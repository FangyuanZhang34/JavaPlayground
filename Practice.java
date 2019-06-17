
import java.util.*;

class Practice{
	static class ListNode{
		int val;
		ListNode next;
		ListNode(int val) {
			this.val = val;
		}
	}

	static class TreeNode{
		int val;
		TreeNode left;
		TreeNode right;
		TreeNode(int val) {
			this.val = val;
		}
	}

	static class Point{
		int x;
		int y;
		Point(int x, int y) {
			this.x = x;
			this.y = y;
		}
		@Override
		public String toString() {
			 return new String(x + ", " + y);
		}
	}

	// threaded binary tree(tree with right null pointer pointed to inorder successor)
	// https://www.geeksforgeeks.org/convert-binary-tree-threaded-binary-tree-set-2-efficient/
	// test:
	// 		ThreadedTree threadedTree = new ThreadedTree();
	// 		ThreadedTree.TTNode root = threadedTree.new TTNode(1);
	// 		root.left = threadedTree.new TTNode(2);
	// 		root.right = threadedTree.new TTNode(3);
	// 		root.left.left = threadedTree.new TTNode(4);
	// 		root.left.right = threadedTree.new TTNode(5);
	// 		root.right.left = threadedTree.new TTNode(6);
	// 		root.right.right = threadedTree.new TTNode(7);	
	// 		threadedTree.createThreaded(root);
	// 		threadedTree.inOrder(root);
	static class ThreadedTree{
		class TTNode{
			int key;
			TTNode left, right;
			boolean isThreaded;
			public TTNode(int key) {
				this.key = key;
				left = null;
				right = null;
			}
		}
		public void inOrder(TTNode root) {
			if(root == null) {
				return;
			}
			TTNode cur = leftMost(root);
			while(cur != null) {
				System.out.print(cur.key + " ");
				if(cur.isThreaded) {
					cur = cur.right;
				} else {
					cur = leftMost(cur.right);
				}
			}
			System.out.print("\n");
		}
		private TTNode leftMost(TTNode root) {
			while(root != null && root.left != null) {
				root = root.left;
			}
			return root;
		}
		// return the right most node in the tree
		// and create all the threads in this tree
		public TTNode createThreaded(TTNode root) {
			//base cases:
			if(root == null) {
				return root;
			}
			if(root.left == null && root.right == null) {
				return root;
			}
			// this round:
			// find predecessor of some node
			// left subtree
			if(root.left != null) {
				TTNode leftRightMost = createThreaded(root.left);
				leftRightMost.right = root;
				leftRightMost.isThreaded = true;
			}
			// return to upper level:
			if(root.right != null) {
				return createThreaded(root.right);
			} else {
				return root;
			}

		}
		
	}

	// =====Tree==============================================================
	// test:
	// 		String str1 = "1,2,3,null,null,4,5,null,null";
	// 		String str2 = "3,4,5";
	// 		TreeNode s = dserializeLevel(str1);
	// 		TreeNode t = dserializeLevel(str2);
	// 		printLevel(s);
	// 		printLevel(t);

	// deserialize(create) a tree from level order
	private static TreeNode dserializeLevel(String s) {
		if(s == null || s.length() == 0) {
			return null;
		}
		String[] split = s.split(",");
		if(split[0].equals("null")) {
			return null;
		}
		// create a TreeNode root and use queue to append left & right
		TreeNode root = new TreeNode(Integer.parseInt(split[0]));
		Queue<TreeNode> level = new LinkedList<>();
		level.offer(root);
		int ind = 1;
		while(!level.isEmpty()) {
			int size = level.size();
			for(int i = 0; i < size; i++) {
				// append left and right nodes for current level nodes
				// add non-null left and right nodes into queue for next level
				TreeNode node = level.poll();
				if(ind >= split.length) {
					continue;
				}
				if(!split[ind].equals("null")) {
					node.left = new TreeNode(Integer.parseInt(split[ind]));
					level.offer(node.left);
				}
				ind++;
				if(!split[ind].equals("null")) {
					node.right = new TreeNode(Integer.parseInt(split[ind]));
					level.offer(node.right);
				}
				ind++;
			}
		}
		return root;
	}

	private static void printLevel(TreeNode root) {
		if(root == null) {
			return;
		}
		System.out.println("printLevelOrder");
		Queue<TreeNode> level = new LinkedList<>();
		level.offer(root);
		while(!level.isEmpty()) {
			int size = level.size();
			for(int i = 0; i < size; i++) {
				TreeNode node = level.poll();
				if(node == null) {
					System.out.print(" null ");
				} else {
					System.out.print(" " + node.val + " ");
					level.offer(node.left);
					level.offer(node.right);
				}
			}
			System.out.print("\n");
		}
		System.out.print("\n");
	}

	// ===List================================================================
	private static void printList(ListNode head) {
		if(head == null) {
			System.out.println("Null");
			return;
		}
		ListNode cur = head;
		while(cur != null) {
			if(cur.next == null) {
				System.out.println(cur.val);
			} else {
				System.out.print(cur.val+" ");
			}
			cur = cur.next;
		}
	}

	public static ListNode createList(int[] arr) {
		if(arr == null || arr.length == 0) {
			return null;
		}
		ListNode head = new ListNode(arr[0]);
		ListNode cur = head;
		for(int i = 1; i < arr.length; i++) {
			ListNode node = new ListNode(arr[i]);
			cur.next = node;
			cur = cur.next;
		}
		return head;
	}



	// ===================================================================
	public static void main(String[] args) {
		System.out.println(permutationsI("1234"));
	}
	// ===================================================================


	// 22. PermutationsI
	public static List<String> permutationsI(String set) {
		List<String> result = new ArrayList<>();
		if(set == null) {
			return result;
		}
		char[] array = set.toCharArray();
		permutationsI(result, array, 0);
		return result;
	}
	private static void permutationsI(List<String> result, char[] array, int ind) {
		//base case:
		if(ind == array.length) {
			result.add(new String(array));
			return;
		}
		//recursive rule:
		for(int i = ind; i < array.length; i++) {
			permuSwap(array, ind, i);

			permutationsI(result, array, ind + 1);
			permuSwap(array, ind, i);
		}
	}
	private static void permuSwap(char[] array, int i, int j) {
		char temp = array[i];
		array[i] = array[j];
		array[j] = temp;
	}

	// 21. substrings with k distinct chars
	// input: an input String; K distinct chars
	// output: a Set of all substrings with k distinct chars
	// System.out.println(kdistinctChar("rwrqertrteewerertretw", 4));
	// [qert, retw, wrqe]
	public static Set<String> kdistinctChar(String input, int K) {
		Set<String> res = new HashSet<>();
		if(input == null || input.length() == 0 || K > input.length()) {
			return res;
		}
		int n = input.length();
		HashSet<Character> set = new HashSet<>();
		int l = 0, r = 0;
		while(r < n) {
			char cl = input.charAt(l);
			char cr = input.charAt(r);
			if(!set.contains(cr) && (r - l) < K) {
				set.add(cr);
				r++;
			} else {
				set.remove(cl);
				l++;
			}
			if(r - l == K) {
				String str = input.substring(l, r);
				res.add(str);
			}
		}
		return res;
	}

	public static Set<String> kdistinctChar2(String input, int K) {
		Set<String> res = new HashSet<>();
		if(input == null || input.length() == 0 || K > input.length()) {
			return res;
		}
		int n = input.length();
		Map<Character, Integer> map = new HashMap<>();
		int end = 0;
		for(int start = 0; start < n; start++) {
			while(end < n && end - start < K) {
				char c = input.charAt(end);
				map.put(c, map.getOrDefault(c,0)+1);
				end++;
			}
			if(map.size() == K) {
				res.add(input.substring(start, end));
			}
			char c = input.charAt(start);
			map.put(c,map.get(c) - 1);
			if(map.get(c) == 0) {
				map.remove(c);
			}
		}
		return res;
	}

	// practice
	public static Set<String> kDistinct(String input, int K) {
		Set<String> res = new HashSet<>();
		if(input == null || input.length() == 0 || K > input.length()) {
			return res;
		}
		int n = input.length();
		Map<Character, Integer> map = new HashMap<>();
		for(int start = 0, end = 0; start < n; start++) {
			while(end < n && end - start < K) {
				char c = input.charAt(end);
				map.put(c, map.getOrDefault(c, 0) + 1);
				end++;
			}
			if(end - start == K && map.size() == K) {
				res.add(input.substring(start, end));
			}
			char c = input.charAt(start);
			map.put(c, map.get(c) - 1);
			if(map.get(c) == 0) {
				map.remove(c);
			} 
		}
		return res;
	}

	// practice2
	public static Set<String> kDistinct2(String input, int K) {
		Set<String> res = new HashSet<>();
		if(input == null || input.length() == 0 || K > input.length()) {
			return res;
		}
		int n = input.length();
		Map<Character, Integer> map = new HashMap<>();
		int end = 0;
		for(int start = 0; start < n; start++) {
			while(end < n && end - start < K) {
				char c = input.charAt(end);
				map.put(c, map.getOrDefault(c,0) + 1);
				end++;
			}

			int i = start;
			for(;i < end; i++) {
				if(map.get(input.charAt(i)) != 1) {
					break;
				}
			}
			if((end - start == K) && i == end) {
				res.add(input.substring(start, end));
			}

			char c = input.charAt(start);
			map.put(c, map.get(c) - 1);
			if(map.get(c) == 0) {
				map.remove(c);
			}
		}
		return res;
	}

	// 20. Most common words(with the same max frequency)  https://leetcode.com/problems/most-common-word/
	// input: String literatureText, List wordToExclude, 
	// output: List res
	// test: 
	// 		String s = "ball time time beer time beer beer ball ball ball";
	// 		List<String> banned = new ArrayList<>();
	// 		banned.add("ball");
	// 		System.out.println(retrieveMostFrequentlyUsedWords(s, banned));	
	public static List<String> retrieveMostFrequentlyUsedWords (String literatureText, 
		List<String> wordToExclude) {
		List<String> res = new ArrayList<>();
		literatureText += ".";
		Set<String> excludeSet = new HashSet<>();
		for(String word : wordToExclude) {
			excludeSet.add(word);
		} 
		Map<String, Integer> freqMap = new HashMap<>();
		int maxFreq = 0;
		String word = "";
		for(int i = 0; i < literatureText.length(); i++) {
			char c = literatureText.charAt(i);
			if(Character.isLetter(c)) {
				word += c;
			} else if(word.length() > 0) {
				if(!excludeSet.contains(word)) {
					freqMap.put(word, freqMap.getOrDefault(word, 0) + 1);
					if(freqMap.get(word) > maxFreq) {
						maxFreq = freqMap.get(word);
						res.clear();
						res.add(word);
					} else if(freqMap.get(word) == maxFreq) {
						res.add(word);
					}					
				}
				word = "";
			}
		}
		return res;
	}

	// practice 
	private static List<String> re(String literatureText, List<String> wordToExclude) {
		List<String> res = new ArrayList<>();
		if(literatureText == null || literatureText.length() == 0) {
			return res;
		}
		literatureText += ".";
		literatureText = literatureText.toLowerCase();
		Set<String> excluded = new HashSet<>();
		for(String s : wordToExclude) {
			excluded.add(s.toLowerCase());
		}
		Map<String, Integer> map = new HashMap<>();
		int maxFreq = 0;
		StringBuilder word = new StringBuilder();
		for(char c : literatureText.toCharArray()) {
			if(Character.isLetter(c)) {
				word.append(c);
			} else if(word.length() > 0) {
				String finalWord = word.toString();
				map.put(finalWord, map.getOrDefault(finalWord, 0) + 1);
				int freq = map.get(finalWord);
				if(freq == maxFreq) {
					res.add(finalWord);
				} else if(freq > maxFreq) {
					res.clear();
					maxFreq = freq;
					res.add(finalWord);
				}
				word = new StringBuilder();
			} 
		}
		return res;
	}


	// 19. Shortest Job First waiting time: 
	// input: the arr time and execution duration of each request sorted by arr time,
	// rule: 	-first come first go;
	// 			-if there are jobs waiting in line:
	// 				-execute the one with shortest execution duration
	// 				-for jobs with same execution duration: 
	// 					execute the job comes first
	// return: average waiting time
	// test:
	// 		int[] req = new int[]{0,1,4,6,8,8,8,10};
	// 		int[] dur = new int[]{10,2,3,4,5,6,4,3};
	// 		System.out.println(shortestJobFirst(req, dur));
	public static float shortestJobFirst(int[] req, int[] dur) {
		class Job {
			int arrT;
			int dur;
			Job(int arrT, int dur) {
				this.arrT = arrT;
				this.dur = dur;
			}
		}

		if(req == null || dur == null || req.length != dur.length) {
			return 0;
		}
		int curT = 0, wait = 0, ind = 0;
		// store all the jobs comes before curT
		PriorityQueue<Job> pq = new PriorityQueue<>((job1, job2) -> {
			if(job1.dur == job2.dur) {
				return job1.arrT - job2.arrT;
			}
			return job1.dur - job2.dur;
		});
		// add each job into the req pq and update curT
		while(!pq.isEmpty() || ind < req.length) {
			if(!pq.isEmpty()) {
				// poll the first job to execute
				Job job = pq.poll();
				// update the wait, curT 
				wait += curT - job.arrT;
				curT += job.dur;
				// offer all the jobs that arrive before the curT into the pq
				while(ind < req.length && req[ind] <= curT) {
					pq.offer(new Job(req[ind], dur[ind++]));
				}
			} else {
				// if the pq is empty
				// offer first job in the array, update (wait) and curT
				pq.offer(new Job(req[ind], dur[ind]));
				curT = req[ind++];
			}
		}
		return wait / (float)req.length;
	}
	// ===================================================================



	// 18-1. check if the linked list has a cycle or not
	// input: 	the head of a linked list
	// return: 	has a cycle or not
	// test:
	// 		ListNode head = createList(new int[]{1,2,3,4,5,3,2,1});
	// 		// create an entry at val = 4
	// 		ListNode curr = head, entry = head, tail = head;
	// 		while(curr != null) {
	// 			if(curr.val == 4) {
	// 				entry = curr;
	// 			}
	// 			if(curr.next == null) {
	// 				tail = curr;
	// 			}
	// 			curr = curr.next;
	// 		}
	// 		tail.next = entry;
	// 		System.out.println("hasCycle : " + hasCycle(head));
	// 		System.out.println("detectCycle : " + detectCycle(head).val);		
	public static boolean hasCycle(ListNode head) {
		if(head == null) {
			return false;
		}
		ListNode slow = head, fast = head;
		while(fast.next != null && fast.next.next != null) {
			slow = slow.next;
			fast = fast.next.next;
			if(slow == fast) {
				return true;
			}
		}
		return false;
	}
	// ===================================================================


	// 18-2. detect a cycle in a linked list return the entry list node
	// 		https://leetcode.com/problems/linked-list-cycle-ii/submissions/
	// input: 	the head of a linked list
	// return: 	the entry of cycle inside the linked list or null if there is no cycle
	// test:
	// 		ListNode head = createList(new int[]{1,2,3,4,5,3,2,1});
	// 		// create an entry at val = 4
	// 		ListNode curr = head, entry = head, tail = head;
	// 		while(curr != null) {
	// 			if(curr.val == 4) {
	// 				entry = curr;
	// 			}
	// 			if(curr.next == null) {
	// 				tail = curr;
	// 			}
	// 			curr = curr.next;
	// 		}
	// 		tail.next = entry;
	// 		System.out.println("hasCycle : " + hasCycle(head));
	// 		System.out.println("detectCycle : " + detectCycle(head).val);
	public static ListNode detectCycle(ListNode head) {
		if(head == null) {
			return null;
		}
		ListNode slow = head, fast = head, entry = head;
		while(fast.next != null && fast.next.next != null) {
			slow = slow.next;
			fast = fast.next.next;
			if(slow == fast) {
				while(entry != slow) {
					entry = entry.next;
					slow = slow.next;
				}
				return entry;
			}
		}
		return null;
	}
	// ===================================================================



	// 17. Insert into cycle linked list : https://leetcode.com/problems/insert-into-a-cyclic-sorted-list/
	// input : the head node of a sorted cycle linked list; the val we want to insert into the linked list
	// return : the head of the sorted cycle linked list after insertion
	// test:
	// 		ListNode head = createList(new int[]{1,2,3,3,3,6,10});
	// 		printList(head);
	// 		ListNode curr = head;
	// 		while(curr != null) {
	// 			if(curr.next == null) {
	// 				break;
	// 			}
	// 			curr = curr.next;
	// 		}
	// 		curr.next = head;
	// 		ListNode res = linkedListInsert(head, 5);
	// 		curr = res;
	// 		do{
	// 			if(curr.next == res) {
	// 				System.out.println(curr.val);
	// 			} else {
	// 				System.out.print(curr.val+" ");
	// 			}
	// 			curr = curr.next;
	// 		}
	// 		while(curr != res);
	public static ListNode linkedListInsert(ListNode head, int val) {
		ListNode res = new ListNode(val);
		if(head == null) {
			res.next = res;
			return res;
		}
		// find the curr node after which we insert the new node
		ListNode curr = head;
		do {
			// insert between start and end
			if(curr.val <= val && val <= curr.next.val) {
				break;
			}
			// insert at end & start intersection
			if(curr.val > curr.next.val && 
				(curr.val <= val || val <= curr.next.val)) {
				break;
			}
			curr = curr.next;
		} while(curr != head);
		// insert the node after curr node
		res.next = curr.next;
		curr.next = res;
		return head;
	}
	// ===================================================================



	// 16. Rotate Matrix clockwise / counterclockwise 90 degress
	// https://leetcode.com/problems/transpose-matrix/submissions/
	// input: 	an m * n int matrix; 
	// 			flag = 1 : clockwise / flag != 1 : counterclockwise
	// return: 	a new matrix after rotation of 90 degrees
	// test:
	// 			int[][] matrix = {{1,2}, {3,4}, {5,6}};
	// 			int flag = 0;
	// 			matrix = rotateMatrix(matrix, flag);
	// 			System.out.println(Arrays.toString(matrix[0]));
	// 			System.out.println(Arrays.toString(matrix[1]));
	// 
	// First transpose the matrix
	// Then reverse rows(clockwise) / columns(counterclockwise)
	public static int[][] rotateMatrix(int[][] matrix, int flag) {
		if(matrix == null || matrix.length == 0 || matrix[0].length == 0) {
			return new int[0][0];
		}
		int nr = matrix.length, nc = matrix[0].length;
		int[][] res = new int[nc][nr];
		// first transpose the matrix
		transpose(res, matrix);
		// then if clockwise : 
		if(flag == 1) {
			reverseR(res);
		} else {
			reverseC(res);
		}
		return res;
	}
	private static void transpose(int[][] res, int[][] matrix) {
		for(int i = 0; i < matrix.length; i++) {
			for(int j = 0; j < matrix[0].length; j++) {
				res[j][i] = matrix[i][j];
			}
		}
	}
	private static void reverseR(int[][] res) {
		for(int i = 0; i < res.length; i++) {
			for(int j = 0; j < res[0].length / 2; j++) {
				int temp = res[i][j];
				res[i][j] = res[i][res[0].length - j - 1];
				res[i][res[0].length - j - 1] = temp;
			}
		}
	}
	private static void reverseC(int[][] res) {
		for(int i = 0; i < res[0].length; i++) {
			for(int j = 0; j < res.length / 2; j++) {
				int temp = res[j][i];
				res[j][i] = res[res.length - j - 1][i];
				res[res.length - j - 1][i] = temp;
			}
		}
	}
	// ===================================================================


	// 15. Round Robin: https://www.geeksforgeeks.org/program-round-robin-scheduling-set-1/
	// input 	Atime: the initialized arrival time for each process;
	// 			Etime: the total execution time for each process;
	// 			q: the quantum of each Robin round
	// return  	the average waiting time for each process
	// test: 	
	//		int[] Atime = new int[]{0, 0, 0};
	// 		int[] Etime = new int[]{3, 4, 3};
	// 		int q = 1;
	// 		System.out.println(roundRobin(Atime, Etime, q));
	public static float roundRobin(int[] Atime, int[] Etime, int q) {
		class Process {
			int arrT; // arrival time
			int remP; // remaining period
			Process(int arrT, int remP) {
				this.arrT = arrT;
				this.remP = remP;
			}
		}	

		if(Atime == null || Etime == null || Atime.length != Etime.length || q <= 0) {
			return 0;
		}
		int ind = 0;
		int curr = 0;
		int wait = 0;
		// use a queue to store all the Processes waiting to be executed
		Queue<Process> queue = new LinkedList<>();
		while(!queue.isEmpty() || ind < Atime.length) {
			if(!queue.isEmpty()) {
				// ===========execute thisProcess===========
				// if queue is not empty, poll thisProcess from queue, execute it, compute wait time
				// curr += min(q, remP)
				Process process = queue.poll();
				wait += curr - process.arrT;
				curr += Math.min(q, process.remP);
				// ==============new curr===================
				// offer all new Processes have arrived after ind before curr into the queue
				for(; ind < Atime.length && Atime[ind] <= curr; ind++) {
					queue.offer(new Process(Atime[ind], Etime[ind]));
				}
				// last if thisProcess not finished, the thisProcess should be offered into the queue again				
				if(q < process.remP) {
					// process haven't finished
					// offer again into the queue with new arrT, and remP
					queue.offer(new Process(curr, process.remP - q));
				}
			} else {
				// if queue is empty, offer the next process in the array and set curr time to Atime
				queue.offer(new Process(Atime[ind], Etime[ind]));
				curr = Atime[ind++];
			}
		}
		return wait / (float)Atime.length;
	}
	// ===================================================================


	// 14. LRU count missed requests  (hit <-> miss)
	// use Double LinkedList and HashMap to implement a LRU class
	// has a counter to count the missed requests
	// test: 
	// 		LRUCache cache = new LRUCache(2);
	// 		cache.put(1, 1);
	// 		cache.put(2, 2);
	// 		System.out.println("1 : 1 : " + cache.get(1));       // returns 1
	// 		cache.put(3, 3);    // evicts key 2
	// 		System.out.println("2 : -1 : " + cache.get(2));       // returns -1 (not found)
	// 		cache.put(4, 4);    // evicts key 1
	// 		System.out.println("1 : -1 : " + cache.get(1));       // returns -1 (not found)
	// 		System.out.println("3 : 3 : " + cache.get(3));       // returns 3
	// 		System.out.println("4 : 4 : " + cache.get(4));       // returns 4
	// 		System.out.println("miss : 2 : " + cache.miss);
	static class LRUCache {
		class DLinkedNode {
			int key;
			int value;
			DLinkedNode prev;
			DLinkedNode next;
			DLinkedNode(int key, int value) {
				this.key = key;
				this.value = value;
			}
		}
		Map<Integer, DLinkedNode> map;
		DLinkedNode head;
		DLinkedNode tail;
		int size;
		int capacity;
		int miss;
		public LRUCache (int capacity){
			map = new HashMap<>();
			head = new DLinkedNode(0, 0);
			tail = new DLinkedNode(0, 0);
			head.next = tail;
			tail.prev = head;
			size = 0;
			this.capacity = capacity;
			miss = 0;
		}
		public void put (int key, int value) {
			DLinkedNode node = map.get(key);
			if(node == null) {
				// create a new node
				DLinkedNode newNode = new DLinkedNode(key, value);
				// add to map
				map.put(key, newNode);
				// add to head
				addToHead(newNode);
				// update size
				size++;
				// check size
				if(size > capacity) {
					DLinkedNode last = tail.prev;
					remove(last);
					map.remove(last.key);
					size--;
				}
			} else {
				// update value
				node.value = value;
				// move to head
				moveToHead(node);
			}
		}
		private int get(int key) {
			DLinkedNode node = map.get(key);
			if(node != null){
				// move to head
				moveToHead(node);
				// return value
				return node.value;
			}
			miss++;
			return -1;
		}
		private void remove(DLinkedNode node) {
			node.prev.next = node.next;
			node.next.prev = node.prev;
		}
		private void addToHead(DLinkedNode node) {
			node.prev = head;
			node.next = head.next;
			head.next.prev = node;
			head.next = node;
		}
		private void moveToHead(DLinkedNode node) {
			remove(node);
			addToHead(node);
		}
	}
	// ===================================================================


	// 13. Greatest Common Divisor
	// input: an array of numbers
	// return: an integer, which is the greatest common divisor of this array of integers
	// test: 
	// 		System.out.println(gcd(new int[]{50, 35, 70, 1000}));
	public static int gcd(int[] nums) {
		if(nums == null || nums.length == 0) {
			return 0;
		}
		int gcd = nums[0];
		for(int i = 1; i < nums.length; i++) {
			gcd = gcd(gcd, nums[i]);
		}
		return gcd;
	}
	// Euclid method: https://www.cnblogs.com/drizzlecrj/archive/2007/09/14/892340.html
	private static int gcd(int num1, int num2) {
		if(num2 == 0) {
			return num1;
		}
		return gcd(num2, num1 % num2);
	}
	// ===================================================================


	// 12. window sum
	// input: an arraylist A, and a size of sliding window k
	// return an arraylist that lists all the sum inside the window.
	// test:
	// 		Integer[] nums = {1,2,3,2,1,3,4,10};
	// 		List<Integer> A = Arrays.asList(nums);
	// 		int k = 3;
	// 		System.out.println(windowSum(A, k));	
	public static List<Integer> windowSum(List<Integer> A, int k) {
		List<Integer> res = new ArrayList<>();
		if(A == null || A.size() == 0) {
			return res;
		}
		for(int i = 0; i + k - 1 < A.size(); i++) {
			int sum = 0;
			// add i ~ i + k - 1 to sum
			for(int j = 0; j < k; j++) {
				sum += A.get(i + j);
			}
			res.add(sum);
		}
		return res;
	}
	// ===================================================================


	// Version 2. combination sum  
	// 40. Combination Sum II https://leetcode.com/problems/combination-sum-ii/submissions/
	// test: 
	// 		int[] nums = {1,2,3,2,1,4,3,5,3};
	// 		int target = 5;
	// 		System.out.println(comSum(nums, target));
	public static List<List<Integer>> comSum(int[] nums, int target) {
		if(nums == null || nums.length == 0) {
			return new ArrayList<>();
		}
		Arrays.sort(nums);
		List<List<Integer>> res = new ArrayList<>();
		List<Integer> list = new ArrayList<>();
		dfs(nums, target, res, list, 0);
		return res;
	}
	private static void dfs(int[] nums, int target, List<List<Integer>> res, List<Integer> list, int ind) {
		if(target == 0) {
			res.add(new ArrayList<>(list));
			return;
		}
		if(target < 0 || ind >= nums.length) {
			return;
		}
		for(int i = ind; i < nums.length; i++) {
			// only dfs the first number when there are duplicates
			if(i > ind && nums[i] == nums[i - 1]) {
				continue;
			}
			if(nums[i] <= target) {
				list.add(nums[i]);
				dfs(nums, target - nums[i], res, list, i + 1);  // pay attention to the next index is "i+1"
				list.remove(list.size() - 1);
			}
		}
	}
	// ===================================================================




	// 11. Overlap rectangle 
	// given four points, upper-left and bottom-right for each rectangle
	// return: 	these two rectangles are overlapping or not
	// test:
	// 		System.out.println(doOverlap(new Point(-3,4),new Point(3,0),new Point(0,2),new Point(9,-1)));
	public static boolean doOverlap(Point l1, Point r1, Point l2, Point r2) {
		// if one rectangle is on the right side of the other
		if(l1.x > r2.x || l2.x > r1.x) {
			return false;
		}
		// if one rectangle is on the top of the other
		if(r1.y > l2.y || r2.y > l1.y) {
			return false;
		}
		return true;
	}

	// Version 2: compute total area covered by two rectangles
	// https://leetcode.com/problems/rectangle-area/
	public static int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
		// compute overlapped area
        // add two area minus overlapped area
        int area1 = Math.abs(A - C) * Math.abs(B - D);
        int area2 = Math.abs(E - G) * Math.abs(F - H);
        int innerL = Math.max(A, E);
        int innerR = Math.max(innerL, Math.min(C, G));
        int innerB = Math.max(B, F);
        int innerT = Math.max(innerB, Math.min(D, H));
        return (area1 + area2 - (innerR - innerL) * (innerT - innerB));
	}
	// ===================================================================



	// 10. find k nearest point to a given origin 
	// input: ArrayList of Points; integer k
	// return: ArrayList of all k closest points
	// test:
	//		List<List<Integer>> locations = new ArrayList<>();
	// 		locations.add(Arrays.asList(new Integer[]{1,2}));
	// 		locations.add(Arrays.asList(new Integer[]{3,4}));
	// 		locations.add(Arrays.asList(new Integer[]{1,-1}));
	// 		System.out.println(closestLocations(3, locations, 2));
	public static List<List<Integer>> closestLocations(int totalCrates, List<List<Integer>> allLocations,
												int truckCapacity) {
		if(allLocations == null || allLocations.size() == 0 || truckCapacity == 0) {
			return new ArrayList<List<Integer>>();
		}
		// use a max heap to maintain k closest points inside the heap
		PriorityQueue<List<Integer>> maxHeap = new PriorityQueue<List<Integer>>(truckCapacity, (loc1, loc2) -> {
			return (int)(dist(loc2) - dist(loc1));
		});
		for(int i = 0; i < allLocations.size(); i++) {
			List<Integer> loc = allLocations.get(i);
			if(i < truckCapacity) {
				maxHeap.offer(loc);
			} else if(dist(maxHeap.peek()) > dist(loc)){
				maxHeap.poll();
				maxHeap.offer(loc);
			}
		}
		List<List<Integer>> res = new ArrayList<>();
		while(!maxHeap.isEmpty()) {
			res.add(maxHeap.poll());
		}
		return res;
	}
	private static double dist(List<Integer> loc) {
		return Math.pow(loc.get(0), 2) + Math.pow(loc.get(1), 2);
	}
	// ===================================================================




	// 9. two sum existence
	// unsorted，duplicated，exists or not. https://app.laicode.io/app/problem/180
	// Set<Integer(val)>
	public boolean twoSumExist(int[] array, int target) {
		if(array == null || array.length == 0) {
			return false;
		}
		Set<Integer> visited = new HashSet<>();
		for(int i = 0; i < array.length; i++) {
			int complement = target - array[i];
			if(visited.contains(complement)) {
				return true;
			}
			visited.add(array[i]);
		}
		return false;
	}


	// 9. two sum one pair
	// unsorted，duplicated，one pair。 https://leetcode.com/problems/two-sum/
	// Map<Integer(val), Integer(ind)> 
	public int[] twoSumOnePair(int[] nums, int target) {
        if(nums == null || nums.length == 0) {
            return new int[0];
        }
        Map<Integer, Integer> visited = new HashMap<>();
        for(int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if(visited.containsKey(complement)) {
                return new int[]{visited.get(complement), i};
            }
            visited.put(nums[i], i);
        }
        return new int[0];
    }

    // 9. two sum all pairs
    // unsorted，duplicated，all pairs. https://app.laicode.io/app/problem/181
	// Map<Integer(val), List<Integer>(inds)>
	public List<List<Integer>> twoSumAllPairs(int[] array, int target) {
		List<List<Integer>> res = new ArrayList<>();
		if(array == null || array.length == 0) {
			return res;
		}
		Map<Integer, List<Integer>> visited = new HashMap<>();
		for(int i = 0; i < array.length; i++) {
			int complement = target - array[i];
			if(visited.containsKey(complement)){
			for(int comInd : visited.get(complement)) {
				res.add(Arrays.asList(comInd, i));
			}
			}
			if(!visited.containsKey(array[i])) {
				visited.put(array[i], new ArrayList<>());
			}
			visited.get(array[i]).add(i);
		}
		return res;
	}

	// 9. two sum distinct val pairs
	// unsorted，duplicated，all distinct value pairs. https://app.laicode.io/app/problem/182   
	// Map<Integer(val), Integer(count)>
	public List<List<Integer>> twoSumDistValPairs(int[] array, int target) {
		List<List<Integer>> res = new ArrayList<>();
		if(array == null || array.length == 0) {
			return res;
		}
		Map<Integer, Integer> counter = new HashMap<>();
		for(int i = 0; i < array.length; i++) {
			int complement = target - array[i];
			Integer comCount = counter.get(complement);
			if(array[i] == complement && comCount != null && comCount == 1) {
				res.add(Arrays.asList(array[i], array[i]));
			} else if(array[i] != complement && comCount != null && 
				!counter.containsKey(array[i])) {
				res.add(Arrays.asList(complement, array[i]));
			}
			counter.put(array[i], counter.getOrDefault(array[i], 0) + 1);
		}
		return res;
	}

	// 9. two sum counter 
	// unsorted，duplicated，count of pairs. https://app.laicode.io/app/problem/181
	// test: 
	// 		int[] nums = {1,2,5,3,4,5,2,6,7};
	// 		int target = 4;
	// 		System.out.println(twoSumI(nums, target));
	public static int twoSumCounter(int[] nums, int target) {
		if(nums == null || nums.length == 0) {
			return 0;
		}
		Map<Integer, Integer> map = new HashMap<>(); // number -> count of the number
		int res = 0;
		for(int i = 0; i < nums.length; i++) {
			// find complement and update res
			int complement = target - nums[i];
			if(map.containsKey(complement)) {
				res += map.get(complement);
			}
			// for nums[i] itself
			// updata map
			map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
		}
		return res;
	}
	// 9. =======================================


	// 8. subtree https://leetcode.com/problems/subtree-of-another-tree/submissions/
	// check is tree t has same values as tree s's subtree
	// traverse all the subtrees of s, check if there is a subtree same as t
	// return -1 => not a subtree
	// return  1 => is a subtree
	// test:
	// 		String str1 = "1,2,3,null,null,4,5,null,null";
	// 		String str2 = "3,4,5";
	// 		TreeNode s = dserializeLevel(str1);
	// 		TreeNode t = dserializeLevel(str2);
	// 		printLevel(s);
	// 		printLevel(t);
	// 		System.out.println(isSubtree(s,t));
	public static int isSubtree(TreeNode s, TreeNode t) {
		if(s == null) {
			return -1;
		}
		if(t == null) {
			return 1;
		}
		// check if s, t are same
		// or t is the subtree of s
		if(isSameTree(s, t)) {
			return 1;
		}
		if(isSubtree(s.left, t) == 1 || isSubtree(s.right, t) == 1) {
			return 1;
		}
		return -1;
	}
	// check if s, t are same
	private static boolean isSameTree(TreeNode s, TreeNode t) {
		if(s == null && t == null) {
			return true;
		}
		if(s == null || t == null) {
			return false;
		}
		// if s and t are not null but s t have different values => false
		if(s.val != t.val) {
			return false;
		}
		// if s and t has the same values
		// then check if left and right subtrees are same for s and t
		return isSameTree(s.left, t.left) && isSameTree(s.right, t.right);
	}
	// 8. =======================================




	// 7. reverseSecondHalfLinkedList
	// reverse the secon half of a linkedlist
	// return the new list
	// test: 	int[] arr = {1,2,3,4,5,6,7,8,9,10};
	// 			printList(createList(arr));
	// 			printList(reverseHalfList(createList(arr)));
	//			1 2 3 4 5 6 7 8 9 10
    // 			1 2 3 4 5 10 9 8 7 6
	public static ListNode reverseHalfList(ListNode head) {
		// 1. corner case
		if(head == null || head.next == null) {
			return head;
		}
		// 2. find the middle point and its prev point(!!slow!!)
		ListNode tail1 = head, slow = head, fast = head;
		while(fast.next != null && fast.next.next != null) {
			slow = slow.next;
			fast = fast.next.next;
		}
		// 3. reverse the second half list (slow.next - null)
		tail1 = slow; // at middle point
		ListNode prev = null;
		ListNode cur = slow.next; // the new tail, the first cur.next we have to change
		while(cur != null) {
			ListNode next = cur.next;
			cur.next = prev; // main operation in each run
			prev = cur;
			cur = next;
		}
		// 4. combine the two parts together
		tail1.next = prev; // prev is the new head

		return head;
	}
	// 7. =======================================

 
	// 6. merge two sorted lists https://leetcode.com/problems/merge-two-sorted-lists/
	// return the sorted list
	public static ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		if(l1 == null || l2 == null) {
			return (l1 == null) ? l2 : l1;
		}
		ListNode dummy = new ListNode(0); // create a dummy head
		ListNode cur = dummy;
		while(l1 != null && l2 != null) {
			if(l1.val <= l2.val) {
				cur.next = l1;
				l1 = l1.next;
			} else {
				cur.next = l2;
				l2 = l2.next;
			}
			cur = cur.next;
		}
		if(l1 != null) {
			cur.next = l1;
		} else if(l2 != null) {
			cur.next = l2;
		}
		return dummy.next;
	}
	// 6. =======================================


	// 5. longestParlin1   https://leetcode.com/problems/longest-palindromic-substring/
	// return the longest palindrome substring
	// test:	String s = "abcdceed";
	// 			System.out.println(longestParlin1(s));
	public static String longestParlin1(String s) {
		if(s == null || s.length() == 0) {
			return s;
		}
		boolean[][] M = new boolean[s.length()][s.length()]; // M[i][j] if s i-j is palin or not
		int start = 0, end = 0; // max palin substring start and end
		int maxLen = 1; // max palin substring length
		// base case: diagonal -> true
		for(int i = 0; i < s.length(); i++) {
			M[i][i] = true;
		}
		// induction rule: from diagonal to right upper corner
		// M[i][i + 1] = if si == s(i+1)
		// M[i][j] = (M[i + 1][j - 1]) && (si == sj)
		for(int gap = 1; gap < s.length(); gap++) {
			for(int i = 0; i + gap < s.length(); i++) {
				if(gap == 1) {
					M[i][i + gap] = (s.charAt(i) == s.charAt(i + gap));
				} else {
					M[i][i + gap] = M[i + 1][i + gap - 1] && (s.charAt(i) == s.charAt(i + gap));
				}
				if(M[i][i + gap]) {
					if(maxLen < gap + 1) {
						maxLen = gap + 1;
						start = i;
						end = i + gap;
					}
				}
			}
		}
		return (start == end) ? "OOXX" : s.substring(start, end + 1);
	}
	// 5. =======================================

	// 4. valid parenthesis
	// return -1 if it is not valid parenthesis string, 
	// o.w. return num of pairs
	// test: 	String s = "(())()";
	// 			System.out.println(validParen(s));
	public static int validParen(String s) {
		if(s == null) {
			return -1;
		}
		if(s.length() == 0) {
			return 0;
		}
		int total = 0; // keep track of num of pairs
		Stack<Character> stack = new Stack<>();
		for(char c : s.toCharArray()) {
			// once meet a left paranthese
			// push into stack
			if(c == '(') {
				stack.push(c);
			} else {
				// once meet a right parenthese
				// pop one left parenthese from the stack
				// means there is a pair
				// total + 1
				if(stack.isEmpty()) {
					return -1;
				}
				stack.pop();
				total++;
			}
		}
		// if there is still left parenthesis left => not valid
		// else, is valid and return total num
		return stack.isEmpty() ? total : -1;
	}
	// 4. =======================================


	// 3. remove vowels
	// to remove all vowels from a String, return the result String
	// test : 	String s = "FangyuanZhang";
	// 			System.out.println(removeVowel(s));
	public static String removeVowel(String s) {
		if(s == null || s.length() == 0) {
			return s;
		}
		StringBuilder sb = new StringBuilder();
		String vowels = "aeiouAEIOU";
		for(char c : s.toCharArray()) {
			if(!vowels.contains(c+"")) {
				sb.append(c);
			}
		}
		return sb.toString();
	}
	// 3. =======================================

	// 2. Gray Code:
	// 	to check if two bytes are adjacent bytes as gray codes
	// 	test: 	byte term1 = (byte)0b10000000;
	// 			byte term2 = (byte)0b10011000;
	// 			System.out.println(grayCode(term1, term2));
	public static boolean grayCode(byte term1, byte term2) {
		byte diff = (byte)(term1 ^ term2);
		// check if diff is power of 2
		// which means to check if there is only one 1 in the byte
		while((diff & 1) == 0) {
			diff >>>= 1;
		}
		return diff == 1;	
	}
	// 2. =======================================
	
}

