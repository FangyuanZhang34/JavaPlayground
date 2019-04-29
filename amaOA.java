
import java.util.*;

class amaOA{
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

	public static List<List<Integer>> createListListInt(Integer[][] matrix) {
		List<List<Integer>> roads = new ArrayList<>();
		for(int i = 0; i < matrix.length; i++) {
			List<Integer> road = Arrays.asList(matrix[i]);
			roads.add(road);
		}
		return roads;
	}



	// ===================================================================
	public static void main(String[] args) {
		
	}
	// ===================================================================



	// 37. build new roads with minimum cost (MST)
	// input: 	two arraylist with old roads and new roads, and an integer of total number of cities
	// print: 	list of new roads we have to build
	// test:
	// 		Integer[][] oldR = {{1,4},{4,5},{2,3}};
	// 		Integer[][] newR = {{1,2,5},{1,3,10},{1,6,2},{5,6,5}};
	// 		List<List<Integer>> oldRoads = createListListInt(oldR);	
	// 		List<List<Integer>> newRoads = createListListInt(newR);			
	// 		minCostNewMST(oldRoads,newRoads,6);
	public static void minCostNewMST(List<List<Integer>> oldRoads, List<List<Integer>> newRoads, int numCities) {
		// map all the old cities to unionfind indices
		// build old roads in Unionfind object
		// map all the new citites to unionfind indices
		// store all new roads into pq
		// check and build new roads in Unionfind object, add to res List
		class Unionfind {
			int[] arr;
			int size;
			Unionfind(int size) {
				arr = new int[size];
				Arrays.fill(arr, -1);
				size = 0;
			}
			public int root(int i) {
				while(arr[i] != -1) {
					i = arr[i];
				} 
				return i;
			}
			public boolean find(int i, int j) {
				return root(i) == root(j);
			}
			// assign i to j's subset
			//		assign i's root's root as j's root
			public void union(int i, int j) {
				int rooti = root(i);
				int rootj = root(j);
				arr[rooti] = rootj;
				size++;
			}
		}
		if(oldRoads == null || newRoads == null) {
			System.out.println("null input");
			return;
		}

		// all old cites into map
		int ind = 0;
		Map<Integer, Integer> cityMap = new HashMap<>();
		// build all old roads using UF
		Unionfind uf = new Unionfind(numCities);
		for(List<Integer> road : oldRoads) {
			Integer city1 = road.get(0);
			Integer city2 = road.get(1);
			// put cities into map
			if(!cityMap.containsKey(city1)) {
				cityMap.put(city1, ind++);
			}
			if(!cityMap.containsKey(city2)) {
				cityMap.put(city2, ind++);
			}
			// build roads in UF
			if(!uf.find(cityMap.get(city1), cityMap.get(city2))) {
				uf.union(cityMap.get(city1), cityMap.get(city2));
			}
		}
		int oldRoadsNum = uf.size;
		PriorityQueue<List<Integer>> minHeap = new PriorityQueue<List<Integer>>(newRoads.size(),(road1, road2) -> {
			return road1.get(2) - road2.get(2);
		});
		// store all the new citites into map
		// store all new roads into pq
		for(List<Integer> road : newRoads) {
			Integer city1 = road.get(0);
			Integer city2 = road.get(1);
			// put cities into map
			if(!cityMap.containsKey(city1)) {
				cityMap.put(city1, ind++);
			}
			if(!cityMap.containsKey(city2)) {
				cityMap.put(city2, ind++);
			}
			// into pq
			minHeap.offer(road);
		}
		// 结果：res ArrayList
		List<List<Integer>> res = new ArrayList<>();
		// use pq to poll min-cost road each round and check if it is needed in a spanning tree
		while(!minHeap.isEmpty() && oldRoadsNum + res.size() < numCities - 1) {
			List<Integer> polledRoad = minHeap.poll();
			Integer city1 = polledRoad.get(0);
			Integer city2 = polledRoad.get(1);
			// build roads in UF
			if(!uf.find(cityMap.get(city1), cityMap.get(city2))) {
				uf.union(cityMap.get(city1), cityMap.get(city2));
				res.add(polledRoad);
			}
		}
		// can not build a spanning tree
		if(oldRoadsNum + res.size() < numCities - 1) {
			System.out.println("cannot!");
			return;
		}
		for(List<Integer> road : res) {
			System.out.println(road.toString());
		}

	}



	// 37. Construct an MST 
	// input: list of all the roads and its costs List<List<Integer>> [[xxx],[],[],[],[],[],[],[]]
	// return: a list of all the roads(with sorted cost) of a min-cost MST List<List<Integer>> [[xxx],[],[]]
	// test:
	// 		Integer[][] matrix = {{1,2,1},{1,3,2},{2,3,3},{2,4,2},{3,4,3}};
	// 		List<List<Integer>> roads = createListListInt(matrix);
	// 		System.out.println(roads);
	// 		System.out.println(minCostMST(roads));
	// res:	[[1, 2, 1], [1, 3, 2], [2, 3, 3], [2, 4, 2], [3, 4, 3]]
	// 		[[1, 2, 1], [1, 3, 2], [2, 4, 2]]
	// cityMap : map all the cities to unionfind indices
	// Unionfind : store the graph, build edges, used to check if it is a spanning tree
	// PriorityQueue : poll min-cost new roads
	// mst List : result all the edges we build, 只用做返回值记录，不参与checkMST
	public static List<List<Integer>> minCostMST(List<List<Integer>> roads) {
		class Unionfind {
			int[] arr;
			Unionfind(int size) {
				arr = new int[size];
				Arrays.fill(arr, -1);
			}
			public int root(int i) {
				while(arr[i] != -1) {
					i = arr[i];
				} 
				return i;
			}
			public boolean find(int i, int j) {
				return root(i) == root(j);
			}
			// assign i to j's subset
			//		assign i's root's root as j's root
			public void union(int i, int j) {
				int rooti = root(i);
				int rootj = root(j);
				arr[rooti] = rootj;
			}
		}
		if(roads == null || roads.size() == 0) {
			return new ArrayList<>();
		}
		// map from all cityId -> unionfind ind & to avoid road duplication
		Map<Integer, Integer> cityMap = new HashMap<>();
		// poll min-cost road from minHeap each round
		PriorityQueue<List<Integer>> minHeap = new PriorityQueue<List<Integer>>(roads.size(), (road1, road2) -> {
			return road1.get(2) - road2.get(2);
		});
		// offer all the unique cities and roads into the cityMap and minHeap
		int ind = 0;
		for(List<Integer> road : roads) {
			Integer city1 = road.get(0);
			Integer city2 = road.get(1);
			if(!cityMap.containsKey(city1)) {
				cityMap.put(city1, ind++);
			}
			if(!cityMap.containsKey(city2)) {
				cityMap.put(city2, ind++);
			}
			minHeap.offer(road);
		}
		int cityN = cityMap.size();

		// create a new Unionfind class to expand an MST
		Unionfind uf = new Unionfind(cityMap.size());
		List<List<Integer>> mst = new ArrayList<>();

		// check each road from the minHeap in the order of cost
		// stop until we have checked all the roads(minheap empty) / we have built a MST(mst.size() == cityN - 1)
		while(!minHeap.isEmpty() && mst.size() < cityN - 1) {
			List<Integer> minRoad = minHeap.poll();
			Integer city1 = minRoad.get(0), city2 = minRoad.get(1);
			// city 1, 2 not connected, union the 1, 2 cities, add road into mst 
			if(!uf.find(cityMap.get(city1), cityMap.get(city2))) {
				uf.union(cityMap.get(city1), cityMap.get(city2));
				mst.add(minRoad);
			}
		}
		// cannot build a spanning tree
		if(mst.size() < cityN - 1) {
			return new ArrayList<>();
		}
		// sort the roads in MST by cost
		Collections.sort(mst, (List<Integer> road1, List<Integer> road2) -> {
			// 1. compare cost
			int cmp = road1.get(2) - road2.get(2);
			if(cmp == 0) {
				// 2. compare city1
				int cmp1 = road1.get(0) - road2.get(0);
				if(cmp1 == 0) {
					// 3. compare city2
					return road1.get(1) - road2.get(1);
				}
				return cmp1;
			}
			return cmp;
		});
		return mst;
	}

	// 36. 零件组装 / merge subfiles (greedy 不一定连续)/ (dp 连续)merge consecutive stone piles min cost lc1000 https://leetcode.com/problems/minimum-cost-to-merge-stones/
	// input: 	an array of costs of merge each pile of stone; 
	//			K exact number of piles to merge in each round
	// return: 	the minimum cost to merge together all piles of stones （和1000不一样，不用merge连续的piles）
	//			return -1 if not possible to merge exact K piles each round
	// maintain a total coast variable; use a pq minheap to store all the piles, 
	//  	each round poll K least-cost piles to merge,
	// 		offer the new merged pile with new cost back to the pq again
	//  	until there is only one pile in the pq, return the total
	public static int mergeStones(int[] stones, int K) {
        // check if it is possible or not
        if(stones == null || stones.length == 0) {
            return -1;
        }
        int size = stones.length;
        while(size > K) {
            size = size - K + 1;
        }
        if(size != K) {
            return -1;
        }
        // it is possible
        // use a minHeap, each time poll K itmes; add cost; offer into again
        // maintain a total cost
        int cost = 0;
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        // offer all piles into the pq 
        for(int i = 0; i < stones.length; i++) {
            minHeap.offer(stones[i]);
        }
        while(!minHeap.isEmpty()) {
            // only one pile left 
            if(minHeap.size() == 1) {
                break;
            }
            // each level, poll K items
            int sum = 0;
            for(int i = 0; i < K; i++) {
                int num = minHeap.poll();
                sum += num;
            }
            // offer back the merged stone
            minHeap.offer(sum);
            cost += sum;
        }
        return cost;
    }
	// ===================================================================




	// 34. k closest points to origin (result no need in order) 
	// https://leetcode.com/problems/k-closest-points-to-origin/
	public static int[][] kClosest(int[][] points, int K) {
        int kClosest = 0;
        int[] dist = new int[points.length];
        for(int i = 0; i < points.length; i++) {
            dist[i] = getDist(points, i);
        }
        Arrays.sort(dist);
        kClosest = dist[K - 1];
        int[][] res = new int[K][2];
        int ind = 0;
        for(int i = 0; i < points.length; i++) {
            if(getDist(points, i) <= kClosest) {
                res[ind++] = points[i];
            }
        }
        return res;
    }
    private static int getDist(int[][] points, int i) {
        return points[i][0] * points[i][0] + points[i][1] * points[i][1];
    }


    // 34. k closest points to origin (pq : result in order)
	public static List<List<Integer>> kClosest(List<List<Integer>> points, int k) {
		if(points == null || points.size() == 0 || k == 0) {
			return new ArrayList<>();
		}
		PriorityQueue<List<Integer>> maxHeap = new PriorityQueue<List<Integer>>(k, (p1, p2) -> {
			return (int)(getDist(p2) - getDist(p1));
		});
		for(int i = 0; i < points.size(); i++) {
			List<Integer> point = points.get(i);
			if(i < k) {
				maxHeap.offer(point);
			} else {
				if(getDist(point) <= getDist(maxHeap.peek())) {
					maxHeap.poll();
					maxHeap.offer(point);
				}
			}
		}
		List<List<Integer>> res = new ArrayList<>(maxHeap);
		return res;
	}
	private static double getDist(List<Integer> point) {
		return point.get(0) * point.get(0) + point.get(1) * point.get(1);
	}
	// ===================================================================




	// 33. partition labels  https://leetcode.com/problems/partition-labels/submissions/
	// input:	a String needed partition
	//			partition the string into as many parts as possible 
	//			so that each letter appears in at most one partition
	// return:	list of size of paritions
	// greedy--
	// choose the smallest left-justified partition
    // for example, the string 'abadd'
    // for the first label 'a', the first partition should include 
    //     its first occurrance and last occurrance
    // So for each label, we extend the current partition,
    //     until we find the right most last occurrance of the letter
    //     we do the partition then.
    public static List<Integer> partitionLabels(String S) {
        List<Integer> res = new ArrayList<>();
        if(S == null || S.length() == 0) {
            return res;
        }
        int[] lastOcc = new int[26];
        for(int i = 0; i < S.length(); i++) {
            char c = S.charAt(i);
            lastOcc[c - 'a'] = i;
        }
        int start = 0, end = 0;
        for(int i = 0; i < S.length(); i++) {
            char c = S.charAt(i);
            end = Math.max(end, lastOcc[c - 'a']);
            if(i == end) {
                res.add(end - start + 1);
                end++;
                start = end;
            }
        }
        return res;
    }
	// ===================================================================



	// 32. log sort https://leetcode.com/problems/reorder-log-files/
	// test: 
	// 		String[] logs = {"a1 9 2 3 1", "g1 act car", "zo4 4 7", "ab1 off key dog", "a8 act zoo"};
	// 		System.out.println(Arrays.toString(reorderLogFiles(logs)));
	public static String[] reorderLogFiles(String[] logs) {
        if(logs == null || logs.length == 0) {
            return logs;
        }
        Arrays.sort(logs, (log1, log2) -> {
            String[] split1 = log1.split(" ", 2);
            String[] split2 = log2.split(" ", 2);
            boolean isLetter1 = Character.isLetter(split1[1].charAt(0));
            boolean isLetter2 = Character.isLetter(split2[1].charAt(0));
            if(isLetter1 && isLetter2) {
                int cmp = split1[1].compareTo(split2[1]);
                if(cmp == 0) {
                    return split1[0].compareTo(split2[0]);
                }
                return cmp;
            }
            return isLetter1 ? -1 : (isLetter2) ? 1 : 0;
        });
        return logs;
    }
	// ===================================================================



	// 22. maze
	// input: 	a maze 2d matrix
	// 			start from top-left corner to "9"
	// return: 	minimum steps
	// test:
	// 		int[][] maze1 = new int[][]{{1,0,0},{1,1,1},{1,9,1}};
	// 		int[][] maze2 = new int[][]{{9,0,0},{1,1,1},{1,0,1}};
	// 		System.out.println(minSteps(maze1));
	// BFS
	public static int minSteps(int[][] maze) {
		if(maze == null || maze.length == 0 || maze[0].length == 0) {
			return -1;
		}
		int nr = maze.length, nc = maze[0].length;
		int[][] dirs = new int[][]{{0,1},{0,-1},{1,0},{-1,0}};
		Set<List<Integer>> visited = new HashSet<>();
		Queue<List<Integer>> level = new LinkedList<>();
		List<Integer> origin = createPos(0, 0);
		level.offer(origin);
		visited.add(origin);
		int step = -1;
		while(!level.isEmpty()) {
			int size = level.size();
			step++;
			for(int i = 0; i < size; i++) {
				// poll one position from this level
				List<Integer> pos = level.poll();
				int row = pos.get(0), col = pos.get(1);
				// once we reach destination, return step number
				if(maze[row][col] == 9) {
					return step;
				}
				// add all the neighbors into the level queue
				for(int[] dir : dirs) {
					int newRow = row + dir[0], newCol = col + dir[1];
					if(newRow < nr && newRow >= 0 && newCol < nc && newCol >= 0 && maze[newRow][newCol] != 0) {
						List<Integer> newList = createPos(newRow, newCol);
						if(!visited.contains(newList)) {
							level.offer(newList);
							visited.add(newList);
						}
					}
				}
			}
		}
		return -1;
	}
	private static List<Integer> createPos(int i, int j) {
		List<Integer> list = new ArrayList<>();
		list.add(i);
		list.add(j);
		return list;
	}
	// ===================================================================




	// 27. max min path 
	// input: a 2d matrix
	// find the minimum in each path from top-left corner to bottom-right corner
	// and then find the maximum among these minimums.
	// return: a maximum number
	// test:
	// 		int[][] matrix = new int[][]{{8,4,7},{6,5,9}};
	// 		System.out.println(maxMinPath(matrix));		
	public static int maxMinPath(int[][] matrix){
		if(matrix == null || matrix.length == 0) {
			return 0;
		}
		int[] max = new int[]{Integer.MIN_VALUE};
		List<Integer> mins = new ArrayList<>();
		mins.add(Integer.MAX_VALUE);
		dfs(matrix, mins, max, 0, 0);
		return max[0];
	}
	private static void dfs(int[][] matrix, List<Integer> mins, int[] max, int i, int j) {
		int nr = matrix.length, nc = matrix[0].length;
		// out of bound, update max, return
		if(i >= nr || j >= nc) {
			return;
		}
		// update mins with current matrix[i][j]
		mins.add(Math.min(mins.get(mins.size() - 1), matrix[i][j]));
		// once reach the right-bottom corner, update the max, but do not return
		if(i == nr - 1 && j == nc - 1) {
			max[0] = Math.max(max[0], mins.get(mins.size() - 1));
		}
		// go to next step
		dfs(matrix, mins, max, i, j + 1);
		dfs(matrix, mins, max, i + 1, j);
		// backtrack
		mins.remove(mins.size() - 1);
	}
	// ===================================================================





	// 20. Binary Search Tree Min Sum Root to Leaf 
	// input: the root node of the tree
	// return: the maximum sum of a path from root to leaf
	// test:
	// 		TreeNode root = dserializeLevel("1,2,-3,null,null,4,5");
	// 		System.out.println(minSumRootLeaf(root));
	public static int minSumRootLeaf(TreeNode root){
		if(root == null) {
			return 0;
		}
		if(root.left != null && root.right == null) {
			return root.val + minSumRootLeaf(root.left);
		} else if(root.left == null && root.right != null) {
			return root.val + minSumRootLeaf(root.right);
		}
		return root.val + Math.min(minSumRootLeaf(root.left), minSumRootLeaf(root.right));
	}
	// ===================================================================


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


	// 12. Version 2. combination sum  
	// 40. Combination Sum II https://leetcode.com/problems/combination-sum-ii/submissions/
	// find all unique combinations in nums that sums to target.
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
	// input: an array of Points; an origin point; an integer k
	// return: an array of all k closest points, sorted by ascending distance
	// test:
	// 		Point[] array = new Point[]{new Point(212,0),new Point(0,2),new Point(1,0),new Point(3,1)};
	// 		Point ori = new Point(1,1);
	// 		Point[] res = kNearestPoint(array, ori, 3);
	// 		for(Point p : res) {
	// 			System.out.println(p.toString());
	// 		}
	public static Point[] kNearestPoint(Point[] array, Point origin, int k) {
		if(array == null || array.length == 0) {
			return new Point[0];
		}
		// use a max heap to maintain k closest points inside the heap
		PriorityQueue<Point> maxHeap = new PriorityQueue<>(k, (point1, point2) -> {
			return (int)(dist(point2, origin) - dist(point1, origin));
		});
		for(int i = 0; i < array.length; i++) {
			if(i < k) {
				maxHeap.offer(array[i]);
			} else {
				if(dist(maxHeap.peek(), origin) > dist(array[i], origin)){
					maxHeap.poll();
					maxHeap.offer(array[i]);
				}
			}
		}
		Point[] res = new Point[k];
		for(int i = k - 1; i >= 0; i--) {
			res[i] = maxHeap.poll();
		}
		return res;
	}
	private static double dist(Point p1, Point p2) {
		return (p1.x-p2.x) * (p1.x-p2.x) + (p1.y-p2.y) * (double)(p1.y-p2.y);
	}
	// ===================================================================




	// 9. two sum 
	// input: 	an unsorted array and a target number, no duplicates
	// return: 	once find a match to target number, return index + 1
	// 			o.w. {0, 0}
	// test:
	// 		int[] nums = {1,2,3,2,1,3};
	// 		int target = 4;
	// 		System.out.println(Arrays.toString(twoSum(nums, target)));
	public static int[] twoSum(int[] nums, int target) {
		if(nums == null || nums.length == 0) {
			return new int[]{0, 0};
		}
		Map<Integer, Integer> map = new HashMap<>();
		for(int i = 0; i < nums.length; i++) {
			// find complement
			int complement = target - nums[i];
			if(map.containsKey(complement)) {
				return new int[]{map.get(complement) + 1, i + 1};
			}
			// update map
			map.put(nums[i], i);
		}
		return new int[]{0, 0}; 
	}
	// ===================================================================


	// 9. two sum all pair I -- https://app.laicode.io/app/problem/181
	// input: an unsorted array and a target number
	// return: the count of all the matches for two sum
	// test: 
	// 		int[] nums = {1,2,5,3,4,5,2,6,7};
	// 		int target = 4;
	// 		System.out.println(twoSumI(nums, target));
	public static int twoSumI(int[] nums, int target) {
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
	// 9 ===================================================================

	// 36. twoSumMaxWeight
	// input: 	an array weights of all packages, a target weight
	// return: 	an array of two indices that the weights add up target,
	// 			if there are multiple answers, return the one with largest package weight
	// test:
	// 		int[] weights = {1,2,4,3,2,1,3,0};
	// 		int target = 4;
	// 		System.out.println(Arrays.toString(twoSumMaxWeight(weights, target)));
	// return: [2, 7]
	public static int[] twoSumMaxWeight(int[] weights, int target) {
		// find all pairs of indices
		// return the pair with largest item weight
		if(weights == null || weights.length == 0) {
			return new int[]{-1,-1};
		}
		Map<Integer, List<Integer>> map = new HashMap<>();
		List<List<Integer>> res = new ArrayList<>();
		for(int i = 0; i < weights.length; i++) {
			int complement = target - weights[i];
			// if contains complement
			// find a pair
			if(map.containsKey(complement)) {
				List<Integer> comList = map.get(complement);
				for(int j = 0; j < comList.size(); j++) {
					int comInd = comList.get(j);
					List<Integer> pair = new ArrayList<>();
					pair.add(comInd);
					pair.add(i);
					res.add(pair);
				}
			}
			// update item key's list
			if(!map.containsKey(weights[i])) {
				List<Integer> newList = new ArrayList<>();
				map.put(weights[i], newList);
			}
			map.get(weights[i]).add(i);
		}
		// find the result pair with largest item weight
		int[] result = new int[]{-1, -1};
		int maxItemWeight = 0;
		for(int i = 0; i < res.size(); i++) {
			List<Integer> pair = res.get(i);
			int item1Ind = pair.get(0), item2Ind = pair.get(1);
			if(weights[item1Ind] > maxItemWeight || weights[item2Ind] > maxItemWeight) {
				result[0] = item1Ind;
				result[1] = item2Ind;
			}
		}
		return result;
 	}
	// 36 ===================================================================


	// 35. memory consumption / flight 无人机 其实不是two sum的解法，这个是两个list 但是题目目的上像
	// input: 	two lists : jobs memory comsumptions / flight departure and return duration ;
	//			int : a memory limit / time limit
	// return:	max memory comsumption / flight duration that is not out of limit
	// test:
	// 		int[][] list1 = new int[][]{{1,100}, {2,200}, {3,500}};
	// 		int[][] list2 = new int[][]{{1,200}, {2,300}, {3,900}};
	// 		System.out.println(Arrays.toString(twoSumMax(list1, list2, 1000)));
	// 直接暴力解
	public static int[] twoSumMax(int[][] list1, int[][] list2, int limit) {
		if(list1 == null || list2 == null || list1.length == 0 || list2.length == 0) {
			return new int[]{-1,-1};
		}
		int maxSum = 0;
		int id1 = -1, id2 = -1;
		for(int i = 0; i < list1.length; i++) {
			for(int j = 0; j < list2.length; j++) {
				int sum = list1[i][1] + list2[j][1];
				if(sum < limit && sum > maxSum) {
					maxSum = sum;
					id1 = list1[i][0];
					id2 = list2[j][0];
				}
			}
		}
		return new int[]{id1, id2};
	}
	// 35 ===================================================================



	// 8. check is tree t has same values as tree s's subtree
	// https://leetcode.com/problems/subtree-of-another-tree/submissions/
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

	// 7. reverse from m to n : https://leetcode.com/problems/reverse-linked-list-ii/
	public static ListNode reverseBetween(ListNode head, int m, int n) {
        if(head == null || m == n || n == 1) {
            return head;
        }
        // split into three parts
        // record the tail1; head2, tail2; head3
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode tail1 = dummy;
        ListNode head2 = dummy; 
        ListNode tail2 = dummy; 
        ListNode head3 = dummy; 
        ListNode cur = dummy;
        int ind = 0;
        while(cur != null) {
            if(tail1 == dummy && ind + 1 == m) {
                tail1 = cur;
                head2 = cur.next;
            }
            if(tail2 == dummy && ind == n) {
                tail2 = cur;
                head3 = cur.next;
            }
            cur = cur.next;
            ind++;
        }
        // reverse head2 - tail2
        ListNode prev = null;
        cur = head2;
        tail2.next = null;
        while(cur != null) {
            ListNode next = cur.next;
            cur.next = prev;
            prev = cur;
            cur = next;
        }
        // combine three parts together
        tail1.next = prev;
        head2.next = head3;
        
        return dummy.next;
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


	// 5. longestParlin1 
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
		return (start == end) ? "WTF" : s.substring(start, end + 1);
	}
	// 5. =======================================

	// 4. valid parenthesis
	// return -1 if it is not valid parenthesis string, 
	// o.w. return num of pairs
	// test: 	String s = "(())()";
	// 			System.out.println(validParen(s));
	// also can use a variable to count "(" instead of maintaining a stack
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

