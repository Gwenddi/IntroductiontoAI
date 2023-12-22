package SearchAlgorithm;

import java.awt.Point;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class AStarSearch {
    private static final char WALL = '%';
    private static final char START = 'P';
    private static final char END = '.';
    // 定义上下左右四个方向
    private static final int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
    private static final char[][] maze = readMazeFromFile(
            "C:\\Users\\yueyue\\Desktop\\人工智能导论\\SearchAlgorithm\\mediumMaze.txt");

    static class Node {
        Point position;
        Node parent;
        int g; // 从起始节点走到当前节点的路径长度
        int h; // 从当前节点到目标节点的估计代价
        int f; // 下一步探索的节点的估计代价 f=g+h，A*一般会选择最小的f值的节点作为下一步搜索

        Node(Point position, Node parent, int g, int h) {
            this.position = position;
            this.parent = parent;
            this.g = g;
            this.h = h;
        }

    }

    public static void main(String[] args) {
        // 找到终点和起点
        Point start = findStart();
        Point end = findEnd();
        // 找到路径
        List<Point> path = findPath(start, end);

        if (path != null) {
            System.out.println("找到路径！");
            printPath(path);
        } else {
            System.out.println("未找到路径！");
        }
    }

    /**
     * 从txt文件中读取迷宫
     * 
     * @param filename
     * @return
     */
    public static char[][] readMazeFromFile(String filename) {
        List<String> lines = new ArrayList<>();
        try {
            FileReader fileReader = new FileReader(filename);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String line;
            while ((line = bufferedReader.readLine()) != null) {
                lines.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        char[][] maze = new char[lines.size()][lines.get(0).length()];

        for (int i = 0; i < lines.size(); i++) {
            maze[i] = lines.get(i).toCharArray();
        }

        return maze;
    }

    private static Point findStart() {
        for (int i = 0; i < maze.length; i++) {
            for (int j = 0; j < maze[i].length; j++) {
                if (maze[i][j] == START) {
                    return new Point(i, j);
                }
            }
        }
        return null;
    }

    private static Point findEnd() {
        for (int i = 0; i < maze.length; i++) {
            for (int j = 0; j < maze[i].length; j++) {
                if (maze[i][j] == END) {
                    return new Point(i, j);
                }
            }
        }
        return null;
    }

    private static List<Point> findPath(Point start, Point end) {
        Set<Point> visited = new HashSet<>();
        // 可以保证每次都是按照节点的f值的大小排序
        PriorityQueue<Node> queue = new PriorityQueue<>(Comparator.comparingInt(a -> a.f));

        queue.add(new Node(start, null, 0, heuristic(start, end)));

        while (!queue.isEmpty()) {
            Node current = queue.poll(); // 去除并删除优先级队列中的头部元素

            // 遍历路径
            if (current.position.equals(end)) {
                List<Point> path = new ArrayList<>();
                while (current.parent != null) {
                    path.add((current.position));
                    current = current.parent;
                }
                // 反转元素顺序
                Collections.reverse(path);
                return path;
            }

            visited.add(current.position);
            for (int[] dir : directions) {
                int newRow = current.position.x + dir[0];
                int newCol = current.position.y + dir[1];

                if (isValid(newRow, newCol) && maze[newRow][newCol] != WALL) {
                    Point newPosition = new Point(newRow, newCol);
                    if (!visited.contains(newPosition)) {
                        int g = current.g + 1;
                        int h = heuristic(newPosition, end);
                        Node newNode = new Node(newPosition, current, g, h);
                        newNode.f = g + h;
                        queue.add(newNode);
                    }
                }
            }
        }
        return null;
    }

    /**
     * 使用曼哈顿距离作为启发式函数
     * 
     * @param a
     * @param b
     * @return 两个节点之间的曼哈顿距离
     */
    private static int heuristic(Point a, Point b) {
        return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
    }

    /*
     * 判断当前节点是否在迷宫中
     */
    private static boolean isValid(int r, int c) {
        return r >= 0 && r < maze.length && c >= 0 && c < maze[r].length;
    }

    /**
     * 将路径用^标注出来
     * 
     * @param path
     */
    private static void printPath(List<Point> path) {
        for (Point p : path) {
            if (maze[p.x][p.y] != END)
                maze[p.x][p.y] = '^';
        }
        for (char[] row : maze) {
            System.out.println(row);
        }
    }
}
