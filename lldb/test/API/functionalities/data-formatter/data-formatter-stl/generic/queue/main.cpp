#include <queue>
#include <stack>
#include <vector>

int main() {
  std::queue<int> q1{{1, 2, 3, 4, 5}};
  std::queue<int, std::vector<int>> q2{{1, 2, 3, 4, 5}};
  std::stack<int> s1{{1, 2, 3, 4, 5}};
  std::stack<int, std::vector<int>> s2{{1, 2, 3, 4, 5}};
  std::priority_queue<int> pq;
  for (int v : {1, 2, 3, 4, 5})
    pq.push(v);
  int ret = q1.size() + q2.size() + s1.size() + s2.size() + pq.size();
  return ret; // break here
}
