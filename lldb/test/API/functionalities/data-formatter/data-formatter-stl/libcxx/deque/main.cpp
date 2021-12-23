#include <deque>

int main() {
  std::deque<int> empty;
  std::deque<int> deque_1 = {1};
  std::deque<int> deque_3 = {3, 1, 2};
  return empty.size() + deque_1.front() + deque_3.front(); // break here
}
