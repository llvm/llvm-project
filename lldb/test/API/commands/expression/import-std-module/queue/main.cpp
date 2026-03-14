#include <deque>
#include <list>
#include <queue>

struct C {
  // Constructor for testing emplace.
  C(int i) : i(i) {};
  int i;
};

int main(int argc, char **argv) {
  // std::deque is the default container.
  std::queue<C> q_deque({{1}});
  std::queue<C, std::list<C>> q_list({{1}});
  return 0; // Set break point at this line.
}
