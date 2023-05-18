#include <iterator>
#include <vector>

int main() {
  std::vector<int> v{1, 2, 3};
  auto move_begin = std::make_move_iterator(v.begin());
  auto move_end = std::make_move_iterator(v.end());
  return 0; // Set break point at this line.
}
