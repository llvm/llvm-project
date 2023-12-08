#include <tuple>
#include <string>

int main() {
  std::tuple<> empty;
  std::tuple<int> one_elt{47};
  std::tuple<int, long, std::string> three_elts{1, 47l, "foo"};
  return 0; // break here
}
