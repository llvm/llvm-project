#include <string>
#include <tuple>

int main() {
  std::tuple<> empty;
  std::tuple<int> one_elt{47};
  std::tuple<std::string> string_elt{"foobar"};
  std::tuple<int, long, std::string> three_elts{1, 47l, "foo"};
  return 0; // break here
}
