#include <memory>
#include <string>

int main() {
  std::tuple<int> ti{1};
  std::tuple<std::string> ts{"foobar"};
  std::tuple<int, std::string, int> tt{1, "baz", 2};
  return 0; // Set break point at this line.
}
