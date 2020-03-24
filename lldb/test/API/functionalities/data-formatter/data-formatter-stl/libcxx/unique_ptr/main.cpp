#include <cstdio>
#include <memory>
#include <string>

int main() {
  std::unique_ptr<int> up_empty;
  std::unique_ptr<int> up_int = std::make_unique<int>(10);
  std::unique_ptr<std::string> up_str = std::make_unique<std::string>("hello");
  std::unique_ptr<int> &up_int_ref = up_int;
  std::unique_ptr<int> &&up_int_ref_ref = std::make_unique<int>(10);

  return 0; // break here
}
