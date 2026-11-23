#include <system_error>

int main() {
  std::error_code ec(2, std::generic_category());
  std::error_condition econd(7, std::generic_category());
  std::error_code negative(-1, std::generic_category());
  std::error_code default_ec;
  return 0; // break here
}
