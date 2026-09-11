#include <expected>

int main() {
  std::expected<int, int> ok = 7;
  std::expected<int, int> err = std::unexpected(42);
  std::expected<void, int> void_ok;
  std::expected<void, int> void_err = std::unexpected(11);
  std::expected<int, int> &ok_ref = ok;
  std::expected<int, int> &err_ref = err;

  return ok.value(); // break here
}
