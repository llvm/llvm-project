#include <system_error>

int main() {
  std::error_code ec =
      std::make_error_code(std::errc::no_such_file_or_directory);
  std::error_condition econd = std::errc::no_such_file_or_directory;
  std::error_code default_ec;
  return 0; // break here
}
