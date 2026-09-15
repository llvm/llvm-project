#include <filesystem>

int main() {
  std::filesystem::path p("dir/file.txt");
  std::filesystem::path empty;
  std::filesystem::path abs_win("C:\\tmp\\file.txt");
  std::filesystem::path abs_unix("/usr/local/lib/file.txt");
  std::filesystem::path extensionless("README");
  return 0; // break here
}
