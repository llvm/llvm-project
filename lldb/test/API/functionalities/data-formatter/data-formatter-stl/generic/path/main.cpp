#include <filesystem>

int main() {
  std::filesystem::path p("dir/file.txt");
  std::filesystem::path empty;
  std::filesystem::path abs_win("C:\\tmp\\file.txt");
  return 0; // break here
}
