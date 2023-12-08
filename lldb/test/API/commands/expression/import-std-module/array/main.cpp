#include <array>

struct DbgInfo {
  int v = 4;
};

int main(int argc, char **argv) {
  std::array<int, 3> a = {3, 1, 2};
  std::array<DbgInfo, 1> b{DbgInfo()};
  return 0; // Set break point at this line.
}
