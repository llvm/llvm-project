#include <__verbose_abort>
#include <array>

// Some expressons from the test need this symbol to be compiled when libcxx is
// built statically.
void *libcpp_verbose_abort_ptr = (void *)&std::__libcpp_verbose_abort;

struct DbgInfo {
  int v = 4;
};

int main(int argc, char **argv) {
  std::array<int, 3> a = {3, 1, 2};
  std::array<DbgInfo, 1> b{DbgInfo()};
  return 0; // Set break point at this line.
}
