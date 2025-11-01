#include <__verbose_abort>
#include <vector>

// Some expressons from the test need this symbol to be compiled when libcxx is
// built statically.
void *libcpp_verbose_abort_ptr = (void *)&std::__libcpp_verbose_abort;

int main(int argc, char **argv) {
  std::vector<std::vector<int> > a = {{1, 2, 3}, {3, 2, 1}};
  return 0; // Set break point at this line.
}
