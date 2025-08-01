#include <__verbose_abort>
#include <vector>

// Some expressons from the test need this symbol to be compiled when libcxx is
// built statically.
void *libcpp_verbose_abort_ptr = (void *)&std::__libcpp_verbose_abort;

struct Foo {
  int a;
};

int main(int argc, char **argv) {
  std::vector<Foo> a = {{3}, {1}, {2}};
  return 0; // Set break point at this line.
}
