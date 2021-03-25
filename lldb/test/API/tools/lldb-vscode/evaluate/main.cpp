#include "foo.h"

static int static_int = 42;

int non_static_int = 43;

int a_function(int var) {
  return var; // breakpoint 3
}

struct my_struct {
  int foo;
};

int main(int argc, char const *argv[]) {
  my_struct struct1 = {15};
  my_struct *struct2 = new my_struct{16};
  int var1 = 20;
  int var2 = 21;
  int var3 = static_int; // breakpoint 1
  {
    int non_static_int = 10;
    int var2 = 2;
    int var3 = non_static_int; // breakpoint 2
  }
  a_function(var3);
  foo_func();
  return 0;
}
