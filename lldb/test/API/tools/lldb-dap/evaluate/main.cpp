#include "foo.h"

#include <cstdint>
#include <map>
#include <vector>

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
  my_struct *struct3 = nullptr;
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

  std::vector<int> my_vec;
  my_vec.push_back(1);
  my_vec.push_back(2);
  my_vec.push_back(3); // breakpoint 4

  std::map<int, int> my_map;
  my_map[1] = 2;
  my_map[2] = 3;
  my_map[3] = 4; // breakpoint 5

  std::vector<bool> my_bool_vec;
  my_bool_vec.push_back(true);
  my_bool_vec.push_back(false); // breakpoint 6
  my_bool_vec.push_back(true);  // breakpoint 7

  uint8_t my_ints[] = {5, 10, 15, 20, 25, 30};
  return 0; // breakpoint 8
}
