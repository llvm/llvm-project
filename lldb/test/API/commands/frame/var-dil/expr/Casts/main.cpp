// Type Casting, main.cpp

#include <limits>
#include <cstddef>

namespace ns {

typedef int myint;

class Foo {};

namespace inner {

using mydouble = double;

class Foo {};

} // namespace inner

} // namespace ns

// Global variable
bool myGlobalName = true;

int main(int argc, char **argv) {
  int a = 1;
  int *ap = &a;
  void *vp = &a;
  int arr[2] = {1, 2};

  int na = -1;
  float f = 1.1;

  typedef int myint;
  std::nullptr_t std_nullptr_t = nullptr;
  bool found_it = false;
  if (std_nullptr_t) {
    found_it = true;
  } else {
    found_it = (bool)0;
  }

  myint myint_ = 1;
  ns::myint ns_myint_ = 2;
  ns::Foo ns_foo_;
  ns::Foo *ns_foo_ptr_ = &ns_foo_;

  ns::inner::mydouble ns_inner_mydouble_ = 1.2;
  ns::inner::Foo ns_inner_foo_;
  ns::inner::Foo *ns_inner_foo_ptr_ = &ns_inner_foo_;

  float finf = std::numeric_limits<float>::infinity();
  float fnan = std::numeric_limits<float>::quiet_NaN();
  float fsnan = std::numeric_limits<float>::signaling_NaN();
  float fmax = std::numeric_limits<float>::max();
  float fdenorm = std::numeric_limits<float>::denorm_min();

  struct InnerFoo {
    int a;
    int b;
  };

  InnerFoo ifoo;
  (void)ifoo;

  int arr_1d[] = {1, 2, 3, 4};
  int arr_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};

  struct myName {
    int x;
    int y;
  };

  struct myName myStruct = {98, 99};
  int myName = 37;

  struct myGlobalName {
    int m;
    bool bval;
  };

  struct myGlobalName secondStruct = {42, false};

  return 0; // Set a breakpoint here
}
