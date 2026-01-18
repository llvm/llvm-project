// Type Casting, main.cpp

#include <limits>

namespace ns {

typedef int myint;

class Foo {};

} // namespace ns

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

  float finf = std::numeric_limits<float>::infinity();
  float fnan = std::numeric_limits<float>::quiet_NaN();
  float fsnan = std::numeric_limits<float>::signaling_NaN();
  float fmax = std::numeric_limits<float>::max();
  float fdenorm = std::numeric_limits<float>::denorm_min();

  int arr_1d[] = {1, 2, 3, 4};
  int arr_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};

  return 0; // Set a breakpoint here
}
