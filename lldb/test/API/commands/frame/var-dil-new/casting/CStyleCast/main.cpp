// CStyleCast, main.cpp

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>

namespace ns {

typedef int myint;

class Foo {};

namespace inner {

using mydouble = double;

class Foo {};

}  // namespace inner

}  // namespace ns

/*
void TestUniquePtr() {
  struct NodeU {
    std::unique_ptr<NodeU> next;
    int value;
  };
  auto ptr_node = std::unique_ptr<NodeU>(new NodeU{nullptr, 2});
  ptr_node = std::unique_ptr<NodeU>(new NodeU{std::move(ptr_node), 1});

  std::unique_ptr<char> ptr_null;
  auto ptr_int = std::make_unique<int>(1);
  auto ptr_float = std::make_unique<float>(1.1f);

  auto deleter = [](void const* data) { delete static_cast<int const*>(data); }\
;
  std::unique_ptr<void, decltype(deleter)> ptr_void(new int(42), deleter);

  // BREAK(TestUniquePtr)
  // BREAK(TestUniquePtrDeref)
  // BREAK(TestUniquePtrCompare)
}
*/

int main (int argc, char** argv) {
  int a = 1;
  int* ap = &a;
  void* vp = &a;
  int arr[2] = {1, 2};

  int na = -1;
  float f = 1.1;

  typedef int myint;
  std::nullptr_t std_nullptr_t = nullptr;
  bool found_it = false;
  if (std_nullptr_t) {
    found_it = true;
  } else {
    found_it = (bool) 0;
  }


  myint myint_ = 1;
  ns::myint ns_myint_ = 2;
  ns::Foo ns_foo_;
  ns::Foo* ns_foo_ptr_ = &ns_foo_;

  ns::inner::mydouble ns_inner_mydouble_ = 1.2;
  ns::inner::Foo ns_inner_foo_;
  ns::inner::Foo* ns_inner_foo_ptr_ = &ns_inner_foo_;

  float finf = std::numeric_limits<float>::infinity();
  float fnan = std::numeric_limits<float>::quiet_NaN();
  float fsnan = std::numeric_limits<float>::signaling_NaN();
  float fmax = std::numeric_limits<float>::max();
  float fdenorm = std::numeric_limits<float>::denorm_min();

  // BREAK(TestCStyleCastBuiltins)
  // BREAK(TestCStyleCastBasicType)
  // BREAK(TestCStyleCastPointer)
  // BREAK(TestCStyleCastNullptrType)
  if (false) { // Set a breakpoint here
  }

  struct InnerFoo {
    int a;
    int b;
  };

  InnerFoo ifoo;
  (void)ifoo;

  int arr_1d[] = {1, 2, 3, 4};
  int arr_2d[2][3] = {{1, 2, 3}, {4, 5, 6}};

  // BREAK(TestCStyleCastArray)
  // BREAK(TestCStyleCastReference)
  return 0; // Set a breakpoint here
}
