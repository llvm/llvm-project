#include <atomic>
#include <version>

#if __cpp_lib_atomic_shared_ptr
#include <memory>
#endif // __cpp_lib_atomic_shared_ptr

// Define a Parent and Child struct that can point to each other.
class Parent;
struct Child {
  // This should point to the parent which in turn owns this
  // child instance. This cycle should not cause LLDB to infinite loop
  // during printing.
  std::atomic<Parent *> parent{nullptr};
};
struct Parent {
  Child child;
};

struct S {
  int x = 1;
  int y = 2;
};

typedef float vector4float __attribute__((__vector_size__(16)));

static void func() {}
using func_t = void (*)();

int main() {
  std::atomic<S> s;
  s.store(S());
  std::atomic<int> i;
  i.store(5);

  Parent p;
  // Let the child node know what its parent is.
  p.child.parent = &p;

  // libstdcpp has different layout depending on the data structure
  std::atomic<bool> atomic_bool{true};
#if __cpp_lib_atomic_float
  std::atomic<float> atomic_float{3.14};
  std::atomic<double> atomic_double{6.28};
#endif // __cpp_lib_atomic_float

  // The type system treats these as floating point types as well, but they
  // use the generic std::atomic layout.
#ifdef __GLIBCXX__
  // Other standard libraries cannot instantiate std::atomic<_Complex float>.
  _Complex float complex_value;
  __real__ complex_value = 1.5f;
  __imag__ complex_value = 2.5f;
  std::atomic<_Complex float> atomic_complex_float;
  atomic_complex_float.store(complex_value);
#endif // __GLIBCXX__
  std::atomic<vector4float> atomic_vector_float{vector4float{1, 2, 3, 4}};

  std::atomic<func_t> atomic_func(func);

  S data;
  std::atomic<S *> atomic_pointer{&data};

  // atomic smart pointers were added in c++20
#if __cpp_lib_atomic_shared_ptr
  std::shared_ptr<int> s_value = std::make_shared<int>(300);
  std::atomic<std::shared_ptr<int>> atomic_shared{s_value};
  std::atomic<std::weak_ptr<int>> atomic_weak{s_value};
#endif // __cpp_lib_atomic_shared_ptr

  return 0; // Set break point at this line.
}
