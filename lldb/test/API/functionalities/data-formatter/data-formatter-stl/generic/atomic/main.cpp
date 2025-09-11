#include <atomic>

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

int main() {
  std::atomic<S> s;
  s.store(S());
  std::atomic<int> i;
  i.store(5);

  Parent p;
  // Let the child node know what its parent is.
  p.child.parent = &p;

  return 0; // Set break point at this line.
}
