template <typename T> struct Base {
  Base(T &t) : ref(t), pointer(&t) {}
  // Try referencing `Derived` via different ways to potentially make LLDB
  // pull in the definition (which would recurse back to this base class).
  T &ref;
  T *pointer;
  T func() { return ref; }
};

struct Derived : Base<Derived> {
  Derived() : Base<Derived>(*this) {}
  int member = 0;
};

Derived derived;

int main() { return derived.member; }
