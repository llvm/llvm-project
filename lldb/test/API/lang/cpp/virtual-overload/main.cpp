// Test that lldb doesn't get confused by an overload of a virtual
// function of the same name.
struct Base {
  virtual void f(int i) {}
  virtual ~Base() {}
};

struct Derived : Base {
  virtual void f(int i, int j) {}
};

int main(int argc, char **argv) {
  Derived obj;
  obj.f(1, 2); //% self.expect("fr var", "not crashing", substrs = ["obj"])
  return 0;
}

