class Trivial {
public:
  Trivial(int input) : m_int(input) {}
private:
  int m_int;
};

class Foo {
private:
  Trivial m_trivial = Trivial(100); // Set the before constructor breakpoint here

public:
  Foo(int input) {
    ++input;
  }

private:
  Trivial m_other_trivial = Trivial(200); // Set the after constructor breakpoint here
};

int main() {
  Foo myFoo(10); // Set a breakpoint here to get started
  return 0;
}
