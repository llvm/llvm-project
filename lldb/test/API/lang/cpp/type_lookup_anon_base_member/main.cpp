struct Base {
  int b;
  struct {
    int x;
  };
};

struct Derived : public Base {
  int a;
};

int main() {
  Derived d;
  d.b = 1;
  d.x = 2;
  d.a = 3;
  return 0; // Set breakpoint here
}
