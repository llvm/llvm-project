struct Base {
  int b;
  struct {
    int x;
    int y;
  };
};

struct Derived : public Base {
  int d;
};

int main() {
  Derived d;
  d.b = 1;
  d.x = 2;
  d.d = 3;
  d.y = 4;
  return 0; // Set breakpoint here
}
