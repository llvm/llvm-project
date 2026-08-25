namespace ns {
class Calculator {
public:
  int add(int a, int b) { return a + b; }
  int mul(int a, int b) { return a * b; }
};
}

int entry(int x, int y) {
  ns::Calculator c;
  return c.add(x, y) * c.mul(x, y);
}
