#define C c
#define D 1
#define E (C != a) && (C > a)
#define F E

void __attribute__((noinline)) func1(void) { return; }

void __attribute__((noinline)) func(int a, int b, int c) {
  if (a && D && E || b)
    func1();
  if (b && D)
    func1();
  if (a && (b && C) || (D && F))
    func1();
}

int main() {
  func(2, 3, 3);
  return 0;
}
