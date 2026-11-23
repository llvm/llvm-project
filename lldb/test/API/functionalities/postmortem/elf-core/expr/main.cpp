struct Inner {
  Inner(int val) : val(val) {}
  int val;
};

struct Outer {
  Outer(int val) : inner(val) {}
  Inner inner;
};

extern "C" void _start(void) {
  Outer outer(5);
  char *boom = (char *)0;
  *boom = 47;
}
