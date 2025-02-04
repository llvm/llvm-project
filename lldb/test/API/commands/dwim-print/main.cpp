extern "C" int puts(const char *s);

struct Structure {
  int number = 30;
  void f() { puts("break inside"); }
};

struct Wrapper {
  Structure s;
};

struct Opaque;

int main(int argc, char **argv) {
  Structure s;
  Wrapper w;
  Wrapper *wp = &w;
  Opaque *opaque = (Opaque *)(void *)&s;
  puts("break here");
  s.f();
  return 0;
}
