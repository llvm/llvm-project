struct Foo;

extern void lib1_func(Foo *);
extern void lib2_func(Foo *);

int main() {
  lib1_func(nullptr);
  lib2_func(nullptr);
  return 0;
}
