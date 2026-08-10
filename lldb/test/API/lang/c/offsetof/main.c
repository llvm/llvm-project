#include <stdint.h>

struct Foo {
  int8_t a;
  int16_t b;
};

int main (int argc, char const *argv[]) {
    struct Foo f;
    return f.a; //% self.expect("expr offsetof(Foo, a)", substrs = ['= 0'])
                //% self.expect("expr offsetof(Foo, b)", substrs = ['= 2'])
}
