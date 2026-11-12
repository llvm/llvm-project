#include "lib.h"

struct Local : public virtual Foo {
  Local();
  ~Local();
  int y;
};

Local::Local() : Foo(5) { y = x; }
Local::~Local() {}

int main() {
  Foo f(5);
  Base b1;
  Bar b2;
  Local l1;
  return f.method();
}
