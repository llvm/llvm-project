#include <memory>

struct Foo {
  int a;
};

int main() { int argc = 0; char **argv = (char **)0; 
  std::shared_ptr<Foo> s(new Foo);
  s->a = 3;
  return s->a; // Set break point at this line.
}
