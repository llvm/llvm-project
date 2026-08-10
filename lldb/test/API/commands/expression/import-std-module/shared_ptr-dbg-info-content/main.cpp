#include <memory>

struct Foo {
  int a;
};

int main(int argc, char **argv) {
  std::shared_ptr<Foo> s(new Foo);
  s->a = 3;
  return s->a; // Set break point at this line.
}
