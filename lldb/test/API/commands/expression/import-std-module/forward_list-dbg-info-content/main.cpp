#include <forward_list>

struct Foo {
  int a;
};

int main(int argc, char **argv) {
  std::forward_list<Foo> a = {{3}, {1}, {2}};
  return 0; // Set break point at this line.
}
