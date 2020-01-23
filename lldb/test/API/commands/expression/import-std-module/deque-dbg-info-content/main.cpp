#include <deque>

struct Foo {
  int a;
};

int main() { int argc = 0; char **argv = (char **)0; 
  std::deque<Foo> a = {{3}, {1}, {2}};
  return 0; // Set break point at this line.
}
