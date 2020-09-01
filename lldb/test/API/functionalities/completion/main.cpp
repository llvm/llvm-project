#include <iostream>

class Foo
{
public:
    int Bar(int x, int y)
    {
        return x + y;
    }
};

namespace { int Quux (void) { return 0; } }

struct Container { int MemberVar; };

int main(int argc, char *argv[]) {
  if (argc > 1 && std::string(argv[1]) == "-x")
    std::cin.get();

  Foo fooo;
  Foo *ptr_fooo = &fooo;
  fooo.Bar(1, 2);

  Container container;
  Container *ptr_container = &container;
  int q = Quux();
  return container.MemberVar = 3; // Break here
}
