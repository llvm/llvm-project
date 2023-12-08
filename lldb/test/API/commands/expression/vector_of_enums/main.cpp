#include <vector>

enum E {
a,
b,
c,
d
} ;

int main() {
  std::vector<E> v = {E::a, E::b, E::c};

  return v.size(); // break here
}
