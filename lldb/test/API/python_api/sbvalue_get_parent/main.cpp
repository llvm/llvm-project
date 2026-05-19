#include <vector>

struct Child {
  int x;
};

struct Parent {
  Child child;
};

int main() {
  Parent parent = {{1}};
  std::vector<int> vec = {10, 20, 30};
  return 0; // break here
}
