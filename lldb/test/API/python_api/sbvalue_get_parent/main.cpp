#include <vector>

struct Child {
  int x;
};

struct Parent {
  Child child;
};

struct MyContainer {
  int *data;
  unsigned count;
};

int main() {
  Parent parent = {{1}};
  std::vector<int> vec = {10, 20, 30};
  int arr[] = {100, 200, 300};
  MyContainer container = {arr, 3};
  return 0; // break here
}
