struct Child {
  int x;
};

struct Parent {
  Child child;
};

int main() {
  Parent parent = {{1}};
  return 0; // break here
}
