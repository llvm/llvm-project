template <typename> struct pair {};
struct A {
  using iterator = pair<char *>;
  pair<char *> a_[];
};
struct B {
  using iterator = A::iterator;
  iterator begin();
  A *tag_set_;
};
B b;
int main() {};
