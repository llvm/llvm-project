template <typename T> struct TestObj {
  int f;
  T g;
};

int main() {
  TestObj<int> t{42, 21};
  return t.f + t.g; // break here
}
