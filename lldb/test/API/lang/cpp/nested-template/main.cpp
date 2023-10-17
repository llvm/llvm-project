struct Outer {
  Outer() {}

  template <class T>
  struct Inner {};
};

int main() {
  Outer::Inner<int> oi;
}
