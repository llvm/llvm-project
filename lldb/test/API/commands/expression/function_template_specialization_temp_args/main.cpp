template <typename T> struct M {};

template <typename T> void f(T &t);

template <> void f<int>(int &t) {
  typedef M<int> VType;

  VType p0; // break here
}

int main() {
  int x;

  f(x);

  return 0;
}
