namespace std {
template <typename T> struct atomic {
  int foo;
  int bar;
};

namespace __1 {
template <typename T> struct atomic {
  int foo;
  int bar;
};
} // namespace __1
} // namespace std

int main() {
  std::atomic<int> a{1, 2};
  std::atomic<void> b{3, 4};

  std::__1::atomic<int> c{5, 6};
  std::__1::atomic<void> d{7, 8};

  return 0; // Set break point at this line.
}
