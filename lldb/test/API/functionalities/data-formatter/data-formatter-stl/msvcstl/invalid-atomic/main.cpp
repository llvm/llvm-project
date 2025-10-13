namespace std {
template <typename T> struct atomic {
  int foo;
  int bar;
};
} // namespace std

int main() {
  std::atomic<int> a{1, 2};
  std::atomic<void> b{3, 4};

  return 0; // Set break point at this line.
}
