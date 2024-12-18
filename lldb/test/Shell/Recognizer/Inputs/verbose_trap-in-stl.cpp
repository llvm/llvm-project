namespace std {
inline namespace __1 {
template <typename T> struct vector {
  T& operator[](unsigned) { __builtin_verbose_trap("Bounds error", "out-of-bounds access"); }
};
} // namespace __1
} // namespace std

void g() {
  std::vector<int> v;
  v[10];
}

int main() {
  g();
  return 0;
}
