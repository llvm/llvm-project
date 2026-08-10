namespace std {
namespace detail {
void function_that_aborts() { __builtin_verbose_trap("Bounds error", "out-of-bounds access"); }
} // namespace detail

inline namespace __1 {
template <typename T> struct vector {
  void operator[](unsigned) { detail::function_that_aborts(); }
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
