namespace ignore {
template <typename T> auto auto_ret(T x) { return 0; }
[[gnu::abi_tag("test")]] int with_tag() { return 0; }
template <typename T> [[gnu::abi_tag("test")]] int with_tag_template() {
  return 0;
}

template <typename T> decltype(auto) decltype_auto_ret(T x) { return 0; }
} // namespace ignore

int main() {
  auto v1 = ignore::auto_ret<int>(5);
  auto v2 = ignore::with_tag();
  auto v3 = ignore::decltype_auto_ret<int>(5);
  auto v4 = ignore::with_tag_template<int>();
}
