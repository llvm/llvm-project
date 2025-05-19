namespace std {
void recursively_aborts(int depth) {
  if (depth == 0)
    __builtin_verbose_trap("Error", "max depth");

  recursively_aborts(--depth);
}
} // namespace std

int main() {
  std::recursively_aborts(256);
  return 0;
}
