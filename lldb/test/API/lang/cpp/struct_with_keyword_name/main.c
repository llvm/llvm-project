struct class {
  int class;
};

int main() {
  struct class constexpr;
  constexpr.class = 3;
  return constexpr.class; // break here
}
