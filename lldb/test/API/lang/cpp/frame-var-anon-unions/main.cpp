int main() {
  union {
    int i;
    char c;
  };
  i = 0xFFFFFF00;
  c = 'A';
  return c; // break here
}
