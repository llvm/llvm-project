int main() {
  union {
    int i;
    char c;
  };
  struct {
    int x;
    char y;
    short z;
  } s{3,'B',14};
  i = 0xFFFFFF00;
  c = 'A';
  return c; // break here
}
