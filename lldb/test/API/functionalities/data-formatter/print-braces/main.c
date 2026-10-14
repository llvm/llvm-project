struct Inner {
  int x;
  int y;
};

struct Outer {
  struct Inner inner;
  int z;
};

int main(void) {
  struct Outer s = {{1, 2}, 3};
  return s.z; // break here
}
