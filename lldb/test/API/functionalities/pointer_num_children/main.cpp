struct Inner {
  int a;
  int b;
};

struct Outer {
  Inner *inner;
};

int main() {
  Inner inner{42, 56};
  Outer outer{&inner};
  Inner **Ptr = &(outer.inner);
  Inner *&Ref = outer.inner;
  return 0; // break here
}
