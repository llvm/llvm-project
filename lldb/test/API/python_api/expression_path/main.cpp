struct StructA {
  int x;
  int y;
};

struct StructB {
  int x;
  StructA &a_ref;
  StructA *&a_ptr_ref;
};

struct StructC : public StructB {
  int y;

  StructC(int x, StructA &a_ref, StructA *&a_ref_ptr, int y)
      : StructB{x, a_ref, a_ref_ptr}, y(y) {}
};

int main() {
  StructA a{1, 2};
  StructA *a_ptr = &a;

  StructB b{3, a, a_ptr};
  StructB *b_ptr = &b;
  StructB &b_ref = b;
  StructB *&b_ptr_ref = b_ptr;

  StructC c(4, a, a_ptr, 5);
  StructC *c_ptr = &c;
  StructC &c_ref = c;
  StructC *&c_ptr_ref = c_ptr;

  return 0; // Set breakpoint here
}