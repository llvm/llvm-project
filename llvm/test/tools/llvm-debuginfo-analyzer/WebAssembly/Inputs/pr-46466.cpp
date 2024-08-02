struct Struct {
  union Union {
    enum NestedEnum { RED, BLUE };
  };
  Union U;
};

Struct S;
int test() {
  return S.U.BLUE;
}
