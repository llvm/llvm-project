#define CLASS(NAME)                             \
  class NAME {                                  \
  public:                                       \
    struct Inner;                               \
    Inner *i = nullptr;                         \
  };                                            \
NAME::Inner &getInner##NAME();

CLASS(A)
CLASS(B)
CLASS(C)
CLASS(D)
CLASS(E)
CLASS(F)
CLASS(G)

int main()
{
  A::Inner &inner_a = getInnerA();
  B::Inner &inner_b = getInnerB();
  C::Inner &inner_c = getInnerC();
  D::Inner &inner_d = getInnerD();
  E::Inner &inner_e = getInnerE();
  F::Inner &inner_f = getInnerF();
  G::Inner &inner_g = getInnerG();

  return 0; // break here
}
