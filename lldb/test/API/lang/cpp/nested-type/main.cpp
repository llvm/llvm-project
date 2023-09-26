struct PointerIntPairInfo {
  enum MaskAndShiftConstants : unsigned long {
    PointerBitMask = 42,
  };

  union B {};
  B b;

  struct C {};
  C c;

  int a{};
};

static unsigned long foo() {
  return PointerIntPairInfo::PointerBitMask;
}

int main()
{
    PointerIntPairInfo p;
    return p.a + foo(); // breakpoint 1
}