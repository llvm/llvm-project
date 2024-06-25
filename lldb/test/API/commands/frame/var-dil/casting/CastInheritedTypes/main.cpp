int main(int argc, char** argv) {
  struct CxxEmpty {};
  struct CxxA {
    short a;
  };
  struct CxxB {
    long long b;
  };
  struct CxxC : CxxEmpty, CxxA, CxxB {
    int c;
  };
  struct CxxD {
    long long d;
  };
  struct CxxE : CxxD, CxxC {
    int e;
  };

  CxxA a{1};
  CxxB b{2};
  CxxC c;
  c.a = 3;
  c.b = 4;
  c.c = 5;
  CxxD d{6};
  CxxE e;
  e.a = 7;
  e.b = 8;
  e.c = 9;
  e.d = 10;
  e.e = 11;

  struct CxxVC : virtual CxxA, virtual CxxB {
    int c;
  };
  struct CxxVE : CxxD, CxxVC {
    int e;
  };

  CxxVC vc;
  vc.a = 12;
  vc.b = 13;
  vc.c = 14;
  CxxVE ve;
  ve.a = 15;
  ve.b = 16;
  ve.c = 17;
  ve.d = 18;
  ve.e = 19;

  CxxB* e_as_b = &e;
  CxxB* ve_as_b = &ve;

  // BREAK(TestCastBaseToDerived)
  // BREAK(TestCastDerivedToBase)
  return 0; // Set a breakpoint here
}
