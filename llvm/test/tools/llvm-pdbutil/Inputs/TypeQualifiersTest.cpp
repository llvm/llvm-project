// Compile with "cl /c /Zi /GR- TypeQualifiersTest.cpp"
// Link with "link TypeQualifiersTest.obj /debug /nodefaultlib /entry:main"

union Union {
  int * __restrict x_member;
  float * __restrict y_member;
  int* volatile __restrict m_volatile;
  const char* m_const;
};

int f(const volatile int* __restrict arg_crv) {
  Union u;
  return 1;
}

void g(int& __restrict arg_ref) {
}

namespace NS {
  class Class {
  public:
    int get() const { return 1;}
    int set() __restrict { return 2; }
    void help() volatile { return; }
  };

  struct Foo {
    int a;
    int b;
    int func(int x) __restrict { return 1; }
  };

  Foo s = { 10 };

  const int* __restrict p_object = &s.a;

  volatile int Foo:: * __restrict p_data_member = &Foo::a;

  int (Foo::* p_member_func)(int) __restrict = &Foo::func;
}

typedef long* __restrict RestrictTypedef;
RestrictTypedef RestrictVar;

typedef volatile int* __restrict RankNArray[10][100];
RankNArray ArrayVar;

int main() {
  NS::Class ClassVar;
  ClassVar.get();
  ClassVar.help();
  ClassVar.set();

  return 0;
}
