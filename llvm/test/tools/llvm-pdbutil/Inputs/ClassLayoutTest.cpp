// Compile with "cl /c /Zi /GR- ClassLayoutTest.cpp"
// Link with "link ClassLayoutTest.obj /debug /nodefaultlib /entry:main"

namespace MembersTest {
  class A {
  public:
    typedef int NestedTypedef;
    enum NestedEnum {
      NestedEnumValue1
    };

    void MemberFunc() {}

  private:
    int IntMemberVar;
    double DoubleMemberVar;
  };
}

namespace GlobalsTest {
  int IntVar;
  double DoubleVar;
  
  typedef int Typedef;
  enum Enum {
    Val1
  } EnumVar;
  Typedef TypedefVar;
}

namespace BaseClassTest {
  class A {};
  class B : public virtual A {};
  class C : public virtual A {};
  class D : protected B, private C {};
}

namespace UdtKindTest {
  struct A {};
  class B {};
  union C {};
}

namespace BitFieldTest {
  struct A {
    int Bits1 : 1;
    int Bits2 : 2;
    int Bits3 : 3;
    int Bits4 : 4;
    int Bits22 : 22;
    int Offset0x04;
  };
};

int main(int argc, char **argv) {
  MembersTest::A v1;
  v1.MemberFunc();
  BaseClassTest::D v2;
  UdtKindTest::A v3;
  UdtKindTest::B v4;
  UdtKindTest::C v5;
  BitFieldTest::A v7;
  return 0;
}
