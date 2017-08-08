// Build with "cl.exe /Zi /GR- /GX- every-type.cpp /link /debug /nodefaultlib /entry:main"

// clang-format off
void *__purecall = 0;

void __cdecl operator delete(void *,unsigned int) {}

struct FooStruct { };                      // LF_STRUCTURE

class FooClass {                           // LF_CLASS
                                           // LF_FIELDLIST
  enum NestedEnum {                        // LF_ENUM
                                           // LF_NESTTYPE
    A, B, C                                // LF_ENUMERATE
  };

  void RegularMethod() {}                  // LF_ARGLIST
                                           // LF_ONEMETHOD
                                           // LF_MFUNCTION

  void OverloadedMethod(int) {}            // LF_METHODLIST
                                           // LF_METHOD
  void OverloadedMethod(int, int) {}

  int HiNibble : 4;                        // LF_BITFIELD
  int LoNibble : 4;
  NestedEnum EnumVariable;                 // LF_MEMBER
  static void *StaticMember;               // LF_POINTER
                                           // LF_STMEMBER
};

void *FooClass::StaticMember = nullptr;

class Inherit : public FooClass {           // LF_BCLASS
public:
  virtual ~Inherit() {}                     // LF_VTSHAPE
                                            // LF_VFUNCTAB
};

class VInherit : public virtual FooClass {  // LF_VBCLASS

};

class IVInherit : public VInherit {         // LF_IVBCLASS
};

union TheUnion {
  int X;                                    // LF_UNION
};

int SomeArray[7] = {1, 2, 3, 4, 5, 6, 7};   // LF_ARRAY

int main(int argc, char **argv) {           // LF_PROCEDURE
  const int X = 7;                          // LF_MODIFIER

  FooStruct FooStructInstance;
  FooClass FooClassInstance;
  Inherit InheritInstance;
  VInherit VInheritInstance;
  IVInherit IVInheritInstance;
  TheUnion UnionInstance;
  return SomeArray[argc];
}
