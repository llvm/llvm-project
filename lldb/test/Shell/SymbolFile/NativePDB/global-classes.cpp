// clang-format off
// REQUIRES: lld, x86

// Test that we can display tag types.
// RUN: %clang_cl --target=x86_64-windows-msvc -Od -Z7 \
// RUN:   -Xclang -fkeep-static-consts -c /Fo%t.obj -- %s
// RUN: lld-link -debug:full -nodefaultlib -entry:main %t.obj -out:%t.exe -pdb:%t.pdb
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/globals-classes.lldbinit | FileCheck %s

enum class EnumType : unsigned {
  A = 1,
  B = 2
};

class ClassNoPadding {
  /* [ 0] */ unsigned char a = 86;
  /* [ 1] */ char b = 'a';
  /* [ 2] */ bool c = false;
  /* [ 3] */ bool d = true;
  /* [ 4] */ short e = -1234;
  /* [ 6] */ unsigned short f = 8123;
  /* [ 8] */ unsigned int g = 123908;
  /* [12] */ int h = -890234;
  /* [16] */ unsigned long i = 2908234;
  /* [20] */ long j = 7234890;
  /* [24] */ float k = 908234.12392;
  /* [28] */ EnumType l = EnumType::A;
  /* [32] */ double m = 23890.897423;
  /* [40] */ unsigned long long n = 23490782;
  /* [48] */ long long o = -923409823;
  /* [56] */ int p[5] = { 2, 3, 5, 8, 13 };
};

class ClassWithPadding {
  /* [ 0] */ char a = '0';
  //         char padding[1];
  /* [ 2] */ short b = 50;
  /* [ 4] */ char c[2] = { '0', '1' };
  //         char padding[2];
  /* [ 8] */ int d = 100;
  /* [12] */ char e = '0';
  //         char padding[3];
  /* [16] */ int f = 200;
  //         char padding[4];
  /* [24] */ long long g = 300;
  /* [32] */ char h[3] = { '0', '1', '2' };
  //         char padding[5];
  /* [40] */ long long i = 400;
  /* [48] */ char j[2] = { '0', '1' };
  //         char padding[6];
  /* [56] */ long long k = 500;
  /* [64] */ char l = '0';
  //         char padding[7];
  /* [72] */ long long m = 600;
} ;

struct EmptyBase {};

template<typename T>
struct BaseClass {
  constexpr BaseClass(int N)
    : BaseMember(N) {}

  int BaseMember;
};

struct DerivedClass : public BaseClass<int> {
  constexpr DerivedClass(int Base, int Derived)
    : BaseClass(Base), DerivedMember(Derived) {}

  int DerivedMember;
};

struct EBO : public EmptyBase {
  constexpr EBO(int N) : Member(N) {}
  int Member;
};

struct PaddedBases : public BaseClass<char>, public BaseClass<short>, BaseClass<int> {
  constexpr PaddedBases(char CH, short S, int N, long long D)
    : BaseClass<char>(CH), BaseClass<short>(S), BaseClass<int>(N), DerivedMember(D) {}
  long long DerivedMember;
};

struct Statics {
  static char a;
  static bool b;
  static short c;
  static unsigned short d;
  static unsigned int e;
  static int f;
  static unsigned long g;
  static long h;
  static float i;
  static EnumType j;
  static double k;
  static unsigned long long l;
  static long long m;
};

char Statics::a = 'a';
bool Statics::b = true;
short Statics::c = 1234;
unsigned short Statics::d = 2345;
unsigned int Statics::e = 3456;
int Statics::f = 4567;
unsigned long Statics::g = 5678;
long Statics::h = 6789;
float Statics::i = 7890.1234;
EnumType Statics::j = EnumType::A;
double Statics::k = 8901.2345;
unsigned long long Statics::l = 9012345;
long long Statics::m = 1234567;


struct Pointers {
  void *a = nullptr;
  char *b = &Statics::a;
  bool *c = &Statics::b;
  short *e = &Statics::c;
  unsigned short *f = &Statics::d;
  unsigned int *g = &Statics::e;
  int *h = &Statics::f;
  unsigned long *i = &Statics::g;
  long *j = &Statics::h;
  float *k = &Statics::i;
  EnumType *l = &Statics::j;
  double *m = &Statics::k;
  unsigned long long *n = &Statics::l;
  long long *o = &Statics::m;
};

struct References {
  char &a = Statics::a;
  bool &b = Statics::b;
  short &c = Statics::c;
  unsigned short &d = Statics::d;
  unsigned int &e = Statics::e;
  int &f = Statics::f;
  unsigned long &g = Statics::g;
  long &h = Statics::h;
  float &i = Statics::i;
  EnumType &j = Statics::j;
  double &k = Statics::k;
  unsigned long long &l = Statics::l;
  long long &m = Statics::m;
};


constexpr ClassWithPadding ClassWithPaddingInstance;
// CHECK:      (lldb) target variable -T ClassWithPaddingInstance
// CHECK-NEXT: (const ClassWithPadding) ClassWithPaddingInstance = {
// CHECK-NEXT:   (char) a = '0'
// CHECK-NEXT:   (short) b = 50
// CHECK-NEXT:   (char [2]) c = "01"
// CHECK-NEXT:   (int) d = 100
// CHECK-NEXT:   (char) e = '0'
// CHECK-NEXT:   (int) f = 200
// CHECK-NEXT:   (long long) g = 300
// CHECK-NEXT:   (char [3]) h = "012"
// CHECK-NEXT:   (long long) i = 400
// CHECK-NEXT:   (char [2]) j = "01"
// CHECK-NEXT:   (long long) k = 500
// CHECK-NEXT:   (char) l = '0'
// CHECK-NEXT:   (long long) m = 600
// CHECK-NEXT: }

constexpr ClassNoPadding ClassNoPaddingInstance;
// CHECK:      (lldb) target variable -T ClassNoPaddingInstance
// CHECK-NEXT: (const ClassNoPadding) ClassNoPaddingInstance = {
// CHECK-NEXT:   (unsigned char) a = 'V'
// CHECK-NEXT:   (char) b = 'a'
// CHECK-NEXT:   (bool) c = false
// CHECK-NEXT:   (bool) d = true
// CHECK-NEXT:   (short) e = -1234
// CHECK-NEXT:   (unsigned short) f = 8123
// CHECK-NEXT:   (unsigned int) g = 123908
// CHECK-NEXT:   (int) h = -890234
// CHECK-NEXT:   (unsigned long) i = 2908234
// CHECK-NEXT:   (long) j = 7234890
// CHECK-NEXT:   (float) k = 908234.125
// CHECK-NEXT:   (EnumType) l = A
// CHECK-NEXT:   (double) m = 23890.897422999999
// CHECK-NEXT:   (unsigned long long) n = 23490782
// CHECK-NEXT:     (long long) o = -923409823
// CHECK-NEXT:     (int [5]) p = {
// CHECK-NEXT:       (int) [0] = 2
// CHECK-NEXT:       (int) [1] = 3
// CHECK-NEXT:       (int) [2] = 5
// CHECK-NEXT:       (int) [3] = 8
// CHECK-NEXT:       (int) [4] = 13
// CHECK-NEXT:     }
// CHECK-NEXT:   }

constexpr DerivedClass DC(10, 20);
// CHECK:      (lldb) target variable -T DC
// CHECK-NEXT: (const DerivedClass) DC = {
// CHECK-NEXT:   (BaseClass<int>) BaseClass<int> = {
// CHECK-NEXT:     (int) BaseMember = 10
// CHECK-NEXT:   }
// CHECK-NEXT:   (int) DerivedMember = 20
// CHECK-NEXT: }

constexpr EBO EBOC(20);
// CHECK:      (lldb) target variable -T EBOC
// CHECK-NEXT: (const EBO) EBOC = {
// CHECK-NEXT:   (int) Member = 20
// CHECK-NEXT: }

constexpr PaddedBases PBC('a', 12, 120, 1200);
// CHECK:      (lldb) target variable -T PBC
// CHECK-NEXT: (const PaddedBases) PBC = {
// CHECK-NEXT:   (BaseClass<char>) BaseClass<char> = {
// CHECK-NEXT:     (int) BaseMember = 97
// CHECK-NEXT:   }
// CHECK-NEXT:   (BaseClass<short>) BaseClass<short> = {
// CHECK-NEXT:     (int) BaseMember = 12
// CHECK-NEXT:   }
// CHECK-NEXT:   (BaseClass<int>) BaseClass<int> = {
// CHECK-NEXT:     (int) BaseMember = 120
// CHECK-NEXT:   }
// CHECK-NEXT:   (long long) DerivedMember = 1200
// CHECK-NEXT: }

constexpr struct {
  int x = 12;
  EBO EBOC{ 42 };
} UnnamedClassInstance;
// CHECK:      (lldb) target variable -T UnnamedClassInstance
// CHECK-NEXT: (const <unnamed-type-UnnamedClassInstance>) UnnamedClassInstance = {
// CHECK-NEXT:   (int) x = 12
// CHECK-NEXT:   (EBO) EBOC = {
// CHECK-NEXT:     (int) Member = 42
// CHECK-NEXT:   }
// CHECK-NEXT: }

constexpr Pointers PointersInstance;
// CHECK:      (lldb) target variable -T PointersInstance
// CHECK-NEXT: (const Pointers) PointersInstance = {
// CHECK-NEXT:   (void *) a = {{.*}}
// CHECK-NEXT:   (char *) b = {{.*}}
// CHECK-NEXT:   (bool *) c = {{.*}}
// CHECK-NEXT:   (short *) e = {{.*}}
// CHECK-NEXT:   (unsigned short *) f = {{.*}}
// CHECK-NEXT:   (unsigned int *) g = {{.*}}
// CHECK-NEXT:   (int *) h = {{.*}}
// CHECK-NEXT:   (unsigned long *) i = {{.*}}
// CHECK-NEXT:   (long *) j = {{.*}}
// CHECK-NEXT:   (float *) k = {{.*}}
// CHECK-NEXT:   (EnumType *) l = {{.*}}
// CHECK-NEXT:   (double *) m = {{.*}}
// CHECK-NEXT:   (unsigned long long *) n = {{.*}}
// CHECK-NEXT:   (long long *) o = {{.*}}
// CHECK-NEXT: }
constexpr References ReferencesInstance;
// CHECK:      (lldb) target variable -T ReferencesInstance
// CHECK-NEXT: (const References) ReferencesInstance = {
// CHECK-NEXT:   (char &) a = {{.*}}
// CHECK-NEXT:   (bool &) b = {{.*}}
// CHECK-NEXT:   (short &) c = {{.*}}
// CHECK-NEXT:   (unsigned short &) d = {{.*}}
// CHECK-NEXT:   (unsigned int &) e = {{.*}}
// CHECK-NEXT:   (int &) f = {{.*}}
// CHECK-NEXT:   (unsigned long &) g = {{.*}}
// CHECK-NEXT:   (long &) h = {{.*}}
// CHECK-NEXT:   (float &) i = {{.*}}
// CHECK-NEXT:   (EnumType &) j = {{.*}}
// CHECK-NEXT:   (double &) k = {{.*}}
// CHECK-NEXT:   (unsigned long long &) l = {{.*}}
// CHECK-NEXT:   (long long &) m = {{.*}}
// CHECK-NEXT: }

// CHECK: Dumping clang ast for 1 modules.
// CHECK: TranslationUnitDecl {{.*}}
// CHECK: |-CXXRecordDecl {{.*}} class ClassWithPadding definition
// CHECK: | |-FieldDecl {{.*}} a 'char'
// CHECK: | |-FieldDecl {{.*}} b 'short'
// CHECK: | |-FieldDecl {{.*}} c 'char [2]'
// CHECK: | |-FieldDecl {{.*}} d 'int'
// CHECK: | |-FieldDecl {{.*}} e 'char'
// CHECK: | |-FieldDecl {{.*}} f 'int'
// CHECK: | |-FieldDecl {{.*}} g 'long long'
// CHECK: | |-FieldDecl {{.*}} h 'char [3]'
// CHECK: | |-FieldDecl {{.*}} i 'long long'
// CHECK: | |-FieldDecl {{.*}} j 'char [2]'
// CHECK: | |-FieldDecl {{.*}} k 'long long'
// CHECK: | |-FieldDecl {{.*}} l 'char'
// CHECK: | `-FieldDecl {{.*}} m 'long long'
// CHECK: |-VarDecl {{.*}} ClassWithPaddingInstance 'const ClassWithPadding'
// CHECK: |-CXXRecordDecl {{.*}} class ClassNoPadding definition
// CHECK: | |-FieldDecl {{.*}} a 'unsigned char'
// CHECK: | |-FieldDecl {{.*}} b 'char'
// CHECK: | |-FieldDecl {{.*}} c 'bool'
// CHECK: | |-FieldDecl {{.*}} d 'bool'
// CHECK: | |-FieldDecl {{.*}} e 'short'
// CHECK: | |-FieldDecl {{.*}} f 'unsigned short'
// CHECK: | |-FieldDecl {{.*}} g 'unsigned int'
// CHECK: | |-FieldDecl {{.*}} h 'int'
// CHECK: | |-FieldDecl {{.*}} i 'unsigned long'
// CHECK: | |-FieldDecl {{.*}} j 'long'
// CHECK: | |-FieldDecl {{.*}} k 'float'
// CHECK: | |-FieldDecl {{.*}} l 'EnumType'
// CHECK: | |-FieldDecl {{.*}} m 'double'
// CHECK: | |-FieldDecl {{.*}} n 'unsigned long long'
// CHECK: | |-FieldDecl {{.*}} o 'long long'
// CHECK: | `-FieldDecl {{.*}} p 'int [5]'
// CHECK: |-VarDecl {{.*}} ClassNoPaddingInstance 'const ClassNoPadding'
// CHECK: |-EnumDecl {{.*}} EnumType
// CHECK: | |-EnumConstantDecl {{.*}} A 'EnumType'
// CHECK: | `-EnumConstantDecl {{.*}} B 'EnumType'
// CHECK: |-CXXRecordDecl {{.*}} struct DerivedClass definition
// CHECK: | |-public 'BaseClass<int>'
// CHECK: | |-FieldDecl {{.*}} DerivedMember 'int'
// CHECK: | `-CXXConstructorDecl {{.*}} DerivedClass 'void (int, int)'
// CHECK: |   |-ParmVarDecl {{.*}} 'int'
// CHECK: |   `-ParmVarDecl {{.*}} 'int'
// CHECK: |-VarDecl {{.*}} DC 'const DerivedClass'
// CHECK: |-CXXRecordDecl {{.*}} struct BaseClass<int> definition
// CHECK: | |-FieldDecl {{.*}} BaseMember 'int'
// CHECK: | `-CXXMethodDecl {{.*}} BaseClass 'void (int)'
// CHECK: |   `-ParmVarDecl {{.*}} 'int'
// CHECK: |-CXXRecordDecl {{.*}} struct EBO definition
// CHECK: | |-public 'EmptyBase'
// CHECK: | |-FieldDecl {{.*}} Member 'int'
// CHECK: | `-CXXConstructorDecl {{.*}} EBO 'void (int)'
// CHECK: |   `-ParmVarDecl {{.*}} 'int'
// CHECK: |-VarDecl {{.*}} EBOC 'const EBO'
// CHECK: |-CXXRecordDecl {{.*}} struct EmptyBase definition
// CHECK: |-CXXRecordDecl {{.*}} struct PaddedBases definition
// CHECK: | |-public 'BaseClass<char>'
// CHECK: | |-public 'BaseClass<short>'
// CHECK: | |-public 'BaseClass<int>'
// CHECK: | |-FieldDecl {{.*}} DerivedMember 'long long'
// CHECK: | `-CXXConstructorDecl {{.*}} PaddedBases 'void (char, short, int, long long)'
// CHECK: |   |-ParmVarDecl {{.*}} 'char'
// CHECK: |   |-ParmVarDecl {{.*}} 'short'
// CHECK: |   |-ParmVarDecl {{.*}} 'int'
// CHECK: |   `-ParmVarDecl {{.*}} 'long long'
// CHECK: |-VarDecl {{.*}} PBC 'const PaddedBases'
// CHECK: |-CXXRecordDecl {{.*}} struct BaseClass<char> definition
// CHECK: | |-FieldDecl {{.*}} BaseMember 'int'
// CHECK: | `-CXXMethodDecl {{.*}} BaseClass 'void (int)'
// CHECK: |   `-ParmVarDecl {{.*}} 'int'
// CHECK: |-CXXRecordDecl {{.*}} struct BaseClass<short> definition
// CHECK: | |-FieldDecl {{.*}} BaseMember 'int'
// CHECK: | `-CXXMethodDecl {{.*}} BaseClass 'void (int)'
// CHECK: |   `-ParmVarDecl {{.*}} 'int'
// CHECK: |-CXXRecordDecl {{.*}} struct <unnamed-type-UnnamedClassInstance> definition
// CHECK: | |-FieldDecl {{.*}} x 'int'
// CHECK: | `-FieldDecl {{.*}} EBOC 'EBO'
// CHECK: |-VarDecl {{.*}} UnnamedClassInstance 'const <unnamed-type-UnnamedClassInstance>'
// CHECK: |-CXXRecordDecl {{.*}} struct Pointers definition
// CHECK: | |-FieldDecl {{.*}} a 'void *'
// CHECK: | |-FieldDecl {{.*}} b 'char *'
// CHECK: | |-FieldDecl {{.*}} c 'bool *'
// CHECK: | |-FieldDecl {{.*}} e 'short *'
// CHECK: | |-FieldDecl {{.*}} f 'unsigned short *'
// CHECK: | |-FieldDecl {{.*}} g 'unsigned int *'
// CHECK: | |-FieldDecl {{.*}} h 'int *'
// CHECK: | |-FieldDecl {{.*}} i 'unsigned long *'
// CHECK: | |-FieldDecl {{.*}} j 'long *'
// CHECK: | |-FieldDecl {{.*}} k 'float *'
// CHECK: | |-FieldDecl {{.*}} l 'EnumType *'
// CHECK: | |-FieldDecl {{.*}} m 'double *'
// CHECK: | |-FieldDecl {{.*}} n 'unsigned long long *'
// CHECK: | `-FieldDecl {{.*}} o 'long long *'
// CHECK: |-VarDecl {{.*}} PointersInstance 'const Pointers'
// CHECK: |-CXXRecordDecl {{.*}} struct References definition
// CHECK: | |-FieldDecl {{.*}} a 'char &'
// CHECK: | |-FieldDecl {{.*}} b 'bool &'
// CHECK: | |-FieldDecl {{.*}} c 'short &'
// CHECK: | |-FieldDecl {{.*}} d 'unsigned short &'
// CHECK: | |-FieldDecl {{.*}} e 'unsigned int &'
// CHECK: | |-FieldDecl {{.*}} f 'int &'
// CHECK: | |-FieldDecl {{.*}} g 'unsigned long &'
// CHECK: | |-FieldDecl {{.*}} h 'long &'
// CHECK: | |-FieldDecl {{.*}} i 'float &'
// CHECK: | |-FieldDecl {{.*}} j 'EnumType &'
// CHECK: | |-FieldDecl {{.*}} k 'double &'
// CHECK: | |-FieldDecl {{.*}} l 'unsigned long long &'
// CHECK: | `-FieldDecl {{.*}} m 'long long &'
// CHECK: `-VarDecl {{.*}} ReferencesInstance 'const References'

int main(int argc, char **argv) {
  return 0;
}
