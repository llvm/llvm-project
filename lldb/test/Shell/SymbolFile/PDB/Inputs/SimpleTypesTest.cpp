// typedef
typedef unsigned long ULongArrayTypedef[10];
ULongArrayTypedef ULongArrayVar;

typedef long double*& RefTypedef;
long double* LongDoublePtrVar = 0;
RefTypedef RefVar = LongDoublePtrVar;

typedef long long (*FuncPtrTypedef)(int&, unsigned char**, short[], const double, volatile bool);
FuncPtrTypedef FuncVar;

typedef char (*VarArgsFuncTypedef)(void*, long, unsigned short, unsigned int, ...);
VarArgsFuncTypedef VarArgsFuncVar;

typedef float (*VarArgsFuncTypedefA)(...);
VarArgsFuncTypedefA VarArgsFuncVarA;

// unscoped enum
enum Enum { RED, GREEN, BLUE };
Enum EnumVar;

enum EnumConst { LOW, NORMAL = 10, HIGH };
EnumConst EnumConstVar;

enum EnumEmpty {};
EnumEmpty EnumEmptyVar;

enum EnumUChar : unsigned char { ON, OFF, AUTO };
EnumUChar EnumCharVar;

// scoped enum
enum class EnumClass { YES, NO, DEFAULT };
EnumClass EnumClassVar;

enum struct EnumStruct { red, blue, black };
EnumStruct EnumStructVar;

typedef signed char SCharTypedef;
SCharTypedef SCVar;

typedef char16_t WChar16Typedef;
WChar16Typedef WC16Var;

typedef char32_t WChar32Typedef;
WChar32Typedef WC32Var;

typedef wchar_t WCharTypedef;
WCharTypedef WCVar;

int main() {
  return 0;
}
