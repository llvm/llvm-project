//===- MicrosoftDemangle.cpp ----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a demangler for MSVC-style mangled symbols.
//
// This file has no dependencies on the rest of LLVM so that it can be
// easily reused in other programs such as libcxxabi.
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"

#include "Compiler.h"
#include "StringView.h"
#include "Utility.h"

#include <cctype>
#include <cstdio>
#include <tuple>

// This memory allocator is extremely fast, but it doesn't call dtors
// for allocated objects. That means you can't use STL containers
// (such as std::vector) with this allocator. But it pays off --
// the demangler is 3x faster with this allocator compared to one with
// STL containers.
namespace {
  constexpr size_t AllocUnit = 4096;

class ArenaAllocator {
  struct AllocatorNode {
    uint8_t *Buf = nullptr;
    size_t Used = 0;
    size_t Capacity = 0;
    AllocatorNode *Next = nullptr;
  };

  void addNode(size_t Capacity) {
    AllocatorNode *NewHead = new AllocatorNode;
    NewHead->Buf = new uint8_t[Capacity];
    NewHead->Next = Head;
    NewHead->Capacity = Capacity;
    Head = NewHead;
    NewHead->Used = 0;
  }

public:
  ArenaAllocator() { addNode(AllocUnit); }

  ~ArenaAllocator() {
    while (Head) {
      assert(Head->Buf);
      delete[] Head->Buf;
      AllocatorNode *Next = Head->Next;
      delete Head;
      Head = Next;
    }
  }

  char *allocUnalignedBuffer(size_t Length) {
    uint8_t *Buf = Head->Buf + Head->Used;

    Head->Used += Length;
    if (Head->Used > Head->Capacity) {
      // It's possible we need a buffer which is larger than our default unit
      // size, so we need to be careful to add a node with capacity that is at
      // least as large as what we need.
      addNode(std::max(AllocUnit, Length));
      Head->Used = Length;
      Buf = Head->Buf;
    }

    return reinterpret_cast<char *>(Buf);
  }

  template <typename T, typename... Args> T *alloc(Args &&... ConstructorArgs) {

    size_t Size = sizeof(T);
    assert(Head && Head->Buf);

    size_t P = (size_t)Head->Buf + Head->Used;
    uintptr_t AlignedP =
        (((size_t)P + alignof(T) - 1) & ~(size_t)(alignof(T) - 1));
    uint8_t *PP = (uint8_t *)AlignedP;
    size_t Adjustment = AlignedP - P;

    Head->Used += Size + Adjustment;
    if (Head->Used < Head->Capacity)
      return new (PP) T(std::forward<Args>(ConstructorArgs)...);

    addNode(AllocUnit);
    Head->Used = Size;
    return new (Head->Buf) T(std::forward<Args>(ConstructorArgs)...);
  }

private:
  AllocatorNode *Head = nullptr;
};
} // namespace

static bool startsWithDigit(StringView S) {
  return !S.empty() && std::isdigit(S.front());
}

// Writes a space if the last token does not end with a punctuation.
static void outputSpaceIfNecessary(OutputStream &OS) {
  if (OS.empty())
    return;

  char C = OS.back();
  if (isalnum(C) || C == '>')
    OS << " ";
}

// Storage classes
enum Qualifiers : uint8_t {
  Q_None = 0,
  Q_Const = 1 << 0,
  Q_Volatile = 1 << 1,
  Q_Far = 1 << 2,
  Q_Huge = 1 << 3,
  Q_Unaligned = 1 << 4,
  Q_Restrict = 1 << 5,
  Q_Pointer64 = 1 << 6
};

enum class StorageClass : uint8_t {
  None,
  PrivateStatic,
  ProtectedStatic,
  PublicStatic,
  Global,
  FunctionLocalStatic,
};

enum class QualifierMangleMode { Drop, Mangle, Result };

enum class PointerAffinity { Pointer, Reference, RValueReference };

// Calling conventions
enum class CallingConv : uint8_t {
  None,
  Cdecl,
  Pascal,
  Thiscall,
  Stdcall,
  Fastcall,
  Clrcall,
  Eabi,
  Vectorcall,
  Regcall,
};

enum class ReferenceKind : uint8_t { None, LValueRef, RValueRef };

// Types
enum class PrimTy : uint8_t {
  Unknown,
  None,
  Function,
  Ptr,
  MemberPtr,
  Array,

  Struct,
  Union,
  Class,
  Enum,

  Void,
  Bool,
  Char,
  Schar,
  Uchar,
  Char16,
  Char32,
  Short,
  Ushort,
  Int,
  Uint,
  Long,
  Ulong,
  Int64,
  Uint64,
  Wchar,
  Float,
  Double,
  Ldouble,
  Nullptr,
  Vftable,
  Vbtable,
  LocalStaticGuard
};

enum class OperatorTy : uint8_t {
  Ctor,                    // ?0 # Foo::Foo()
  Dtor,                    // ?1 # Foo::~Foo()
  New,                     // ?2 # operator new
  Delete,                  // ?3 # operator delete
  Assign,                  // ?4 # operator=
  RightShift,              // ?5 # operator>>
  LeftShift,               // ?6 # operator<<
  LogicalNot,              // ?7 # operator!
  Equals,                  // ?8 # operator==
  NotEquals,               // ?9 # operator!=
  ArraySubscript,          // ?A # operator[]
  Conversion,              // ?B # Foo::operator <type>()
  Pointer,                 // ?C # operator->
  Dereference,             // ?D # operator*
  Increment,               // ?E # operator++
  Decrement,               // ?F # operator--
  Minus,                   // ?G # operator-
  Plus,                    // ?H # operator+
  BitwiseAnd,              // ?I # operator&
  MemberPointer,           // ?J # operator->*
  Divide,                  // ?K # operator/
  Modulus,                 // ?L # operator%
  LessThan,                // ?M operator<
  LessThanEqual,           // ?N operator<=
  GreaterThan,             // ?O operator>
  GreaterThanEqual,        // ?P operator>=
  Comma,                   // ?Q operator,
  Parens,                  // ?R operator()
  BitwiseNot,              // ?S operator~
  BitwiseXor,              // ?T operator^
  BitwiseOr,               // ?U operator|
  LogicalAnd,              // ?V operator&&
  LogicalOr,               // ?W operator||
  TimesEqual,              // ?X operator*=
  PlusEqual,               // ?Y operator+=
  MinusEqual,              // ?Z operator-=
  DivEqual,                // ?_0 operator/=
  ModEqual,                // ?_1 operator%=
  RshEqual,                // ?_2 operator>>=
  LshEqual,                // ?_3 operator<<=
  BitwiseAndEqual,         // ?_4 operator&=
  BitwiseOrEqual,          // ?_5 operator|=
  BitwiseXorEqual,         // ?_6 operator^=
  Vftable,                 // ?_7 # vftable
  Vbtable,                 // ?_8 # vbtable
  Vcall,                   // ?_9 # vcall
  Typeof,                  // ?_A # typeof
  LocalStaticGuard,        // ?_B # local static guard
  StringLiteral,           // ?_C # string literal
  VbaseDtor,               // ?_D # vbase destructor
  VecDelDtor,              // ?_E # vector deleting destructor
  DefaultCtorClosure,      // ?_F # default constructor closure
  ScalarDelDtor,           // ?_G # scalar deleting destructor
  VecCtorIter,             // ?_H # vector constructor iterator
  VecDtorIter,             // ?_I # vector destructor iterator
  VecVbaseCtorIter,        // ?_J # vector vbase constructor iterator
  VdispMap,                // ?_K # virtual displacement map
  EHVecCtorIter,           // ?_L # eh vector constructor iterator
  EHVecDtorIter,           // ?_M # eh vector destructor iterator
  EHVecVbaseCtorIter,      // ?_N # eh vector vbase constructor iterator
  CopyCtorClosure,         // ?_O # copy constructor closure
  UdtReturning,            // ?_P<name> # udt returning <name>
  Unknown,                 // ?_Q # <unknown>
  RttiTypeDescriptor,      // ?_R0 # RTTI Type Descriptor
  RttiBaseClassDescriptor, // ?_R1 # RTTI Base Class Descriptor at (a,b,c,d)
  RttiBaseClassArray,      // ?_R2 # RTTI Base Class Array
  RttiClassHierarchyDescriptor, // ?_R3 # RTTI Class Hierarchy Descriptor
  RttiCompleteObjLocator,       // ?_R4 # RTTI Complete Object Locator
  LocalVftable,                 // ?_S # local vftable
  LocalVftableCtorClosure,      // ?_T # local vftable constructor closure
  ArrayNew,                     // ?_U operator new[]
  ArrayDelete,                  // ?_V operator delete[]
  LiteralOperator,              // ?__K operator ""_name
  CoAwait,                      // ?__L co_await
  Spaceship,                    // operator<=>
};

// A map to translate from operator prefix to operator type.
struct OperatorMapEntry {
  StringView Prefix;
  StringView Name;
  OperatorTy Operator;
};

// The entries here must be in the same order as the enumeration so that it can
// be indexed by enum value.
OperatorMapEntry OperatorMap[] = {
    {"0", " <ctor>", OperatorTy::Ctor},
    {"1", " <dtor>", OperatorTy::Dtor},
    {"2", "operator new", OperatorTy::New},
    {"3", "operator delete", OperatorTy::Delete},
    {"4", "operator=", OperatorTy::Assign},
    {"5", "operator>>", OperatorTy::RightShift},
    {"6", "operator<<", OperatorTy::LeftShift},
    {"7", "operator!", OperatorTy::LogicalNot},
    {"8", "operator==", OperatorTy::Equals},
    {"9", "operator!=", OperatorTy::NotEquals},
    {"A", "operator[]", OperatorTy::ArraySubscript},
    {"B", "operator <conversion>", OperatorTy::Conversion},
    {"C", "operator->", OperatorTy::Pointer},
    {"D", "operator*", OperatorTy::Dereference},
    {"E", "operator++", OperatorTy::Increment},
    {"F", "operator--", OperatorTy::Decrement},
    {"G", "operator-", OperatorTy::Minus},
    {"H", "operator+", OperatorTy::Plus},
    {"I", "operator&", OperatorTy::BitwiseAnd},
    {"J", "operator->*", OperatorTy::MemberPointer},
    {"K", "operator/", OperatorTy::Divide},
    {"L", "operator%", OperatorTy::Modulus},
    {"M", "operator<", OperatorTy::LessThan},
    {"N", "operator<=", OperatorTy::LessThanEqual},
    {"O", "operator>", OperatorTy::GreaterThan},
    {"P", "operator>=", OperatorTy::GreaterThanEqual},
    {"Q", "operator,", OperatorTy::Comma},
    {"R", "operator()", OperatorTy::Parens},
    {"S", "operator~", OperatorTy::BitwiseNot},
    {"T", "operator^", OperatorTy::BitwiseXor},
    {"U", "operator|", OperatorTy::BitwiseOr},
    {"V", "operator&&", OperatorTy::LogicalAnd},
    {"W", "operator||", OperatorTy::LogicalOr},
    {"X", "operator*=", OperatorTy::TimesEqual},
    {"Y", "operator+=", OperatorTy::PlusEqual},
    {"Z", "operator-=", OperatorTy::MinusEqual},
    {"_0", "operator/=", OperatorTy::DivEqual},
    {"_1", "operator%=", OperatorTy::ModEqual},
    {"_2", "operator>>=", OperatorTy::RshEqual},
    {"_3", "operator<<=", OperatorTy::LshEqual},
    {"_4", "operator&=", OperatorTy::BitwiseAndEqual},
    {"_5", "operator|=", OperatorTy::BitwiseOrEqual},
    {"_6", "operator^=", OperatorTy::BitwiseXorEqual},
    {"_7", "`vftable'", OperatorTy::Vftable},
    {"_8", "`vbtable'", OperatorTy::Vbtable},
    {"_9", "`vcall'", OperatorTy::Vcall},
    {"_A", "`typeof'", OperatorTy::Typeof},
    {"_B", "`local static guard'", OperatorTy::LocalStaticGuard},
    {"_C", "`string'", OperatorTy::StringLiteral},
    {"_D", "`vbase dtor'", OperatorTy::VbaseDtor},
    {"_E", "`vector deleting dtor'", OperatorTy::VecDelDtor},
    {"_F", "`default ctor closure'", OperatorTy::DefaultCtorClosure},
    {"_G", "`scalar deleting dtor'", OperatorTy::ScalarDelDtor},
    {"_H", "`vector ctor iterator'", OperatorTy::VecCtorIter},
    {"_I", "`vector dtor iterator'", OperatorTy::VecDtorIter},
    {"_J", "`vector vbase ctor iterator'", OperatorTy::VecVbaseCtorIter},
    {"_K", "`virtual displacement map'", OperatorTy::VdispMap},
    {"_L", "`eh vector ctor iterator'", OperatorTy::EHVecCtorIter},
    {"_M", "`eh vector dtor iterator'", OperatorTy::EHVecDtorIter},
    {"_N", "`eh vector vbase ctor iterator'", OperatorTy::EHVecVbaseCtorIter},
    {"_O", "`copy ctor closure'", OperatorTy::CopyCtorClosure},
    {"_P", "`udt returning'", OperatorTy::UdtReturning},
    {"_Q", "`unknown'", OperatorTy::Unknown},
    {"_R0", "`RTTI Type Descriptor'", OperatorTy::RttiTypeDescriptor},
    {"_R1", "RTTI Base Class Descriptor", OperatorTy::RttiBaseClassDescriptor},
    {"_R2", "`RTTI Base Class Array'", OperatorTy::RttiBaseClassArray},
    {"_R3", "`RTTI Class Hierarchy Descriptor'",
     OperatorTy::RttiClassHierarchyDescriptor},
    {"_R4", "`RTTI Complete Object Locator'",
     OperatorTy::RttiCompleteObjLocator},
    {"_S", "`local vftable'", OperatorTy::LocalVftable},
    {"_T", "`local vftable ctor closure'", OperatorTy::LocalVftableCtorClosure},
    {"_U", "operator new[]", OperatorTy::ArrayNew},
    {"_V", "operator delete[]", OperatorTy::ArrayDelete},
    {"__K", "operator \"\"", OperatorTy::LiteralOperator},
    {"__L", "co_await", OperatorTy::CoAwait},
};

// Function classes
enum FuncClass : uint16_t {
  None = 0,
  Public = 1 << 0,
  Protected = 1 << 1,
  Private = 1 << 2,
  Global = 1 << 3,
  Static = 1 << 4,
  Virtual = 1 << 5,
  Far = 1 << 6,
  ExternC = 1 << 7,
  NoPrototype = 1 << 8,
  VirtualThisAdjust = 1 << 9,
  VirtualThisAdjustEx = 1 << 10,
  StaticThisAdjust = 1 << 11
};

enum NameBackrefBehavior : uint8_t {
  NBB_None = 0,          // don't save any names as backrefs.
  NBB_Template = 1 << 0, // save template instanations.
  NBB_Simple = 1 << 1,   // save simple names.
};

enum class SymbolCategory {
  Unknown,
  NamedFunction,
  NamedVariable,
  UnnamedFunction,
  UnnamedVariable,
  SpecialOperator
};

namespace {

struct NameResolver {
  virtual ~NameResolver() = default;
  virtual StringView resolve(StringView S) = 0;
};

struct Type;
struct Name;

struct FunctionParams {
  bool IsVariadic = false;

  Type *Current = nullptr;

  FunctionParams *Next = nullptr;
};

struct TemplateParams {
  bool IsTemplateTemplate = false;
  bool IsAliasTemplate = false;
  bool IsIntegerLiteral = false;
  bool IntegerLiteralIsNegative = false;
  bool IsEmptyParameterPack = false;
  bool PointerToSymbol = false;
  bool ReferenceToSymbol = false;

  // If IsIntegerLiteral is true, this is a non-type template parameter
  // whose value is contained in this field.
  uint64_t IntegralValue = 0;

  // Type can be null if this is a template template parameter.  In that case
  // only Name will be valid.
  Type *ParamType = nullptr;

  // Name can be valid if this is a template template parameter (see above) or
  // this is a function declaration (e.g. foo<&SomeFunc>).  In the latter case
  // Name contains the name of the function and Type contains the signature.
  Name *ParamName = nullptr;

  TemplateParams *Next = nullptr;
};

// The type class. Mangled symbols are first parsed and converted to
// this type and then converted to string.
struct Type {
  virtual ~Type() {}

  virtual Type *clone(ArenaAllocator &Arena) const;

  // Write the "first half" of a given type.  This is a static functions to
  // give the code a chance to do processing that is common to a subset of
  // subclasses
  static void outputPre(OutputStream &OS, Type &Ty, NameResolver &Resolver);

  // Write the "second half" of a given type.  This is a static functions to
  // give the code a chance to do processing that is common to a subset of
  // subclasses
  static void outputPost(OutputStream &OS, Type &Ty, NameResolver &Resolver);

  virtual void outputPre(OutputStream &OS, NameResolver &Resolver);
  virtual void outputPost(OutputStream &OS, NameResolver &Resolver);

  // Primitive type such as Int.
  PrimTy Prim = PrimTy::Unknown;

  Qualifiers Quals = Q_None;
  StorageClass Storage = StorageClass::None; // storage class
};

// Represents an identifier which may be a template.
struct Name {
  virtual ~Name() = default;

  bool IsTemplateInstantiation = false;
  bool IsOperator = false;
  bool IsBackReference = false;

  bool isStringLiteralOperatorInfo() const;

  // Name read from an MangledName string.
  StringView Str;

  // Template parameters. Only valid if IsTemplateInstantiation is true.
  TemplateParams *TParams = nullptr;

  // Nested BackReferences (e.g. "A::B::C") are represented as a linked list.
  Name *Next = nullptr;
};

struct OperatorInfo : public Name {
  explicit OperatorInfo(const OperatorMapEntry &Info) : Info(&Info) {
    this->IsOperator = true;
  }
  explicit OperatorInfo(OperatorTy OpType)
      : OperatorInfo(OperatorMap[(int)OpType]) {}

  const OperatorMapEntry *Info = nullptr;
};

struct StringLiteral : public OperatorInfo {
  StringLiteral() : OperatorInfo(OperatorTy::StringLiteral) {}

  PrimTy CharType;
  bool IsTruncated = false;
};

struct RttiBaseClassDescriptor : public OperatorInfo {
  RttiBaseClassDescriptor()
      : OperatorInfo(OperatorTy::RttiBaseClassDescriptor) {}

  uint32_t NVOffset = 0;
  int32_t VBPtrOffset = 0;
  uint32_t VBTableOffset = 0;
  uint32_t Flags = 0;
};

struct LocalStaticGuardVariable : public OperatorInfo {
  LocalStaticGuardVariable() : OperatorInfo(OperatorTy::LocalStaticGuard) {}

  uint32_t ScopeIndex = 0;
  bool IsVisible = false;
};

struct VirtualMemberPtrThunk : public OperatorInfo {
  VirtualMemberPtrThunk() : OperatorInfo(OperatorTy::Vcall) {}

  uint64_t OffsetInVTable = 0;
  CallingConv CC = CallingConv::Cdecl;
};

struct PointerType : public Type {
  Type *clone(ArenaAllocator &Arena) const override;
  void outputPre(OutputStream &OS, NameResolver &Resolver) override;
  void outputPost(OutputStream &OS, NameResolver &Resolver) override;

  PointerAffinity Affinity;

  // Represents a type X in "a pointer to X", "a reference to X",
  // "an array of X", or "a function returning X".
  Type *Pointee = nullptr;
};

struct MemberPointerType : public Type {
  Type *clone(ArenaAllocator &Arena) const override;
  void outputPre(OutputStream &OS, NameResolver &Resolver) override;
  void outputPost(OutputStream &OS, NameResolver &Resolver) override;

  Name *MemberName = nullptr;

  // Represents a type X in "a pointer to X", "a reference to X",
  // "an array of X", or "a function returning X".
  Type *Pointee = nullptr;
};

struct FunctionType : public Type {
  struct ThisAdjustor {
    uint32_t StaticOffset = 0;
    int32_t VBPtrOffset = 0;
    int32_t VBOffsetOffset = 0;
    int32_t VtordispOffset = 0;
  };

  Type *clone(ArenaAllocator &Arena) const override;
  void outputPre(OutputStream &OS, NameResolver &Resolver) override;
  void outputPost(OutputStream &OS, NameResolver &Resolver) override;

  // True if this FunctionType instance is the Pointee of a PointerType or
  // MemberPointerType.
  bool IsFunctionPointer = false;
  bool IsThunk = false;

  Type *ReturnType = nullptr;
  // If this is a reference, the type of reference.
  ReferenceKind RefKind;

  CallingConv CallConvention;
  FuncClass FunctionClass;

  // Valid if IsThunk is true.
  ThisAdjustor *ThisAdjust = nullptr;

  FunctionParams Params;
};

struct UdtType : public Type {
  Type *clone(ArenaAllocator &Arena) const override;
  void outputPre(OutputStream &OS, NameResolver &Resolver) override;

  Name *UdtName = nullptr;
};

struct ArrayDimension {
  uint64_t Dim = 0;
  ArrayDimension *Next = nullptr;
};

struct ArrayType : public Type {
  Type *clone(ArenaAllocator &Arena) const override;
  void outputPre(OutputStream &OS, NameResolver &Resolver) override;
  void outputPost(OutputStream &OS, NameResolver &Resolver) override;

  // Either NextDimension or ElementType will be valid.
  ArrayDimension *Dims = nullptr;

  Type *ElementType = nullptr;
};

} // namespace

static bool isMemberPointer(StringView MangledName) {
  switch (MangledName.popFront()) {
  case '$':
    // This is probably an rvalue reference (e.g. $$Q), and you cannot have an
    // rvalue reference to a member.
    return false;
  case 'A':
    // 'A' indicates a reference, and you cannot have a reference to a member
    // function or member.
    return false;
  case 'P':
  case 'Q':
  case 'R':
  case 'S':
    // These 4 values indicate some kind of pointer, but we still don't know
    // what.
    break;
  default:
    assert(false && "Ty is not a pointer type!");
  }

  // If it starts with a number, then 6 indicates a non-member function
  // pointer, and 8 indicates a member function pointer.
  if (startsWithDigit(MangledName)) {
    assert(MangledName[0] == '6' || MangledName[0] == '8');
    return (MangledName[0] == '8');
  }

  // Remove ext qualifiers since those can appear on either type and are
  // therefore not indicative.
  MangledName.consumeFront('E'); // 64-bit
  MangledName.consumeFront('I'); // restrict
  MangledName.consumeFront('F'); // unaligned

  assert(!MangledName.empty());

  // The next value should be either ABCD (non-member) or QRST (member).
  switch (MangledName.front()) {
  case 'A':
  case 'B':
  case 'C':
  case 'D':
    return false;
  case 'Q':
  case 'R':
  case 'S':
  case 'T':
    return true;
  default:
    assert(false);
  }
  return false;
}

static void outputCallingConvention(OutputStream &OS, CallingConv CC) {
  outputSpaceIfNecessary(OS);

  switch (CC) {
  case CallingConv::Cdecl:
    OS << "__cdecl";
    break;
  case CallingConv::Fastcall:
    OS << "__fastcall";
    break;
  case CallingConv::Pascal:
    OS << "__pascal";
    break;
  case CallingConv::Regcall:
    OS << "__regcall";
    break;
  case CallingConv::Stdcall:
    OS << "__stdcall";
    break;
  case CallingConv::Thiscall:
    OS << "__thiscall";
    break;
  case CallingConv::Eabi:
    OS << "__eabi";
    break;
  case CallingConv::Vectorcall:
    OS << "__vectorcall";
    break;
  case CallingConv::Clrcall:
    OS << "__clrcall";
    break;
  default:
    break;
  }
}

static bool startsWithLocalScopePattern(StringView S) {
  if (!S.consumeFront('?'))
    return false;
  if (S.size() < 2)
    return false;

  size_t End = S.find('?');
  if (End == StringView::npos)
    return false;
  StringView Candidate = S.substr(0, End);
  if (Candidate.empty())
    return false;

  // \?[0-9]\?
  // ?@? is the discriminator 0.
  if (Candidate.size() == 1)
    return Candidate[0] == '@' || (Candidate[0] >= '0' && Candidate[0] <= '9');

  // If it's not 0-9, then it's an encoded number terminated with an @
  if (Candidate.back() != '@')
    return false;
  Candidate = Candidate.dropBack();

  // An encoded number starts with B-P and all subsequent digits are in A-P.
  // Note that the reason the first digit cannot be A is two fold.  First, it
  // would create an ambiguity with ?A which delimits the beginning of an
  // anonymous namespace.  Second, A represents 0, and you don't start a multi
  // digit number with a leading 0.  Presumably the anonymous namespace
  // ambiguity is also why single digit encoded numbers use 0-9 rather than A-J.
  if (Candidate[0] < 'B' || Candidate[0] > 'P')
    return false;
  Candidate = Candidate.dropFront();
  while (!Candidate.empty()) {
    if (Candidate[0] < 'A' || Candidate[0] > 'P')
      return false;
    Candidate = Candidate.dropFront();
  }

  return true;
}

// Write a function or template parameter list.
static void outputParameterList(OutputStream &OS, const FunctionParams &Params,
                                NameResolver &Resolver) {
  if (!Params.Current) {
    OS << "void";
    return;
  }

  const FunctionParams *Head = &Params;
  while (Head) {
    Type::outputPre(OS, *Head->Current, Resolver);
    Type::outputPost(OS, *Head->Current, Resolver);

    Head = Head->Next;

    if (Head)
      OS << ", ";
  }
}

static void outputStringLiteral(OutputStream &OS, const StringLiteral &Str) {
  switch (Str.CharType) {
  case PrimTy::Wchar:
    OS << "const wchar_t * {L\"";
    break;
  case PrimTy::Char:
    OS << "const char * {\"";
    break;
  case PrimTy::Char16:
    OS << "const char16_t * {u\"";
    break;
  case PrimTy::Char32:
    OS << "const char32_t * {U\"";
    break;
  default:
    LLVM_BUILTIN_UNREACHABLE;
  }
  OS << Str.Str << "\"";
  if (Str.IsTruncated)
    OS << "...";
  OS << "}";
}

static void outputName(OutputStream &OS, const Name *TheName, const Type *Ty,
                       NameResolver &Resolver);

static void outputParameterList(OutputStream &OS, const TemplateParams &Params,
                                NameResolver &Resolver) {
  if (Params.IsEmptyParameterPack) {
    OS << "<>";
    return;
  }

  OS << "<";
  const TemplateParams *Head = &Params;
  while (Head) {
    // Type can be null if this is a template template parameter,
    // and Name can be null if this is a simple type.

    if (Head->IsIntegerLiteral) {
      if (Head->IntegerLiteralIsNegative)
        OS << '-';
      OS << Head->IntegralValue;
    } else if (Head->PointerToSymbol || Head->ReferenceToSymbol) {
      if (Head->PointerToSymbol)
        OS << "&";
      Type::outputPre(OS, *Head->ParamType, Resolver);
      outputName(OS, Head->ParamName, Head->ParamType, Resolver);
      Type::outputPost(OS, *Head->ParamType, Resolver);
    } else if (Head->ParamType) {
      // simple type.
      Type::outputPre(OS, *Head->ParamType, Resolver);
      Type::outputPost(OS, *Head->ParamType, Resolver);
    } else {
      // Template alias.
      outputName(OS, Head->ParamName, Head->ParamType, Resolver);
    }

    Head = Head->Next;

    if (Head)
      OS << ", ";
  }
  OS << ">";
}

static void outputQualifiers(OutputStream &OS, Qualifiers Q) {
  if (Q & Q_Const) {
    outputSpaceIfNecessary(OS);
    OS << "const";
  }

  if (Q & Q_Volatile) {
    outputSpaceIfNecessary(OS);
    OS << "volatile";
  }

  if (Q & Q_Restrict) {
    outputSpaceIfNecessary(OS);
    OS << "__restrict";
  }
}

static void outputNameComponent(OutputStream &OS, bool IsBackReference,
                                const TemplateParams *TParams, StringView Str,
                                NameResolver &Resolver) {
  if (IsBackReference)
    Str = Resolver.resolve(Str);
  OS << Str;

  if (TParams)
    outputParameterList(OS, *TParams, Resolver);
}

static void outputNameComponent(OutputStream &OS, const Name &N,
                                NameResolver &Resolver) {
  outputNameComponent(OS, N.IsBackReference, N.TParams, N.Str, Resolver);
}

static void outputName(OutputStream &OS, const Name *TheName, const Type *Ty,
                       NameResolver &Resolver) {
  if (!TheName)
    return;

  outputSpaceIfNecessary(OS);

  const Name *Previous = nullptr;
  // Print out namespaces or outer class BackReferences.
  for (; TheName->Next; TheName = TheName->Next) {
    Previous = TheName;
    outputNameComponent(OS, *TheName, Resolver);
    OS << "::";
  }

  // Print out a regular name.
  if (!TheName->IsOperator) {
    outputNameComponent(OS, *TheName, Resolver);
    return;
  }

  const OperatorInfo &Operator = static_cast<const OperatorInfo &>(*TheName);

  // Print out ctor or dtor.
  switch (Operator.Info->Operator) {
  case OperatorTy::Dtor:
    OS << "~";
    LLVM_FALLTHROUGH;
  case OperatorTy::Ctor:
    outputNameComponent(OS, *Previous, Resolver);
    break;
  case OperatorTy::Conversion:
    OS << "operator";
    if (TheName->IsTemplateInstantiation && TheName->TParams)
      outputParameterList(OS, *TheName->TParams, Resolver);
    OS << " ";
    if (Ty) {
      const FunctionType *FTy = static_cast<const FunctionType *>(Ty);
      Type::outputPre(OS, *FTy->ReturnType, Resolver);
      Type::outputPost(OS, *FTy->ReturnType, Resolver);
    } else {
      OS << "<conversion>";
    }
    break;
  case OperatorTy::LiteralOperator:
    OS << Operator.Info->Name;
    outputNameComponent(OS, *TheName, Resolver);
    break;
  case OperatorTy::RttiBaseClassDescriptor: {
    const RttiBaseClassDescriptor &BCD =
        static_cast<const RttiBaseClassDescriptor &>(Operator);
    OS << "`" << Operator.Info->Name << " at (";
    OS << BCD.NVOffset << ", " << BCD.VBPtrOffset << ", " << BCD.VBTableOffset
       << ", " << BCD.Flags;
    OS << ")'";
    break;
  }
  case OperatorTy::LocalStaticGuard: {
    const LocalStaticGuardVariable &LSG =
        static_cast<const LocalStaticGuardVariable &>(Operator);
    OS << Operator.Info->Name;
    if (LSG.ScopeIndex > 0)
      OS << "{" << LSG.ScopeIndex << "}";
    break;
  }
  default:
    OS << Operator.Info->Name;
    if (Operator.IsTemplateInstantiation)
      outputParameterList(OS, *Operator.TParams, Resolver);
    break;
  }
}

static void outputSpecialOperator(OutputStream &OS, const Name *OuterName,
                                  NameResolver &Resolver) {
  assert(OuterName);
  // The last component should be an operator.
  const Name *LastComponent = OuterName;
  while (LastComponent->Next)
    LastComponent = LastComponent->Next;

  assert(LastComponent->IsOperator);
  const OperatorInfo &Oper = static_cast<const OperatorInfo &>(*LastComponent);
  switch (Oper.Info->Operator) {
  case OperatorTy::StringLiteral: {
    const StringLiteral &SL = static_cast<const StringLiteral &>(Oper);
    outputStringLiteral(OS, SL);
    break;
  }
  case OperatorTy::Vcall: {
    const VirtualMemberPtrThunk &Thunk =
        static_cast<const VirtualMemberPtrThunk &>(Oper);
    OS << "[thunk]: ";
    outputCallingConvention(OS, Thunk.CC);
    OS << " ";
    // Print out namespaces or outer class BackReferences.
    const Name *N = OuterName;
    for (; N->Next; N = N->Next) {
      outputNameComponent(OS, *N, Resolver);
      OS << "::";
    }
    OS << "`vcall'{";
    OS << Thunk.OffsetInVTable << ", {flat}}";
    break;
  }
  default:
    // There are no other special operator categories.
    LLVM_BUILTIN_UNREACHABLE;
  }
}

namespace {

bool Name::isStringLiteralOperatorInfo() const {
  if (!IsOperator)
    return false;
  const OperatorInfo &O = static_cast<const OperatorInfo &>(*this);
  return O.Info->Operator == OperatorTy::StringLiteral;
}

Type *Type::clone(ArenaAllocator &Arena) const {
  return Arena.alloc<Type>(*this);
}

// Write the "first half" of a given type.
void Type::outputPre(OutputStream &OS, Type &Ty, NameResolver &Resolver) {
  // Function types require custom handling of const and static so we
  // handle them separately.  All other types use the same decoration
  // for these modifiers, so handle them here in common code.
  if (Ty.Prim == PrimTy::Function) {
    Ty.outputPre(OS, Resolver);
    return;
  }

  switch (Ty.Storage) {
  case StorageClass::PrivateStatic:
  case StorageClass::PublicStatic:
  case StorageClass::ProtectedStatic:
    OS << "static ";
  default:
    break;
  }
  Ty.outputPre(OS, Resolver);

  outputQualifiers(OS, Ty.Quals);
}

// Write the "second half" of a given type.
void Type::outputPost(OutputStream &OS, Type &Ty, NameResolver &Resolver) {
  Ty.outputPost(OS, Resolver);
}

void Type::outputPre(OutputStream &OS, NameResolver &Resolver) {
  switch (Prim) {
  case PrimTy::Void:
    OS << "void";
    break;
  case PrimTy::Bool:
    OS << "bool";
    break;
  case PrimTy::Char:
    OS << "char";
    break;
  case PrimTy::Schar:
    OS << "signed char";
    break;
  case PrimTy::Uchar:
    OS << "unsigned char";
    break;
  case PrimTy::Char16:
    OS << "char16_t";
    break;
  case PrimTy::Char32:
    OS << "char32_t";
    break;
  case PrimTy::Short:
    OS << "short";
    break;
  case PrimTy::Ushort:
    OS << "unsigned short";
    break;
  case PrimTy::Int:
    OS << "int";
    break;
  case PrimTy::Uint:
    OS << "unsigned int";
    break;
  case PrimTy::Long:
    OS << "long";
    break;
  case PrimTy::Ulong:
    OS << "unsigned long";
    break;
  case PrimTy::Int64:
    OS << "__int64";
    break;
  case PrimTy::Uint64:
    OS << "unsigned __int64";
    break;
  case PrimTy::Wchar:
    OS << "wchar_t";
    break;
  case PrimTy::Float:
    OS << "float";
    break;
  case PrimTy::Double:
    OS << "double";
    break;
  case PrimTy::Ldouble:
    OS << "long double";
    break;
  case PrimTy::Nullptr:
    OS << "std::nullptr_t";
    break;
  case PrimTy::Vbtable:
  case PrimTy::Vftable:
    break;
  default:
    assert(false && "Invalid primitive type!");
  }
}
void Type::outputPost(OutputStream &OS, NameResolver &Resolver) {}

Type *PointerType::clone(ArenaAllocator &Arena) const {
  return Arena.alloc<PointerType>(*this);
}

static void outputPointerIndicator(OutputStream &OS, PointerAffinity Affinity,
                                   const Name *MemberName, const Type *Pointee,
                                   NameResolver &Resolver) {
  // "[]" and "()" (for function parameters) take precedence over "*",
  // so "int *x(int)" means "x is a function returning int *". We need
  // parentheses to supercede the default precedence. (e.g. we want to
  // emit something like "int (*x)(int)".)
  if (Pointee->Prim == PrimTy::Function || Pointee->Prim == PrimTy::Array) {
    OS << "(";
    if (Pointee->Prim == PrimTy::Function) {
      const FunctionType *FTy = static_cast<const FunctionType *>(Pointee);
      assert(FTy->IsFunctionPointer);
      outputCallingConvention(OS, FTy->CallConvention);
      OS << " ";
    }
  }

  if (MemberName) {
    outputName(OS, MemberName, Pointee, Resolver);
    OS << "::";
  }

  if (Affinity == PointerAffinity::Pointer)
    OS << "*";
  else if (Affinity == PointerAffinity::Reference)
    OS << "&";
  else
    OS << "&&";
}

void PointerType::outputPre(OutputStream &OS, NameResolver &Resolver) {
  Type::outputPre(OS, *Pointee, Resolver);

  outputSpaceIfNecessary(OS);

  if (Quals & Q_Unaligned)
    OS << "__unaligned ";

  outputPointerIndicator(OS, Affinity, nullptr, Pointee, Resolver);

  // FIXME: We should output this, but it requires updating lots of tests.
  // if (Ty.Quals & Q_Pointer64)
  //  OS << " __ptr64";
}

void PointerType::outputPost(OutputStream &OS, NameResolver &Resolver) {
  if (Pointee->Prim == PrimTy::Function || Pointee->Prim == PrimTy::Array)
    OS << ")";

  Type::outputPost(OS, *Pointee, Resolver);
}

Type *MemberPointerType::clone(ArenaAllocator &Arena) const {
  return Arena.alloc<MemberPointerType>(*this);
}

void MemberPointerType::outputPre(OutputStream &OS, NameResolver &Resolver) {
  Type::outputPre(OS, *Pointee, Resolver);

  outputSpaceIfNecessary(OS);

  outputPointerIndicator(OS, PointerAffinity::Pointer, MemberName, Pointee,
                         Resolver);

  // FIXME: We should output this, but it requires updating lots of tests.
  // if (Ty.Quals & Q_Pointer64)
  //  OS << " __ptr64";
}

void MemberPointerType::outputPost(OutputStream &OS, NameResolver &Resolver) {
  if (Pointee->Prim == PrimTy::Function || Pointee->Prim == PrimTy::Array)
    OS << ")";

  Type::outputPost(OS, *Pointee, Resolver);
}

Type *FunctionType::clone(ArenaAllocator &Arena) const {
  return Arena.alloc<FunctionType>(*this);
}

void FunctionType::outputPre(OutputStream &OS, NameResolver &Resolver) {
  if ((FunctionClass & StaticThisAdjust) || (FunctionClass & VirtualThisAdjust))
    OS << "[thunk]: ";

  if (!(FunctionClass & Global)) {
    if (FunctionClass & Static)
      OS << "static ";
  }
  if (FunctionClass & ExternC)
    OS << "extern \"C\" ";

  if (FunctionClass & Virtual)
    OS << "virtual ";

  if (ReturnType) {
    Type::outputPre(OS, *ReturnType, Resolver);
    OS << " ";
  }

  // Function pointers print the calling convention as void (__cdecl *)(params)
  // rather than void __cdecl (*)(params).  So we need to let the PointerType
  // class handle this.
  if (!IsFunctionPointer)
    outputCallingConvention(OS, CallConvention);
}

void FunctionType::outputPost(OutputStream &OS, NameResolver &Resolver) {
  // extern "C" functions don't have a prototype.
  if (FunctionClass & NoPrototype)
    return;

  if (FunctionClass & StaticThisAdjust) {
    OS << "`adjustor{" << ThisAdjust->StaticOffset << "}'";
  } else if (FunctionClass & VirtualThisAdjust) {
    if (FunctionClass & VirtualThisAdjustEx) {
      OS << "`vtordispex{" << ThisAdjust->VBPtrOffset << ", "
         << ThisAdjust->VBOffsetOffset << ", " << ThisAdjust->VtordispOffset
         << ", " << ThisAdjust->StaticOffset << "}'";
    } else {
      OS << "`vtordisp{" << ThisAdjust->VtordispOffset << ", "
         << ThisAdjust->StaticOffset << "}'";
    }
  }

  OS << "(";
  outputParameterList(OS, Params, Resolver);
  OS << ")";
  if (Quals & Q_Const)
    OS << " const";
  if (Quals & Q_Volatile)
    OS << " volatile";
  if (Quals & Q_Restrict)
    OS << " __restrict";
  if (Quals & Q_Unaligned)
    OS << " __unaligned";

  if (RefKind == ReferenceKind::LValueRef)
    OS << " &";
  else if (RefKind == ReferenceKind::RValueRef)
    OS << " &&";

  if (ReturnType)
    Type::outputPost(OS, *ReturnType, Resolver);
  return;
}

Type *UdtType::clone(ArenaAllocator &Arena) const {
  return Arena.alloc<UdtType>(*this);
}

void UdtType::outputPre(OutputStream &OS, NameResolver &Resolver) {
  switch (Prim) {
  case PrimTy::Class:
    OS << "class ";
    break;
  case PrimTy::Struct:
    OS << "struct ";
    break;
  case PrimTy::Union:
    OS << "union ";
    break;
  case PrimTy::Enum:
    OS << "enum ";
    break;
  default:
    assert(false && "Not a udt type!");
  }

  outputName(OS, UdtName, this, Resolver);
}

Type *ArrayType::clone(ArenaAllocator &Arena) const {
  return Arena.alloc<ArrayType>(*this);
}

void ArrayType::outputPre(OutputStream &OS, NameResolver &Resolver) {
  Type::outputPre(OS, *ElementType, Resolver);
}

void ArrayType::outputPost(OutputStream &OS, NameResolver &Resolver) {
  ArrayDimension *D = Dims;
  while (D) {
    OS << "[";
    if (D->Dim > 0)
      OS << D->Dim;
    OS << "]";
    D = D->Next;
  }

  Type::outputPost(OS, *ElementType, Resolver);
}

struct Symbol {
  SymbolCategory Category;

  Qualifiers SymbolQuals = Q_None;
  Name *SymbolName = nullptr;
  Type *SymbolType = nullptr;
};

} // namespace

namespace {

struct BackrefContext {
  static constexpr size_t Max = 10;

  Type *FunctionParams[Max];
  size_t FunctionParamCount = 0;

  // The first 10 BackReferences in a mangled name can be back-referenced by
  // special name @[0-9]. This is a storage for the first 10 BackReferences.
  StringView Names[Max];
  size_t NamesCount = 0;
};

// Demangler class takes the main role in demangling symbols.
// It has a set of functions to parse mangled symbols into Type instances.
// It also has a set of functions to cnovert Type instances to strings.
class Demangler : public NameResolver {
public:
  Demangler() = default;
  virtual ~Demangler() = default;

  // You are supposed to call parse() first and then check if error is true.  If
  // it is false, call output() to write the formatted name to the given stream.
  Symbol *parse(StringView &MangledName);
  Symbol *parseOperator(StringView &MangledName);

  void output(const Symbol *S, OutputStream &OS);

  StringView resolve(StringView N) override;

  // True if an error occurred.
  bool Error = false;

  void dumpBackReferences();

private:
  std::pair<SymbolCategory, Type *>
  demangleSymbolCategoryAndType(StringView &MangledName);

  Type *demangleVariableEncoding(StringView &MangledName, StorageClass SC);
  Type *demangleFunctionEncoding(StringView &MangledName);
  uint64_t demangleThunkThisAdjust(StringView &MangledName);

  Qualifiers demanglePointerExtQualifiers(StringView &MangledName);

  // Parser functions. This is a recursive-descent parser.
  Type *demangleType(StringView &MangledName, QualifierMangleMode QMM);
  Type *demangleBasicType(StringView &MangledName);
  UdtType *demangleClassType(StringView &MangledName);
  PointerType *demanglePointerType(StringView &MangledName);
  MemberPointerType *demangleMemberPointerType(StringView &MangledName);
  FunctionType *demangleFunctionType(StringView &MangledName, bool HasThisQuals,
                                     bool IsFunctionPointer);

  ArrayType *demangleArrayType(StringView &MangledName);

  TemplateParams *demangleTemplateParameterList(StringView &MangledName);
  FunctionParams demangleFunctionParameterList(StringView &MangledName);

  std::pair<uint64_t, bool> demangleNumber(StringView &MangledName);
  uint64_t demangleUnsigned(StringView &MangledName);
  int64_t demangleSigned(StringView &MangledName);

  void memorizeString(StringView s);

  /// Allocate a copy of \p Borrowed into memory that we own.
  StringView copyString(StringView Borrowed);

  Name *demangleFullyQualifiedTypeName(StringView &MangledName);
  Name *demangleFullyQualifiedSymbolName(StringView &MangledName);

  Name *demangleUnqualifiedTypeName(StringView &MangledName, bool Memorize);
  Name *demangleUnqualifiedSymbolName(StringView &MangledName,
                                      NameBackrefBehavior NBB);

  Name *demangleNameScopeChain(StringView &MangledName, Name *UnqualifiedName);
  Name *demangleNameScopePiece(StringView &MangledName);

  Name *demangleBackRefName(StringView &MangledName);
  Name *demangleTemplateInstantiationName(StringView &MangledName,
                                          NameBackrefBehavior NBB);
  std::pair<OperatorTy, Name *> demangleOperatorName(StringView &MangledName,
                                                     bool FullyQualified);
  Name *demangleSimpleName(StringView &MangledName, bool Memorize);
  Name *demangleAnonymousNamespaceName(StringView &MangledName);
  Name *demangleLocallyScopedNamePiece(StringView &MangledName);
  StringLiteral *demangleStringLiteral(StringView &MangledName);

  StringView demangleSimpleString(StringView &MangledName, bool Memorize);

  FuncClass demangleFunctionClass(StringView &MangledName);
  CallingConv demangleCallingConvention(StringView &MangledName);
  StorageClass demangleVariableStorageClass(StringView &MangledName);
  ReferenceKind demangleReferenceKind(StringView &MangledName);
  void demangleThrowSpecification(StringView &MangledName);
  wchar_t demangleWcharLiteral(StringView &MangledName);
  uint8_t demangleCharLiteral(StringView &MangledName);

  std::pair<Qualifiers, bool> demangleQualifiers(StringView &MangledName);

  // Memory allocator.
  ArenaAllocator Arena;

  // A single type uses one global back-ref table for all function params.
  // This means back-refs can even go "into" other types.  Examples:
  //
  //  // Second int* is a back-ref to first.
  //  void foo(int *, int*);
  //
  //  // Second int* is not a back-ref to first (first is not a function param).
  //  int* foo(int*);
  //
  //  // Second int* is a back-ref to first (ALL function types share the same
  //  // back-ref map.
  //  using F = void(*)(int*);
  //  F G(int *);
  BackrefContext Backrefs;
};
} // namespace

StringView Demangler::copyString(StringView Borrowed) {
  char *Stable = Arena.allocUnalignedBuffer(Borrowed.size() + 1);
  std::strcpy(Stable, Borrowed.begin());

  return {Stable, Borrowed.size()};
}

Symbol *Demangler::parseOperator(StringView &MangledName) {
  Symbol *S = Arena.alloc<Symbol>();

  bool IsMember = false;
  OperatorTy OTy;
  std::tie(OTy, S->SymbolName) = demangleOperatorName(MangledName, true);
  switch (OTy) {
  case OperatorTy::StringLiteral:
  case OperatorTy::Vcall:
    S->Category = SymbolCategory::SpecialOperator;
    break;
  case OperatorTy::Vftable:                // Foo@@6B@
  case OperatorTy::LocalVftable:           // Foo@@6B@
  case OperatorTy::RttiCompleteObjLocator: // Foo@@6B@
  case OperatorTy::Vbtable:                // Foo@@7B@
    S->Category = SymbolCategory::UnnamedVariable;
    switch (MangledName.popFront()) {
    case '6':
    case '7':
      std::tie(S->SymbolQuals, IsMember) = demangleQualifiers(MangledName);
      if (!MangledName.consumeFront('@'))
        Error = true;
      break;
    default:
      Error = true;
      break;
    }
    break;
  case OperatorTy::RttiTypeDescriptor: // <type>@@8
    S->Category = SymbolCategory::UnnamedVariable;
    S->SymbolType = demangleType(MangledName, QualifierMangleMode::Result);
    if (Error)
      break;
    if (!MangledName.consumeFront("@8"))
      Error = true;
    if (!MangledName.empty())
      Error = true;
    break;
  case OperatorTy::LocalStaticGuard: {
    S->Category = SymbolCategory::UnnamedVariable;
    break;
  }
  default:
    if (!Error)
      std::tie(S->Category, S->SymbolType) =
          demangleSymbolCategoryAndType(MangledName);
    break;
  }

  return (Error) ? nullptr : S;
}

std::pair<SymbolCategory, Type *>
Demangler::demangleSymbolCategoryAndType(StringView &MangledName) {
  // Read a variable.
  switch (MangledName.front()) {
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
    return std::make_pair(
        SymbolCategory::NamedVariable,
        demangleVariableEncoding(MangledName,
                                 demangleVariableStorageClass(MangledName)));
  case '8':
    MangledName.consumeFront('8');
    return std::pair<SymbolCategory, Type *>(SymbolCategory::UnnamedVariable,
                                             nullptr);
  }
  return std::make_pair(SymbolCategory::NamedFunction,
                        demangleFunctionEncoding(MangledName));
}

// Parser entry point.
Symbol *Demangler::parse(StringView &MangledName) {
  // We can't demangle MD5 names, just output them as-is.
  // Also, MSVC-style mangled symbols must start with '?'.
  if (MangledName.startsWith("??@") || !MangledName.startsWith('?')) {
    Symbol *S = Arena.alloc<Symbol>();
    S->Category = SymbolCategory::Unknown;
    S->SymbolName = Arena.alloc<Name>();
    S->SymbolName->Str = MangledName;
    S->SymbolType = nullptr;
    MangledName = StringView();
    return S;
  }

  MangledName.consumeFront('?');

  // ?$ is a template instantiation, but all other names that start with ? are
  // operators / special names.
  if (MangledName.startsWith('?') && !MangledName.startsWith("?$"))
    return parseOperator(MangledName);

  Symbol *S = Arena.alloc<Symbol>();
  // What follows is a main symbol name. This may include namespaces or class
  // back references.
  S->SymbolName = demangleFullyQualifiedSymbolName(MangledName);
  if (Error)
    return nullptr;

  std::tie(S->Category, S->SymbolType) =
      demangleSymbolCategoryAndType(MangledName);

  if (Error)
    return nullptr;

  return S;
}

// <type-encoding> ::= <storage-class> <variable-type>
// <storage-class> ::= 0  # private static member
//                 ::= 1  # protected static member
//                 ::= 2  # public static member
//                 ::= 3  # global
//                 ::= 4  # static local

Type *Demangler::demangleVariableEncoding(StringView &MangledName,
                                          StorageClass SC) {
  Type *Ty = demangleType(MangledName, QualifierMangleMode::Drop);

  Ty->Storage = SC;

  // <variable-type> ::= <type> <cvr-qualifiers>
  //                 ::= <type> <pointee-cvr-qualifiers> # pointers, references
  switch (Ty->Prim) {
  case PrimTy::Ptr:
  case PrimTy::MemberPtr: {
    Qualifiers ExtraChildQuals = Q_None;
    Ty->Quals =
        Qualifiers(Ty->Quals | demanglePointerExtQualifiers(MangledName));

    bool IsMember = false;
    std::tie(ExtraChildQuals, IsMember) = demangleQualifiers(MangledName);

    if (Ty->Prim == PrimTy::MemberPtr) {
      assert(IsMember);
      Name *BackRefName = demangleFullyQualifiedTypeName(MangledName);
      (void)BackRefName;
      MemberPointerType *MPTy = static_cast<MemberPointerType *>(Ty);
      MPTy->Pointee->Quals = Qualifiers(MPTy->Pointee->Quals | ExtraChildQuals);
    } else {
      PointerType *PTy = static_cast<PointerType *>(Ty);
      PTy->Pointee->Quals = Qualifiers(PTy->Pointee->Quals | ExtraChildQuals);
    }

    break;
  }
  default:
    Ty->Quals = demangleQualifiers(MangledName).first;
    break;
  }

  return Ty;
}

// Sometimes numbers are encoded in mangled symbols. For example,
// "int (*x)[20]" is a valid C type (x is a pointer to an array of
// length 20), so we need some way to embed numbers as part of symbols.
// This function parses it.
//
// <number>               ::= [?] <non-negative integer>
//
// <non-negative integer> ::= <decimal digit> # when 1 <= Number <= 10
//                        ::= <hex digit>+ @  # when Numbrer == 0 or >= 10
//
// <hex-digit>            ::= [A-P]           # A = 0, B = 1, ...
std::pair<uint64_t, bool> Demangler::demangleNumber(StringView &MangledName) {
  bool IsNegative = MangledName.consumeFront('?');

  if (startsWithDigit(MangledName)) {
    uint64_t Ret = MangledName[0] - '0' + 1;
    MangledName = MangledName.dropFront(1);
    return {Ret, IsNegative};
  }

  uint64_t Ret = 0;
  for (size_t i = 0; i < MangledName.size(); ++i) {
    char C = MangledName[i];
    if (C == '@') {
      MangledName = MangledName.dropFront(i + 1);
      return {Ret, IsNegative};
    }
    if ('A' <= C && C <= 'P') {
      Ret = (Ret << 4) + (C - 'A');
      continue;
    }
    break;
  }

  Error = true;
  return {0ULL, false};
}

uint64_t Demangler::demangleUnsigned(StringView &MangledName) {
  bool IsNegative = false;
  uint64_t Number = 0;
  std::tie(Number, IsNegative) = demangleNumber(MangledName);
  if (IsNegative)
    Error = true;
  return Number;
}

int64_t Demangler::demangleSigned(StringView &MangledName) {
  bool IsNegative = false;
  uint64_t Number = 0;
  std::tie(Number, IsNegative) = demangleNumber(MangledName);
  if (Number > INT64_MAX)
    Error = true;
  int64_t I = static_cast<int64_t>(Number);
  return IsNegative ? -I : I;
}

// First 10 strings can be referenced by special BackReferences ?0, ?1, ..., ?9.
// Memorize it.
void Demangler::memorizeString(StringView S) {
  if (Backrefs.NamesCount >= BackrefContext::Max)
    return;
  for (size_t i = 0; i < Backrefs.NamesCount; ++i)
    if (S == Backrefs.Names[i])
      return;
  Backrefs.Names[Backrefs.NamesCount++] = S;
}

Name *Demangler::demangleBackRefName(StringView &MangledName) {
  assert(startsWithDigit(MangledName));
  Name *Node = Arena.alloc<Name>();
  Node->IsBackReference = true;
  Node->Str = {MangledName.begin(), 1};
  MangledName = MangledName.dropFront();
  return Node;
}

Name *Demangler::demangleTemplateInstantiationName(StringView &MangledName,
                                                   NameBackrefBehavior NBB) {
  assert(MangledName.startsWith("?$"));
  MangledName.consumeFront("?$");

  BackrefContext OuterContext;
  std::swap(OuterContext, Backrefs);

  Name *Node = demangleUnqualifiedSymbolName(MangledName, NBB_None);
  if (!Error)
    Node->TParams = demangleTemplateParameterList(MangledName);

  std::swap(OuterContext, Backrefs);
  if (Error)
    return nullptr;

  Node->IsTemplateInstantiation = true;

  if (NBB & NBB_Template) {
    // Render this class template name into a string buffer so that we can
    // memorize it for the purpose of back-referencing.
    OutputStream OS = OutputStream::create(nullptr, nullptr, 1024);
    outputName(OS, Node, nullptr, *this);
    OS << '\0';
    char *Name = OS.getBuffer();

    StringView Owned = copyString(Name);
    memorizeString(Owned);
    std::free(Name);
  }

  return Node;
}

std::pair<OperatorTy, Name *>
Demangler::demangleOperatorName(StringView &MangledName, bool FullyQualified) {
  assert(MangledName.startsWith('?'));
  MangledName.consumeFront('?');

  const OperatorMapEntry *Entry = nullptr;
  for (const auto &MapEntry : OperatorMap) {
    if (!MangledName.consumeFront(MapEntry.Prefix))
      continue;
    Entry = &MapEntry;
    break;
  }
  if (!Entry) {
    Error = true;
    return std::make_pair(OperatorTy::Unknown, nullptr);
  }

  Name *N = nullptr;
  switch (Entry->Operator) {
  case OperatorTy::Vftable:                // Foo@@6B@
  case OperatorTy::LocalVftable:           // Foo@@6B@
  case OperatorTy::RttiCompleteObjLocator: // Foo@@6B@
  case OperatorTy::Vbtable: {              // Foo@@7B@
    OperatorInfo *Oper = Arena.alloc<OperatorInfo>(*Entry);
    N = (FullyQualified) ? demangleNameScopeChain(MangledName, Oper) : Oper;
    break;
  }

  case OperatorTy::StringLiteral:
    N = demangleStringLiteral(MangledName);
    break;
  case OperatorTy::LiteralOperator:
    N = Arena.alloc<OperatorInfo>(*Entry);
    N->Str = demangleSimpleString(MangledName, false);
    if (!MangledName.consumeFront('@'))
      Error = true;
    break;
  case OperatorTy::RttiBaseClassDescriptor: {
    RttiBaseClassDescriptor *Temp = Arena.alloc<RttiBaseClassDescriptor>();
    Temp->NVOffset = demangleUnsigned(MangledName);
    Temp->VBPtrOffset = demangleSigned(MangledName);
    Temp->VBTableOffset = demangleUnsigned(MangledName);
    Temp->Flags = demangleUnsigned(MangledName);
    N = (FullyQualified) ? demangleNameScopeChain(MangledName, Temp) : Temp;
    break;
  }
  case OperatorTy::Vcall: {
    VirtualMemberPtrThunk *Temp = Arena.alloc<VirtualMemberPtrThunk>();
    N = demangleNameScopeChain(MangledName, Temp);
    if (Error)
      break;
    if (!MangledName.consumeFront("$B"))
      Error = true;
    Temp->OffsetInVTable = demangleUnsigned(MangledName);
    if (!MangledName.consumeFront('A'))
      Error = true;
    Temp->CC = demangleCallingConvention(MangledName);
    break;
  }
  case OperatorTy::RttiTypeDescriptor:
    // This one is just followed by a type, not a name scope.
    N = Arena.alloc<OperatorInfo>(*Entry);
    break;
  case OperatorTy::LocalStaticGuard: {
    LocalStaticGuardVariable *Temp = Arena.alloc<LocalStaticGuardVariable>();
    N = (FullyQualified) ? demangleNameScopeChain(MangledName, Temp) : Temp;
    if (MangledName.consumeFront("4IA"))
      Temp->IsVisible = false;
    else if (MangledName.consumeFront("5"))
      Temp->IsVisible = true;
    else
      Error = true;
    if (!MangledName.empty())
      Temp->ScopeIndex = demangleUnsigned(MangledName);
    break;
  }
  default:
    N = Arena.alloc<OperatorInfo>(*Entry);
    N = (FullyQualified) ? demangleNameScopeChain(MangledName, N) : N;
    break;
  }
  if (Error)
    return std::make_pair(OperatorTy::Unknown, nullptr);

  return std::make_pair(Entry->Operator, N);
}

Name *Demangler::demangleSimpleName(StringView &MangledName, bool Memorize) {
  StringView S = demangleSimpleString(MangledName, Memorize);
  if (Error)
    return nullptr;

  Name *Node = Arena.alloc<Name>();
  Node->Str = S;
  return Node;
}

static bool isRebasedHexDigit(char C) { return (C >= 'A' && C <= 'P'); }

static uint8_t rebasedHexDigitToNumber(char C) {
  assert(isRebasedHexDigit(C));
  return (C <= 'J') ? (C - 'A') : (10 + C - 'K');
}

uint8_t Demangler::demangleCharLiteral(StringView &MangledName) {
  if (!MangledName.startsWith('?'))
    return MangledName.popFront();

  MangledName = MangledName.dropFront();
  if (MangledName.empty())
    goto CharLiteralError;

  if (MangledName.consumeFront('$')) {
    // Two hex digits
    if (MangledName.size() < 2)
      goto CharLiteralError;
    StringView Nibbles = MangledName.substr(0, 2);
    if (!isRebasedHexDigit(Nibbles[0]) || !isRebasedHexDigit(Nibbles[1]))
      goto CharLiteralError;
    // Don't append the null terminator.
    uint8_t C1 = rebasedHexDigitToNumber(Nibbles[0]);
    uint8_t C2 = rebasedHexDigitToNumber(Nibbles[1]);
    MangledName = MangledName.dropFront(2);
    return (C1 << 4) | C2;
  }

  if (startsWithDigit(MangledName)) {
    const char *Lookup = ",/\\:. \n\t'-";
    char C = Lookup[MangledName[0] - '0'];
    MangledName = MangledName.dropFront();
    return C;
  }

  if (MangledName[0] >= 'a' && MangledName[0] <= 'z') {
    char Lookup[26] = {'\xE1', '\xE2', '\xE3', '\xE4', '\xE5', '\xE6', '\xE7',
                       '\xE8', '\xE9', '\xEA', '\xEB', '\xEC', '\xED', '\xEE',
                       '\xEF', '\xF0', '\xF1', '\xF2', '\xF3', '\xF4', '\xF5',
                       '\xF6', '\xF7', '\xF8', '\xF9', '\xFA'};
    char C = Lookup[MangledName[0] - 'a'];
    MangledName = MangledName.dropFront();
    return C;
  }

  if (MangledName[0] >= 'A' && MangledName[0] <= 'Z') {
    char Lookup[26] = {'\xC1', '\xC2', '\xC3', '\xC4', '\xC5', '\xC6', '\xC7',
                       '\xC8', '\xC9', '\xCA', '\xCB', '\xCC', '\xCD', '\xCE',
                       '\xCF', '\xD0', '\xD1', '\xD2', '\xD3', '\xD4', '\xD5',
                       '\xD6', '\xD7', '\xD8', '\xD9', '\xDA'};
    char C = Lookup[MangledName[0] - 'A'];
    MangledName = MangledName.dropFront();
    return C;
  }

CharLiteralError:
  Error = true;
  return '\0';
}

wchar_t Demangler::demangleWcharLiteral(StringView &MangledName) {
  uint8_t C1, C2;

  C1 = demangleCharLiteral(MangledName);
  if (Error)
    goto WCharLiteralError;
  C2 = demangleCharLiteral(MangledName);
  if (Error)
    goto WCharLiteralError;

  return ((wchar_t)C1 << 8) | (wchar_t)C2;

WCharLiteralError:
  Error = true;
  return L'\0';
}

static void writeHexDigit(char *Buffer, uint8_t Digit) {
  assert(Digit <= 15);
  *Buffer = (Digit < 10) ? ('0' + Digit) : ('A' + Digit - 10);
}

static void outputHex(OutputStream &OS, unsigned C) {
  if (C == 0) {
    OS << "\\x00";
    return;
  }
  // It's easier to do the math if we can work from right to left, but we need
  // to print the numbers from left to right.  So render this into a temporary
  // buffer first, then output the temporary buffer.  Each byte is of the form
  // \xAB, which means that each byte needs 4 characters.  Since there are at
  // most 4 bytes, we need a 4*4+1 = 17 character temporary buffer.
  char TempBuffer[17];

  ::memset(TempBuffer, 0, sizeof(TempBuffer));
  constexpr int MaxPos = 15;

  int Pos = MaxPos - 1;
  while (C != 0) {
    for (int I = 0; I < 2; ++I) {
      writeHexDigit(&TempBuffer[Pos--], C % 16);
      C /= 16;
    }
    TempBuffer[Pos--] = 'x';
    TempBuffer[Pos--] = '\\';
    assert(Pos >= 0);
  }
  OS << StringView(&TempBuffer[Pos + 1]);
}

static void outputEscapedChar(OutputStream &OS, unsigned C) {
  switch (C) {
  case '\'': // single quote
    OS << "\\\'";
    return;
  case '\"': // double quote
    OS << "\\\"";
    return;
  case '\\': // backslash
    OS << "\\\\";
    return;
  case '\a': // bell
    OS << "\\a";
    return;
  case '\b': // backspace
    OS << "\\b";
    return;
  case '\f': // form feed
    OS << "\\f";
    return;
  case '\n': // new line
    OS << "\\n";
    return;
  case '\r': // carriage return
    OS << "\\r";
    return;
  case '\t': // tab
    OS << "\\t";
    return;
  case '\v': // vertical tab
    OS << "\\v";
    return;
  default:
    break;
  }

  if (C > 0x1F && C < 0x7F) {
    // Standard ascii char.
    OS << (char)C;
    return;
  }

  outputHex(OS, C);
}

unsigned countTrailingNullBytes(const uint8_t *StringBytes, int Length) {
  const uint8_t *End = StringBytes + Length - 1;
  unsigned Count = 0;
  while (Length > 0 && *End == 0) {
    --Length;
    --End;
    ++Count;
  }
  return Count;
}

unsigned countEmbeddedNulls(const uint8_t *StringBytes, unsigned Length) {
  unsigned Result = 0;
  for (unsigned I = 0; I < Length; ++I) {
    if (*StringBytes++ == 0)
      ++Result;
  }
  return Result;
}

unsigned guessCharByteSize(const uint8_t *StringBytes, unsigned NumChars,
                           unsigned NumBytes) {
  assert(NumBytes > 0);

  // If the number of bytes is odd, this is guaranteed to be a char string.
  if (NumBytes % 2 == 1)
    return 1;

  // All strings can encode at most 32 bytes of data.  If it's less than that,
  // then we encoded the entire string.  In this case we check for a 1-byte,
  // 2-byte, or 4-byte null terminator.
  if (NumBytes < 32) {
    unsigned TrailingNulls = countTrailingNullBytes(StringBytes, NumChars);
    if (TrailingNulls >= 4)
      return 4;
    if (TrailingNulls >= 2)
      return 2;
    return 1;
  }

  // The whole string was not able to be encoded.  Try to look at embedded null
  // terminators to guess.  The heuristic is that we count all embedded null
  // terminators.  If more than 2/3 are null, it's a char32.  If more than 1/3
  // are null, it's a char16.  Otherwise it's a char8.  This obviously isn't
  // perfect and is biased towards languages that have ascii alphabets, but this
  // was always going to be best effort since the encoding is lossy.
  unsigned Nulls = countEmbeddedNulls(StringBytes, NumChars);
  if (Nulls >= 2 * NumChars / 3)
    return 4;
  if (Nulls >= NumChars / 3)
    return 2;
  return 1;
}

static unsigned decodeMultiByteChar(const uint8_t *StringBytes,
                                    unsigned CharIndex, unsigned CharBytes) {
  assert(CharBytes == 1 || CharBytes == 2 || CharBytes == 4);
  unsigned Offset = CharIndex * CharBytes;
  unsigned Result = 0;
  StringBytes = StringBytes + Offset;
  for (unsigned I = 0; I < CharBytes; ++I) {
    unsigned C = static_cast<unsigned>(StringBytes[I]);
    Result |= C << (8 * I);
  }
  return Result;
}

StringLiteral *Demangler::demangleStringLiteral(StringView &MangledName) {
  // This function uses goto, so declare all variables up front.
  OutputStream OS;
  StringView CRC;
  uint64_t StringByteSize;
  bool IsWcharT = false;
  bool IsNegative = false;
  size_t CrcEndPos = 0;
  char *ResultBuffer = nullptr;

  StringLiteral *Result = Arena.alloc<StringLiteral>();

  // Prefix indicating the beginning of a string literal
  if (!MangledName.consumeFront("@_"))
    goto StringLiteralError;
  if (MangledName.empty())
    goto StringLiteralError;

  // Char Type (regular or wchar_t)
  switch (MangledName.popFront()) {
  case '1':
    IsWcharT = true;
    LLVM_FALLTHROUGH;
  case '0':
    break;
  default:
    goto StringLiteralError;
  }

  // Encoded Length
  std::tie(StringByteSize, IsNegative) = demangleNumber(MangledName);
  if (Error || IsNegative)
    goto StringLiteralError;

  // CRC 32 (always 8 characters plus a terminator)
  CrcEndPos = MangledName.find('@');
  if (CrcEndPos == StringView::npos)
    goto StringLiteralError;
  CRC = MangledName.substr(0, CrcEndPos);
  MangledName = MangledName.dropFront(CrcEndPos + 1);
  if (MangledName.empty())
    goto StringLiteralError;

  OS = OutputStream::create(nullptr, nullptr, 1024);
  if (IsWcharT) {
    Result->CharType = PrimTy::Wchar;
    if (StringByteSize > 64)
      Result->IsTruncated = true;

    while (!MangledName.consumeFront('@')) {
      assert(StringByteSize >= 2);
      wchar_t W = demangleWcharLiteral(MangledName);
      if (StringByteSize != 2 || Result->IsTruncated)
        outputEscapedChar(OS, W);
      StringByteSize -= 2;
      if (Error)
        goto StringLiteralError;
    }
  } else {
    if (StringByteSize > 32)
      Result->IsTruncated = true;

    constexpr unsigned MaxStringByteLength = 32;
    uint8_t StringBytes[MaxStringByteLength];

    unsigned BytesDecoded = 0;
    while (!MangledName.consumeFront('@')) {
      assert(StringByteSize >= 1);
      StringBytes[BytesDecoded++] = demangleCharLiteral(MangledName);
    }

    unsigned CharBytes =
        guessCharByteSize(StringBytes, BytesDecoded, StringByteSize);
    assert(StringByteSize % CharBytes == 0);
    switch (CharBytes) {
    case 1:
      Result->CharType = PrimTy::Char;
      break;
    case 2:
      Result->CharType = PrimTy::Char16;
      break;
    case 4:
      Result->CharType = PrimTy::Char32;
      break;
    default:
      LLVM_BUILTIN_UNREACHABLE;
    }
    const unsigned NumChars = BytesDecoded / CharBytes;
    for (unsigned CharIndex = 0; CharIndex < NumChars; ++CharIndex) {
      unsigned NextChar =
          decodeMultiByteChar(StringBytes, CharIndex, CharBytes);
      if (CharIndex + 1 < NumChars || Result->IsTruncated)
        outputEscapedChar(OS, NextChar);
    }
  }

  OS << '\0';
  ResultBuffer = OS.getBuffer();
  Result->Str = copyString(ResultBuffer);
  std::free(ResultBuffer);
  return Result;

StringLiteralError:
  Error = true;
  return nullptr;
}

StringView Demangler::demangleSimpleString(StringView &MangledName,
                                           bool Memorize) {
  StringView S;
  for (size_t i = 0; i < MangledName.size(); ++i) {
    if (MangledName[i] != '@')
      continue;
    S = MangledName.substr(0, i);
    MangledName = MangledName.dropFront(i + 1);

    if (Memorize)
      memorizeString(S);
    return S;
  }

  Error = true;
  return {};
}

Name *Demangler::demangleAnonymousNamespaceName(StringView &MangledName) {
  assert(MangledName.startsWith("?A"));
  MangledName.consumeFront("?A");

  Name *Node = Arena.alloc<Name>();
  Node->Str = "`anonymous namespace'";
  if (MangledName.consumeFront('@'))
    return Node;

  Error = true;
  return nullptr;
}

Name *Demangler::demangleLocallyScopedNamePiece(StringView &MangledName) {
  assert(startsWithLocalScopePattern(MangledName));

  Name *Node = Arena.alloc<Name>();
  MangledName.consumeFront('?');
  auto Number = demangleNumber(MangledName);
  assert(!Number.second);

  // One ? to terminate the number
  MangledName.consumeFront('?');

  assert(!Error);
  Symbol *Scope = parse(MangledName);
  if (Error)
    return nullptr;

  // Render the parent symbol's name into a buffer.
  OutputStream OS = OutputStream::create(nullptr, nullptr, 1024);
  OS << '`';
  output(Scope, OS);
  OS << '\'';
  OS << "::`" << Number.first << "'";
  OS << '\0';
  char *Result = OS.getBuffer();
  Node->Str = copyString(Result);
  std::free(Result);
  return Node;
}

// Parses a type name in the form of A@B@C@@ which represents C::B::A.
Name *Demangler::demangleFullyQualifiedTypeName(StringView &MangledName) {
  Name *TypeName = demangleUnqualifiedTypeName(MangledName, true);
  if (Error)
    return nullptr;
  assert(TypeName);

  Name *QualName = demangleNameScopeChain(MangledName, TypeName);
  if (Error)
    return nullptr;
  assert(QualName);
  return QualName;
}

// Parses a symbol name in the form of A@B@C@@ which represents C::B::A.
// Symbol names have slightly different rules regarding what can appear
// so we separate out the implementations for flexibility.
Name *Demangler::demangleFullyQualifiedSymbolName(StringView &MangledName) {
  // This is the final component of a symbol name (i.e. the leftmost component
  // of a mangled name.  Since the only possible template instantiation that
  // can appear in this context is a function template, and since those are
  // not saved for the purposes of name backreferences, only backref simple
  // names.
  Name *SymbolName = demangleUnqualifiedSymbolName(MangledName, NBB_Simple);
  if (Error)
    return nullptr;

  Name *QualName = demangleNameScopeChain(MangledName, SymbolName);
  if (Error)
    return nullptr;
  assert(QualName);
  return QualName;
}

Name *Demangler::demangleUnqualifiedTypeName(StringView &MangledName,
                                             bool Memorize) {
  // An inner-most name can be a back-reference, because a fully-qualified name
  // (e.g. Scope + Inner) can contain other fully qualified names inside of
  // them (for example template parameters), and these nested parameters can
  // refer to previously mangled types.
  if (startsWithDigit(MangledName))
    return demangleBackRefName(MangledName);

  if (MangledName.startsWith("?$"))
    return demangleTemplateInstantiationName(MangledName, NBB_Template);

  return demangleSimpleName(MangledName, Memorize);
}

Name *Demangler::demangleUnqualifiedSymbolName(StringView &MangledName,
                                               NameBackrefBehavior NBB) {
  if (startsWithDigit(MangledName))
    return demangleBackRefName(MangledName);
  if (MangledName.startsWith("?$"))
    return demangleTemplateInstantiationName(MangledName, NBB);
  if (MangledName.startsWith('?'))
    return demangleOperatorName(MangledName, false).second;
  return demangleSimpleName(MangledName, (NBB & NBB_Simple) != 0);
}

Name *Demangler::demangleNameScopePiece(StringView &MangledName) {
  if (startsWithDigit(MangledName))
    return demangleBackRefName(MangledName);

  if (MangledName.startsWith("?$"))
    return demangleTemplateInstantiationName(MangledName, NBB_Template);

  if (MangledName.startsWith("?A"))
    return demangleAnonymousNamespaceName(MangledName);

  if (startsWithLocalScopePattern(MangledName))
    return demangleLocallyScopedNamePiece(MangledName);

  return demangleSimpleName(MangledName, true);
}

Name *Demangler::demangleNameScopeChain(StringView &MangledName,
                                        Name *UnqualifiedName) {
  Name *Head = UnqualifiedName;

  while (!MangledName.consumeFront("@")) {
    if (MangledName.empty()) {
      Error = true;
      return nullptr;
    }

    assert(!Error);
    Name *Elem = demangleNameScopePiece(MangledName);
    if (Error)
      return nullptr;

    Elem->Next = Head;
    Head = Elem;
  }
  return Head;
}

FuncClass Demangler::demangleFunctionClass(StringView &MangledName) {
  SwapAndRestore<StringView> RestoreOnError(MangledName, MangledName);
  RestoreOnError.shouldRestore(false);

  switch (MangledName.popFront()) {
  case '9':
    return FuncClass(ExternC | NoPrototype);
  case 'A':
    return Private;
  case 'B':
    return FuncClass(Private | Far);
  case 'C':
    return FuncClass(Private | Static);
  case 'D':
    return FuncClass(Private | Static);
  case 'E':
    return FuncClass(Private | Virtual);
  case 'F':
    return FuncClass(Private | Virtual);
  case 'I':
    return FuncClass(Protected);
  case 'J':
    return FuncClass(Protected | Far);
  case 'K':
    return FuncClass(Protected | Static);
  case 'L':
    return FuncClass(Protected | Static | Far);
  case 'M':
    return FuncClass(Protected | Virtual);
  case 'N':
    return FuncClass(Protected | Virtual | Far);
  case 'O':
    return FuncClass(Protected | Virtual | StaticThisAdjust);
  case 'P':
    return FuncClass(Protected | Virtual | StaticThisAdjust | Far);
  case 'Q':
    return FuncClass(Public);
  case 'R':
    return FuncClass(Public | Far);
  case 'S':
    return FuncClass(Public | Static);
  case 'T':
    return FuncClass(Public | Static | Far);
  case 'U':
    return FuncClass(Public | Virtual);
  case 'V':
    return FuncClass(Public | Virtual | Far);
  case 'W':
    return FuncClass(Public | Virtual | StaticThisAdjust);
  case 'X':
    return FuncClass(Public | Virtual | StaticThisAdjust | Far);
  case 'Y':
    return FuncClass(Global);
  case 'Z':
    return FuncClass(Global | Far);
  case '$': {
    FuncClass VFlag = VirtualThisAdjust;
    if (MangledName.consumeFront('R'))
      VFlag = FuncClass(VFlag | VirtualThisAdjustEx);

    switch (MangledName.popFront()) {
    case '0':
      return FuncClass(Private | Virtual | VFlag);
    case '1':
      return FuncClass(Private | Virtual | VFlag | Far);
    case '2':
      return FuncClass(Protected | Virtual | VFlag);
    case '3':
      return FuncClass(Protected | Virtual | VFlag | Far);
    case '4':
      return FuncClass(Public | Virtual | VFlag);
    case '5':
      return FuncClass(Public | Virtual | VFlag | Far);
    }
  }
  }

  Error = true;
  RestoreOnError.shouldRestore(true);
  return Public;
}

CallingConv Demangler::demangleCallingConvention(StringView &MangledName) {
  switch (MangledName.popFront()) {
  case 'A':
  case 'B':
    return CallingConv::Cdecl;
  case 'C':
  case 'D':
    return CallingConv::Pascal;
  case 'E':
  case 'F':
    return CallingConv::Thiscall;
  case 'G':
  case 'H':
    return CallingConv::Stdcall;
  case 'I':
  case 'J':
    return CallingConv::Fastcall;
  case 'M':
  case 'N':
    return CallingConv::Clrcall;
  case 'O':
  case 'P':
    return CallingConv::Eabi;
  case 'Q':
    return CallingConv::Vectorcall;
  }

  return CallingConv::None;
}

StorageClass Demangler::demangleVariableStorageClass(StringView &MangledName) {
  assert(std::isdigit(MangledName.front()));

  switch (MangledName.popFront()) {
  case '0':
    return StorageClass::PrivateStatic;
  case '1':
    return StorageClass::ProtectedStatic;
  case '2':
    return StorageClass::PublicStatic;
  case '3':
    return StorageClass::Global;
  case '4':
    return StorageClass::FunctionLocalStatic;
  }
  Error = true;
  return StorageClass::None;
}

std::pair<Qualifiers, bool>
Demangler::demangleQualifiers(StringView &MangledName) {

  switch (MangledName.popFront()) {
  // Member qualifiers
  case 'Q':
    return std::make_pair(Q_None, true);
  case 'R':
    return std::make_pair(Q_Const, true);
  case 'S':
    return std::make_pair(Q_Volatile, true);
  case 'T':
    return std::make_pair(Qualifiers(Q_Const | Q_Volatile), true);
  // Non-Member qualifiers
  case 'A':
    return std::make_pair(Q_None, false);
  case 'B':
    return std::make_pair(Q_Const, false);
  case 'C':
    return std::make_pair(Q_Volatile, false);
  case 'D':
    return std::make_pair(Qualifiers(Q_Const | Q_Volatile), false);
  }
  Error = true;
  return std::make_pair(Q_None, false);
}

static bool isTagType(StringView S) {
  switch (S.front()) {
  case 'T': // union
  case 'U': // struct
  case 'V': // class
  case 'W': // enum
    return true;
  }
  return false;
}

static bool isPointerType(StringView S) {
  if (S.startsWith("$$Q")) // foo &&
    return true;

  switch (S.front()) {
  case 'A': // foo &
  case 'P': // foo *
  case 'Q': // foo *const
  case 'R': // foo *volatile
  case 'S': // foo *const volatile
    return true;
  }
  return false;
}

static bool isArrayType(StringView S) { return S[0] == 'Y'; }

static bool isFunctionType(StringView S) {
  return S.startsWith("$$A8@@") || S.startsWith("$$A6");
}

// <variable-type> ::= <type> <cvr-qualifiers>
//                 ::= <type> <pointee-cvr-qualifiers> # pointers, references
Type *Demangler::demangleType(StringView &MangledName,
                              QualifierMangleMode QMM) {
  Qualifiers Quals = Q_None;
  bool IsMember = false;
  bool IsMemberKnown = false;
  if (QMM == QualifierMangleMode::Mangle) {
    std::tie(Quals, IsMember) = demangleQualifiers(MangledName);
    IsMemberKnown = true;
  } else if (QMM == QualifierMangleMode::Result) {
    if (MangledName.consumeFront('?')) {
      std::tie(Quals, IsMember) = demangleQualifiers(MangledName);
      IsMemberKnown = true;
    }
  }

  Type *Ty = nullptr;
  if (isTagType(MangledName))
    Ty = demangleClassType(MangledName);
  else if (isPointerType(MangledName)) {
    if (!IsMemberKnown)
      IsMember = isMemberPointer(MangledName);

    if (IsMember)
      Ty = demangleMemberPointerType(MangledName);
    else
      Ty = demanglePointerType(MangledName);
  } else if (isArrayType(MangledName))
    Ty = demangleArrayType(MangledName);
  else if (isFunctionType(MangledName)) {
    if (MangledName.consumeFront("$$A8@@"))
      Ty = demangleFunctionType(MangledName, true, false);
    else {
      assert(MangledName.startsWith("$$A6"));
      MangledName.consumeFront("$$A6");
      Ty = demangleFunctionType(MangledName, false, false);
    }
  } else {
    Ty = demangleBasicType(MangledName);
    assert(Ty && !Error);
    if (!Ty || Error)
      return Ty;
  }

  Ty->Quals = Qualifiers(Ty->Quals | Quals);
  return Ty;
}

ReferenceKind Demangler::demangleReferenceKind(StringView &MangledName) {
  if (MangledName.consumeFront('G'))
    return ReferenceKind::LValueRef;
  else if (MangledName.consumeFront('H'))
    return ReferenceKind::RValueRef;
  return ReferenceKind::None;
}

void Demangler::demangleThrowSpecification(StringView &MangledName) {
  if (MangledName.consumeFront('Z'))
    return;

  Error = true;
}

FunctionType *Demangler::demangleFunctionType(StringView &MangledName,
                                              bool HasThisQuals,
                                              bool IsFunctionPointer) {
  FunctionType *FTy = Arena.alloc<FunctionType>();
  FTy->Prim = PrimTy::Function;
  FTy->IsFunctionPointer = IsFunctionPointer;

  if (HasThisQuals) {
    FTy->Quals = demanglePointerExtQualifiers(MangledName);
    FTy->RefKind = demangleReferenceKind(MangledName);
    FTy->Quals = Qualifiers(FTy->Quals | demangleQualifiers(MangledName).first);
  }

  // Fields that appear on both member and non-member functions.
  FTy->CallConvention = demangleCallingConvention(MangledName);

  // <return-type> ::= <type>
  //               ::= @ # structors (they have no declared return type)
  bool IsStructor = MangledName.consumeFront('@');
  if (!IsStructor)
    FTy->ReturnType = demangleType(MangledName, QualifierMangleMode::Result);

  FTy->Params = demangleFunctionParameterList(MangledName);

  demangleThrowSpecification(MangledName);

  return FTy;
}

Type *Demangler::demangleFunctionEncoding(StringView &MangledName) {
  FuncClass ExtraFlags = FuncClass::None;
  if (MangledName.consumeFront("$$J0"))
    ExtraFlags = FuncClass::ExternC;

  FuncClass FC = demangleFunctionClass(MangledName);
  FC = FuncClass(ExtraFlags | FC);

  FunctionType::ThisAdjustor *Adjustor = nullptr;
  if (FC & FuncClass::StaticThisAdjust) {
    Adjustor = Arena.alloc<FunctionType::ThisAdjustor>();
    Adjustor->StaticOffset = demangleSigned(MangledName);
  } else if (FC & FuncClass::VirtualThisAdjust) {
    Adjustor = Arena.alloc<FunctionType::ThisAdjustor>();
    if (FC & FuncClass::VirtualThisAdjustEx) {
      Adjustor->VBPtrOffset = demangleSigned(MangledName);
      Adjustor->VBOffsetOffset = demangleSigned(MangledName);
    }
    Adjustor->VtordispOffset = demangleSigned(MangledName);
    Adjustor->StaticOffset = demangleSigned(MangledName);
  }

  FunctionType *FTy = nullptr;
  if (FC & NoPrototype) {
    // This is an extern "C" function whose full signature hasn't been mangled.
    // This happens when we need to mangle a local symbol inside of an extern
    // "C" function.
    FTy = Arena.alloc<FunctionType>();
  } else {
    bool HasThisQuals = !(FC & (Global | Static));
    FTy = demangleFunctionType(MangledName, HasThisQuals, false);
  }
  FTy->ThisAdjust = Adjustor;
  FTy->FunctionClass = FC;

  return FTy;
}

// Reads a primitive type.
Type *Demangler::demangleBasicType(StringView &MangledName) {
  Type *Ty = Arena.alloc<Type>();

  if (MangledName.consumeFront("$$T")) {
    Ty->Prim = PrimTy::Nullptr;
    return Ty;
  }

  switch (MangledName.popFront()) {
  case 'X':
    Ty->Prim = PrimTy::Void;
    break;
  case 'D':
    Ty->Prim = PrimTy::Char;
    break;
  case 'C':
    Ty->Prim = PrimTy::Schar;
    break;
  case 'E':
    Ty->Prim = PrimTy::Uchar;
    break;
  case 'F':
    Ty->Prim = PrimTy::Short;
    break;
  case 'G':
    Ty->Prim = PrimTy::Ushort;
    break;
  case 'H':
    Ty->Prim = PrimTy::Int;
    break;
  case 'I':
    Ty->Prim = PrimTy::Uint;
    break;
  case 'J':
    Ty->Prim = PrimTy::Long;
    break;
  case 'K':
    Ty->Prim = PrimTy::Ulong;
    break;
  case 'M':
    Ty->Prim = PrimTy::Float;
    break;
  case 'N':
    Ty->Prim = PrimTy::Double;
    break;
  case 'O':
    Ty->Prim = PrimTy::Ldouble;
    break;
  case '_': {
    if (MangledName.empty()) {
      Error = true;
      return nullptr;
    }
    switch (MangledName.popFront()) {
    case 'N':
      Ty->Prim = PrimTy::Bool;
      break;
    case 'J':
      Ty->Prim = PrimTy::Int64;
      break;
    case 'K':
      Ty->Prim = PrimTy::Uint64;
      break;
    case 'W':
      Ty->Prim = PrimTy::Wchar;
      break;
    case 'S':
      Ty->Prim = PrimTy::Char16;
      break;
    case 'U':
      Ty->Prim = PrimTy::Char32;
      break;
    default:
      Error = true;
      return nullptr;
    }
    break;
  }
  default:
    Error = true;
    return nullptr;
  }
  return Ty;
}

UdtType *Demangler::demangleClassType(StringView &MangledName) {
  UdtType *UTy = Arena.alloc<UdtType>();

  switch (MangledName.popFront()) {
  case 'T':
    UTy->Prim = PrimTy::Union;
    break;
  case 'U':
    UTy->Prim = PrimTy::Struct;
    break;
  case 'V':
    UTy->Prim = PrimTy::Class;
    break;
  case 'W':
    if (MangledName.popFront() != '4') {
      Error = true;
      return nullptr;
    }
    UTy->Prim = PrimTy::Enum;
    break;
  default:
    assert(false);
  }

  UTy->UdtName = demangleFullyQualifiedTypeName(MangledName);
  return UTy;
}

static std::pair<Qualifiers, PointerAffinity>
demanglePointerCVQualifiers(StringView &MangledName) {
  if (MangledName.consumeFront("$$Q"))
    return std::make_pair(Q_None, PointerAffinity::RValueReference);

  switch (MangledName.popFront()) {
  case 'A':
    return std::make_pair(Q_None, PointerAffinity::Reference);
  case 'P':
    return std::make_pair(Q_None, PointerAffinity::Pointer);
  case 'Q':
    return std::make_pair(Q_Const, PointerAffinity::Pointer);
  case 'R':
    return std::make_pair(Q_Volatile, PointerAffinity::Pointer);
  case 'S':
    return std::make_pair(Qualifiers(Q_Const | Q_Volatile),
                          PointerAffinity::Pointer);
  default:
    assert(false && "Ty is not a pointer type!");
  }
  return std::make_pair(Q_None, PointerAffinity::Pointer);
}

// <pointer-type> ::= E? <pointer-cvr-qualifiers> <ext-qualifiers> <type>
//                       # the E is required for 64-bit non-static pointers
PointerType *Demangler::demanglePointerType(StringView &MangledName) {
  PointerType *Pointer = Arena.alloc<PointerType>();

  std::tie(Pointer->Quals, Pointer->Affinity) =
      demanglePointerCVQualifiers(MangledName);

  Pointer->Prim = PrimTy::Ptr;
  if (MangledName.consumeFront("6")) {
    Pointer->Pointee = demangleFunctionType(MangledName, false, true);
    return Pointer;
  }

  Qualifiers ExtQuals = demanglePointerExtQualifiers(MangledName);
  Pointer->Quals = Qualifiers(Pointer->Quals | ExtQuals);

  Pointer->Pointee = demangleType(MangledName, QualifierMangleMode::Mangle);
  return Pointer;
}

MemberPointerType *
Demangler::demangleMemberPointerType(StringView &MangledName) {
  MemberPointerType *Pointer = Arena.alloc<MemberPointerType>();
  Pointer->Prim = PrimTy::MemberPtr;

  PointerAffinity Affinity;
  std::tie(Pointer->Quals, Affinity) = demanglePointerCVQualifiers(MangledName);
  assert(Affinity == PointerAffinity::Pointer);

  Qualifiers ExtQuals = demanglePointerExtQualifiers(MangledName);
  Pointer->Quals = Qualifiers(Pointer->Quals | ExtQuals);

  if (MangledName.consumeFront("8")) {
    Pointer->MemberName = demangleFullyQualifiedSymbolName(MangledName);
    Pointer->Pointee = demangleFunctionType(MangledName, true, true);
  } else {
    Qualifiers PointeeQuals = Q_None;
    bool IsMember = false;
    std::tie(PointeeQuals, IsMember) = demangleQualifiers(MangledName);
    assert(IsMember);
    Pointer->MemberName = demangleFullyQualifiedSymbolName(MangledName);

    Pointer->Pointee = demangleType(MangledName, QualifierMangleMode::Drop);
    Pointer->Pointee->Quals = PointeeQuals;
  }

  return Pointer;
}

Qualifiers Demangler::demanglePointerExtQualifiers(StringView &MangledName) {
  Qualifiers Quals = Q_None;
  if (MangledName.consumeFront('E'))
    Quals = Qualifiers(Quals | Q_Pointer64);
  if (MangledName.consumeFront('I'))
    Quals = Qualifiers(Quals | Q_Restrict);
  if (MangledName.consumeFront('F'))
    Quals = Qualifiers(Quals | Q_Unaligned);

  return Quals;
}

ArrayType *Demangler::demangleArrayType(StringView &MangledName) {
  assert(MangledName.front() == 'Y');
  MangledName.popFront();

  uint64_t Rank = 0;
  bool IsNegative = false;
  std::tie(Rank, IsNegative) = demangleNumber(MangledName);
  if (IsNegative || Rank == 0) {
    Error = true;
    return nullptr;
  }

  ArrayType *ATy = Arena.alloc<ArrayType>();
  ATy->Prim = PrimTy::Array;
  ATy->Dims = Arena.alloc<ArrayDimension>();
  ArrayDimension *Dim = ATy->Dims;
  for (uint64_t I = 0; I < Rank; ++I) {
    std::tie(Dim->Dim, IsNegative) = demangleNumber(MangledName);
    if (IsNegative) {
      Error = true;
      return nullptr;
    }
    if (I + 1 < Rank) {
      Dim->Next = Arena.alloc<ArrayDimension>();
      Dim = Dim->Next;
    }
  }

  if (MangledName.consumeFront("$$C")) {
    bool IsMember = false;
    std::tie(ATy->Quals, IsMember) = demangleQualifiers(MangledName);
    if (IsMember) {
      Error = true;
      return nullptr;
    }
  }

  ATy->ElementType = demangleType(MangledName, QualifierMangleMode::Drop);
  return ATy;
}

// Reads a function or a template parameters.
FunctionParams
Demangler::demangleFunctionParameterList(StringView &MangledName) {
  // Empty parameter list.
  if (MangledName.consumeFront('X'))
    return {};

  FunctionParams *Head;
  FunctionParams **Current = &Head;
  while (!Error && !MangledName.startsWith('@') &&
         !MangledName.startsWith('Z')) {

    if (startsWithDigit(MangledName)) {
      size_t N = MangledName[0] - '0';
      if (N >= Backrefs.FunctionParamCount) {
        Error = true;
        return {};
      }
      MangledName = MangledName.dropFront();

      *Current = Arena.alloc<FunctionParams>();
      (*Current)->Current = Backrefs.FunctionParams[N]->clone(Arena);
      Current = &(*Current)->Next;
      continue;
    }

    size_t OldSize = MangledName.size();

    *Current = Arena.alloc<FunctionParams>();
    (*Current)->Current = demangleType(MangledName, QualifierMangleMode::Drop);

    size_t CharsConsumed = OldSize - MangledName.size();
    assert(CharsConsumed != 0);

    // Single-letter types are ignored for backreferences because memorizing
    // them doesn't save anything.
    if (Backrefs.FunctionParamCount <= 9 && CharsConsumed > 1)
      Backrefs.FunctionParams[Backrefs.FunctionParamCount++] =
          (*Current)->Current;

    Current = &(*Current)->Next;
  }

  if (Error)
    return {};

  // A non-empty parameter list is terminated by either 'Z' (variadic) parameter
  // list or '@' (non variadic).  Careful not to consume "@Z", as in that case
  // the following Z could be a throw specifier.
  if (MangledName.consumeFront('@'))
    return *Head;

  if (MangledName.consumeFront('Z')) {
    Head->IsVariadic = true;
    return *Head;
  }

  Error = true;
  return {};
}

TemplateParams *
Demangler::demangleTemplateParameterList(StringView &MangledName) {
  TemplateParams *Head;
  TemplateParams **Current = &Head;
  while (!Error && !MangledName.startsWith('@')) {
    // Template parameter lists don't participate in back-referencing.
    *Current = Arena.alloc<TemplateParams>();

    // Empty parameter pack.
    if (MangledName.consumeFront("$S") || MangledName.consumeFront("$$V") ||
        MangledName.consumeFront("$$$V")) {
      (*Current)->IsEmptyParameterPack = true;
      break;
    }

    if (MangledName.consumeFront("$$Y")) {
      // Template alias
      (*Current)->IsTemplateTemplate = true;
      (*Current)->IsAliasTemplate = true;
      (*Current)->ParamName = demangleFullyQualifiedTypeName(MangledName);
    } else if (MangledName.consumeFront("$$B")) {
      // Array
      (*Current)->ParamType =
          demangleType(MangledName, QualifierMangleMode::Drop);
    } else if (MangledName.consumeFront("$$C")) {
      // Type has qualifiers.
      (*Current)->ParamType =
          demangleType(MangledName, QualifierMangleMode::Mangle);
    } else if (MangledName.startsWith("$1?")) {
      MangledName.consumeFront("$1");
      // Pointer to symbol
      Symbol *S = parse(MangledName);
      (*Current)->ParamName = S->SymbolName;
      (*Current)->ParamType = S->SymbolType;
      (*Current)->PointerToSymbol = true;
    } else if (MangledName.startsWith("$E?")) {
      MangledName.consumeFront("$E");
      // Reference to symbol
      Symbol *S = parse(MangledName);
      (*Current)->ParamName = S->SymbolName;
      (*Current)->ParamType = S->SymbolType;
      (*Current)->ReferenceToSymbol = true;
    } else if (MangledName.consumeFront("$0")) {
      // Integral non-type template parameter
      bool IsNegative = false;
      uint64_t Value = 0;
      std::tie(Value, IsNegative) = demangleNumber(MangledName);

      (*Current)->IsIntegerLiteral = true;
      (*Current)->IntegerLiteralIsNegative = IsNegative;
      (*Current)->IntegralValue = Value;
    } else {
      (*Current)->ParamType =
          demangleType(MangledName, QualifierMangleMode::Drop);
    }
    if (Error)
      return nullptr;

    Current = &(*Current)->Next;
  }

  if (Error)
    return nullptr;

  // Template parameter lists cannot be variadic, so it can only be terminated
  // by @.
  if (MangledName.consumeFront('@'))
    return Head;
  Error = true;
  return nullptr;
}

StringView Demangler::resolve(StringView N) {
  assert(N.size() == 1 && isdigit(N[0]));
  size_t Digit = N[0] - '0';
  if (Digit >= Backrefs.NamesCount)
    return N;
  return Backrefs.Names[Digit];
}

void Demangler::output(const Symbol *S, OutputStream &OS) {
  if (S->Category == SymbolCategory::Unknown) {
    outputName(OS, S->SymbolName, S->SymbolType, *this);
    return;
  }

  if (S->Category == SymbolCategory::SpecialOperator) {
    outputSpecialOperator(OS, S->SymbolName, *this);
    return;
  }

  // Converts an AST to a string.
  //
  // Converting an AST representing a C++ type to a string is tricky due
  // to the bad grammar of the C++ declaration inherited from C. You have
  // to construct a string from inside to outside. For example, if a type
  // X is a pointer to a function returning int, the order you create a
  // string becomes something like this:
  //
  //   (1) X is a pointer: *X
  //   (2) (1) is a function returning int: int (*X)()
  //
  // So you cannot construct a result just by appending strings to a result.
  //
  // To deal with this, we split the function into two. outputPre() writes
  // the "first half" of type declaration, and outputPost() writes the
  // "second half". For example, outputPre() writes a return type for a
  // function and outputPost() writes an parameter list.
  if (S->SymbolType) {
    Type::outputPre(OS, *S->SymbolType, *this);
    outputName(OS, S->SymbolName, S->SymbolType, *this);
    Type::outputPost(OS, *S->SymbolType, *this);
  } else {
    outputQualifiers(OS, S->SymbolQuals);
    outputName(OS, S->SymbolName, nullptr, *this);
  }
}

void Demangler::dumpBackReferences() {
  std::printf("%d function parameter backreferences\n",
              (int)Backrefs.FunctionParamCount);

  // Create an output stream so we can render each type.
  OutputStream OS = OutputStream::create(nullptr, 0, 1024);
  for (size_t I = 0; I < Backrefs.FunctionParamCount; ++I) {
    OS.setCurrentPosition(0);

    Type *T = Backrefs.FunctionParams[I];
    Type::outputPre(OS, *T, *this);
    Type::outputPost(OS, *T, *this);

    std::printf("  [%d] - %.*s\n", (int)I, (int)OS.getCurrentPosition(),
                OS.getBuffer());
  }
  std::free(OS.getBuffer());

  if (Backrefs.FunctionParamCount > 0)
    std::printf("\n");
  std::printf("%d name backreferences\n", (int)Backrefs.NamesCount);
  for (size_t I = 0; I < Backrefs.NamesCount; ++I) {
    std::printf("  [%d] - %.*s\n", (int)I, (int)Backrefs.Names[I].size(),
                Backrefs.Names[I].begin());
  }
  if (Backrefs.NamesCount > 0)
    std::printf("\n");
}

char *llvm::microsoftDemangle(const char *MangledName, char *Buf, size_t *N,
                              int *Status, MSDemangleFlags Flags) {
  Demangler D;
  StringView Name{MangledName};
  Symbol *S = D.parse(Name);

  if (Flags & MSDF_DumpBackrefs)
    D.dumpBackReferences();
  OutputStream OS = OutputStream::create(Buf, N, 1024);
  if (D.Error) {
    OS << MangledName;
    *Status = llvm::demangle_invalid_mangled_name;
  } else {
    D.output(S, OS);
    *Status = llvm::demangle_success;
  }

  OS << '\0';
  return OS.getBuffer();
}
