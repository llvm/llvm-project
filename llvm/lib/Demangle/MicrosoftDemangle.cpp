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
  FunctionLocalStatic
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
  Nullptr
};

// Function classes
enum FuncClass : uint16_t {
  Public = 1 << 0,
  Protected = 1 << 1,
  Private = 1 << 2,
  Global = 1 << 3,
  Static = 1 << 4,
  Virtual = 1 << 5,
  Far = 1 << 6,
  ExternC = 1 << 7,
  NoPrototype = 1 << 8,
};

enum NameBackrefBehavior : uint8_t {
  NBB_None = 0,          // don't save any names as backrefs.
  NBB_Template = 1 << 0, // save template instanations.
  NBB_Simple = 1 << 1,   // save simple names.
};

enum class SymbolCategory { Unknown, Function, Variable, StringLiteral };

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
  bool IsTemplateInstantiation = false;
  bool IsOperator = false;
  bool IsBackReference = false;
  bool IsConversionOperator = false;
  bool IsStringLiteral = false;
  bool IsLongStringLiteral = false;

  // If IsStringLiteral is true, this is the character type.
  PrimTy StringLiteralType = PrimTy::None;

  // Name read from an MangledName string.
  StringView Str;

  // Template parameters. Only valid if Flags contains NF_TemplateInstantiation.
  TemplateParams *TParams = nullptr;

  // Nested BackReferences (e.g. "A::B::C") are represented as a linked list.
  Name *Next = nullptr;
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
  Type *clone(ArenaAllocator &Arena) const override;
  void outputPre(OutputStream &OS, NameResolver &Resolver) override;
  void outputPost(OutputStream &OS, NameResolver &Resolver) override;

  // True if this FunctionType instance is the Pointee of a PointerType or
  // MemberPointerType.
  bool IsFunctionPointer = false;

  Type *ReturnType = nullptr;
  // If this is a reference, the type of reference.
  ReferenceKind RefKind;

  CallingConv CallConvention;
  FuncClass FunctionClass;

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

static void outputStringLiteral(OutputStream &OS, const Name &TheString) {
  assert(TheString.IsStringLiteral);
  switch (TheString.StringLiteralType) {
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
  OS << TheString.Str << "\"";
  if (TheString.IsLongStringLiteral)
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

static void outputNameComponent(OutputStream &OS, const Name &N,
                                NameResolver &Resolver) {
  if (N.IsConversionOperator) {
    OS << " conv";
  } else {
    StringView S = N.Str;

    if (N.IsBackReference)
      S = Resolver.resolve(N.Str);
    OS << S;
  }

  if (N.IsTemplateInstantiation && N.TParams)
    outputParameterList(OS, *N.TParams, Resolver);
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

  // Print out ctor or dtor.
  if (TheName->Str == "dtor")
    OS << "~";

  if (TheName->Str == "ctor" || TheName->Str == "dtor") {
    outputNameComponent(OS, *Previous, Resolver);
    return;
  }

  if (TheName->IsConversionOperator) {
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
  } else {
    // Print out an overloaded operator.
    OS << "operator";
    outputNameComponent(OS, *TheName, Resolver);
  }
}

namespace {

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

  if (Ty.Quals & Q_Const) {
    outputSpaceIfNecessary(OS);
    OS << "const";
  }

  if (Ty.Quals & Q_Volatile) {
    outputSpaceIfNecessary(OS);
    OS << "volatile";
  }

  if (Ty.Quals & Q_Restrict) {
    outputSpaceIfNecessary(OS);
    OS << "__restrict";
  }
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
  if (!(FunctionClass & Global)) {
    if (FunctionClass & Static)
      OS << "static ";
  }
  if (FunctionClass & ExternC) {
    OS << "extern \"C\" ";
  }

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
  void output(const Symbol *S, OutputStream &OS);

  StringView resolve(StringView N) override;

  // True if an error occurred.
  bool Error = false;

  void dumpBackReferences();

private:
  Type *demangleVariableEncoding(StringView &MangledName);
  Type *demangleFunctionEncoding(StringView &MangledName);

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
  Name *demangleOperatorName(StringView &MangledName);
  Name *demangleSimpleName(StringView &MangledName, bool Memorize);
  Name *demangleAnonymousNamespaceName(StringView &MangledName);
  Name *demangleLocallyScopedNamePiece(StringView &MangledName);
  Name *demangleStringLiteral(StringView &MangledName);

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

// Parser entry point.
Symbol *Demangler::parse(StringView &MangledName) {
  Symbol *S = Arena.alloc<Symbol>();

  // We can't demangle MD5 names, just output them as-is.
  if (MangledName.startsWith("??@")) {
    S->Category = SymbolCategory::Unknown;
    S->SymbolName = Arena.alloc<Name>();
    S->SymbolName->Str = MangledName;
    S->SymbolType = nullptr;
    MangledName = StringView();
    return S;
  }

  // MSVC-style mangled symbols must start with '?'.
  if (!MangledName.consumeFront("?")) {
    S->Category = SymbolCategory::Unknown;
    S->SymbolName = Arena.alloc<Name>();
    S->SymbolName->Str = MangledName;
    S->SymbolType = nullptr;
    return S;
  }

  if (MangledName.consumeFront("?_C@_")) {
    // This is a string literal.  Just demangle it and return.
    S->Category = SymbolCategory::StringLiteral;
    S->SymbolName = demangleStringLiteral(MangledName);
    S->SymbolType = nullptr;
    return S;
  }

  // What follows is a main symbol name. This may include
  // namespaces or class BackReferences.
  S->SymbolName = demangleFullyQualifiedSymbolName(MangledName);
  if (Error)
    return nullptr;
  // Read a variable.
  if (startsWithDigit(MangledName) && !MangledName.startsWith('9')) {
    // 9 is a special marker for an extern "C" function with
    // no prototype.
    S->Category = SymbolCategory::Variable;
    S->SymbolType = demangleVariableEncoding(MangledName);
  } else {
    S->Category = SymbolCategory::Function;
    S->SymbolType = demangleFunctionEncoding(MangledName);
  }

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

Type *Demangler::demangleVariableEncoding(StringView &MangledName) {
  StorageClass SC = demangleVariableStorageClass(MangledName);

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

Name *Demangler::demangleOperatorName(StringView &MangledName) {
  assert(MangledName.startsWith('?'));
  MangledName.consumeFront('?');

  auto NameString = [this, &MangledName]() -> StringView {
    switch (MangledName.popFront()) {
    case '0':
      return "ctor";
    case '1':
      return "dtor";
    case '2':
      return " new";
    case '3':
      return " delete";
    case '4':
      return "=";
    case '5':
      return ">>";
    case '6':
      return "<<";
    case '7':
      return "!";
    case '8':
      return "==";
    case '9':
      return "!=";
    case 'A':
      return "[]";
    case 'C':
      return "->";
    case 'D':
      return "*";
    case 'E':
      return "++";
    case 'F':
      return "--";
    case 'G':
      return "-";
    case 'H':
      return "+";
    case 'I':
      return "&";
    case 'J':
      return "->*";
    case 'K':
      return "/";
    case 'L':
      return "%";
    case 'M':
      return "<";
    case 'N':
      return "<=";
    case 'O':
      return ">";
    case 'P':
      return ">=";
    case 'Q':
      return ",";
    case 'R':
      return "()";
    case 'S':
      return "~";
    case 'T':
      return "^";
    case 'U':
      return "|";
    case 'V':
      return "&&";
    case 'W':
      return "||";
    case 'X':
      return "*=";
    case 'Y':
      return "+=";
    case 'Z':
      return "-=";
    case '_': {
      if (MangledName.empty())
        break;

      switch (MangledName.popFront()) {
      case '0':
        return "/=";
      case '1':
        return "%=";
      case '2':
        return ">>=";
      case '3':
        return "<<=";
      case '4':
        return "&=";
      case '5':
        return "|=";
      case '6':
        return "^=";
      // case '7': # vftable
      // case '8': # vbtable
      // case '9': # vcall
      // case 'A': # typeof
      // case 'B': # local static guard
      // case 'D': # vbase destructor
      // case 'E': # vector deleting destructor
      // case 'F': # default constructor closure
      // case 'G': # scalar deleting destructor
      // case 'H': # vector constructor iterator
      // case 'I': # vector destructor iterator
      // case 'J': # vector vbase constructor iterator
      // case 'K': # virtual displacement map
      // case 'L': # eh vector constructor iterator
      // case 'M': # eh vector destructor iterator
      // case 'N': # eh vector vbase constructor iterator
      // case 'O': # copy constructor closure
      // case 'P<name>': # udt returning <name>
      // case 'Q': # <unknown>
      // case 'R0': # RTTI Type Descriptor
      // case 'R1': # RTTI Base Class Descriptor at (a,b,c,d)
      // case 'R2': # RTTI Base Class Array
      // case 'R3': # RTTI Class Hierarchy Descriptor
      // case 'R4': # RTTI Complete Object Locator
      // case 'S': # local vftable
      // case 'T': # local vftable constructor closure
      case 'U':
        return " new[]";
      case 'V':
        return " delete[]";
      case '_':
        if (MangledName.consumeFront("L"))
          return " co_await";
        if (MangledName.consumeFront("K")) {
          size_t EndPos = MangledName.find('@');
          if (EndPos == StringView::npos)
            break;
          StringView OpName = demangleSimpleString(MangledName, false);
          size_t FullSize = OpName.size() + 3; // <space>""OpName
          char *Buffer = Arena.allocUnalignedBuffer(FullSize);
          Buffer[0] = ' ';
          Buffer[1] = '"';
          Buffer[2] = '"';
          std::memcpy(Buffer + 3, OpName.begin(), OpName.size());
          return {Buffer, FullSize};
        }
      }
    }
    }
    Error = true;
    return "";
  };

  Name *Node = Arena.alloc<Name>();
  if (MangledName.consumeFront('B')) {
    // Handle conversion operator specially.
    Node->IsConversionOperator = true;
  } else {
    Node->Str = NameString();
  }
  if (Error)
    return nullptr;

  Node->IsOperator = true;
  return Node;
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
  while (Length > 0 && *End == 0) {
    --Length;
    --End;
  }
  return End - StringBytes + 1;
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

Name *Demangler::demangleStringLiteral(StringView &MangledName) {
  // This function uses goto, so declare all variables up front.
  OutputStream OS;
  StringView CRC;
  uint64_t StringByteSize;
  bool IsWcharT = false;
  bool IsNegative = false;
  size_t CrcEndPos = 0;
  char *ResultBuffer = nullptr;

  Name *Result = Arena.alloc<Name>();
  Result->IsStringLiteral = true;

  // Prefix indicating the beginning of a string literal
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
    Result->StringLiteralType = PrimTy::Wchar;
    if (StringByteSize > 64)
      Result->IsLongStringLiteral = true;

    while (!MangledName.consumeFront('@')) {
      assert(StringByteSize >= 2);
      wchar_t W = demangleWcharLiteral(MangledName);
      if (StringByteSize != 2 || Result->IsLongStringLiteral)
        outputEscapedChar(OS, W);
      StringByteSize -= 2;
      if (Error)
        goto StringLiteralError;
    }
  } else {
    if (StringByteSize > 32)
      Result->IsLongStringLiteral = true;

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
      Result->StringLiteralType = PrimTy::Char;
      break;
    case 2:
      Result->StringLiteralType = PrimTy::Char16;
      break;
    case 4:
      Result->StringLiteralType = PrimTy::Char32;
      break;
    default:
      LLVM_BUILTIN_UNREACHABLE;
    }
    const unsigned NumChars = BytesDecoded / CharBytes;
    for (unsigned CharIndex = 0; CharIndex < NumChars; ++CharIndex) {
      unsigned NextChar =
          decodeMultiByteChar(StringBytes, CharIndex, CharBytes);
      if (CharIndex + 1 < NumChars || Result->IsLongStringLiteral)
        outputEscapedChar(OS, NextChar);
    }
  }

  OS << '\0';
  ResultBuffer = OS.getBuffer();
  Result->Str = copyString(ResultBuffer);
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
  assert(SymbolName);

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
    return demangleOperatorName(MangledName);
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

  FuncClass TempFlags = FuncClass(0);
  if (MangledName.consumeFront("$$J0"))
    TempFlags = ExternC;

  switch (MangledName.popFront()) {
  case '9':
    return FuncClass(TempFlags | ExternC | NoPrototype);
  case 'A':
    return Private;
  case 'B':
    return FuncClass(TempFlags | Private | Far);
  case 'C':
    return FuncClass(TempFlags | Private | Static);
  case 'D':
    return FuncClass(TempFlags | Private | Static);
  case 'E':
    return FuncClass(TempFlags | Private | Virtual);
  case 'F':
    return FuncClass(TempFlags | Private | Virtual);
  case 'I':
    return FuncClass(TempFlags | Protected);
  case 'J':
    return FuncClass(TempFlags | Protected | Far);
  case 'K':
    return FuncClass(TempFlags | Protected | Static);
  case 'L':
    return FuncClass(TempFlags | Protected | Static | Far);
  case 'M':
    return FuncClass(TempFlags | Protected | Virtual);
  case 'N':
    return FuncClass(TempFlags | Protected | Virtual | Far);
  case 'Q':
    return FuncClass(TempFlags | Public);
  case 'R':
    return FuncClass(TempFlags | Public | Far);
  case 'S':
    return FuncClass(TempFlags | Public | Static);
  case 'T':
    return FuncClass(TempFlags | Public | Static | Far);
  case 'U':
    return FuncClass(TempFlags | Public | Virtual);
  case 'V':
    return FuncClass(TempFlags | Public | Virtual | Far);
  case 'Y':
    return FuncClass(TempFlags | Global);
  case 'Z':
    return FuncClass(TempFlags | Global | Far);
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
  FuncClass FC = demangleFunctionClass(MangledName);
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
  if (S->Category == SymbolCategory::StringLiteral) {
    outputStringLiteral(OS, *S->SymbolName);
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
  Type::outputPre(OS, *S->SymbolType, *this);
  outputName(OS, S->SymbolName, S->SymbolType, *this);
  Type::outputPost(OS, *S->SymbolType, *this);
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
