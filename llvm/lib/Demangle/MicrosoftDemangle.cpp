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
enum FuncClass : uint8_t {
  Public = 1 << 0,
  Protected = 1 << 1,
  Private = 1 << 2,
  Global = 1 << 3,
  Static = 1 << 4,
  Virtual = 1 << 5,
  Far = 1 << 6,
};

namespace {

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
  static void outputPre(OutputStream &OS, Type &Ty);

  // Write the "second half" of a given type.  This is a static functions to
  // give the code a chance to do processing that is common to a subset of
  // subclasses
  static void outputPost(OutputStream &OS, Type &Ty);

  virtual void outputPre(OutputStream &OS);
  virtual void outputPost(OutputStream &OS);

  // Primitive type such as Int.
  PrimTy Prim = PrimTy::Unknown;

  Qualifiers Quals = Q_None;
  StorageClass Storage = StorageClass::None; // storage class
};

// Represents an identifier which may be a template.
struct Name {
  // Name read from an MangledName string.
  StringView Str;

  // Overloaded operators are represented as special BackReferences in mangled
  // symbols. If this is an operator name, "op" has an operator name (e.g.
  // ">>"). Otherwise, empty.
  StringView Operator;

  // Template parameters. Null if not a template.
  TemplateParams *TParams = nullptr;

  // Nested BackReferences (e.g. "A::B::C") are represented as a linked list.
  Name *Next = nullptr;
};

struct PointerType : public Type {
  Type *clone(ArenaAllocator &Arena) const override;
  void outputPre(OutputStream &OS) override;
  void outputPost(OutputStream &OS) override;

  PointerAffinity Affinity;

  // Represents a type X in "a pointer to X", "a reference to X",
  // "an array of X", or "a function returning X".
  Type *Pointee = nullptr;
};

struct MemberPointerType : public Type {
  Type *clone(ArenaAllocator &Arena) const override;
  void outputPre(OutputStream &OS) override;
  void outputPost(OutputStream &OS) override;

  Name *MemberName = nullptr;

  // Represents a type X in "a pointer to X", "a reference to X",
  // "an array of X", or "a function returning X".
  Type *Pointee = nullptr;
};

struct FunctionType : public Type {
  Type *clone(ArenaAllocator &Arena) const override;
  void outputPre(OutputStream &OS) override;
  void outputPost(OutputStream &OS) override;

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
  void outputPre(OutputStream &OS) override;

  Name *UdtName = nullptr;
};

struct ArrayType : public Type {
  Type *clone(ArenaAllocator &Arena) const override;
  void outputPre(OutputStream &OS) override;
  void outputPost(OutputStream &OS) override;

  // Either NextDimension or ElementType will be valid.
  ArrayType *NextDimension = nullptr;
  uint32_t ArrayDimension = 0;

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

static void outputName(OutputStream &OS, const Name *TheName);

// Write a function or template parameter list.
static void outputParameterList(OutputStream &OS,
                                const FunctionParams &Params) {
  if (!Params.Current) {
    OS << "void";
    return;
  }

  const FunctionParams *Head = &Params;
  while (Head) {
    Type::outputPre(OS, *Head->Current);
    Type::outputPost(OS, *Head->Current);

    Head = Head->Next;

    if (Head)
      OS << ", ";
  }
}

static void outputParameterList(OutputStream &OS,
                                const TemplateParams &Params) {
  if (!Params.ParamType && !Params.ParamName) {
    OS << "<>";
    return;
  }

  OS << "<";
  const TemplateParams *Head = &Params;
  while (Head) {
    // Type can be null if this is a template template parameter,
    // and Name can be null if this is a simple type.

    if (Head->ParamType && Head->ParamName) {
      // Function pointer.
      OS << "&";
      Type::outputPre(OS, *Head->ParamType);
      outputName(OS, Head->ParamName);
      Type::outputPost(OS, *Head->ParamType);
    } else if (Head->ParamType) {
      // simple type.
      Type::outputPre(OS, *Head->ParamType);
      Type::outputPost(OS, *Head->ParamType);
    } else {
      // Template alias.
      outputName(OS, Head->ParamName);
    }

    Head = Head->Next;

    if (Head)
      OS << ", ";
  }
  OS << ">";
}

static void outputName(OutputStream &OS, const Name *TheName) {
  if (!TheName)
    return;

  outputSpaceIfNecessary(OS);

  const Name *Previous = nullptr;
  // Print out namespaces or outer class BackReferences.
  for (; TheName->Next; TheName = TheName->Next) {
    Previous = TheName;
    OS << TheName->Str;
    if (TheName->TParams)
      outputParameterList(OS, *TheName->TParams);
    OS << "::";
  }

  // Print out a regular name.
  if (TheName->Operator.empty()) {
    OS << TheName->Str;
    if (TheName->TParams)
      outputParameterList(OS, *TheName->TParams);
    return;
  }

  // Print out ctor or dtor.
  if (TheName->Operator == "dtor")
    OS << "~";

  if (TheName->Operator == "ctor" || TheName->Operator == "dtor") {
    OS << Previous->Str;
    if (Previous->TParams)
      outputParameterList(OS, *Previous->TParams);
    return;
  }

  // Print out an overloaded operator.
  if (!TheName->Str.empty())
    OS << TheName->Str << "::";
  OS << "operator" << TheName->Operator;
}

namespace {

Type *Type::clone(ArenaAllocator &Arena) const {
  return Arena.alloc<Type>(*this);
}

// Write the "first half" of a given type.
void Type::outputPre(OutputStream &OS, Type &Ty) {
  // Function types require custom handling of const and static so we
  // handle them separately.  All other types use the same decoration
  // for these modifiers, so handle them here in common code.
  if (Ty.Prim == PrimTy::Function) {
    Ty.outputPre(OS);
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
  Ty.outputPre(OS);

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
void Type::outputPost(OutputStream &OS, Type &Ty) { Ty.outputPost(OS); }

void Type::outputPre(OutputStream &OS) {
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
void Type::outputPost(OutputStream &OS) {}

Type *PointerType::clone(ArenaAllocator &Arena) const {
  return Arena.alloc<PointerType>(*this);
}

static void outputPointerIndicator(OutputStream &OS, PointerAffinity Affinity,
                                   const Name *MemberName,
                                   const Type *Pointee) {
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
    outputName(OS, MemberName);
    OS << "::";
  }

  if (Affinity == PointerAffinity::Pointer)
    OS << "*";
  else if (Affinity == PointerAffinity::Reference)
    OS << "&";
  else
    OS << "&&";
}

void PointerType::outputPre(OutputStream &OS) {
  Type::outputPre(OS, *Pointee);

  outputSpaceIfNecessary(OS);

  if (Quals & Q_Unaligned)
    OS << "__unaligned ";

  outputPointerIndicator(OS, Affinity, nullptr, Pointee);

  // FIXME: We should output this, but it requires updating lots of tests.
  // if (Ty.Quals & Q_Pointer64)
  //  OS << " __ptr64";
}

void PointerType::outputPost(OutputStream &OS) {
  if (Pointee->Prim == PrimTy::Function || Pointee->Prim == PrimTy::Array)
    OS << ")";

  Type::outputPost(OS, *Pointee);
}

Type *MemberPointerType::clone(ArenaAllocator &Arena) const {
  return Arena.alloc<MemberPointerType>(*this);
}

void MemberPointerType::outputPre(OutputStream &OS) {
  Type::outputPre(OS, *Pointee);

  outputSpaceIfNecessary(OS);

  outputPointerIndicator(OS, PointerAffinity::Pointer, MemberName, Pointee);

  // FIXME: We should output this, but it requires updating lots of tests.
  // if (Ty.Quals & Q_Pointer64)
  //  OS << " __ptr64";
  if (Quals & Q_Restrict)
    OS << " __restrict";
}

void MemberPointerType::outputPost(OutputStream &OS) {
  if (Pointee->Prim == PrimTy::Function || Pointee->Prim == PrimTy::Array)
    OS << ")";

  Type::outputPost(OS, *Pointee);
}

Type *FunctionType::clone(ArenaAllocator &Arena) const {
  return Arena.alloc<FunctionType>(*this);
}

void FunctionType::outputPre(OutputStream &OS) {
  if (!(FunctionClass & Global)) {
    if (FunctionClass & Static)
      OS << "static ";
  }

  if (ReturnType) {
    Type::outputPre(OS, *ReturnType);
    OS << " ";
  }

  // Function pointers print the calling convention as void (__cdecl *)(params)
  // rather than void __cdecl (*)(params).  So we need to let the PointerType
  // class handle this.
  if (!IsFunctionPointer)
    outputCallingConvention(OS, CallConvention);
}

void FunctionType::outputPost(OutputStream &OS) {
  OS << "(";
  outputParameterList(OS, Params);
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
    Type::outputPost(OS, *ReturnType);
  return;
}

Type *UdtType::clone(ArenaAllocator &Arena) const {
  return Arena.alloc<UdtType>(*this);
}

void UdtType::outputPre(OutputStream &OS) {
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

  outputName(OS, UdtName);
}

Type *ArrayType::clone(ArenaAllocator &Arena) const {
  return Arena.alloc<ArrayType>(*this);
}

void ArrayType::outputPre(OutputStream &OS) {
  Type::outputPre(OS, *ElementType);
}

void ArrayType::outputPost(OutputStream &OS) {
  if (ArrayDimension > 0)
    OS << "[" << ArrayDimension << "]";
  if (NextDimension)
    Type::outputPost(OS, *NextDimension);
  else if (ElementType)
    Type::outputPost(OS, *ElementType);
}

struct Symbol {
  Name *SymbolName = nullptr;
  Type *SymbolType = nullptr;
};

} // namespace

namespace {

// Demangler class takes the main role in demangling symbols.
// It has a set of functions to parse mangled symbols into Type instances.
// It also has a set of functions to cnovert Type instances to strings.
class Demangler {
public:
  Demangler() = default;

  // You are supposed to call parse() first and then check if error is true.  If
  // it is false, call output() to write the formatted name to the given stream.
  Symbol *parse(StringView &MangledName);
  void output(const Symbol *S, OutputStream &OS);

  // True if an error occurred.
  bool Error = false;

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

  int demangleNumber(StringView &MangledName);

  void memorizeString(StringView s);

  /// Allocate a copy of \p Borrowed into memory that we own.
  StringView copyString(StringView Borrowed);

  Name *demangleFullyQualifiedTypeName(StringView &MangledName);
  Name *demangleFullyQualifiedSymbolName(StringView &MangledName);

  Name *demangleUnqualifiedTypeName(StringView &MangledName);
  Name *demangleUnqualifiedSymbolName(StringView &MangledName);

  Name *demangleNameScopeChain(StringView &MangledName, Name *UnqualifiedName);
  Name *demangleNameScopePiece(StringView &MangledName);

  Name *demangleBackRefName(StringView &MangledName);
  Name *demangleClassTemplateName(StringView &MangledName);
  Name *demangleOperatorName(StringView &MangledName);
  Name *demangleSimpleName(StringView &MangledName, bool Memorize);
  Name *demangleAnonymousNamespaceName(StringView &MangledName);
  Name *demangleLocallyScopedNamePiece(StringView &MangledName);

  StringView demangleSimpleString(StringView &MangledName, bool Memorize);

  FuncClass demangleFunctionClass(StringView &MangledName);
  CallingConv demangleCallingConvention(StringView &MangledName);
  StorageClass demangleVariableStorageClass(StringView &MangledName);
  ReferenceKind demangleReferenceKind(StringView &MangledName);
  void demangleThrowSpecification(StringView &MangledName);

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
  Type *FunctionParamBackRefs[10];
  size_t FunctionParamBackRefCount = 0;

  // The first 10 BackReferences in a mangled name can be back-referenced by
  // special name @[0-9]. This is a storage for the first 10 BackReferences.
  StringView BackReferences[10];
  size_t BackRefCount = 0;
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

  // MSVC-style mangled symbols must start with '?'.
  if (!MangledName.consumeFront("?")) {
    S->SymbolName = Arena.alloc<Name>();
    S->SymbolName->Str = MangledName;
    S->SymbolType = Arena.alloc<Type>();
    S->SymbolType->Prim = PrimTy::Unknown;
    return S;
  }

  // What follows is a main symbol name. This may include
  // namespaces or class BackReferences.
  S->SymbolName = demangleFullyQualifiedSymbolName(MangledName);

  // Read a variable.
  S->SymbolType = startsWithDigit(MangledName)
                      ? demangleVariableEncoding(MangledName)
                      : demangleFunctionEncoding(MangledName);

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
int Demangler::demangleNumber(StringView &MangledName) {
  bool neg = MangledName.consumeFront("?");

  if (startsWithDigit(MangledName)) {
    int32_t Ret = MangledName[0] - '0' + 1;
    MangledName = MangledName.dropFront(1);
    return neg ? -Ret : Ret;
  }

  int Ret = 0;
  for (size_t i = 0; i < MangledName.size(); ++i) {
    char C = MangledName[i];
    if (C == '@') {
      MangledName = MangledName.dropFront(i + 1);
      return neg ? -Ret : Ret;
    }
    if ('A' <= C && C <= 'P') {
      Ret = (Ret << 4) + (C - 'A');
      continue;
    }
    break;
  }

  Error = true;
  return 0;
}

// First 10 strings can be referenced by special BackReferences ?0, ?1, ..., ?9.
// Memorize it.
void Demangler::memorizeString(StringView S) {
  if (BackRefCount >= sizeof(BackReferences) / sizeof(*BackReferences))
    return;
  for (size_t i = 0; i < BackRefCount; ++i)
    if (S == BackReferences[i])
      return;
  BackReferences[BackRefCount++] = S;
}

Name *Demangler::demangleBackRefName(StringView &MangledName) {
  assert(startsWithDigit(MangledName));

  size_t I = MangledName[0] - '0';
  if (I >= BackRefCount) {
    Error = true;
    return nullptr;
  }

  MangledName = MangledName.dropFront();
  Name *Node = Arena.alloc<Name>();
  Node->Str = BackReferences[I];
  return Node;
}

Name *Demangler::demangleClassTemplateName(StringView &MangledName) {
  assert(MangledName.startsWith("?$"));
  MangledName.consumeFront("?$");

  Name *Node = demangleSimpleName(MangledName, false);
  Node->TParams = demangleTemplateParameterList(MangledName);

  // Render this class template name into a string buffer so that we can
  // memorize it for the purpose of back-referencing.
  OutputStream OS = OutputStream::create(nullptr, nullptr, 1024);
  outputName(OS, Node);
  OS << '\0';
  char *Name = OS.getBuffer();

  StringView Owned = copyString(Name);
  memorizeString(Owned);
  std::free(Name);

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
  Node->Operator = NameString();
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
  int ScopeIdentifier = demangleNumber(MangledName);

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
  OS << "::`" << ScopeIdentifier << "'";
  OS << '\0';
  char *Result = OS.getBuffer();
  Node->Str = copyString(Result);
  std::free(Result);
  return Node;
}

// Parses a type name in the form of A@B@C@@ which represents C::B::A.
Name *Demangler::demangleFullyQualifiedTypeName(StringView &MangledName) {
  Name *TypeName = demangleUnqualifiedTypeName(MangledName);
  assert(TypeName);

  Name *QualName = demangleNameScopeChain(MangledName, TypeName);
  assert(QualName);
  return QualName;
}

// Parses a symbol name in the form of A@B@C@@ which represents C::B::A.
// Symbol names have slightly different rules regarding what can appear
// so we separate out the implementations for flexibility.
Name *Demangler::demangleFullyQualifiedSymbolName(StringView &MangledName) {
  Name *SymbolName = demangleUnqualifiedSymbolName(MangledName);
  assert(SymbolName);

  Name *QualName = demangleNameScopeChain(MangledName, SymbolName);
  assert(QualName);
  return QualName;
}

Name *Demangler::demangleUnqualifiedTypeName(StringView &MangledName) {
  // An inner-most name can be a back-reference, because a fully-qualified name
  // (e.g. Scope + Inner) can contain other fully qualified names inside of
  // them (for example template parameters), and these nested parameters can
  // refer to previously mangled types.
  if (startsWithDigit(MangledName))
    return demangleBackRefName(MangledName);

  if (MangledName.startsWith("?$"))
    return demangleClassTemplateName(MangledName);

  return demangleSimpleName(MangledName, true);
}

Name *Demangler::demangleUnqualifiedSymbolName(StringView &MangledName) {
  if (startsWithDigit(MangledName))
    return demangleBackRefName(MangledName);
  if (MangledName.startsWith("?$"))
    return demangleClassTemplateName(MangledName);
  if (MangledName.startsWith('?'))
    return demangleOperatorName(MangledName);
  return demangleSimpleName(MangledName, true);
}

Name *Demangler::demangleNameScopePiece(StringView &MangledName) {
  if (startsWithDigit(MangledName))
    return demangleBackRefName(MangledName);

  if (MangledName.startsWith("?$"))
    return demangleClassTemplateName(MangledName);

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
    return Protected;
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
  case 'Q':
    return Public;
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
  case 'Y':
    return Global;
  case 'Z':
    return FuncClass(Global | Far);
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

  bool HasThisQuals = !(FC & (Global | Static));
  FunctionType *FTy = demangleFunctionType(MangledName, HasThisQuals, false);
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

  int Dimension = demangleNumber(MangledName);
  if (Dimension <= 0) {
    Error = true;
    return nullptr;
  }

  ArrayType *ATy = Arena.alloc<ArrayType>();
  ArrayType *Dim = ATy;
  for (int I = 0; I < Dimension; ++I) {
    Dim->Prim = PrimTy::Array;
    Dim->ArrayDimension = demangleNumber(MangledName);
    Dim->NextDimension = Arena.alloc<ArrayType>();
    Dim = Dim->NextDimension;
  }

  if (MangledName.consumeFront("$$C")) {
    if (MangledName.consumeFront("B"))
      ATy->Quals = Q_Const;
    else if (MangledName.consumeFront("C") || MangledName.consumeFront("D"))
      ATy->Quals = Qualifiers(Q_Const | Q_Volatile);
    else if (!MangledName.consumeFront("A"))
      Error = true;
  }

  ATy->ElementType = demangleType(MangledName, QualifierMangleMode::Drop);
  Dim->ElementType = ATy->ElementType;
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
      if (N >= FunctionParamBackRefCount) {
        Error = true;
        return {};
      }
      MangledName = MangledName.dropFront();

      *Current = Arena.alloc<FunctionParams>();
      (*Current)->Current = FunctionParamBackRefs[N]->clone(Arena);
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
    if (FunctionParamBackRefCount <= 9 && CharsConsumed > 1)
      FunctionParamBackRefs[FunctionParamBackRefCount++] = (*Current)->Current;

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
      if (!MangledName.startsWith('@'))
        Error = true;
      continue;
    }

    if (MangledName.consumeFront("$$Y")) {
      (*Current)->IsTemplateTemplate = true;
      (*Current)->IsAliasTemplate = true;
      (*Current)->ParamName = demangleFullyQualifiedTypeName(MangledName);
    } else if (MangledName.consumeFront("$1?")) {
      (*Current)->ParamName = demangleFullyQualifiedSymbolName(MangledName);
      (*Current)->ParamType = demangleFunctionEncoding(MangledName);
    } else {
      (*Current)->ParamType =
          demangleType(MangledName, QualifierMangleMode::Drop);
    }

    Current = &(*Current)->Next;
  }

  if (Error)
    return {};

  // Template parameter lists cannot be variadic, so it can only be terminated
  // by @.
  if (MangledName.consumeFront('@'))
    return Head;
  Error = true;
  return {};
}

void Demangler::output(const Symbol *S, OutputStream &OS) {
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
  Type::outputPre(OS, *S->SymbolType);
  outputName(OS, S->SymbolName);
  Type::outputPost(OS, *S->SymbolType);
}

char *llvm::microsoftDemangle(const char *MangledName, char *Buf, size_t *N,
                              int *Status) {
  Demangler D;
  StringView Name{MangledName};
  Symbol *S = D.parse(Name);

  if (D.Error)
    *Status = llvm::demangle_invalid_mangled_name;
  else
    *Status = llvm::demangle_success;

  OutputStream OS = OutputStream::create(Buf, N, 1024);
  D.output(S, OS);
  OS << '\0';
  return OS.getBuffer();
}
