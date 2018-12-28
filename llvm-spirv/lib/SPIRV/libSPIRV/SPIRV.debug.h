#ifndef SPIRV_DEBUG_H
#define SPIRV_DEBUG_H
#include "SPIRVUtil.h"
#include "llvm/BinaryFormat/Dwarf.h"

namespace SPIRVDebug {

const unsigned int DebugInfoVersion = 0x00010000;

// clang-format off

enum Instruction {
  DebugInfoNone                 = 0,
  CompilationUnit               = 1,
  TypeBasic                     = 2,
  TypePointer                   = 3,
  TypeQualifier                 = 4,
  TypeArray                     = 5,
  TypeVector                    = 6,
  Typedef                       = 7,
  TypeFunction                  = 8,
  TypeEnum                      = 9,
  TypeComposite                 = 10,
  TypeMember                    = 11,
  Inheritance                   = 12,
  TypePtrToMember               = 13,
  TypeTemplate                  = 14,
  TypeTemplateParameter         = 15,
  TypeTemplateParameterPack     = 16,
  TypeTemplateTemplateParameter = 17,
  GlobalVariable                = 18,
  FunctionDecl                  = 19,
  Function                      = 20,
  LexicalBlock                  = 21,
  LexicalBlockDiscriminator     = 22,
  Scope                         = 23,
  NoScope                       = 24,
  InlinedAt                     = 25,
  LocalVariable                 = 26,
  InlinedVariable               = 27,
  Declare                       = 28,
  Value                         = 29,
  Operation                     = 30,
  Expression                    = 31,
  MacroDef                      = 32,
  MacroUndef                    = 33,
  ImportedEntity                = 34,
  Source                        = 35,
  InstCount                     = 36
};

enum Flag {
  FlagIsPrivate          = 1 << 0,
  FlagIsProtected        = 1 << 1,
  FlagIsPublic           = FlagIsPrivate | FlagIsProtected,
  FlagAccess             = FlagIsPublic,
  FlagIsLocal            = 1 << 2,
  FlagIsDefinition       = 1 << 3,
  FlagIsFwdDecl          = 1 << 4,
  FlagIsArtificial       = 1 << 5,
  FlagIsExplicit         = 1 << 6,
  FlagIsPrototyped       = 1 << 7,
  FlagIsObjectPointer    = 1 << 8,
  FlagIsStaticMember     = 1 << 9,
  FlagIsIndirectVariable = 1 << 10,
  FlagIsLValueReference  = 1 << 11,
  FlagIsRValueReference  = 1 << 12,
  FlagIsOptimized        = 1 << 13,
  FlagIsEnumClass        = 1 << 14,
};

enum EncodingTag {
  Unspecified  = 0,
  Address      = 1,
  Boolean      = 2,
  Float        = 3,
  Signed       = 4,
  SignedChar   = 5,
  Unsigned     = 6,
  UnsignedChar = 7
};

enum CompositeTypeTag {
  Class     = 0,
  Structure = 1,
  Union     = 2
};

enum TypeQualifierTag {
  ConstType    = 0,
  VolatileType = 1,
  RestrictType = 2,
  AtomicType   = 3
};

enum ExpressionOpCode {
  Deref      = 0,
  Plus       = 1,
  Minus      = 2,
  PlusUconst = 3,
  BitPiece   = 4,
  Swap       = 5,
  Xderef     = 6,
  StackValue = 7,
  Constu     = 8,
  Fragment   = 9
};

enum ImportedEntityTag {
  ImportedModule      = 0,
  ImportedDeclaration = 1,
};

namespace Operand {

namespace CompilationUnit {
enum {
  SPIRVDebugInfoVersionIdx = 0,
  DWARFVersionIdx          = 1,
  SourceIdx                = 2,
  LanguageIdx              = 3,
  OperandCount             = 4
};
}

namespace Source {
enum {
  FileIdx      = 0,
  TextIdx      = 1,
  OperandCount = 2
};
}

namespace TypeBasic {
enum {
  NameIdx      = 0,
  SizeIdx      = 1,
  EncodingIdx  = 2,
  OperandCount = 3
};
}

namespace TypePointer {
enum {
  BaseTypeIdx     = 0,
  StorageClassIdx = 1,
  FlagsIdx        = 2,
  OperandCount    = 3
};
}

namespace TypeQualifier {
enum {
  BaseTypeIdx  = 0,
  QualifierIdx = 1,
  OperandCount = 2
};
}

namespace TypeArray {
enum {
  BaseTypeIdx       = 0,
  ComponentCountIdx = 1,
  MinOperandCount   = 2
};
}

namespace TypeVector = TypeArray;

namespace Typedef {
enum {
  NameIdx      = 0,
  BaseTypeIdx  = 1,
  SourceIdx    = 2,
  LineIdx      = 3,
  ColumnIdx    = 4,
  ParentIdx    = 5,
  OperandCount = 6
};
}

namespace TypeFunction {
enum {
  FlagsIdx          = 0,
  ReturnTypeIdx     = 1,
  FirstParameterIdx = 2,
  MinOperandCount   = 2
};
}

namespace TypeEnum {
enum {
  NameIdx            = 0,
  UnderlyingTypeIdx  = 1,
  SourceIdx          = 2,
  LineIdx            = 3,
  ColumnIdx          = 4,
  ParentIdx          = 5,
  SizeIdx            = 6,
  FlagsIdx           = 7,
  FirstEnumeratorIdx = 8,
  MinOperandCount    = 8
};
}

namespace TypeComposite {
enum {
  NameIdx         = 0,
  TagIdx          = 1,
  SourceIdx       = 2,
  LineIdx         = 3,
  ColumnIdx       = 4,
  ParentIdx       = 5,
  LinkageNameIdx  = 6,
  SizeIdx         = 7,
  FlagsIdx        = 8,
  FirstMemberIdx  = 9,
  MinOperandCount = 9
};
}

namespace TypeMember {
enum {
  NameIdx         = 0,
  TypeIdx         = 1,
  SourceIdx       = 2,
  LineIdx         = 3,
  ColumnIdx       = 4,
  ParentIdx       = 5,
  OffsetIdx       = 6,
  SizeIdx         = 7,
  FlagsIdx        = 8,
  ValueIdx        = 9,
  MinOperandCount = 9
};
}

namespace TypeInheritance {
enum {
  ChildIdx     = 0,
  ParentIdx    = 1,
  OffsetIdx    = 2,
  SizeIdx      = 3,
  FlagsIdx     = 4,
  OperandCount = 5
};
}

namespace PtrToMember {
enum {
  MemberTypeIdx = 0,
  ParentIdx     = 1,
  OperandCount  = 2
};
}

namespace Template {
enum {
  TargetIdx         = 0,
  FirstParameterIdx = 1,
  MinOperandCount   = 1
};
}

namespace TemplateParameter {
enum {
  NameIdx      = 0,
  TypeIdx      = 1,
  ValueIdx     = 2,
  SourceIdx    = 3,
  LineIdx      = 4,
  ColumnIdx    = 5,
  OperandCount = 6
};
}

namespace TemplateTemplateParameter {
enum {
  NameIdx         = 0,
  TemplateNameIdx = 1,
  SourceIdx       = 2,
  LineIdx         = 3,
  ColumnIdx       = 4,
  OperandCount    = 4
};
}

namespace TemplateParameterPack {
enum {
  NameIdx           = 0,
  SourceIdx         = 1,
  LineIdx           = 2,
  ColumnIdx         = 3,
  FirstParameterIdx = 4,
  MinOperandCount   = 4
};
}

namespace GlobalVariable {
enum {
  NameIdx                    = 0,
  TypeIdx                    = 1,
  SourceIdx                  = 2,
  LineIdx                    = 3,
  ColumnIdx                  = 4,
  ParentIdx                  = 5,
  LinkageNameIdx             = 6,
  VariableIdx                = 7,
  FlagsIdx                   = 8,
  StaticMemberDeclarationIdx = 9,
  MinOperandCount            = 9
};
}

namespace FunctionDeclaration {
enum {
  NameIdx        = 0,
  TypeIdx        = 1,
  SourceIdx      = 2,
  LineIdx        = 3,
  ColumnIdx      = 4,
  ParentIdx      = 5,
  LinkageNameIdx = 6,
  FlagsIdx       = 7,
  OperandCount   = 8
};
}

namespace Function {
enum {
  NameIdx         = 0,
  TypeIdx         = 1,
  SourceIdx       = 2,
  LineIdx         = 3,
  ColumnIdx       = 4,
  ParentIdx       = 5,
  LinkageNameIdx  = 6,
  FlagsIdx        = 7,
  ScopeLineIdx    = 8,
  FunctionIdIdx   = 9,
  DeclarationIdx  = 10,
  MinOperandCount = 10
};
}

namespace LexicalBlock {
enum {
  SourceIdx       = 0,
  LineIdx         = 1,
  ColumnIdx       = 2,
  ParentIdx       = 3,
  NameIdx         = 4,
  MinOperandCount = 4
};
}

namespace LexicalBlockDiscriminator {
enum {
  SourceIdx        = 0,
  DiscriminatorIdx = 1,
  ParentIdx        = 2,
  OperandCount     = 3
};
}

namespace Scope {
enum {
  ScopeIdx        = 0,
  InlinedAtIdx    = 1,
  MinOperandCount = 1
};
}

namespace NoScope {
// No operands
}

namespace InlinedAt {
enum {
  LineIdx         = 0,
  ScopeIdx        = 1,
  InlinedIdx      = 2,
  MinOperandCount = 2
};
}

namespace LocalVariable {
enum {
  NameIdx         = 0,
  TypeIdx         = 1,
  SourceIdx       = 2,
  LineIdx         = 3,
  ColumnIdx       = 4,
  ParentIdx       = 5,
  FlagsIdx        = 6,
  ArgNumberIdx    = 7,
  MinOperandCount = 7
};
}

namespace InlinedVariable {
enum {
  VariableIdx  = 0,
  InlinedIdx   = 1,
  OperandCount = 2
};
}

namespace DebugDeclare {
enum {
  DebugLocalVarIdx = 0,
  VariableIdx      = 1,
  ExpressionIdx    = 2,
  OperandCount     = 3
};
}

namespace DebugValue {
enum {
  DebugLocalVarIdx     = 0,
  ValueIdx             = 1,
  ExpressionIdx        = 2,
  FirstIndexOperandIdx = 3,
  MinOperandCount      = 3
};
}

namespace Operation {
enum {
  OpCodeIdx = 0
};
static std::map<ExpressionOpCode, unsigned> OpCountMap {
  { Deref,      1 },
  { Plus,       2 },
  { Minus,      2 },
  { PlusUconst, 2 },
  { BitPiece,   3 },
  { Swap,       1 },
  { Xderef,     1 },
  { StackValue, 1 },
  { Constu,     2 },
  { Fragment,   3 }
};
}

namespace ImportedEntity {
enum {
  NameIdx      = 0,
  TagIdx       = 1,
  SourceIdx    = 3,
  EntityIdx    = 4,
  LineIdx      = 5,
  ColumnIdx    = 6,
  ParentIdx    = 7,
  OperandCount = 8
};
}

} // namespace Operand
} // namespace SPIRVDebug

using namespace llvm;

namespace SPIRV {
typedef SPIRVMap<dwarf::TypeKind, SPIRVDebug::EncodingTag> DbgEncodingMap;
template <>
inline void DbgEncodingMap::init() {
  add(static_cast<dwarf::TypeKind>(0), SPIRVDebug::Unspecified);
  add(dwarf::DW_ATE_address,           SPIRVDebug::Address);
  add(dwarf::DW_ATE_boolean,           SPIRVDebug::Boolean);
  add(dwarf::DW_ATE_float,             SPIRVDebug::Float);
  add(dwarf::DW_ATE_signed,            SPIRVDebug::Signed);
  add(dwarf::DW_ATE_signed_char,       SPIRVDebug::SignedChar);
  add(dwarf::DW_ATE_unsigned,          SPIRVDebug::Unsigned);
  add(dwarf::DW_ATE_unsigned_char,     SPIRVDebug::UnsignedChar);
}

typedef SPIRVMap<dwarf::Tag, SPIRVDebug::TypeQualifierTag> DbgTypeQulifierMap;
template <>
inline void DbgTypeQulifierMap::init() {
  add(dwarf::DW_TAG_const_type,    SPIRVDebug::ConstType);
  add(dwarf::DW_TAG_volatile_type, SPIRVDebug::VolatileType);
  add(dwarf::DW_TAG_restrict_type, SPIRVDebug::RestrictType);
  add(dwarf::DW_TAG_atomic_type,   SPIRVDebug::AtomicType);
}

typedef SPIRVMap<dwarf::Tag, SPIRVDebug::CompositeTypeTag> DbgCompositeTypeMap;
template <>
inline void DbgCompositeTypeMap::init() {
  add(dwarf::DW_TAG_class_type,     SPIRVDebug::Class);
  add(dwarf::DW_TAG_structure_type, SPIRVDebug::Structure);
  add(dwarf::DW_TAG_union_type,     SPIRVDebug::Union);
}

typedef SPIRVMap<dwarf::LocationAtom, SPIRVDebug::ExpressionOpCode>
  DbgExpressionOpCodeMap;
template <>
inline void DbgExpressionOpCodeMap::init() {
  add(dwarf::DW_OP_deref,         SPIRVDebug::Deref);
  add(dwarf::DW_OP_plus,          SPIRVDebug::Plus);
  add(dwarf::DW_OP_minus,         SPIRVDebug::Minus);
  add(dwarf::DW_OP_plus_uconst,   SPIRVDebug::PlusUconst);
  add(dwarf::DW_OP_bit_piece,     SPIRVDebug::BitPiece);
  add(dwarf::DW_OP_swap,          SPIRVDebug::Swap);
  add(dwarf::DW_OP_xderef,        SPIRVDebug::Xderef);
  add(dwarf::DW_OP_stack_value,   SPIRVDebug::StackValue);
  add(dwarf::DW_OP_constu,        SPIRVDebug::Constu);
  add(dwarf::DW_OP_LLVM_fragment, SPIRVDebug::Fragment);
}

typedef SPIRVMap<dwarf::Tag, SPIRVDebug::ImportedEntityTag>
  DbgImportedEntityMap;
template <>
inline void DbgImportedEntityMap::init() {
  add(dwarf::DW_TAG_imported_module,      SPIRVDebug::ImportedModule);
  add(dwarf::DW_TAG_imported_declaration, SPIRVDebug::ImportedDeclaration);
}

} // namespace SPIRV

#endif // SPIRV_DEBUG_H
