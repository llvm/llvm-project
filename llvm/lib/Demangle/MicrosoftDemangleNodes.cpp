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
//===----------------------------------------------------------------------===//

#include "MicrosoftDemangleNodes.h"
#include "llvm/Demangle/Compiler.h"
#include "llvm/Demangle/Utility.h"
#include <cctype>

using namespace llvm;
using namespace ms_demangle;

#define OUTPUT_ENUM_CLASS_VALUE(Enum, Value, Desc)                             \
  case Enum::Value:                                                            \
    OS << Desc;                                                                \
    break;

// Writes a space if the last token does not end with a punctuation.
static void outputSpaceIfNecessary(OutputStream &OS) {
  if (OS.empty())
    return;

  char C = OS.back();
  if (std::isalnum(C) || C == '>')
    OS << " ";
}

static bool outputSingleQualifier(OutputStream &OS, Qualifiers Q) {
  switch (Q) {
  case Q_Const:
    OS << "const";
    return true;
  case Q_Volatile:
    OS << "volatile";
    return true;
  case Q_Restrict:
    OS << "__restrict";
    return true;
  default:
    break;
  }
  return false;
}

static bool outputQualifierIfPresent(OutputStream &OS, Qualifiers Q,
                                     Qualifiers Mask, bool NeedSpace) {
  if (!(Q & Mask))
    return NeedSpace;

  if (NeedSpace)
    OS << " ";

  outputSingleQualifier(OS, Mask);
  return true;
}

static void outputQualifiers(OutputStream &OS, Qualifiers Q, bool SpaceBefore,
                             bool SpaceAfter) {
  if (Q == Q_None)
    return;

  size_t Pos1 = OS.getCurrentPosition();
  SpaceBefore = outputQualifierIfPresent(OS, Q, Q_Const, SpaceBefore);
  SpaceBefore = outputQualifierIfPresent(OS, Q, Q_Volatile, SpaceBefore);
  SpaceBefore = outputQualifierIfPresent(OS, Q, Q_Restrict, SpaceBefore);
  size_t Pos2 = OS.getCurrentPosition();
  if (SpaceAfter && Pos2 > Pos1)
    OS << " ";
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

void TypeNode::outputQuals(bool SpaceBefore, bool SpaceAfter) const {}

void PrimitiveTypeNode::outputPre(OutputStream &OS) const {
  switch (PrimKind) {
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Void, "void");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Bool, "bool");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Char, "char");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Schar, "signed char");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Uchar, "unsigned char");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Char16, "char16_t");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Char32, "char32_t");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Short, "short");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Ushort, "unsigned short");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Int, "int");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Uint, "unsigned int");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Long, "long");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Ulong, "unsigned long");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Int64, "__int64");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Uint64, "unsigned __int64");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Wchar, "wchar_t");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Float, "float");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Double, "double");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Ldouble, "long double");
    OUTPUT_ENUM_CLASS_VALUE(PrimitiveKind, Nullptr, "std::nullptr_t");
  }
  outputQualifiers(OS, Quals, true, false);
}

void NodeArrayNode::output(OutputStream &OS) const { output(OS, ", "); }

void NodeArrayNode::output(OutputStream &OS, StringView Separator) const {
  if (Count == 0)
    return;
  if (Nodes[0])
    Nodes[0]->output(OS);
  for (size_t I = 1; I < Count; ++I) {
    OS << Separator;
    Nodes[I]->output(OS);
  }
}

void EncodedStringLiteralNode::output(OutputStream &OS) const {
  switch (Char) {
  case CharKind::Wchar:
    OS << "const wchar_t * {L\"";
    break;
  case CharKind::Char:
    OS << "const char * {\"";
    break;
  case CharKind::Char16:
    OS << "const char16_t * {u\"";
    break;
  case CharKind::Char32:
    OS << "const char32_t * {U\"";
    break;
  }
  OS << DecodedString << "\"";
  if (IsTruncated)
    OS << "...";
  OS << "}";
}

void IntegerLiteralNode::output(OutputStream &OS) const {
  if (IsNegative)
    OS << '-';
  OS << Value;
}

void TemplateParameterReferenceNode::output(OutputStream &OS) const {
  if (ThunkOffsetCount > 0)
    OS << "{";
  else if (Affinity == PointerAffinity::Pointer)
    OS << "&";

  if (Symbol) {
    Symbol->output(OS);
    if (ThunkOffsetCount > 0)
      OS << ", ";
  }

  if (ThunkOffsetCount > 0)
    OS << ThunkOffsets[0];
  for (int I = 1; I < ThunkOffsetCount; ++I) {
    OS << ", " << ThunkOffsets[I];
  }
  if (ThunkOffsetCount > 0)
    OS << "}";
}

void IdentifierNode::outputTemplateParameters(OutputStream &OS) const {
  if (!TemplateParams)
    return;
  OS << "<";
  TemplateParams->output(OS);
  OS << ">";
}

void DynamicStructorIdentifierNode::output(OutputStream &OS) const {
  if (IsDestructor)
    OS << "`dynamic atexit destructor for ";
  else
    OS << "`dynamic initializer for ";

  OS << "'";
  Name->output(OS);
  OS << "''";
}

void NamedIdentifierNode::output(OutputStream &OS) const {
  OS << Name;
  outputTemplateParameters(OS);
}

void IntrinsicFunctionIdentifierNode::output(OutputStream &OS) const {
  switch (Operator) {
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, New, "operator new");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, Delete, "operator delete");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, Assign, "operator=");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, RightShift, "operator>>");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, LeftShift, "operator<<");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, LogicalNot, "operator!");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, Equals, "operator==");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, NotEquals, "operator!=");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, ArraySubscript,
                            "operator[]");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, Pointer, "operator->");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, Increment, "operator++");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, Decrement, "operator--");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, Minus, "operator-");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, Plus, "operator+");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, Dereference, "operator*");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, BitwiseAnd, "operator&");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, MemberPointer,
                            "operator->*");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, Divide, "operator/");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, Modulus, "operator%");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, LessThan, "operator<");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, LessThanEqual, "operator<=");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, GreaterThan, "operator>");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, GreaterThanEqual,
                            "operator>=");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, Comma, "operator,");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, Parens, "operator()");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, BitwiseNot, "operator~");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, BitwiseXor, "operator^");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, BitwiseOr, "operator|");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, LogicalAnd, "operator&&");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, LogicalOr, "operator||");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, TimesEqual, "operator*=");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, PlusEqual, "operator+=");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, MinusEqual, "operator-=");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, DivEqual, "operator/=");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, ModEqual, "operator%=");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, RshEqual, "operator>>=");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, LshEqual, "operator<<=");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, BitwiseAndEqual,
                            "operator&=");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, BitwiseOrEqual,
                            "operator|=");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, BitwiseXorEqual,
                            "operator^=");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, VbaseDtor, "`vbase dtor'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, VecDelDtor,
                            "`vector deleting dtor'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, DefaultCtorClosure,
                            "`default ctor closure'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, ScalarDelDtor,
                            "`scalar deleting dtor'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, VecCtorIter,
                            "`vector ctor iterator'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, VecDtorIter,
                            "`vector dtor iterator'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, VecVbaseCtorIter,
                            "`vector vbase ctor iterator'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, VdispMap,
                            "`virtual displacement map'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, EHVecCtorIter,
                            "`eh vector ctor iterator'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, EHVecDtorIter,
                            "`eh vector dtor iterator'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, EHVecVbaseCtorIter,
                            "`eh vector vbase ctor iterator'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, CopyCtorClosure,
                            "`copy ctor closure'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, LocalVftableCtorClosure,
                            "`local vftable ctor closure'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, ArrayNew, "operator new[]");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, ArrayDelete,
                            "operator delete[]");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, ManVectorCtorIter,
                            "`managed vector ctor iterator'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, ManVectorDtorIter,
                            "`managed vector dtor iterator'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, EHVectorCopyCtorIter,
                            "`EH vector copy ctor iterator'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, EHVectorVbaseCopyCtorIter,
                            "`EH vector vbase copy ctor iterator'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, VectorCopyCtorIter,
                            "`vector copy ctor iterator'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, VectorVbaseCopyCtorIter,
                            "`vector vbase copy constructor iterator'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, ManVectorVbaseCopyCtorIter,
                            "`managed vector vbase copy constructor iterator'");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, CoAwait, "co_await");
    OUTPUT_ENUM_CLASS_VALUE(IntrinsicFunctionKind, Spaceship, "operator <=>");
  case IntrinsicFunctionKind::MaxIntrinsic:
  case IntrinsicFunctionKind::None:
    break;
  }
  outputTemplateParameters(OS);
}

void LocalStaticGuardIdentifierNode::output(OutputStream &OS) const {
  OS << "`local static guard'";
  if (ScopeIndex > 0)
    OS << "{" << ScopeIndex << "}";
}

void ConversionOperatorIdentifierNode::output(OutputStream &OS) const {
  OS << "operator";
  outputTemplateParameters(OS);
  OS << " ";
  TargetType->output(OS);
}

void StructorIdentifierNode::output(OutputStream &OS) const {
  if (IsDestructor)
    OS << "~";
  Class->output(OS);
  outputTemplateParameters(OS);
}

void LiteralOperatorIdentifierNode::output(OutputStream &OS) const {
  OS << "operator \"\"" << Name;
  outputTemplateParameters(OS);
}

void FunctionSignatureNode::outputPre(OutputStream &OS,
                                      FunctionSigFlags Flags) const {
  if (!(FunctionClass & FC_Global)) {
    if (FunctionClass & FC_Static)
      OS << "static ";
  }
  if (FunctionClass & FC_ExternC)
    OS << "extern \"C\" ";

  if (FunctionClass & FC_Virtual)
    OS << "virtual ";

  if (ReturnType) {
    ReturnType->outputPre(OS);
    OS << " ";
  }

  if (!(Flags & FSF_NoCallingConvention))
    outputCallingConvention(OS, CallConvention);
}

void FunctionSignatureNode::outputPost(OutputStream &OS,
                                       FunctionSigFlags Flags) const {
  if (!(FunctionClass & FC_NoParameterList)) {
    OS << "(";
    if (Params)
      Params->output(OS);
    else
      OS << "void";
    OS << ")";
  }

  if (Quals & Q_Const)
    OS << " const";
  if (Quals & Q_Volatile)
    OS << " volatile";
  if (Quals & Q_Restrict)
    OS << " __restrict";
  if (Quals & Q_Unaligned)
    OS << " __unaligned";

  if (RefQualifier == FunctionRefQualifier::Reference)
    OS << " &";
  else if (RefQualifier == FunctionRefQualifier::RValueReference)
    OS << " &&";

  if (ReturnType)
    ReturnType->outputPost(OS);
}

void ThunkSignatureNode::outputPre(OutputStream &OS,
                                   FunctionSigFlags Flags) const {
  OS << "[thunk]: ";

  FunctionSignatureNode::outputPre(OS, Flags);
}

void ThunkSignatureNode::outputPost(OutputStream &OS,
                                    FunctionSigFlags Flags) const {
  if (FunctionClass & FC_StaticThisAdjust) {
    OS << "`adjustor{" << ThisAdjust.StaticOffset << "}'";
  } else if (FunctionClass & FC_VirtualThisAdjust) {
    if (FunctionClass & FC_VirtualThisAdjustEx) {
      OS << "`vtordispex{" << ThisAdjust.VBPtrOffset << ", "
         << ThisAdjust.VBOffsetOffset << ", " << ThisAdjust.VtordispOffset
         << ", " << ThisAdjust.StaticOffset << "}'";
    } else {
      OS << "`vtordisp{" << ThisAdjust.VtordispOffset << ", "
         << ThisAdjust.StaticOffset << "}'";
    }
  }

  FunctionSignatureNode::outputPost(OS, Flags);
}

void PointerTypeNode::outputPre(OutputStream &OS) const {
  if (Pointee->kind() == NodeKind::FunctionSignature) {
    // If this is a pointer to a function, don't output the calling convention.
    // It needs to go inside the parentheses.
    const FunctionSignatureNode *Sig =
        static_cast<const FunctionSignatureNode *>(Pointee);
    Sig->outputPre(OS, FSF_NoCallingConvention);
  } else
    Pointee->outputPre(OS);

  outputSpaceIfNecessary(OS);

  if (Quals & Q_Unaligned)
    OS << "__unaligned ";

  if (Pointee->kind() == NodeKind::ArrayType) {
    OS << "(";
  } else if (Pointee->kind() == NodeKind::FunctionSignature) {
    OS << "(";
    const FunctionSignatureNode *Sig =
        static_cast<const FunctionSignatureNode *>(Pointee);
    outputCallingConvention(OS, Sig->CallConvention);
    OS << " ";
  }

  if (ClassParent) {
    ClassParent->output(OS);
    OS << "::";
  }

  switch (Affinity) {
  case PointerAffinity::Pointer:
    OS << "*";
    break;
  case PointerAffinity::Reference:
    OS << "&";
    break;
  case PointerAffinity::RValueReference:
    OS << "&&";
    break;
  default:
    assert(false);
  }
  outputQualifiers(OS, Quals, false, false);
}

void PointerTypeNode::outputPost(OutputStream &OS) const {
  if (Pointee->kind() == NodeKind::ArrayType ||
      Pointee->kind() == NodeKind::FunctionSignature)
    OS << ")";

  Pointee->outputPost(OS);
}

void TagTypeNode::outputPre(OutputStream &OS) const {
  switch (Tag) {
    OUTPUT_ENUM_CLASS_VALUE(TagKind, Class, "class");
    OUTPUT_ENUM_CLASS_VALUE(TagKind, Struct, "struct");
    OUTPUT_ENUM_CLASS_VALUE(TagKind, Union, "union");
    OUTPUT_ENUM_CLASS_VALUE(TagKind, Enum, "enum");
  }
  OS << " ";
  QualifiedName->output(OS);
  outputQualifiers(OS, Quals, true, false);
}

void TagTypeNode::outputPost(OutputStream &OS) const {}

void ArrayTypeNode::outputPre(OutputStream &OS) const {
  ElementType->outputPre(OS);
  outputQualifiers(OS, Quals, true, false);
}

void ArrayTypeNode::outputOneDimension(OutputStream &OS, Node *N) const {
  assert(N->kind() == NodeKind::IntegerLiteral);
  IntegerLiteralNode *ILN = static_cast<IntegerLiteralNode *>(N);
  if (ILN->Value != 0)
    ILN->output(OS);
}

void ArrayTypeNode::outputDimensionsImpl(OutputStream &OS) const {
  if (Dimensions->Count == 0)
    return;

  outputOneDimension(OS, Dimensions->Nodes[0]);
  for (size_t I = 1; I < Dimensions->Count; ++I) {
    OS << "][";
    outputOneDimension(OS, Dimensions->Nodes[I]);
  }
}

void ArrayTypeNode::outputPost(OutputStream &OS) const {
  OS << "[";
  outputDimensionsImpl(OS);
  OS << "]";

  ElementType->outputPost(OS);
}

void SymbolNode::output(OutputStream &OS) const { Name->output(OS); }

void FunctionSymbolNode::output(OutputStream &OS) const {
  output(OS, FunctionSigFlags::FSF_Default);
}

void FunctionSymbolNode::output(OutputStream &OS,
                                FunctionSigFlags Flags) const {
  Signature->outputPre(OS, Flags);
  outputSpaceIfNecessary(OS);
  Name->output(OS);
  Signature->outputPost(OS, Flags);
}

void VariableSymbolNode::output(OutputStream &OS) const {
  switch (SC) {
  case StorageClass::PrivateStatic:
  case StorageClass::PublicStatic:
  case StorageClass::ProtectedStatic:
    OS << "static ";
  default:
    break;
  }

  if (Type) {
    Type->outputPre(OS);
    outputSpaceIfNecessary(OS);
  }
  Name->output(OS);
  if (Type)
    Type->outputPost(OS);
}

void CustomNode::output(OutputStream &OS) const { OS << Name; }

void QualifiedNameNode::output(OutputStream &OS) const {
  Components->output(OS, "::");
}

void RttiBaseClassDescriptorNode::output(OutputStream &OS) const {
  OS << "`RTTI Base Class Descriptor at (";
  OS << NVOffset << ", " << VBPtrOffset << ", " << VBTableOffset << ", "
     << Flags;
  OS << ")'";
}

void LocalStaticGuardVariableNode::output(OutputStream &OS) const {
  Name->output(OS);
}

void VcallThunkIdentifierNode::output(OutputStream &OS) const {
  OS << "`vcall'{" << OffsetInVTable << ", {flat}}";
}

void SpecialTableSymbolNode::output(OutputStream &OS) const {
  outputQualifiers(OS, Quals, false, true);
  Name->output(OS);
  if (TargetName) {
    OS << "{for `";
    TargetName->output(OS);
    OS << "'}";
  }
  return;
}
