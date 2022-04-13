//===- DWARFDie.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFAbbreviationDeclaration.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLine.h"
#include "llvm/DebugInfo/DWARF/DWARFDebugLoc.h"
#include "llvm/DebugInfo/DWARF/DWARFExpression.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFTypeUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <string>
#include <utility>

using namespace llvm;
using namespace dwarf;
using namespace object;

static void dumpApplePropertyAttribute(raw_ostream &OS, uint64_t Val) {
  OS << " (";
  do {
    uint64_t Shift = countTrailingZeros(Val);
    assert(Shift < 64 && "undefined behavior");
    uint64_t Bit = 1ULL << Shift;
    auto PropName = ApplePropertyString(Bit);
    if (!PropName.empty())
      OS << PropName;
    else
      OS << format("DW_APPLE_PROPERTY_0x%" PRIx64, Bit);
    if (!(Val ^= Bit))
      break;
    OS << ", ";
  } while (true);
  OS << ")";
}

static void dumpRanges(const DWARFObject &Obj, raw_ostream &OS,
                       const DWARFAddressRangesVector &Ranges,
                       unsigned AddressSize, unsigned Indent,
                       const DIDumpOptions &DumpOpts) {
  if (!DumpOpts.ShowAddresses)
    return;

  for (const DWARFAddressRange &R : Ranges) {
    OS << '\n';
    OS.indent(Indent);
    R.dump(OS, AddressSize, DumpOpts, &Obj);
  }
}

static void dumpLocationList(raw_ostream &OS, const DWARFFormValue &FormValue,
                             DWARFUnit *U, unsigned Indent,
                             DIDumpOptions DumpOpts) {
  assert(FormValue.isFormClass(DWARFFormValue::FC_SectionOffset) &&
         "bad FORM for location list");
  DWARFContext &Ctx = U->getContext();
  const MCRegisterInfo *MRI = Ctx.getRegisterInfo();
  uint64_t Offset = *FormValue.getAsSectionOffset();

  if (FormValue.getForm() == DW_FORM_loclistx) {
    FormValue.dump(OS, DumpOpts);

    if (auto LoclistOffset = U->getLoclistOffset(Offset))
      Offset = *LoclistOffset;
    else
      return;
  }
  U->getLocationTable().dumpLocationList(&Offset, OS, U->getBaseAddress(), MRI,
                                         Ctx.getDWARFObj(), U, DumpOpts,
                                         Indent);
}

static void dumpLocationExpr(raw_ostream &OS, const DWARFFormValue &FormValue,
                             DWARFUnit *U, unsigned Indent,
                             DIDumpOptions DumpOpts) {
  assert((FormValue.isFormClass(DWARFFormValue::FC_Block) ||
          FormValue.isFormClass(DWARFFormValue::FC_Exprloc)) &&
         "bad FORM for location expression");
  DWARFContext &Ctx = U->getContext();
  const MCRegisterInfo *MRI = Ctx.getRegisterInfo();
  ArrayRef<uint8_t> Expr = *FormValue.getAsBlock();
  DataExtractor Data(StringRef((const char *)Expr.data(), Expr.size()),
                     Ctx.isLittleEndian(), 0);
  DWARFExpression(Data, U->getAddressByteSize(), U->getFormParams().Format)
      .print(OS, DumpOpts, MRI, U);
}

static DWARFDie resolveReferencedType(DWARFDie D,
                                      dwarf::Attribute Attr = DW_AT_type) {
  return D.getAttributeValueAsReferencedDie(Attr).resolveTypeUnitReference();
}
static DWARFDie resolveReferencedType(DWARFDie D, DWARFFormValue F) {
  return D.getAttributeValueAsReferencedDie(F).resolveTypeUnitReference();
}

namespace {

// FIXME: We should have pretty printers per language. Currently we print
// everything as if it was C++ and fall back to the TAG type name.
struct DWARFTypePrinter {
  raw_ostream &OS;
  bool Word = true;
  bool EndedWithTemplate = false;

  DWARFTypePrinter(raw_ostream &OS) : OS(OS) {}

  /// Dump the name encoded in the type tag.
  void appendTypeTagName(dwarf::Tag T) {
    StringRef TagStr = TagString(T);
    static constexpr StringRef Prefix = "DW_TAG_";
    static constexpr StringRef Suffix = "_type";
    if (!TagStr.startswith(Prefix) || !TagStr.endswith(Suffix))
      return;
    OS << TagStr.substr(Prefix.size(),
                        TagStr.size() - (Prefix.size() + Suffix.size()))
       << " ";
  }

  void appendArrayType(const DWARFDie &D) {
    for (const DWARFDie &C : D.children()) {
      if (C.getTag() != DW_TAG_subrange_type)
        continue;
      Optional<uint64_t> LB;
      Optional<uint64_t> Count;
      Optional<uint64_t> UB;
      Optional<unsigned> DefaultLB;
      if (Optional<DWARFFormValue> L = C.find(DW_AT_lower_bound))
        LB = L->getAsUnsignedConstant();
      if (Optional<DWARFFormValue> CountV = C.find(DW_AT_count))
        Count = CountV->getAsUnsignedConstant();
      if (Optional<DWARFFormValue> UpperV = C.find(DW_AT_upper_bound))
        UB = UpperV->getAsUnsignedConstant();
      if (Optional<DWARFFormValue> LV =
              D.getDwarfUnit()->getUnitDIE().find(DW_AT_language))
        if (Optional<uint64_t> LC = LV->getAsUnsignedConstant())
          if ((DefaultLB =
                   LanguageLowerBound(static_cast<dwarf::SourceLanguage>(*LC))))
            if (LB && *LB == *DefaultLB)
              LB = None;
      if (!LB && !Count && !UB)
        OS << "[]";
      else if (!LB && (Count || UB) && DefaultLB)
        OS << '[' << (Count ? *Count : *UB - *DefaultLB + 1) << ']';
      else {
        OS << "[[";
        if (LB)
          OS << *LB;
        else
          OS << '?';
        OS << ", ";
        if (Count)
          if (LB)
            OS << *LB + *Count;
          else
            OS << "? + " << *Count;
        else if (UB)
          OS << *UB + 1;
        else
          OS << '?';
        OS << ")]";
      }
    }
    EndedWithTemplate = false;
  }

  DWARFDie skipQualifiers(DWARFDie D) {
    while (D && (D.getTag() == DW_TAG_const_type ||
                 D.getTag() == DW_TAG_volatile_type))
      D = resolveReferencedType(D);
    return D;
  }

  bool needsParens(DWARFDie D) {
    D = skipQualifiers(D);
    return D && (D.getTag() == DW_TAG_subroutine_type || D.getTag() == DW_TAG_array_type);
  }

  void appendPointerLikeTypeBefore(DWARFDie D, DWARFDie Inner, StringRef Ptr) {
    appendQualifiedNameBefore(Inner);
    if (Word)
      OS << ' ';
    if (needsParens(Inner))
      OS << '(';
    OS << Ptr;
    Word = false;
    EndedWithTemplate = false;
  }

  DWARFDie
  appendUnqualifiedNameBefore(DWARFDie D,
                              std::string *OriginalFullName = nullptr) {
    Word = true;
    if (!D) {
      OS << "void";
      return DWARFDie();
    }
    DWARFDie InnerDIE;
    auto Inner = [&] { return InnerDIE = resolveReferencedType(D); };
    const dwarf::Tag T = D.getTag();
    switch (T) {
    case DW_TAG_pointer_type: {
      appendPointerLikeTypeBefore(D, Inner(), "*");
      break;
    }
    case DW_TAG_subroutine_type: {
      appendQualifiedNameBefore(Inner());
      if (Word) {
        OS << ' ';
      }
      Word = false;
      break;
    }
    case DW_TAG_array_type: {
      appendQualifiedNameBefore(Inner());
      break;
    }
    case DW_TAG_reference_type:
      appendPointerLikeTypeBefore(D, Inner(), "&");
      break;
    case DW_TAG_rvalue_reference_type:
      appendPointerLikeTypeBefore(D, Inner(), "&&");
      break;
    case DW_TAG_ptr_to_member_type: {
      appendQualifiedNameBefore(Inner());
      if (needsParens(InnerDIE))
        OS << '(';
      else if (Word)
        OS << ' ';
      if (DWARFDie Cont = resolveReferencedType(D, DW_AT_containing_type)) {
        appendQualifiedName(Cont);
        EndedWithTemplate = false;
        OS << "::";
      }
      OS << "*";
      Word = false;
      break;
    }
    case DW_TAG_const_type:
    case DW_TAG_volatile_type:
      appendConstVolatileQualifierBefore(D);
      break;
    case DW_TAG_namespace: {
      if (const char *Name = dwarf::toString(D.find(DW_AT_name), nullptr))
        OS << Name;
      else
        OS << "(anonymous namespace)";
      break;
    }
    case DW_TAG_unspecified_type: {
      StringRef TypeName = D.getShortName();
      if (TypeName == "decltype(nullptr)")
        TypeName = "std::nullptr_t";
      Word = true;
      OS << TypeName;
      EndedWithTemplate = false;
      break;
    }
      /*
    case DW_TAG_structure_type:
    case DW_TAG_class_type:
    case DW_TAG_enumeration_type:
    case DW_TAG_base_type:
    */
    default: {
      const char *NamePtr = dwarf::toString(D.find(DW_AT_name), nullptr);
      if (!NamePtr) {
        appendTypeTagName(D.getTag());
        return DWARFDie();
      }
      Word = true;
      StringRef Name = NamePtr;
      static constexpr StringRef MangledPrefix = "_STN|";
      if (Name.startswith(MangledPrefix)) {
        Name = Name.drop_front(MangledPrefix.size());
        auto Separator = Name.find('|');
        assert(Separator != StringRef::npos);
        StringRef BaseName = Name.substr(0, Separator);
        StringRef TemplateArgs = Name.substr(Separator + 1);
        if (OriginalFullName)
          *OriginalFullName = (BaseName + TemplateArgs).str();
        Name = BaseName;
      } else
        EndedWithTemplate = Name.endswith(">");
      OS << Name;
      // This check would be insufficient for operator overloads like
      // "operator>>" - but for now Clang doesn't try to simplify them, so this
      // is OK. Add more nuanced operator overload handling here if/when needed.
      if (Name.endswith(">"))
        break;
      if (!appendTemplateParameters(D))
        break;

      if (EndedWithTemplate)
        OS << ' ';
      OS << '>';
      EndedWithTemplate = true;
      Word = true;
      break;
    }
    }
    return InnerDIE;
  }

  void appendUnqualifiedNameAfter(DWARFDie D, DWARFDie Inner,
                                  bool SkipFirstParamIfArtificial = false) {
    if (!D)
      return;
    switch (D.getTag()) {
    case DW_TAG_subroutine_type: {
      appendSubroutineNameAfter(D, Inner, SkipFirstParamIfArtificial, false,
                                false);
      break;
    }
    case DW_TAG_array_type: {
      appendArrayType(D);
      break;
    }
    case DW_TAG_const_type:
    case DW_TAG_volatile_type:
      appendConstVolatileQualifierAfter(D);
      break;
    case DW_TAG_ptr_to_member_type:
    case DW_TAG_reference_type:
    case DW_TAG_rvalue_reference_type:
    case DW_TAG_pointer_type: {
      if (needsParens(Inner))
        OS << ')';
      appendUnqualifiedNameAfter(Inner, resolveReferencedType(Inner),
                                 /*SkipFirstParamIfArtificial=*/D.getTag() ==
                                     DW_TAG_ptr_to_member_type);
      break;
    }
      /*
    case DW_TAG_structure_type:
    case DW_TAG_class_type:
    case DW_TAG_enumeration_type:
    case DW_TAG_base_type:
    case DW_TAG_namespace:
    */
    default:
      break;
    }
  }

  void appendQualifiedName(DWARFDie D) {
    if (D)
      appendScopes(D.getParent());
    appendUnqualifiedName(D);
  }
  DWARFDie appendQualifiedNameBefore(DWARFDie D) {
    if (D)
      appendScopes(D.getParent());
    return appendUnqualifiedNameBefore(D);
  }
  bool appendTemplateParameters(DWARFDie D, bool *FirstParameter = nullptr) {
    bool FirstParameterValue = true;
    bool IsTemplate = false;
    if (!FirstParameter)
      FirstParameter = &FirstParameterValue;
    for (const DWARFDie &C : D) {
      auto Sep = [&] {
        if (*FirstParameter)
          OS << '<';
        else
          OS << ", ";
        IsTemplate = true;
        EndedWithTemplate = false;
        *FirstParameter = false;
      };
      if (C.getTag() == dwarf::DW_TAG_GNU_template_parameter_pack) {
        IsTemplate = true;
        appendTemplateParameters(C, FirstParameter);
      }
      if (C.getTag() == dwarf::DW_TAG_template_value_parameter) {
        DWARFDie T = resolveReferencedType(C);
        Sep();
        if (T.getTag() == DW_TAG_enumeration_type) {
          OS << '(';
          appendQualifiedName(T);
          OS << ')';
          auto V = C.find(DW_AT_const_value);
          OS << to_string(*V->getAsSignedConstant());
          continue;
        }
        // /Maybe/ we could do pointer type parameters, looking for the
        // symbol in the ELF symbol table to get back to the variable...
        // but probably not worth it.
        if (T.getTag() == DW_TAG_pointer_type)
          continue;
        const char *RawName = dwarf::toString(T.find(DW_AT_name), nullptr);
        assert(RawName);
        StringRef Name = RawName;
        auto V = C.find(DW_AT_const_value);
        bool IsQualifiedChar = false;
        if (Name == "bool") {
          OS << (*V->getAsUnsignedConstant() ? "true" : "false");
        } else if (Name == "short") {
          OS << "(short)";
          OS << to_string(*V->getAsSignedConstant());
        } else if (Name == "unsigned short") {
          OS << "(unsigned short)";
          OS << to_string(*V->getAsSignedConstant());
        } else if (Name == "int")
          OS << to_string(*V->getAsSignedConstant());
        else if (Name == "long") {
          OS << to_string(*V->getAsSignedConstant());
          OS << "L";
        } else if (Name == "long long") {
          OS << to_string(*V->getAsSignedConstant());
          OS << "LL";
        } else if (Name == "unsigned int") {
          OS << to_string(*V->getAsUnsignedConstant());
          OS << "U";
        } else if (Name == "unsigned long") {
          OS << to_string(*V->getAsUnsignedConstant());
          OS << "UL";
        } else if (Name == "unsigned long long") {
          OS << to_string(*V->getAsUnsignedConstant());
          OS << "ULL";
        } else if (Name == "char" ||
                   (IsQualifiedChar =
                        (Name == "unsigned char" || Name == "signed char"))) {
          // FIXME: check T's DW_AT_type to see if it's signed or not (since
          // char signedness is implementation defined).
          auto Val = *V->getAsSignedConstant();
          // Copied/hacked up from Clang's CharacterLiteral::print - incomplete
          // (doesn't actually support different character types/widths, sign
          // handling's not done, and doesn't correctly test if a character is
          // printable or needs to use a numeric escape sequence instead)
          if (IsQualifiedChar) {
            OS << '(';
            OS << Name;
            OS << ')';
          }
          switch (Val) {
          case '\\':
            OS << "'\\\\'";
            break;
          case '\'':
            OS << "'\\''";
            break;
          case '\a':
            // TODO: K&R: the meaning of '\\a' is different in traditional C
            OS << "'\\a'";
            break;
          case '\b':
            OS << "'\\b'";
            break;
          case '\f':
            OS << "'\\f'";
            break;
          case '\n':
            OS << "'\\n'";
            break;
          case '\r':
            OS << "'\\r'";
            break;
          case '\t':
            OS << "'\\t'";
            break;
          case '\v':
            OS << "'\\v'";
            break;
          default:
            if ((Val & ~0xFFu) == ~0xFFu)
              Val &= 0xFFu;
            if (Val < 127 && Val >= 32) {
              OS << "'";
              OS << (char)Val;
              OS << "'";
            } else if (Val < 256)
              OS << to_string(llvm::format("'\\x%02x'", Val));
            else if (Val <= 0xFFFF)
              OS << to_string(llvm::format("'\\u%04x'", Val));
            else
              OS << to_string(llvm::format("'\\U%08x'", Val));
          }
        }
        continue;
      }
      if (C.getTag() == dwarf::DW_TAG_GNU_template_template_param) {
        const char *RawName =
            dwarf::toString(C.find(DW_AT_GNU_template_name), nullptr);
        assert(RawName);
        StringRef Name = RawName;
        Sep();
        OS << Name;
        continue;
      }
      if (C.getTag() != dwarf::DW_TAG_template_type_parameter)
        continue;
      auto TypeAttr = C.find(DW_AT_type);
      Sep();
      appendQualifiedName(TypeAttr ? resolveReferencedType(C, *TypeAttr)
                                   : DWARFDie());
    }
    if (IsTemplate && *FirstParameter && FirstParameter == &FirstParameterValue) {
      OS << '<';
      EndedWithTemplate = false;
    }
    return IsTemplate;
  }
  void decomposeConstVolatile(DWARFDie &N, DWARFDie &T, DWARFDie &C,
                              DWARFDie &V) {
    (N.getTag() == DW_TAG_const_type ? C : V) = N;
    T = resolveReferencedType(N);
    if (T) {
      auto Tag = T.getTag();
      if (Tag == DW_TAG_const_type) {
        C = T;
        T = resolveReferencedType(T);
      } else if (Tag == DW_TAG_volatile_type) {
        V = T;
        T = resolveReferencedType(T);
      }
    }
  }
  void appendConstVolatileQualifierAfter(DWARFDie N) {
    DWARFDie C;
    DWARFDie V;
    DWARFDie T;
    decomposeConstVolatile(N, T, C, V);
    if (T && T.getTag() == DW_TAG_subroutine_type)
      appendSubroutineNameAfter(T, resolveReferencedType(T), false, C.isValid(),
                                V.isValid());
    else
      appendUnqualifiedNameAfter(T, resolveReferencedType(T));
  }
  void appendConstVolatileQualifierBefore(DWARFDie N) {
    DWARFDie C;
    DWARFDie V;
    DWARFDie T;
    decomposeConstVolatile(N, T, C, V);
    bool Subroutine = T && T.getTag() == DW_TAG_subroutine_type;
    DWARFDie A = T;
    while (A && A.getTag() == DW_TAG_array_type)
      A = resolveReferencedType(A);
    bool Leading =
        (!A || (A.getTag() != DW_TAG_pointer_type &&
                A.getTag() != llvm::dwarf::DW_TAG_ptr_to_member_type)) &&
        !Subroutine;
    if (Leading) {
      if (C)
        OS << "const ";
      if (V)
        OS << "volatile ";
    }
    appendQualifiedNameBefore(T);
    if (!Leading && !Subroutine) {
      Word = true;
      if (C)
        OS << "const";
      if (V) {
        if (C)
          OS << ' ';
        OS << "volatile";
      }
    }
  }

  /// Recursively append the DIE type name when applicable.
  void appendUnqualifiedName(DWARFDie D,
                             std::string *OriginalFullName = nullptr) {
    // FIXME: We should have pretty printers per language. Currently we print
    // everything as if it was C++ and fall back to the TAG type name.
    DWARFDie Inner = appendUnqualifiedNameBefore(D, OriginalFullName);
    appendUnqualifiedNameAfter(D, Inner);
  }

  void appendSubroutineNameAfter(DWARFDie D, DWARFDie Inner,
                                 bool SkipFirstParamIfArtificial, bool Const,
                                 bool Volatile) {
    DWARFDie FirstParamIfArtificial;
    OS << '(';
    EndedWithTemplate = false;
    bool First = true;
    bool RealFirst = true;
    for (DWARFDie P : D) {
      if (P.getTag() != DW_TAG_formal_parameter &&
          P.getTag() != DW_TAG_unspecified_parameters)
        return;
      DWARFDie T = resolveReferencedType(P);
      if (SkipFirstParamIfArtificial && RealFirst && P.find(DW_AT_artificial)) {
        FirstParamIfArtificial = T;
        RealFirst = false;
        continue;
      }
      if (!First) {
        OS << ", ";
      }
      First = false;
      if (P.getTag() == DW_TAG_unspecified_parameters)
        OS << "...";
      else
        appendQualifiedName(T);
    }
    EndedWithTemplate = false;
    OS << ')';
    if (FirstParamIfArtificial) {
      if (DWARFDie P = FirstParamIfArtificial) {
        if (P.getTag() == DW_TAG_pointer_type) {
          auto CVStep = [&](DWARFDie CV) {
            if (DWARFDie U = resolveReferencedType(CV)) {
              Const |= U.getTag() == DW_TAG_const_type;
              Volatile |= U.getTag() == DW_TAG_volatile_type;
              return U;
            }
            return DWARFDie();
          };
          if (DWARFDie CV = CVStep(P)) {
            CVStep(CV);
          }
        }
      }
    }

    if (auto CC = D.find(DW_AT_calling_convention)) {
      switch (*CC->getAsUnsignedConstant()) {
      case CallingConvention::DW_CC_BORLAND_stdcall:
        OS << " __attribute__((stdcall))";
        break;
      case CallingConvention::DW_CC_BORLAND_msfastcall:
        OS << " __attribute__((fastcall))";
        break;
      case CallingConvention::DW_CC_BORLAND_thiscall:
        OS << " __attribute__((thiscall))";
        break;
      case CallingConvention::DW_CC_LLVM_vectorcall:
        OS << " __attribute__((vectorcall))";
        break;
      case CallingConvention::DW_CC_BORLAND_pascal:
        OS << " __attribute__((pascal))";
        break;
      case CallingConvention::DW_CC_LLVM_Win64:
        OS << " __attribute__((ms_abi))";
        break;
      case CallingConvention::DW_CC_LLVM_X86_64SysV:
        OS << " __attribute__((sysv_abi))";
        break;
      case CallingConvention::DW_CC_LLVM_AAPCS:
        // AArch64VectorCall missing?
        OS << " __attribute__((pcs(\"aapcs\")))";
        break;
      case CallingConvention::DW_CC_LLVM_AAPCS_VFP:
        OS << " __attribute__((pcs(\"aapcs-vfp\")))";
        break;
      case CallingConvention::DW_CC_LLVM_IntelOclBicc:
        OS << " __attribute__((intel_ocl_bicc))";
        break;
      case CallingConvention::DW_CC_LLVM_SpirFunction:
      case CallingConvention::DW_CC_LLVM_OpenCLKernel:
        // These aren't available as attributes, but maybe we should still
        // render them somehow? (Clang doesn't render them, but that's an issue
        // for template names too - since then the DWARF names of templates
        // instantiated with function types with these calling conventions won't
        // have distinct names - so we'd need to fix that too)
        break;
      case CallingConvention::DW_CC_LLVM_Swift:
        // SwiftAsync missing
        OS << " __attribute__((swiftcall))";
        break;
      case CallingConvention::DW_CC_LLVM_PreserveMost:
        OS << " __attribute__((preserve_most))";
        break;
      case CallingConvention::DW_CC_LLVM_PreserveAll:
        OS << " __attribute__((preserve_all))";
        break;
      case CallingConvention::DW_CC_LLVM_X86RegCall:
        OS << " __attribute__((regcall))";
        break;
      }
    }

    if (Const)
      OS << " const";
    if (Volatile)
      OS << " volatile";
    if (D.find(DW_AT_reference))
      OS << " &";
    if (D.find(DW_AT_rvalue_reference))
      OS << " &&";

    appendUnqualifiedNameAfter(Inner, resolveReferencedType(Inner));
  }
  void appendScopes(DWARFDie D) {
    if (D.getTag() == DW_TAG_compile_unit)
      return;
    if (D.getTag() == DW_TAG_type_unit)
      return;
    if (D.getTag() == DW_TAG_skeleton_unit)
      return;
    if (D.getTag() == DW_TAG_subprogram)
      return;
    if (D.getTag() == DW_TAG_lexical_block)
      return;
    D = D.resolveTypeUnitReference();
    if (DWARFDie P = D.getParent())
      appendScopes(P);
    appendUnqualifiedName(D);
    OS << "::";
  }
};
} // anonymous namespace

static void dumpAttribute(raw_ostream &OS, const DWARFDie &Die,
                          const DWARFAttribute &AttrValue, unsigned Indent,
                          DIDumpOptions DumpOpts) {
  if (!Die.isValid())
    return;
  const char BaseIndent[] = "            ";
  OS << BaseIndent;
  OS.indent(Indent + 2);
  dwarf::Attribute Attr = AttrValue.Attr;
  WithColor(OS, HighlightColor::Attribute) << formatv("{0}", Attr);

  dwarf::Form Form = AttrValue.Value.getForm();
  if (DumpOpts.Verbose || DumpOpts.ShowForm)
    OS << formatv(" [{0}]", Form);

  DWARFUnit *U = Die.getDwarfUnit();
  const DWARFFormValue &FormValue = AttrValue.Value;

  OS << "\t(";

  StringRef Name;
  std::string File;
  auto Color = HighlightColor::Enumerator;
  if (Attr == DW_AT_decl_file || Attr == DW_AT_call_file) {
    Color = HighlightColor::String;
    if (const auto *LT = U->getContext().getLineTableForUnit(U))
      if (LT->getFileNameByIndex(
              FormValue.getAsUnsignedConstant().getValue(),
              U->getCompilationDir(),
              DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath, File)) {
        File = '"' + File + '"';
        Name = File;
      }
  } else if (Optional<uint64_t> Val = FormValue.getAsUnsignedConstant())
    Name = AttributeValueString(Attr, *Val);

  if (!Name.empty())
    WithColor(OS, Color) << Name;
  else if (Attr == DW_AT_decl_line || Attr == DW_AT_call_line)
    OS << *FormValue.getAsUnsignedConstant();
  else if (Attr == DW_AT_low_pc &&
           (FormValue.getAsAddress() ==
            dwarf::computeTombstoneAddress(U->getAddressByteSize()))) {
    if (DumpOpts.Verbose) {
      FormValue.dump(OS, DumpOpts);
      OS << " (";
    }
    OS << "dead code";
    if (DumpOpts.Verbose)
      OS << ')';
  } else if (Attr == DW_AT_high_pc && !DumpOpts.ShowForm && !DumpOpts.Verbose &&
             FormValue.getAsUnsignedConstant()) {
    if (DumpOpts.ShowAddresses) {
      // Print the actual address rather than the offset.
      uint64_t LowPC, HighPC, Index;
      if (Die.getLowAndHighPC(LowPC, HighPC, Index))
        DWARFFormValue::dumpAddress(OS, U->getAddressByteSize(), HighPC);
      else
        FormValue.dump(OS, DumpOpts);
    }
  } else if (DWARFAttribute::mayHaveLocationList(Attr) &&
             FormValue.isFormClass(DWARFFormValue::FC_SectionOffset))
    dumpLocationList(OS, FormValue, U, sizeof(BaseIndent) + Indent + 4,
                     DumpOpts);
  else if (FormValue.isFormClass(DWARFFormValue::FC_Exprloc) ||
           (DWARFAttribute::mayHaveLocationExpr(Attr) &&
            FormValue.isFormClass(DWARFFormValue::FC_Block)))
    dumpLocationExpr(OS, FormValue, U, sizeof(BaseIndent) + Indent + 4,
                     DumpOpts);
  else
    FormValue.dump(OS, DumpOpts);

  std::string Space = DumpOpts.ShowAddresses ? " " : "";

  // We have dumped the attribute raw value. For some attributes
  // having both the raw value and the pretty-printed value is
  // interesting. These attributes are handled below.
  if (Attr == DW_AT_specification || Attr == DW_AT_abstract_origin) {
    if (const char *Name =
            Die.getAttributeValueAsReferencedDie(FormValue).getName(
                DINameKind::LinkageName))
      OS << Space << "\"" << Name << '\"';
  } else if (Attr == DW_AT_type) {
    DWARFDie D = resolveReferencedType(Die, FormValue);
    if (D && !D.isNULL()) {
      OS << Space << "\"";
      dumpTypeQualifiedName(D, OS);
      OS << '"';
    }
  } else if (Attr == DW_AT_APPLE_property_attribute) {
    if (Optional<uint64_t> OptVal = FormValue.getAsUnsignedConstant())
      dumpApplePropertyAttribute(OS, *OptVal);
  } else if (Attr == DW_AT_ranges) {
    const DWARFObject &Obj = Die.getDwarfUnit()->getContext().getDWARFObj();
    // For DW_FORM_rnglistx we need to dump the offset separately, since
    // we have only dumped the index so far.
    if (FormValue.getForm() == DW_FORM_rnglistx)
      if (auto RangeListOffset =
              U->getRnglistOffset(*FormValue.getAsSectionOffset())) {
        DWARFFormValue FV = DWARFFormValue::createFromUValue(
            dwarf::DW_FORM_sec_offset, *RangeListOffset);
        FV.dump(OS, DumpOpts);
      }
    if (auto RangesOrError = Die.getAddressRanges())
      dumpRanges(Obj, OS, RangesOrError.get(), U->getAddressByteSize(),
                 sizeof(BaseIndent) + Indent + 4, DumpOpts);
    else
      DumpOpts.RecoverableErrorHandler(createStringError(
          errc::invalid_argument, "decoding address ranges: %s",
          toString(RangesOrError.takeError()).c_str()));
  }

  OS << ")\n";
}

void DWARFDie::getFullName(raw_string_ostream &OS,
                           std::string *OriginalFullName) const {
  const char *NamePtr = getShortName();
  if (!NamePtr)
    return;
  if (getTag() == DW_TAG_GNU_template_parameter_pack)
    return;
  dumpTypeUnqualifiedName(*this, OS, OriginalFullName);
}

bool DWARFDie::isSubprogramDIE() const { return getTag() == DW_TAG_subprogram; }

bool DWARFDie::isSubroutineDIE() const {
  auto Tag = getTag();
  return Tag == DW_TAG_subprogram || Tag == DW_TAG_inlined_subroutine;
}

Optional<DWARFFormValue> DWARFDie::find(dwarf::Attribute Attr) const {
  if (!isValid())
    return None;
  auto AbbrevDecl = getAbbreviationDeclarationPtr();
  if (AbbrevDecl)
    return AbbrevDecl->getAttributeValue(getOffset(), Attr, *U);
  return None;
}

Optional<DWARFFormValue>
DWARFDie::find(ArrayRef<dwarf::Attribute> Attrs) const {
  if (!isValid())
    return None;
  auto AbbrevDecl = getAbbreviationDeclarationPtr();
  if (AbbrevDecl) {
    for (auto Attr : Attrs) {
      if (auto Value = AbbrevDecl->getAttributeValue(getOffset(), Attr, *U))
        return Value;
    }
  }
  return None;
}

Optional<DWARFFormValue>
DWARFDie::findRecursively(ArrayRef<dwarf::Attribute> Attrs) const {
  SmallVector<DWARFDie, 3> Worklist;
  Worklist.push_back(*this);

  // Keep track if DIEs already seen to prevent infinite recursion.
  // Empirically we rarely see a depth of more than 3 when dealing with valid
  // DWARF. This corresponds to following the DW_AT_abstract_origin and
  // DW_AT_specification just once.
  SmallSet<DWARFDie, 3> Seen;
  Seen.insert(*this);

  while (!Worklist.empty()) {
    DWARFDie Die = Worklist.pop_back_val();

    if (!Die.isValid())
      continue;

    if (auto Value = Die.find(Attrs))
      return Value;

    if (auto D = Die.getAttributeValueAsReferencedDie(DW_AT_abstract_origin))
      if (Seen.insert(D).second)
        Worklist.push_back(D);

    if (auto D = Die.getAttributeValueAsReferencedDie(DW_AT_specification))
      if (Seen.insert(D).second)
        Worklist.push_back(D);
  }

  return None;
}

DWARFDie
DWARFDie::getAttributeValueAsReferencedDie(dwarf::Attribute Attr) const {
  if (Optional<DWARFFormValue> F = find(Attr))
    return getAttributeValueAsReferencedDie(*F);
  return DWARFDie();
}

DWARFDie
DWARFDie::getAttributeValueAsReferencedDie(const DWARFFormValue &V) const {
  DWARFDie Result;
  if (auto SpecRef = V.getAsRelativeReference()) {
    if (SpecRef->Unit)
      Result = SpecRef->Unit->getDIEForOffset(SpecRef->Unit->getOffset() +
                                              SpecRef->Offset);
    else if (auto SpecUnit =
                 U->getUnitVector().getUnitForOffset(SpecRef->Offset))
      Result = SpecUnit->getDIEForOffset(SpecRef->Offset);
  }
  return Result;
}

DWARFDie DWARFDie::resolveTypeUnitReference() const {
  if (auto Attr = find(DW_AT_signature)) {
    if (Optional<uint64_t> Sig = Attr->getAsReferenceUVal()) {
      if (DWARFTypeUnit *TU = U->getContext().getTypeUnitForHash(
              U->getVersion(), *Sig, U->isDWOUnit()))
        return TU->getDIEForOffset(TU->getTypeOffset() + TU->getOffset());
    }
  }
  return *this;
}

Optional<uint64_t> DWARFDie::getRangesBaseAttribute() const {
  return toSectionOffset(find({DW_AT_rnglists_base, DW_AT_GNU_ranges_base}));
}

Optional<uint64_t> DWARFDie::getLocBaseAttribute() const {
  return toSectionOffset(find(DW_AT_loclists_base));
}

Optional<uint64_t> DWARFDie::getHighPC(uint64_t LowPC) const {
  uint64_t Tombstone = dwarf::computeTombstoneAddress(U->getAddressByteSize());
  if (LowPC == Tombstone)
    return None;
  if (auto FormValue = find(DW_AT_high_pc)) {
    if (auto Address = FormValue->getAsAddress()) {
      // High PC is an address.
      return Address;
    }
    if (auto Offset = FormValue->getAsUnsignedConstant()) {
      // High PC is an offset from LowPC.
      return LowPC + *Offset;
    }
  }
  return None;
}

bool DWARFDie::getLowAndHighPC(uint64_t &LowPC, uint64_t &HighPC,
                               uint64_t &SectionIndex) const {
  auto F = find(DW_AT_low_pc);
  auto LowPcAddr = toSectionedAddress(F);
  if (!LowPcAddr)
    return false;
  if (auto HighPcAddr = getHighPC(LowPcAddr->Address)) {
    LowPC = LowPcAddr->Address;
    HighPC = *HighPcAddr;
    SectionIndex = LowPcAddr->SectionIndex;
    return true;
  }
  return false;
}

Expected<DWARFAddressRangesVector> DWARFDie::getAddressRanges() const {
  if (isNULL())
    return DWARFAddressRangesVector();
  // Single range specified by low/high PC.
  uint64_t LowPC, HighPC, Index;
  if (getLowAndHighPC(LowPC, HighPC, Index))
    return DWARFAddressRangesVector{{LowPC, HighPC, Index}};

  Optional<DWARFFormValue> Value = find(DW_AT_ranges);
  if (Value) {
    if (Value->getForm() == DW_FORM_rnglistx)
      return U->findRnglistFromIndex(*Value->getAsSectionOffset());
    return U->findRnglistFromOffset(*Value->getAsSectionOffset());
  }
  return DWARFAddressRangesVector();
}

bool DWARFDie::addressRangeContainsAddress(const uint64_t Address) const {
  auto RangesOrError = getAddressRanges();
  if (!RangesOrError) {
    llvm::consumeError(RangesOrError.takeError());
    return false;
  }

  for (const auto &R : RangesOrError.get())
    if (R.LowPC <= Address && Address < R.HighPC)
      return true;
  return false;
}

Expected<DWARFLocationExpressionsVector>
DWARFDie::getLocations(dwarf::Attribute Attr) const {
  Optional<DWARFFormValue> Location = find(Attr);
  if (!Location)
    return createStringError(inconvertibleErrorCode(), "No %s",
                             dwarf::AttributeString(Attr).data());

  if (Optional<uint64_t> Off = Location->getAsSectionOffset()) {
    uint64_t Offset = *Off;

    if (Location->getForm() == DW_FORM_loclistx) {
      if (auto LoclistOffset = U->getLoclistOffset(Offset))
        Offset = *LoclistOffset;
      else
        return createStringError(inconvertibleErrorCode(),
                                 "Loclist table not found");
    }
    return U->findLoclistFromOffset(Offset);
  }

  if (Optional<ArrayRef<uint8_t>> Expr = Location->getAsBlock()) {
    return DWARFLocationExpressionsVector{
        DWARFLocationExpression{None, to_vector<4>(*Expr)}};
  }

  return createStringError(
      inconvertibleErrorCode(), "Unsupported %s encoding: %s",
      dwarf::AttributeString(Attr).data(),
      dwarf::FormEncodingString(Location->getForm()).data());
}

const char *DWARFDie::getSubroutineName(DINameKind Kind) const {
  if (!isSubroutineDIE())
    return nullptr;
  return getName(Kind);
}

const char *DWARFDie::getName(DINameKind Kind) const {
  if (!isValid() || Kind == DINameKind::None)
    return nullptr;
  // Try to get mangled name only if it was asked for.
  if (Kind == DINameKind::LinkageName) {
    if (auto Name = getLinkageName())
      return Name;
  }
  return getShortName();
}

const char *DWARFDie::getShortName() const {
  if (!isValid())
    return nullptr;

  return dwarf::toString(findRecursively(dwarf::DW_AT_name), nullptr);
}

const char *DWARFDie::getLinkageName() const {
  if (!isValid())
    return nullptr;

  return dwarf::toString(findRecursively({dwarf::DW_AT_MIPS_linkage_name,
                                          dwarf::DW_AT_linkage_name}),
                         nullptr);
}

uint64_t DWARFDie::getDeclLine() const {
  return toUnsigned(findRecursively(DW_AT_decl_line), 0);
}

std::string
DWARFDie::getDeclFile(DILineInfoSpecifier::FileLineInfoKind Kind) const {
  if (auto FormValue = findRecursively(DW_AT_decl_file))
    if (auto OptString = FormValue->getAsFile(Kind))
      return *OptString;
  return {};
}

void DWARFDie::getCallerFrame(uint32_t &CallFile, uint32_t &CallLine,
                              uint32_t &CallColumn,
                              uint32_t &CallDiscriminator) const {
  CallFile = toUnsigned(find(DW_AT_call_file), 0);
  CallLine = toUnsigned(find(DW_AT_call_line), 0);
  CallColumn = toUnsigned(find(DW_AT_call_column), 0);
  CallDiscriminator = toUnsigned(find(DW_AT_GNU_discriminator), 0);
}

/// Helper to dump a DIE with all of its parents, but no siblings.
static unsigned dumpParentChain(DWARFDie Die, raw_ostream &OS, unsigned Indent,
                                DIDumpOptions DumpOpts, unsigned Depth = 0) {
  if (!Die)
    return Indent;
  if (DumpOpts.ParentRecurseDepth > 0 && Depth >= DumpOpts.ParentRecurseDepth)
    return Indent;
  Indent = dumpParentChain(Die.getParent(), OS, Indent, DumpOpts, Depth + 1);
  Die.dump(OS, Indent, DumpOpts);
  return Indent + 2;
}

void DWARFDie::dump(raw_ostream &OS, unsigned Indent,
                    DIDumpOptions DumpOpts) const {
  if (!isValid())
    return;
  DWARFDataExtractor debug_info_data = U->getDebugInfoExtractor();
  const uint64_t Offset = getOffset();
  uint64_t offset = Offset;
  if (DumpOpts.ShowParents) {
    DIDumpOptions ParentDumpOpts = DumpOpts;
    ParentDumpOpts.ShowParents = false;
    ParentDumpOpts.ShowChildren = false;
    Indent = dumpParentChain(getParent(), OS, Indent, ParentDumpOpts);
  }

  if (debug_info_data.isValidOffset(offset)) {
    uint32_t abbrCode = debug_info_data.getULEB128(&offset);
    if (DumpOpts.ShowAddresses)
      WithColor(OS, HighlightColor::Address).get()
          << format("\n0x%8.8" PRIx64 ": ", Offset);

    if (abbrCode) {
      auto AbbrevDecl = getAbbreviationDeclarationPtr();
      if (AbbrevDecl) {
        WithColor(OS, HighlightColor::Tag).get().indent(Indent)
            << formatv("{0}", getTag());
        if (DumpOpts.Verbose) {
          OS << format(" [%u] %c", abbrCode,
                       AbbrevDecl->hasChildren() ? '*' : ' ');
          if (Optional<uint32_t> ParentIdx = Die->getParentIdx())
            OS << format(" (0x%8.8" PRIx64 ")",
                         U->getDIEAtIndex(*ParentIdx).getOffset());
        }
        OS << '\n';

        // Dump all data in the DIE for the attributes.
        for (const DWARFAttribute &AttrValue : attributes())
          dumpAttribute(OS, *this, AttrValue, Indent, DumpOpts);

        if (DumpOpts.ShowChildren && DumpOpts.ChildRecurseDepth > 0) {
          DWARFDie Child = getFirstChild();
          DumpOpts.ChildRecurseDepth--;
          DIDumpOptions ChildDumpOpts = DumpOpts;
          ChildDumpOpts.ShowParents = false;
          while (Child) {
            Child.dump(OS, Indent + 2, ChildDumpOpts);
            Child = Child.getSibling();
          }
        }
      } else {
        OS << "Abbreviation code not found in 'debug_abbrev' class for code: "
           << abbrCode << '\n';
      }
    } else {
      OS.indent(Indent) << "NULL\n";
    }
  }
}

LLVM_DUMP_METHOD void DWARFDie::dump() const { dump(llvm::errs(), 0); }

DWARFDie DWARFDie::getParent() const {
  if (isValid())
    return U->getParent(Die);
  return DWARFDie();
}

DWARFDie DWARFDie::getSibling() const {
  if (isValid())
    return U->getSibling(Die);
  return DWARFDie();
}

DWARFDie DWARFDie::getPreviousSibling() const {
  if (isValid())
    return U->getPreviousSibling(Die);
  return DWARFDie();
}

DWARFDie DWARFDie::getFirstChild() const {
  if (isValid())
    return U->getFirstChild(Die);
  return DWARFDie();
}

DWARFDie DWARFDie::getLastChild() const {
  if (isValid())
    return U->getLastChild(Die);
  return DWARFDie();
}

iterator_range<DWARFDie::attribute_iterator> DWARFDie::attributes() const {
  return make_range(attribute_iterator(*this, false),
                    attribute_iterator(*this, true));
}

DWARFDie::attribute_iterator::attribute_iterator(DWARFDie D, bool End)
    : Die(D), Index(0) {
  auto AbbrDecl = Die.getAbbreviationDeclarationPtr();
  assert(AbbrDecl && "Must have abbreviation declaration");
  if (End) {
    // This is the end iterator so we set the index to the attribute count.
    Index = AbbrDecl->getNumAttributes();
  } else {
    // This is the begin iterator so we extract the value for this->Index.
    AttrValue.Offset = D.getOffset() + AbbrDecl->getCodeByteSize();
    updateForIndex(*AbbrDecl, 0);
  }
}

void DWARFDie::attribute_iterator::updateForIndex(
    const DWARFAbbreviationDeclaration &AbbrDecl, uint32_t I) {
  Index = I;
  // AbbrDecl must be valid before calling this function.
  auto NumAttrs = AbbrDecl.getNumAttributes();
  if (Index < NumAttrs) {
    AttrValue.Attr = AbbrDecl.getAttrByIndex(Index);
    // Add the previous byte size of any previous attribute value.
    AttrValue.Offset += AttrValue.ByteSize;
    uint64_t ParseOffset = AttrValue.Offset;
    if (AbbrDecl.getAttrIsImplicitConstByIndex(Index))
      AttrValue.Value = DWARFFormValue::createFromSValue(
          AbbrDecl.getFormByIndex(Index),
          AbbrDecl.getAttrImplicitConstValueByIndex(Index));
    else {
      auto U = Die.getDwarfUnit();
      assert(U && "Die must have valid DWARF unit");
      AttrValue.Value = DWARFFormValue::createFromUnit(
          AbbrDecl.getFormByIndex(Index), U, &ParseOffset);
    }
    AttrValue.ByteSize = ParseOffset - AttrValue.Offset;
  } else {
    assert(Index == NumAttrs && "Indexes should be [0, NumAttrs) only");
    AttrValue = {};
  }
}

DWARFDie::attribute_iterator &DWARFDie::attribute_iterator::operator++() {
  if (auto AbbrDecl = Die.getAbbreviationDeclarationPtr())
    updateForIndex(*AbbrDecl, Index + 1);
  return *this;
}

bool DWARFAttribute::mayHaveLocationList(dwarf::Attribute Attr) {
  switch(Attr) {
  case DW_AT_location:
  case DW_AT_string_length:
  case DW_AT_return_addr:
  case DW_AT_data_member_location:
  case DW_AT_frame_base:
  case DW_AT_static_link:
  case DW_AT_segment:
  case DW_AT_use_location:
  case DW_AT_vtable_elem_location:
    return true;
  default:
    return false;
  }
}

bool DWARFAttribute::mayHaveLocationExpr(dwarf::Attribute Attr) {
  switch (Attr) {
  // From the DWARF v5 specification.
  case DW_AT_location:
  case DW_AT_byte_size:
  case DW_AT_bit_offset:
  case DW_AT_bit_size:
  case DW_AT_string_length:
  case DW_AT_lower_bound:
  case DW_AT_return_addr:
  case DW_AT_bit_stride:
  case DW_AT_upper_bound:
  case DW_AT_count:
  case DW_AT_data_member_location:
  case DW_AT_frame_base:
  case DW_AT_segment:
  case DW_AT_static_link:
  case DW_AT_use_location:
  case DW_AT_vtable_elem_location:
  case DW_AT_allocated:
  case DW_AT_associated:
  case DW_AT_data_location:
  case DW_AT_byte_stride:
  case DW_AT_rank:
  case DW_AT_call_value:
  case DW_AT_call_origin:
  case DW_AT_call_target:
  case DW_AT_call_target_clobbered:
  case DW_AT_call_data_location:
  case DW_AT_call_data_value:
  // Extensions.
  case DW_AT_GNU_call_site_value:
  case DW_AT_GNU_call_site_target:
    return true;
  default:
    return false;
  }
}

namespace llvm {

void dumpTypeQualifiedName(const DWARFDie &DIE, raw_ostream &OS) {
  DWARFTypePrinter(OS).appendQualifiedName(DIE);
}

void dumpTypeUnqualifiedName(const DWARFDie &DIE, raw_ostream &OS,
                             std::string *OriginalFullName) {
  DWARFTypePrinter(OS).appendUnqualifiedName(DIE, OriginalFullName);
}

} // namespace llvm
