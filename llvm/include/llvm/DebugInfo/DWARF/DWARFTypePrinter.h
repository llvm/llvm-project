//===- DWARFTypePrinter.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFTYPEPRINTER_H
#define LLVM_DEBUGINFO_DWARF_DWARFTYPEPRINTER_H

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Error.h"

#include <string>

namespace llvm {

class raw_ostream;

// FIXME: We should have pretty printers per language. Currently we print
// everything as if it was C++ and fall back to the TAG type name.
template <typename DieType> struct DWARFTypePrinter {
  raw_ostream &OS;
  bool Word = true;
  bool EndedWithTemplate = false;

  DWARFTypePrinter(raw_ostream &OS) : OS(OS) {}

  /// Dump the name encoded in the type tag.
  void appendTypeTagName(dwarf::Tag T);

  void appendArrayType(const DieType &D);

  DieType skipQualifiers(DieType D);

  bool needsParens(DieType D);

  void appendPointerLikeTypeBefore(DieType D, DieType Inner, StringRef Ptr);

  DieType appendUnqualifiedNameBefore(DieType D,
                                      std::string *OriginalFullName = nullptr);

  void appendUnqualifiedNameAfter(DieType D, DieType Inner,
                                  bool SkipFirstParamIfArtificial = false);
  void appendQualifiedName(DieType D);
  DieType appendQualifiedNameBefore(DieType D);
  bool appendTemplateParameters(DieType D, bool *FirstParameter = nullptr);
  void appendAndTerminateTemplateParameters(DieType D);
  void decomposeConstVolatile(DieType &N, DieType &T, DieType &C, DieType &V);
  void appendConstVolatileQualifierAfter(DieType N);
  void appendConstVolatileQualifierBefore(DieType N);

  /// Recursively append the DIE type name when applicable.
  void appendUnqualifiedName(DieType D,
                             std::string *OriginalFullName = nullptr);

  void appendSubroutineNameAfter(DieType D, DieType Inner,
                                 bool SkipFirstParamIfArtificial, bool Const,
                                 bool Volatile);
  void appendScopes(DieType D);

private:
  /// Returns True if the DIE TAG is one of the ones that is scopped.
  static inline bool scopedTAGs(dwarf::Tag Tag) {
    switch (Tag) {
    case dwarf::DW_TAG_structure_type:
    case dwarf::DW_TAG_class_type:
    case dwarf::DW_TAG_union_type:
    case dwarf::DW_TAG_namespace:
    case dwarf::DW_TAG_enumeration_type:
    case dwarf::DW_TAG_typedef:
      return true;
    default:
      break;
    }
    return false;
  }
};

template <typename DieType>
void DWARFTypePrinter<DieType>::appendTypeTagName(dwarf::Tag T) {
  StringRef TagStr = TagString(T);
  static constexpr StringRef Prefix = "DW_TAG_";
  static constexpr StringRef Suffix = "_type";
  if (!TagStr.starts_with(Prefix) || !TagStr.ends_with(Suffix))
    return;
  OS << TagStr.substr(Prefix.size(),
                      TagStr.size() - (Prefix.size() + Suffix.size()))
     << " ";
}

template <typename DieType>
void DWARFTypePrinter<DieType>::appendArrayType(const DieType &D) {
  for (const DieType &C : D.children()) {
    if (C.getTag() != dwarf::DW_TAG_subrange_type)
      continue;
    std::optional<uint64_t> LB;
    std::optional<uint64_t> Count;
    std::optional<uint64_t> UB;
    std::optional<unsigned> DefaultLB;
    if (std::optional<typename DieType::DWARFFormValue> L =
            C.find(dwarf::DW_AT_lower_bound))
      LB = L->getAsUnsignedConstant();
    if (std::optional<typename DieType::DWARFFormValue> CountV =
            C.find(dwarf::DW_AT_count))
      Count = CountV->getAsUnsignedConstant();
    if (std::optional<typename DieType::DWARFFormValue> UpperV =
            C.find(dwarf::DW_AT_upper_bound))
      UB = UpperV->getAsUnsignedConstant();
    if (std::optional<uint64_t> LV = D.getLanguage())
      if ((DefaultLB =
               LanguageLowerBound(static_cast<dwarf::SourceLanguage>(*LV))))
        if (LB && *LB == *DefaultLB)
          LB = std::nullopt;
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

namespace detail {
template <typename DieType>
DieType resolveReferencedType(DieType D,
                              dwarf::Attribute Attr = dwarf::DW_AT_type) {
  return D.resolveReferencedType(Attr);
}
template <typename DieType>
DieType resolveReferencedType(DieType D, typename DieType::DWARFFormValue F) {
  return D.resolveReferencedType(F);
}
template <typename DWARFFormValueType>
const char *toString(std::optional<DWARFFormValueType> F) {
  if (F) {
    llvm::Expected<const char *> E = F->getAsCString();
    if (E)
      return *E;
    llvm::consumeError(E.takeError());
  }
  return nullptr;
}
} // namespace detail

template <typename DieType>
DieType DWARFTypePrinter<DieType>::skipQualifiers(DieType D) {
  while (D && (D.getTag() == dwarf::DW_TAG_const_type ||
               D.getTag() == dwarf::DW_TAG_volatile_type))
    D = detail::resolveReferencedType(D);
  return D;
}

template <typename DieType>
bool DWARFTypePrinter<DieType>::needsParens(DieType D) {
  D = skipQualifiers(D);
  return D && (D.getTag() == dwarf::DW_TAG_subroutine_type ||
               D.getTag() == dwarf::DW_TAG_array_type);
}

template <typename DieType>
void DWARFTypePrinter<DieType>::appendPointerLikeTypeBefore(DieType D,
                                                            DieType Inner,
                                                            StringRef Ptr) {
  appendQualifiedNameBefore(Inner);
  if (Word)
    OS << ' ';
  if (needsParens(Inner))
    OS << '(';
  OS << Ptr;
  Word = false;
  EndedWithTemplate = false;
}

template <typename DieType>
DieType DWARFTypePrinter<DieType>::appendUnqualifiedNameBefore(
    DieType D, std::string *OriginalFullName) {
  Word = true;
  if (!D) {
    OS << "void";
    return DieType();
  }
  DieType InnerDIE;
  auto Inner = [&] { return InnerDIE = detail::resolveReferencedType(D); };
  const dwarf::Tag T = D.getTag();
  switch (T) {
  case dwarf::DW_TAG_pointer_type: {
    appendPointerLikeTypeBefore(D, Inner(), "*");
    break;
  }
  case dwarf::DW_TAG_subroutine_type: {
    appendQualifiedNameBefore(Inner());
    if (Word) {
      OS << ' ';
    }
    Word = false;
    break;
  }
  case dwarf::DW_TAG_array_type: {
    appendQualifiedNameBefore(Inner());
    break;
  }
  case dwarf::DW_TAG_reference_type:
    appendPointerLikeTypeBefore(D, Inner(), "&");
    break;
  case dwarf::DW_TAG_rvalue_reference_type:
    appendPointerLikeTypeBefore(D, Inner(), "&&");
    break;
  case dwarf::DW_TAG_ptr_to_member_type: {
    appendQualifiedNameBefore(Inner());
    if (needsParens(InnerDIE))
      OS << '(';
    else if (Word)
      OS << ' ';
    if (DieType Cont =
            detail::resolveReferencedType(D, dwarf::DW_AT_containing_type)) {
      appendQualifiedName(Cont);
      EndedWithTemplate = false;
      OS << "::";
    }
    OS << "*";
    Word = false;
    break;
  }
  case dwarf::DW_TAG_LLVM_ptrauth_type:
    appendQualifiedNameBefore(Inner());
    break;
  case dwarf::DW_TAG_const_type:
  case dwarf::DW_TAG_volatile_type:
    appendConstVolatileQualifierBefore(D);
    break;
  case dwarf::DW_TAG_namespace: {
    if (const char *Name = detail::toString(D.find(dwarf::DW_AT_name)))
      OS << Name;
    else
      OS << "(anonymous namespace)";
    break;
  }
  case dwarf::DW_TAG_unspecified_type: {
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
    const char *NamePtr = detail::toString(D.find(dwarf::DW_AT_name));
    if (!NamePtr) {
      appendTypeTagName(D.getTag());
      return DieType();
    }
    Word = true;
    StringRef Name = NamePtr;
    static constexpr StringRef MangledPrefix = "_STN|";
    if (Name.consume_front(MangledPrefix)) {
      auto Separator = Name.find('|');
      assert(Separator != StringRef::npos);
      StringRef BaseName = Name.substr(0, Separator);
      StringRef TemplateArgs = Name.substr(Separator + 1);
      if (OriginalFullName)
        *OriginalFullName = (BaseName + TemplateArgs).str();
      Name = BaseName;
    } else
      EndedWithTemplate = Name.ends_with(">");
    OS << Name;
    // This check would be insufficient for operator overloads like
    // "operator>>" - but for now Clang doesn't try to simplify them, so this
    // is OK. Add more nuanced operator overload handling here if/when needed.
    if (Name.ends_with(">"))
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

template <typename DieType>
void DWARFTypePrinter<DieType>::appendUnqualifiedNameAfter(
    DieType D, DieType Inner, bool SkipFirstParamIfArtificial) {
  if (!D)
    return;
  switch (D.getTag()) {
  case dwarf::DW_TAG_subroutine_type: {
    appendSubroutineNameAfter(D, Inner, SkipFirstParamIfArtificial, false,
                              false);
    break;
  }
  case dwarf::DW_TAG_array_type: {
    appendArrayType(D);
    break;
  }
  case dwarf::DW_TAG_const_type:
  case dwarf::DW_TAG_volatile_type:
    appendConstVolatileQualifierAfter(D);
    break;
  case dwarf::DW_TAG_ptr_to_member_type:
  case dwarf::DW_TAG_reference_type:
  case dwarf::DW_TAG_rvalue_reference_type:
  case dwarf::DW_TAG_pointer_type: {
    if (needsParens(Inner))
      OS << ')';
    appendUnqualifiedNameAfter(Inner, detail::resolveReferencedType(Inner),
                               /*SkipFirstParamIfArtificial=*/D.getTag() ==
                                   dwarf::DW_TAG_ptr_to_member_type);
    break;
  }
  case dwarf::DW_TAG_LLVM_ptrauth_type: {
    auto getValOrNull = [&](dwarf::Attribute Attr) -> uint64_t {
      if (auto Form = D.find(Attr))
        return *Form->getAsUnsignedConstant();
      return 0;
    };
    SmallVector<const char *, 2> optionsVec;
    if (getValOrNull(dwarf::DW_AT_LLVM_ptrauth_isa_pointer))
      optionsVec.push_back("isa-pointer");
    if (getValOrNull(dwarf::DW_AT_LLVM_ptrauth_authenticates_null_values))
      optionsVec.push_back("authenticates-null-values");
    if (auto AuthenticationMode =
            D.find(dwarf::DW_AT_LLVM_ptrauth_authentication_mode)) {
      switch (*AuthenticationMode->getAsUnsignedConstant()) {
      case 0:
      case 1:
        optionsVec.push_back("strip");
        break;
      case 2:
        optionsVec.push_back("sign-and-strip");
        break;
      default:
        // Default authentication policy
        break;
      }
    }
    std::string options;
    for (const auto *option : optionsVec) {
      if (options.size())
        options += ",";
      options += option;
    }
    if (options.size())
      options = ", \"" + options + "\"";
    std::string PtrauthString;
    llvm::raw_string_ostream PtrauthStream(PtrauthString);
    PtrauthStream
        << "__ptrauth(" << getValOrNull(dwarf::DW_AT_LLVM_ptrauth_key) << ", "
        << getValOrNull(dwarf::DW_AT_LLVM_ptrauth_address_discriminated)
        << ", 0x0"
        << utohexstr(
               getValOrNull(dwarf::DW_AT_LLVM_ptrauth_extra_discriminator),
               true)
        << options << ")";
    OS << PtrauthStream.str();
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

template <typename DieType>
void DWARFTypePrinter<DieType>::appendQualifiedName(DieType D) {
  if (D && scopedTAGs(D.getTag()))
    appendScopes(D.getParent());
  appendUnqualifiedName(D);
}

template <typename DieType>
DieType DWARFTypePrinter<DieType>::appendQualifiedNameBefore(DieType D) {
  if (D && scopedTAGs(D.getTag()))
    appendScopes(D.getParent());
  return appendUnqualifiedNameBefore(D);
}

template <typename DieType>
bool DWARFTypePrinter<DieType>::appendTemplateParameters(DieType D,
                                                         bool *FirstParameter) {
  bool FirstParameterValue = true;
  bool IsTemplate = false;
  if (!FirstParameter)
    FirstParameter = &FirstParameterValue;
  for (const DieType &C : D) {
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
      DieType T = detail::resolveReferencedType(C);
      Sep();
      if (T.getTag() == dwarf::DW_TAG_enumeration_type) {
        OS << '(';
        appendQualifiedName(T);
        OS << ')';
        auto V = C.find(dwarf::DW_AT_const_value);
        OS << std::to_string(*V->getAsSignedConstant());
        continue;
      }
      // /Maybe/ we could do pointer/reference type parameters, looking for the
      // symbol in the ELF symbol table to get back to the variable...
      // but probably not worth it.
      if (T.getTag() == dwarf::DW_TAG_pointer_type ||
          T.getTag() == dwarf::DW_TAG_reference_type)
        continue;
      const char *RawName = detail::toString(T.find(dwarf::DW_AT_name));
      assert(RawName);
      StringRef Name = RawName;
      auto V = C.find(dwarf::DW_AT_const_value);
      bool IsQualifiedChar = false;
      if (Name == "bool") {
        OS << (*V->getAsUnsignedConstant() ? "true" : "false");
      } else if (Name == "short") {
        OS << "(short)";
        OS << std::to_string(*V->getAsSignedConstant());
      } else if (Name == "unsigned short") {
        OS << "(unsigned short)";
        OS << std::to_string(*V->getAsSignedConstant());
      } else if (Name == "int")
        OS << std::to_string(*V->getAsSignedConstant());
      else if (Name == "long") {
        OS << std::to_string(*V->getAsSignedConstant());
        OS << "L";
      } else if (Name == "long long") {
        OS << std::to_string(*V->getAsSignedConstant());
        OS << "LL";
      } else if (Name == "unsigned int") {
        OS << std::to_string(*V->getAsUnsignedConstant());
        OS << "U";
      } else if (Name == "unsigned long") {
        OS << std::to_string(*V->getAsUnsignedConstant());
        OS << "UL";
      } else if (Name == "unsigned long long") {
        OS << std::to_string(*V->getAsUnsignedConstant());
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
            OS << llvm::format("'\\x%02" PRIx64 "'", Val);
          else if (Val <= 0xFFFF)
            OS << llvm::format("'\\u%04" PRIx64 "'", Val);
          else
            OS << llvm::format("'\\U%08" PRIx64 "'", Val);
        }
      }
      continue;
    }
    if (C.getTag() == dwarf::DW_TAG_GNU_template_template_param) {
      const char *RawName =
          detail::toString(C.find(dwarf::DW_AT_GNU_template_name));
      assert(RawName);
      StringRef Name = RawName;
      Sep();
      OS << Name;
      continue;
    }
    if (C.getTag() != dwarf::DW_TAG_template_type_parameter)
      continue;
    auto TypeAttr = C.find(dwarf::DW_AT_type);
    Sep();
    appendQualifiedName(TypeAttr ? detail::resolveReferencedType(C, *TypeAttr)
                                 : DieType());
  }
  if (IsTemplate && *FirstParameter && FirstParameter == &FirstParameterValue) {
    OS << '<';
    EndedWithTemplate = false;
  }
  return IsTemplate;
}

template <typename DieType>
void DWARFTypePrinter<DieType>::appendAndTerminateTemplateParameters(
    DieType D) {
  bool R = appendTemplateParameters(D);
  if (!R)
    return;

  if (EndedWithTemplate)
    OS << " ";
  OS << ">";
  EndedWithTemplate = true;
  Word = true;
}

template <typename DieType>
void DWARFTypePrinter<DieType>::decomposeConstVolatile(DieType &N, DieType &T,
                                                       DieType &C, DieType &V) {
  (N.getTag() == dwarf::DW_TAG_const_type ? C : V) = N;
  T = detail::resolveReferencedType(N);
  if (T) {
    auto Tag = T.getTag();
    if (Tag == dwarf::DW_TAG_const_type) {
      C = T;
      T = detail::resolveReferencedType(T);
    } else if (Tag == dwarf::DW_TAG_volatile_type) {
      V = T;
      T = detail::resolveReferencedType(T);
    }
  }
}

template <typename DieType>
void DWARFTypePrinter<DieType>::appendConstVolatileQualifierAfter(DieType N) {
  DieType C;
  DieType V;
  DieType T;
  decomposeConstVolatile(N, T, C, V);
  if (T && T.getTag() == dwarf::DW_TAG_subroutine_type)
    appendSubroutineNameAfter(T, detail::resolveReferencedType(T), false,
                              static_cast<bool>(C), static_cast<bool>(V));
  else
    appendUnqualifiedNameAfter(T, detail::resolveReferencedType(T));
}

template <typename DieType>
void DWARFTypePrinter<DieType>::appendConstVolatileQualifierBefore(DieType N) {
  DieType C;
  DieType V;
  DieType T;
  decomposeConstVolatile(N, T, C, V);
  bool Subroutine = T && T.getTag() == dwarf::DW_TAG_subroutine_type;
  DieType A = T;
  while (A && A.getTag() == dwarf::DW_TAG_array_type)
    A = detail::resolveReferencedType(A);
  bool Leading =
      (!A || (A.getTag() != dwarf::DW_TAG_pointer_type &&
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

template <typename DieType>
void DWARFTypePrinter<DieType>::appendUnqualifiedName(
    DieType D, std::string *OriginalFullName) {
  // FIXME: We should have pretty printers per language. Currently we print
  // everything as if it was C++ and fall back to the TAG type name.
  DieType Inner = appendUnqualifiedNameBefore(D, OriginalFullName);
  appendUnqualifiedNameAfter(D, Inner);
}

template <typename DieType>
void DWARFTypePrinter<DieType>::appendSubroutineNameAfter(
    DieType D, DieType Inner, bool SkipFirstParamIfArtificial, bool Const,
    bool Volatile) {
  DieType FirstParamIfArtificial;
  OS << '(';
  EndedWithTemplate = false;
  bool First = true;
  bool RealFirst = true;
  for (DieType P : D) {
    if (P.getTag() != dwarf::DW_TAG_formal_parameter &&
        P.getTag() != dwarf::DW_TAG_unspecified_parameters)
      return;
    DieType T = detail::resolveReferencedType(P);
    if (SkipFirstParamIfArtificial && RealFirst &&
        P.find(dwarf::DW_AT_artificial)) {
      FirstParamIfArtificial = T;
      RealFirst = false;
      continue;
    }
    if (!First) {
      OS << ", ";
    }
    First = false;
    if (P.getTag() == dwarf::DW_TAG_unspecified_parameters)
      OS << "...";
    else
      appendQualifiedName(T);
  }
  EndedWithTemplate = false;
  OS << ')';
  if (FirstParamIfArtificial) {
    if (DieType P = FirstParamIfArtificial) {
      if (P.getTag() == dwarf::DW_TAG_pointer_type) {
        auto CVStep = [&](DieType CV) {
          if (DieType U = detail::resolveReferencedType(CV)) {
            Const |= U.getTag() == dwarf::DW_TAG_const_type;
            Volatile |= U.getTag() == dwarf::DW_TAG_volatile_type;
            return U;
          }
          return DieType();
        };
        if (DieType CV = CVStep(P)) {
          CVStep(CV);
        }
      }
    }
  }

  if (auto CC = D.find(dwarf::DW_AT_calling_convention)) {
    switch (*CC->getAsUnsignedConstant()) {
    case dwarf::CallingConvention::DW_CC_BORLAND_stdcall:
      OS << " __attribute__((stdcall))";
      break;
    case dwarf::CallingConvention::DW_CC_BORLAND_msfastcall:
      OS << " __attribute__((fastcall))";
      break;
    case dwarf::CallingConvention::DW_CC_BORLAND_thiscall:
      OS << " __attribute__((thiscall))";
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_vectorcall:
      OS << " __attribute__((vectorcall))";
      break;
    case dwarf::CallingConvention::DW_CC_BORLAND_pascal:
      OS << " __attribute__((pascal))";
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_Win64:
      OS << " __attribute__((ms_abi))";
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_X86_64SysV:
      OS << " __attribute__((sysv_abi))";
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_AAPCS:
      // AArch64VectorCall missing?
      OS << " __attribute__((pcs(\"aapcs\")))";
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_AAPCS_VFP:
      OS << " __attribute__((pcs(\"aapcs-vfp\")))";
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_IntelOclBicc:
      OS << " __attribute__((intel_ocl_bicc))";
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_SpirFunction:
    case dwarf::CallingConvention::DW_CC_LLVM_OpenCLKernel:
      // These aren't available as attributes, but maybe we should still
      // render them somehow? (Clang doesn't render them, but that's an issue
      // for template names too - since then the DWARF names of templates
      // instantiated with function types with these calling conventions won't
      // have distinct names - so we'd need to fix that too)
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_Swift:
      // SwiftAsync missing
      OS << " __attribute__((swiftcall))";
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_PreserveMost:
      OS << " __attribute__((preserve_most))";
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_PreserveAll:
      OS << " __attribute__((preserve_all))";
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_PreserveNone:
      OS << " __attribute__((preserve_none))";
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_X86RegCall:
      OS << " __attribute__((regcall))";
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_M68kRTD:
      OS << " __attribute__((m68k_rtd))";
      break;
    }
  }

  if (Const)
    OS << " const";
  if (Volatile)
    OS << " volatile";
  if (D.find(dwarf::DW_AT_reference))
    OS << " &";
  if (D.find(dwarf::DW_AT_rvalue_reference))
    OS << " &&";

  appendUnqualifiedNameAfter(Inner, detail::resolveReferencedType(Inner));
}

template <typename DieType>
void DWARFTypePrinter<DieType>::appendScopes(DieType D) {
  if (D.getTag() == dwarf::DW_TAG_compile_unit)
    return;
  if (D.getTag() == dwarf::DW_TAG_type_unit)
    return;
  if (D.getTag() == dwarf::DW_TAG_skeleton_unit)
    return;
  if (D.getTag() == dwarf::DW_TAG_subprogram)
    return;
  if (D.getTag() == dwarf::DW_TAG_lexical_block)
    return;
  D = D.resolveTypeUnitReference();
  if (DieType P = D.getParent())
    appendScopes(P);
  appendUnqualifiedName(D);
  OS << "::";
}
} // namespace llvm

#endif // LLVM_DEBUGINFO_DWARF_DWARFTYPEPRINTER_H
