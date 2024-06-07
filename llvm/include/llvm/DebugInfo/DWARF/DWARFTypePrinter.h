//===- DWARFTypePrinter.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARF_DWARFTYPEPRINTER_H
#define LLVM_DEBUGINFO_DWARF_DWARFTYPEPRINTER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/ADT/SmallString.h"

#include <string>

namespace llvm {

class raw_ostream;

inline std::optional<uint64_t> getLanguage(DWARFDie D) {
  if (std::optional<DWARFFormValue> LV =
          D.getDwarfUnit()->getUnitDIE().find(dwarf::DW_AT_language))
    return LV->getAsUnsignedConstant();
  return std::nullopt;
}

namespace detail {
struct DefaultVisitor {
  raw_ostream &OS;
  DefaultVisitor(raw_ostream &OS) : OS(OS) { }
  bool operator()(StringRef S) const {
    OS << S;
    return true;
  }
};

inline llvm::SmallString<128> toString(const llvm::format_object_base &Fmt) {
  size_t NextBufferSize = 127;
  llvm::SmallString<128> V;

  while (true) {
    V.resize_for_overwrite(NextBufferSize);

    // Try formatting into the SmallVector.
    size_t BytesUsed = Fmt.print(V.data(), NextBufferSize);

    // If BytesUsed fit into the vector, we win.
    if (BytesUsed <= NextBufferSize) {
      V.resize(BytesUsed);
      return V;
    }

    // Otherwise, try again with a new size.
    assert(BytesUsed > NextBufferSize && "Didn't grow buffer!?");
    NextBufferSize = BytesUsed;
  }
}
}

#define LLVM_QUICK_EXIT(expr) if (!(expr)) return false
#define LLVM_QUICK_EXIT_NONE(expr) if (!(expr)) return std::nullopt

// FIXME: We should have pretty printers per language. Currently we print
// everything as if it was C++ and fall back to the TAG type name.
template <typename DieType, typename Visitor = detail::DefaultVisitor>
struct DWARFTypePrinter {
  Visitor V;
  bool Word = true;
  bool EndedWithTemplate = false;

  DWARFTypePrinter(raw_ostream &OS) : V(OS) {
  }
  template<typename T>
  DWARFTypePrinter(T &&V) : V(std::forward<T>(V)) {}

  /// Dump the name encoded in the type tag.
  bool appendTypeTagName(dwarf::Tag T);

  bool appendArrayType(const DieType &D);

  DieType skipQualifiers(DieType D);

  bool needsParens(DieType D);

  bool appendPointerLikeTypeBefore(DieType D, DieType Inner, StringRef Ptr);

  std::optional<DieType>
  appendUnqualifiedNameBefore(DieType D,
                              std::string *OriginalFullName = nullptr);

  bool appendUnqualifiedNameAfter(DieType D, DieType Inner,
                                  bool SkipFirstParamIfArtificial = false);
  bool appendQualifiedName(DieType D);
  std::optional<DieType> appendQualifiedNameBefore(DieType D);
  std::optional<bool> appendTemplateParameters(DieType D, bool *FirstParameter = nullptr);
  bool appendAndTerminateTemplateParameters(DieType D);
  void decomposeConstVolatile(DieType &N, DieType &T, DieType &C, DieType &V);
  bool appendConstVolatileQualifierAfter(DieType N);
  bool appendConstVolatileQualifierBefore(DieType N);

  /// Recursively append the DIE type name when applicable.
  bool appendUnqualifiedName(DieType D,
                             std::string *OriginalFullName = nullptr);

  bool appendSubroutineNameAfter(DieType D, DieType Inner,
                                 bool SkipFirstParamIfArtificial, bool Const,
                                 bool Volatile);
  bool appendScopes(DieType D);
};

template <typename DieType, typename Visitor>
bool DWARFTypePrinter<DieType, Visitor>::appendTypeTagName(dwarf::Tag T) {
  StringRef TagStr = TagString(T);
  static constexpr StringRef Prefix = "dwarf::DW_TAG_";
  static constexpr StringRef Suffix = "_type";
  if (!TagStr.starts_with(Prefix) || !TagStr.ends_with(Suffix))
    return true;
  LLVM_QUICK_EXIT(V(TagStr.substr(
      Prefix.size(), TagStr.size() - (Prefix.size() + Suffix.size()))));
  LLVM_QUICK_EXIT(V(" "));
  return true;
}

template <typename DieType, typename Visitor>
bool DWARFTypePrinter<DieType, Visitor>::appendArrayType(const DieType &D) {
  for (const DieType &C : D.children()) {
    if (C.getTag() != dwarf::DW_TAG_subrange_type)
      continue;
    std::optional<uint64_t> LB;
    std::optional<uint64_t> Count;
    std::optional<uint64_t> UB;
    std::optional<uint64_t> DefaultLB;
    if (std::optional<typename DieType::DWARFFormValue> L = C.find(dwarf::DW_AT_lower_bound))
      LB = L->getAsUnsignedConstant();
    if (std::optional<typename DieType::DWARFFormValue> CountV = C.find(dwarf::DW_AT_count))
      Count = CountV->getAsUnsignedConstant();
    if (std::optional<typename DieType::DWARFFormValue> UpperV = C.find(dwarf::DW_AT_upper_bound))
      UB = UpperV->getAsUnsignedConstant();

    if (std::optional<uint64_t> LC = getLanguage(D))
        DefaultLB =
                 LanguageLowerBound(static_cast<dwarf::SourceLanguage>(*LC));

    if (DefaultLB == LB)
      LB = std::nullopt;
    if (!LB && !Count && !UB) {
      LLVM_QUICK_EXIT(V("[]"));
    } else if (!LB && (Count || UB) && DefaultLB) {
      LLVM_QUICK_EXIT(V("[") &&
                      V(Twine(Count ? *Count : *UB - *DefaultLB + 1).str()) &&
                      V("]"));
    } else {
      LLVM_QUICK_EXIT(V("[["));
      if (LB) {
        LLVM_QUICK_EXIT(V(Twine(*LB).str()));
      } else {
        LLVM_QUICK_EXIT(V("?"));
      }
      LLVM_QUICK_EXIT(V(", "));
      if (Count) {
        if (LB) {
          LLVM_QUICK_EXIT(V(Twine(*LB + *Count).str()));
        } else {
          LLVM_QUICK_EXIT(V("? + ") && V(Twine(*Count).str()));
        }
      } else if (UB) {
        LLVM_QUICK_EXIT(V(Twine(*UB + 1).str()));
      } else {
        LLVM_QUICK_EXIT(V("?"));
      }
      LLVM_QUICK_EXIT(V(")]"));
    }
  }
  EndedWithTemplate = false;
  return true;
}

namespace detail {
template<typename DieType>
DieType resolveReferencedType(DieType D,
                                     dwarf::Attribute Attr = dwarf::DW_AT_type) {
  return D.getAttributeValueAsReferencedDie(Attr); // .resolveTypeUnitReference();
}
template <typename DieType>
DieType resolveReferencedType(DieType D, typename DieType::DWARFFormValue F) {
  return D.getAttributeValueAsReferencedDie(F); // .resolveTypeUnitReference();
}
} // namespace detail

template <typename DieType, typename Visitor>
DieType DWARFTypePrinter<DieType, Visitor>::skipQualifiers(DieType D) {
  while (D && (D.getTag() == dwarf::DW_TAG_const_type ||
               D.getTag() == dwarf::DW_TAG_volatile_type))
    D = detail::resolveReferencedType(D);
  return D;
}

template <typename DieType, typename Visitor>
bool DWARFTypePrinter<DieType, Visitor>::needsParens(DieType D) {
  D = skipQualifiers(D);
  return D && (D.getTag() == dwarf::DW_TAG_subroutine_type ||
               D.getTag() == dwarf::DW_TAG_array_type);
}

template <typename DieType, typename Visitor>
bool DWARFTypePrinter<DieType, Visitor>::appendPointerLikeTypeBefore(
    DieType D, DieType Inner, StringRef Ptr) {
  LLVM_QUICK_EXIT(appendQualifiedNameBefore(Inner));
  if (Word)
    LLVM_QUICK_EXIT(V(" "));
  if (needsParens(Inner))
    LLVM_QUICK_EXIT(V("("));
  LLVM_QUICK_EXIT(V(Ptr));
  Word = false;
  EndedWithTemplate = false;
  return true;
}

template <typename DieType, typename Visitor>
std::optional<DieType> DWARFTypePrinter<DieType, Visitor>::appendUnqualifiedNameBefore(
    DieType D, std::string *OriginalFullName) {
  Word = true;
  if (!D) {
    LLVM_QUICK_EXIT_NONE(V("void"));
    return DieType();
  }
  DieType InnerDIE;
  auto Inner = [&] { return InnerDIE = detail::resolveReferencedType(D); };
  const dwarf::Tag T = D.getTag();
  switch (T) {
  case dwarf::DW_TAG_pointer_type: {
    LLVM_QUICK_EXIT_NONE(appendPointerLikeTypeBefore(D, Inner(), "*"));
    break;
  }
  case dwarf::DW_TAG_subroutine_type: {
    LLVM_QUICK_EXIT_NONE(appendQualifiedNameBefore(Inner()));
    if (Word) {
      LLVM_QUICK_EXIT_NONE(V(" "));
    }
    Word = false;
    break;
  }
  case dwarf::DW_TAG_array_type: {
    LLVM_QUICK_EXIT_NONE(appendQualifiedNameBefore(Inner()));
    break;
  }
  case dwarf::DW_TAG_reference_type:
    LLVM_QUICK_EXIT_NONE(appendPointerLikeTypeBefore(D, Inner(), "&"));
    break;
  case dwarf::DW_TAG_rvalue_reference_type:
    LLVM_QUICK_EXIT_NONE(appendPointerLikeTypeBefore(D, Inner(), "&&"));
    break;
  case dwarf::DW_TAG_ptr_to_member_type: {
    LLVM_QUICK_EXIT_NONE(appendQualifiedNameBefore(Inner()));
    if (needsParens(InnerDIE)) {
      LLVM_QUICK_EXIT_NONE(V("("));
    } else if (Word) {
      LLVM_QUICK_EXIT_NONE(V(" "));
    } if (DieType Cont = detail::resolveReferencedType(D, dwarf::DW_AT_containing_type)) {
      appendQualifiedName(Cont);
      EndedWithTemplate = false;
      LLVM_QUICK_EXIT_NONE(V("::"));
    }
    LLVM_QUICK_EXIT_NONE(V("*"));
    Word = false;
    break;
  }
  case dwarf::DW_TAG_LLVM_ptrauth_type:
    LLVM_QUICK_EXIT_NONE(appendQualifiedNameBefore(Inner()));
    break;
  case dwarf::DW_TAG_const_type:
  case dwarf::DW_TAG_volatile_type:
    LLVM_QUICK_EXIT_NONE(appendConstVolatileQualifierBefore(D));
    break;
  case dwarf::DW_TAG_namespace: {
    if (const char *Name = toString(D.find(dwarf::DW_AT_name), nullptr)) {
      LLVM_QUICK_EXIT_NONE(V(Name));
    } else {
      LLVM_QUICK_EXIT_NONE(V("(anonymous namespace)"));
    }
    break;
  }
  case dwarf::DW_TAG_unspecified_type: {
    StringRef TypeName = D.getShortName();
    if (TypeName == "decltype(nullptr)")
      TypeName = "std::nullptr_t";
    Word = true;
    LLVM_QUICK_EXIT_NONE(V(TypeName));
    EndedWithTemplate = false;
    break;
  }
    /*
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_class_type:
  case dwarf::DW_TAG_enumeration_type:
  case dwarf::DW_TAG_base_type:
  */
  default: {
    const char *NamePtr = toString(D.find(dwarf::DW_AT_name), nullptr);
    if (!NamePtr) {
      LLVM_QUICK_EXIT_NONE(appendTypeTagName(D.getTag()));
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
    LLVM_QUICK_EXIT_NONE(V(Name));
    // This check would be insufficient for operator overloads like
    // "operator>>" - but for now Clang doesn't try to simplify them, so this
    // is OK. Add more nuanced operator overload handling here if/when needed.
    if (Name.ends_with(">"))
      break;
    LLVM_QUICK_EXIT_NONE(appendAndTerminateTemplateParameters(D));
    break;
  }
  }
  return InnerDIE;
}

template <typename DieType, typename Visitor>
bool DWARFTypePrinter<DieType, Visitor>::appendAndTerminateTemplateParameters(DieType D) {
  std::optional<bool> R = appendTemplateParameters(D);
  if (!R)
    return false;
  if (!*R)
    return true;

  if (EndedWithTemplate)
    LLVM_QUICK_EXIT(V(" "));
  LLVM_QUICK_EXIT(V(">"));
  EndedWithTemplate = true;
  Word = true;
  return true;
}

template <typename DieType, typename Visitor>
bool DWARFTypePrinter<DieType, Visitor>::appendUnqualifiedNameAfter(
    DieType D, DieType Inner, bool SkipFirstParamIfArtificial) {
  if (!D)
    return true;
  switch (D.getTag()) {
  case dwarf::DW_TAG_subroutine_type: {
    LLVM_QUICK_EXIT(appendSubroutineNameAfter(
        D, Inner, SkipFirstParamIfArtificial, false, false));
    break;
  }
  case dwarf::DW_TAG_array_type: {
    LLVM_QUICK_EXIT(appendArrayType(D));
    break;
  }
  case dwarf::DW_TAG_const_type:
  case dwarf::DW_TAG_volatile_type:
    LLVM_QUICK_EXIT(appendConstVolatileQualifierAfter(D));
    break;
  case dwarf::DW_TAG_ptr_to_member_type:
  case dwarf::DW_TAG_reference_type:
  case dwarf::DW_TAG_rvalue_reference_type:
  case dwarf::DW_TAG_pointer_type: {
    if (needsParens(Inner))
      LLVM_QUICK_EXIT(V(")"));
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
        << getValOrNull(dwarf::DW_AT_LLVM_ptrauth_address_discriminated) << ", 0x0"
        << utohexstr(getValOrNull(dwarf::DW_AT_LLVM_ptrauth_extra_discriminator), true)
        << options << ")";
    LLVM_QUICK_EXIT(V(PtrauthStream.str()));
    break;
  }
    /*
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_class_type:
  case dwarf::DW_TAG_enumeration_type:
  case dwarf::DW_TAG_base_type:
  case dwarf::DW_TAG_namespace:
  */
  default:
    break;
  }
  return true;
}

namespace detail {
/// Returns True if the DIE TAG is one of the ones that is scopped.
inline bool scopedTAGs(dwarf::Tag Tag) {
  switch (Tag) {
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_class_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_namespace:
  case dwarf::DW_TAG_enumeration_type:
    return true;
  default:
    break;
  }
  return false;
}
} // namespace detail
template <typename DieType, typename Visitor>
bool DWARFTypePrinter<DieType, Visitor>::appendQualifiedName(DieType D) {
  if (D && detail::scopedTAGs(D.getTag()))
    LLVM_QUICK_EXIT(appendScopes(D.getParent()));
  LLVM_QUICK_EXIT(appendUnqualifiedName(D));
  return true;
}
template <typename DieType, typename Visitor>
std::optional<DieType> DWARFTypePrinter<DieType, Visitor>::appendQualifiedNameBefore(DieType D) {
  if (D && detail::scopedTAGs(D.getTag()))
    LLVM_QUICK_EXIT_NONE(appendScopes(D.getParent()));
  return appendUnqualifiedNameBefore(D);
}
template <typename DieType, typename Visitor>
std::optional<bool>
DWARFTypePrinter<DieType, Visitor>::appendTemplateParameters(
    DieType D, bool *FirstParameter) {
  bool FirstParameterValue = true;
  bool IsTemplate = false;
  if (!FirstParameter)
    FirstParameter = &FirstParameterValue;
  for (const DieType &C : D) {
    auto Sep = [&] {
      if (*FirstParameter) {
        LLVM_QUICK_EXIT(V("<"));
      } else {
        LLVM_QUICK_EXIT(V(", "));
      }
      IsTemplate = true;
      EndedWithTemplate = false;
      *FirstParameter = false;
      return true;
    };
    if (C.getTag() == dwarf::DW_TAG_GNU_template_parameter_pack) {
      IsTemplate = true;
      appendTemplateParameters(C, FirstParameter);
    }
    if (C.getTag() == dwarf::DW_TAG_template_value_parameter) {
      DieType T = detail::resolveReferencedType(C);
      LLVM_QUICK_EXIT_NONE(Sep());
      if (T.getTag() == dwarf::DW_TAG_enumeration_type) {
        LLVM_QUICK_EXIT_NONE(V("("));
        appendQualifiedName(T);
        LLVM_QUICK_EXIT_NONE(V(")"));
        auto Value = C.find(dwarf::DW_AT_const_value);
        LLVM_QUICK_EXIT_NONE(V(std::to_string(*Value->getAsSignedConstant())));
        continue;
      }
      // /Maybe/ we could do pointer type parameters, looking for the
      // symbol in the ELF symbol table to get back to the variable...
      // but probably not worth it.
      if (T.getTag() == dwarf::DW_TAG_pointer_type)
        continue;
      const char *RawName = toString(T.find(dwarf::DW_AT_name), nullptr);
      assert(RawName);
      StringRef Name = RawName;
      auto Value = C.find(dwarf::DW_AT_const_value);
      bool IsQualifiedChar = false;
      if (Name == "bool") {
        LLVM_QUICK_EXIT_NONE(V((*Value->getAsUnsignedConstant() ? "true" : "false")));
      } else if (Name == "short") {
        LLVM_QUICK_EXIT_NONE(V("(short)"));
        LLVM_QUICK_EXIT_NONE(V(std::to_string(*Value->getAsSignedConstant())));
      } else if (Name == "unsigned short") {
        LLVM_QUICK_EXIT_NONE(V("(unsigned short)"));
        LLVM_QUICK_EXIT_NONE(V(std::to_string(*Value->getAsSignedConstant())));
      } else if (Name == "int") {
        LLVM_QUICK_EXIT_NONE(V(std::to_string(*Value->getAsSignedConstant())));
      } else if (Name == "long") {
        LLVM_QUICK_EXIT_NONE(V(std::to_string(*Value->getAsSignedConstant())));
        LLVM_QUICK_EXIT_NONE(V("L"));
      } else if (Name == "long long") {
        LLVM_QUICK_EXIT_NONE(V(std::to_string(*Value->getAsSignedConstant())));
        LLVM_QUICK_EXIT_NONE(V("LL"));
      } else if (Name == "unsigned int") {
        LLVM_QUICK_EXIT_NONE(V(std::to_string(*Value->getAsUnsignedConstant())));
        LLVM_QUICK_EXIT_NONE(V("U"));
      } else if (Name == "unsigned long") {
        LLVM_QUICK_EXIT_NONE(V(std::to_string(*Value->getAsUnsignedConstant())));
        LLVM_QUICK_EXIT_NONE(V("UL"));
      } else if (Name == "unsigned long long") {
        LLVM_QUICK_EXIT_NONE(V(std::to_string(*Value->getAsUnsignedConstant())));
        LLVM_QUICK_EXIT_NONE(V("ULL"));
      } else if (Name == "char" ||
                 (IsQualifiedChar =
                      (Name == "unsigned char" || Name == "signed char"))) {
        // FIXME: check T's dwarf::DW_AT_type to see if it's signed or not (since
        // char signedness is implementation defined).
        auto Val = *Value->getAsSignedConstant();
        // Copied/hacked up from Clang's CharacterLiteral::print - incomplete
        // (doesn't actually support different character types/widths, sign
        // handling's not done, and doesn't correctly test if a character is
        // printable or needs to use a numeric escape sequence instead)
        if (IsQualifiedChar) {
          LLVM_QUICK_EXIT_NONE(V("("));
          LLVM_QUICK_EXIT_NONE(V(Name));
          LLVM_QUICK_EXIT_NONE(V(")"));
        }
        switch (Val) {
        case '\\':
          LLVM_QUICK_EXIT_NONE(V("'\\\\'"));
          break;
        case '\'':
          LLVM_QUICK_EXIT_NONE(V("'\\''"));
          break;
        case '\a':
          // TODO: K&R: the meaning of '\\a' is different in traditional C
          LLVM_QUICK_EXIT_NONE(V("'\\a'"));
          break;
        case '\b':
          LLVM_QUICK_EXIT_NONE(V("'\\b'"));
          break;
        case '\f':
          LLVM_QUICK_EXIT_NONE(V("'\\f'"));
          break;
        case '\n':
          LLVM_QUICK_EXIT_NONE(V("'\\n'"));
          break;
        case '\r':
          LLVM_QUICK_EXIT_NONE(V("'\\r'"));
          break;
        case '\t':
          LLVM_QUICK_EXIT_NONE(V("'\\t'"));
          break;
        case '\v':
          LLVM_QUICK_EXIT_NONE(V("'\\v'"));
          break;
        default:
          if ((Val & ~0xFFu) == ~0xFFu)
            Val &= 0xFFu;
          if (Val < 127 && Val >= 32) {
            LLVM_QUICK_EXIT_NONE(V("'"));
            LLVM_QUICK_EXIT_NONE(V(Twine((char)Val).str()));
            LLVM_QUICK_EXIT_NONE(V("'"));
          } else if (Val < 256) {
            LLVM_QUICK_EXIT_NONE(V(detail::toString(llvm::format("'\\x%02" PRIx64 "'", Val))));
          } else if (Val <= 0xFFFF) {
            LLVM_QUICK_EXIT_NONE(
                V(detail::toString(llvm::format("'\\u%04" PRIx64 "'", Val))));
          } else {
            LLVM_QUICK_EXIT_NONE(
                V(detail::toString(llvm::format("'\\U%08" PRIx64 "'", Val))));
          }
        }
      }
      continue;
    }
    if (C.getTag() == dwarf::DW_TAG_GNU_template_template_param) {
      const char *RawName =
          toString(C.find(dwarf::DW_AT_GNU_template_name), nullptr);
      assert(RawName);
      StringRef Name = RawName;
      LLVM_QUICK_EXIT_NONE(Sep());
      LLVM_QUICK_EXIT_NONE(V(Name));
      continue;
    }
    if (C.getTag() != dwarf::DW_TAG_template_type_parameter)
      continue;
    auto TypeAttr = C.find(dwarf::DW_AT_type);
    LLVM_QUICK_EXIT_NONE(Sep());
    appendQualifiedName(TypeAttr ? detail::resolveReferencedType(C, *TypeAttr)
                                 : DieType());
  }
  if (IsTemplate && *FirstParameter && FirstParameter == &FirstParameterValue) {
    LLVM_QUICK_EXIT_NONE(V("<"));
    EndedWithTemplate = false;
  }
  return IsTemplate;
}
template <typename DieType, typename Visitor>
void DWARFTypePrinter<DieType, Visitor>::decomposeConstVolatile(DieType &N, DieType &T,
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
template <typename DieType, typename Visitor>
bool DWARFTypePrinter<DieType, Visitor>::appendConstVolatileQualifierAfter(DieType N) {
  DieType C;
  DieType V;
  DieType T;
  decomposeConstVolatile(N, T, C, V);
  if (T && T.getTag() == dwarf::DW_TAG_subroutine_type) {
    LLVM_QUICK_EXIT(appendSubroutineNameAfter(T, detail::resolveReferencedType(T), false, C.isValid(),
                              V.isValid()));
  } else {
    LLVM_QUICK_EXIT(appendUnqualifiedNameAfter(T, detail::resolveReferencedType(T)));
  }
  return true;
}
template <typename DieType, typename Visitor>
bool DWARFTypePrinter<DieType, Visitor>::appendConstVolatileQualifierBefore(DieType N) {
  DieType C;
  DieType Vol;
  DieType T;
  decomposeConstVolatile(N, T, C, Vol);
  bool Subroutine = T && T.getTag() == dwarf::DW_TAG_subroutine_type;
  DieType A = T;
  while (A && A.getTag() == dwarf::DW_TAG_array_type)
    A = detail::resolveReferencedType(A);
  bool Leading =
      (!A || (A.getTag() != dwarf::DW_TAG_pointer_type &&
              A.getTag() != llvm::dwarf::DW_TAG_ptr_to_member_type)) &&
      !Subroutine;
  if (Leading) {
    if (C) {
      LLVM_QUICK_EXIT(V("const "));
    }
    if (Vol) {
      LLVM_QUICK_EXIT(V("volatile "));
    }
  }
  appendQualifiedNameBefore(T);
  if (!Leading && !Subroutine) {
    Word = true;
    if (C) {
      LLVM_QUICK_EXIT(V("const"));
    }
    if (Vol) {
      if (C) {
        LLVM_QUICK_EXIT(V(" "));
      }
      LLVM_QUICK_EXIT(V("volatile"));
    }
  }
  return true;
}
template <typename DieType, typename Visitor>
bool DWARFTypePrinter<DieType, Visitor>::appendUnqualifiedName(
    DieType D, std::string *OriginalFullName) {
  // FIXME: We should have pretty printers per language. Currently we print
  // everything as if it was C++ and fall back to the TAG type name.
  std::optional<DieType> Inner =
      appendUnqualifiedNameBefore(D, OriginalFullName);
  LLVM_QUICK_EXIT(Inner);
  LLVM_QUICK_EXIT(appendUnqualifiedNameAfter(D, *Inner));
  return true;
}
template <typename DieType, typename Visitor>
bool DWARFTypePrinter<DieType, Visitor>::appendSubroutineNameAfter(
    DieType D, DieType Inner, bool SkipFirstParamIfArtificial, bool Const,
    bool Volatile) {
  DieType FirstParamIfArtificial;
  LLVM_QUICK_EXIT(V("("));
  EndedWithTemplate = false;
  bool First = true;
  bool RealFirst = true;
  for (DieType P : D) {
    if (P.getTag() != dwarf::DW_TAG_formal_parameter &&
        P.getTag() != dwarf::DW_TAG_unspecified_parameters)
      return true;
    DieType T = detail::resolveReferencedType(P);
    if (SkipFirstParamIfArtificial && RealFirst && P.find(dwarf::DW_AT_artificial)) {
      FirstParamIfArtificial = T;
      RealFirst = false;
      continue;
    }
    if (!First) {
      LLVM_QUICK_EXIT(V(", "));
    }
    First = false;
    if (P.getTag() == dwarf::DW_TAG_unspecified_parameters) {
      LLVM_QUICK_EXIT(V("..."));
    } else {
      LLVM_QUICK_EXIT(appendQualifiedName(T));
    }
  }
  EndedWithTemplate = false;
  LLVM_QUICK_EXIT(V(")"));
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
      LLVM_QUICK_EXIT(V(" __attribute__((stdcall))"));
      break;
    case dwarf::CallingConvention::DW_CC_BORLAND_msfastcall:
      LLVM_QUICK_EXIT(V(" __attribute__((fastcall))"));
      break;
    case dwarf::CallingConvention::DW_CC_BORLAND_thiscall:
      LLVM_QUICK_EXIT(V(" __attribute__((thiscall))"));
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_vectorcall:
      LLVM_QUICK_EXIT(V(" __attribute__((vectorcall))"));
      break;
    case dwarf::CallingConvention::DW_CC_BORLAND_pascal:
      LLVM_QUICK_EXIT(V(" __attribute__((pascal))"));
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_Win64:
      LLVM_QUICK_EXIT(V(" __attribute__((ms_abi))"));
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_X86_64SysV:
      LLVM_QUICK_EXIT(V(" __attribute__((sysv_abi))"));
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_AAPCS:
      // AArch64VectorCall missing?
      LLVM_QUICK_EXIT(V(" __attribute__((pcs(\"aapcs\")))"));
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_AAPCS_VFP:
      LLVM_QUICK_EXIT(V(" __attribute__((pcs(\"aapcs-vfp\")))"));
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_IntelOclBicc:
      LLVM_QUICK_EXIT(V(" __attribute__((intel_ocl_bicc))"));
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
      LLVM_QUICK_EXIT(V(" __attribute__((swiftcall))"));
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_PreserveMost:
      LLVM_QUICK_EXIT(V(" __attribute__((preserve_most))"));
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_PreserveAll:
      LLVM_QUICK_EXIT(V(" __attribute__((preserve_all))"));
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_PreserveNone:
      LLVM_QUICK_EXIT(V(" __attribute__((preserve_none))"));
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_X86RegCall:
      LLVM_QUICK_EXIT(V(" __attribute__((regcall))"));
      break;
    case dwarf::CallingConvention::DW_CC_LLVM_M68kRTD:
      LLVM_QUICK_EXIT(V(" __attribute__((m68k_rtd))"));
      break;
    }
  }

  if (Const) {
    LLVM_QUICK_EXIT(V(" const"));
  }
  if (Volatile) {
    LLVM_QUICK_EXIT(V(" volatile"));
  }
  if (D.find(dwarf::DW_AT_reference)) {
    LLVM_QUICK_EXIT(V(" &"));
  }
  if (D.find(dwarf::DW_AT_rvalue_reference)) {
    LLVM_QUICK_EXIT(V(" &&"));
  }

  LLVM_QUICK_EXIT(
      appendUnqualifiedNameAfter(Inner, detail::resolveReferencedType(Inner)));
  return true;
}
template <typename DieType, typename Visitor>
bool DWARFTypePrinter<DieType, Visitor>::appendScopes(DieType D) {
  if (D.getTag() == dwarf::DW_TAG_compile_unit)
    return true;
  if (D.getTag() == dwarf::DW_TAG_type_unit)
    return true;
  if (D.getTag() == dwarf::DW_TAG_skeleton_unit)
    return true;
  if (D.getTag() == dwarf::DW_TAG_subprogram)
    return true;
  if (D.getTag() == dwarf::DW_TAG_lexical_block)
    return true;
  //D = D.resolveTypeUnitReference();
  if (DieType P = D.getParent()) {
    LLVM_QUICK_EXIT(appendScopes(P));
  }
  LLVM_QUICK_EXIT(appendUnqualifiedName(D));
  LLVM_QUICK_EXIT(V("::"));
  return true;
}
} // namespace llvm

#undef LLVM_QUICK_EXIT
#undef LLVM_QUICK_EXIT_NONE

#endif // LLVM_DEBUGINFO_DWARF_DWARFTYPEPRINTER_H
