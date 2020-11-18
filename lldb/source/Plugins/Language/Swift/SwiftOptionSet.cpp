//===-- SwiftOptionSet.cpp --------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftOptionSet.h"

#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "Plugins/TypeSystem/Swift/SwiftASTContext.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Utility/StreamString.h"

#include "swift/AST/Decl.h"
#include "swift/ClangImporter/ClangImporter.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

/// If this is a Clang enum wrapped in a Swift type, return the clang::EnumDecl.
static std::pair<clang::EnumDecl *, TypeSystemClang *>
GetAsEnumDecl(CompilerType swift_type) {
  swift_type = swift_type.GetCanonicalType();
  if (!swift_type)
    return {nullptr, nullptr};

  TypeSystemSwift *swift_ast_ctx =
      llvm::dyn_cast_or_null<TypeSystemSwift>(swift_type.GetTypeSystem());
  if (!swift_ast_ctx)
    return {nullptr, nullptr};

  CompilerType clang_type;
  if (!swift_ast_ctx->IsImportedType(swift_type.GetOpaqueQualType(),
                                     &clang_type))
    return {nullptr, nullptr};

  if (!clang_type.IsValid())
    return {nullptr, nullptr};

  auto *clang_ts = llvm::dyn_cast<TypeSystemClang>(clang_type.GetTypeSystem());
  if (!clang_ts)
    return {nullptr, nullptr};

  auto qual_type =
      clang::QualType::getFromOpaquePtr(clang_type.GetOpaqueQualType());
  if (qual_type->getTypeClass() != clang::Type::TypeClass::Enum)
    return {nullptr, nullptr};

  if (const clang::EnumType *enum_type = qual_type->getAs<clang::EnumType>())
    return {enum_type->getDecl(), clang_ts};
  return {nullptr, nullptr};
}

bool lldb_private::formatters::swift::SwiftOptionSetSummaryProvider::
    WouldEvenConsiderFormatting(CompilerType swift_type) {
  return GetAsEnumDecl(swift_type).first;
}

lldb_private::formatters::swift::SwiftOptionSetSummaryProvider::
    SwiftOptionSetSummaryProvider(CompilerType clang_type)
    : TypeSummaryImpl(TypeSummaryImpl::Kind::eInternal,
                      TypeSummaryImpl::Flags()),
      m_type(clang_type), m_cases() {}

void lldb_private::formatters::swift::SwiftOptionSetSummaryProvider::
    FillCasesIfNeeded() {
  if (m_cases.hasValue())
    return;

  m_cases = CasesVector();
  auto decl_ts = GetAsEnumDecl(m_type);
  clang::EnumDecl *enum_decl = decl_ts.first;
  if (!enum_decl)
    return;

  // FIXME: Delete this type reconstruction block. For GetSwiftName() to
  // fully work, ClangImporter's ImportName class needs to be made
  // standalone and provided with a callback to read the APINote
  // information.
  auto *ts = llvm::cast<TypeSystemSwift>(m_type.GetTypeSystem());
  if (auto *trts = llvm::dyn_cast_or_null<TypeSystemSwiftTypeRef>(ts)) {
    m_type = trts->ReconstructType(m_type);
    ts = llvm::cast<SwiftASTContext>(m_type.GetTypeSystem());
    decl_ts = GetAsEnumDecl(m_type);
    enum_decl = decl_ts.first;
    if (!enum_decl)
      return;
  }

  SwiftASTContext *swift_ast_ctx =
      llvm::dyn_cast_or_null<SwiftASTContext>(m_type.GetTypeSystem());
  auto iter = enum_decl->enumerator_begin(), end = enum_decl->enumerator_end();
  for (; iter != end; ++iter) {
    clang::EnumConstantDecl *case_decl = *iter;
    if (case_decl) {
      llvm::APInt case_init_val(case_decl->getInitVal());
      // Extend all cases to 64 bits so that equality check is fast
      // but if they are larger than 64, I am going to get out of that
      // case and then pick it up again as unmatched data at the end.
      if (case_init_val.getBitWidth() < 64)
        case_init_val = case_init_val.zext(64);
      if (case_init_val.getBitWidth() > 64)
        continue;
      ConstString case_name(ts->GetSwiftName(case_decl, *decl_ts.second));
      m_cases->push_back({case_init_val, case_name});
    }
  }
}

std::string lldb_private::formatters::swift::SwiftOptionSetSummaryProvider::
    GetDescription() {
  StreamString sstr;
  sstr.Printf("`%s `%s%s%s%s%s%s%s", "Swift OptionSet summary provider",
              Cascades() ? "" : " (not cascading)", " (may show children)",
              !DoesPrintValue(nullptr) ? " (hide value)" : "",
              IsOneLiner() ? " (one-line printout)" : "",
              SkipsPointers() ? " (skip pointers)" : "",
              SkipsReferences() ? " (skip references)" : "",
              HideNames(nullptr) ? " (hide member names)" : "");
  return sstr.GetString().str();
}

static bool ReadValueIfAny(ValueObject &valobj, llvm::APInt &value) {
  ValueObjectSP most_qualified_sp(valobj.GetQualifiedRepresentationIfAvailable(
      lldb::eDynamicDontRunTarget, true));

  bool success;
  value = llvm::APInt(64, most_qualified_sp->GetValueAsUnsigned(0, &success));
  return success;
}

static ValueObjectSP GetRawValue(ValueObject *valobj) {
  if (!valobj)
    return nullptr;

  static ConstString g_rawValue("rawValue");

  auto rawValue_sp = valobj->GetChildMemberWithName(g_rawValue, true);

  return rawValue_sp;
}

bool lldb_private::formatters::swift::SwiftOptionSetSummaryProvider::
    FormatObject(ValueObject *valobj, std::string &dest,
                 const TypeSummaryOptions &options) {
  auto rawValue_sp = GetRawValue(valobj);
  if (!rawValue_sp)
    return false;

  llvm::APInt value;
  if (!ReadValueIfAny(*rawValue_sp, value))
    return false;

  FillCasesIfNeeded();

  StreamString ss;
  bool first_match = true;
  bool any_match = false;

  llvm::APInt matched_value(llvm::APInt::getNullValue(64));

  for (auto val_name : *m_cases) {
    llvm::APInt case_value = val_name.first;
    // Print single valued sets without using enclosing brackets.
    // `WouldEvenConsiderFormatting` can't opt out early because it
    // has only the type, but needs the value for this case.
    if (case_value == value) {
      ss << '.' << val_name.second;
      dest.assign(ss.GetData());
      return true;
    }
    // Don't display the zero case in an option set unless it's the
    // only value.
    if (case_value == 0 && value != 0)
      continue;
    if ((case_value & value) == case_value) {
      any_match = true;
      if (first_match) {
        ss << "[." << val_name.second;
        first_match = false;
      } else {
        ss << ", ." << val_name.second;
      }

      matched_value |= case_value;

      // If we matched everything, get out.
      if (matched_value == value)
        break;
    }
  }

  if (!any_match)
    return false;

  if (matched_value != value) {
    // Print the unaccounted-for bits separately.
    llvm::APInt residual = value & ~matched_value;
    ss << ", 0x" << residual.toString(16, false);
  }
  ss << ']';

  dest.assign(ss.GetData());
  return true;
}

bool lldb_private::formatters::swift::SwiftOptionSetSummaryProvider::
    DoesPrintChildren(ValueObject *valobj) const {
  auto rawValue_sp = GetRawValue(valobj);
  if (!rawValue_sp)
    return false;

  llvm::APInt value;
  // only show children if you couldn't read the value of rawValue
  return (false == ReadValueIfAny(*rawValue_sp, value));
}
