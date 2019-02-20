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

#include "lldb/Utility/StreamString.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/SwiftASTContext.h"

#include "swift/AST/Decl.h"
#include "swift/ClangImporter/ClangImporter.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/StringRef.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using namespace lldb_private::formatters::swift;

bool lldb_private::formatters::swift::SwiftOptionSetSummaryProvider::
    WouldEvenConsiderFormatting(CompilerType clang_type) {
  SwiftASTContext *swift_ast_ctx =
      llvm::dyn_cast_or_null<SwiftASTContext>(clang_type.GetTypeSystem());
  if (!swift_ast_ctx)
    return false;

  return clang_type.IsValid() &&
         swift_ast_ctx->IsTrivialOptionSetType(clang_type) &&
         swift_ast_ctx->IsImportedType(clang_type, nullptr);
}

lldb_private::formatters::swift::SwiftOptionSetSummaryProvider::
    SwiftOptionSetSummaryProvider(CompilerType clang_type)
    : TypeSummaryImpl(TypeSummaryImpl::Kind::eInternal,
                      TypeSummaryImpl::Flags()),
      m_type(clang_type), m_cases() {}

static ConstString GetDisplayCaseName(::swift::ClangImporter *clang_importer,
                                      clang::EnumConstantDecl *case_decl) {
  if (clang_importer) {
    ::swift::Identifier imported_identifier =
        clang_importer->getEnumConstantName(case_decl);
    if (false == imported_identifier.empty())
      return ConstString(imported_identifier.str());
  }
  return ConstString(case_decl->getName());
}

static clang::EnumDecl *GetAsEnumDecl(const CompilerType &compiler_type) {
  if (compiler_type.IsValid() &&
      llvm::dyn_cast_or_null<ClangASTContext>(compiler_type.GetTypeSystem())) {
    opaque_compiler_type_t clang_type = compiler_type.GetOpaqueQualType();
    clang::QualType qual_type = clang::QualType::getFromOpaquePtr(clang_type);
    const clang::Type::TypeClass type_class = qual_type->getTypeClass();
    switch (type_class) {
    case clang::Type::TypeClass::Enum: {
      if (const clang::EnumType *enum_type =
              qual_type->getAs<clang::EnumType>())
        return enum_type->getDecl();
      break;
    }
    default:
      break;
    }
  }

  return nullptr;
}

void lldb_private::formatters::swift::SwiftOptionSetSummaryProvider::
    FillCasesIfNeeded() {
  if (m_cases.hasValue())
    return;

  m_cases = CasesVector();
  SwiftASTContext *swift_ast_ctx =
      llvm::dyn_cast_or_null<SwiftASTContext>(m_type.GetTypeSystem());

  if (swift_ast_ctx && swift_ast_ctx->IsTrivialOptionSetType(m_type)) {
    CompilerType original_type;
    if (swift_ast_ctx->IsImportedType(m_type, &original_type)) {
      clang::EnumDecl *enum_decl = GetAsEnumDecl(original_type);
      if (enum_decl) {
        ::swift::ClangImporter *clang_importer =
            swift_ast_ctx->GetClangImporter();
        auto iter = enum_decl->enumerator_begin(),
             end = enum_decl->enumerator_end();
        for (; iter != end; ++iter) {
          clang::EnumConstantDecl *case_decl = *iter;
          if (case_decl) {
            llvm::APInt case_init_val(case_decl->getInitVal());
            // extend all cases to 64 bits so that equality check is fast
            // but if they are larger than 64, I am going to get out of that
            // case
            // and then pick it up again as unmatched data at the end
            if (case_init_val.getBitWidth() < 64)
              case_init_val = case_init_val.zext(64);
            if (case_init_val.getBitWidth() > 64)
              continue;
            ConstString case_name(
                GetDisplayCaseName(clang_importer, case_decl));
            m_cases->push_back({case_init_val, case_name});
          }
        }
      }
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
  return sstr.GetString();
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
  if (ReadValueIfAny(*rawValue_sp, value)) {
    FillCasesIfNeeded();

    StreamString ss;
    bool first_match = true;
    bool any_match = false;

    llvm::APInt matched_value(llvm::APInt::getNullValue(64));

    auto iter = m_cases->begin(), end = m_cases->end();
    for (; iter != end; ++iter) {
      llvm::APInt case_value = iter->first;
      if ((case_value & value) == case_value) {
        // hey a case matched!!
        any_match = true;
        if (first_match) {
          ss.Printf("[.%s", iter->second.AsCString());
          first_match = false;
        } else {
          ss.Printf(", .%s", iter->second.AsCString());
        }

        matched_value |= case_value;

        // if we matched everything, get out
        if (matched_value == value)
          break;
      }
    }

    if (any_match) {
      // if we found a full match, then close the list
      if (matched_value == value)
        ss.PutChar(']');
      else {
        // print the unaccounted-for bits separately
        llvm::APInt residual = (value & ~matched_value);
        ss.Printf(", 0x%s]", residual.toString(16, false).c_str());
      }
    }

    // if we printed anything, use it
    const char *data = ss.GetData();
    if (data && data[0]) {
      dest.assign(data);
      return true;
    }
  }
  return false;
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
