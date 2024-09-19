//===-- SwiftDWARFImporterForClangTypes.cpp ------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2022 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "SwiftDWARFImporterForClangTypes.h"
#include "SwiftASTContext.h"

#include "swift/ClangImporter/ClangImporter.h"
#include "clang/AST/DeclObjC.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringRef.h"

#include "Plugins/ExpressionParser/Clang/ClangASTImporter.h"
#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/TypeMap.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Timer.h"

using namespace lldb;
using llvm::StringRef;

namespace lldb_private {

/// Used to filter out types with mismatching kinds.
static bool HasTypeKind(TypeSP clang_type_sp, swift::ClangTypeKind kind) {
  CompilerType fwd_type = clang_type_sp->GetForwardCompilerType();
  clang::QualType qual_type = ClangUtil::GetQualType(fwd_type);
  switch (kind) {
  case swift::ClangTypeKind::Typedef:
    /*=swift::ClangTypeKind::ObjCClass:*/
    return !qual_type->isObjCObjectOrInterfaceType() &&
           !qual_type->getAs<clang::TypedefType>();
  case swift::ClangTypeKind::Tag:
    return !qual_type->isStructureOrClassType() &&
           !qual_type->isEnumeralType() && !qual_type->isUnionType();
  case swift::ClangTypeKind::ObjCProtocol:
    // Not implemented since Objective-C protocols aren't yet
    // described in DWARF.
    return true;
  }
}

SwiftDWARFImporterForClangTypes::SwiftDWARFImporterForClangTypes(
    TypeSystemSwiftTypeRef &ts)
    : m_swift_typesystem(ts),
      m_description(ts.GetDescription() + "::SwiftDWARFImporterForClangTypes") {
}

void SwiftDWARFImporterForClangTypes::lookupValue(
    StringRef name, std::optional<swift::ClangTypeKind> kind,
    StringRef inModule, llvm::SmallVectorImpl<CompilerType> &results) {
  LLDB_SCOPED_TIMER();
  LLDB_LOG(GetLog(LLDBLog::Types), "{0}::lookupValue(\"{1}\")", m_description,
           name.str());

  // We will not find any Swift types in the Clang compile units.
  ConstString name_cs(name);
  if (SwiftLanguageRuntime::IsSwiftMangledName(name_cs.GetStringRef()))
    return;

  // Find the type in the debug info.
  TypeSP clang_type_sp;
  // FIXME: LookupClangType won't work for nested C++ types.
  if (m_swift_typesystem.GetModule())
    clang_type_sp = m_swift_typesystem.LookupClangType(name);
  else if (TargetSP target_sp = m_swift_typesystem.GetTargetWP().lock()) {
    // In a scratch context, check the module's DWARFImporterDelegates
    // first.
    //
    // It's a common pattern that a type is revisited immediately
    // after looking it up in a per-module context in the scratch
    // context for dynamic type resolution.

    auto images = target_sp->GetImages();
    for (size_t i = 0; i != images.GetSize(); ++i) {
      auto module_sp = images.GetModuleAtIndex(i);
      if (!module_sp)
        continue;
      auto ts = module_sp->GetTypeSystemForLanguage(lldb::eLanguageTypeSwift);
      if (!ts) {
        llvm::consumeError(ts.takeError());
        continue;
      }
      auto swift_ts = llvm::dyn_cast_or_null<TypeSystemSwift>(ts->get());
      if (!swift_ts)
        continue;
      // FIXME: LookupClangType won't work for nested C++ types.
      clang_type_sp =
          swift_ts->GetTypeSystemSwiftTypeRef().LookupClangType(name);
      if (clang_type_sp)
        break;
    }
  }

  if (!clang_type_sp)
    return;

  // Filter out types with a mismatching type kind.
  if (kind && HasTypeKind(clang_type_sp, *kind))
    return;

  // Realize the full type.
  CompilerType fwd_type = clang_type_sp->GetForwardCompilerType();
  // Filter out non-Clang types.
  if (!fwd_type.GetTypeSystem().dyn_cast_or_null<TypeSystemClang>())
    return;

  results.push_back(clang_type_sp->GetFullCompilerType());
}

void SwiftDWARFImporterDelegate::importType(
    clang::QualType qual_type, clang::ASTContext &from_ctx,
    clang::ASTContext &to_ctx, std::optional<swift::ClangTypeKind> kind,
    llvm::SmallVectorImpl<clang::Decl *> &results) {
  clang::ASTImporter importer(
      to_ctx, to_ctx.getSourceManager().getFileManager(), from_ctx,
      from_ctx.getSourceManager().getFileManager(), false);
  llvm::Expected<clang::QualType> clang_type(importer.Import(qual_type));
  if (!clang_type) {
    llvm::consumeError(clang_type.takeError());
    return;
  }

  // Retrieve the imported type's Decl.
  if (kind) {
    if (clang::Decl *clang_decl = GetDeclForTypeAndKind(*clang_type, *kind))
      results.push_back(clang_decl);
  } else {
    swift::ClangTypeKind kinds[] = {
        swift::ClangTypeKind::Typedef, // =swift::ClangTypeKind::ObjCClass,
        swift::ClangTypeKind::Tag, swift::ClangTypeKind::ObjCProtocol};
    for (auto kind : kinds)
      if (clang::Decl *clang_decl = GetDeclForTypeAndKind(*clang_type, kind))
        results.push_back(clang_decl);
  }
}

clang::Decl *
SwiftDWARFImporterDelegate::GetDeclForTypeAndKind(clang::QualType qual_type,
                                                  swift::ClangTypeKind kind) {
  switch (kind) {
  case swift::ClangTypeKind::Typedef:
    /*=swift::ClangTypeKind::ObjCClass:*/
    if (auto *obj_type = qual_type->getAsObjCInterfaceType())
      return obj_type->getInterface();
    if (auto *typedef_type = qual_type->getAs<clang::TypedefType>())
      return typedef_type->getDecl();
    break;
  case swift::ClangTypeKind::Tag:
    return qual_type->getAsTagDecl();
  case swift::ClangTypeKind::ObjCProtocol:
    // Not implemented since Objective-C protocols aren't yet
    // described in DWARF.
    break;
  }
  return nullptr;
}

SwiftDWARFImporterDelegate::SwiftDWARFImporterDelegate(SwiftASTContext &ts)
    : m_swift_ast_ctx(ts),
      m_importer(m_swift_ast_ctx.GetTypeSystemSwiftTypeRef()
                     .GetSwiftDWARFImporterForClangTypes()),
      m_description(ts.GetDescription() + "::SwiftDWARFImporterDelegate") {}

void SwiftDWARFImporterDelegate::lookupValue(
    StringRef name, std::optional<swift::ClangTypeKind> kind,
    StringRef inModule, llvm::SmallVectorImpl<clang::Decl *> &results) {
  LLDB_LOG(GetLog(LLDBLog::Types), "{0}::lookupValue(\"{1}\")", m_description,
           name.str());
  if (!name.size() || name[0] < 0) {
    LLDB_LOG(GetLog(LLDBLog::Types),
             "SwiftDWARFImporterDelegate was asked to look up a type with a "
             "non-ASCII or empty type name");
    return;
  }
  auto clang_importer = m_swift_ast_ctx.GetClangImporter();
  if (!clang_importer) {
    LLDB_LOG(GetLog(LLDBLog::Types), "no clangimporter");
    return;
  }

  llvm::SmallVector<CompilerType, 1> types;
  m_importer.lookupValue(name, kind, inModule, types);
  for (auto &compiler_type : types) {
    auto type_system =
        compiler_type.GetTypeSystem().dyn_cast_or_null<TypeSystemClang>();
    if (!type_system)
      continue;

    // Import the type into SwiftASTContext's ClangImporter's clang::ASTContext.
    clang::ASTContext &to_ctx = clang_importer->getClangASTContext();
    clang::ASTContext &from_ctx = type_system->getASTContext();
    clang::QualType qual_type = ClangUtil::GetQualType(compiler_type);
    importType(qual_type, from_ctx, to_ctx, kind, results);

  }
  LLDB_LOG(GetLog(LLDBLog::Types),
           "{0}::lookupValue() -- imported {1} types from debug info.",
           m_description.c_str(), results.size());
}

} // namespace lldb_private
