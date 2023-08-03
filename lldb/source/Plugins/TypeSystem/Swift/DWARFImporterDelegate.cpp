//===-- DWARFImporterDelegate.cpp -----------------------------------------===//
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

/// Implements a swift::DWARFImporterDelegate to look up Clang types in DWARF.
///
/// During compile time, ClangImporter-imported Clang modules are compiled with
/// -gmodules, which emits a DWARF rendition of all types defined in the module
/// into the .pcm file. On Darwin, these types can be collected by
/// dsymutil. This delegate allows DWARFImporter to ask LLDB to look up a Clang
/// type by name, synthesize a Clang AST from it. DWARFImporter then hands this
/// Clang AST to ClangImporter to import the type into Swift.
class SwiftDWARFImporterDelegate : public swift::DWARFImporterDelegate {
  TypeSystemSwiftTypeRef &m_swift_typesystem;
  using ModuleAndName = std::pair<const char *, const char *>;
  std::string m_description;

  /// Used to filter out types with mismatching kinds.
  bool HasTypeKind(TypeSP clang_type_sp, swift::ClangTypeKind kind) {
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

  clang::Decl *GetDeclForTypeAndKind(clang::QualType qual_type,
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

  static CompilerContextKind
  GetCompilerContextKind(llvm::Optional<swift::ClangTypeKind> kind) {
    if (!kind)
      return CompilerContextKind::AnyType;
    switch (*kind) {
    case swift::ClangTypeKind::Typedef:
      /*=swift::ClangTypeKind::ObjCClass:*/
      return (CompilerContextKind)((uint16_t)CompilerContextKind::Any |
                                   (uint16_t)CompilerContextKind::Typedef |
                                   (uint16_t)CompilerContextKind::Struct);
      break;
    case swift::ClangTypeKind::Tag:
      return (CompilerContextKind)((uint16_t)CompilerContextKind::Any |
                                   (uint16_t)CompilerContextKind::Class |
                                   (uint16_t)CompilerContextKind::Struct |
                                   (uint16_t)CompilerContextKind::Union |
                                   (uint16_t)CompilerContextKind::Enum);
      // case swift::ClangTypeKind::ObjCProtocol:
      // Not implemented since Objective-C protocols aren't yet
      // described in DWARF.
    default:
      return CompilerContextKind::Invalid;
    }
  }

  /// Import \p qual_type from one clang ASTContext to another and
  /// add it to \p results if successful.
  void importType(clang::QualType qual_type, clang::ASTContext &from_ctx,
                  clang::ASTContext &to_ctx,
                  llvm::Optional<swift::ClangTypeKind> kind,
                  llvm::SmallVectorImpl<clang::Decl *> &results) {
    clang::ASTImporter importer(to_ctx,
                                to_ctx.getSourceManager().getFileManager(),
                                from_ctx,
                                from_ctx.getSourceManager().getFileManager(),
                                false);
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

public:
  SwiftDWARFImporterDelegate(TypeSystemSwiftTypeRef &ts)
      : m_swift_typesystem(ts),
        m_description(ts.GetDescription() + "::SwiftDWARFImporterDelegate") {}

  /// Look up a clang::Decl by name.
  ///
  /// There are two primary ways that this delegate method is called:
  ///
  ///    1. When resolving a type from a mangled name. In this case \p
  ///       kind will be known, but the owning module of a Clang type
  ///       in a mangled name is always __ObjC or __C.
  ///
  ///    2. When resolving a type from a serialized module
  ///       cross reference. In this case \c kind will be unspecified,
  ///       but the (top-level) module that the type is defined in
  ///       will be known.
  ///
  /// The following diagram shows how the various components
  /// interact. All paths lead to a call to the function
  /// \c ClangImporter::Implementation::importDeclReal(), which turns
  /// a \c clang::Decl into a \c swift::Decl.  The return paths leading
  /// back from \c importDeclReal() are omitted from the diagram. Also
  /// some auxiliary intermediate function calls are be omitted for
  /// brevity.
  ///
  /// \verbatim
  /// ╔═LLDB═════════════════════════════════════════════════════════════════╗
  /// ║                                                                      ║
  /// ║  ┌─DWARFASTParserSwift──────────┐   ┌─DWARFImporterDelegate──────┐   ║
  /// ║  │                              │   │                            │   ║
  /// ║  │ GetTypeFromMangledTypename() │ ┌─├→lookupValue()─────┐        │   ║
  /// ║  │             │                │ │ │                   │        │   ║
  /// ║  └─────────────┬────────────────┘ │ └───────────────────┬────────┘   ║
  /// ║                │                  │                     │            ║
  /// ╚════════════════╤══════════════════╧═════════════════════╤════════════╝
  ///                  │                  │                     │
  /// ╔═Swift Compiler═╤══════════════════╧═════════════════════╤════════════╗
  /// ║                │                  │                     │            ║
  /// ║  ┌─ASTDemangler┬─────────────┐    │ ┌─ClangImporter─────┬────┐       ║
  /// ║  │             ↓             │    │ │                   │    │       ║
  /// ║  │ findForeignTypeDecl()─────├──────├→lookupTypeDecl()  │    │       ║
  /// ║  │                           │    │ │      ⇣            ↓    │       ║
  /// ║  └───────────────────────────┘    └─┤─lookupTypeDeclDWARF()  │       ║
  /// ║                                     │      ↓                 │       ║
  /// ║                                     │ *importDeclReal()*     │       ║
  /// ║                                     │      ↑                 │       ║
  /// ║                                     │ lookupValueDWARF()     │       ║
  /// ║                                     │      ↑                 │       ║
  /// ║                                     └──────┴─────────────────┘       ║
  /// ║                                            │                         ║
  /// ║  ┌─Deserialization─────────┐               └──────────────────────┐  ║
  /// ║  │ loadAllMembers()        │                                      │  ║
  /// ║  │        ↓                │  ┌─ModuleDecl────┐ ┌─DWARFModuleUnit─┴┐ ║
  /// ║  │ resolveCrossReference()─├──├→lookupValue()─├─├→lookupValue()───┘│ ║
  /// ║  │                         │  └───────────────┘ └──────────────────┘ ║
  /// ║  └─────────────────────────┘                                         ║
  /// ╚══════════════════════════════════════════════════════════════════════╝
  /// \endverbatim
  void lookupValue(StringRef name, llvm::Optional<swift::ClangTypeKind> kind,
                   StringRef inModule,
                   llvm::SmallVectorImpl<clang::Decl *> &results) override {
    LLDB_SCOPED_TIMER();
    LLDB_LOG(GetLog(LLDBLog::Types), "{0}::lookupValue(\"{1}\")", m_description,
             name.str());

    // We will not find any Swift types in the Clang compile units.
    ConstString name_cs(name);
    if (SwiftLanguageRuntime::IsSwiftMangledName(name_cs.GetStringRef()))
      return;

    auto *swift_ast_ctx = m_swift_typesystem.GetSwiftASTContext();
    if (!swift_ast_ctx)
      return;
    auto clang_importer = swift_ast_ctx->GetClangImporter();
    if (!clang_importer)
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
        clang_type_sp = swift_ts->GetTypeSystemSwiftTypeRef().LookupClangType(name);
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
    CompilerType compiler_type = clang_type_sp->GetFullCompilerType();

    // Filter our non-Clang types.
    auto type_system =
        compiler_type.GetTypeSystem().dyn_cast_or_null<TypeSystemClang>();
    if (!type_system)
      return;

    // Import the type into the DWARFImporter's context.
    clang::ASTContext &to_ctx = clang_importer->getClangASTContext();
    clang::ASTContext &from_ctx = type_system->getASTContext();

    clang::QualType qual_type = ClangUtil::GetQualType(compiler_type);
    importType(qual_type, from_ctx, to_ctx, kind, results);

    LLDB_LOG(GetLog(LLDBLog::Types),
             "{0}::lookupValue() -- imported {1} types from debug info.",
             m_description.c_str(), results.size());
  }
};

swift::DWARFImporterDelegate *
CreateSwiftDWARFImporterDelegate(TypeSystemSwiftTypeRef &ts) {
  return new SwiftDWARFImporterDelegate(ts);
}

} // namespace lldb_private
