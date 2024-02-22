//===-- SwiftDWARFImporterForClangTypes.h -------------------*- C++ -*-----===//
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

//#include "TypeSystemSwiftTypeRef.h"
//#include "SwiftASTContext.h"
// 
#include "swift/ClangImporter/ClangImporter.h"
//#include "clang/AST/DeclObjC.h"
//#include "clang/Basic/SourceManager.h"
//#include "llvm/ADT/StringRef.h"
// 
//#include "Plugins/ExpressionParser/Clang/ClangASTImporter.h"
//#include "Plugins/ExpressionParser/Clang/ClangUtil.h"
//#include "Plugins/LanguageRuntime/Swift/SwiftLanguageRuntime.h"
#include "Plugins/TypeSystem/Clang/TypeSystemClang.h"
//#include "lldb/Core/Module.h"
//#include "lldb/Symbol/TypeMap.h"
//#include "lldb/Utility/LLDBLog.h"
//#include "lldb/Utility/Timer.h"

using namespace lldb;
using llvm::StringRef;

namespace lldb_private {
class TypeSystemSwiftTypeRef;
class SwiftASTContext;

/// Owned by a TypeSystemSwiftTypeRef. Imports Clang types from DWARF.
class SwiftDWARFImporterForClangTypes {
  TypeSystemSwiftTypeRef &m_swift_typesystem;
  using ModuleAndName = std::pair<const char *, const char *>;
  std::string m_description;

public:
  SwiftDWARFImporterForClangTypes(TypeSystemSwiftTypeRef &ts);
  void lookupValue(StringRef name, std::optional<swift::ClangTypeKind> kind,
                   StringRef inModule,
                   llvm::SmallVectorImpl<CompilerType> &results);
};

/// Implements a swift::DWARFImporterDelegate to look up Clang types in
/// DWARF.
///
/// During compile time, ClangImporter-imported Clang modules are compiled
/// with -gmodules, which emits a DWARF rendition of all types defined in
/// the module into the .pcm file. On Darwin, these types can be collected
/// by dsymutil. This delegate allows DWARFImporter to ask LLDB to look up a
/// Clang type by name, synthesize a Clang AST from it. DWARFImporter then
/// hands this Clang AST to ClangImporter to import the type into Swift.
class SwiftDWARFImporterDelegate : public swift::DWARFImporterDelegate {
  SwiftASTContext &m_swift_ast_ctx;
  SwiftDWARFImporterForClangTypes &m_importer;
  std::string m_description;

  /// Import \p qual_type from one clang ASTContext to another and
  /// add it to \p results if successful.
  void importType(clang::QualType qual_type, clang::ASTContext &from_ctx,
                  clang::ASTContext &to_ctx,
                  std::optional<swift::ClangTypeKind> kind,
                  llvm::SmallVectorImpl<clang::Decl *> &results);

  clang::Decl *GetDeclForTypeAndKind(clang::QualType qual_type,
                                     swift::ClangTypeKind kind);

public:
  SwiftDWARFImporterDelegate(SwiftASTContext &ts);

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
  void lookupValue(StringRef name, std::optional<swift::ClangTypeKind> kind,
                   StringRef inModule,
                   llvm::SmallVectorImpl<clang::Decl *> &results) override;
};

} // namespace lldb_private
