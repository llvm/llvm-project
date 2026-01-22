//===--LLDBImplicitModuleLoader.h -----------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2025 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_LLDBImplicitModuleLoader_h_
#define liblldb_LLDBImplicitModuleLoader_h_

#include "SwiftASTContext.h"
#include <swift/Serialization/SerializedModuleLoader.h>
namespace swift {
class ImplicitSerializedModuleLoader;
} // namespace swift

namespace lldb_private {
/// This is a wrapper around the ImplicitSwiftSerializedModuleLoader
/// that can be selectively disabled.
class LLDBImplicitSwiftModuleLoader : public swift::SerializedModuleLoaderBase {
public:
  LLDBImplicitSwiftModuleLoader(
      swift::ASTContext &ctx, swift::DependencyTracker *tracker,
      swift::ModuleLoadingMode LoadMode,
      std::weak_ptr<SwiftASTContext> swift_ast_ctx_wp,
      std::unique_ptr<swift::ImplicitSerializedModuleLoader> isml);

  static std::unique_ptr<LLDBImplicitSwiftModuleLoader>
  create(swift::ASTContext &ctx, swift::DependencyTracker *tracker,
         swift::ModuleLoadingMode loadMode,
         std::weak_ptr<SwiftASTContext> swift_ast_ctx_wp);

  void collectVisibleTopLevelModuleNames(
      llvm::SmallVectorImpl<swift::Identifier> &names) const override;
  std::error_code findModuleFilesInDirectory(
      swift::ImportPath::Element ModuleID,
      const swift::SerializedModuleBaseName &BaseName,
      llvm::SmallVectorImpl<char> *ModuleInterfacePath,
      llvm::SmallVectorImpl<char> *ModuleInterfaceSourcePath,
      std::unique_ptr<llvm::MemoryBuffer> *ModuleBuffer,
      std::unique_ptr<llvm::MemoryBuffer> *ModuleDocBuffer,
      std::unique_ptr<llvm::MemoryBuffer> *ModuleSourceInfoBuffer,
      bool IsCanImportLookup, bool IsFramework,
      bool IsTestableDependencyLookup = false) override;

  bool canImportModule(swift::ImportPath::Module named, swift::SourceLoc loc,
                       ModuleVersionInfo *versionInfo,
                       bool isTestableImport = false) override;

  swift::ModuleDecl *loadModule(swift::SourceLoc importLoc,
                                swift::ImportPath::Module path,
                                bool AllowMemoryCache = true) override;
  void loadExtensions(swift::NominalTypeDecl *nominal,
                      unsigned previousGeneration) override;
  void loadObjCMethods(
      swift::NominalTypeDecl *typeDecl, swift::ObjCSelector selector,
      bool isInstanceMethod, unsigned previousGeneration,
      llvm::TinyPtrVector<swift::AbstractFunctionDecl *> &methods) override;

  void loadDerivativeFunctionConfigurations(
      swift::AbstractFunctionDecl *originalAFD, unsigned previousGeneration,
      llvm::SetVector<swift::AutoDiffConfig> &results) override;

  void verifyAllModules() override;

protected:
  bool enabled() const;
  std::weak_ptr<SwiftASTContext> m_swift_ast_ctx_wp;
  /// The implicit Swift module loader.
  std::unique_ptr<swift::ImplicitSerializedModuleLoader> m_isml;
};
} // namespace lldb_private
#endif
