//===--LLDBExplicitModuleLoader.h -------------------------------*- C++-*-===//
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

#ifndef liblldb_LLDBExplicitModuleLoader_h_
#define liblldb_LLDBExplicitModuleLoader_h_

#include <swift/Serialization/SerializedModuleLoader.h>
namespace swift {
class ExplicitSwiftModuleLoader;
class ExplicitCASModuleLoader;
} // namespace swift

namespace lldb_private {
/// This Swift module loader implementation wraps a CAS module loader
/// and an ESML and forwards calls to addExplicitModulePath to both of
/// them.
class LLDBExplicitSwiftModuleLoader : public swift::SerializedModuleLoaderBase {
public:
  LLDBExplicitSwiftModuleLoader(
      swift::ASTContext &ctx, llvm::cas::ObjectStore *CAS,
      swift::DependencyTracker *tracker, swift::ModuleLoadingMode LoadMode,
      bool IgnoreSwiftSourceInfoFile,
      std::unique_ptr<swift::ExplicitCASModuleLoader> casml,
      std::unique_ptr<swift::ExplicitSwiftModuleLoader> esml);

  static std::unique_ptr<LLDBExplicitSwiftModuleLoader>
  create(swift::ASTContext &ctx, llvm::cas::ObjectStore *CAS,
         llvm::cas::ActionCache *cache, swift::DependencyTracker *tracker,
         swift::ModuleLoadingMode loadMode,
         llvm::StringRef ExplicitSwiftModuleMap,
         const llvm::StringMap<std::string> &ExplicitSwiftModuleInputs,
         bool IgnoreSwiftSourceInfoFile);

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
  void addExplicitModulePath(llvm::StringRef name, std::string path) override;

protected:
  llvm::cas::ObjectStore *m_cas;
  /// The CAS Swift module loader.
  std::unique_ptr<swift::ExplicitCASModuleLoader> m_casml;
  /// The explicit Swift module loader (ESML).
  std::unique_ptr<swift::ExplicitSwiftModuleLoader> m_esml;
};
} // namespace lldb_private
#endif
