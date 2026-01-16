//===--LLDBExplicitModuleLoader.cpp --------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2025 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===---------------------------------------------------------------------===//

#include "LLDBExplicitModuleLoader.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include "swift/Serialization/SerializedModuleLoader.h"
#include <swift/Frontend/ModuleInterfaceLoader.h>

namespace lldb_private {

LLDBExplicitSwiftModuleLoader::LLDBExplicitSwiftModuleLoader(
    swift::ASTContext &ctx, llvm::cas::ObjectStore *CAS,
    swift::DependencyTracker *tracker, swift::ModuleLoadingMode loadMode,
    bool IgnoreSwiftSourceInfoFile,
    std::unique_ptr<swift::ExplicitCASModuleLoader> casml,
    std::unique_ptr<swift::ExplicitSwiftModuleLoader> esml)
    : swift::SerializedModuleLoaderBase(ctx, tracker, loadMode,
                                        IgnoreSwiftSourceInfoFile),
      m_cas(CAS), m_casml(std::move(casml)), m_esml(std::move(esml)) {}

std::unique_ptr<LLDBExplicitSwiftModuleLoader>
LLDBExplicitSwiftModuleLoader::create(
    swift::ASTContext &ctx, llvm::cas::ObjectStore *CAS,
    llvm::cas::ActionCache *cache, swift::DependencyTracker *tracker,
    swift::ModuleLoadingMode loadMode, llvm::StringRef ExplicitSwiftModuleMap,
    const llvm::StringMap<std::string> &ExplicitSwiftModuleInputs,
    bool IgnoreSwiftSourceInfoFile) {
  auto esml = swift::ExplicitSwiftModuleLoader::create(
      ctx, tracker, loadMode, ExplicitSwiftModuleMap, ExplicitSwiftModuleInputs,
      IgnoreSwiftSourceInfoFile);
  std::unique_ptr<swift::ExplicitCASModuleLoader> casml;
  if (CAS && cache) {
    casml = swift::ExplicitCASModuleLoader::create(
        ctx, *CAS, *cache, tracker, loadMode, ExplicitSwiftModuleMap,
        ExplicitSwiftModuleInputs, IgnoreSwiftSourceInfoFile);
  }
  return std::make_unique<LLDBExplicitSwiftModuleLoader>(
      ctx, CAS, tracker, loadMode, IgnoreSwiftSourceInfoFile, std::move(casml),
      std::move(esml));
}

void LLDBExplicitSwiftModuleLoader::collectVisibleTopLevelModuleNames(
    llvm::SmallVectorImpl<swift::Identifier> &names) const {
  if (m_casml)
    m_casml->collectVisibleTopLevelModuleNames(names);
  m_esml->collectVisibleTopLevelModuleNames(names);
}

std::error_code LLDBExplicitSwiftModuleLoader::findModuleFilesInDirectory(
    swift::ImportPath::Element ModuleID,
    const swift::SerializedModuleBaseName &BaseName,
    llvm::SmallVectorImpl<char> *ModuleInterfacePath,
    llvm::SmallVectorImpl<char> *ModuleInterfaceSourcePath,
    std::unique_ptr<llvm::MemoryBuffer> *ModuleBuffer,
    std::unique_ptr<llvm::MemoryBuffer> *ModuleDocBuffer,
    std::unique_ptr<llvm::MemoryBuffer> *ModuleSourceInfoBuffer,
    bool IsCanImportLookup, bool IsFramework, bool IsTestableDependencyLookup) {
  // This is a protected member and probably also not useful.
  return {};
}
bool LLDBExplicitSwiftModuleLoader::canImportModule(
    swift::ImportPath::Module named, swift::SourceLoc loc,
    ModuleVersionInfo *versionInfo, bool isTestableImport) {
  if (m_casml &&
      llvm::cast<swift::SerializedModuleLoaderBase>(m_casml.get())
          ->canImportModule(named, loc, versionInfo, isTestableImport))
    return true;
  return llvm::cast<swift::SerializedModuleLoaderBase>(m_esml.get())
      ->canImportModule(named, loc, versionInfo, isTestableImport);
}

swift::ModuleDecl *
LLDBExplicitSwiftModuleLoader::loadModule(swift::SourceLoc importLoc,
                                          swift::ImportPath::Module path,
                                          bool AllowMemoryCache) {
  if (m_casml)
    if (swift::ModuleDecl *decl =
            m_casml->loadModule(importLoc, path, AllowMemoryCache))
      return decl;
  return m_esml->loadModule(importLoc, path, AllowMemoryCache);
}

void LLDBExplicitSwiftModuleLoader::loadExtensions(
    swift::NominalTypeDecl *nominal, unsigned previousGeneration) {
  if (m_casml)
    m_casml->loadExtensions(nominal, previousGeneration);
  m_esml->loadExtensions(nominal, previousGeneration);
}

void LLDBExplicitSwiftModuleLoader::loadObjCMethods(
    swift::NominalTypeDecl *typeDecl, swift::ObjCSelector selector,
    bool isInstanceMethod, unsigned previousGeneration,
    llvm::TinyPtrVector<swift::AbstractFunctionDecl *> &methods) {
  if (m_casml)
    m_casml->loadObjCMethods(typeDecl, selector, isInstanceMethod,
                             previousGeneration, methods);
  m_esml->loadObjCMethods(typeDecl, selector, isInstanceMethod,
                          previousGeneration, methods);
}

void LLDBExplicitSwiftModuleLoader::loadDerivativeFunctionConfigurations(
    swift::AbstractFunctionDecl *originalAFD, unsigned previousGeneration,
    llvm::SetVector<swift::AutoDiffConfig> &results) {
  if (m_casml)
    m_casml->loadDerivativeFunctionConfigurations(originalAFD,
                                                  previousGeneration, results);
  m_esml->loadDerivativeFunctionConfigurations(originalAFD, previousGeneration,
                                               results);
}

void LLDBExplicitSwiftModuleLoader::verifyAllModules() {
  if (m_casml)
    m_casml->verifyAllModules();
  m_esml->verifyAllModules();
}

void LLDBExplicitSwiftModuleLoader::addExplicitModulePath(llvm::StringRef name,
                                                          std::string path) {
  if (m_cas && m_casml) {
    llvm::Expected<llvm::cas::CASID> parsed_id = m_cas->parseID(path);
    if (parsed_id) {
      LLDB_LOG(GetLog(LLDBLog::Types),
               "discovered explicitly tracked module \"{0}\" at CAS id \"{1}\"",
               name, path);
      return m_casml->addExplicitModulePath(name, path);
    }
    llvm::consumeError(parsed_id.takeError());
  }
  LLDB_LOG(GetLog(LLDBLog::Types),
           "discovered explicitly tracked module \"{0}\" at \"{1}\"", name,
           path);
  m_esml->addExplicitModulePath(name, path);
}

} // namespace lldb_private
