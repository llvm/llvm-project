//===--LLDBImplicitModuleLoader.cpp --------------------------------------===//
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

#include "LLDBImplicitModuleLoader.h"
#include "SwiftASTContext.h"
#include "lldb/Target/Target.h"
#include "swift/Serialization/SerializedModuleLoader.h"

namespace lldb_private {

LLDBImplicitSwiftModuleLoader::LLDBImplicitSwiftModuleLoader(
    swift::ASTContext &ctx, swift::DependencyTracker *tracker,
    swift::ModuleLoadingMode loadMode,
    std::weak_ptr<SwiftASTContext> swift_ast_ctx_wp,
    std::unique_ptr<swift::ImplicitSerializedModuleLoader> isml)
    : swift::SerializedModuleLoaderBase(ctx, tracker, loadMode, true),
      m_swift_ast_ctx_wp(swift_ast_ctx_wp), m_isml(std::move(isml)) {}

std::unique_ptr<LLDBImplicitSwiftModuleLoader>
LLDBImplicitSwiftModuleLoader::create(
    swift::ASTContext &ctx, swift::DependencyTracker *tracker,
    swift::ModuleLoadingMode loadMode,
    std::weak_ptr<SwiftASTContext> swift_ast_ctx_wp) {
  auto isml =
      swift::ImplicitSerializedModuleLoader::create(ctx, tracker, loadMode);
  return std::make_unique<LLDBImplicitSwiftModuleLoader>(
      ctx, tracker, loadMode, swift_ast_ctx_wp, std::move(isml));
}

bool LLDBImplicitSwiftModuleLoader::enabled() const {
  if (Target::GetGlobalProperties().GetSwiftAllowImplicitModuleLoader())
    return true;
  if (auto swift_ast_ctx_sp = m_swift_ast_ctx_wp.lock())
    return !swift_ast_ctx_sp->ImplicitModulesDisabled();
  return false;
}

void LLDBImplicitSwiftModuleLoader::collectVisibleTopLevelModuleNames(
    llvm::SmallVectorImpl<swift::Identifier> &names) const {
  if (enabled())
    m_isml->collectVisibleTopLevelModuleNames(names);
}

std::error_code LLDBImplicitSwiftModuleLoader::findModuleFilesInDirectory(
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
bool LLDBImplicitSwiftModuleLoader::canImportModule(
    swift::ImportPath::Module named, swift::SourceLoc loc,
    ModuleVersionInfo *versionInfo, bool isTestableImport) {
  if (enabled())
    return llvm::cast<swift::SerializedModuleLoaderBase>(m_isml.get())
        ->canImportModule(named, loc, versionInfo, isTestableImport);
  return false;
}

swift::ModuleDecl *
LLDBImplicitSwiftModuleLoader::loadModule(swift::SourceLoc importLoc,
                                          swift::ImportPath::Module path,
                                          bool AllowMemoryCache) {
  if (enabled())
    return m_isml->loadModule(importLoc, path, AllowMemoryCache);
  return nullptr;
}

void LLDBImplicitSwiftModuleLoader::loadExtensions(
    swift::NominalTypeDecl *nominal, unsigned previousGeneration) {
  if (enabled())
    m_isml->loadExtensions(nominal, previousGeneration);
}

void LLDBImplicitSwiftModuleLoader::loadObjCMethods(
    swift::NominalTypeDecl *typeDecl, swift::ObjCSelector selector,
    bool isInstanceMethod, unsigned previousGeneration,
    llvm::TinyPtrVector<swift::AbstractFunctionDecl *> &methods) {
  if (enabled())
    m_isml->loadObjCMethods(typeDecl, selector, isInstanceMethod,
                            previousGeneration, methods);
}

void LLDBImplicitSwiftModuleLoader::loadDerivativeFunctionConfigurations(
    swift::AbstractFunctionDecl *originalAFD, unsigned previousGeneration,
    llvm::SetVector<swift::AutoDiffConfig> &results) {
  if (enabled())
    m_isml->loadDerivativeFunctionConfigurations(originalAFD,
                                                 previousGeneration, results);
}

void LLDBImplicitSwiftModuleLoader::verifyAllModules() {
  if (enabled())
    m_isml->verifyAllModules();
}

} // namespace lldb_private
