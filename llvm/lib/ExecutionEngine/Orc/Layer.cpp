//===-------------------- Layer.cpp - Layer interfaces --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {
namespace orc {

MangleAndInterner::MangleAndInterner(ExecutionSession &ES, const DataLayout &DL)
    : ES(ES), DL(DL) {}

SymbolStringPtr MangleAndInterner::operator()(StringRef Name) {
  std::string MangledName;
  {
    raw_string_ostream MangledNameStream(MangledName);
    Mangler::getNameWithPrefix(MangledNameStream, Name, DL);
  }
  return ES.getSymbolStringPool().intern(MangledName);
}

IRLayer::IRLayer(ExecutionSession &ES) : ES(ES) {}
IRLayer::~IRLayer() {}

Error IRLayer::add(VSO &V, VModuleKey K, std::unique_ptr<Module> M) {
  return V.define(llvm::make_unique<BasicIRLayerMaterializationUnit>(
      *this, std::move(K), std::move(M)));
}

IRMaterializationUnit::IRMaterializationUnit(ExecutionSession &ES,
                                             std::unique_ptr<Module> M)
  : MaterializationUnit(SymbolFlagsMap()), M(std::move(M)) {

  MangleAndInterner Mangle(ES, this->M->getDataLayout());
  for (auto &G : this->M->global_values()) {
    if (G.hasName() && !G.isDeclaration() &&
        !G.hasLocalLinkage() &&
        !G.hasAvailableExternallyLinkage()) {
      auto MangledName = Mangle(G.getName());
      SymbolFlags[MangledName] = JITSymbolFlags::fromGlobalValue(G);
      SymbolToDefinition[MangledName] = &G;
    }
  }
}

void IRMaterializationUnit::discard(const VSO &V, SymbolStringPtr Name) {
  auto I = SymbolToDefinition.find(Name);
  assert(I != SymbolToDefinition.end() &&
         "Symbol not provided by this MU, or previously discarded");
  assert(!I->second->isDeclaration() &&
         "Discard should only apply to definitions");
  I->second->setLinkage(GlobalValue::AvailableExternallyLinkage);
  SymbolToDefinition.erase(I);
}

BasicIRLayerMaterializationUnit::BasicIRLayerMaterializationUnit(
    IRLayer &L, VModuleKey K, std::unique_ptr<Module> M)
  : IRMaterializationUnit(L.getExecutionSession(), std::move(M)),
      L(L), K(std::move(K)) {}

void BasicIRLayerMaterializationUnit::materialize(
    MaterializationResponsibility R) {
  L.emit(std::move(R), std::move(K), std::move(M));
}

ObjectLayer::ObjectLayer(ExecutionSession &ES) : ES(ES) {}

ObjectLayer::~ObjectLayer() {}

Error ObjectLayer::add(VSO &V, VModuleKey K, std::unique_ptr<MemoryBuffer> O) {
  return V.define(llvm::make_unique<BasicObjectLayerMaterializationUnit>(
      *this, std::move(K), std::move(O)));
}

BasicObjectLayerMaterializationUnit::BasicObjectLayerMaterializationUnit(
    ObjectLayer &L, VModuleKey K, std::unique_ptr<MemoryBuffer> O)
    : MaterializationUnit(SymbolFlagsMap()), L(L), K(std::move(K)),
      O(std::move(O)) {

  auto &ES = L.getExecutionSession();
  auto Obj = cantFail(
      object::ObjectFile::createObjectFile(this->O->getMemBufferRef()));

  for (auto &Sym : Obj->symbols()) {
    if (!(Sym.getFlags() & object::BasicSymbolRef::SF_Undefined) &&
         (Sym.getFlags() & object::BasicSymbolRef::SF_Exported)) {
      auto InternedName =
          ES.getSymbolStringPool().intern(cantFail(Sym.getName()));
      SymbolFlags[InternedName] = JITSymbolFlags::fromObjectSymbol(Sym);
    }
  }
}

void BasicObjectLayerMaterializationUnit::materialize(
    MaterializationResponsibility R) {
  L.emit(std::move(R), std::move(K), std::move(O));
}

void BasicObjectLayerMaterializationUnit::discard(const VSO &V,
                                                  SymbolStringPtr Name) {
  // FIXME: Support object file level discard. This could be done by building a
  //        filter to pass to the object layer along with the object itself.
}

} // End namespace orc.
} // End namespace llvm.
