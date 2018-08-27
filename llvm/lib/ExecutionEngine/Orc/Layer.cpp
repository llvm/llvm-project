//===-------------------- Layer.cpp - Layer interfaces --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/Object/ObjectFile.h"

namespace llvm {
namespace orc {

IRLayer::IRLayer(ExecutionSession &ES) : ES(ES) {}
IRLayer::~IRLayer() {}

Error IRLayer::add(JITDylib &JD, VModuleKey K, std::unique_ptr<Module> M) {
  return JD.define(llvm::make_unique<BasicIRLayerMaterializationUnit>(
      *this, std::move(K), std::move(M)));
}

IRMaterializationUnit::IRMaterializationUnit(ExecutionSession &ES,
                                             std::unique_ptr<Module> M)
  : MaterializationUnit(SymbolFlagsMap()), M(std::move(M)) {

  MangleAndInterner Mangle(ES, this->M->getDataLayout());
  for (auto &G : this->M->global_values()) {
    if (G.hasName() && !G.isDeclaration() && !G.hasLocalLinkage() &&
        !G.hasAvailableExternallyLinkage() && !G.hasAppendingLinkage()) {
      auto MangledName = Mangle(G.getName());
      SymbolFlags[MangledName] = JITSymbolFlags::fromGlobalValue(G);
      SymbolToDefinition[MangledName] = &G;
    }
  }
}

IRMaterializationUnit::IRMaterializationUnit(
    std::unique_ptr<Module> M, SymbolFlagsMap SymbolFlags,
    SymbolNameToDefinitionMap SymbolToDefinition)
    : MaterializationUnit(std::move(SymbolFlags)), M(std::move(M)),
      SymbolToDefinition(std::move(SymbolToDefinition)) {}

void IRMaterializationUnit::discard(const JITDylib &JD, SymbolStringPtr Name) {
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

Error ObjectLayer::add(JITDylib &JD, VModuleKey K,
                       std::unique_ptr<MemoryBuffer> O) {
  auto ObjMU = BasicObjectLayerMaterializationUnit::Create(*this, std::move(K),
                                                           std::move(O));
  if (!ObjMU)
    return ObjMU.takeError();
  return JD.define(std::move(*ObjMU));
}

Expected<std::unique_ptr<BasicObjectLayerMaterializationUnit>>
BasicObjectLayerMaterializationUnit::Create(ObjectLayer &L, VModuleKey K,
                                            std::unique_ptr<MemoryBuffer> O) {
  auto SymbolFlags =
      getObjectSymbolFlags(L.getExecutionSession(), O->getMemBufferRef());

  if (!SymbolFlags)
    return SymbolFlags.takeError();

  return std::unique_ptr<BasicObjectLayerMaterializationUnit>(
      new BasicObjectLayerMaterializationUnit(L, K, std::move(O),
                                              std::move(*SymbolFlags)));
}

BasicObjectLayerMaterializationUnit::BasicObjectLayerMaterializationUnit(
    ObjectLayer &L, VModuleKey K, std::unique_ptr<MemoryBuffer> O,
    SymbolFlagsMap SymbolFlags)
    : MaterializationUnit(std::move(SymbolFlags)), L(L), K(std::move(K)),
      O(std::move(O)) {}

void BasicObjectLayerMaterializationUnit::materialize(
    MaterializationResponsibility R) {
  L.emit(std::move(R), std::move(K), std::move(O));
}

void BasicObjectLayerMaterializationUnit::discard(const JITDylib &JD,
                                                  SymbolStringPtr Name) {
  // FIXME: Support object file level discard. This could be done by building a
  //        filter to pass to the object layer along with the object itself.
}

Expected<SymbolFlagsMap> getObjectSymbolFlags(ExecutionSession &ES,
                                              MemoryBufferRef ObjBuffer) {
  auto Obj = object::ObjectFile::createObjectFile(ObjBuffer);

  if (!Obj)
    return Obj.takeError();

  SymbolFlagsMap SymbolFlags;
  for (auto &Sym : (*Obj)->symbols()) {
    // Skip symbols not defined in this object file.
    if (Sym.getFlags() & object::BasicSymbolRef::SF_Undefined)
      continue;

    // Skip symbols that are not global.
    if (!(Sym.getFlags() & object::BasicSymbolRef::SF_Global))
      continue;

    auto Name = Sym.getName();
    if (!Name)
      return Name.takeError();
    auto InternedName = ES.getSymbolStringPool().intern(*Name);
    auto SymFlags = JITSymbolFlags::fromObjectSymbol(Sym);
    if (!SymFlags)
      return SymFlags.takeError();
    SymbolFlags[InternedName] = std::move(*SymFlags);
  }

  return SymbolFlags;
}

} // End namespace orc.
} // End namespace llvm.
