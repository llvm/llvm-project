//===-------------------- Layer.cpp - Layer interfaces --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/IR/Constants.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

IRLayer::~IRLayer() {}

Error IRLayer::add(JITDylib &JD, ThreadSafeModule TSM, VModuleKey K) {
  return JD.define(std::make_unique<BasicIRLayerMaterializationUnit>(
      *this, *getManglingOptions(), std::move(TSM), std::move(K)));
}

IRMaterializationUnit::IRMaterializationUnit(ExecutionSession &ES,
                                             const ManglingOptions &MO,
                                             ThreadSafeModule TSM, VModuleKey K)
    : MaterializationUnit(SymbolFlagsMap(), std::move(K)), TSM(std::move(TSM)) {

  assert(this->TSM && "Module must not be null");

  MangleAndInterner Mangle(ES, this->TSM.getModuleUnlocked()->getDataLayout());
  this->TSM.withModuleDo([&](Module &M) {
    for (auto &G : M.global_values()) {
      // Skip globals that don't generate symbols.
      if (!G.hasName() || G.isDeclaration() || G.hasLocalLinkage() ||
          G.hasAvailableExternallyLinkage() || G.hasAppendingLinkage())
        continue;

      // thread locals generate different symbols depending on whether or not
      // emulated TLS is enabled.
      if (G.isThreadLocal() && MO.EmulatedTLS) {
        auto &GV = cast<GlobalVariable>(G);

        auto Flags = JITSymbolFlags::fromGlobalValue(GV);

        auto EmuTLSV = Mangle(("__emutls_v." + GV.getName()).str());
        SymbolFlags[EmuTLSV] = Flags;
        SymbolToDefinition[EmuTLSV] = &GV;

        // If this GV has a non-zero initializer we'll need to emit an
        // __emutls.t symbol too.
        if (GV.hasInitializer()) {
          const auto *InitVal = GV.getInitializer();

          // Skip zero-initializers.
          if (isa<ConstantAggregateZero>(InitVal))
            continue;
          const auto *InitIntValue = dyn_cast<ConstantInt>(InitVal);
          if (InitIntValue && InitIntValue->isZero())
            continue;

          auto EmuTLST = Mangle(("__emutls_t." + GV.getName()).str());
          SymbolFlags[EmuTLST] = Flags;
        }
        continue;
      }

      // Otherwise we just need a normal linker mangling.
      auto MangledName = Mangle(G.getName());
      SymbolFlags[MangledName] = JITSymbolFlags::fromGlobalValue(G);
      SymbolToDefinition[MangledName] = &G;
    }
  });
}

IRMaterializationUnit::IRMaterializationUnit(
    ThreadSafeModule TSM, VModuleKey K, SymbolFlagsMap SymbolFlags,
    SymbolNameToDefinitionMap SymbolToDefinition)
    : MaterializationUnit(std::move(SymbolFlags), std::move(K)),
      TSM(std::move(TSM)), SymbolToDefinition(std::move(SymbolToDefinition)) {}

StringRef IRMaterializationUnit::getName() const {
  if (TSM)
    return TSM.withModuleDo(
        [](const Module &M) -> StringRef { return M.getModuleIdentifier(); });
  return "<null module>";
}

void IRMaterializationUnit::discard(const JITDylib &JD,
                                    const SymbolStringPtr &Name) {
  LLVM_DEBUG(JD.getExecutionSession().runSessionLocked([&]() {
    dbgs() << "In " << JD.getName() << " discarding " << *Name << " from MU@"
           << this << " (" << getName() << ")\n";
  }););

  auto I = SymbolToDefinition.find(Name);
  assert(I != SymbolToDefinition.end() &&
         "Symbol not provided by this MU, or previously discarded");
  assert(!I->second->isDeclaration() &&
         "Discard should only apply to definitions");
  I->second->setLinkage(GlobalValue::AvailableExternallyLinkage);
  SymbolToDefinition.erase(I);
}

BasicIRLayerMaterializationUnit::BasicIRLayerMaterializationUnit(
    IRLayer &L, const ManglingOptions &MO, ThreadSafeModule TSM, VModuleKey K)
    : IRMaterializationUnit(L.getExecutionSession(), MO, std::move(TSM),
                            std::move(K)),
      L(L), K(std::move(K)) {}

void BasicIRLayerMaterializationUnit::materialize(
    MaterializationResponsibility R) {

  // Throw away the SymbolToDefinition map: it's not usable after we hand
  // off the module.
  SymbolToDefinition.clear();

  // If cloneToNewContextOnEmit is set, clone the module now.
  if (L.getCloneToNewContextOnEmit())
    TSM = cloneToNewContext(TSM);

#ifndef NDEBUG
  auto &ES = R.getTargetJITDylib().getExecutionSession();
  auto &N = R.getTargetJITDylib().getName();
#endif // NDEBUG

  LLVM_DEBUG(ES.runSessionLocked(
      [&]() { dbgs() << "Emitting, for " << N << ", " << *this << "\n"; }););
  L.emit(std::move(R), std::move(TSM));
  LLVM_DEBUG(ES.runSessionLocked([&]() {
    dbgs() << "Finished emitting, for " << N << ", " << *this << "\n";
  }););
}

ObjectLayer::ObjectLayer(ExecutionSession &ES) : ES(ES) {}

ObjectLayer::~ObjectLayer() {}

Error ObjectLayer::add(JITDylib &JD, std::unique_ptr<MemoryBuffer> O,
                       VModuleKey K) {
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
    : MaterializationUnit(std::move(SymbolFlags), std::move(K)), L(L),
      O(std::move(O)) {}

StringRef BasicObjectLayerMaterializationUnit::getName() const {
  if (O)
    return O->getBufferIdentifier();
  return "<null object>";
}

void BasicObjectLayerMaterializationUnit::materialize(
    MaterializationResponsibility R) {
  L.emit(std::move(R), std::move(O));
}

void BasicObjectLayerMaterializationUnit::discard(const JITDylib &JD,
                                                  const SymbolStringPtr &Name) {
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
    auto InternedName = ES.intern(*Name);
    auto SymFlags = JITSymbolFlags::fromObjectSymbol(Sym);
    if (!SymFlags)
      return SymFlags.takeError();
    SymbolFlags[InternedName] = std::move(*SymFlags);
  }

  return SymbolFlags;
}

} // End namespace orc.
} // End namespace llvm.
