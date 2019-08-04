//===---------- speculation.cpp - Utilities for Speculation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Speculation.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"

#include <vector>

namespace llvm {

namespace orc {

// ImplSymbolMap methods
void ImplSymbolMap::trackImpls(SymbolAliasMap ImplMaps, JITDylib *SrcJD) {
  assert(SrcJD && "Tracking on Null Source .impl dylib");
  std::lock_guard<std::mutex> Lockit(ConcurrentAccess);
  for (auto &I : ImplMaps) {
    auto It = Maps.insert({I.first, {I.second.Aliasee, SrcJD}});
    // check rationale when independent dylibs have same symbol name?
    assert(It.second && "ImplSymbols are already tracked for this Symbol?");
    (void)(It);
  }
}

// If two modules, share the same LLVMContext, different threads must
// not access those modules concurrently, doing so leave the
// LLVMContext in in-consistent state.
// But here since each TSM has a unique Context associated with it,
// on locking is necessary!
void IRSpeculationLayer::emit(MaterializationResponsibility R,
                              ThreadSafeModule TSM) {

  assert(TSM && "Speculation Layer received Null Module ?");
  assert(TSM.getContext().getContext() != nullptr &&
         "Module with null LLVMContext?");

  // Instrumentation of runtime calls
  auto &InContext = *TSM.getContext().getContext();
  auto SpeculatorVTy = StructType::create(InContext, "Class.Speculator");
  auto RuntimeCallTy = FunctionType::get(
      Type::getVoidTy(InContext),
      {SpeculatorVTy->getPointerTo(), Type::getInt64Ty(InContext)}, false);
  auto RuntimeCall =
      Function::Create(RuntimeCallTy, Function::LinkageTypes::ExternalLinkage,
                       "__orc_speculate_for", TSM.getModuleUnlocked());
  auto SpeclAddr = new GlobalVariable(
      *TSM.getModuleUnlocked(), SpeculatorVTy, false,
      GlobalValue::LinkageTypes::ExternalLinkage, nullptr, "__orc_speculator");

  IRBuilder<> Mutator(InContext);

  // QueryAnalysis allowed to transform the IR source, one such example is
  // Simplify CFG helps the static branch prediction heuristics!
  for (auto &Fn : TSM.getModuleUnlocked()->getFunctionList()) {
    if (!Fn.isDeclaration()) {
      auto IRNames = QueryAnalysis(Fn, FAM);
      // Instrument and register if Query has result
      if (IRNames.hasValue()) {
        Mutator.SetInsertPoint(&(Fn.getEntryBlock().front()));
        auto ImplAddrToUint =
            Mutator.CreatePtrToInt(&Fn, Type::getInt64Ty(InContext));
        Mutator.CreateCall(RuntimeCallTy, RuntimeCall,
                           {SpeclAddr, ImplAddrToUint});
        S.registerSymbols(internToJITSymbols(IRNames.getValue()),
                          &R.getTargetJITDylib());
      }
    }
  }
  // No locking needed read only operation.
  assert(!(verifyModule(*TSM.getModuleUnlocked())) &&
         "Speculation Instrumentation breaks IR?");

  NextLayer.emit(std::move(R), std::move(TSM));
}

// Runtime Function Implementation
extern "C" void __orc_speculate_for(Speculator *Ptr, uint64_t StubId) {
  assert(Ptr && " Null Address Received in orc_speculate_for ");
  Ptr->speculateFor(StubId);
}

} // namespace orc
} // namespace llvm
