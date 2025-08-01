//===-- ThreadSafeModule.cpp - Thread safe Module, Context, and Utilities -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace llvm {
namespace orc {

ThreadSafeModule cloneToContext(const ThreadSafeModule &TSM,
                                ThreadSafeContext TSCtx,
                                GVPredicate ShouldCloneDef,
                                GVModifier UpdateClonedDefSource) {
  assert(TSM && "Can not clone null module");

  if (!ShouldCloneDef)
    ShouldCloneDef = [](const GlobalValue &) { return true; };

  // First copy the source module into a buffer.
  std::string ModuleName;
  SmallVector<char, 1> ClonedModuleBuffer;
  TSM.withModuleDo([&](Module &M) {
    ModuleName = M.getModuleIdentifier();
    std::set<GlobalValue *> ClonedDefsInSrc;
    ValueToValueMapTy VMap;
    auto Tmp = CloneModule(M, VMap, [&](const GlobalValue *GV) {
      if (ShouldCloneDef(*GV)) {
        ClonedDefsInSrc.insert(const_cast<GlobalValue *>(GV));
        return true;
      }
      return false;
    });

    if (UpdateClonedDefSource)
      for (auto *GV : ClonedDefsInSrc)
        UpdateClonedDefSource(*GV);

    BitcodeWriter BCWriter(ClonedModuleBuffer);
    BCWriter.writeModule(*Tmp);
    BCWriter.writeSymtab();
    BCWriter.writeStrtab();
  });

  MemoryBufferRef ClonedModuleBufferRef(
      StringRef(ClonedModuleBuffer.data(), ClonedModuleBuffer.size()),
      "cloned module buffer");

  // Then parse the buffer into the new Module.
  auto M = TSCtx.withContextDo([&](LLVMContext *Ctx) {
    assert(Ctx && "No LLVMContext provided");
    auto TmpM = cantFail(parseBitcodeFile(ClonedModuleBufferRef, *Ctx));
    TmpM->setModuleIdentifier(ModuleName);
    return TmpM;
  });

  return ThreadSafeModule(std::move(M), std::move(TSCtx));
}

ThreadSafeModule cloneToNewContext(const ThreadSafeModule &TSM,
                                   GVPredicate ShouldCloneDef,
                                   GVModifier UpdateClonedDefSource) {
  assert(TSM && "Can not clone null module");

  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());
  return cloneToContext(TSM, std::move(TSCtx), std::move(ShouldCloneDef),
                        std::move(UpdateClonedDefSource));
}

} // end namespace orc
} // end namespace llvm
