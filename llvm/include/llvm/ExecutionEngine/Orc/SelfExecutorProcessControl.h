//===-- SelfExecutorProcessControl.h - EPC for in-process JITs --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Executor process control implementation for JITs that run JIT'd code in the
// same process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_SELFEXECUTORPROCESSCONTROL_H
#define LLVM_EXECUTIONENGINE_ORC_SELFEXECUTORPROCESSCONTROL_H

#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/InProcessMemoryAccess.h"

#include <memory>

namespace llvm::orc {

/// A ExecutorProcessControl implementation targeting the current process.
class LLVM_ABI SelfExecutorProcessControl : public ExecutorProcessControl,
                                            private DylibManager {
public:
  SelfExecutorProcessControl(
      std::shared_ptr<SymbolStringPool> SSP, std::unique_ptr<TaskDispatcher> D,
      Triple TargetTriple, unsigned PageSize,
      std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr);

  /// Create a SelfExecutorProcessControl with the given symbol string pool and
  /// memory manager.
  /// If no symbol string pool is given then one will be created.
  /// If no memory manager is given a jitlink::InProcessMemoryManager will
  /// be created and used by default.
  static Expected<std::unique_ptr<SelfExecutorProcessControl>>
  Create(std::shared_ptr<SymbolStringPool> SSP = nullptr,
         std::unique_ptr<TaskDispatcher> D = nullptr,
         std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr = nullptr);

  Expected<int32_t> runAsMain(ExecutorAddr MainFnAddr,
                              ArrayRef<std::string> Args) override;

  Expected<int32_t> runAsVoidFunction(ExecutorAddr VoidFnAddr) override;

  Expected<int32_t> runAsIntFunction(ExecutorAddr IntFnAddr, int Arg) override;

  void callWrapperAsync(ExecutorAddr WrapperFnAddr,
                        IncomingWFRHandler OnComplete,
                        ArrayRef<char> ArgBuffer) override;

  Error disconnect() override;

private:
  static shared::CWrapperFunctionResult
  jitDispatchViaWrapperFunctionManager(void *Ctx, const void *FnTag,
                                       const char *Data, size_t Size);

  Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) override;

  void lookupSymbolsAsync(ArrayRef<LookupRequest> Request,
                          SymbolLookupCompleteFn F) override;

  std::unique_ptr<jitlink::JITLinkMemoryManager> OwnedMemMgr;
#ifdef __APPLE__
  std::unique_ptr<UnwindInfoManager> UnwindInfoMgr;
#endif // __APPLE__
  char GlobalManglingPrefix = 0;
  InProcessMemoryAccess IPMA;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_SELFEXECUTORPROCESSCONTROL_H
