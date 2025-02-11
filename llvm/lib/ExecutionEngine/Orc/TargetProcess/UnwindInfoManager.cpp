//===------- UnwindInfoManager.cpp - Register unwind info sections --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/UnwindInfoManager.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"

#ifdef __APPLE__
#include <dlfcn.h>
#endif // __APPLE__

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

static orc::shared::CWrapperFunctionResult
llvm_orc_rt_alt_UnwindInfoManager_register(const char *Data, uint64_t Size) {
  using SPSSig = SPSError(SPSSequence<SPSExecutorAddrRange>, SPSExecutorAddr,
                          SPSExecutorAddrRange, SPSExecutorAddrRange);

  return WrapperFunction<SPSSig>::handle(
             Data, Size,
             [](std::vector<ExecutorAddrRange> CodeRanges, ExecutorAddr DSOBase,
                ExecutorAddrRange DWARFRange,
                ExecutorAddrRange CompactUnwindRange) {
               return UnwindInfoManager::registerSections(
                   CodeRanges, DSOBase, DWARFRange, CompactUnwindRange);
             })
      .release();
}

static orc::shared::CWrapperFunctionResult
llvm_orc_rt_alt_UnwindInfoManager_deregister(const char *Data, uint64_t Size) {
  using SPSSig = SPSError(SPSSequence<SPSExecutorAddrRange>);

  return WrapperFunction<SPSSig>::handle(
             Data, Size,
             [](std::vector<ExecutorAddrRange> CodeRanges) {
               return UnwindInfoManager::deregisterSections(CodeRanges);
             })
      .release();
}

namespace llvm::orc {

static const char *AddFnName = "__unw_add_find_dynamic_unwind_sections";
[[maybe_unused]] static const char *RemoveFnName = "__unw_remove_find_dynamic_unwind_sections";
static std::unique_ptr<UnwindInfoManager> Instance;
static int (*RemoveFindDynamicUnwindSections)(void *) = nullptr;

UnwindInfoManager::~UnwindInfoManager() {
  if (int Err = RemoveFindDynamicUnwindSections((void *)&findSections)) {
    LLVM_DEBUG({
      dbgs() << "Failed call to " << RemoveFnName << ": error = " << Err
             << "\n";
    });
  }
}

bool UnwindInfoManager::TryEnable() {
#ifdef __APPLE__
  static std::mutex M;
  std::lock_guard<std::mutex> Lock(M);

  if (Instance)
    return true;

  auto AddFn = (int (*)(void *))dlsym(RTLD_DEFAULT, AddFnName);
  if (!AddFn)
    return false;

  auto RemoveFn = (int (*)(void *))dlsym(RTLD_DEFAULT, RemoveFnName);
  if (!RemoveFn)
    return false;

  Instance.reset(new UnwindInfoManager());

  if (auto Err = AddFn((void *)&findSections)) {
    LLVM_DEBUG({
      dbgs() << "Failed call to " << AddFnName << ": error = " << Err << "\n";
    });
    Instance = nullptr;
    return false;
  }

  RemoveFindDynamicUnwindSections = RemoveFn;
  return true;

#else
  return false;
#endif // __APPLE__
}

void UnwindInfoManager::addBootstrapSymbols(StringMap<ExecutorAddr> &M) {
  M[rt_alt::UnwindInfoManagerRegisterActionName] =
      ExecutorAddr::fromPtr(llvm_orc_rt_alt_UnwindInfoManager_register);
  M[rt_alt::UnwindInfoManagerDeregisterActionName] =
      ExecutorAddr::fromPtr(llvm_orc_rt_alt_UnwindInfoManager_deregister);
}

Error UnwindInfoManager::registerSections(
    ArrayRef<orc::ExecutorAddrRange> CodeRanges, orc::ExecutorAddr DSOBase,
    orc::ExecutorAddrRange DWARFEHFrame, orc::ExecutorAddrRange CompactUnwind) {
  return Instance->registerSectionsImpl(CodeRanges, DSOBase, DWARFEHFrame,
                                        CompactUnwind);
}

Error UnwindInfoManager::deregisterSections(
    ArrayRef<orc::ExecutorAddrRange> CodeRanges) {
  return Instance->deregisterSectionsImpl(CodeRanges);
}

int UnwindInfoManager::findSectionsImpl(uintptr_t Addr, UnwindSections *Info) {
  std::lock_guard<std::mutex> Lock(M);
  auto I = UWSecs.upper_bound(Addr);
  if (I == UWSecs.begin())
    return 0;
  --I;
  *Info = I->second;
  return 1;
}

int UnwindInfoManager::findSections(uintptr_t Addr, UnwindSections *Info) {
  return Instance->findSectionsImpl(Addr, Info);
}

Error UnwindInfoManager::registerSectionsImpl(
    ArrayRef<ExecutorAddrRange> CodeRanges, ExecutorAddr DSOBase,
    ExecutorAddrRange DWARFEHFrame, ExecutorAddrRange CompactUnwind) {
  std::lock_guard<std::mutex> Lock(M);
  for (auto &R : CodeRanges)
    UWSecs[R.Start.getValue()] =
        UnwindSections{static_cast<uintptr_t>(DSOBase.getValue()),
                       static_cast<uintptr_t>(DWARFEHFrame.Start.getValue()),
                       static_cast<size_t>(DWARFEHFrame.size()),
                       static_cast<uintptr_t>(CompactUnwind.Start.getValue()),
                       static_cast<size_t>(CompactUnwind.size())};
  return Error::success();
}

Error UnwindInfoManager::deregisterSectionsImpl(
    ArrayRef<ExecutorAddrRange> CodeRanges) {
  std::lock_guard<std::mutex> Lock(M);
  for (auto &R : CodeRanges) {
    auto I = UWSecs.find(R.Start.getValue());
    if (I == UWSecs.end())
      return make_error<StringError>(
          "No unwind-info sections registered for range " +
              formatv("{0:x} - {1:x}", R.Start, R.End),
          inconvertibleErrorCode());
    UWSecs.erase(I);
  }
  return Error::success();
}

} // namespace llvm::orc
