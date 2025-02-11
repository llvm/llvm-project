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
#include "llvm/Support/DynamicLibrary.h"

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

static orc::shared::CWrapperFunctionResult
llvm_orc_rt_alt_UnwindInfoManager_enable(const char *Data, uint64_t Size) {
  return WrapperFunction<SPSError(SPSExecutorAddr, SPSExecutorAddr)>::handle(
             Data, Size,
             [](ExecutorAddr Instance, ExecutorAddr FindFn) {
               return Instance.toPtr<UnwindInfoManager *>()->enable(
                   FindFn.toPtr<void *>());
             })
      .release();
}

static orc::shared::CWrapperFunctionResult
llvm_orc_rt_alt_UnwindInfoManager_disable(const char *Data, uint64_t Size) {
  return WrapperFunction<SPSError(SPSExecutorAddr)>::handle(
             Data, Size,
             [](ExecutorAddr Instance) {
               return Instance.toPtr<UnwindInfoManager *>()->disable();
             })
      .release();
}

static orc::shared::CWrapperFunctionResult
llvm_orc_rt_alt_UnwindInfoManager_register(const char *Data, uint64_t Size) {
  using SPSSig =
      SPSError(SPSExecutorAddr, SPSSequence<SPSExecutorAddrRange>,
               SPSExecutorAddr, SPSExecutorAddrRange, SPSExecutorAddrRange);

  return WrapperFunction<SPSSig>::handle(
             Data, Size,
             [](ExecutorAddr Instance,
                std::vector<ExecutorAddrRange> CodeRanges, ExecutorAddr DSOBase,
                ExecutorAddrRange DWARFRange,
                ExecutorAddrRange CompactUnwindRange) {
               return Instance.toPtr<UnwindInfoManager *>()->registerSections(
                   CodeRanges, DSOBase, DWARFRange, CompactUnwindRange);
             })
      .release();
}

static orc::shared::CWrapperFunctionResult
llvm_orc_rt_alt_UnwindInfoManager_deregister(const char *Data, uint64_t Size) {
  using SPSSig = SPSError(SPSExecutorAddr, SPSSequence<SPSExecutorAddrRange>);

  return WrapperFunction<SPSSig>::handle(
             Data, Size,
             [](ExecutorAddr Instance,
                std::vector<ExecutorAddrRange> CodeRanges) {
               return Instance.toPtr<UnwindInfoManager *>()->deregisterSections(
                   CodeRanges);
             })
      .release();
}

namespace llvm::orc {

const char *UnwindInfoManager::AddFnName =
    "__unw_add_find_dynamic_unwind_sections";
const char *UnwindInfoManager::RemoveFnName =
    "__unw_remove_find_dynamic_unwind_sections";

std::unique_ptr<UnwindInfoManager> UnwindInfoManager::TryCreate() {
  std::string ErrMsg;
  auto DL = sys::DynamicLibrary::getPermanentLibrary(nullptr, &ErrMsg);
  if (!DL.isValid())
    return nullptr;

  auto AddFindDynamicUnwindSections =
      (int (*)(void *))DL.getAddressOfSymbol(AddFnName);
  if (!AddFindDynamicUnwindSections)
    return nullptr;

  auto RemoveFindDynamicUnwindSections =
      (int (*)(void *))DL.getAddressOfSymbol(RemoveFnName);
  if (!RemoveFindDynamicUnwindSections)
    return nullptr;

  return std::unique_ptr<UnwindInfoManager>(new UnwindInfoManager(
      AddFindDynamicUnwindSections, RemoveFindDynamicUnwindSections));
}

Error UnwindInfoManager::shutdown() { return Error::success(); }

void UnwindInfoManager::addBootstrapSymbols(StringMap<ExecutorAddr> &M) {
  M[rt_alt::UnwindInfoManagerInstanceName] = ExecutorAddr::fromPtr(this);
  M[rt_alt::UnwindInfoManagerFindSectionsHelperName] =
      ExecutorAddr::fromPtr(&findSectionsHelper);
  M[rt_alt::UnwindInfoManagerEnableWrapperName] =
      ExecutorAddr::fromPtr(llvm_orc_rt_alt_UnwindInfoManager_enable);
  M[rt_alt::UnwindInfoManagerDisableWrapperName] =
      ExecutorAddr::fromPtr(llvm_orc_rt_alt_UnwindInfoManager_disable);
  M[rt_alt::UnwindInfoManagerRegisterActionName] =
      ExecutorAddr::fromPtr(llvm_orc_rt_alt_UnwindInfoManager_register);
  M[rt_alt::UnwindInfoManagerDeregisterActionName] =
      ExecutorAddr::fromPtr(llvm_orc_rt_alt_UnwindInfoManager_deregister);
}

Error UnwindInfoManager::enable(void *FindDynamicUnwindSections) {
  LLVM_DEBUG(dbgs() << "Enabling UnwindInfoManager.\n");

  if (auto Err = AddFindDynamicUnwindSections(FindDynamicUnwindSections))
    return make_error<StringError>(Twine("Could not register function via ") +
                                       AddFnName +
                                       ", error code = " + Twine(Err),
                                   inconvertibleErrorCode());

  this->FindDynamicUnwindSections = FindDynamicUnwindSections;
  return Error::success();
}

Error UnwindInfoManager::disable(void) {
  LLVM_DEBUG(dbgs() << "Disabling UnwindInfoManager.\n");

  if (FindDynamicUnwindSections)
    if (auto Err = RemoveFindDynamicUnwindSections(FindDynamicUnwindSections))
      return make_error<StringError>(
          Twine("Could not deregister function via ") + RemoveFnName +
              "error code = " + Twine(Err),
          inconvertibleErrorCode());

  FindDynamicUnwindSections = nullptr;
  return Error::success();
}

Error UnwindInfoManager::registerSections(
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

Error UnwindInfoManager::deregisterSections(
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

int UnwindInfoManager::findSections(uintptr_t Addr, UnwindSections *Info) {
  std::lock_guard<std::mutex> Lock(M);
  auto I = UWSecs.upper_bound(Addr);
  if (I == UWSecs.begin())
    return 0;
  --I;
  *Info = I->second;
  return 1;
}

int UnwindInfoManager::findSectionsHelper(UnwindInfoManager *Instance,
                                          uintptr_t Addr,
                                          UnwindSections *Info) {
  return Instance->findSections(Addr, Info);
}

} // namespace llvm::orc
