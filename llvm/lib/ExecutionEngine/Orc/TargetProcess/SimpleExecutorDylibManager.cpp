//===--- SimpleExecutorDylibManager.cpp - Executor-side dylib management --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/SimpleExecutorDylibManager.h"

#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"

#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <future>

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {
namespace rt_bootstrap {

SimpleExecutorDylibManager::~SimpleExecutorDylibManager() {
  assert(Dylibs.empty() && "shutdown not called?");
}

Expected<tpctypes::DylibHandle>
SimpleExecutorDylibManager::open(const std::string &Path, uint64_t Mode) {
  if (Mode != 0)
    return make_error<StringError>("open: non-zero mode bits not yet supported",
                                   inconvertibleErrorCode());

  const char *PathCStr = Path.empty() ? nullptr : Path.c_str();
  std::string ErrMsg;

  auto DL = sys::DynamicLibrary::getPermanentLibrary(PathCStr, &ErrMsg);
  if (!DL.isValid())
    return make_error<StringError>(std::move(ErrMsg), inconvertibleErrorCode());

  std::lock_guard<std::mutex> Lock(M);
  auto H = ExecutorAddr::fromPtr(DL.getOSSpecificHandle());
  Resolvers.push_back(std::make_unique<DylibSymbolResolver>(H));
  Dylibs.insert(DL.getOSSpecificHandle());
  return ExecutorAddr::fromPtr(Resolvers.back().get());
}

Error SimpleExecutorDylibManager::shutdown() {

  DylibSet DS;
  {
    std::lock_guard<std::mutex> Lock(M);
    std::swap(DS, Dylibs);
  }

  // There is no removal of dylibs at the moment, so nothing to do here.
  return Error::success();
}

void SimpleExecutorDylibManager::addBootstrapSymbols(
    StringMap<ExecutorAddr> &M) {
  M[rt::SimpleExecutorDylibManagerInstanceName] = ExecutorAddr::fromPtr(this);
  M[rt::SimpleExecutorDylibManagerOpenWrapperName] =
      ExecutorAddr::fromPtr(&openWrapper);
  M[rt::SimpleExecutorDylibManagerResolveWrapperName] =
      ExecutorAddr::fromPtr(&resolveWrapper);
}

llvm::orc::shared::CWrapperFunctionResult
SimpleExecutorDylibManager::openWrapper(const char *ArgData, size_t ArgSize) {
  return shared::
      WrapperFunction<rt::SPSSimpleExecutorDylibManagerOpenSignature>::handle(
             ArgData, ArgSize,
             shared::makeMethodWrapperHandler(
                 &SimpleExecutorDylibManager::open))
          .release();
}

llvm::orc::shared::CWrapperFunctionResult
SimpleExecutorDylibManager::resolveWrapper(const char *ArgData,
                                           size_t ArgSize) {
  using ResolveResult = ExecutorResolver::ResolveResult;
  return shared::WrapperFunction<
             rt::SPSSimpleExecutorDylibManagerResolveSignature>::
      handle(ArgData, ArgSize,
             [](ExecutorAddr Obj, RemoteSymbolLookupSet L) -> ResolveResult {
               using TmpResult =
                   MSVCPExpected<std::vector<std::optional<ExecutorSymbolDef>>>;
               std::promise<TmpResult> P;
               auto F = P.get_future();
               Obj.toPtr<ExecutorResolver *>()->resolveAsync(
                   std::move(L),
                   [&](TmpResult R) { P.set_value(std::move(R)); });
               return F.get();
             })
          .release();
}

} // namespace rt_bootstrap
} // end namespace orc
} // end namespace llvm
