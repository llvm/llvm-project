//===------- SimpleEPCServer.cpp - EPC over simple abstract channel -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/SimpleRemoteEPCServer.h"

#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Process.h"

#include "OrcRTBootstrap.h"

#define DEBUG_TYPE "orc"

using namespace llvm::orc::shared;

namespace llvm {
namespace orc {

SimpleRemoteEPCServer::Dispatcher::~Dispatcher() {}

#if LLVM_ENABLE_THREADS
void SimpleRemoteEPCServer::ThreadDispatcher::dispatch(
    unique_function<void()> Work) {
  {
    std::lock_guard<std::mutex> Lock(DispatchMutex);
    if (!Running)
      return;
    ++Outstanding;
  }

  std::thread([this, Work = std::move(Work)]() mutable {
    Work();
    std::lock_guard<std::mutex> Lock(DispatchMutex);
    --Outstanding;
    OutstandingCV.notify_all();
  }).detach();
}

void SimpleRemoteEPCServer::ThreadDispatcher::shutdown() {
  std::unique_lock<std::mutex> Lock(DispatchMutex);
  Running = false;
  OutstandingCV.wait(Lock, [this]() { return Outstanding == 0; });
}
#endif

StringMap<ExecutorAddress> SimpleRemoteEPCServer::defaultBootstrapSymbols() {
  StringMap<ExecutorAddress> DBS;
  rt_bootstrap::addTo(DBS);
  DBS["__llvm_orc_load_dylib"] = ExecutorAddress::fromPtr(&loadDylibWrapper);
  DBS["__llvm_orc_lookup_symbols"] =
      ExecutorAddress::fromPtr(&lookupSymbolsWrapper);
  return DBS;
}

Expected<SimpleRemoteEPCTransportClient::HandleMessageAction>
SimpleRemoteEPCServer::handleMessage(SimpleRemoteEPCOpcode OpC, uint64_t SeqNo,
                                     ExecutorAddress TagAddr,
                                     SimpleRemoteEPCArgBytesVector ArgBytes) {
  using UT = std::underlying_type_t<SimpleRemoteEPCOpcode>;
  if (static_cast<UT>(OpC) > static_cast<UT>(SimpleRemoteEPCOpcode::LastOpC))
    return make_error<StringError>("Unexpected opcode",
                                   inconvertibleErrorCode());

  // TODO: Clean detach message?
  switch (OpC) {
  case SimpleRemoteEPCOpcode::Setup:
    return make_error<StringError>("Unexpected Setup opcode",
                                   inconvertibleErrorCode());
  case SimpleRemoteEPCOpcode::Hangup:
    return SimpleRemoteEPCTransportClient::EndSession;
  case SimpleRemoteEPCOpcode::Result:
    if (auto Err = handleResult(SeqNo, TagAddr, std::move(ArgBytes)))
      return std::move(Err);
    break;
  case SimpleRemoteEPCOpcode::CallWrapper:
    handleCallWrapper(SeqNo, TagAddr, std::move(ArgBytes));
    break;
  }
  return ContinueSession;
}

Error SimpleRemoteEPCServer::waitForDisconnect() {
  std::unique_lock<std::mutex> Lock(ServerStateMutex);
  ShutdownCV.wait(Lock, [this]() { return RunState == ServerShutDown; });
  return std::move(ShutdownErr);
}

void SimpleRemoteEPCServer::handleDisconnect(Error Err) {
  PendingJITDispatchResultsMap TmpPending;

  {
    std::lock_guard<std::mutex> Lock(ServerStateMutex);
    std::swap(TmpPending, PendingJITDispatchResults);
    RunState = ServerShuttingDown;
  }

  // Send out-of-band errors to any waiting threads.
  for (auto &KV : TmpPending)
    KV.second->set_value(
        shared::WrapperFunctionResult::createOutOfBandError("disconnecting"));

  // TODO: Free attached resources.
  // 1. Close libraries in DylibHandles.

  // Wait for dispatcher to clear.
  D->shutdown();

  std::lock_guard<std::mutex> Lock(ServerStateMutex);
  ShutdownErr = joinErrors(std::move(ShutdownErr), std::move(Err));
  RunState = ServerShutDown;
  ShutdownCV.notify_all();
}

Error SimpleRemoteEPCServer::sendSetupMessage(
    StringMap<ExecutorAddress> BootstrapSymbols) {

  using namespace SimpleRemoteEPCDefaultBootstrapSymbolNames;

  std::vector<char> SetupPacket;
  SimpleRemoteEPCExecutorInfo EI;
  EI.TargetTriple = sys::getProcessTriple();
  if (auto PageSize = sys::Process::getPageSize())
    EI.PageSize = *PageSize;
  else
    return PageSize.takeError();
  EI.BootstrapSymbols = std::move(BootstrapSymbols);

  assert(!EI.BootstrapSymbols.count(ExecutorSessionObjectName) &&
         "Dispatch context name should not be set");
  assert(!EI.BootstrapSymbols.count(DispatchFnName) &&
         "Dispatch function name should not be set");
  EI.BootstrapSymbols[ExecutorSessionObjectName] =
      ExecutorAddress::fromPtr(this);
  EI.BootstrapSymbols[DispatchFnName] =
      ExecutorAddress::fromPtr(jitDispatchEntry);

  using SPSSerialize =
      shared::SPSArgList<shared::SPSSimpleRemoteEPCExecutorInfo>;
  auto SetupPacketBytes =
      shared::WrapperFunctionResult::allocate(SPSSerialize::size(EI));
  shared::SPSOutputBuffer OB(SetupPacketBytes.data(), SetupPacketBytes.size());
  if (!SPSSerialize::serialize(OB, EI))
    return make_error<StringError>("Could not send setup packet",
                                   inconvertibleErrorCode());

  return T->sendMessage(SimpleRemoteEPCOpcode::Setup, 0, ExecutorAddress(),
                        {SetupPacketBytes.data(), SetupPacketBytes.size()});
}

Error SimpleRemoteEPCServer::handleResult(
    uint64_t SeqNo, ExecutorAddress TagAddr,
    SimpleRemoteEPCArgBytesVector ArgBytes) {
  std::promise<shared::WrapperFunctionResult> *P = nullptr;
  {
    std::lock_guard<std::mutex> Lock(ServerStateMutex);
    auto I = PendingJITDispatchResults.find(SeqNo);
    if (I == PendingJITDispatchResults.end())
      return make_error<StringError>("No call for sequence number " +
                                         Twine(SeqNo),
                                     inconvertibleErrorCode());
    P = I->second;
    PendingJITDispatchResults.erase(I);
    releaseSeqNo(SeqNo);
  }
  auto R = shared::WrapperFunctionResult::allocate(ArgBytes.size());
  memcpy(R.data(), ArgBytes.data(), ArgBytes.size());
  P->set_value(std::move(R));
  return Error::success();
}

void SimpleRemoteEPCServer::handleCallWrapper(
    uint64_t RemoteSeqNo, ExecutorAddress TagAddr,
    SimpleRemoteEPCArgBytesVector ArgBytes) {
  D->dispatch([this, RemoteSeqNo, TagAddr, ArgBytes = std::move(ArgBytes)]() {
    using WrapperFnTy =
        shared::detail::CWrapperFunctionResult (*)(const char *, size_t);
    auto *Fn = TagAddr.toPtr<WrapperFnTy>();
    shared::WrapperFunctionResult ResultBytes(
        Fn(ArgBytes.data(), ArgBytes.size()));
    if (auto Err = T->sendMessage(SimpleRemoteEPCOpcode::Result, RemoteSeqNo,
                                  ExecutorAddress(),
                                  {ResultBytes.data(), ResultBytes.size()}))
      ReportError(std::move(Err));
  });
}

shared::detail::CWrapperFunctionResult
SimpleRemoteEPCServer::loadDylibWrapper(const char *ArgData, size_t ArgSize) {
  return shared::WrapperFunction<shared::SPSLoadDylibSignature>::handle(
             ArgData, ArgSize,
             [](ExecutorAddress ExecutorSessionObj, std::string Path,
                uint64_t Flags) -> Expected<uint64_t> {
               return ExecutorSessionObj.toPtr<SimpleRemoteEPCServer *>()
                   ->loadDylib(Path, Flags);
             })
      .release();
}

shared::detail::CWrapperFunctionResult
SimpleRemoteEPCServer::lookupSymbolsWrapper(const char *ArgData,
                                            size_t ArgSize) {
  return shared::WrapperFunction<shared::SPSLookupSymbolsSignature>::handle(
             ArgData, ArgSize,
             [](ExecutorAddress ExecutorSessionObj,
                std::vector<RemoteSymbolLookup> Lookup) {
               return ExecutorSessionObj.toPtr<SimpleRemoteEPCServer *>()
                   ->lookupSymbols(Lookup);
             })
      .release();
}

Expected<tpctypes::DylibHandle>
SimpleRemoteEPCServer::loadDylib(const std::string &Path, uint64_t Mode) {
  std::string ErrMsg;
  const char *P = Path.empty() ? nullptr : Path.c_str();
  auto DL = sys::DynamicLibrary::getPermanentLibrary(P, &ErrMsg);
  if (!DL.isValid())
    return make_error<StringError>(std::move(ErrMsg), inconvertibleErrorCode());
  std::lock_guard<std::mutex> Lock(ServerStateMutex);
  uint64_t Id = Dylibs.size();
  Dylibs.push_back(std::move(DL));
  return Id;
}

Expected<std::vector<std::vector<ExecutorAddress>>>
SimpleRemoteEPCServer::lookupSymbols(const std::vector<RemoteSymbolLookup> &L) {
  std::vector<std::vector<ExecutorAddress>> Result;

  for (const auto &E : L) {
    if (E.H >= Dylibs.size())
      return make_error<StringError>("Unrecognized handle",
                                     inconvertibleErrorCode());
    auto &DL = Dylibs[E.H];
    Result.push_back({});

    for (const auto &Sym : E.Symbols) {

      const char *DemangledSymName = Sym.Name.c_str();
#ifdef __APPLE__
      if (*DemangledSymName == '_')
        ++DemangledSymName;
#endif

      void *Addr = DL.getAddressOfSymbol(DemangledSymName);
      if (!Addr && Sym.Required)
        return make_error<StringError>(Twine("Missing definition for ") +
                                           DemangledSymName,
                                       inconvertibleErrorCode());

      Result.back().push_back(ExecutorAddress::fromPtr(Addr));
    }
  }

  return std::move(Result);
}

shared::WrapperFunctionResult
SimpleRemoteEPCServer::doJITDispatch(const void *FnTag, const char *ArgData,
                                     size_t ArgSize) {
  uint64_t SeqNo;
  std::promise<shared::WrapperFunctionResult> ResultP;
  auto ResultF = ResultP.get_future();
  {
    std::lock_guard<std::mutex> Lock(ServerStateMutex);
    if (RunState != ServerRunning)
      return shared::WrapperFunctionResult::createOutOfBandError(
          "jit_dispatch not available (EPC server shut down)");

    SeqNo = getNextSeqNo();
    assert(!PendingJITDispatchResults.count(SeqNo) && "SeqNo already in use");
    PendingJITDispatchResults[SeqNo] = &ResultP;
  }

  if (auto Err =
          T->sendMessage(SimpleRemoteEPCOpcode::CallWrapper, SeqNo,
                         ExecutorAddress::fromPtr(FnTag), {ArgData, ArgSize}))
    ReportError(std::move(Err));

  return ResultF.get();
}

shared::detail::CWrapperFunctionResult
SimpleRemoteEPCServer::jitDispatchEntry(void *DispatchCtx, const void *FnTag,
                                        const char *ArgData, size_t ArgSize) {
  return reinterpret_cast<SimpleRemoteEPCServer *>(DispatchCtx)
      ->doJITDispatch(FnTag, ArgData, ArgSize)
      .release();
}

} // end namespace orc
} // end namespace llvm
