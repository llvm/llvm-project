//===---------- InProcessEPC.cpp -- In-process EPC for new ORC runtime ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/InProcessEPC.h"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/EPCGenericDylibManager.h"
#include "llvm/ExecutionEngine/Orc/EPCGenericJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/EPCGenericMemoryAccess.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/TargetExecutionUtils.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Process.h"

#define DEBUG_TYPE "orc"

namespace llvm::orc {

Expected<std::unique_ptr<InProcessEPC>>
InProcessEPC::Create(Connection *C, BootstrapInfoAccess *BIA,
                     std::shared_ptr<SymbolStringPool> SSP,
                     std::unique_ptr<TaskDispatcher> D) {
  assert(C && "C must not be null");
  assert(BIA && "BIA must not be null");

  // Lifecycle and IPCA-side fields must be populated by the controller side
  // before OnConnect is invoked.
  assert(C->Retain && "C->Retain not set by controller");
  assert(C->Release && "C->Release not set by controller");
  assert(C->Disconnect && "C->Disconnect not set by controller");
  assert(C->EnterMessageScope && "C->EnterMessageScope not set by controller");
  assert(C->LeaveMessageScope && "C->LeaveMessageScope not set by controller");
  assert(C->IPCA && "C->IPCA not set by controller");
  assert(C->CallWrapper && "C->CallWrapper not set by controller");
  assert(C->ReturnJITDispatchResult &&
         "C->ReturnJITDispatchResult not set by controller");

  if (!SSP)
    SSP = std::make_shared<SymbolStringPool>();

  if (!D)
    D = std::make_unique<InPlaceTaskDispatcher>();

  std::unique_ptr<InProcessEPC> IPEPC(
      new InProcessEPC(C, std::move(SSP), std::move(D)));

  // First set values in C.
  C->IPEPC = IPEPC.get();
  C->CallJITDispatch = callJITDispatchEntry;
  C->ReturnWrapperResult = returnWrapperResultEntry;

  // Then grab bootstrap values.
  if (auto PageSize = BIA->GetPageSize(BIA))
    IPEPC->PageSize = PageSize;
  else
    return make_error<StringError>(
        "Cannot create InProcessEPC with page-size = 0",
        inconvertibleErrorCode());

  if (auto TT = BIA->GetTargetTriple(BIA)) {
    IPEPC->TargetTriple = llvm::Triple(TT);
  } else
    return make_error<StringError>(
        "Cannot create InProcessEPC with target-triple = \"\"",
        inconvertibleErrorCode());

  {
    const char *Name;
    const char *ValBytes;
    uint64_t ValSize;
    int RC;
    while ((RC = BIA->GetNextValue(BIA, &Name, &ValBytes, &ValSize)) == 1) {
      if (!IPEPC->BootstrapMap
               .try_emplace(Name,
                            std::vector<char>(ValBytes, ValBytes + ValSize))
               .second)
        return make_error<StringError>(
            ("Cannot create InProcessEPC: bootstrap-value map contains "
             "duplicate key \"" +
             StringRef(Name) + "\""),
            inconvertibleErrorCode());
    }
    if (RC < 0)
      return make_error<StringError>(
          "Cannot create InProcessEPC: bootstrap-value map corrupted",
          inconvertibleErrorCode());
  }

  {
    const char *SymName;
    uint64_t SymAddr;
    int RC;
    while ((RC = BIA->GetNextSymbol(BIA, &SymName, &SymAddr)) == 1) {
      if (!IPEPC->BootstrapSymbols.try_emplace(SymName, ExecutorAddr(SymAddr))
               .second)
        return make_error<StringError>(
            ("Cannot create InProcessEPC: bootstrap-symbol map contains "
             "duplicate symbol \"" +
             StringRef(SymName) + "\""),
            inconvertibleErrorCode());
    }
    if (RC < 0)
      return make_error<StringError>(
          "Cannot create InProcessEPC: bootstrap-symbol map corrupted",
          inconvertibleErrorCode());
  }

  return std::move(IPEPC);
}

InProcessEPC::~InProcessEPC() {
  // Guarantee that a discarded InProcessEPC initiates disconnect, even if it
  // was never attached to an ExecutionSession (e.g. Create failed partway
  // through, or the caller dropped the returned object without handing it to
  // a session). When the InProcessEPC *is* attached, the ExecutionSession is
  // guaranteed to call disconnect() during shutdown, and this call becomes a
  // no-op via the idempotency of C->Disconnect.
  doDisconnect();

  // Shut down the dispatcher.
  D->shutdown();

  // Release the connection object.
  C->Release(C);
}

Expected<int32_t> InProcessEPC::runAsMain(ExecutorAddr MainFnAddr,
                                          ArrayRef<std::string> Args) {
  using MainTy = int (*)(int, char *[]);
  return orc::runAsMain(MainFnAddr.toPtr<MainTy>(), Args);
}

Expected<int32_t> InProcessEPC::runAsVoidFunction(ExecutorAddr VoidFnAddr) {
  using VoidTy = int (*)();
  return orc::runAsVoidFunction(VoidFnAddr.toPtr<VoidTy>());
}

Expected<int32_t> InProcessEPC::runAsIntFunction(ExecutorAddr IntFnAddr,
                                                 int Arg) {
  using IntTy = int (*)(int);
  return orc::runAsIntFunction(IntFnAddr.toPtr<IntTy>(), Arg);
}

void InProcessEPC::callWrapperAsync(ExecutorAddr WrapperFnAddr,
                                    IncomingWFRHandler OnComplete,
                                    ArrayRef<char> ArgBuffer) {
  if (C->EnterMessageScope(C)) {
    auto CallId = registerPendingCallWrapperResult(std::move(OnComplete));
    auto ArgBytes = shared::WrapperFunctionBuffer::copyFrom(ArgBuffer.data(),
                                                            ArgBuffer.size());

    LLVM_DEBUG(dbgs() << "InProcessEPC: callWrapperAsync call id " << CallId
                      << " to " << WrapperFnAddr << "\n");

    C->CallWrapper(C->IPCA, CallId, WrapperFnAddr.toPtr<void *>(),
                   ArgBytes.release());
    C->LeaveMessageScope(C);
  } else
    OnComplete(shared::WrapperFunctionBuffer::createOutOfBandError(
        "connection closed"));
}

Expected<std::unique_ptr<jitlink::JITLinkMemoryManager>>
InProcessEPC::createDefaultMemoryManager() {
  // FIXME: Should actually use InProcessMemoryManager for this.
  return EPCGenericJITLinkMemoryManager::Create(getExecutionSession());
}

Expected<std::unique_ptr<DylibManager>> InProcessEPC::createDefaultDylibMgr() {
  // FIXME: Should actually use in-process for this.
  auto DM = EPCGenericDylibManager::Create(getExecutionSession());
  if (!DM)
    return DM.takeError();
  return std::make_unique<EPCGenericDylibManager>(std::move(*DM));
}

Expected<std::unique_ptr<MemoryAccess>>
InProcessEPC::createDefaultMemoryAccess() {
  // FIXME: Should actually use in-process for this.
  EPCGenericMemoryAccess::FuncAddrs FAs;
  if (auto Err = getBootstrapSymbols(
          {{FAs.WriteUInt8s, rt::MemoryWriteUInt8sWrapperName},
           {FAs.WriteUInt16s, rt::MemoryWriteUInt16sWrapperName},
           {FAs.WriteUInt32s, rt::MemoryWriteUInt32sWrapperName},
           {FAs.WriteUInt64s, rt::MemoryWriteUInt64sWrapperName},
           {FAs.WriteBuffers, rt::MemoryWriteBuffersWrapperName},
           {FAs.WritePointers, rt::MemoryWritePointersWrapperName},
           {FAs.ReadUInt8s, rt::MemoryReadUInt8sWrapperName},
           {FAs.ReadUInt16s, rt::MemoryReadUInt16sWrapperName},
           {FAs.ReadUInt32s, rt::MemoryReadUInt32sWrapperName},
           {FAs.ReadUInt64s, rt::MemoryReadUInt64sWrapperName},
           {FAs.ReadBuffers, rt::MemoryReadBuffersWrapperName},
           {FAs.ReadStrings, rt::MemoryReadStringsWrapperName}}))
    return std::move(Err);

  return std::make_unique<EPCGenericMemoryAccess>(*this, FAs);
}

Error InProcessEPC::disconnect() {
  doDisconnect();
  return Error::success();
}

uint64_t InProcessEPC::registerPendingCallWrapperResult(IncomingWFRHandler H) {
  std::scoped_lock<std::mutex> Lock(M);
  assert(!PendingCallWrapperResults.count(NextCallId) &&
         "CallId already in use");
  PendingCallWrapperResults[NextCallId] = std::move(H);
  return NextCallId++;
}

void InProcessEPC::doDisconnect() {
  // Disconnect from InProcessControllerAccess. This should prevent any further
  // incoming or outgoing calls.
  C->Disconnect(C);

  // Drain any pending handlers.
  DenseMap<uint64_t, IncomingWFRHandler> HandlersToDrain;
  {
    std::scoped_lock<std::mutex> Lock(M);
    HandlersToDrain = std::move(PendingCallWrapperResults);
  }

  for (auto &[_, H] : HandlersToDrain)
    H(shared::WrapperFunctionBuffer::createOutOfBandError("disconnected"));
}

void InProcessEPC::callJITDispatch(uint64_t CallId, void *HandlerTag,
                                   shared::CWrapperFunctionBuffer ArgBytes) {
  assert(C->ReturnJITDispatchResult && "ReturnJITDispatchResult not set");

  LLVM_DEBUG(dbgs() << "InProcessEPC: JIT-dispatch call id " << CallId << " to "
                    << HandlerTag << "\n");

  getExecutionSession().runJITDispatchHandler(
      [this, CallId](shared::WrapperFunctionBuffer ResultBytes) {
        LLVM_DEBUG(dbgs() << "InProcessEPC: Returning JIT-dispatch result for "
                             "call id "
                          << CallId << "\n");
        if (C->EnterMessageScope(C)) {
          C->ReturnJITDispatchResult(C->IPCA, CallId, ResultBytes.release());
          C->LeaveMessageScope(C);
        }
      },
      ExecutorAddr::fromPtr(HandlerTag),
      shared::WrapperFunctionBuffer(ArgBytes));
}

void InProcessEPC::callJITDispatchEntry(
    void *IPEPC, uint64_t CallId, void *HandlerTag,
    shared::CWrapperFunctionBuffer ArgBytes) {
  static_cast<InProcessEPC *>(IPEPC)->callJITDispatch(CallId, HandlerTag,
                                                      ArgBytes);
}

void InProcessEPC::returnWrapperResult(
    uint64_t CallId, shared::CWrapperFunctionBuffer ResultBytes) {

  LLVM_DEBUG(dbgs() << "InProcessEPC: Wrapper result for call id " << CallId
                    << "\n");

  IncomingWFRHandler H;
  {
    std::scoped_lock<std::mutex> Lock(M);
    auto I = PendingCallWrapperResults.find(CallId);
    if (I != PendingCallWrapperResults.end()) {
      H = std::move(I->second);
      PendingCallWrapperResults.erase(I);
    }
  }

  if (!H) {
    getExecutionSession().reportError(make_error<StringError>(
        "InProcessEPC received result for invalid call id " + Twine(CallId),
        inconvertibleErrorCode()));
    return;
  }

  H(shared::WrapperFunctionBuffer(ResultBytes));
}

void InProcessEPC::returnWrapperResultEntry(
    void *IPEPC, uint64_t CallId, shared::CWrapperFunctionBuffer ResultBytes) {
  static_cast<InProcessEPC *>(IPEPC)->returnWrapperResult(CallId, ResultBytes);
}

} // namespace llvm::orc
