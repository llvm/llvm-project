//===------- SimpleRemoteEPC.cpp -- Simple remote executor control --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SimpleRemoteEPC.h"
#include "llvm/ExecutionEngine/Orc/CallProxiesSPS.h"
#include "llvm/ExecutionEngine/Orc/EPCGenericDylibManagerSPS.h"
#include "llvm/ExecutionEngine/Orc/EPCGenericJITLinkMemoryManagerSPS.h"
#include "llvm/ExecutionEngine/Orc/EPCGenericMemoryAccessSPS.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

SimpleRemoteEPC::~SimpleRemoteEPC() {
#ifndef NDEBUG
  std::lock_guard<std::mutex> Lock(SimpleRemoteEPCMutex);
  assert(Disconnected && "Destroyed without disconnection");
#endif // NDEBUG
}

Expected<int32_t> SimpleRemoteEPC::runAsMain(ExecutorAddr MainFnAddr,
                                             ArrayRef<std::string> Args) {
  int64_t Result = 0;
  if (auto Err = callSPSWrapper<rt::sps_ci::CallMain::SPSSig>(
          RunAsMainAddr, Result, MainFnAddr, Args))
    return std::move(Err);
  return Result;
}

void SimpleRemoteEPC::callWrapperAsync(ExecutorAddr WrapperFnAddr,
                                       IncomingWFRHandler OnComplete,
                                       ArrayRef<char> ArgBuffer) {
  uint64_t SeqNo;
  {
    std::lock_guard<std::mutex> Lock(SimpleRemoteEPCMutex);
    SeqNo = getNextSeqNo();
    assert(!PendingCallWrapperResults.count(SeqNo) && "SeqNo already in use");
    PendingCallWrapperResults[SeqNo] = std::move(OnComplete);
  }

  if (auto Err = sendMessage(SimpleRemoteEPCOpcode::CallWrapper, SeqNo,
                             WrapperFnAddr, ArgBuffer)) {
    IncomingWFRHandler H;

    // We just registered OnComplete, but there may be a race between this
    // thread returning from sendMessage and handleDisconnect being called from
    // the transport's listener thread. If handleDisconnect gets there first
    // then it will have failed 'H' for us. If we get there first (or if
    // handleDisconnect already ran) then we need to take care of it.
    {
      std::lock_guard<std::mutex> Lock(SimpleRemoteEPCMutex);
      auto I = PendingCallWrapperResults.find(SeqNo);
      if (I != PendingCallWrapperResults.end()) {
        H = std::move(I->second);
        PendingCallWrapperResults.erase(I);
      }
    }

    if (H)
      H(shared::WrapperFunctionBuffer::createOutOfBandError("disconnecting"));

    getExecutionSession().reportError(std::move(Err));
  }
}

Expected<std::unique_ptr<jitlink::JITLinkMemoryManager>>
SimpleRemoteEPC::createDefaultMemoryManager() {
  return sps::createEPCGenericJITLinkMemoryManager(getExecutionSession());
}

Expected<std::unique_ptr<DylibManager>>
SimpleRemoteEPC::createDefaultDylibMgr() {
  return sps::createEPCGenericDylibManager(getExecutionSession());
}

Expected<std::unique_ptr<MemoryAccess>>
SimpleRemoteEPC::createDefaultMemoryAccess() {
  return sps::createEPCGenericMemoryAccess(getExecutionSession());
}

Error SimpleRemoteEPC::disconnect() {
  // disconnect is idempotent, so the first caller owns the hangup. There is
  // also nothing to announce to an executor that has already announced its own
  // departure.
  bool SendHangup = false;
  {
    std::lock_guard<std::mutex> Lock(SimpleRemoteEPCMutex);
    SendHangup = !LocalHangup && !RemoteHangup;
    LocalHangup = true;
  }

  // Tell the executor we're going away, so that it can distinguish this from
  // losing us unexpectedly. Best-effort: if the send fails there is nothing to
  // do but tear down anyway, and the executor will report the disconnection as
  // unexpected. A locally requested disconnect is orderly, so the hangup
  // carries a success value.
  if (SendHangup) {
    auto Payload = encodeHangupPayload(Error::success());
    if (auto Err = sendMessage(SimpleRemoteEPCOpcode::Hangup, 0, ExecutorAddr(),
                               {Payload.data(), Payload.size()}))
      consumeError(std::move(Err));
  }

  T->disconnect();
  D->shutdown();
  std::unique_lock<std::mutex> Lock(SimpleRemoteEPCMutex);
  DisconnectCV.wait(Lock, [this] { return Disconnected; });
  return std::move(DisconnectErr);
}

Expected<SimpleRemoteEPCTransportClient::HandleMessageAction>
SimpleRemoteEPC::handleMessage(SimpleRemoteEPCOpcode OpC, uint64_t SeqNo,
                               ExecutorAddr TagAddr,
                               shared::WrapperFunctionBuffer ArgBytes) {

  LLVM_DEBUG({
    dbgs() << "SimpleRemoteEPC::handleMessage: opc = ";
    switch (OpC) {
    case SimpleRemoteEPCOpcode::Setup:
      dbgs() << "Setup";
      assert(SeqNo == 0 && "Non-zero SeqNo for Setup?");
      assert(!TagAddr && "Non-zero TagAddr for Setup?");
      break;
    case SimpleRemoteEPCOpcode::Hangup:
      dbgs() << "Hangup";
      assert(SeqNo == 0 && "Non-zero SeqNo for Hangup?");
      assert(!TagAddr && "Non-zero TagAddr for Hangup?");
      break;
    case SimpleRemoteEPCOpcode::Result:
      dbgs() << "Result";
      assert(!TagAddr && "Non-zero TagAddr for Result?");
      break;
    case SimpleRemoteEPCOpcode::CallWrapper:
      dbgs() << "CallWrapper";
      break;
    }
    dbgs() << ", seqno = " << SeqNo << ", tag-addr = " << TagAddr
           << ", arg-buffer = " << formatv("{0:x}", ArgBytes.size())
           << " bytes\n";
  });

  using UT = std::underlying_type_t<SimpleRemoteEPCOpcode>;
  if (static_cast<UT>(OpC) > static_cast<UT>(SimpleRemoteEPCOpcode::LastOpC))
    return make_error<StringError>("Unexpected opcode",
                                   inconvertibleErrorCode());

  switch (OpC) {
  case SimpleRemoteEPCOpcode::Setup:
    if (auto Err = handleSetup(SeqNo, TagAddr, std::move(ArgBytes)))
      return std::move(Err);
    break;
  case SimpleRemoteEPCOpcode::Hangup:
    T->disconnect();
    {
      std::lock_guard<std::mutex> Lock(SimpleRemoteEPCMutex);
      RemoteHangup = true;
    }
    if (auto Err = handleHangup(std::move(ArgBytes)))
      return std::move(Err);
    return EndSession;
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

void SimpleRemoteEPC::handleDisconnect(Error Err) {
  LLVM_DEBUG({
    dbgs() << "SimpleRemoteEPC::handleDisconnect: "
           << (Err ? "failure" : "success") << "\n";
  });

  PendingCallWrapperResultsMap TmpPending;

  {
    std::lock_guard<std::mutex> Lock(SimpleRemoteEPCMutex);
    std::swap(TmpPending, PendingCallWrapperResults);
  }

  for (auto &KV : TmpPending)
    KV.second(
        shared::WrapperFunctionBuffer::createOutOfBandError("disconnecting"));

  std::lock_guard<std::mutex> Lock(SimpleRemoteEPCMutex);

  // If the transport reported no error, but neither side announced the end of
  // the session, then the executor went away without telling us. The cause is
  // not knowable from here -- it may have crashed, been killed, or become
  // unreachable -- so report what was observed rather than a cause.
  //
  // A missing hangup is evidence, not proof: a hangup can also be lost in
  // transit, since closing a TCP socket with unread data queued sends an RST,
  // which can discard bytes the peer had already delivered. We accept that
  // rather than draining the read side before closing -- the cost is a
  // misleading diagnostic on a session that is ending regardless, whereas a
  // drain risks stalling teardown on a peer that never closes.
  Error DisconnectReason =
      (!Err && !LocalHangup && !RemoteHangup)
          ? make_error<StringError>("Connection closed without hangup",
                                    inconvertibleErrorCode())
          : std::move(Err);

  DisconnectErr =
      joinErrors(std::move(DisconnectErr), std::move(DisconnectReason));
  Disconnected = true;
  DisconnectCV.notify_all();
}

Error SimpleRemoteEPC::sendMessage(SimpleRemoteEPCOpcode OpC, uint64_t SeqNo,
                                   ExecutorAddr TagAddr,
                                   ArrayRef<char> ArgBytes) {
  assert(OpC != SimpleRemoteEPCOpcode::Setup &&
         "SimpleRemoteEPC sending Setup message? That's the wrong direction.");

  LLVM_DEBUG({
    dbgs() << "SimpleRemoteEPC::sendMessage: opc = ";
    switch (OpC) {
    case SimpleRemoteEPCOpcode::Hangup:
      dbgs() << "Hangup";
      assert(SeqNo == 0 && "Non-zero SeqNo for Hangup?");
      assert(!TagAddr && "Non-zero TagAddr for Hangup?");
      break;
    case SimpleRemoteEPCOpcode::Result:
      dbgs() << "Result";
      assert(!TagAddr && "Non-zero TagAddr for Result?");
      break;
    case SimpleRemoteEPCOpcode::CallWrapper:
      dbgs() << "CallWrapper";
      break;
    default:
      llvm_unreachable("Invalid opcode");
    }
    dbgs() << ", seqno = " << SeqNo << ", tag-addr = " << TagAddr
           << ", arg-buffer = " << formatv("{0:x}", ArgBytes.size())
           << " bytes\n";
  });
  auto Err = T->sendMessage(OpC, SeqNo, TagAddr, ArgBytes);
  LLVM_DEBUG({
    if (Err)
      dbgs() << "  \\--> SimpleRemoteEPC::sendMessage failed\n";
  });
  return Err;
}

Error SimpleRemoteEPC::handleSetup(uint64_t SeqNo, ExecutorAddr TagAddr,
                                   shared::WrapperFunctionBuffer ArgBytes) {
  if (SeqNo != 0)
    return make_error<StringError>("Setup packet SeqNo not zero",
                                   inconvertibleErrorCode());

  if (TagAddr)
    return make_error<StringError>("Setup packet TagAddr not zero",
                                   inconvertibleErrorCode());

  std::lock_guard<std::mutex> Lock(SimpleRemoteEPCMutex);
  auto I = PendingCallWrapperResults.find(0);
  assert(PendingCallWrapperResults.size() == 1 &&
         I != PendingCallWrapperResults.end() &&
         "Setup message handler not connectly set up");
  auto SetupMsgHandler = std::move(I->second);
  PendingCallWrapperResults.erase(I);

  auto WFR =
      shared::WrapperFunctionBuffer::copyFrom(ArgBytes.data(), ArgBytes.size());
  SetupMsgHandler(std::move(WFR));
  return Error::success();
}

Error SimpleRemoteEPC::setup() {
  using namespace SimpleRemoteEPCDefaultBootstrapSymbolNames;

  std::promise<MSVCPExpected<SimpleRemoteEPCExecutorInfo>> EIP;
  auto EIF = EIP.get_future();

  // Prepare a handler for the setup packet.
  PendingCallWrapperResults[0] =
    RunInPlace()(
      [&](shared::WrapperFunctionBuffer SetupMsgBytes) {
        if (const char *ErrMsg = SetupMsgBytes.getOutOfBandError()) {
          EIP.set_value(
              make_error<StringError>(ErrMsg, inconvertibleErrorCode()));
          return;
        }
        using SPSSerialize =
            shared::SPSArgList<shared::SPSSimpleRemoteEPCExecutorInfo>;
        shared::SPSInputBuffer IB(SetupMsgBytes.data(), SetupMsgBytes.size());
        SimpleRemoteEPCExecutorInfo EI;
        if (SPSSerialize::deserialize(IB, EI))
          EIP.set_value(EI);
        else
          EIP.set_value(make_error<StringError>(
              "Could not deserialize setup message", inconvertibleErrorCode()));
      });

  // Start the transport.
  if (auto Err = T->start())
    return Err;

  // Wait for setup packet to arrive.
  auto EI = EIF.get();
  if (!EI) {
    T->disconnect();
    return EI.takeError();
  }

  LLVM_DEBUG({
    dbgs() << "SimpleRemoteEPC received setup message:\n"
           << "  Triple: " << EI->TargetTriple << "\n"
           << "  Page size: " << EI->PageSize << "\n"
           << "  Bootstrap map" << (EI->BootstrapMap.empty() ? " empty" : ":")
           << "\n";
    for (const auto &KV : EI->BootstrapMap)
      dbgs() << "    " << KV.first() << ": " << KV.second.size()
             << "-byte SPS encoded buffer\n";
    dbgs() << "  Bootstrap symbols"
           << (EI->BootstrapSymbols.empty() ? " empty" : ":") << "\n";
    for (const auto &KV : EI->BootstrapSymbols)
      dbgs() << "    " << KV.first() << ": " << KV.second << "\n";
  });
  TargetTriple = Triple(EI->TargetTriple);
  PageSize = EI->PageSize;
  BootstrapMap = std::move(EI->BootstrapMap);
  BootstrapSymbols = std::move(EI->BootstrapSymbols);

  BootstrapSymbols[rt::DispatchName] = BootstrapSymbols[DispatchFnName];
  BootstrapSymbols[rt::DispatchCtxName] =
      BootstrapSymbols[ExecutorSessionObjectName];

  if (auto Err =
          getBootstrapSymbols({{RunAsMainAddr, rt::sps_ci::CallMain::Name}}))
    return Err;

  return Error::success();
}

Error SimpleRemoteEPC::handleResult(uint64_t SeqNo, ExecutorAddr TagAddr,
                                    shared::WrapperFunctionBuffer ArgBytes) {
  IncomingWFRHandler SendResult;

  if (TagAddr)
    return make_error<StringError>("Unexpected TagAddr in result message",
                                   inconvertibleErrorCode());

  {
    std::lock_guard<std::mutex> Lock(SimpleRemoteEPCMutex);
    auto I = PendingCallWrapperResults.find(SeqNo);
    if (I == PendingCallWrapperResults.end())
      return make_error<StringError>("No call for sequence number " +
                                         Twine(SeqNo),
                                     inconvertibleErrorCode());
    SendResult = std::move(I->second);
    PendingCallWrapperResults.erase(I);
    releaseSeqNo(SeqNo);
  }

  auto WFR =
      shared::WrapperFunctionBuffer::copyFrom(ArgBytes.data(), ArgBytes.size());
  SendResult(std::move(WFR));
  return Error::success();
}

void SimpleRemoteEPC::handleCallWrapper(
    uint64_t RemoteSeqNo, ExecutorAddr TagAddr,
    shared::WrapperFunctionBuffer ArgBytes) {
  assert(ES && "No ExecutionSession attached");
  D->dispatch(makeGenericNamedTask(
      [this, RemoteSeqNo, TagAddr, ArgBytes = std::move(ArgBytes)]() mutable {
        ES->runJITDispatchHandler(
            [this, RemoteSeqNo](shared::WrapperFunctionBuffer WFR) {
              if (auto Err =
                      sendMessage(SimpleRemoteEPCOpcode::Result, RemoteSeqNo,
                                  ExecutorAddr(), {WFR.data(), WFR.size()}))
                getExecutionSession().reportError(std::move(Err));
            },
            TagAddr, std::move(ArgBytes));
      },
      "callWrapper task"));
}

Error SimpleRemoteEPC::handleHangup(shared::WrapperFunctionBuffer ArgBytes) {
  return decodeHangupPayload(std::move(ArgBytes));
}

} // end namespace orc
} // end namespace llvm
