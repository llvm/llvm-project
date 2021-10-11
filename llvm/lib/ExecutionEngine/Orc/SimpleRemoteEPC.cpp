//===------- SimpleRemoteEPC.cpp -- Simple remote executor control --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/SimpleRemoteEPC.h"
#include "llvm/ExecutionEngine/Orc/EPCGenericJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/EPCGenericMemoryAccess.h"
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

Expected<tpctypes::DylibHandle>
SimpleRemoteEPC::loadDylib(const char *DylibPath) {
  return DylibMgr->open(DylibPath, 0);
}

Expected<std::vector<tpctypes::LookupResult>>
SimpleRemoteEPC::lookupSymbols(ArrayRef<LookupRequest> Request) {
  std::vector<tpctypes::LookupResult> Result;

  for (auto &Element : Request) {
    if (auto R = DylibMgr->lookup(Element.Handle, Element.Symbols)) {
      Result.push_back({});
      Result.back().reserve(R->size());
      for (auto Addr : *R)
        Result.back().push_back(Addr.getValue());
    } else
      return R.takeError();
  }
  return std::move(Result);
}

Expected<int32_t> SimpleRemoteEPC::runAsMain(ExecutorAddr MainFnAddr,
                                             ArrayRef<std::string> Args) {
  int64_t Result = 0;
  if (auto Err = callSPSWrapper<rt::SPSRunAsMainSignature>(
          RunAsMainAddr, Result, ExecutorAddr(MainFnAddr), Args))
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
    getExecutionSession().reportError(std::move(Err));
  }
}

Error SimpleRemoteEPC::disconnect() {
  T->disconnect();
  std::unique_lock<std::mutex> Lock(SimpleRemoteEPCMutex);
  DisconnectCV.wait(Lock, [this] { return Disconnected; });
  return std::move(DisconnectErr);
}

Expected<SimpleRemoteEPCTransportClient::HandleMessageAction>
SimpleRemoteEPC::handleMessage(SimpleRemoteEPCOpcode OpC, uint64_t SeqNo,
                               ExecutorAddr TagAddr,
                               SimpleRemoteEPCArgBytesVector ArgBytes) {

  LLVM_DEBUG({
    dbgs() << "SimpleRemoteEPC::handleMessage: opc = ";
    switch (OpC) {
    case SimpleRemoteEPCOpcode::Setup:
      dbgs() << "Setup";
      assert(SeqNo == 0 && "Non-zero SeqNo for Setup?");
      assert(TagAddr.getValue() == 0 && "Non-zero TagAddr for Setup?");
      break;
    case SimpleRemoteEPCOpcode::Hangup:
      dbgs() << "Hangup";
      assert(SeqNo == 0 && "Non-zero SeqNo for Hangup?");
      assert(TagAddr.getValue() == 0 && "Non-zero TagAddr for Hangup?");
      break;
    case SimpleRemoteEPCOpcode::Result:
      dbgs() << "Result";
      assert(TagAddr.getValue() == 0 && "Non-zero TagAddr for Result?");
      break;
    case SimpleRemoteEPCOpcode::CallWrapper:
      dbgs() << "CallWrapper";
      break;
    }
    dbgs() << ", seqno = " << SeqNo
           << ", tag-addr = " << formatv("{0:x}", TagAddr.getValue())
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
    return make_error<StringError>("Unexpected Hangup opcode",
                                   inconvertibleErrorCode());
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
        shared::WrapperFunctionResult::createOutOfBandError("disconnecting"));

  std::lock_guard<std::mutex> Lock(SimpleRemoteEPCMutex);
  DisconnectErr = joinErrors(std::move(DisconnectErr), std::move(Err));
  Disconnected = true;
  DisconnectCV.notify_all();
}

Expected<std::unique_ptr<jitlink::JITLinkMemoryManager>>
SimpleRemoteEPC::createMemoryManager() {
  EPCGenericJITLinkMemoryManager::SymbolAddrs SAs;
  if (auto Err = getBootstrapSymbols(
          {{SAs.Allocator, rt::SimpleExecutorMemoryManagerInstanceName},
           {SAs.Reserve, rt::SimpleExecutorMemoryManagerReserveWrapperName},
           {SAs.Finalize, rt::SimpleExecutorMemoryManagerFinalizeWrapperName},
           {SAs.Deallocate,
            rt::SimpleExecutorMemoryManagerDeallocateWrapperName}}))
    return std::move(Err);

  return std::make_unique<EPCGenericJITLinkMemoryManager>(*this, SAs);
}

Expected<std::unique_ptr<ExecutorProcessControl::MemoryAccess>>
SimpleRemoteEPC::createMemoryAccess() {

  return nullptr;
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
      assert(TagAddr.getValue() == 0 && "Non-zero TagAddr for Hangup?");
      break;
    case SimpleRemoteEPCOpcode::Result:
      dbgs() << "Result";
      assert(TagAddr.getValue() == 0 && "Non-zero TagAddr for Result?");
      break;
    case SimpleRemoteEPCOpcode::CallWrapper:
      dbgs() << "CallWrapper";
      break;
    default:
      llvm_unreachable("Invalid opcode");
    }
    dbgs() << ", seqno = " << SeqNo
           << ", tag-addr = " << formatv("{0:x}", TagAddr.getValue())
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
                                   SimpleRemoteEPCArgBytesVector ArgBytes) {
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
      shared::WrapperFunctionResult::copyFrom(ArgBytes.data(), ArgBytes.size());
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
      [&](shared::WrapperFunctionResult SetupMsgBytes) {
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
           << "  Bootstrap symbols:\n";
    for (const auto &KV : EI->BootstrapSymbols)
      dbgs() << "    " << KV.first() << ": "
             << formatv("{0:x16}", KV.second.getValue()) << "\n";
  });
  TargetTriple = Triple(EI->TargetTriple);
  PageSize = EI->PageSize;
  BootstrapSymbols = std::move(EI->BootstrapSymbols);

  if (auto Err = getBootstrapSymbols(
          {{JDI.JITDispatchContext, ExecutorSessionObjectName},
           {JDI.JITDispatchFunction, DispatchFnName},
           {RunAsMainAddr, rt::RunAsMainWrapperName}}))
    return Err;

  if (auto DM =
          EPCGenericDylibManager::CreateWithDefaultBootstrapSymbols(*this))
    DylibMgr = std::make_unique<EPCGenericDylibManager>(std::move(*DM));
  else
    return DM.takeError();

  if (auto MemMgr = createMemoryManager()) {
    OwnedMemMgr = std::move(*MemMgr);
    this->MemMgr = OwnedMemMgr.get();
  } else
    return MemMgr.takeError();

  if (auto MemAccess = createMemoryAccess()) {
    OwnedMemAccess = std::move(*MemAccess);
    this->MemAccess = OwnedMemAccess.get();
  } else
    return MemAccess.takeError();

  return Error::success();
}

Error SimpleRemoteEPC::handleResult(uint64_t SeqNo, ExecutorAddr TagAddr,
                                    SimpleRemoteEPCArgBytesVector ArgBytes) {
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
      shared::WrapperFunctionResult::copyFrom(ArgBytes.data(), ArgBytes.size());
  SendResult(std::move(WFR));
  return Error::success();
}

void SimpleRemoteEPC::handleCallWrapper(
    uint64_t RemoteSeqNo, ExecutorAddr TagAddr,
    SimpleRemoteEPCArgBytesVector ArgBytes) {
  assert(ES && "No ExecutionSession attached");
  ES->runJITDispatchHandler(
      [this, RemoteSeqNo](shared::WrapperFunctionResult WFR) {
        if (auto Err = sendMessage(SimpleRemoteEPCOpcode::Result, RemoteSeqNo,
                                   ExecutorAddr(), {WFR.data(), WFR.size()}))
          getExecutionSession().reportError(std::move(Err));
      },
      TagAddr.getValue(), ArgBytes);
}

} // end namespace orc
} // end namespace llvm
