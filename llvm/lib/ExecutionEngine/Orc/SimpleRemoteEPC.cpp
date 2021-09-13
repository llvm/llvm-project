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
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {
namespace shared {

template <>
class SPSSerializationTraits<SPSRemoteSymbolLookupSetElement,
                             SymbolLookupSet::value_type> {
public:
  static size_t size(const SymbolLookupSet::value_type &V) {
    return SPSArgList<SPSString, bool>::size(
        *V.first, V.second == SymbolLookupFlags::RequiredSymbol);
  }

  static bool serialize(SPSOutputBuffer &OB,
                        const SymbolLookupSet::value_type &V) {
    return SPSArgList<SPSString, bool>::serialize(
        OB, *V.first, V.second == SymbolLookupFlags::RequiredSymbol);
  }
};

template <>
class TrivialSPSSequenceSerialization<SPSRemoteSymbolLookupSetElement,
                                      SymbolLookupSet> {
public:
  static constexpr bool available = true;
};

template <>
class SPSSerializationTraits<SPSRemoteSymbolLookup,
                             ExecutorProcessControl::LookupRequest> {
  using MemberSerialization =
      SPSArgList<SPSExecutorAddress, SPSRemoteSymbolLookupSet>;

public:
  static size_t size(const ExecutorProcessControl::LookupRequest &LR) {
    return MemberSerialization::size(ExecutorAddress(LR.Handle), LR.Symbols);
  }

  static bool serialize(SPSOutputBuffer &OB,
                        const ExecutorProcessControl::LookupRequest &LR) {
    return MemberSerialization::serialize(OB, ExecutorAddress(LR.Handle),
                                          LR.Symbols);
  }
};

} // end namespace shared

SimpleRemoteEPC::~SimpleRemoteEPC() {
  assert(Disconnected && "Destroyed without disconnection");
}

Expected<tpctypes::DylibHandle>
SimpleRemoteEPC::loadDylib(const char *DylibPath) {
  Expected<tpctypes::DylibHandle> H((tpctypes::DylibHandle()));
  if (auto Err = callSPSWrapper<shared::SPSLoadDylibSignature>(
          LoadDylibAddr.getValue(), H, JDI.JITDispatchContextAddress,
          StringRef(DylibPath), (uint64_t)0))
    return std::move(Err);
  return H;
}

Expected<std::vector<tpctypes::LookupResult>>
SimpleRemoteEPC::lookupSymbols(ArrayRef<LookupRequest> Request) {
  Expected<std::vector<tpctypes::LookupResult>> R(
      (std::vector<tpctypes::LookupResult>()));

  if (auto Err = callSPSWrapper<shared::SPSLookupSymbolsSignature>(
          LookupSymbolsAddr.getValue(), R, JDI.JITDispatchContextAddress,
          Request))
    return std::move(Err);
  return R;
}

Expected<int32_t> SimpleRemoteEPC::runAsMain(JITTargetAddress MainFnAddr,
                                             ArrayRef<std::string> Args) {
  int64_t Result = 0;
  if (auto Err = callSPSWrapper<shared::SPSRunAsMainSignature>(
          RunAsMainAddr.getValue(), Result, ExecutorAddress(MainFnAddr), Args))
    return std::move(Err);
  return Result;
}

void SimpleRemoteEPC::callWrapperAsync(SendResultFunction OnComplete,
                                       JITTargetAddress WrapperFnAddr,
                                       ArrayRef<char> ArgBuffer) {
  uint64_t SeqNo;
  {
    std::lock_guard<std::mutex> Lock(SimpleRemoteEPCMutex);
    SeqNo = getNextSeqNo();
    assert(!PendingCallWrapperResults.count(SeqNo) && "SeqNo already in use");
    PendingCallWrapperResults[SeqNo] = std::move(OnComplete);
  }

  if (auto Err = T->sendMessage(SimpleRemoteEPCOpcode::CallWrapper, SeqNo,
                                ExecutorAddress(WrapperFnAddr), ArgBuffer)) {
    getExecutionSession().reportError(std::move(Err));
  }
}

Error SimpleRemoteEPC::disconnect() {
  Disconnected = true;
  T->disconnect();
  return Error::success();
}

Expected<SimpleRemoteEPCTransportClient::HandleMessageAction>
SimpleRemoteEPC::handleMessage(SimpleRemoteEPCOpcode OpC, uint64_t SeqNo,
                               ExecutorAddress TagAddr,
                               SimpleRemoteEPCArgBytesVector ArgBytes) {
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
    // FIXME: Put EPC into 'detached' state.
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

void SimpleRemoteEPC::handleDisconnect(Error Err) {
  PendingCallWrapperResultsMap TmpPending;

  {
    std::lock_guard<std::mutex> Lock(SimpleRemoteEPCMutex);
    std::swap(TmpPending, PendingCallWrapperResults);
  }

  for (auto &KV : TmpPending)
    KV.second(
        shared::WrapperFunctionResult::createOutOfBandError("disconnecting"));

  if (Err) {
    // FIXME: Move ReportError to EPC.
    if (ES)
      ES->reportError(std::move(Err));
    else
      logAllUnhandledErrors(std::move(Err), errs(), "SimpleRemoteEPC: ");
  }
}

Expected<std::unique_ptr<jitlink::JITLinkMemoryManager>>
SimpleRemoteEPC::createMemoryManager() {
  EPCGenericJITLinkMemoryManager::FuncAddrs FAs;
  if (auto Err = getBootstrapSymbols(
          {{FAs.Reserve, "__llvm_orc_memory_reserve"},
           {FAs.Finalize, "__llvm_orc_memory_finalize"},
           {FAs.Deallocate, "__llvm_orc_memory_deallocate"}}))
    return std::move(Err);

  return std::make_unique<EPCGenericJITLinkMemoryManager>(*this, FAs);
}

Expected<std::unique_ptr<ExecutorProcessControl::MemoryAccess>>
SimpleRemoteEPC::createMemoryAccess() {

  return nullptr;
}

Error SimpleRemoteEPC::handleSetup(uint64_t SeqNo, ExecutorAddress TagAddr,
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

void SimpleRemoteEPC::prepareToReceiveSetupMessage(
    std::promise<MSVCPExpected<SimpleRemoteEPCExecutorInfo>> &ExecInfoP) {
  PendingCallWrapperResults[0] =
      [&](shared::WrapperFunctionResult SetupMsgBytes) {
        if (const char *ErrMsg = SetupMsgBytes.getOutOfBandError()) {
          ExecInfoP.set_value(
              make_error<StringError>(ErrMsg, inconvertibleErrorCode()));
          return;
        }
        using SPSSerialize =
            shared::SPSArgList<shared::SPSSimpleRemoteEPCExecutorInfo>;
        shared::SPSInputBuffer IB(SetupMsgBytes.data(), SetupMsgBytes.size());
        SimpleRemoteEPCExecutorInfo EI;
        if (SPSSerialize::deserialize(IB, EI))
          ExecInfoP.set_value(EI);
        else
          ExecInfoP.set_value(make_error<StringError>(
              "Could not deserialize setup message", inconvertibleErrorCode()));
      };
}

Error SimpleRemoteEPC::setup(std::unique_ptr<SimpleRemoteEPCTransport> T,
                             SimpleRemoteEPCExecutorInfo EI) {
  using namespace SimpleRemoteEPCDefaultBootstrapSymbolNames;
  LLVM_DEBUG({
    dbgs() << "SimpleRemoteEPC received setup message:\n"
           << "  Triple: " << EI.TargetTriple << "\n"
           << "  Page size: " << EI.PageSize << "\n"
           << "  Bootstrap symbols:\n";
    for (const auto &KV : EI.BootstrapSymbols)
      dbgs() << "    " << KV.first() << ": "
             << formatv("{0:x16}", KV.second.getValue()) << "\n";
  });
  this->T = std::move(T);
  TargetTriple = Triple(EI.TargetTriple);
  PageSize = EI.PageSize;
  BootstrapSymbols = std::move(EI.BootstrapSymbols);

  if (auto Err = getBootstrapSymbols(
          {{JDI.JITDispatchContextAddress, ExecutorSessionObjectName},
           {JDI.JITDispatchFunctionAddress, DispatchFnName},
           {LoadDylibAddr, "__llvm_orc_load_dylib"},
           {LookupSymbolsAddr, "__llvm_orc_lookup_symbols"},
           {RunAsMainAddr, "__llvm_orc_run_as_main"}}))
    return Err;

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

Error SimpleRemoteEPC::handleResult(uint64_t SeqNo, ExecutorAddress TagAddr,
                                    SimpleRemoteEPCArgBytesVector ArgBytes) {
  SendResultFunction SendResult;

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
    uint64_t RemoteSeqNo, ExecutorAddress TagAddr,
    SimpleRemoteEPCArgBytesVector ArgBytes) {
  assert(ES && "No ExecutionSession attached");
  ES->runJITDispatchHandler(
      [this, RemoteSeqNo](shared::WrapperFunctionResult WFR) {
        if (auto Err =
                T->sendMessage(SimpleRemoteEPCOpcode::Result, RemoteSeqNo,
                               ExecutorAddress(), {WFR.data(), WFR.size()}))
          getExecutionSession().reportError(std::move(Err));
      },
      TagAddr.getValue(), ArgBytes);
}

} // end namespace orc
} // end namespace llvm
