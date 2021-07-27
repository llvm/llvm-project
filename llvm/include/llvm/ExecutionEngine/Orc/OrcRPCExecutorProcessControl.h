//===-- OrcRPCExecutorProcessControl.h - Remote target control --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Executor control via ORC RPC.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_ORCRPCEXECUTORPROCESSCONTROL_H
#define LLVM_EXECUTIONENGINE_ORC_ORCRPCEXECUTORPROCESSCONTROL_H

#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/Shared/RPCUtils.h"
#include "llvm/ExecutionEngine/Orc/Shared/RawByteChannel.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/OrcRPCTPCServer.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"

namespace llvm {
namespace orc {

/// JITLinkMemoryManager implementation for a process connected via an ORC RPC
/// endpoint.
template <typename OrcRPCEPCImplT>
class OrcRPCEPCJITLinkMemoryManager : public jitlink::JITLinkMemoryManager {
private:
  struct HostAlloc {
    std::unique_ptr<char[]> Mem;
    uint64_t Size;
  };

  struct TargetAlloc {
    JITTargetAddress Address = 0;
    uint64_t AllocatedSize = 0;
  };

  using HostAllocMap = DenseMap<int, HostAlloc>;
  using TargetAllocMap = DenseMap<int, TargetAlloc>;

public:
  class OrcRPCAllocation : public Allocation {
  public:
    OrcRPCAllocation(OrcRPCEPCJITLinkMemoryManager<OrcRPCEPCImplT> &Parent,
                     HostAllocMap HostAllocs, TargetAllocMap TargetAllocs)
        : Parent(Parent), HostAllocs(std::move(HostAllocs)),
          TargetAllocs(std::move(TargetAllocs)) {
      assert(HostAllocs.size() == TargetAllocs.size() &&
             "HostAllocs size should match TargetAllocs");
    }

    ~OrcRPCAllocation() override {
      assert(TargetAllocs.empty() && "failed to deallocate");
    }

    MutableArrayRef<char> getWorkingMemory(ProtectionFlags Seg) override {
      auto I = HostAllocs.find(Seg);
      assert(I != HostAllocs.end() && "No host allocation for segment");
      auto &HA = I->second;
      return {HA.Mem.get(), static_cast<size_t>(HA.Size)};
    }

    JITTargetAddress getTargetMemory(ProtectionFlags Seg) override {
      auto I = TargetAllocs.find(Seg);
      assert(I != TargetAllocs.end() && "No target allocation for segment");
      return I->second.Address;
    }

    void finalizeAsync(FinalizeContinuation OnFinalize) override {

      std::vector<tpctypes::BufferWrite> BufferWrites;
      orcrpctpc::ReleaseOrFinalizeMemRequest FMR;

      for (auto &KV : HostAllocs) {
        assert(TargetAllocs.count(KV.first) &&
               "No target allocation for buffer");
        auto &HA = KV.second;
        auto &TA = TargetAllocs[KV.first];
        BufferWrites.push_back({TA.Address, StringRef(HA.Mem.get(), HA.Size)});
        FMR.push_back({orcrpctpc::toWireProtectionFlags(
                           static_cast<sys::Memory::ProtectionFlags>(KV.first)),
                       TA.Address, TA.AllocatedSize});
      }

      DEBUG_WITH_TYPE("orc", {
        dbgs() << "finalizeAsync " << (void *)this << ":\n";
        auto FMRI = FMR.begin();
        for (auto &B : BufferWrites) {
          auto Prot = FMRI->Prot;
          ++FMRI;
          dbgs() << "  Writing " << formatv("{0:x16}", B.Buffer.size())
                 << " bytes to " << ((Prot & orcrpctpc::WPF_Read) ? 'R' : '-')
                 << ((Prot & orcrpctpc::WPF_Write) ? 'W' : '-')
                 << ((Prot & orcrpctpc::WPF_Exec) ? 'X' : '-')
                 << " segment: local " << (const void *)B.Buffer.data()
                 << " -> target " << formatv("{0:x16}", B.Address) << "\n";
        }
      });
      if (auto Err =
              Parent.Parent.getMemoryAccess().writeBuffers(BufferWrites)) {
        OnFinalize(std::move(Err));
        return;
      }

      DEBUG_WITH_TYPE("orc", dbgs() << " Applying permissions...\n");
      if (auto Err =
              Parent.getEndpoint().template callAsync<orcrpctpc::FinalizeMem>(
                  [OF = std::move(OnFinalize)](Error Err2) {
                    // FIXME: Dispatch to work queue.
                    std::thread([OF = std::move(OF),
                                 Err3 = std::move(Err2)]() mutable {
                      DEBUG_WITH_TYPE(
                          "orc", { dbgs() << "  finalizeAsync complete\n"; });
                      OF(std::move(Err3));
                    }).detach();
                    return Error::success();
                  },
                  FMR)) {
        DEBUG_WITH_TYPE("orc", dbgs() << "    failed.\n");
        Parent.getEndpoint().abandonPendingResponses();
        Parent.reportError(std::move(Err));
      }
      DEBUG_WITH_TYPE("orc", {
        dbgs() << "Leaving finalizeAsync (finalization may continue in "
                  "background)\n";
      });
    }

    Error deallocate() override {
      orcrpctpc::ReleaseOrFinalizeMemRequest RMR;
      for (auto &KV : TargetAllocs)
        RMR.push_back({orcrpctpc::toWireProtectionFlags(
                           static_cast<sys::Memory::ProtectionFlags>(KV.first)),
                       KV.second.Address, KV.second.AllocatedSize});
      TargetAllocs.clear();

      return Parent.getEndpoint().template callB<orcrpctpc::ReleaseMem>(RMR);
    }

  private:
    OrcRPCEPCJITLinkMemoryManager<OrcRPCEPCImplT> &Parent;
    HostAllocMap HostAllocs;
    TargetAllocMap TargetAllocs;
  };

  OrcRPCEPCJITLinkMemoryManager(OrcRPCEPCImplT &Parent) : Parent(Parent) {}

  Expected<std::unique_ptr<Allocation>>
  allocate(const jitlink::JITLinkDylib *JD,
           const SegmentsRequestMap &Request) override {
    orcrpctpc::ReserveMemRequest RMR;
    HostAllocMap HostAllocs;

    for (auto &KV : Request) {
      assert(KV.second.getContentSize() <= std::numeric_limits<size_t>::max() &&
             "Content size is out-of-range for host");

      RMR.push_back({orcrpctpc::toWireProtectionFlags(
                         static_cast<sys::Memory::ProtectionFlags>(KV.first)),
                     KV.second.getContentSize() + KV.second.getZeroFillSize(),
                     KV.second.getAlignment()});
      HostAllocs[KV.first] = {
          std::make_unique<char[]>(KV.second.getContentSize()),
          KV.second.getContentSize()};
    }

    DEBUG_WITH_TYPE("orc", {
      dbgs() << "Orc remote memmgr got request:\n";
      for (auto &KV : Request)
        dbgs() << "  permissions: "
               << ((KV.first & sys::Memory::MF_READ) ? 'R' : '-')
               << ((KV.first & sys::Memory::MF_WRITE) ? 'W' : '-')
               << ((KV.first & sys::Memory::MF_EXEC) ? 'X' : '-')
               << ", content size: "
               << formatv("{0:x16}", KV.second.getContentSize())
               << " + zero-fill-size: "
               << formatv("{0:x16}", KV.second.getZeroFillSize())
               << ", align: " << KV.second.getAlignment() << "\n";
    });

    // FIXME: LLVM RPC needs to be fixed to support alt
    // serialization/deserialization on return types. For now just
    // translate from std::map to DenseMap manually.
    auto TmpTargetAllocs =
        Parent.getEndpoint().template callB<orcrpctpc::ReserveMem>(RMR);
    if (!TmpTargetAllocs)
      return TmpTargetAllocs.takeError();

    if (TmpTargetAllocs->size() != RMR.size())
      return make_error<StringError>(
          "Number of target allocations does not match request",
          inconvertibleErrorCode());

    TargetAllocMap TargetAllocs;
    for (auto &E : *TmpTargetAllocs)
      TargetAllocs[orcrpctpc::fromWireProtectionFlags(E.Prot)] = {
          E.Address, E.AllocatedSize};

    DEBUG_WITH_TYPE("orc", {
      auto HAI = HostAllocs.begin();
      for (auto &KV : TargetAllocs)
        dbgs() << "  permissions: "
               << ((KV.first & sys::Memory::MF_READ) ? 'R' : '-')
               << ((KV.first & sys::Memory::MF_WRITE) ? 'W' : '-')
               << ((KV.first & sys::Memory::MF_EXEC) ? 'X' : '-')
               << " assigned local " << (void *)HAI->second.Mem.get()
               << ", target " << formatv("{0:x16}", KV.second.Address) << "\n";
    });

    return std::make_unique<OrcRPCAllocation>(*this, std::move(HostAllocs),
                                              std::move(TargetAllocs));
  }

private:
  void reportError(Error Err) { Parent.reportError(std::move(Err)); }

  decltype(std::declval<OrcRPCEPCImplT>().getEndpoint()) getEndpoint() {
    return Parent.getEndpoint();
  }

  OrcRPCEPCImplT &Parent;
};

/// ExecutorProcessControl::MemoryAccess implementation for a process connected
/// via an ORC RPC endpoint.
template <typename OrcRPCEPCImplT>
class OrcRPCEPCMemoryAccess : public ExecutorProcessControl::MemoryAccess {
public:
  OrcRPCEPCMemoryAccess(OrcRPCEPCImplT &Parent) : Parent(Parent) {}

  void writeUInt8s(ArrayRef<tpctypes::UInt8Write> Ws,
                   WriteResultFn OnWriteComplete) override {
    writeViaRPC<orcrpctpc::WriteUInt8s>(Ws, std::move(OnWriteComplete));
  }

  void writeUInt16s(ArrayRef<tpctypes::UInt16Write> Ws,
                    WriteResultFn OnWriteComplete) override {
    writeViaRPC<orcrpctpc::WriteUInt16s>(Ws, std::move(OnWriteComplete));
  }

  void writeUInt32s(ArrayRef<tpctypes::UInt32Write> Ws,
                    WriteResultFn OnWriteComplete) override {
    writeViaRPC<orcrpctpc::WriteUInt32s>(Ws, std::move(OnWriteComplete));
  }

  void writeUInt64s(ArrayRef<tpctypes::UInt64Write> Ws,
                    WriteResultFn OnWriteComplete) override {
    writeViaRPC<orcrpctpc::WriteUInt64s>(Ws, std::move(OnWriteComplete));
  }

  void writeBuffers(ArrayRef<tpctypes::BufferWrite> Ws,
                    WriteResultFn OnWriteComplete) override {
    writeViaRPC<orcrpctpc::WriteBuffers>(Ws, std::move(OnWriteComplete));
  }

private:
  template <typename WriteRPCFunction, typename WriteElementT>
  void writeViaRPC(ArrayRef<WriteElementT> Ws, WriteResultFn OnWriteComplete) {
    if (auto Err = Parent.getEndpoint().template callAsync<WriteRPCFunction>(
            [OWC = std::move(OnWriteComplete)](Error Err2) mutable -> Error {
              OWC(std::move(Err2));
              return Error::success();
            },
            Ws)) {
      Parent.reportError(std::move(Err));
      Parent.getEndpoint().abandonPendingResponses();
    }
  }

  OrcRPCEPCImplT &Parent;
};

// ExecutorProcessControl for a process connected via an ORC RPC Endpoint.
template <typename RPCEndpointT>
class OrcRPCExecutorProcessControlBase : public ExecutorProcessControl {
public:
  using ErrorReporter = unique_function<void(Error)>;

  using OnCloseConnectionFunction = unique_function<Error(Error)>;

  OrcRPCExecutorProcessControlBase(std::shared_ptr<SymbolStringPool> SSP,
                                   RPCEndpointT &EP, ErrorReporter ReportError)
      : ExecutorProcessControl(std::move(SSP)),
        ReportError(std::move(ReportError)), EP(EP) {
    using ThisT = OrcRPCExecutorProcessControlBase<RPCEndpointT>;
    EP.template addAsyncHandler<orcrpctpc::RunWrapper>(*this,
                                                       &ThisT::runWrapperInJIT);
  }

  void reportError(Error Err) { ReportError(std::move(Err)); }

  RPCEndpointT &getEndpoint() { return EP; }

  Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) override {
    DEBUG_WITH_TYPE("orc", {
      dbgs() << "Loading dylib \"" << (DylibPath ? DylibPath : "") << "\" ";
      if (!DylibPath)
        dbgs() << "(process symbols)";
      dbgs() << "\n";
    });
    if (!DylibPath)
      DylibPath = "";
    auto H = EP.template callB<orcrpctpc::LoadDylib>(DylibPath);
    DEBUG_WITH_TYPE("orc", {
      if (H)
        dbgs() << "  got handle " << formatv("{0:x16}", *H) << "\n";
      else
        dbgs() << "  error, unable to load\n";
    });
    return H;
  }

  Expected<std::vector<tpctypes::LookupResult>>
  lookupSymbols(ArrayRef<LookupRequest> Request) override {
    std::vector<orcrpctpc::RemoteLookupRequest> RR;
    for (auto &E : Request) {
      RR.push_back({});
      RR.back().first = E.Handle;
      for (auto &KV : E.Symbols)
        RR.back().second.push_back(
            {(*KV.first).str(),
             KV.second == SymbolLookupFlags::WeaklyReferencedSymbol});
    }
    DEBUG_WITH_TYPE("orc", {
      dbgs() << "Compound lookup:\n";
      for (auto &R : Request) {
        dbgs() << "  In " << formatv("{0:x16}", R.Handle) << ": {";
        bool First = true;
        for (auto &KV : R.Symbols) {
          dbgs() << (First ? "" : ",") << " " << *KV.first;
          First = false;
        }
        dbgs() << " }\n";
      }
    });
    return EP.template callB<orcrpctpc::LookupSymbols>(RR);
  }

  Expected<int32_t> runAsMain(JITTargetAddress MainFnAddr,
                              ArrayRef<std::string> Args) override {
    DEBUG_WITH_TYPE("orc", {
      dbgs() << "Running as main: " << formatv("{0:x16}", MainFnAddr)
             << ", args = [";
      for (unsigned I = 0; I != Args.size(); ++I)
        dbgs() << (I ? "," : "") << " \"" << Args[I] << "\"";
      dbgs() << "]\n";
    });
    auto Result = EP.template callB<orcrpctpc::RunMain>(MainFnAddr, Args);
    DEBUG_WITH_TYPE("orc", {
      dbgs() << "  call to " << formatv("{0:x16}", MainFnAddr);
      if (Result)
        dbgs() << " returned result " << *Result << "\n";
      else
        dbgs() << " failed\n";
    });
    return Result;
  }

  void callWrapperAsync(SendResultFunction OnComplete,
                        JITTargetAddress WrapperFnAddr,
                        ArrayRef<char> ArgBuffer) override {
    DEBUG_WITH_TYPE("orc", {
      dbgs() << "Running as wrapper function "
             << formatv("{0:x16}", WrapperFnAddr) << " with "
             << formatv("{0:x16}", ArgBuffer.size()) << " argument buffer\n";
    });
    auto Result = EP.template callB<orcrpctpc::RunWrapper>(
        WrapperFnAddr,
        ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(ArgBuffer.data()),
                          ArgBuffer.size()));

    if (!Result)
      OnComplete(shared::WrapperFunctionResult::createOutOfBandError(
          toString(Result.takeError())));
    OnComplete(std::move(*Result));
  }

  Error closeConnection(OnCloseConnectionFunction OnCloseConnection) {
    DEBUG_WITH_TYPE("orc", dbgs() << "Closing connection to remote\n");
    return EP.template callAsync<orcrpctpc::CloseConnection>(
        std::move(OnCloseConnection));
  }

  Error closeConnectionAndWait() {
    std::promise<MSVCPError> P;
    auto F = P.get_future();
    if (auto Err = closeConnection([&](Error Err2) -> Error {
          P.set_value(std::move(Err2));
          return Error::success();
        })) {
      EP.abandonAllPendingResponses();
      return joinErrors(std::move(Err), F.get());
    }
    return F.get();
  }

protected:
  /// Subclasses must call this during construction to initialize the
  /// TargetTriple and PageSize members.
  Error initializeORCRPCEPCBase() {
    if (auto EPI = EP.template callB<orcrpctpc::GetExecutorProcessInfo>()) {
      this->TargetTriple = Triple(EPI->Triple);
      this->PageSize = PageSize;
      this->JDI = {ExecutorAddress(EPI->DispatchFuncAddr),
                   ExecutorAddress(EPI->DispatchCtxAddr)};
      return Error::success();
    } else
      return EPI.takeError();
  }

private:
  Error runWrapperInJIT(
      std::function<Error(Expected<shared::WrapperFunctionResult>)> SendResult,
      JITTargetAddress FunctionTag, std::vector<uint8_t> ArgBuffer) {

    getExecutionSession().runJITDispatchHandler(
        [this, SendResult = std::move(SendResult)](
            Expected<shared::WrapperFunctionResult> R) {
          if (auto Err = SendResult(std::move(R)))
            ReportError(std::move(Err));
        },
        FunctionTag,
        {reinterpret_cast<const char *>(ArgBuffer.data()), ArgBuffer.size()});
    return Error::success();
  }

  ErrorReporter ReportError;
  RPCEndpointT &EP;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_ORCRPCEXECUTORPROCESSCONTROL_H
