//===--- OrcRemoteTargetRPCAPI.h - Orc Remote-target RPC API ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the Orc remote-target RPC API. It should not be used
// directly, but is used by the RemoteTargetClient and RemoteTargetServer
// classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_ORCREMOTETARGETRPCAPI_H
#define LLVM_EXECUTIONENGINE_ORC_ORCREMOTETARGETRPCAPI_H

#include "JITSymbol.h"
#include "RPCChannel.h"
#include "RPCUtils.h"

namespace llvm {
namespace orc {
namespace remote {

class OrcRemoteTargetRPCAPI : public RPC<RPCChannel> {
protected:
  class ResourceIdMgr {
  public:
    typedef uint64_t ResourceId;
    ResourceId getNext() {
      if (!FreeIds.empty()) {
        ResourceId I = FreeIds.back();
        FreeIds.pop_back();
        return I;
      }
      return NextId++;
    }
    void release(ResourceId I) { FreeIds.push_back(I); }

  private:
    ResourceId NextId = 0;
    std::vector<ResourceId> FreeIds;
  };

public:
  enum JITProcId : uint32_t {
    InvalidId = 0,
    CallIntVoidId,
    CallIntVoidResponseId,
    CallMainId,
    CallMainResponseId,
    CallVoidVoidId,
    CallVoidVoidResponseId,
    CreateRemoteAllocatorId,
    CreateIndirectStubsOwnerId,
    DeregisterEHFramesId,
    DestroyRemoteAllocatorId,
    DestroyIndirectStubsOwnerId,
    EmitIndirectStubsId,
    EmitIndirectStubsResponseId,
    EmitResolverBlockId,
    EmitTrampolineBlockId,
    EmitTrampolineBlockResponseId,
    GetSymbolAddressId,
    GetSymbolAddressResponseId,
    GetRemoteInfoId,
    GetRemoteInfoResponseId,
    ReadMemId,
    ReadMemResponseId,
    RegisterEHFramesId,
    ReserveMemId,
    ReserveMemResponseId,
    RequestCompileId,
    RequestCompileResponseId,
    SetProtectionsId,
    TerminateSessionId,
    WriteMemId,
    WritePtrId
  };

  static const char *getJITProcIdName(JITProcId Id);

  typedef Procedure<CallIntVoidId, void(TargetAddress Addr)> CallIntVoid;

  typedef Procedure<CallIntVoidResponseId, void(int Result)>
    CallIntVoidResponse;

  typedef Procedure<CallMainId, void(TargetAddress Addr,
                                     std::vector<std::string> Args)>
      CallMain;

  typedef Procedure<CallMainResponseId, void(int Result)> CallMainResponse;

  typedef Procedure<CallVoidVoidId, void(TargetAddress FnAddr)> CallVoidVoid;

  typedef Procedure<CallVoidVoidResponseId, void()> CallVoidVoidResponse;

  typedef Procedure<CreateRemoteAllocatorId,
                    void(ResourceIdMgr::ResourceId AllocatorID)>
      CreateRemoteAllocator;

  typedef Procedure<CreateIndirectStubsOwnerId,
                    void(ResourceIdMgr::ResourceId StubOwnerID)>
    CreateIndirectStubsOwner;

  typedef Procedure<DeregisterEHFramesId,
                    void(TargetAddress Addr, uint32_t Size)>
      DeregisterEHFrames;

  typedef Procedure<DestroyRemoteAllocatorId,
                    void(ResourceIdMgr::ResourceId AllocatorID)>
      DestroyRemoteAllocator;

  typedef Procedure<DestroyIndirectStubsOwnerId,
                    void(ResourceIdMgr::ResourceId StubsOwnerID)>
      DestroyIndirectStubsOwner;

  typedef Procedure<EmitIndirectStubsId,
                    void(ResourceIdMgr::ResourceId StubsOwnerID,
                         uint32_t NumStubsRequired)>
      EmitIndirectStubs;

  typedef Procedure<EmitIndirectStubsResponseId,
                    void(TargetAddress StubsBaseAddr,
                         TargetAddress PtrsBaseAddr,
                         uint32_t NumStubsEmitted)>
      EmitIndirectStubsResponse;

  typedef Procedure<EmitResolverBlockId, void()> EmitResolverBlock;

  typedef Procedure<EmitTrampolineBlockId, void()> EmitTrampolineBlock;

  typedef Procedure<EmitTrampolineBlockResponseId,
                    void(TargetAddress BlockAddr, uint32_t NumTrampolines)>
      EmitTrampolineBlockResponse;

  typedef Procedure<GetSymbolAddressId, void(std::string SymbolName)>
      GetSymbolAddress;

  typedef Procedure<GetSymbolAddressResponseId, void(uint64_t SymbolAddr)>
      GetSymbolAddressResponse;

  typedef Procedure<GetRemoteInfoId, void()> GetRemoteInfo;

  typedef Procedure<GetRemoteInfoResponseId,
                    void(std::string Triple, uint32_t PointerSize,
                         uint32_t PageSize, uint32_t TrampolineSize,
                         uint32_t IndirectStubSize)>
      GetRemoteInfoResponse;

  typedef Procedure<ReadMemId, void(TargetAddress Src, uint64_t Size)>
      ReadMem;

  typedef Procedure<ReadMemResponseId, void()> ReadMemResponse;

  typedef Procedure<RegisterEHFramesId,
                    void(TargetAddress Addr, uint32_t Size)>
      RegisterEHFrames;

  typedef Procedure<ReserveMemId,
                    void(ResourceIdMgr::ResourceId AllocID, uint64_t Size,
                         uint32_t Align)>
      ReserveMem;

  typedef Procedure<ReserveMemResponseId, void(TargetAddress Addr)>
      ReserveMemResponse;

  typedef Procedure<RequestCompileId, void(TargetAddress TrampolineAddr)>
      RequestCompile;

  typedef Procedure<RequestCompileResponseId, void(TargetAddress ImplAddr)>
      RequestCompileResponse;

  typedef Procedure<SetProtectionsId,
                    void(ResourceIdMgr::ResourceId AllocID, TargetAddress Dst,
                         uint32_t ProtFlags)>
      SetProtections;

  typedef Procedure<TerminateSessionId, void()> TerminateSession;

  typedef Procedure<WriteMemId,
                    void(TargetAddress Dst, uint64_t Size /* Data to follow */)>
      WriteMem;

  typedef Procedure<WritePtrId, void(TargetAddress Dst, TargetAddress Val)>
      WritePtr;
};

} // end namespace remote
} // end namespace orc
} // end namespace llvm

#endif
