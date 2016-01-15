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

  typedef Procedure<CallIntVoidId, TargetAddress /* FnAddr */> CallIntVoid;

  typedef Procedure<CallIntVoidResponseId, int /* Result */>
      CallIntVoidResponse;

  typedef Procedure<CallMainId, TargetAddress /* FnAddr */,
                    std::vector<std::string> /* Args */>
      CallMain;

  typedef Procedure<CallMainResponseId, int /* Result */> CallMainResponse;

  typedef Procedure<CallVoidVoidId, TargetAddress /* FnAddr */> CallVoidVoid;

  typedef Procedure<CallVoidVoidResponseId> CallVoidVoidResponse;

  typedef Procedure<CreateRemoteAllocatorId,
                    ResourceIdMgr::ResourceId /* Allocator ID */>
      CreateRemoteAllocator;

  typedef Procedure<CreateIndirectStubsOwnerId,
                    ResourceIdMgr::ResourceId /* StubsOwner ID */>
      CreateIndirectStubsOwner;

  typedef Procedure<DeregisterEHFramesId, TargetAddress /* Addr */,
                    uint32_t /* Size */>
      DeregisterEHFrames;

  typedef Procedure<DestroyRemoteAllocatorId,
                    ResourceIdMgr::ResourceId /* Allocator ID */>
      DestroyRemoteAllocator;

  typedef Procedure<DestroyIndirectStubsOwnerId,
                    ResourceIdMgr::ResourceId /* StubsOwner ID */>
      DestroyIndirectStubsOwner;

  typedef Procedure<EmitIndirectStubsId,
                    ResourceIdMgr::ResourceId /* StubsOwner ID */,
                    uint32_t /* NumStubsRequired */>
      EmitIndirectStubs;

  typedef Procedure<
      EmitIndirectStubsResponseId, TargetAddress /* StubsBaseAddr */,
      TargetAddress /* PtrsBaseAddr */, uint32_t /* NumStubsEmitted */>
      EmitIndirectStubsResponse;

  typedef Procedure<EmitResolverBlockId> EmitResolverBlock;

  typedef Procedure<EmitTrampolineBlockId> EmitTrampolineBlock;

  typedef Procedure<EmitTrampolineBlockResponseId,
                    TargetAddress /* BlockAddr */,
                    uint32_t /* NumTrampolines */>
      EmitTrampolineBlockResponse;

  typedef Procedure<GetSymbolAddressId, std::string /*SymbolName*/>
      GetSymbolAddress;

  typedef Procedure<GetSymbolAddressResponseId, uint64_t /* SymbolAddr */>
      GetSymbolAddressResponse;

  typedef Procedure<GetRemoteInfoId> GetRemoteInfo;

  typedef Procedure<GetRemoteInfoResponseId, std::string /* Triple */,
                    uint32_t /* PointerSize */, uint32_t /* PageSize */,
                    uint32_t /* TrampolineSize */,
                    uint32_t /* IndirectStubSize */>
      GetRemoteInfoResponse;

  typedef Procedure<ReadMemId, TargetAddress /* Src */, uint64_t /* Size */>
      ReadMem;

  typedef Procedure<ReadMemResponseId> ReadMemResponse;

  typedef Procedure<RegisterEHFramesId, TargetAddress /* Addr */,
                    uint32_t /* Size */>
      RegisterEHFrames;

  typedef Procedure<ReserveMemId, ResourceIdMgr::ResourceId /* Id */,
                    uint64_t /* Size */, uint32_t /* Align */>
      ReserveMem;

  typedef Procedure<ReserveMemResponseId, TargetAddress /* Addr */>
      ReserveMemResponse;

  typedef Procedure<RequestCompileId, TargetAddress /* TrampolineAddr */>
      RequestCompile;

  typedef Procedure<RequestCompileResponseId, TargetAddress /* ImplAddr */>
      RequestCompileResponse;

  typedef Procedure<SetProtectionsId, ResourceIdMgr::ResourceId /* Id */,
                    TargetAddress /* Dst */, uint32_t /* ProtFlags */>
      SetProtections;

  typedef Procedure<TerminateSessionId> TerminateSession;

  typedef Procedure<WriteMemId, TargetAddress /* Dst */, uint64_t /* Size */
                    /* Data should follow */>
      WriteMem;

  typedef Procedure<WritePtrId, TargetAddress /* Dst */,
                    TargetAddress /* Val */>
      WritePtr;
};

} // end namespace remote
} // end namespace orc
} // end namespace llvm

#endif
