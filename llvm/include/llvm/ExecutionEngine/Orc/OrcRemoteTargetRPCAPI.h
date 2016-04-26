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

class DirectBufferWriter {
public:
  DirectBufferWriter() = default;
  DirectBufferWriter(const char *Src, TargetAddress Dst, uint64_t Size)
      : Src(Src), Dst(Dst), Size(Size) {}

  const char *getSrc() const { return Src; }
  TargetAddress getDst() const { return Dst; }
  uint64_t getSize() const { return Size; }

private:
  const char *Src;
  TargetAddress Dst;
  uint64_t Size;
};

inline Error serialize(RPCChannel &C, const DirectBufferWriter &DBW) {
  if (auto EC = serialize(C, DBW.getDst()))
    return EC;
  if (auto EC = serialize(C, DBW.getSize()))
    return EC;
  return C.appendBytes(DBW.getSrc(), DBW.getSize());
}

inline Error deserialize(RPCChannel &C, DirectBufferWriter &DBW) {
  TargetAddress Dst;
  if (auto EC = deserialize(C, Dst))
    return EC;
  uint64_t Size;
  if (auto EC = deserialize(C, Size))
    return EC;
  char *Addr = reinterpret_cast<char *>(static_cast<uintptr_t>(Dst));

  DBW = DirectBufferWriter(0, Dst, Size);

  return C.readBytes(Addr, Size);
}

class OrcRemoteTargetRPCAPI : public RPC<RPCChannel> {
protected:
  class ResourceIdMgr {
  public:
    typedef uint64_t ResourceId;
    static const ResourceId InvalidId = ~0U;

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
  // FIXME: Remove constructors once MSVC supports synthesizing move-ops.
  OrcRemoteTargetRPCAPI() = default;
  OrcRemoteTargetRPCAPI(const OrcRemoteTargetRPCAPI &) = delete;
  OrcRemoteTargetRPCAPI &operator=(const OrcRemoteTargetRPCAPI &) = delete;

  OrcRemoteTargetRPCAPI(OrcRemoteTargetRPCAPI &&) {}
  OrcRemoteTargetRPCAPI &operator=(OrcRemoteTargetRPCAPI &&) { return *this; }

  enum JITFuncId : uint32_t {
    InvalidId = RPCFunctionIdTraits<JITFuncId>::InvalidId,
    CallIntVoidId = RPCFunctionIdTraits<JITFuncId>::FirstValidId,
    CallMainId,
    CallVoidVoidId,
    CreateRemoteAllocatorId,
    CreateIndirectStubsOwnerId,
    DeregisterEHFramesId,
    DestroyRemoteAllocatorId,
    DestroyIndirectStubsOwnerId,
    EmitIndirectStubsId,
    EmitResolverBlockId,
    EmitTrampolineBlockId,
    GetSymbolAddressId,
    GetRemoteInfoId,
    ReadMemId,
    RegisterEHFramesId,
    ReserveMemId,
    RequestCompileId,
    SetProtectionsId,
    TerminateSessionId,
    WriteMemId,
    WritePtrId
  };

  static const char *getJITFuncIdName(JITFuncId Id);

  typedef Function<CallIntVoidId, int32_t(TargetAddress Addr)> CallIntVoid;

  typedef Function<CallMainId,
                   int32_t(TargetAddress Addr, std::vector<std::string> Args)>
      CallMain;

  typedef Function<CallVoidVoidId, void(TargetAddress FnAddr)> CallVoidVoid;

  typedef Function<CreateRemoteAllocatorId,
                   void(ResourceIdMgr::ResourceId AllocatorID)>
      CreateRemoteAllocator;

  typedef Function<CreateIndirectStubsOwnerId,
                   void(ResourceIdMgr::ResourceId StubOwnerID)>
      CreateIndirectStubsOwner;

  typedef Function<DeregisterEHFramesId,
                   void(TargetAddress Addr, uint32_t Size)>
      DeregisterEHFrames;

  typedef Function<DestroyRemoteAllocatorId,
                   void(ResourceIdMgr::ResourceId AllocatorID)>
      DestroyRemoteAllocator;

  typedef Function<DestroyIndirectStubsOwnerId,
                   void(ResourceIdMgr::ResourceId StubsOwnerID)>
      DestroyIndirectStubsOwner;

  /// EmitIndirectStubs result is (StubsBase, PtrsBase, NumStubsEmitted).
  typedef Function<EmitIndirectStubsId,
                   std::tuple<TargetAddress, TargetAddress, uint32_t>(
                       ResourceIdMgr::ResourceId StubsOwnerID,
                       uint32_t NumStubsRequired)>
      EmitIndirectStubs;

  typedef Function<EmitResolverBlockId, void()> EmitResolverBlock;

  /// EmitTrampolineBlock result is (BlockAddr, NumTrampolines).
  typedef Function<EmitTrampolineBlockId, std::tuple<TargetAddress, uint32_t>()>
      EmitTrampolineBlock;

  typedef Function<GetSymbolAddressId, TargetAddress(std::string SymbolName)>
      GetSymbolAddress;

  /// GetRemoteInfo result is (Triple, PointerSize, PageSize, TrampolineSize,
  ///                          IndirectStubsSize).
  typedef Function<GetRemoteInfoId, std::tuple<std::string, uint32_t, uint32_t,
                                               uint32_t, uint32_t>()>
      GetRemoteInfo;

  typedef Function<ReadMemId,
                   std::vector<char>(TargetAddress Src, uint64_t Size)>
      ReadMem;

  typedef Function<RegisterEHFramesId, void(TargetAddress Addr, uint32_t Size)>
      RegisterEHFrames;

  typedef Function<ReserveMemId,
                   TargetAddress(ResourceIdMgr::ResourceId AllocID,
                                 uint64_t Size, uint32_t Align)>
      ReserveMem;

  typedef Function<RequestCompileId,
                   TargetAddress(TargetAddress TrampolineAddr)>
      RequestCompile;

  typedef Function<SetProtectionsId,
                   void(ResourceIdMgr::ResourceId AllocID, TargetAddress Dst,
                        uint32_t ProtFlags)>
      SetProtections;

  typedef Function<TerminateSessionId, void()> TerminateSession;

  typedef Function<WriteMemId, void(DirectBufferWriter DB)> WriteMem;

  typedef Function<WritePtrId, void(TargetAddress Dst, TargetAddress Val)>
      WritePtr;
};

} // end namespace remote
} // end namespace orc
} // end namespace llvm

#endif
