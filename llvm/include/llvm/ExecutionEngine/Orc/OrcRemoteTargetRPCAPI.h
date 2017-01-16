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

#include "RPCUtils.h"
#include "RawByteChannel.h"
#include "llvm/ExecutionEngine/JITSymbol.h"

namespace llvm {
namespace orc {
namespace remote {

class DirectBufferWriter {
public:
  DirectBufferWriter() = default;
  DirectBufferWriter(const char *Src, JITTargetAddress Dst, uint64_t Size)
      : Src(Src), Dst(Dst), Size(Size) {}

  const char *getSrc() const { return Src; }
  JITTargetAddress getDst() const { return Dst; }
  uint64_t getSize() const { return Size; }

private:
  const char *Src;
  JITTargetAddress Dst;
  uint64_t Size;
};

} // end namespace remote

namespace rpc {

template <> class RPCTypeName<remote::DirectBufferWriter> {
public:
  static const char *getName() { return "DirectBufferWriter"; }
};

template <typename ChannelT>
class SerializationTraits<
    ChannelT, remote::DirectBufferWriter, remote::DirectBufferWriter,
    typename std::enable_if<
        std::is_base_of<RawByteChannel, ChannelT>::value>::type> {
public:
  static Error serialize(ChannelT &C, const remote::DirectBufferWriter &DBW) {
    if (auto EC = serializeSeq(C, DBW.getDst()))
      return EC;
    if (auto EC = serializeSeq(C, DBW.getSize()))
      return EC;
    return C.appendBytes(DBW.getSrc(), DBW.getSize());
  }

  static Error deserialize(ChannelT &C, remote::DirectBufferWriter &DBW) {
    JITTargetAddress Dst;
    if (auto EC = deserializeSeq(C, Dst))
      return EC;
    uint64_t Size;
    if (auto EC = deserializeSeq(C, Size))
      return EC;
    char *Addr = reinterpret_cast<char *>(static_cast<uintptr_t>(Dst));

    DBW = remote::DirectBufferWriter(0, Dst, Size);

    return C.readBytes(Addr, Size);
  }
};

} // end namespace rpc

namespace remote {

class OrcRemoteTargetRPCAPI
    : public rpc::SingleThreadedRPCEndpoint<rpc::RawByteChannel> {
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
  OrcRemoteTargetRPCAPI(rpc::RawByteChannel &C)
      : rpc::SingleThreadedRPCEndpoint<rpc::RawByteChannel>(C, true) {}

  class CallIntVoid
      : public rpc::Function<CallIntVoid, int32_t(JITTargetAddress Addr)> {
  public:
    static const char *getName() { return "CallIntVoid"; }
  };

  class CallMain
      : public rpc::Function<CallMain, int32_t(JITTargetAddress Addr,
                                               std::vector<std::string> Args)> {
  public:
    static const char *getName() { return "CallMain"; }
  };

  class CallVoidVoid
      : public rpc::Function<CallVoidVoid, void(JITTargetAddress FnAddr)> {
  public:
    static const char *getName() { return "CallVoidVoid"; }
  };

  class CreateRemoteAllocator
      : public rpc::Function<CreateRemoteAllocator,
                             void(ResourceIdMgr::ResourceId AllocatorID)> {
  public:
    static const char *getName() { return "CreateRemoteAllocator"; }
  };

  class CreateIndirectStubsOwner
      : public rpc::Function<CreateIndirectStubsOwner,
                             void(ResourceIdMgr::ResourceId StubOwnerID)> {
  public:
    static const char *getName() { return "CreateIndirectStubsOwner"; }
  };

  class DeregisterEHFrames
      : public rpc::Function<DeregisterEHFrames,
                             void(JITTargetAddress Addr, uint32_t Size)> {
  public:
    static const char *getName() { return "DeregisterEHFrames"; }
  };

  class DestroyRemoteAllocator
      : public rpc::Function<DestroyRemoteAllocator,
                             void(ResourceIdMgr::ResourceId AllocatorID)> {
  public:
    static const char *getName() { return "DestroyRemoteAllocator"; }
  };

  class DestroyIndirectStubsOwner
      : public rpc::Function<DestroyIndirectStubsOwner,
                             void(ResourceIdMgr::ResourceId StubsOwnerID)> {
  public:
    static const char *getName() { return "DestroyIndirectStubsOwner"; }
  };

  /// EmitIndirectStubs result is (StubsBase, PtrsBase, NumStubsEmitted).
  class EmitIndirectStubs
      : public rpc::Function<
            EmitIndirectStubs,
            std::tuple<JITTargetAddress, JITTargetAddress, uint32_t>(
                ResourceIdMgr::ResourceId StubsOwnerID,
                uint32_t NumStubsRequired)> {
  public:
    static const char *getName() { return "EmitIndirectStubs"; }
  };

  class EmitResolverBlock : public rpc::Function<EmitResolverBlock, void()> {
  public:
    static const char *getName() { return "EmitResolverBlock"; }
  };

  /// EmitTrampolineBlock result is (BlockAddr, NumTrampolines).
  class EmitTrampolineBlock
      : public rpc::Function<EmitTrampolineBlock,
                             std::tuple<JITTargetAddress, uint32_t>()> {
  public:
    static const char *getName() { return "EmitTrampolineBlock"; }
  };

  class GetSymbolAddress
      : public rpc::Function<GetSymbolAddress,
                             JITTargetAddress(std::string SymbolName)> {
  public:
    static const char *getName() { return "GetSymbolAddress"; }
  };

  /// GetRemoteInfo result is (Triple, PointerSize, PageSize, TrampolineSize,
  ///                          IndirectStubsSize).
  class GetRemoteInfo
      : public rpc::Function<
            GetRemoteInfo,
            std::tuple<std::string, uint32_t, uint32_t, uint32_t, uint32_t>()> {
  public:
    static const char *getName() { return "GetRemoteInfo"; }
  };

  class ReadMem
      : public rpc::Function<ReadMem, std::vector<uint8_t>(JITTargetAddress Src,
                                                           uint64_t Size)> {
  public:
    static const char *getName() { return "ReadMem"; }
  };

  class RegisterEHFrames
      : public rpc::Function<RegisterEHFrames,
                             void(JITTargetAddress Addr, uint32_t Size)> {
  public:
    static const char *getName() { return "RegisterEHFrames"; }
  };

  class ReserveMem
      : public rpc::Function<ReserveMem,
                             JITTargetAddress(ResourceIdMgr::ResourceId AllocID,
                                              uint64_t Size, uint32_t Align)> {
  public:
    static const char *getName() { return "ReserveMem"; }
  };

  class RequestCompile
      : public rpc::Function<
            RequestCompile, JITTargetAddress(JITTargetAddress TrampolineAddr)> {
  public:
    static const char *getName() { return "RequestCompile"; }
  };

  class SetProtections
      : public rpc::Function<SetProtections,
                             void(ResourceIdMgr::ResourceId AllocID,
                                  JITTargetAddress Dst, uint32_t ProtFlags)> {
  public:
    static const char *getName() { return "SetProtections"; }
  };

  class TerminateSession : public rpc::Function<TerminateSession, void()> {
  public:
    static const char *getName() { return "TerminateSession"; }
  };

  class WriteMem
      : public rpc::Function<WriteMem, void(remote::DirectBufferWriter DB)> {
  public:
    static const char *getName() { return "WriteMem"; }
  };

  class WritePtr : public rpc::Function<WritePtr, void(JITTargetAddress Dst,
                                                       JITTargetAddress Val)> {
  public:
    static const char *getName() { return "WritePtr"; }
  };
};

} // end namespace remote
} // end namespace orc
} // end namespace llvm

#endif
