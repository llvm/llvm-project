//===---- OrcRemoteTargetServer.h - Orc Remote-target Server ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the OrcRemoteTargetServer class. It can be used to build a
// JIT server that can execute code sent from an OrcRemoteTargetClient.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_ORCREMOTETARGETSERVER_H
#define LLVM_EXECUTIONENGINE_ORC_ORCREMOTETARGETSERVER_H

#include "OrcRemoteTargetRPCAPI.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

#define DEBUG_TYPE "orc-remote"

namespace llvm {
namespace orc {
namespace remote {

template <typename ChannelT, typename TargetT>
class OrcRemoteTargetServer : public OrcRemoteTargetRPCAPI {
public:
  typedef std::function<TargetAddress(const std::string &Name)>
      SymbolLookupFtor;

  OrcRemoteTargetServer(ChannelT &Channel, SymbolLookupFtor SymbolLookup)
      : Channel(Channel), SymbolLookup(std::move(SymbolLookup)) {}

  std::error_code getNextProcId(JITProcId &Id) {
    return deserialize(Channel, Id);
  }

  std::error_code handleKnownProcedure(JITProcId Id) {
    DEBUG(dbgs() << "Handling known proc: " << getJITProcIdName(Id) << "\n");

    switch (Id) {
    case CallIntVoidId:
      return handleCallIntVoid();
    case CallMainId:
      return handleCallMain();
    case CallVoidVoidId:
      return handleCallVoidVoid();
    case CreateRemoteAllocatorId:
      return handleCreateRemoteAllocator();
    case CreateIndirectStubsOwnerId:
      return handleCreateIndirectStubsOwner();
    case DestroyRemoteAllocatorId:
      return handleDestroyRemoteAllocator();
    case EmitIndirectStubsId:
      return handleEmitIndirectStubs();
    case EmitResolverBlockId:
      return handleEmitResolverBlock();
    case EmitTrampolineBlockId:
      return handleEmitTrampolineBlock();
    case GetSymbolAddressId:
      return handleGetSymbolAddress();
    case GetRemoteInfoId:
      return handleGetRemoteInfo();
    case ReadMemId:
      return handleReadMem();
    case ReserveMemId:
      return handleReserveMem();
    case SetProtectionsId:
      return handleSetProtections();
    case WriteMemId:
      return handleWriteMem();
    case WritePtrId:
      return handleWritePtr();
    default:
      return orcError(OrcErrorCode::UnexpectedRPCCall);
    }

    llvm_unreachable("Unhandled JIT RPC procedure Id.");
  }

  std::error_code requestCompile(TargetAddress &CompiledFnAddr,
                                 TargetAddress TrampolineAddr) {
    if (auto EC = call<RequestCompile>(Channel, TrampolineAddr))
      return EC;

    while (1) {
      JITProcId Id = InvalidId;
      if (auto EC = getNextProcId(Id))
        return EC;

      switch (Id) {
      case RequestCompileResponseId:
        return handle<RequestCompileResponse>(Channel,
                                              readArgs(CompiledFnAddr));
      default:
        if (auto EC = handleKnownProcedure(Id))
          return EC;
      }
    }

    llvm_unreachable("Fell through request-compile command loop.");
  }

private:
  struct Allocator {
    Allocator() = default;
    Allocator(Allocator &&) = default;
    Allocator &operator=(Allocator &&) = default;

    ~Allocator() {
      for (auto &Alloc : Allocs)
        sys::Memory::releaseMappedMemory(Alloc.second);
    }

    std::error_code allocate(void *&Addr, size_t Size, uint32_t Align) {
      std::error_code EC;
      sys::MemoryBlock MB = sys::Memory::allocateMappedMemory(
          Size, nullptr, sys::Memory::MF_READ | sys::Memory::MF_WRITE, EC);
      if (EC)
        return EC;

      Addr = MB.base();
      assert(Allocs.find(MB.base()) == Allocs.end() && "Duplicate alloc");
      Allocs[MB.base()] = std::move(MB);
      return std::error_code();
    }

    std::error_code setProtections(void *block, unsigned Flags) {
      auto I = Allocs.find(block);
      if (I == Allocs.end())
        return orcError(OrcErrorCode::RemoteMProtectAddrUnrecognized);
      return sys::Memory::protectMappedMemory(I->second, Flags);
    }

  private:
    std::map<void *, sys::MemoryBlock> Allocs;
  };

  static std::error_code doNothing() { return std::error_code(); }

  static TargetAddress reenter(void *JITTargetAddr, void *TrampolineAddr) {
    TargetAddress CompiledFnAddr = 0;

    auto T = static_cast<OrcRemoteTargetServer *>(JITTargetAddr);
    auto EC = T->requestCompile(
        CompiledFnAddr, static_cast<TargetAddress>(
                            reinterpret_cast<uintptr_t>(TrampolineAddr)));
    assert(!EC && "Compile request failed");
    return CompiledFnAddr;
  }

  std::error_code handleCallIntVoid() {
    typedef int (*IntVoidFnTy)();

    IntVoidFnTy Fn = nullptr;
    if (auto EC = handle<CallIntVoid>(Channel, [&](TargetAddress Addr) {
          Fn = reinterpret_cast<IntVoidFnTy>(static_cast<uintptr_t>(Addr));
          return std::error_code();
        }))
      return EC;

    DEBUG(dbgs() << "  Calling " << reinterpret_cast<void *>(Fn) << "\n");
    int Result = Fn();
    DEBUG(dbgs() << "  Result = " << Result << "\n");

    return call<CallIntVoidResponse>(Channel, Result);
  }

  std::error_code handleCallMain() {
    typedef int (*MainFnTy)(int, const char *[]);

    MainFnTy Fn = nullptr;
    std::vector<std::string> Args;
    if (auto EC = handle<CallMain>(
            Channel, [&](TargetAddress Addr, std::vector<std::string> &A) {
              Fn = reinterpret_cast<MainFnTy>(static_cast<uintptr_t>(Addr));
              Args = std::move(A);
              return std::error_code();
            }))
      return EC;

    int ArgC = Args.size() + 1;
    int Idx = 1;
    std::unique_ptr<const char *[]> ArgV(new const char *[ArgC + 1]);
    ArgV[0] = "<jit process>";
    for (auto &Arg : Args)
      ArgV[Idx++] = Arg.c_str();

    DEBUG(dbgs() << "  Calling " << reinterpret_cast<void *>(Fn) << "\n");
    int Result = Fn(ArgC, ArgV.get());
    DEBUG(dbgs() << "  Result = " << Result << "\n");

    return call<CallMainResponse>(Channel, Result);
  }

  std::error_code handleCallVoidVoid() {
    typedef void (*VoidVoidFnTy)();

    VoidVoidFnTy Fn = nullptr;
    if (auto EC = handle<CallIntVoid>(Channel, [&](TargetAddress Addr) {
          Fn = reinterpret_cast<VoidVoidFnTy>(static_cast<uintptr_t>(Addr));
          return std::error_code();
        }))
      return EC;

    DEBUG(dbgs() << "  Calling " << reinterpret_cast<void *>(Fn) << "\n");
    Fn();
    DEBUG(dbgs() << "  Complete.\n");

    return call<CallVoidVoidResponse>(Channel);
  }

  std::error_code handleCreateRemoteAllocator() {
    return handle<CreateRemoteAllocator>(
        Channel, [&](ResourceIdMgr::ResourceId Id) {
          auto I = Allocators.find(Id);
          if (I != Allocators.end())
            return orcError(OrcErrorCode::RemoteAllocatorIdAlreadyInUse);
          DEBUG(dbgs() << "  Created allocator " << Id << "\n");
          Allocators[Id] = Allocator();
          return std::error_code();
        });
  }

  std::error_code handleCreateIndirectStubsOwner() {
    return handle<CreateIndirectStubsOwner>(
        Channel, [&](ResourceIdMgr::ResourceId Id) {
          auto I = IndirectStubsOwners.find(Id);
          if (I != IndirectStubsOwners.end())
            return orcError(
                OrcErrorCode::RemoteIndirectStubsOwnerIdAlreadyInUse);
          DEBUG(dbgs() << "  Create indirect stubs owner " << Id << "\n");
          IndirectStubsOwners[Id] = ISBlockOwnerList();
          return std::error_code();
        });
  }

  std::error_code handleDestroyRemoteAllocator() {
    return handle<DestroyRemoteAllocator>(
        Channel, [&](ResourceIdMgr::ResourceId Id) {
          auto I = Allocators.find(Id);
          if (I == Allocators.end())
            return orcError(OrcErrorCode::RemoteAllocatorDoesNotExist);
          Allocators.erase(I);
          DEBUG(dbgs() << "  Destroyed allocator " << Id << "\n");
          return std::error_code();
        });
  }

  std::error_code handleDestroyIndirectStubsOwner() {
    return handle<DestroyIndirectStubsOwner>(
        Channel, [&](ResourceIdMgr::ResourceId Id) {
          auto I = IndirectStubsOwners.find(Id);
          if (I == IndirectStubsOwners.end())
            return orcError(OrcErrorCode::RemoteIndirectStubsOwnerDoesNotExist);
          IndirectStubsOwners.erase(I);
          return std::error_code();
        });
  }

  std::error_code handleEmitIndirectStubs() {
    ResourceIdMgr::ResourceId ISOwnerId = ~0U;
    uint32_t NumStubsRequired = 0;

    if (auto EC = handle<EmitIndirectStubs>(
            Channel, readArgs(ISOwnerId, NumStubsRequired)))
      return EC;

    DEBUG(dbgs() << "  ISMgr " << ISOwnerId << " request " << NumStubsRequired
                 << " stubs.\n");

    auto StubOwnerItr = IndirectStubsOwners.find(ISOwnerId);
    if (StubOwnerItr == IndirectStubsOwners.end())
      return orcError(OrcErrorCode::RemoteIndirectStubsOwnerDoesNotExist);

    typename TargetT::IndirectStubsInfo IS;
    if (auto EC =
            TargetT::emitIndirectStubsBlock(IS, NumStubsRequired, nullptr))
      return EC;

    TargetAddress StubsBase =
        static_cast<TargetAddress>(reinterpret_cast<uintptr_t>(IS.getStub(0)));
    TargetAddress PtrsBase =
        static_cast<TargetAddress>(reinterpret_cast<uintptr_t>(IS.getPtr(0)));
    uint32_t NumStubsEmitted = IS.getNumStubs();

    auto &BlockList = StubOwnerItr->second;
    BlockList.push_back(std::move(IS));

    return call<EmitIndirectStubsResponse>(Channel, StubsBase, PtrsBase,
                                           NumStubsEmitted);
  }

  std::error_code handleEmitResolverBlock() {
    if (auto EC = handle<EmitResolverBlock>(Channel, doNothing))
      return EC;

    std::error_code EC;
    ResolverBlock = sys::OwningMemoryBlock(sys::Memory::allocateMappedMemory(
        TargetT::ResolverCodeSize, nullptr,
        sys::Memory::MF_READ | sys::Memory::MF_WRITE, EC));
    if (EC)
      return EC;

    TargetT::writeResolverCode(static_cast<uint8_t *>(ResolverBlock.base()),
                               &reenter, this);

    return sys::Memory::protectMappedMemory(ResolverBlock.getMemoryBlock(),
                                            sys::Memory::MF_READ |
                                                sys::Memory::MF_EXEC);
  }

  std::error_code handleEmitTrampolineBlock() {
    if (auto EC = handle<EmitTrampolineBlock>(Channel, doNothing))
      return EC;

    std::error_code EC;

    auto TrampolineBlock =
        sys::OwningMemoryBlock(sys::Memory::allocateMappedMemory(
            sys::Process::getPageSize(), nullptr,
            sys::Memory::MF_READ | sys::Memory::MF_WRITE, EC));
    if (EC)
      return EC;

    unsigned NumTrampolines =
        (sys::Process::getPageSize() - TargetT::PointerSize) /
        TargetT::TrampolineSize;

    uint8_t *TrampolineMem = static_cast<uint8_t *>(TrampolineBlock.base());
    TargetT::writeTrampolines(TrampolineMem, ResolverBlock.base(),
                              NumTrampolines);

    EC = sys::Memory::protectMappedMemory(TrampolineBlock.getMemoryBlock(),
                                          sys::Memory::MF_READ |
                                              sys::Memory::MF_EXEC);

    TrampolineBlocks.push_back(std::move(TrampolineBlock));

    return call<EmitTrampolineBlockResponse>(
        Channel,
        static_cast<TargetAddress>(reinterpret_cast<uintptr_t>(TrampolineMem)),
        NumTrampolines);
  }

  std::error_code handleGetSymbolAddress() {
    std::string SymbolName;
    if (auto EC = handle<GetSymbolAddress>(Channel, readArgs(SymbolName)))
      return EC;

    TargetAddress SymbolAddr = SymbolLookup(SymbolName);
    DEBUG(dbgs() << "  Symbol '" << SymbolName
                 << "' =  " << format("0x%016x", SymbolAddr) << "\n");
    return call<GetSymbolAddressResponse>(Channel, SymbolAddr);
  }

  std::error_code handleGetRemoteInfo() {
    if (auto EC = handle<GetRemoteInfo>(Channel, doNothing))
      return EC;

    std::string ProcessTriple = sys::getProcessTriple();
    uint32_t PointerSize = TargetT::PointerSize;
    uint32_t PageSize = sys::Process::getPageSize();
    uint32_t TrampolineSize = TargetT::TrampolineSize;
    uint32_t IndirectStubSize = TargetT::IndirectStubsInfo::StubSize;
    DEBUG(dbgs() << "  Remote info:\n"
                 << "    triple             = '" << ProcessTriple << "'\n"
                 << "    pointer size       = " << PointerSize << "\n"
                 << "    page size          = " << PageSize << "\n"
                 << "    trampoline size    = " << TrampolineSize << "\n"
                 << "    indirect stub size = " << IndirectStubSize << "\n");
    return call<GetRemoteInfoResponse>(Channel, ProcessTriple, PointerSize,
                                       PageSize, TrampolineSize,
                                       IndirectStubSize);
  }

  std::error_code handleReadMem() {
    char *Src = nullptr;
    uint64_t Size = 0;
    if (auto EC =
            handle<ReadMem>(Channel, [&](TargetAddress RSrc, uint64_t RSize) {
              Src = reinterpret_cast<char *>(static_cast<uintptr_t>(RSrc));
              Size = RSize;
              return std::error_code();
            }))
      return EC;

    DEBUG(dbgs() << "  Reading " << Size << " bytes from "
                 << static_cast<void *>(Src) << "\n");

    if (auto EC = call<ReadMemResponse>(Channel))
      return EC;

    if (auto EC = Channel.appendBytes(Src, Size))
      return EC;

    return Channel.send();
  }

  std::error_code handleReserveMem() {
    void *LocalAllocAddr = nullptr;

    if (auto EC =
            handle<ReserveMem>(Channel, [&](ResourceIdMgr::ResourceId Id,
                                            uint64_t Size, uint32_t Align) {
              auto I = Allocators.find(Id);
              if (I == Allocators.end())
                return orcError(OrcErrorCode::RemoteAllocatorDoesNotExist);
              auto &Allocator = I->second;
              auto EC2 = Allocator.allocate(LocalAllocAddr, Size, Align);
              DEBUG(dbgs() << "  Allocator " << Id << " reserved "
                           << LocalAllocAddr << " (" << Size
                           << " bytes, alignment " << Align << ")\n");
              return EC2;
            }))
      return EC;

    TargetAddress AllocAddr =
        static_cast<TargetAddress>(reinterpret_cast<uintptr_t>(LocalAllocAddr));

    return call<ReserveMemResponse>(Channel, AllocAddr);
  }

  std::error_code handleSetProtections() {
    return handle<ReserveMem>(Channel, [&](ResourceIdMgr::ResourceId Id,
                                           TargetAddress Addr, uint32_t Flags) {
      auto I = Allocators.find(Id);
      if (I == Allocators.end())
        return orcError(OrcErrorCode::RemoteAllocatorDoesNotExist);
      auto &Allocator = I->second;
      void *LocalAddr = reinterpret_cast<void *>(static_cast<uintptr_t>(Addr));
      DEBUG(dbgs() << "  Allocator " << Id << " set permissions on "
                   << LocalAddr << " to "
                   << (Flags & sys::Memory::MF_READ ? 'R' : '-')
                   << (Flags & sys::Memory::MF_WRITE ? 'W' : '-')
                   << (Flags & sys::Memory::MF_EXEC ? 'X' : '-') << "\n");
      return Allocator.setProtections(LocalAddr, Flags);
    });
  }

  std::error_code handleWriteMem() {
    return handle<WriteMem>(Channel, [&](TargetAddress RDst, uint64_t Size) {
      char *Dst = reinterpret_cast<char *>(static_cast<uintptr_t>(RDst));
      return Channel.readBytes(Dst, Size);
    });
  }

  std::error_code handleWritePtr() {
    return handle<WritePtr>(
        Channel, [&](TargetAddress Addr, TargetAddress PtrVal) {
          uintptr_t *Ptr =
              reinterpret_cast<uintptr_t *>(static_cast<uintptr_t>(Addr));
          *Ptr = static_cast<uintptr_t>(PtrVal);
          return std::error_code();
        });
  }

  ChannelT &Channel;
  SymbolLookupFtor SymbolLookup;
  std::map<ResourceIdMgr::ResourceId, Allocator> Allocators;
  typedef std::vector<typename TargetT::IndirectStubsInfo> ISBlockOwnerList;
  std::map<ResourceIdMgr::ResourceId, ISBlockOwnerList> IndirectStubsOwners;
  sys::OwningMemoryBlock ResolverBlock;
  std::vector<sys::OwningMemoryBlock> TrampolineBlocks;
};

} // end namespace remote
} // end namespace orc
} // end namespace llvm

#undef DEBUG_TYPE

#endif
