//===---- OrcRemoteTargetClient.h - Orc Remote-target Client ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the OrcRemoteTargetClient class and helpers. This class
// can be used to communicate over an RPCChannel with an OrcRemoteTargetServer
// instance to support remote-JITing.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_ORCREMOTETARGETCLIENT_H
#define LLVM_EXECUTIONENGINE_ORC_ORCREMOTETARGETCLIENT_H

#include "IndirectionUtils.h"
#include "OrcRemoteTargetRPCAPI.h"
#include <system_error>

#define DEBUG_TYPE "orc-remote"

namespace llvm {
namespace orc {
namespace remote {

/// This class provides utilities (including memory manager, indirect stubs
/// manager, and compile callback manager types) that support remote JITing
/// in ORC.
///
/// Each of the utility classes talks to a JIT server (an instance of the
/// OrcRemoteTargetServer class) via an RPC system (see RPCUtils.h) to carry out
/// its actions.
template <typename ChannelT>
class OrcRemoteTargetClient : public OrcRemoteTargetRPCAPI {
public:

  /// Remote memory manager.
  class RCMemoryManager : public RuntimeDyld::MemoryManager {
  public:
    RCMemoryManager(OrcRemoteTargetClient &Client, ResourceIdMgr::ResourceId Id)
        : Client(Client), Id(Id) {
      DEBUG(dbgs() << "Created remote allocator " << Id << "\n");
    }

    RCMemoryManager(RCMemoryManager &&Other)
        : Client(std::move(Other.Client)), Id(std::move(Other.Id)),
          Unmapped(std::move(Other.Unmapped)),
          Unfinalized(std::move(Other.Unfinalized)) {}

    RCMemoryManager operator=(RCMemoryManager &&Other) {
      Client = std::move(Other.Client);
      Id = std::move(Other.Id);
      Unmapped = std::move(Other.Unmapped);
      Unfinalized = std::move(Other.Unfinalized);
      return *this;
    }

    ~RCMemoryManager() override {
      Client.destroyRemoteAllocator(Id);
      DEBUG(dbgs() << "Destroyed remote allocator " << Id << "\n");
    }

    uint8_t *allocateCodeSection(uintptr_t Size, unsigned Alignment,
                                 unsigned SectionID,
                                 StringRef SectionName) override {
      Unmapped.back().CodeAllocs.emplace_back(Size, Alignment);
      uint8_t *Alloc = reinterpret_cast<uint8_t *>(
          Unmapped.back().CodeAllocs.back().getLocalAddress());
      DEBUG(dbgs() << "Allocator " << Id << " allocated code for "
                   << SectionName << ": " << Alloc << " (" << Size
                   << " bytes, alignment " << Alignment << ")\n");
      return Alloc;
    }

    uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                 unsigned SectionID, StringRef SectionName,
                                 bool IsReadOnly) override {
      if (IsReadOnly) {
        Unmapped.back().RODataAllocs.emplace_back(Size, Alignment);
        uint8_t *Alloc = reinterpret_cast<uint8_t *>(
            Unmapped.back().RODataAllocs.back().getLocalAddress());
        DEBUG(dbgs() << "Allocator " << Id << " allocated ro-data for "
                     << SectionName << ": " << Alloc << " (" << Size
                     << " bytes, alignment " << Alignment << ")\n");
        return Alloc;
      } // else...

      Unmapped.back().RWDataAllocs.emplace_back(Size, Alignment);
      uint8_t *Alloc = reinterpret_cast<uint8_t *>(
          Unmapped.back().RWDataAllocs.back().getLocalAddress());
      DEBUG(dbgs() << "Allocator " << Id << " allocated rw-data for "
                   << SectionName << ": " << Alloc << " (" << Size
                   << " bytes, alignment " << Alignment << ")\n");
      return Alloc;
    }

    void reserveAllocationSpace(uintptr_t CodeSize, uint32_t CodeAlign,
                                uintptr_t RODataSize, uint32_t RODataAlign,
                                uintptr_t RWDataSize,
                                uint32_t RWDataAlign) override {
      Unmapped.push_back(ObjectAllocs());

      DEBUG(dbgs() << "Allocator " << Id << " reserved:\n");

      if (CodeSize != 0) {
	if (auto AddrOrErr = Client.reserveMem(Id, CodeSize, CodeAlign))
	  Unmapped.back().RemoteCodeAddr = *AddrOrErr;
	else {
	  // FIXME; Add error to poll.
	  assert(!AddrOrErr.getError() && "Failed reserving remote memory.");
	}

        DEBUG(dbgs() << "  code: "
                     << format("0x%016x", Unmapped.back().RemoteCodeAddr)
                     << " (" << CodeSize << " bytes, alignment " << CodeAlign
                     << ")\n");
      }

      if (RODataSize != 0) {
        if (auto AddrOrErr = Client.reserveMem(Id, RODataSize, RODataAlign))
	  Unmapped.back().RemoteRODataAddr = *AddrOrErr;
	else {
	  // FIXME; Add error to poll.
	  assert(!AddrOrErr.getError() && "Failed reserving remote memory.");
	}
                                               
        DEBUG(dbgs() << "  ro-data: "
                     << format("0x%016x", Unmapped.back().RemoteRODataAddr)
                     << " (" << RODataSize << " bytes, alignment "
                     << RODataAlign << ")\n");
      }

      if (RWDataSize != 0) {
        if (auto AddrOrErr = Client.reserveMem(Id, RWDataSize, RWDataAlign))
	  Unmapped.back().RemoteRWDataAddr = *AddrOrErr;
	else {
	  // FIXME; Add error to poll.
	  assert(!AddrOrErr.getError() && "Failed reserving remote memory.");
        }

        DEBUG(dbgs() << "  rw-data: "
                     << format("0x%016x", Unmapped.back().RemoteRWDataAddr)
                     << " (" << RWDataSize << " bytes, alignment "
                     << RWDataAlign << ")\n");
      }
    }

    bool needsToReserveAllocationSpace() override { return true; }

    void registerEHFrames(uint8_t *Addr, uint64_t LoadAddr,
                          size_t Size) override {
      UnfinalizedEHFrames.push_back(
          std::make_pair(LoadAddr, static_cast<uint32_t>(Size)));
    }

    void deregisterEHFrames(uint8_t *Addr, uint64_t LoadAddr,
                            size_t Size) override {
      auto EC = Client.deregisterEHFrames(LoadAddr, Size);
      // FIXME: Add error poll.
      assert(!EC && "Failed to register remote EH frames.");
      (void)EC;
    }

    void notifyObjectLoaded(RuntimeDyld &Dyld,
                            const object::ObjectFile &Obj) override {
      DEBUG(dbgs() << "Allocator " << Id << " applied mappings:\n");
      for (auto &ObjAllocs : Unmapped) {
        {
          TargetAddress NextCodeAddr = ObjAllocs.RemoteCodeAddr;
          for (auto &Alloc : ObjAllocs.CodeAllocs) {
            NextCodeAddr = alignTo(NextCodeAddr, Alloc.getAlign());
            Dyld.mapSectionAddress(Alloc.getLocalAddress(), NextCodeAddr);
            DEBUG(dbgs() << "     code: "
                         << static_cast<void *>(Alloc.getLocalAddress())
                         << " -> " << format("0x%016x", NextCodeAddr) << "\n");
            Alloc.setRemoteAddress(NextCodeAddr);
            NextCodeAddr += Alloc.getSize();
          }
        }
        {
          TargetAddress NextRODataAddr = ObjAllocs.RemoteRODataAddr;
          for (auto &Alloc : ObjAllocs.RODataAllocs) {
            NextRODataAddr = alignTo(NextRODataAddr, Alloc.getAlign());
            Dyld.mapSectionAddress(Alloc.getLocalAddress(), NextRODataAddr);
            DEBUG(dbgs() << "  ro-data: "
                         << static_cast<void *>(Alloc.getLocalAddress())
                         << " -> " << format("0x%016x", NextRODataAddr)
                         << "\n");
            Alloc.setRemoteAddress(NextRODataAddr);
            NextRODataAddr += Alloc.getSize();
          }
        }
        {
          TargetAddress NextRWDataAddr = ObjAllocs.RemoteRWDataAddr;
          for (auto &Alloc : ObjAllocs.RWDataAllocs) {
            NextRWDataAddr = alignTo(NextRWDataAddr, Alloc.getAlign());
            Dyld.mapSectionAddress(Alloc.getLocalAddress(), NextRWDataAddr);
            DEBUG(dbgs() << "  rw-data: "
                         << static_cast<void *>(Alloc.getLocalAddress())
                         << " -> " << format("0x%016x", NextRWDataAddr)
                         << "\n");
            Alloc.setRemoteAddress(NextRWDataAddr);
            NextRWDataAddr += Alloc.getSize();
          }
        }
        Unfinalized.push_back(std::move(ObjAllocs));
      }
      Unmapped.clear();
    }

    bool finalizeMemory(std::string *ErrMsg = nullptr) override {
      DEBUG(dbgs() << "Allocator " << Id << " finalizing:\n");

      for (auto &ObjAllocs : Unfinalized) {

        for (auto &Alloc : ObjAllocs.CodeAllocs) {
          DEBUG(dbgs() << "  copying code: "
                       << static_cast<void *>(Alloc.getLocalAddress()) << " -> "
                       << format("0x%016x", Alloc.getRemoteAddress()) << " ("
                       << Alloc.getSize() << " bytes)\n");
          Client.writeMem(Alloc.getRemoteAddress(), Alloc.getLocalAddress(),
                          Alloc.getSize());
        }

        if (ObjAllocs.RemoteCodeAddr) {
          DEBUG(dbgs() << "  setting R-X permissions on code block: "
                       << format("0x%016x", ObjAllocs.RemoteCodeAddr) << "\n");
          Client.setProtections(Id, ObjAllocs.RemoteCodeAddr,
                                sys::Memory::MF_READ | sys::Memory::MF_EXEC);
        }

        for (auto &Alloc : ObjAllocs.RODataAllocs) {
          DEBUG(dbgs() << "  copying ro-data: "
                       << static_cast<void *>(Alloc.getLocalAddress()) << " -> "
                       << format("0x%016x", Alloc.getRemoteAddress()) << " ("
                       << Alloc.getSize() << " bytes)\n");
          Client.writeMem(Alloc.getRemoteAddress(), Alloc.getLocalAddress(),
                          Alloc.getSize());
        }

        if (ObjAllocs.RemoteRODataAddr) {
          DEBUG(dbgs() << "  setting R-- permissions on ro-data block: "
                       << format("0x%016x", ObjAllocs.RemoteRODataAddr)
                       << "\n");
          Client.setProtections(Id, ObjAllocs.RemoteRODataAddr,
                                sys::Memory::MF_READ);
        }

        for (auto &Alloc : ObjAllocs.RWDataAllocs) {
          DEBUG(dbgs() << "  copying rw-data: "
                       << static_cast<void *>(Alloc.getLocalAddress()) << " -> "
                       << format("0x%016x", Alloc.getRemoteAddress()) << " ("
                       << Alloc.getSize() << " bytes)\n");
          Client.writeMem(Alloc.getRemoteAddress(), Alloc.getLocalAddress(),
                          Alloc.getSize());
        }

        if (ObjAllocs.RemoteRWDataAddr) {
          DEBUG(dbgs() << "  setting RW- permissions on rw-data block: "
                       << format("0x%016x", ObjAllocs.RemoteRWDataAddr)
                       << "\n");
          Client.setProtections(Id, ObjAllocs.RemoteRWDataAddr,
                                sys::Memory::MF_READ | sys::Memory::MF_WRITE);
        }
      }
      Unfinalized.clear();

      for (auto &EHFrame : UnfinalizedEHFrames) {
        auto EC = Client.registerEHFrames(EHFrame.first, EHFrame.second);
        // FIXME: Add error poll.
        assert(!EC && "Failed to register remote EH frames.");
        (void)EC;
      }
      UnfinalizedEHFrames.clear();

      return false;
    }

  private:
    class Alloc {
    public:
      Alloc(uint64_t Size, unsigned Align)
          : Size(Size), Align(Align), Contents(new char[Size + Align - 1]) {}

      Alloc(Alloc &&Other)
          : Size(std::move(Other.Size)), Align(std::move(Other.Align)),
            Contents(std::move(Other.Contents)),
            RemoteAddr(std::move(Other.RemoteAddr)) {}

      Alloc &operator=(Alloc &&Other) {
        Size = std::move(Other.Size);
        Align = std::move(Other.Align);
        Contents = std::move(Other.Contents);
        RemoteAddr = std::move(Other.RemoteAddr);
        return *this;
      }

      uint64_t getSize() const { return Size; }

      unsigned getAlign() const { return Align; }

      char *getLocalAddress() const {
        uintptr_t LocalAddr = reinterpret_cast<uintptr_t>(Contents.get());
        LocalAddr = alignTo(LocalAddr, Align);
        return reinterpret_cast<char *>(LocalAddr);
      }

      void setRemoteAddress(TargetAddress RemoteAddr) {
        this->RemoteAddr = RemoteAddr;
      }

      TargetAddress getRemoteAddress() const { return RemoteAddr; }

    private:
      uint64_t Size;
      unsigned Align;
      std::unique_ptr<char[]> Contents;
      TargetAddress RemoteAddr = 0;
    };

    struct ObjectAllocs {
      ObjectAllocs() = default;

      ObjectAllocs(ObjectAllocs &&Other)
          : RemoteCodeAddr(std::move(Other.RemoteCodeAddr)),
            RemoteRODataAddr(std::move(Other.RemoteRODataAddr)),
            RemoteRWDataAddr(std::move(Other.RemoteRWDataAddr)),
            CodeAllocs(std::move(Other.CodeAllocs)),
            RODataAllocs(std::move(Other.RODataAllocs)),
            RWDataAllocs(std::move(Other.RWDataAllocs)) {}

      ObjectAllocs &operator=(ObjectAllocs &&Other) {
        RemoteCodeAddr = std::move(Other.RemoteCodeAddr);
        RemoteRODataAddr = std::move(Other.RemoteRODataAddr);
        RemoteRWDataAddr = std::move(Other.RemoteRWDataAddr);
        CodeAllocs = std::move(Other.CodeAllocs);
        RODataAllocs = std::move(Other.RODataAllocs);
        RWDataAllocs = std::move(Other.RWDataAllocs);
        return *this;
      }

      TargetAddress RemoteCodeAddr = 0;
      TargetAddress RemoteRODataAddr = 0;
      TargetAddress RemoteRWDataAddr = 0;
      std::vector<Alloc> CodeAllocs, RODataAllocs, RWDataAllocs;
    };

    OrcRemoteTargetClient &Client;
    ResourceIdMgr::ResourceId Id;
    std::vector<ObjectAllocs> Unmapped;
    std::vector<ObjectAllocs> Unfinalized;
    std::vector<std::pair<uint64_t, uint32_t>> UnfinalizedEHFrames;
  };

  /// Remote indirect stubs manager.
  class RCIndirectStubsManager : public IndirectStubsManager {
  public:
    RCIndirectStubsManager(OrcRemoteTargetClient &Remote,
                           ResourceIdMgr::ResourceId Id)
        : Remote(Remote), Id(Id) {}

    ~RCIndirectStubsManager() override {
      Remote.destroyIndirectStubsManager(Id);
    }

    std::error_code createStub(StringRef StubName, TargetAddress StubAddr,
                               JITSymbolFlags StubFlags) override {
      if (auto EC = reserveStubs(1))
        return EC;

      return createStubInternal(StubName, StubAddr, StubFlags);
    }

    std::error_code createStubs(const StubInitsMap &StubInits) override {
      if (auto EC = reserveStubs(StubInits.size()))
        return EC;

      for (auto &Entry : StubInits)
        if (auto EC = createStubInternal(Entry.first(), Entry.second.first,
                                         Entry.second.second))
          return EC;

      return std::error_code();
    }

    JITSymbol findStub(StringRef Name, bool ExportedStubsOnly) override {
      auto I = StubIndexes.find(Name);
      if (I == StubIndexes.end())
        return nullptr;
      auto Key = I->second.first;
      auto Flags = I->second.second;
      auto StubSymbol = JITSymbol(getStubAddr(Key), Flags);
      if (ExportedStubsOnly && !StubSymbol.isExported())
        return nullptr;
      return StubSymbol;
    }

    JITSymbol findPointer(StringRef Name) override {
      auto I = StubIndexes.find(Name);
      if (I == StubIndexes.end())
        return nullptr;
      auto Key = I->second.first;
      auto Flags = I->second.second;
      return JITSymbol(getPtrAddr(Key), Flags);
    }

    std::error_code updatePointer(StringRef Name,
                                  TargetAddress NewAddr) override {
      auto I = StubIndexes.find(Name);
      assert(I != StubIndexes.end() && "No stub pointer for symbol");
      auto Key = I->second.first;
      return Remote.writePointer(getPtrAddr(Key), NewAddr);
    }

  private:
    struct RemoteIndirectStubsInfo {
      TargetAddress StubBase;
      TargetAddress PtrBase;
      unsigned NumStubs;
    };

    OrcRemoteTargetClient &Remote;
    ResourceIdMgr::ResourceId Id;
    std::vector<RemoteIndirectStubsInfo> RemoteIndirectStubsInfos;
    typedef std::pair<uint16_t, uint16_t> StubKey;
    std::vector<StubKey> FreeStubs;
    StringMap<std::pair<StubKey, JITSymbolFlags>> StubIndexes;

    std::error_code reserveStubs(unsigned NumStubs) {
      if (NumStubs <= FreeStubs.size())
        return std::error_code();

      unsigned NewStubsRequired = NumStubs - FreeStubs.size();
      TargetAddress StubBase;
      TargetAddress PtrBase;
      unsigned NumStubsEmitted;

      if (auto StubInfoOrErr = Remote.emitIndirectStubs(Id, NewStubsRequired))
	std::tie(StubBase, PtrBase, NumStubsEmitted) = *StubInfoOrErr;
      else
	return StubInfoOrErr.getError();

      unsigned NewBlockId = RemoteIndirectStubsInfos.size();
      RemoteIndirectStubsInfos.push_back({StubBase, PtrBase, NumStubsEmitted});

      for (unsigned I = 0; I < NumStubsEmitted; ++I)
        FreeStubs.push_back(std::make_pair(NewBlockId, I));

      return std::error_code();
    }

    std::error_code createStubInternal(StringRef StubName,
                                       TargetAddress InitAddr,
                                       JITSymbolFlags StubFlags) {
      auto Key = FreeStubs.back();
      FreeStubs.pop_back();
      StubIndexes[StubName] = std::make_pair(Key, StubFlags);
      return Remote.writePointer(getPtrAddr(Key), InitAddr);
    }

    TargetAddress getStubAddr(StubKey K) {
      assert(RemoteIndirectStubsInfos[K.first].StubBase != 0 &&
             "Missing stub address");
      return RemoteIndirectStubsInfos[K.first].StubBase +
             K.second * Remote.getIndirectStubSize();
    }

    TargetAddress getPtrAddr(StubKey K) {
      assert(RemoteIndirectStubsInfos[K.first].PtrBase != 0 &&
             "Missing pointer address");
      return RemoteIndirectStubsInfos[K.first].PtrBase +
             K.second * Remote.getPointerSize();
    }
  };

  /// Remote compile callback manager.
  class RCCompileCallbackManager : public JITCompileCallbackManager {
  public:
    RCCompileCallbackManager(TargetAddress ErrorHandlerAddress,
                             OrcRemoteTargetClient &Remote)
        : JITCompileCallbackManager(ErrorHandlerAddress), Remote(Remote) {
      assert(!Remote.CompileCallback && "Compile callback already set");
      Remote.CompileCallback = [this](TargetAddress TrampolineAddr) {
        return executeCompileCallback(TrampolineAddr);
      };
      Remote.emitResolverBlock();
    }

  private:
    void grow() override {
      TargetAddress BlockAddr = 0;
      uint32_t NumTrampolines = 0;
      if (auto TrampolineInfoOrErr = Remote.emitTrampolineBlock())
	std::tie(BlockAddr, NumTrampolines) = *TrampolineInfoOrErr;
      else {
	// FIXME: Return error.
	llvm_unreachable("Failed to create trampolines");
      }

      uint32_t TrampolineSize = Remote.getTrampolineSize();
      for (unsigned I = 0; I < NumTrampolines; ++I)
        this->AvailableTrampolines.push_back(BlockAddr + (I * TrampolineSize));
    }

    OrcRemoteTargetClient &Remote;
  };

  /// Create an OrcRemoteTargetClient.
  /// Channel is the ChannelT instance to communicate on. It is assumed that
  /// the channel is ready to be read from and written to.
  static ErrorOr<OrcRemoteTargetClient> Create(ChannelT &Channel) {
    std::error_code EC;
    OrcRemoteTargetClient H(Channel, EC);
    if (EC)
      return EC;
    return ErrorOr<OrcRemoteTargetClient>(std::move(H));
  }

  /// Call the int(void) function at the given address in the target and return
  /// its result.
  ErrorOr<int> callIntVoid(TargetAddress Addr) {
    DEBUG(dbgs() << "Calling int(*)(void) " << format("0x%016x", Addr) << "\n");

    auto Listen =
      [&](RPCChannel &C, uint32_t Id) {
        return listenForCompileRequests(C, Id);
      };
    return callSTHandling<CallIntVoid>(Channel, Listen, Addr);
  }

  /// Call the int(int, char*[]) function at the given address in the target and
  /// return its result.
  ErrorOr<int> callMain(TargetAddress Addr,
			const std::vector<std::string> &Args) {
    DEBUG(dbgs() << "Calling int(*)(int, char*[]) " << format("0x%016x", Addr)
                 << "\n");

    auto Listen =
      [&](RPCChannel &C, uint32_t Id) {
        return listenForCompileRequests(C, Id);
      };
    return callSTHandling<CallMain>(Channel, Listen, Addr, Args);
  }

  /// Call the void() function at the given address in the target and wait for
  /// it to finish.
  std::error_code callVoidVoid(TargetAddress Addr) {
    DEBUG(dbgs() << "Calling void(*)(void) " << format("0x%016x", Addr)
                 << "\n");

    auto Listen =
      [&](RPCChannel &C, JITFuncId Id) {
        return listenForCompileRequests(C, Id);
      };
    return callSTHandling<CallVoidVoid>(Channel, Listen, Addr);
  }

  /// Create an RCMemoryManager which will allocate its memory on the remote
  /// target.
  std::error_code
  createRemoteMemoryManager(std::unique_ptr<RCMemoryManager> &MM) {
    assert(!MM && "MemoryManager should be null before creation.");

    auto Id = AllocatorIds.getNext();
    if (auto EC = callST<CreateRemoteAllocator>(Channel, Id))
      return EC;
    MM = llvm::make_unique<RCMemoryManager>(*this, Id);
    return std::error_code();
  }

  /// Create an RCIndirectStubsManager that will allocate stubs on the remote
  /// target.
  std::error_code
  createIndirectStubsManager(std::unique_ptr<RCIndirectStubsManager> &I) {
    assert(!I && "Indirect stubs manager should be null before creation.");
    auto Id = IndirectStubOwnerIds.getNext();
    if (auto EC = callST<CreateIndirectStubsOwner>(Channel, Id))
      return EC;
    I = llvm::make_unique<RCIndirectStubsManager>(*this, Id);
    return std::error_code();
  }

  /// Search for symbols in the remote process. Note: This should be used by
  /// symbol resolvers *after* they've searched the local symbol table in the
  /// JIT stack.
  ErrorOr<TargetAddress> getSymbolAddress(StringRef Name) {
    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return ExistingError;

    return callST<GetSymbolAddress>(Channel, Name);
  }

  /// Get the triple for the remote target.
  const std::string &getTargetTriple() const { return RemoteTargetTriple; }

  std::error_code terminateSession() {
    return callST<TerminateSession>(Channel);
  }

private:
  OrcRemoteTargetClient(ChannelT &Channel, std::error_code &EC)
      : Channel(Channel) {
    if (auto RIOrErr = callST<GetRemoteInfo>(Channel)) {
      std::tie(RemoteTargetTriple, RemotePointerSize, RemotePageSize,
	       RemoteTrampolineSize, RemoteIndirectStubSize) =
	*RIOrErr;
      EC = std::error_code();
    } else
      EC = RIOrErr.getError();
  }

  std::error_code deregisterEHFrames(TargetAddress Addr, uint32_t Size) {
    return callST<RegisterEHFrames>(Channel, Addr, Size);
  }

  void destroyRemoteAllocator(ResourceIdMgr::ResourceId Id) {
    if (auto EC = callST<DestroyRemoteAllocator>(Channel, Id)) {
      // FIXME: This will be triggered by a removeModuleSet call: Propagate
      //        error return up through that.
      llvm_unreachable("Failed to destroy remote allocator.");
      AllocatorIds.release(Id);
    }
  }

  std::error_code destroyIndirectStubsManager(ResourceIdMgr::ResourceId Id) {
    IndirectStubOwnerIds.release(Id);
    return callST<DestroyIndirectStubsOwner>(Channel, Id);
  }

  ErrorOr<std::tuple<TargetAddress, TargetAddress, uint32_t>>
  emitIndirectStubs(ResourceIdMgr::ResourceId Id,
		    uint32_t NumStubsRequired) {
    return callST<EmitIndirectStubs>(Channel, Id, NumStubsRequired);
  }

  std::error_code emitResolverBlock() {
    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return ExistingError;

    return callST<EmitResolverBlock>(Channel);
  }

  ErrorOr<std::tuple<TargetAddress, uint32_t>>
  emitTrampolineBlock() {
    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return ExistingError;

    return callST<EmitTrampolineBlock>(Channel);
  }

  uint32_t getIndirectStubSize() const { return RemoteIndirectStubSize; }
  uint32_t getPageSize() const { return RemotePageSize; }
  uint32_t getPointerSize() const { return RemotePointerSize; }

  uint32_t getTrampolineSize() const { return RemoteTrampolineSize; }

  std::error_code listenForCompileRequests(RPCChannel &C, uint32_t &Id) {
    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return ExistingError;

    if (Id == RequestCompileId) {
      if (auto EC = handle<RequestCompile>(C, CompileCallback))
        return EC;
      return std::error_code();
    }
    // else
    return orcError(OrcErrorCode::UnexpectedRPCCall);
  }

  ErrorOr<std::vector<char>> readMem(char *Dst, TargetAddress Src, uint64_t Size) {
    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return ExistingError;

    return callST<ReadMem>(Channel, Src, Size);
  }

  std::error_code registerEHFrames(TargetAddress &RAddr, uint32_t Size) {
    return callST<RegisterEHFrames>(Channel, RAddr, Size);
  }

  ErrorOr<TargetAddress> reserveMem(ResourceIdMgr::ResourceId Id, uint64_t Size,
				    uint32_t Align) {

    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return ExistingError;

    return callST<ReserveMem>(Channel, Id, Size, Align);
  }

  std::error_code setProtections(ResourceIdMgr::ResourceId Id,
                                 TargetAddress RemoteSegAddr,
                                 unsigned ProtFlags) {
    return callST<SetProtections>(Channel, Id, RemoteSegAddr, ProtFlags);
  }

  std::error_code writeMem(TargetAddress Addr, const char *Src, uint64_t Size) {
    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return ExistingError;

    return callST<WriteMem>(Channel, DirectBufferWriter(Src, Addr, Size));
  }

  std::error_code writePointer(TargetAddress Addr, TargetAddress PtrVal) {
    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return ExistingError;

    return callST<WritePtr>(Channel, Addr, PtrVal);
  }

  static std::error_code doNothing() { return std::error_code(); }

  ChannelT &Channel;
  std::error_code ExistingError;
  std::string RemoteTargetTriple;
  uint32_t RemotePointerSize = 0;
  uint32_t RemotePageSize = 0;
  uint32_t RemoteTrampolineSize = 0;
  uint32_t RemoteIndirectStubSize = 0;
  ResourceIdMgr AllocatorIds, IndirectStubOwnerIds;
  std::function<TargetAddress(TargetAddress)> CompileCallback;
};

} // end namespace remote
} // end namespace orc
} // end namespace llvm

#undef DEBUG_TYPE

#endif // LLVM_EXECUTIONENGINE_ORC_ORCREMOTETARGETCLIENT_H
