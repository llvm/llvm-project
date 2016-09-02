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
  // FIXME: Remove move/copy ops once MSVC supports synthesizing move ops.

  OrcRemoteTargetClient(const OrcRemoteTargetClient &) = delete;
  OrcRemoteTargetClient &operator=(const OrcRemoteTargetClient &) = delete;

  OrcRemoteTargetClient(OrcRemoteTargetClient &&Other)
      : Channel(Other.Channel), ExistingError(std::move(Other.ExistingError)),
        RemoteTargetTriple(std::move(Other.RemoteTargetTriple)),
        RemotePointerSize(std::move(Other.RemotePointerSize)),
        RemotePageSize(std::move(Other.RemotePageSize)),
        RemoteTrampolineSize(std::move(Other.RemoteTrampolineSize)),
        RemoteIndirectStubSize(std::move(Other.RemoteIndirectStubSize)),
        AllocatorIds(std::move(Other.AllocatorIds)),
        IndirectStubOwnerIds(std::move(Other.IndirectStubOwnerIds)),
        CallbackManager(std::move(Other.CallbackManager)) {}

  OrcRemoteTargetClient &operator=(OrcRemoteTargetClient &&) = delete;

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
          assert(!AddrOrErr.takeError() && "Failed reserving remote memory.");
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
          assert(!AddrOrErr.takeError() && "Failed reserving remote memory.");
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
          assert(!AddrOrErr.takeError() && "Failed reserving remote memory.");
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
      auto Err = Client.deregisterEHFrames(LoadAddr, Size);
      // FIXME: Add error poll.
      assert(!Err && "Failed to register remote EH frames.");
      (void)Err;
    }

    void notifyObjectLoaded(RuntimeDyld &Dyld,
                            const object::ObjectFile &Obj) override {
      DEBUG(dbgs() << "Allocator " << Id << " applied mappings:\n");
      for (auto &ObjAllocs : Unmapped) {
        {
          JITTargetAddress NextCodeAddr = ObjAllocs.RemoteCodeAddr;
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
          JITTargetAddress NextRODataAddr = ObjAllocs.RemoteRODataAddr;
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
          JITTargetAddress NextRWDataAddr = ObjAllocs.RemoteRWDataAddr;
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
          if (auto Err =
                  Client.writeMem(Alloc.getRemoteAddress(),
                                  Alloc.getLocalAddress(), Alloc.getSize())) {
            // FIXME: Replace this once finalizeMemory can return an Error.
            handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
              if (ErrMsg) {
                raw_string_ostream ErrOut(*ErrMsg);
                EIB.log(ErrOut);
              }
            });
            return true;
          }
        }

        if (ObjAllocs.RemoteCodeAddr) {
          DEBUG(dbgs() << "  setting R-X permissions on code block: "
                       << format("0x%016x", ObjAllocs.RemoteCodeAddr) << "\n");
          if (auto Err = Client.setProtections(Id, ObjAllocs.RemoteCodeAddr,
                                               sys::Memory::MF_READ |
                                                   sys::Memory::MF_EXEC)) {
            // FIXME: Replace this once finalizeMemory can return an Error.
            handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
              if (ErrMsg) {
                raw_string_ostream ErrOut(*ErrMsg);
                EIB.log(ErrOut);
              }
            });
            return true;
          }
        }

        for (auto &Alloc : ObjAllocs.RODataAllocs) {
          DEBUG(dbgs() << "  copying ro-data: "
                       << static_cast<void *>(Alloc.getLocalAddress()) << " -> "
                       << format("0x%016x", Alloc.getRemoteAddress()) << " ("
                       << Alloc.getSize() << " bytes)\n");
          if (auto Err =
                  Client.writeMem(Alloc.getRemoteAddress(),
                                  Alloc.getLocalAddress(), Alloc.getSize())) {
            // FIXME: Replace this once finalizeMemory can return an Error.
            handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
              if (ErrMsg) {
                raw_string_ostream ErrOut(*ErrMsg);
                EIB.log(ErrOut);
              }
            });
            return true;
          }
        }

        if (ObjAllocs.RemoteRODataAddr) {
          DEBUG(dbgs() << "  setting R-- permissions on ro-data block: "
                       << format("0x%016x", ObjAllocs.RemoteRODataAddr)
                       << "\n");
          if (auto Err = Client.setProtections(Id, ObjAllocs.RemoteRODataAddr,
                                               sys::Memory::MF_READ)) {
            // FIXME: Replace this once finalizeMemory can return an Error.
            handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
              if (ErrMsg) {
                raw_string_ostream ErrOut(*ErrMsg);
                EIB.log(ErrOut);
              }
            });
            return false;
          }
        }

        for (auto &Alloc : ObjAllocs.RWDataAllocs) {
          DEBUG(dbgs() << "  copying rw-data: "
                       << static_cast<void *>(Alloc.getLocalAddress()) << " -> "
                       << format("0x%016x", Alloc.getRemoteAddress()) << " ("
                       << Alloc.getSize() << " bytes)\n");
          if (auto Err =
                  Client.writeMem(Alloc.getRemoteAddress(),
                                  Alloc.getLocalAddress(), Alloc.getSize())) {
            // FIXME: Replace this once finalizeMemory can return an Error.
            handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
              if (ErrMsg) {
                raw_string_ostream ErrOut(*ErrMsg);
                EIB.log(ErrOut);
              }
            });
            return false;
          }
        }

        if (ObjAllocs.RemoteRWDataAddr) {
          DEBUG(dbgs() << "  setting RW- permissions on rw-data block: "
                       << format("0x%016x", ObjAllocs.RemoteRWDataAddr)
                       << "\n");
          if (auto Err = Client.setProtections(Id, ObjAllocs.RemoteRWDataAddr,
                                               sys::Memory::MF_READ |
                                                   sys::Memory::MF_WRITE)) {
            // FIXME: Replace this once finalizeMemory can return an Error.
            handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
              if (ErrMsg) {
                raw_string_ostream ErrOut(*ErrMsg);
                EIB.log(ErrOut);
              }
            });
            return false;
          }
        }
      }
      Unfinalized.clear();

      for (auto &EHFrame : UnfinalizedEHFrames) {
        if (auto Err = Client.registerEHFrames(EHFrame.first, EHFrame.second)) {
          // FIXME: Replace this once finalizeMemory can return an Error.
          handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
            if (ErrMsg) {
              raw_string_ostream ErrOut(*ErrMsg);
              EIB.log(ErrOut);
            }
          });
          return false;
        }
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

      void setRemoteAddress(JITTargetAddress RemoteAddr) {
        this->RemoteAddr = RemoteAddr;
      }

      JITTargetAddress getRemoteAddress() const { return RemoteAddr; }

    private:
      uint64_t Size;
      unsigned Align;
      std::unique_ptr<char[]> Contents;
      JITTargetAddress RemoteAddr = 0;
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

      JITTargetAddress RemoteCodeAddr = 0;
      JITTargetAddress RemoteRODataAddr = 0;
      JITTargetAddress RemoteRWDataAddr = 0;
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
      if (auto Err = Remote.destroyIndirectStubsManager(Id)) {
        // FIXME: Thread this error back to clients.
        consumeError(std::move(Err));
      }
    }

    Error createStub(StringRef StubName, JITTargetAddress StubAddr,
                     JITSymbolFlags StubFlags) override {
      if (auto Err = reserveStubs(1))
        return Err;

      return createStubInternal(StubName, StubAddr, StubFlags);
    }

    Error createStubs(const StubInitsMap &StubInits) override {
      if (auto Err = reserveStubs(StubInits.size()))
        return Err;

      for (auto &Entry : StubInits)
        if (auto Err = createStubInternal(Entry.first(), Entry.second.first,
                                          Entry.second.second))
          return Err;

      return Error::success();
    }

    JITSymbol findStub(StringRef Name, bool ExportedStubsOnly) override {
      auto I = StubIndexes.find(Name);
      if (I == StubIndexes.end())
        return nullptr;
      auto Key = I->second.first;
      auto Flags = I->second.second;
      auto StubSymbol = JITSymbol(getStubAddr(Key), Flags);
      if (ExportedStubsOnly && !StubSymbol.getFlags().isExported())
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

    Error updatePointer(StringRef Name, JITTargetAddress NewAddr) override {
      auto I = StubIndexes.find(Name);
      assert(I != StubIndexes.end() && "No stub pointer for symbol");
      auto Key = I->second.first;
      return Remote.writePointer(getPtrAddr(Key), NewAddr);
    }

  private:
    struct RemoteIndirectStubsInfo {
      JITTargetAddress StubBase;
      JITTargetAddress PtrBase;
      unsigned NumStubs;
    };

    OrcRemoteTargetClient &Remote;
    ResourceIdMgr::ResourceId Id;
    std::vector<RemoteIndirectStubsInfo> RemoteIndirectStubsInfos;
    typedef std::pair<uint16_t, uint16_t> StubKey;
    std::vector<StubKey> FreeStubs;
    StringMap<std::pair<StubKey, JITSymbolFlags>> StubIndexes;

    Error reserveStubs(unsigned NumStubs) {
      if (NumStubs <= FreeStubs.size())
        return Error::success();

      unsigned NewStubsRequired = NumStubs - FreeStubs.size();
      JITTargetAddress StubBase;
      JITTargetAddress PtrBase;
      unsigned NumStubsEmitted;

      if (auto StubInfoOrErr = Remote.emitIndirectStubs(Id, NewStubsRequired))
        std::tie(StubBase, PtrBase, NumStubsEmitted) = *StubInfoOrErr;
      else
        return StubInfoOrErr.takeError();

      unsigned NewBlockId = RemoteIndirectStubsInfos.size();
      RemoteIndirectStubsInfos.push_back({StubBase, PtrBase, NumStubsEmitted});

      for (unsigned I = 0; I < NumStubsEmitted; ++I)
        FreeStubs.push_back(std::make_pair(NewBlockId, I));

      return Error::success();
    }

    Error createStubInternal(StringRef StubName, JITTargetAddress InitAddr,
                             JITSymbolFlags StubFlags) {
      auto Key = FreeStubs.back();
      FreeStubs.pop_back();
      StubIndexes[StubName] = std::make_pair(Key, StubFlags);
      return Remote.writePointer(getPtrAddr(Key), InitAddr);
    }

    JITTargetAddress getStubAddr(StubKey K) {
      assert(RemoteIndirectStubsInfos[K.first].StubBase != 0 &&
             "Missing stub address");
      return RemoteIndirectStubsInfos[K.first].StubBase +
             K.second * Remote.getIndirectStubSize();
    }

    JITTargetAddress getPtrAddr(StubKey K) {
      assert(RemoteIndirectStubsInfos[K.first].PtrBase != 0 &&
             "Missing pointer address");
      return RemoteIndirectStubsInfos[K.first].PtrBase +
             K.second * Remote.getPointerSize();
    }
  };

  /// Remote compile callback manager.
  class RCCompileCallbackManager : public JITCompileCallbackManager {
  public:
    RCCompileCallbackManager(JITTargetAddress ErrorHandlerAddress,
                             OrcRemoteTargetClient &Remote)
        : JITCompileCallbackManager(ErrorHandlerAddress), Remote(Remote) {}

  private:
    void grow() override {
      JITTargetAddress BlockAddr = 0;
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
  static Expected<OrcRemoteTargetClient> Create(ChannelT &Channel) {
    Error Err;
    OrcRemoteTargetClient H(Channel, Err);
    if (Err)
      return std::move(Err);
    return Expected<OrcRemoteTargetClient>(std::move(H));
  }

  /// Call the int(void) function at the given address in the target and return
  /// its result.
  Expected<int> callIntVoid(JITTargetAddress Addr) {
    DEBUG(dbgs() << "Calling int(*)(void) " << format("0x%016x", Addr) << "\n");

    auto Listen = [&](RPCChannel &C, uint32_t Id) {
      return listenForCompileRequests(C, Id);
    };
    return callSTHandling<CallIntVoid>(Channel, Listen, Addr);
  }

  /// Call the int(int, char*[]) function at the given address in the target and
  /// return its result.
  Expected<int> callMain(JITTargetAddress Addr,
                         const std::vector<std::string> &Args) {
    DEBUG(dbgs() << "Calling int(*)(int, char*[]) " << format("0x%016x", Addr)
                 << "\n");

    auto Listen = [&](RPCChannel &C, uint32_t Id) {
      return listenForCompileRequests(C, Id);
    };
    return callSTHandling<CallMain>(Channel, Listen, Addr, Args);
  }

  /// Call the void() function at the given address in the target and wait for
  /// it to finish.
  Error callVoidVoid(JITTargetAddress Addr) {
    DEBUG(dbgs() << "Calling void(*)(void) " << format("0x%016x", Addr)
                 << "\n");

    auto Listen = [&](RPCChannel &C, uint32_t Id) {
      return listenForCompileRequests(C, Id);
    };
    return callSTHandling<CallVoidVoid>(Channel, Listen, Addr);
  }

  /// Create an RCMemoryManager which will allocate its memory on the remote
  /// target.
  Error createRemoteMemoryManager(std::unique_ptr<RCMemoryManager> &MM) {
    assert(!MM && "MemoryManager should be null before creation.");

    auto Id = AllocatorIds.getNext();
    if (auto Err = callST<CreateRemoteAllocator>(Channel, Id))
      return Err;
    MM = llvm::make_unique<RCMemoryManager>(*this, Id);
    return Error::success();
  }

  /// Create an RCIndirectStubsManager that will allocate stubs on the remote
  /// target.
  Error createIndirectStubsManager(std::unique_ptr<RCIndirectStubsManager> &I) {
    assert(!I && "Indirect stubs manager should be null before creation.");
    auto Id = IndirectStubOwnerIds.getNext();
    if (auto Err = callST<CreateIndirectStubsOwner>(Channel, Id))
      return Err;
    I = llvm::make_unique<RCIndirectStubsManager>(*this, Id);
    return Error::success();
  }

  Expected<RCCompileCallbackManager &>
  enableCompileCallbacks(JITTargetAddress ErrorHandlerAddress) {
    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return std::move(ExistingError);

    // Emit the resolver block on the JIT server.
    if (auto Err = callST<EmitResolverBlock>(Channel))
      return std::move(Err);

    // Create the callback manager.
    CallbackManager.emplace(ErrorHandlerAddress, *this);
    RCCompileCallbackManager &Mgr = *CallbackManager;
    return Mgr;
  }

  /// Search for symbols in the remote process. Note: This should be used by
  /// symbol resolvers *after* they've searched the local symbol table in the
  /// JIT stack.
  Expected<JITTargetAddress> getSymbolAddress(StringRef Name) {
    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return std::move(ExistingError);

    return callST<GetSymbolAddress>(Channel, Name);
  }

  /// Get the triple for the remote target.
  const std::string &getTargetTriple() const { return RemoteTargetTriple; }

  Error terminateSession() { return callST<TerminateSession>(Channel); }

private:
  OrcRemoteTargetClient(ChannelT &Channel, Error &Err) : Channel(Channel) {
    ErrorAsOutParameter EAO(&Err);
    if (auto RIOrErr = callST<GetRemoteInfo>(Channel)) {
      std::tie(RemoteTargetTriple, RemotePointerSize, RemotePageSize,
               RemoteTrampolineSize, RemoteIndirectStubSize) = *RIOrErr;
      Err = Error::success();
    } else {
      Err = joinErrors(RIOrErr.takeError(), std::move(ExistingError));
    }
  }

  Error deregisterEHFrames(JITTargetAddress Addr, uint32_t Size) {
    return callST<RegisterEHFrames>(Channel, Addr, Size);
  }

  void destroyRemoteAllocator(ResourceIdMgr::ResourceId Id) {
    if (auto Err = callST<DestroyRemoteAllocator>(Channel, Id)) {
      // FIXME: This will be triggered by a removeModuleSet call: Propagate
      //        error return up through that.
      llvm_unreachable("Failed to destroy remote allocator.");
      AllocatorIds.release(Id);
    }
  }

  Error destroyIndirectStubsManager(ResourceIdMgr::ResourceId Id) {
    IndirectStubOwnerIds.release(Id);
    return callST<DestroyIndirectStubsOwner>(Channel, Id);
  }

  Expected<std::tuple<JITTargetAddress, JITTargetAddress, uint32_t>>
  emitIndirectStubs(ResourceIdMgr::ResourceId Id, uint32_t NumStubsRequired) {
    return callST<EmitIndirectStubs>(Channel, Id, NumStubsRequired);
  }

  Expected<std::tuple<JITTargetAddress, uint32_t>> emitTrampolineBlock() {
    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return std::move(ExistingError);

    return callST<EmitTrampolineBlock>(Channel);
  }

  uint32_t getIndirectStubSize() const { return RemoteIndirectStubSize; }
  uint32_t getPageSize() const { return RemotePageSize; }
  uint32_t getPointerSize() const { return RemotePointerSize; }

  uint32_t getTrampolineSize() const { return RemoteTrampolineSize; }

  Error listenForCompileRequests(RPCChannel &C, uint32_t &Id) {
    assert(CallbackManager &&
           "No calback manager. enableCompileCallbacks must be called first");

    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return std::move(ExistingError);

    // FIXME: CompileCallback could be an anonymous lambda defined at the use
    //        site below, but that triggers a GCC 4.7 ICE. When we move off
    //        GCC 4.7, tidy this up.
    auto CompileCallback =
      [this](JITTargetAddress Addr) -> Expected<JITTargetAddress> {
        return this->CallbackManager->executeCompileCallback(Addr);
      };

    if (Id == RequestCompileId) {
      if (auto Err = handle<RequestCompile>(C, CompileCallback))
        return Err;
      return Error::success();
    }
    // else
    return orcError(OrcErrorCode::UnexpectedRPCCall);
  }

  Expected<std::vector<char>> readMem(char *Dst, JITTargetAddress Src,
                                      uint64_t Size) {
    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return std::move(ExistingError);

    return callST<ReadMem>(Channel, Src, Size);
  }

  Error registerEHFrames(JITTargetAddress &RAddr, uint32_t Size) {
    return callST<RegisterEHFrames>(Channel, RAddr, Size);
  }

  Expected<JITTargetAddress> reserveMem(ResourceIdMgr::ResourceId Id,
                                        uint64_t Size, uint32_t Align) {

    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return std::move(ExistingError);

    return callST<ReserveMem>(Channel, Id, Size, Align);
  }

  Error setProtections(ResourceIdMgr::ResourceId Id,
                       JITTargetAddress RemoteSegAddr, unsigned ProtFlags) {
    return callST<SetProtections>(Channel, Id, RemoteSegAddr, ProtFlags);
  }

  Error writeMem(JITTargetAddress Addr, const char *Src, uint64_t Size) {
    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return std::move(ExistingError);

    return callST<WriteMem>(Channel, DirectBufferWriter(Src, Addr, Size));
  }

  Error writePointer(JITTargetAddress Addr, JITTargetAddress PtrVal) {
    // Check for an 'out-of-band' error, e.g. from an MM destructor.
    if (ExistingError)
      return std::move(ExistingError);

    return callST<WritePtr>(Channel, Addr, PtrVal);
  }

  static Error doNothing() { return Error::success(); }

  ChannelT &Channel;
  Error ExistingError;
  std::string RemoteTargetTriple;
  uint32_t RemotePointerSize = 0;
  uint32_t RemotePageSize = 0;
  uint32_t RemoteTrampolineSize = 0;
  uint32_t RemoteIndirectStubSize = 0;
  ResourceIdMgr AllocatorIds, IndirectStubOwnerIds;
  Optional<RCCompileCallbackManager> CallbackManager;
};

} // end namespace remote
} // end namespace orc
} // end namespace llvm

#undef DEBUG_TYPE

#endif // LLVM_EXECUTIONENGINE_ORC_ORCREMOTETARGETCLIENT_H
