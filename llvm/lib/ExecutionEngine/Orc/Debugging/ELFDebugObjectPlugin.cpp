//===--------- ELFDebugObjectPlugin.cpp - JITLink debug objects -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// FIXME: Update Plugin to poke the debug object into a new JITLink section,
//        rather than creating a new allocation.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Debugging/ELFDebugObjectPlugin.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkDylib.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/MemoryFlags.h"
#include "llvm/ExecutionEngine/Orc/Shared/OrcRTBridge.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <set>

#define DEBUG_TYPE "orc"

using namespace llvm::jitlink;
using namespace llvm::object;

namespace llvm {
namespace orc {

// Helper class to emit and fixup an individual debug object
class DebugObject {
public:
  using FinalizedAlloc = JITLinkMemoryManager::FinalizedAlloc;

  DebugObject(StringRef Name, SimpleSegmentAlloc Alloc, JITLinkContext &Ctx,
              ExecutionSession &ES)
      : Name(Name), WorkingMem(std::move(Alloc)),
        MemMgr(Ctx.getMemoryManager()), ES(ES) {}

  ~DebugObject() {
    // Alloc was finialized
    if (Alloc) {
      std::vector<FinalizedAlloc> Allocs;
      Allocs.push_back(std::move(Alloc));
      if (Error Err = MemMgr.deallocate(std::move(Allocs)))
        ES.reportError(std::move(Err));
      return;
    }
    // Error before step 3: WorkingMem was not collected
    if (!FinalizeFuture.valid()) {
      WorkingMem.abandon(
          [ES = &this->ES](Error Err) { ES->reportError(std::move(Err)); });
      return;
    }
    // Error before step 4: Finalization error was not reported
    Expected<ExecutorAddrRange> TargetMem = FinalizeFuture.get();
    if (!TargetMem)
      ES.reportError(TargetMem.takeError());
  }

  MutableArrayRef<char> getBuffer() {
    auto SegInfo = WorkingMem.getSegInfo(MemProt::Read);
    return SegInfo.WorkingMem;
  }

  SimpleSegmentAlloc collectTargetAlloc() {
    FinalizeFuture = FinalizePromise.get_future();
    return std::move(WorkingMem);
  }

  void trackFinalizedAlloc(FinalizedAlloc FA) { Alloc = std::move(FA); }

  Expected<ExecutorAddrRange> awaitTargetMem() { return FinalizeFuture.get(); }

  void reportTargetMem(ExecutorAddrRange TargetMem) {
    FinalizePromise.set_value(TargetMem);
  }

  void failMaterialization(Error Err) {
    FinalizePromise.set_value(std::move(Err));
  }

  using GetLoadAddressFn = llvm::unique_function<ExecutorAddr(StringRef)>;
  Error visitSections(GetLoadAddressFn Callback);

  template <typename ELFT>
  Error visitSectionLoadAddresses(GetLoadAddressFn Callback);

private:
  std::string Name;
  SimpleSegmentAlloc WorkingMem;
  JITLinkMemoryManager &MemMgr;
  ExecutionSession &ES;

  std::promise<MSVCPExpected<ExecutorAddrRange>> FinalizePromise;
  std::future<MSVCPExpected<ExecutorAddrRange>> FinalizeFuture;

  FinalizedAlloc Alloc;
};

template <typename ELFT>
Error DebugObject::visitSectionLoadAddresses(GetLoadAddressFn Callback) {
  using SectionHeader = typename ELFT::Shdr;

  MutableArrayRef<char> Buffer = getBuffer();
  StringRef BufferRef(Buffer.data(), Buffer.size());
  Expected<ELFFile<ELFT>> ObjRef = ELFFile<ELFT>::create(BufferRef);
  if (!ObjRef)
    return ObjRef.takeError();

  Expected<ArrayRef<SectionHeader>> Sections = ObjRef->sections();
  if (!Sections)
    return Sections.takeError();

  for (const SectionHeader &Header : *Sections) {
    Expected<StringRef> Name = ObjRef->getSectionName(Header);
    if (!Name)
      return Name.takeError();
    if (Name->empty())
      continue;
    ExecutorAddr LoadAddress = Callback(*Name);
    if (LoadAddress)
      const_cast<SectionHeader &>(Header).sh_addr =
          static_cast<typename ELFT::uint>(LoadAddress.getValue());
  }

  LLVM_DEBUG({
    dbgs() << "Section load-addresses in debug object for \"" << Name
           << "\":\n";
    for (const SectionHeader &Header : *Sections) {
      StringRef Name = cantFail(ObjRef->getSectionName(Header));
      if (uint64_t Addr = Header.sh_addr) {
        dbgs() << formatv("  {0:x16} {1}\n", Addr, Name);
      } else {
        dbgs() << formatv("                     {0}\n", Name);
      }
    }
  });

  return Error::success();
}

Error DebugObject::visitSections(GetLoadAddressFn Callback) {
  unsigned char Class, Endian;
  MutableArrayRef<char> Buf = getBuffer();
  std::tie(Class, Endian) = getElfArchType(StringRef(Buf.data(), Buf.size()));

  switch (Class) {
  case ELF::ELFCLASS32:
    if (Endian == ELF::ELFDATA2LSB)
      return visitSectionLoadAddresses<ELF32LE>(std::move(Callback));
    if (Endian == ELF::ELFDATA2MSB)
      return visitSectionLoadAddresses<ELF32BE>(std::move(Callback));
    break;

  case ELF::ELFCLASS64:
    if (Endian == ELF::ELFDATA2LSB)
      return visitSectionLoadAddresses<ELF64LE>(std::move(Callback));
    if (Endian == ELF::ELFDATA2MSB)
      return visitSectionLoadAddresses<ELF64BE>(std::move(Callback));
    break;

  default:
    break;
  }
  llvm_unreachable("Checked class and endian in notifyMaterializing()");
}

ELFDebugObjectPlugin::ELFDebugObjectPlugin(ExecutionSession &ES,
                                           bool RequireDebugSections,
                                           bool AutoRegisterCode, Error &Err)
    : ES(ES), RequireDebugSections(RequireDebugSections),
      AutoRegisterCode(AutoRegisterCode) {
  // Pass bootstrap symbol for registration function to enable debugging
  ErrorAsOutParameter _(&Err);
  Err = ES.getExecutorProcessControl().getBootstrapSymbols(
      {{RegistrationAction, rt::RegisterJITLoaderGDBAllocActionName}});
}

ELFDebugObjectPlugin::~ELFDebugObjectPlugin() = default;

static const std::set<StringRef> DwarfSectionNames = {
#define HANDLE_DWARF_SECTION(ENUM_NAME, ELF_NAME, CMDLINE_NAME, OPTION)        \
  ELF_NAME,
#include "llvm/BinaryFormat/Dwarf.def"
#undef HANDLE_DWARF_SECTION
};

static bool isDwarfSection(StringRef SectionName) {
  return DwarfSectionNames.count(SectionName) == 1;
}

void ELFDebugObjectPlugin::notifyMaterializing(
    MaterializationResponsibility &MR, LinkGraph &G, JITLinkContext &Ctx,
    MemoryBufferRef InputObj) {
  if (InputObj.getBufferSize() == 0)
    return;
  if (G.getTargetTriple().getObjectFormat() != Triple::ELF)
    return;

  unsigned char Class, Endian;
  std::tie(Class, Endian) = getElfArchType(InputObj.getBuffer());
  if (Class != ELF::ELFCLASS64 && Class != ELF::ELFCLASS32)
    return ES.reportError(
        createStringError(object_error::invalid_file_type,
                          "Skipping debug object registration: Invalid arch "
                          "0x%02x in ELF LinkGraph %s",
                          Class, G.getName().c_str()));
  if (Endian != ELF::ELFDATA2LSB && Endian != ELF::ELFDATA2MSB)
    return ES.reportError(
        createStringError(object_error::invalid_file_type,
                          "Skipping debug object registration: Invalid endian "
                          "0x%02x in ELF LinkGraph %s",
                          Endian, G.getName().c_str()));

  // Step 1: We copy the raw input object into the working memory of a
  // single-segment read-only allocation
  size_t Size = InputObj.getBufferSize();
  auto Alignment = sys::Process::getPageSizeEstimate();
  SimpleSegmentAlloc::Segment Segment{Size, Align(Alignment)};

  auto Alloc = SimpleSegmentAlloc::Create(
      Ctx.getMemoryManager(), ES.getSymbolStringPool(), ES.getTargetTriple(),
      Ctx.getJITLinkDylib(), {{MemProt::Read, Segment}});
  if (!Alloc) {
    ES.reportError(Alloc.takeError());
    return;
  }

  std::lock_guard<std::mutex> Lock(PendingObjsLock);
  assert(PendingObjs.count(&MR) == 0 && "One debug object per materialization");
  PendingObjs[&MR] = std::make_unique<DebugObject>(
      InputObj.getBufferIdentifier(), std::move(*Alloc), Ctx, ES);

  MutableArrayRef<char> Buffer = PendingObjs[&MR]->getBuffer();
  memcpy(Buffer.data(), InputObj.getBufferStart(), Size);
}

DebugObject *
ELFDebugObjectPlugin::getPendingDebugObj(MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(PendingObjsLock);
  auto It = PendingObjs.find(&MR);
  return It == PendingObjs.end() ? nullptr : It->second.get();
}

void ELFDebugObjectPlugin::modifyPassConfig(MaterializationResponsibility &MR,
                                            LinkGraph &G,
                                            PassConfiguration &PassConfig) {
  if (!getPendingDebugObj(MR))
    return;

  PassConfig.PostAllocationPasses.push_back([this, &MR](LinkGraph &G) -> Error {
    size_t SectionsPatched = 0;
    bool HasDebugSections = false;
    DebugObject *DebugObj = getPendingDebugObj(MR);
    assert(DebugObj && "Don't inject passes if we have no debug object");

    // Step 2: Once the target memory layout is ready, we write the
    // addresses of the LinkGraph sections into the load-address fields of the
    // section headers in our debug object allocation
    Error Err = DebugObj->visitSections(
        [&G, &SectionsPatched, &HasDebugSections](StringRef Name) {
          Section *S = G.findSectionByName(Name);
          if (!S) {
            // The section may have been merged into a different one during
            // linking, ignore it.
            return ExecutorAddr();
          }

          SectionsPatched += 1;
          if (isDwarfSection(Name))
            HasDebugSections = true;
          return SectionRange(*S).getStart();
        });

    if (Err)
      return Err;
    if (!SectionsPatched) {
      LLVM_DEBUG(dbgs() << "Skipping debug registration for LinkGraph '"
                        << G.getName() << "': no debug info\n");
      return Error::success();
    }

    if (RequireDebugSections && !HasDebugSections) {
      LLVM_DEBUG(dbgs() << "Skipping debug registration for LinkGraph '"
                        << G.getName() << "': no debug info\n");
      return Error::success();
    }

    // Step 3: We start copying the debug object into target memory
    SimpleSegmentAlloc Alloc = DebugObj->collectTargetAlloc();

    // FIXME: FA->getAddress() below is supposed to be the address of the memory
    // range on the target, but InProcessMemoryManager returns the address of a
    // FinalizedAllocInfo helper instead
    auto ROSeg = Alloc.getSegInfo(MemProt::Read);
    ExecutorAddrRange R(ROSeg.Addr, ROSeg.WorkingMem.size());
    Alloc.finalize([this, R, &MR](Expected<DebugObject::FinalizedAlloc> FA) {
      // Bail out if materialization failed in the meantime
      std::lock_guard<std::mutex> Lock(PendingObjsLock);
      auto It = PendingObjs.find(&MR);
      if (It == PendingObjs.end()) {
        if (!FA)
          ES.reportError(FA.takeError());
        return;
      }

      DebugObject *DebugObj = It->second.get();
      if (!FA)
        DebugObj->failMaterialization(FA.takeError());

      // Keep allocation alive until the corresponding code is removed
      DebugObj->trackFinalizedAlloc(std::move(*FA));

      // Unblock post-fixup pass
      DebugObj->reportTargetMem(R);
    });

    return Error::success();
  });

  PassConfig.PostFixupPasses.push_back([this, &MR](LinkGraph &G) -> Error {
    // Step 4: We wait for the debug object copy to finish, so we can
    // register the memory range with the GDB JIT Interface in an allocation
    // action of the LinkGraph's own allocation
    DebugObject *DebugObj = getPendingDebugObj(MR);
    Expected<ExecutorAddrRange> R = DebugObj->awaitTargetMem();
    if (!R)
      return R.takeError();

    // Step 5: We have to keep the allocation alive until the corresponding
    // code is removed
    Error Err = MR.withResourceKeyDo([&](ResourceKey K) {
      std::lock_guard<std::mutex> LockPending(PendingObjsLock);
      std::lock_guard<std::mutex> LockRegistered(RegisteredObjsLock);
      auto It = PendingObjs.find(&MR);
      RegisteredObjs[K].push_back(std::move(It->second));
      PendingObjs.erase(It);
    });

    if (Err)
      return Err;

    if (R->empty())
      return Error::success();

    using namespace shared;
    G.allocActions().push_back(
        {cantFail(WrapperFunctionCall::Create<
                  SPSArgList<SPSExecutorAddrRange, bool>>(
             RegistrationAction, *R, AutoRegisterCode)),
         {/* no deregistration */}});
    return Error::success();
  });
}

Error ELFDebugObjectPlugin::notifyFailed(MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(PendingObjsLock);
  PendingObjs.erase(&MR);
  return Error::success();
}

void ELFDebugObjectPlugin::notifyTransferringResources(JITDylib &JD,
                                                       ResourceKey DstKey,
                                                       ResourceKey SrcKey) {
  // Debug objects are stored by ResourceKey only after registration.
  // Thus, pending objects don't need to be updated here.
  std::lock_guard<std::mutex> Lock(RegisteredObjsLock);
  auto SrcIt = RegisteredObjs.find(SrcKey);
  if (SrcIt != RegisteredObjs.end()) {
    // Resources from distinct MaterializationResponsibilitys can get merged
    // after emission, so we can have multiple debug objects per resource key.
    for (std::unique_ptr<DebugObject> &DebugObj : SrcIt->second)
      RegisteredObjs[DstKey].push_back(std::move(DebugObj));
    RegisteredObjs.erase(SrcIt);
  }
}

Error ELFDebugObjectPlugin::notifyRemovingResources(JITDylib &JD,
                                                    ResourceKey Key) {
  // Removing the resource for a pending object fails materialization, so they
  // get cleaned up in the notifyFailed() handler.
  std::lock_guard<std::mutex> Lock(RegisteredObjsLock);
  RegisteredObjs.erase(Key);

  // TODO: Implement unregister notifications.
  return Error::success();
}

} // namespace orc
} // namespace llvm
