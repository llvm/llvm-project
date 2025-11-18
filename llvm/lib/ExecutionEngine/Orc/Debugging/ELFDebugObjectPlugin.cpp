//===------- ELFDebugObjectPlugin.cpp - JITLink debug objects ---------===//
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
        MemMgr(Ctx.getMemoryManager()), ES(ES) {
    FinalizeFuture = FinalizePromise.get_future();
  }

  StringRef getName() const { return Name; }

  StringRef getBuffer() {
    MutableArrayRef<char> Buffer = getMutBuffer();
    return StringRef(Buffer.data(), Buffer.size());
  }

  MutableArrayRef<char> getMutBuffer() {
    auto SegInfo = WorkingMem.getSegInfo(MemProt::Read);
    return SegInfo.WorkingMem;
  }

  SimpleSegmentAlloc &getTargetAlloc() { return WorkingMem; }

  Expected<ExecutorAddrRange> awaitTargetMem() { return FinalizeFuture.get(); }

  void reportTargetMem(ExecutorAddrRange TargetMem) {
    FinalizePromise.set_value(TargetMem);
  }

  void failMaterialization(Error Err) {
    FinalizePromise.set_value(std::move(Err));
  }

  void reportError(Error Err) { ES.reportError(std::move(Err)); }

  using GetLoadAddressFn = llvm::unique_function<ExecutorAddr(StringRef)>;
  void visitSections(GetLoadAddressFn Callback);

  template <typename ELFT>
  void visitSectionLoadAddresses(GetLoadAddressFn Callback);

private:
  std::string Name;
  SimpleSegmentAlloc WorkingMem;
  JITLinkMemoryManager &MemMgr;
  ExecutionSession &ES;

  std::promise<MSVCPExpected<ExecutorAddrRange>> FinalizePromise;
  std::future<MSVCPExpected<ExecutorAddrRange>> FinalizeFuture;
};

template <typename ELFT>
void DebugObject::visitSectionLoadAddresses(GetLoadAddressFn Callback) {
  using SectionHeader = typename ELFT::Shdr;

  Expected<ELFFile<ELFT>> ObjRef = ELFFile<ELFT>::create(getBuffer());
  if (!ObjRef) {
    reportError(ObjRef.takeError());
    return;
  }

  Expected<ArrayRef<SectionHeader>> Sections = ObjRef->sections();
  if (!Sections) {
    reportError(Sections.takeError());
    return;
  }

  for (const SectionHeader &Header : *Sections) {
    Expected<StringRef> Name = ObjRef->getSectionName(Header);
    if (!Name) {
      reportError(Name.takeError());
      return;
    }
    if (Name->empty())
      continue;
    ExecutorAddr LoadAddress = Callback(*Name);
    const_cast<SectionHeader &>(Header).sh_addr =
        static_cast<typename ELFT::uint>(LoadAddress.getValue());
  }

  LLVM_DEBUG({
    dbgs() << "Section load-addresses in debug object for \"" << getName()
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
}

void DebugObject::visitSections(GetLoadAddressFn Callback) {
  unsigned char Class, Endian;
  std::tie(Class, Endian) = getElfArchType(getBuffer());

  switch (Class) {
  case ELF::ELFCLASS32:
    if (Endian == ELF::ELFDATA2LSB)
      return visitSectionLoadAddresses<ELF32LE>(std::move(Callback));
    if (Endian == ELF::ELFDATA2MSB)
      return visitSectionLoadAddresses<ELF32BE>(std::move(Callback));
    return reportError(createStringError(
        object_error::invalid_file_type,
        "Invalid endian in 32-bit ELF object file: %x", Endian));

  case ELF::ELFCLASS64:
    if (Endian == ELF::ELFDATA2LSB)
      return visitSectionLoadAddresses<ELF64LE>(std::move(Callback));
    if (Endian == ELF::ELFDATA2MSB)
      return visitSectionLoadAddresses<ELF64BE>(std::move(Callback));
    return reportError(createStringError(
        object_error::invalid_file_type,
        "Invalid endian in 64-bit ELF object file: %x", Endian));

  default:
    return reportError(createStringError(object_error::invalid_file_type,
                                         "Invalid arch in ELF object file: %x",
                                         Class));
  }
}

ELFDebugObjectPlugin::ELFDebugObjectPlugin(ExecutionSession &ES,
                                           bool RequireDebugSections,
                                           bool AutoRegisterCode, Error &Err)
    : ES(ES), RequireDebugSections(RequireDebugSections),
      AutoRegisterCode(AutoRegisterCode) {
  // Pass bootstrap symbol for registration function to enable debugging
  ErrorAsOutParameter _(&Err);
  Err = ES.getExecutorProcessControl().getBootstrapSymbols({
      {RegistrationAction, rt::RegisterJITLoaderGDBAllocActionName},
      {DeallocAction, rt::SimpleExecutorMemoryManagerReleaseWrapperName},
      {TargetMemMgr, rt::SimpleExecutorMemoryManagerInstanceName},
  });
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
  if (G.getTargetTriple().getObjectFormat() != Triple::ELF)
    return;

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
  auto [It, Inserted] = PendingObjs.try_emplace(
      &MR, InputObj.getBufferIdentifier(), std::move(*Alloc), Ctx, ES);
  assert(Inserted && "One debug object per materialization");
  memcpy(It->second.getMutBuffer().data(), InputObj.getBufferStart(), Size);
}

DebugObject *
ELFDebugObjectPlugin::getPendingDebugObj(MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(PendingObjsLock);
  auto It = PendingObjs.find(&MR);
  return It == PendingObjs.end() ? nullptr : &It->second;
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
    DebugObj->visitSections(
        [&G, &SectionsPatched, &HasDebugSections](StringRef Name) {
          SectionsPatched += 1;
          if (isDwarfSection(Name))
            HasDebugSections = true;
          Section *S = G.findSectionByName(Name);
          assert(S && "No graph section for object section");
          return SectionRange(*S).getStart();
        });

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
    auto &Alloc = DebugObj->getTargetAlloc();

    // FIXME: The lookup in the segment info here is a workaround. The below
    // FA->release() is supposed to provide the base address in target memory,
    // but InProcessMemoryManager returns the address of a FinalizedAllocInfo
    // helper instead.
    auto ROSeg = Alloc.getSegInfo(MemProt::Read);
    ExecutorAddrRange R(ROSeg.Addr, ROSeg.WorkingMem.size());
    Alloc.finalize([this, R, &MR](Expected<DebugObject::FinalizedAlloc> FA) {
      DebugObject *DebugObj = getPendingDebugObj(MR);
      if (!FA)
        DebugObj->failMaterialization(FA.takeError());

      // Dealloc action from the LinkGraph's allocation will free target memory
      FA->release();

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
    if (R->empty())
      return Error::success();

    // Step 5: Use allocation actions to (1) register the debug object with the
    // GDB JIT Interface and (2) free the debug object when the corresponding
    // code is removed
    using namespace shared;
    G.allocActions().push_back(createAllocActions(*R));
    return Error::success();
  });
}

shared::AllocActionCallPair
ELFDebugObjectPlugin::createAllocActions(ExecutorAddrRange R) {
  using namespace shared;
  // Add the target memory range to __jit_debug_descriptor
  auto Init = cantFail(
      WrapperFunctionCall::Create<SPSArgList<SPSExecutorAddrRange, bool>>(
          RegistrationAction, R, AutoRegisterCode));
  // Free the debug object's target memory block
  auto Fini =
      cantFail(WrapperFunctionCall::Create<
               SPSArgList<SPSExecutorAddr, SPSSequence<SPSExecutorAddr>>>(
          DeallocAction, TargetMemMgr, ArrayRef<ExecutorAddr>(R.Start)));
  return {Init, Fini};
}

Error ELFDebugObjectPlugin::notifyEmitted(MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(PendingObjsLock);
  PendingObjs.erase(&MR);
  return Error::success();
}

Error ELFDebugObjectPlugin::notifyFailed(MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(PendingObjsLock);
  PendingObjs.erase(&MR);
  return Error::success();
}

} // namespace orc
} // namespace llvm
