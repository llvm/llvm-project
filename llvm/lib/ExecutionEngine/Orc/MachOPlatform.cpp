//===------ MachOPlatform.cpp - Utilities for executing MachO in Orc ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/MachOPlatform.h"

#include "llvm/BinaryFormat/MachO.h"
#include "llvm/ExecutionEngine/Orc/DebugUtils.h"
#include "llvm/Support/BinaryByteStream.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

MachOPlatform::MachOPlatform(
    ExecutionSession &ES, ObjectLinkingLayer &ObjLinkingLayer,
    std::unique_ptr<MemoryBuffer> StandardSymbolsObject)
    : ES(ES), ObjLinkingLayer(ObjLinkingLayer),
      StandardSymbolsObject(std::move(StandardSymbolsObject)) {
  ObjLinkingLayer.addPlugin(std::make_unique<InitScraperPlugin>(*this));
}

Error MachOPlatform::setupJITDylib(JITDylib &JD) {
  auto ObjBuffer = MemoryBuffer::getMemBuffer(
      StandardSymbolsObject->getMemBufferRef(), false);
  return ObjLinkingLayer.add(JD, std::move(ObjBuffer));
}

Error MachOPlatform::notifyAdding(ResourceTracker &RT,
                                  const MaterializationUnit &MU) {
  auto &JD = RT.getJITDylib();
  const auto &InitSym = MU.getInitializerSymbol();
  if (!InitSym)
    return Error::success();

  RegisteredInitSymbols[&JD].add(InitSym,
                                 SymbolLookupFlags::WeaklyReferencedSymbol);
  LLVM_DEBUG({
    dbgs() << "MachOPlatform: Registered init symbol " << *InitSym << " for MU "
           << MU.getName() << "\n";
  });
  return Error::success();
}

Error MachOPlatform::notifyRemoving(ResourceTracker &RT) {
  llvm_unreachable("Not supported yet");
}

Expected<MachOPlatform::InitializerSequence>
MachOPlatform::getInitializerSequence(JITDylib &JD) {

  LLVM_DEBUG({
    dbgs() << "MachOPlatform: Building initializer sequence for "
           << JD.getName() << "\n";
  });

  std::vector<JITDylibSP> DFSLinkOrder;

  while (true) {

    DenseMap<JITDylib *, SymbolLookupSet> NewInitSymbols;

    ES.runSessionLocked([&]() {
      DFSLinkOrder = JD.getDFSLinkOrder();

      for (auto &InitJD : DFSLinkOrder) {
        auto RISItr = RegisteredInitSymbols.find(InitJD.get());
        if (RISItr != RegisteredInitSymbols.end()) {
          NewInitSymbols[InitJD.get()] = std::move(RISItr->second);
          RegisteredInitSymbols.erase(RISItr);
        }
      }
    });

    if (NewInitSymbols.empty())
      break;

    LLVM_DEBUG({
      dbgs() << "MachOPlatform: Issuing lookups for new init symbols: "
                "(lookup may require multiple rounds)\n";
      for (auto &KV : NewInitSymbols)
        dbgs() << "  \"" << KV.first->getName() << "\": " << KV.second << "\n";
    });

    // Outside the lock, issue the lookup.
    if (auto R = lookupInitSymbols(JD.getExecutionSession(), NewInitSymbols))
      ; // Nothing to do in the success case.
    else
      return R.takeError();
  }

  LLVM_DEBUG({
    dbgs() << "MachOPlatform: Init symbol lookup complete, building init "
              "sequence\n";
  });

  // Lock again to collect the initializers.
  InitializerSequence FullInitSeq;
  {
    std::lock_guard<std::mutex> Lock(InitSeqsMutex);
    for (auto &InitJD : reverse(DFSLinkOrder)) {
      LLVM_DEBUG({
        dbgs() << "MachOPlatform: Appending inits for \"" << InitJD->getName()
               << "\" to sequence\n";
      });
      auto ISItr = InitSeqs.find(InitJD.get());
      if (ISItr != InitSeqs.end()) {
        FullInitSeq.emplace_back(InitJD.get(), std::move(ISItr->second));
        InitSeqs.erase(ISItr);
      }
    }
  }

  return FullInitSeq;
}

Expected<MachOPlatform::DeinitializerSequence>
MachOPlatform::getDeinitializerSequence(JITDylib &JD) {
  std::vector<JITDylibSP> DFSLinkOrder = JD.getDFSLinkOrder();

  DeinitializerSequence FullDeinitSeq;
  {
    std::lock_guard<std::mutex> Lock(InitSeqsMutex);
    for (auto &DeinitJD : DFSLinkOrder) {
      FullDeinitSeq.emplace_back(DeinitJD.get(), MachOJITDylibDeinitializers());
    }
  }

  return FullDeinitSeq;
}

void MachOPlatform::registerInitInfo(JITDylib &JD,
                                     JITTargetAddress ObjCImageInfoAddr,
                                     ExecutorAddressRange ModInits,
                                     ExecutorAddressRange ObjCSelRefs,
                                     ExecutorAddressRange ObjCClassList) {
  std::lock_guard<std::mutex> Lock(InitSeqsMutex);

  auto &InitSeq = InitSeqs[&JD];

  InitSeq.setObjCImageInfoAddr(ObjCImageInfoAddr);

  if (ModInits.StartAddress)
    InitSeq.addModInitsSection(std::move(ModInits));

  if (ObjCSelRefs.StartAddress)
    InitSeq.addObjCSelRefsSection(std::move(ObjCSelRefs));

  if (ObjCClassList.StartAddress)
    InitSeq.addObjCClassListSection(std::move(ObjCClassList));
}

static Expected<ExecutorAddressRange> getSectionExtent(jitlink::LinkGraph &G,
                                                       StringRef SectionName) {
  auto *Sec = G.findSectionByName(SectionName);
  if (!Sec)
    return ExecutorAddressRange();
  jitlink::SectionRange R(*Sec);
  if (R.getSize() % G.getPointerSize() != 0)
    return make_error<StringError>(SectionName + " section size is not a "
                                                 "multiple of the pointer size",
                                   inconvertibleErrorCode());
  return ExecutorAddressRange(ExecutorAddress(R.getStart()),
                              ExecutorAddress(R.getEnd()));
}

void MachOPlatform::InitScraperPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, jitlink::LinkGraph &LG,
    jitlink::PassConfiguration &Config) {

  if (!MR.getInitializerSymbol())
    return;

  Config.PrePrunePasses.push_back([this, &MR](jitlink::LinkGraph &G) -> Error {
    JITLinkSymbolSet InitSectionSyms;
    preserveInitSectionIfPresent(InitSectionSyms, G, "__DATA,__mod_init_func");
    preserveInitSectionIfPresent(InitSectionSyms, G, "__DATA,__objc_selrefs");
    preserveInitSectionIfPresent(InitSectionSyms, G, "__DATA,__objc_classlist");

    if (!InitSectionSyms.empty()) {
      std::lock_guard<std::mutex> Lock(InitScraperMutex);
      InitSymbolDeps[&MR] = std::move(InitSectionSyms);
    }

    if (auto Err = processObjCImageInfo(G, MR))
      return Err;

    return Error::success();
  });

  Config.PostFixupPasses.push_back([this, &JD = MR.getTargetJITDylib()](
                                       jitlink::LinkGraph &G) -> Error {
    ExecutorAddressRange ModInits, ObjCSelRefs, ObjCClassList;

    JITTargetAddress ObjCImageInfoAddr = 0;
    if (auto *ObjCImageInfoSec =
            G.findSectionByName("__DATA,__objc_image_info")) {
      if (auto Addr = jitlink::SectionRange(*ObjCImageInfoSec).getStart())
        ObjCImageInfoAddr = Addr;
    }

    // Record __mod_init_func.
    if (auto ModInitsOrErr = getSectionExtent(G, "__DATA,__mod_init_func"))
      ModInits = std::move(*ModInitsOrErr);
    else
      return ModInitsOrErr.takeError();

    // Record __objc_selrefs.
    if (auto ObjCSelRefsOrErr = getSectionExtent(G, "__DATA,__objc_selrefs"))
      ObjCSelRefs = std::move(*ObjCSelRefsOrErr);
    else
      return ObjCSelRefsOrErr.takeError();

    // Record __objc_classlist.
    if (auto ObjCClassListOrErr =
            getSectionExtent(G, "__DATA,__objc_classlist"))
      ObjCClassList = std::move(*ObjCClassListOrErr);
    else
      return ObjCClassListOrErr.takeError();

    // Dump the scraped inits.
    LLVM_DEBUG({
      dbgs() << "MachOPlatform: Scraped " << G.getName() << " init sections:\n";
      dbgs() << "  __objc_selrefs: ";
      auto NumObjCSelRefs = ObjCSelRefs.size().getValue() / sizeof(uintptr_t);
      if (NumObjCSelRefs)
        dbgs() << NumObjCSelRefs << " pointer(s) at "
               << formatv("{0:x16}", ObjCSelRefs.StartAddress.getValue())
               << "\n";
      else
        dbgs() << "none\n";

      dbgs() << "  __objc_classlist: ";
      auto NumObjCClasses = ObjCClassList.size().getValue() / sizeof(uintptr_t);
      if (NumObjCClasses)
        dbgs() << NumObjCClasses << " pointer(s) at "
               << formatv("{0:x16}", ObjCClassList.StartAddress.getValue())
               << "\n";
      else
        dbgs() << "none\n";

      dbgs() << "  __mod_init_func: ";
      auto NumModInits = ModInits.size().getValue() / sizeof(uintptr_t);
      if (NumModInits)
        dbgs() << NumModInits << " pointer(s) at "
               << formatv("{0:x16}", ModInits.StartAddress.getValue()) << "\n";
      else
        dbgs() << "none\n";
    });

    MP.registerInitInfo(JD, ObjCImageInfoAddr, std::move(ModInits),
                        std::move(ObjCSelRefs), std::move(ObjCClassList));

    return Error::success();
  });
}

ObjectLinkingLayer::Plugin::SyntheticSymbolDependenciesMap
MachOPlatform::InitScraperPlugin::getSyntheticSymbolDependencies(
    MaterializationResponsibility &MR) {
  std::lock_guard<std::mutex> Lock(InitScraperMutex);
  auto I = InitSymbolDeps.find(&MR);
  if (I != InitSymbolDeps.end()) {
    SyntheticSymbolDependenciesMap Result;
    Result[MR.getInitializerSymbol()] = std::move(I->second);
    InitSymbolDeps.erase(&MR);
    return Result;
  }
  return SyntheticSymbolDependenciesMap();
}

void MachOPlatform::InitScraperPlugin::preserveInitSectionIfPresent(
    JITLinkSymbolSet &Symbols, jitlink::LinkGraph &G, StringRef SectionName) {
  if (auto *Sec = G.findSectionByName(SectionName)) {
    auto SecBlocks = Sec->blocks();
    if (!llvm::empty(SecBlocks))
      Symbols.insert(
          &G.addAnonymousSymbol(**SecBlocks.begin(), 0, 0, false, true));
  }
}

Error MachOPlatform::InitScraperPlugin::processObjCImageInfo(
    jitlink::LinkGraph &G, MaterializationResponsibility &MR) {

  // If there's an ObjC imagine info then either
  //   (1) It's the first __objc_imageinfo we've seen in this JITDylib. In
  //       this case we name and record it.
  // OR
  //   (2) We already have a recorded __objc_imageinfo for this JITDylib,
  //       in which case we just verify it.
  auto *ObjCImageInfo = G.findSectionByName("__objc_imageinfo");
  if (!ObjCImageInfo)
    return Error::success();

  auto ObjCImageInfoBlocks = ObjCImageInfo->blocks();

  // Check that the section is not empty if present.
  if (llvm::empty(ObjCImageInfoBlocks))
    return make_error<StringError>("Empty __objc_imageinfo section in " +
                                       G.getName(),
                                   inconvertibleErrorCode());

  // Check that there's only one block in the section.
  if (std::next(ObjCImageInfoBlocks.begin()) != ObjCImageInfoBlocks.end())
    return make_error<StringError>("Multiple blocks in __objc_imageinfo "
                                   "section in " +
                                       G.getName(),
                                   inconvertibleErrorCode());

  // Check that the __objc_imageinfo section is unreferenced.
  // FIXME: We could optimize this check if Symbols had a ref-count.
  for (auto &Sec : G.sections()) {
    if (&Sec != ObjCImageInfo)
      for (auto *B : Sec.blocks())
        for (auto &E : B->edges())
          if (E.getTarget().isDefined() &&
              &E.getTarget().getBlock().getSection() == ObjCImageInfo)
            return make_error<StringError>("__objc_imageinfo is referenced "
                                           "within file " +
                                               G.getName(),
                                           inconvertibleErrorCode());
  }

  auto &ObjCImageInfoBlock = **ObjCImageInfoBlocks.begin();
  auto *ObjCImageInfoData = ObjCImageInfoBlock.getContent().data();
  auto Version = support::endian::read32(ObjCImageInfoData, G.getEndianness());
  auto Flags =
      support::endian::read32(ObjCImageInfoData + 4, G.getEndianness());

  // Lock the mutex while we verify / update the ObjCImageInfos map.
  std::lock_guard<std::mutex> Lock(InitScraperMutex);

  auto ObjCImageInfoItr = ObjCImageInfos.find(&MR.getTargetJITDylib());
  if (ObjCImageInfoItr != ObjCImageInfos.end()) {
    // We've already registered an __objc_imageinfo section. Verify the
    // content of this new section matches, then delete it.
    if (ObjCImageInfoItr->second.first != Version)
      return make_error<StringError>(
          "ObjC version in " + G.getName() +
              " does not match first registered version",
          inconvertibleErrorCode());
    if (ObjCImageInfoItr->second.second != Flags)
      return make_error<StringError>("ObjC flags in " + G.getName() +
                                         " do not match first registered flags",
                                     inconvertibleErrorCode());

    // __objc_imageinfo is valid. Delete the block.
    for (auto *S : ObjCImageInfo->symbols())
      G.removeDefinedSymbol(*S);
    G.removeBlock(ObjCImageInfoBlock);
  } else {
    // We haven't registered an __objc_imageinfo section yet. Register and
    // move on. The section should already be marked no-dead-strip.
    ObjCImageInfos[&MR.getTargetJITDylib()] = std::make_pair(Version, Flags);
  }

  return Error::success();
}

} // End namespace orc.
} // End namespace llvm.
