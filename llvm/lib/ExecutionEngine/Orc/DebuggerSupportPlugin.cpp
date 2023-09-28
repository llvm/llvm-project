//===------- DebuggerSupportPlugin.cpp - Utils for debugger support -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/DebuggerSupportPlugin.h"
#include "llvm/ExecutionEngine/Orc/MachOBuilder.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/MachO.h"

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;

static const char *SynthDebugSectionName = "__jitlink_synth_debug_object";

namespace {

class MachODebugObjectSynthesizerBase
    : public GDBJITDebugInfoRegistrationPlugin::DebugSectionSynthesizer {
public:
  static bool isDebugSection(Section &Sec) {
    return Sec.getName().startswith("__DWARF,");
  }

  MachODebugObjectSynthesizerBase(LinkGraph &G, ExecutorAddr RegisterActionAddr)
      : G(G), RegisterActionAddr(RegisterActionAddr) {}
  virtual ~MachODebugObjectSynthesizerBase() = default;

  Error preserveDebugSections() {
    if (G.findSectionByName(SynthDebugSectionName)) {
      LLVM_DEBUG({
        dbgs() << "MachODebugObjectSynthesizer skipping graph " << G.getName()
               << " which contains an unexpected existing "
               << SynthDebugSectionName << " section.\n";
      });
      return Error::success();
    }

    LLVM_DEBUG({
      dbgs() << "MachODebugObjectSynthesizer visiting graph " << G.getName()
             << "\n";
    });
    for (auto &Sec : G.sections()) {
      if (!isDebugSection(Sec))
        continue;
      // Preserve blocks in this debug section by marking one existing symbol
      // live for each block, and introducing a new live, anonymous symbol for
      // each currently unreferenced block.
      LLVM_DEBUG({
        dbgs() << "  Preserving debug section " << Sec.getName() << "\n";
      });
      SmallSet<Block *, 8> PreservedBlocks;
      for (auto *Sym : Sec.symbols()) {
        bool NewPreservedBlock =
            PreservedBlocks.insert(&Sym->getBlock()).second;
        if (NewPreservedBlock)
          Sym->setLive(true);
      }
      for (auto *B : Sec.blocks())
        if (!PreservedBlocks.count(B))
          G.addAnonymousSymbol(*B, 0, 0, false, true);
    }

    return Error::success();
  }

protected:
  LinkGraph &G;
  ExecutorAddr RegisterActionAddr;
};

template <typename MachOTraits>
class MachODebugObjectSynthesizer : public MachODebugObjectSynthesizerBase {
public:
  MachODebugObjectSynthesizer(ExecutionSession &ES, LinkGraph &G,
                              ExecutorAddr RegisterActionAddr)
      : MachODebugObjectSynthesizerBase(G, RegisterActionAddr),
        Builder(ES.getPageSize()) {}

  using MachODebugObjectSynthesizerBase::MachODebugObjectSynthesizerBase;

  Error startSynthesis() override {
    LLVM_DEBUG({
      dbgs() << "Creating " << SynthDebugSectionName << " for " << G.getName()
             << "\n";
    });

    auto &SDOSec = G.createSection(SynthDebugSectionName, MemProt::Read);

    for (auto &Sec : G.sections()) {
      if (Sec.blocks().empty())
        continue;

      // Skip sections whose name's don't fit the MachO standard.
      if (Sec.getName().empty() || Sec.getName().size() > 33 ||
          Sec.getName().find(',') > 16)
        continue;

      if (isDebugSection(Sec))
        DebugSections.push_back({&Sec, nullptr});
      else if (Sec.getMemLifetime() != MemLifetime::NoAlloc)
        NonDebugSections.push_back({&Sec, nullptr});
    }

    // Write MachO header and debug section load commands.
    Builder.Header.filetype = MachO::MH_OBJECT;
    switch (G.getTargetTriple().getArch()) {
    case Triple::x86_64:
      Builder.Header.cputype = MachO::CPU_TYPE_X86_64;
      Builder.Header.cpusubtype = MachO::CPU_SUBTYPE_X86_64_ALL;
      break;
    case Triple::aarch64:
      Builder.Header.cputype = MachO::CPU_TYPE_ARM64;
      Builder.Header.cpusubtype = MachO::CPU_SUBTYPE_ARM64_ALL;
      break;
    default:
      llvm_unreachable("Unsupported architecture");
    }

    Seg = &Builder.addSegment("");

    for (auto &DSec : DebugSections) {
      auto [SegName, SecName] = DSec.GraphSec->getName().split(',');
      DSec.BuilderSec = &Seg->addSection(SecName, SegName);

      SectionRange SR(*DSec.GraphSec);
      DSec.BuilderSec->Content.Size = SR.getSize();
      if (!SR.empty())
        DSec.BuilderSec->align = Log2_64(SR.getFirstBlock()->getAlignment());
    }

    for (auto &NDSP : NonDebugSections) {
      auto [SegName, SecName] = NDSP.GraphSec->getName().split(',');
      NDSP.BuilderSec = &Seg->addSection(SecName, SegName);
      SectionRange SR(*NDSP.GraphSec);
      if (!SR.empty())
        NDSP.BuilderSec->align = Log2_64(SR.getFirstBlock()->getAlignment());

      // Add stabs.
      for (auto *Sym : NDSP.GraphSec->symbols()) {
        // Skip anonymous symbols.
        if (!Sym->hasName())
          continue;

        uint8_t SymType = Sym->isCallable() ? MachO::N_FUN : MachO::N_GSYM;

        Builder.addSymbol("", MachO::N_BNSYM, 1, 0, 0);
        StabSymbols.push_back(
            {*Sym, Builder.addSymbol(Sym->getName(), SymType, 1, 0, 0),
             Builder.addSymbol(Sym->getName(), SymType, 0, 0, 0)});
        Builder.addSymbol("", MachO::N_ENSYM, 1, 0, 0);
      }
    }

    size_t DebugObjectSize = Builder.layout();

    MachOContainerBlock = &G.createMutableContentBlock(
        SDOSec, G.allocateBuffer(DebugObjectSize), orc::ExecutorAddr(), 8, 0);

    return Error::success();
  }

  Error completeSynthesisAndRegister() override {
    if (!MachOContainerBlock) {
      LLVM_DEBUG({
        dbgs() << "Not writing MachO debug object header for " << G.getName()
               << " since createDebugSection failed\n";
      });

      return Error::success();
    }
    ExecutorAddr MaxAddr;
    for (auto &NDSec : NonDebugSections) {
      SectionRange SR(*NDSec.GraphSec);
      NDSec.BuilderSec->addr = SR.getStart().getValue();
      NDSec.BuilderSec->size = SR.getSize();
      NDSec.BuilderSec->offset = SR.getStart().getValue();
      if (SR.getEnd() > MaxAddr)
        MaxAddr = SR.getEnd();
    }

    for (auto &DSec : DebugSections) {
      if (DSec.GraphSec->blocks_size() != 1)
        return make_error<StringError>(
            "Unexpected number of blocks in debug info section",
            inconvertibleErrorCode());

      if (ExecutorAddr(DSec.BuilderSec->addr) + DSec.BuilderSec->size > MaxAddr)
        MaxAddr = ExecutorAddr(DSec.BuilderSec->addr) + DSec.BuilderSec->size;

      auto &B = **DSec.GraphSec->blocks().begin();
      DSec.BuilderSec->Content.Data = B.getContent().data();
      DSec.BuilderSec->Content.Size = B.getContent().size();
      DSec.BuilderSec->flags |= MachO::S_ATTR_DEBUG;
    }

    LLVM_DEBUG({
      dbgs() << "Writing MachO debug object header for " << G.getName() << "\n";
    });

    // Update stab symbol addresses.
    for (auto &SS : StabSymbols) {
      SS.StartStab.nlist().n_value = SS.Sym.getAddress().getValue();
      SS.EndStab.nlist().n_value = SS.Sym.getSize();
    }

    Builder.write(MachOContainerBlock->getAlreadyMutableContent());

    static constexpr bool AutoRegisterCode = true;
    SectionRange R(MachOContainerBlock->getSection());
    G.allocActions().push_back(
        {cantFail(shared::WrapperFunctionCall::Create<
                  shared::SPSArgList<shared::SPSExecutorAddrRange, bool>>(
             RegisterActionAddr, R.getRange(), AutoRegisterCode)),
         {}});

    return Error::success();
  }

private:
  struct SectionPair {
    Section *GraphSec = nullptr;
    typename MachOBuilder<MachOTraits>::Section *BuilderSec = nullptr;
  };

  struct StabSymbolsEntry {
    using RelocTarget = typename MachOBuilder<MachOTraits>::RelocTarget;

    StabSymbolsEntry(Symbol &Sym, RelocTarget StartStab, RelocTarget EndStab)
        : Sym(Sym), StartStab(StartStab), EndStab(EndStab) {}

    Symbol &Sym;
    RelocTarget StartStab, EndStab;
  };

  using BuilderType = MachOBuilder<MachOTraits>;

  Block *MachOContainerBlock = nullptr;
  MachOBuilder<MachOTraits> Builder;
  typename MachOBuilder<MachOTraits>::Segment *Seg = nullptr;
  std::vector<StabSymbolsEntry> StabSymbols;
  SmallVector<SectionPair, 16> DebugSections;
  SmallVector<SectionPair, 16> NonDebugSections;
};

} // end anonymous namespace

namespace llvm {
namespace orc {

Expected<std::unique_ptr<GDBJITDebugInfoRegistrationPlugin>>
GDBJITDebugInfoRegistrationPlugin::Create(ExecutionSession &ES,
                                          JITDylib &ProcessJD,
                                          const Triple &TT) {
  auto RegisterActionAddr =
      TT.isOSBinFormatMachO()
          ? ES.intern("_llvm_orc_registerJITLoaderGDBAllocAction")
          : ES.intern("llvm_orc_registerJITLoaderGDBAllocAction");

  if (auto RegisterSym = ES.lookup({&ProcessJD}, RegisterActionAddr))
    return std::make_unique<GDBJITDebugInfoRegistrationPlugin>(
        RegisterSym->getAddress());
  else
    return RegisterSym.takeError();
}

Error GDBJITDebugInfoRegistrationPlugin::notifyFailed(
    MaterializationResponsibility &MR) {
  return Error::success();
}

Error GDBJITDebugInfoRegistrationPlugin::notifyRemovingResources(
    JITDylib &JD, ResourceKey K) {
  return Error::success();
}

void GDBJITDebugInfoRegistrationPlugin::notifyTransferringResources(
    JITDylib &JD, ResourceKey DstKey, ResourceKey SrcKey) {}

void GDBJITDebugInfoRegistrationPlugin::modifyPassConfig(
    MaterializationResponsibility &MR, LinkGraph &LG,
    PassConfiguration &PassConfig) {

  if (LG.getTargetTriple().getObjectFormat() == Triple::MachO)
    modifyPassConfigForMachO(MR, LG, PassConfig);
  else {
    LLVM_DEBUG({
      dbgs() << "GDBJITDebugInfoRegistrationPlugin skipping unspported graph "
             << LG.getName() << "(triple = " << LG.getTargetTriple().str()
             << "\n";
    });
  }
}

void GDBJITDebugInfoRegistrationPlugin::modifyPassConfigForMachO(
    MaterializationResponsibility &MR, jitlink::LinkGraph &LG,
    jitlink::PassConfiguration &PassConfig) {

  switch (LG.getTargetTriple().getArch()) {
  case Triple::x86_64:
  case Triple::aarch64:
    // Supported, continue.
    assert(LG.getPointerSize() == 8 && "Graph has incorrect pointer size");
    assert(LG.getEndianness() == support::little &&
           "Graph has incorrect endianness");
    break;
  default:
    // Unsupported.
    LLVM_DEBUG({
      dbgs() << "GDBJITDebugInfoRegistrationPlugin skipping unsupported "
             << "MachO graph " << LG.getName()
             << "(triple = " << LG.getTargetTriple().str()
             << ", pointer size = " << LG.getPointerSize() << ", endianness = "
             << (LG.getEndianness() == support::big ? "big" : "little")
             << ")\n";
    });
    return;
  }

  // Scan for debug sections. If we find one then install passes.
  bool HasDebugSections = false;
  for (auto &Sec : LG.sections())
    if (MachODebugObjectSynthesizerBase::isDebugSection(Sec)) {
      HasDebugSections = true;
      break;
    }

  if (HasDebugSections) {
    LLVM_DEBUG({
      dbgs() << "GDBJITDebugInfoRegistrationPlugin: Graph " << LG.getName()
             << " contains debug info. Installing debugger support passes.\n";
    });

    auto MDOS = std::make_shared<MachODebugObjectSynthesizer<MachO64LE>>(
        MR.getTargetJITDylib().getExecutionSession(), LG, RegisterActionAddr);
    PassConfig.PrePrunePasses.push_back(
        [=](LinkGraph &G) { return MDOS->preserveDebugSections(); });
    PassConfig.PostPrunePasses.push_back(
        [=](LinkGraph &G) { return MDOS->startSynthesis(); });
    PassConfig.PostFixupPasses.push_back(
        [=](LinkGraph &G) { return MDOS->completeSynthesisAndRegister(); });
  } else {
    LLVM_DEBUG({
      dbgs() << "GDBJITDebugInfoRegistrationPlugin: Graph " << LG.getName()
             << " contains no debug info. Skipping.\n";
    });
  }
}

} // namespace orc
} // namespace llvm
