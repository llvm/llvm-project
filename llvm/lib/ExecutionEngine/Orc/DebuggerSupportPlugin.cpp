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

struct MachO64LE {
  using UIntPtr = uint64_t;

  using Header = MachO::mach_header_64;
  using SegmentLC = MachO::segment_command_64;
  using Section = MachO::section_64;
  using NList = MachO::nlist_64;

  static constexpr support::endianness Endianness = support::little;
  static constexpr const uint32_t Magic = MachO::MH_MAGIC_64;
  static constexpr const uint32_t SegmentCmd = MachO::LC_SEGMENT_64;
};

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
private:
  class MachOStructWriter {
  public:
    MachOStructWriter(MutableArrayRef<char> Buffer) : Buffer(Buffer) {}

    size_t getOffset() const { return Offset; }

    template <typename MachOStruct> void write(MachOStruct S) {
      assert(Offset + sizeof(S) <= Buffer.size() &&
             "Container block overflow while constructing debug MachO");
      if (MachOTraits::Endianness != support::endian::system_endianness())
        MachO::swapStruct(S);
      memcpy(Buffer.data() + Offset, &S, sizeof(S));
      Offset += sizeof(S);
    }

  private:
    MutableArrayRef<char> Buffer;
    size_t Offset = 0;
  };

public:
  using MachODebugObjectSynthesizerBase::MachODebugObjectSynthesizerBase;

  Error startSynthesis() override {
    LLVM_DEBUG({
      dbgs() << "Creating " << SynthDebugSectionName << " for " << G.getName()
             << "\n";
    });
    auto &SDOSec = G.createSection(SynthDebugSectionName, MemProt::Read);

    struct DebugSectionInfo {
      Section *Sec = nullptr;
      StringRef SegName;
      StringRef SecName;
      uint64_t Alignment = 0;
      orc::ExecutorAddr StartAddr;
      uint64_t Size = 0;
    };

    SmallVector<DebugSectionInfo, 12> DebugSecInfos;
    size_t NumSections = 0;
    for (auto &Sec : G.sections()) {
      if (Sec.blocks().empty())
        continue;

      ++NumSections;
      if (isDebugSection(Sec)) {
        size_t SepPos = Sec.getName().find(',');
        if (SepPos > 16 || (Sec.getName().size() - (SepPos + 1) > 16)) {
          LLVM_DEBUG({
            dbgs() << "Skipping debug object synthesis for graph "
                   << G.getName()
                   << ": encountered non-standard DWARF section name \""
                   << Sec.getName() << "\"\n";
          });
          return Error::success();
        }
        DebugSecInfos.push_back({&Sec, Sec.getName().substr(0, SepPos),
                                 Sec.getName().substr(SepPos + 1), 0,
                                 orc::ExecutorAddr(), 0});
      } else {
        NonDebugSections.push_back(&Sec);

        // If the first block in the section has a non-zero alignment offset
        // then we need to add a padding block, since the section command in
        // the header doesn't allow for aligment offsets.
        SectionRange R(Sec);
        if (!R.empty()) {
          auto &FB = *R.getFirstBlock();
          if (FB.getAlignmentOffset() != 0) {
            auto Padding = G.allocateBuffer(FB.getAlignmentOffset());
            memset(Padding.data(), 0, Padding.size());
            G.createContentBlock(Sec, Padding,
                                 FB.getAddress() - FB.getAlignmentOffset(),
                                 FB.getAlignment(), 0);
          }
        }
      }
    }

    // Create container block.
    size_t SectionsCmdSize =
        sizeof(typename MachOTraits::Section) * NumSections;
    size_t SegmentLCSize =
        sizeof(typename MachOTraits::SegmentLC) + SectionsCmdSize;
    size_t ContainerBlockSize =
        sizeof(typename MachOTraits::Header) + SegmentLCSize;
    auto ContainerBlockContent = G.allocateBuffer(ContainerBlockSize);
    MachOContainerBlock = &G.createMutableContentBlock(
        SDOSec, ContainerBlockContent, orc::ExecutorAddr(), 8, 0);

    // Copy debug section blocks and symbols.
    orc::ExecutorAddr NextBlockAddr(MachOContainerBlock->getSize());
    for (auto &SI : DebugSecInfos) {
      assert(!SI.Sec->blocks().empty() && "Empty debug info section?");

      // Update addresses in debug section.
      LLVM_DEBUG({
        dbgs() << "  Appending " << SI.Sec->getName() << " ("
               << SI.Sec->blocks_size() << " block(s)) at "
               << formatv("{0:x8}", NextBlockAddr) << "\n";
      });
      for (auto *B : SI.Sec->blocks()) {
        NextBlockAddr = alignToBlock(NextBlockAddr, *B);
        B->setAddress(NextBlockAddr);
        NextBlockAddr += B->getSize();
      }

      auto &FirstBlock = **SI.Sec->blocks().begin();
      if (FirstBlock.getAlignmentOffset() != 0)
        return make_error<StringError>(
            "First block in " + SI.Sec->getName() +
                " section has non-zero alignment offset",
            inconvertibleErrorCode());
      if (FirstBlock.getAlignment() > std::numeric_limits<uint32_t>::max())
        return make_error<StringError>("First block in " + SI.Sec->getName() +
                                           " has alignment >4Gb",
                                       inconvertibleErrorCode());

      SI.Alignment = FirstBlock.getAlignment();
      SI.StartAddr = FirstBlock.getAddress();
      SI.Size = NextBlockAddr - SI.StartAddr;
      G.mergeSections(SDOSec, *SI.Sec);
      SI.Sec = nullptr;
    }
    size_t DebugSectionsSize =
        NextBlockAddr - orc::ExecutorAddr(MachOContainerBlock->getSize());

    // Write MachO header and debug section load commands.
    MachOStructWriter Writer(MachOContainerBlock->getAlreadyMutableContent());
    typename MachOTraits::Header Hdr;
    memset(&Hdr, 0, sizeof(Hdr));
    Hdr.magic = MachOTraits::Magic;
    switch (G.getTargetTriple().getArch()) {
    case Triple::x86_64:
      Hdr.cputype = MachO::CPU_TYPE_X86_64;
      Hdr.cpusubtype = MachO::CPU_SUBTYPE_X86_64_ALL;
      break;
    case Triple::aarch64:
      Hdr.cputype = MachO::CPU_TYPE_ARM64;
      Hdr.cpusubtype = MachO::CPU_SUBTYPE_ARM64_ALL;
      break;
    default:
      llvm_unreachable("Unsupported architecture");
    }
    Hdr.filetype = MachO::MH_OBJECT;
    Hdr.ncmds = 1;
    Hdr.sizeofcmds = SegmentLCSize;
    Hdr.flags = 0;
    Writer.write(Hdr);

    typename MachOTraits::SegmentLC SegLC;
    memset(&SegLC, 0, sizeof(SegLC));
    SegLC.cmd = MachOTraits::SegmentCmd;
    SegLC.cmdsize = SegmentLCSize;
    SegLC.vmaddr = ContainerBlockSize;
    SegLC.vmsize = DebugSectionsSize;
    SegLC.fileoff = ContainerBlockSize;
    SegLC.filesize = DebugSectionsSize;
    SegLC.maxprot =
        MachO::VM_PROT_READ | MachO::VM_PROT_WRITE | MachO::VM_PROT_EXECUTE;
    SegLC.initprot =
        MachO::VM_PROT_READ | MachO::VM_PROT_WRITE | MachO::VM_PROT_EXECUTE;
    SegLC.nsects = NumSections;
    SegLC.flags = 0;
    Writer.write(SegLC);

    StringSet<> ExistingLongNames;
    for (auto &SI : DebugSecInfos) {
      typename MachOTraits::Section Sec;
      memset(&Sec, 0, sizeof(Sec));
      memcpy(Sec.sectname, SI.SecName.data(), SI.SecName.size());
      memcpy(Sec.segname, SI.SegName.data(), SI.SegName.size());
      Sec.addr = SI.StartAddr.getValue();
      Sec.size = SI.Size;
      Sec.offset = SI.StartAddr.getValue();
      Sec.align = SI.Alignment;
      Sec.reloff = 0;
      Sec.nreloc = 0;
      Sec.flags = MachO::S_ATTR_DEBUG;
      Writer.write(Sec);
    }

    // Set MachOContainerBlock to indicate success to
    // completeSynthesisAndRegister.
    NonDebugSectionsStart = Writer.getOffset();
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

    LLVM_DEBUG({
      dbgs() << "Writing MachO debug object header for " << G.getName() << "\n";
    });

    MachOStructWriter Writer(
        MachOContainerBlock->getAlreadyMutableContent().drop_front(
            NonDebugSectionsStart));

    unsigned LongSectionNameIdx = 0;
    for (auto *Sec : NonDebugSections) {
      size_t SepPos = Sec->getName().find(',');
      StringRef SegName, SecName;
      std::string CustomSecName;

      if ((SepPos == StringRef::npos && Sec->getName().size() <= 16)) {
        // No embedded segment name, short section name.
        SegName = "__JITLINK_CUSTOM";
        SecName = Sec->getName();
      } else if (SepPos < 16 && (Sec->getName().size() - (SepPos + 1) <= 16)) {
        // Canonical embedded segment and section name.
        SegName = Sec->getName().substr(0, SepPos);
        SecName = Sec->getName().substr(SepPos + 1);
      } else {
        // Long section name that needs to be truncated.
        assert(Sec->getName().size() > 16 &&
               "Short section name should have been handled above");
        SegName = "__JITLINK_CUSTOM";
        auto IdxStr = std::to_string(++LongSectionNameIdx);
        CustomSecName = Sec->getName().substr(0, 15 - IdxStr.size()).str();
        CustomSecName += ".";
        CustomSecName += IdxStr;
        SecName = StringRef(CustomSecName.data(), 16);
      }

      SectionRange R(*Sec);
      if (R.getFirstBlock()->getAlignmentOffset() != 0)
        return make_error<StringError>(
            "While building MachO debug object for " + G.getName() +
                " first block has non-zero alignment offset",
            inconvertibleErrorCode());

      typename MachOTraits::Section SecCmd;
      memset(&SecCmd, 0, sizeof(SecCmd));
      memcpy(SecCmd.sectname, SecName.data(), SecName.size());
      memcpy(SecCmd.segname, SegName.data(), SegName.size());
      SecCmd.addr = R.getStart().getValue();
      SecCmd.size = R.getSize();
      SecCmd.offset = 0;
      SecCmd.align = R.getFirstBlock()->getAlignment();
      SecCmd.reloff = 0;
      SecCmd.nreloc = 0;
      SecCmd.flags = 0;
      Writer.write(SecCmd);
    }

    SectionRange R(MachOContainerBlock->getSection());
    G.allocActions().push_back(
        {cantFail(shared::WrapperFunctionCall::Create<
                  shared::SPSArgList<shared::SPSExecutorAddrRange>>(
             RegisterActionAddr, R.getRange())),
         {}});
    return Error::success();
  }

private:
  Block *MachOContainerBlock = nullptr;
  SmallVector<Section *, 16> NonDebugSections;
  size_t NonDebugSectionsStart = 0;
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
        LG, RegisterActionAddr);
    PassConfig.PrePrunePasses.push_back(
        [=](LinkGraph &G) { return MDOS->preserveDebugSections(); });
    PassConfig.PostPrunePasses.push_back(
        [=](LinkGraph &G) { return MDOS->startSynthesis(); });
    PassConfig.PreFixupPasses.push_back(
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
