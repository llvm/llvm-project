//===- llvm-jitlink.cpp -- Command line interface/tester for llvm-jitlink -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This utility provides a simple command line interface to the llvm jitlink
// library, which makes relocatable object files executable in memory. Its
// primary function is as a testing utility for the jitlink library.
//
//===----------------------------------------------------------------------===//

#include "llvm-jitlink.h"

#include "llvm/ExecutionEngine/JITLink/EHFrameSupport.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"

#include <list>
#include <string>

#define DEBUG_TYPE "llvm_jitlink"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;

static cl::list<std::string> InputFiles(cl::Positional, cl::OneOrMore,
                                        cl::desc("input files"));

static cl::opt<bool> NoExec("noexec", cl::desc("Do not execute loaded code"),
                            cl::init(false));

static cl::list<std::string>
    CheckFiles("check", cl::desc("File containing verifier checks"),
               cl::ZeroOrMore);

static cl::opt<std::string>
    CheckName("check-name", cl::desc("Name of checks to match against"),
              cl::init("jitlink-check"));

static cl::opt<std::string>
    EntryPointName("entry", cl::desc("Symbol to call as main entry point"),
                   cl::init(""));

static cl::list<std::string> JITLinkDylibs(
    "jld", cl::desc("Specifies the JITDylib to be used for any subsequent "
                    "input file arguments"));

static cl::list<std::string>
    Dylibs("dlopen", cl::desc("Dynamic libraries to load before linking"),
           cl::ZeroOrMore);

static cl::list<std::string> InputArgv("args", cl::Positional,
                                       cl::desc("<program arguments>..."),
                                       cl::ZeroOrMore, cl::PositionalEatsArgs);

static cl::opt<bool>
    NoProcessSymbols("no-process-syms",
                     cl::desc("Do not resolve to llvm-jitlink process symbols"),
                     cl::init(false));

static cl::list<std::string> AbsoluteDefs(
    "define-abs",
    cl::desc("Inject absolute symbol definitions (syntax: <name>=<addr>)"),
    cl::ZeroOrMore);

static cl::opt<bool> ShowInitialExecutionSessionState(
    "show-init-es",
    cl::desc("Print ExecutionSession state before resolving entry point"),
    cl::init(false));

static cl::opt<bool> ShowAddrs(
    "show-addrs",
    cl::desc("Print registered symbol, section, got and stub addresses"),
    cl::init(false));

static cl::opt<bool> ShowLinkGraph(
    "show-graph",
    cl::desc("Print the link graph after fixups have been applied"),
    cl::init(false));

static cl::opt<bool> ShowSizes(
    "show-sizes",
    cl::desc("Show sizes pre- and post-dead stripping, and allocations"),
    cl::init(false));

static cl::opt<bool> ShowTimes("show-times",
                               cl::desc("Show times for llvm-jitlink phases"),
                               cl::init(false));

static cl::opt<std::string> SlabAllocateSizeString(
    "slab-allocate",
    cl::desc("Allocate from a slab of the given size "
             "(allowable suffixes: Kb, Mb, Gb. default = "
             "Kb)"),
    cl::init(""));

static cl::opt<uint64_t> SlabAddress(
    "slab-address",
    cl::desc("Set slab target address (requires -slab-allocate and -noexec)"),
    cl::init(~0ULL));

static cl::opt<bool> ShowRelocatedSectionContents(
    "show-relocated-section-contents",
    cl::desc("show section contents after fixups have been applied"),
    cl::init(false));

ExitOnError ExitOnErr;

namespace llvm {

static raw_ostream &
operator<<(raw_ostream &OS, const Session::MemoryRegionInfo &MRI) {
  return OS << "target addr = "
            << format("0x%016" PRIx64, MRI.getTargetAddress())
            << ", content: " << (const void *)MRI.getContent().data() << " -- "
            << (const void *)(MRI.getContent().data() + MRI.getContent().size())
            << " (" << MRI.getContent().size() << " bytes)";
}

static raw_ostream &
operator<<(raw_ostream &OS, const Session::SymbolInfoMap &SIM) {
  OS << "Symbols:\n";
  for (auto &SKV : SIM)
    OS << "  \"" << SKV.first() << "\" " << SKV.second << "\n";
  return OS;
}

static raw_ostream &
operator<<(raw_ostream &OS, const Session::FileInfo &FI) {
  for (auto &SIKV : FI.SectionInfos)
    OS << "  Section \"" << SIKV.first() << "\": " << SIKV.second << "\n";
  for (auto &GOTKV : FI.GOTEntryInfos)
    OS << "  GOT \"" << GOTKV.first() << "\": " << GOTKV.second << "\n";
  for (auto &StubKV : FI.StubInfos)
    OS << "  Stub \"" << StubKV.first() << "\": " << StubKV.second << "\n";
  return OS;
}

static raw_ostream &
operator<<(raw_ostream &OS, const Session::FileInfoMap &FIM) {
  for (auto &FIKV : FIM)
    OS << "File \"" << FIKV.first() << "\":\n" << FIKV.second;
  return OS;
}

static uint64_t computeTotalBlockSizes(LinkGraph &G) {
  uint64_t TotalSize = 0;
  for (auto *B : G.blocks())
    TotalSize += B->getSize();
  return TotalSize;
}

static void dumpSectionContents(raw_ostream &OS, LinkGraph &G) {
  constexpr JITTargetAddress DumpWidth = 16;
  static_assert(isPowerOf2_64(DumpWidth), "DumpWidth must be a power of two");

  // Put sections in address order.
  std::vector<Section *> Sections;
  for (auto &S : G.sections())
    Sections.push_back(&S);

  std::sort(Sections.begin(), Sections.end(),
            [](const Section *LHS, const Section *RHS) {
              if (llvm::empty(LHS->symbols()) && llvm::empty(RHS->symbols()))
                return false;
              if (llvm::empty(LHS->symbols()))
                return false;
              if (llvm::empty(RHS->symbols()))
                return true;
              SectionRange LHSRange(*LHS);
              SectionRange RHSRange(*RHS);
              return LHSRange.getStart() < RHSRange.getStart();
            });

  for (auto *S : Sections) {
    OS << S->getName() << " content:";
    if (llvm::empty(S->symbols())) {
      OS << "\n  section empty\n";
      continue;
    }

    // Sort symbols into order, then render.
    std::vector<Symbol *> Syms(S->symbols().begin(), S->symbols().end());
    llvm::sort(Syms, [](const Symbol *LHS, const Symbol *RHS) {
      return LHS->getAddress() < RHS->getAddress();
    });

    JITTargetAddress NextAddr = Syms.front()->getAddress() & ~(DumpWidth - 1);
    for (auto *Sym : Syms) {
      bool IsZeroFill = Sym->getBlock().isZeroFill();
      JITTargetAddress SymStart = Sym->getAddress();
      JITTargetAddress SymSize = Sym->getSize();
      JITTargetAddress SymEnd = SymStart + SymSize;
      const uint8_t *SymData =
          IsZeroFill ? nullptr : Sym->getSymbolContent().bytes_begin();

      // Pad any space before the symbol starts.
      while (NextAddr != SymStart) {
        if (NextAddr % DumpWidth == 0)
          OS << formatv("\n{0:x16}:", NextAddr);
        OS << "   ";
        ++NextAddr;
      }

      // Render the symbol content.
      while (NextAddr != SymEnd) {
        if (NextAddr % DumpWidth == 0)
          OS << formatv("\n{0:x16}:", NextAddr);
        if (IsZeroFill)
          OS << " 00";
        else
          OS << formatv(" {0:x-2}", SymData[NextAddr - SymStart]);
        ++NextAddr;
      }
    }
    OS << "\n";
  }
}

class JITLinkSlabAllocator final : public JITLinkMemoryManager {
public:
  static Expected<std::unique_ptr<JITLinkSlabAllocator>>
  Create(uint64_t SlabSize) {
    Error Err = Error::success();
    std::unique_ptr<JITLinkSlabAllocator> Allocator(
        new JITLinkSlabAllocator(SlabSize, Err));
    if (Err)
      return std::move(Err);
    return std::move(Allocator);
  }

  Expected<std::unique_ptr<JITLinkMemoryManager::Allocation>>
  allocate(const SegmentsRequestMap &Request) override {

    using AllocationMap = DenseMap<unsigned, sys::MemoryBlock>;

    // Local class for allocation.
    class IPMMAlloc : public Allocation {
    public:
      IPMMAlloc(JITLinkSlabAllocator &Parent, AllocationMap SegBlocks)
          : Parent(Parent), SegBlocks(std::move(SegBlocks)) {}
      MutableArrayRef<char> getWorkingMemory(ProtectionFlags Seg) override {
        assert(SegBlocks.count(Seg) && "No allocation for segment");
        return {static_cast<char *>(SegBlocks[Seg].base()),
                SegBlocks[Seg].allocatedSize()};
      }
      JITTargetAddress getTargetMemory(ProtectionFlags Seg) override {
        assert(SegBlocks.count(Seg) && "No allocation for segment");
        return pointerToJITTargetAddress(SegBlocks[Seg].base()) +
               Parent.TargetDelta;
      }
      void finalizeAsync(FinalizeContinuation OnFinalize) override {
        OnFinalize(applyProtections());
      }
      Error deallocate() override {
        for (auto &KV : SegBlocks)
          if (auto EC = sys::Memory::releaseMappedMemory(KV.second))
            return errorCodeToError(EC);
        return Error::success();
      }

    private:
      Error applyProtections() {
        for (auto &KV : SegBlocks) {
          auto &Prot = KV.first;
          auto &Block = KV.second;
          if (auto EC = sys::Memory::protectMappedMemory(Block, Prot))
            return errorCodeToError(EC);
          if (Prot & sys::Memory::MF_EXEC)
            sys::Memory::InvalidateInstructionCache(Block.base(),
                                                    Block.allocatedSize());
        }
        return Error::success();
      }

      JITLinkSlabAllocator &Parent;
      AllocationMap SegBlocks;
    };

    AllocationMap Blocks;

    for (auto &KV : Request) {
      auto &Seg = KV.second;

      if (Seg.getAlignment() > PageSize)
        return make_error<StringError>("Cannot request higher than page "
                                       "alignment",
                                       inconvertibleErrorCode());

      if (PageSize % Seg.getAlignment() != 0)
        return make_error<StringError>("Page size is not a multiple of "
                                       "alignment",
                                       inconvertibleErrorCode());

      uint64_t ZeroFillStart = Seg.getContentSize();
      uint64_t SegmentSize = ZeroFillStart + Seg.getZeroFillSize();

      // Round segment size up to page boundary.
      SegmentSize = (SegmentSize + PageSize - 1) & ~(PageSize - 1);

      // Take segment bytes from the front of the slab.
      void *SlabBase = SlabRemaining.base();
      uint64_t SlabRemainingSize = SlabRemaining.allocatedSize();

      if (SegmentSize > SlabRemainingSize)
        return make_error<StringError>("Slab allocator out of memory",
                                       inconvertibleErrorCode());

      sys::MemoryBlock SegMem(SlabBase, SegmentSize);
      SlabRemaining =
          sys::MemoryBlock(reinterpret_cast<char *>(SlabBase) + SegmentSize,
                           SlabRemainingSize - SegmentSize);

      // Zero out the zero-fill memory.
      memset(static_cast<char *>(SegMem.base()) + ZeroFillStart, 0,
             Seg.getZeroFillSize());

      // Record the block for this segment.
      Blocks[KV.first] = std::move(SegMem);
    }
    return std::unique_ptr<InProcessMemoryManager::Allocation>(
        new IPMMAlloc(*this, std::move(Blocks)));
  }

private:
  JITLinkSlabAllocator(uint64_t SlabSize, Error &Err) {
    ErrorAsOutParameter _(&Err);

    PageSize = sys::Process::getPageSizeEstimate();

    if (!isPowerOf2_64(PageSize)) {
      Err = make_error<StringError>("Page size is not a power of 2",
                                    inconvertibleErrorCode());
      return;
    }

    // Round slab request up to page size.
    SlabSize = (SlabSize + PageSize - 1) & ~(PageSize - 1);

    const sys::Memory::ProtectionFlags ReadWrite =
        static_cast<sys::Memory::ProtectionFlags>(sys::Memory::MF_READ |
                                                  sys::Memory::MF_WRITE);

    std::error_code EC;
    SlabRemaining =
        sys::Memory::allocateMappedMemory(SlabSize, nullptr, ReadWrite, EC);

    if (EC) {
      Err = errorCodeToError(EC);
      return;
    }

    // Calculate the target address delta to link as-if slab were at
    // SlabAddress.
    if (SlabAddress != ~0ULL)
      TargetDelta =
          SlabAddress - pointerToJITTargetAddress(SlabRemaining.base());
  }

  sys::MemoryBlock SlabRemaining;
  uint64_t PageSize = 0;
  int64_t TargetDelta = 0;
};

Expected<uint64_t> getSlabAllocSize(StringRef SizeString) {
  SizeString = SizeString.trim();

  uint64_t Units = 1024;

  if (SizeString.endswith_lower("kb"))
    SizeString = SizeString.drop_back(2).rtrim();
  else if (SizeString.endswith_lower("mb")) {
    Units = 1024 * 1024;
    SizeString = SizeString.drop_back(2).rtrim();
  } else if (SizeString.endswith_lower("gb")) {
    Units = 1024 * 1024 * 1024;
    SizeString = SizeString.drop_back(2).rtrim();
  }

  uint64_t SlabSize = 0;
  if (SizeString.getAsInteger(10, SlabSize))
    return make_error<StringError>("Invalid numeric format for slab size",
                                   inconvertibleErrorCode());

  return SlabSize * Units;
}

static std::unique_ptr<jitlink::JITLinkMemoryManager> createMemoryManager() {
  if (!SlabAllocateSizeString.empty()) {
    auto SlabSize = ExitOnErr(getSlabAllocSize(SlabAllocateSizeString));
    return ExitOnErr(JITLinkSlabAllocator::Create(SlabSize));
  }
  return std::make_unique<jitlink::InProcessMemoryManager>();
}

Expected<std::unique_ptr<Session>> Session::Create(Triple TT) {
  Error Err = Error::success();
  std::unique_ptr<Session> S(new Session(std::move(TT), Err));
  if (Err)
    return std::move(Err);
  return std::move(S);
}

// FIXME: Move to createJITDylib if/when we start using Platform support in
// llvm-jitlink.
Session::Session(Triple TT, Error &Err)
    : ObjLayer(ES, createMemoryManager()), TT(std::move(TT)) {

  /// Local ObjectLinkingLayer::Plugin class to forward modifyPassConfig to the
  /// Session.
  class JITLinkSessionPlugin : public ObjectLinkingLayer::Plugin {
  public:
    JITLinkSessionPlugin(Session &S) : S(S) {}
    void modifyPassConfig(MaterializationResponsibility &MR, const Triple &TT,
                          PassConfiguration &PassConfig) {
      S.modifyPassConfig(TT, PassConfig);
    }

  private:
    Session &S;
  };

  ErrorAsOutParameter _(&Err);

  if (auto MainJDOrErr = ES.createJITDylib("main"))
    MainJD = &*MainJDOrErr;
  else {
    Err = MainJDOrErr.takeError();
    return;
  }

  if (!NoExec && !TT.isOSWindows())
    ObjLayer.addPlugin(std::make_unique<EHFrameRegistrationPlugin>(
        InProcessEHFrameRegistrar::getInstance()));

  ObjLayer.addPlugin(std::make_unique<JITLinkSessionPlugin>(*this));
}

void Session::dumpSessionInfo(raw_ostream &OS) {
  OS << "Registered addresses:\n" << SymbolInfos << FileInfos;
}

void Session::modifyPassConfig(const Triple &FTT,
                               PassConfiguration &PassConfig) {
  if (!CheckFiles.empty())
    PassConfig.PostFixupPasses.push_back([this](LinkGraph &G) {
      if (TT.getObjectFormat() == Triple::MachO)
        return registerMachOStubsAndGOT(*this, G);
      return make_error<StringError>("Unsupported object format for GOT/stub "
                                     "registration",
                                     inconvertibleErrorCode());
    });

  if (ShowLinkGraph)
    PassConfig.PostFixupPasses.push_back([](LinkGraph &G) -> Error {
      outs() << "Link graph post-fixup:\n";
      G.dump(outs());
      return Error::success();
    });

  if (ShowSizes) {
    PassConfig.PrePrunePasses.push_back([this](LinkGraph &G) -> Error {
      SizeBeforePruning += computeTotalBlockSizes(G);
      return Error::success();
    });
    PassConfig.PostFixupPasses.push_back([this](LinkGraph &G) -> Error {
      SizeAfterFixups += computeTotalBlockSizes(G);
      return Error::success();
    });
  }

  if (ShowRelocatedSectionContents)
    PassConfig.PostFixupPasses.push_back([](LinkGraph &G) -> Error {
      outs() << "Relocated section contents for " << G.getName() << ":\n";
      dumpSectionContents(outs(), G);
      return Error::success();
    });
}

Expected<Session::FileInfo &> Session::findFileInfo(StringRef FileName) {
  auto FileInfoItr = FileInfos.find(FileName);
  if (FileInfoItr == FileInfos.end())
    return make_error<StringError>("file \"" + FileName + "\" not recognized",
                                   inconvertibleErrorCode());
  return FileInfoItr->second;
}

Expected<Session::MemoryRegionInfo &>
Session::findSectionInfo(StringRef FileName, StringRef SectionName) {
  auto FI = findFileInfo(FileName);
  if (!FI)
    return FI.takeError();
  auto SecInfoItr = FI->SectionInfos.find(SectionName);
  if (SecInfoItr == FI->SectionInfos.end())
    return make_error<StringError>("no section \"" + SectionName +
                                       "\" registered for file \"" + FileName +
                                       "\"",
                                   inconvertibleErrorCode());
  return SecInfoItr->second;
}

Expected<Session::MemoryRegionInfo &>
Session::findStubInfo(StringRef FileName, StringRef TargetName) {
  auto FI = findFileInfo(FileName);
  if (!FI)
    return FI.takeError();
  auto StubInfoItr = FI->StubInfos.find(TargetName);
  if (StubInfoItr == FI->StubInfos.end())
    return make_error<StringError>("no stub for \"" + TargetName +
                                       "\" registered for file \"" + FileName +
                                       "\"",
                                   inconvertibleErrorCode());
  return StubInfoItr->second;
}

Expected<Session::MemoryRegionInfo &>
Session::findGOTEntryInfo(StringRef FileName, StringRef TargetName) {
  auto FI = findFileInfo(FileName);
  if (!FI)
    return FI.takeError();
  auto GOTInfoItr = FI->GOTEntryInfos.find(TargetName);
  if (GOTInfoItr == FI->GOTEntryInfos.end())
    return make_error<StringError>("no GOT entry for \"" + TargetName +
                                       "\" registered for file \"" + FileName +
                                       "\"",
                                   inconvertibleErrorCode());
  return GOTInfoItr->second;
}

bool Session::isSymbolRegistered(StringRef SymbolName) {
  return SymbolInfos.count(SymbolName);
}

Expected<Session::MemoryRegionInfo &>
Session::findSymbolInfo(StringRef SymbolName, Twine ErrorMsgStem) {
  auto SymInfoItr = SymbolInfos.find(SymbolName);
  if (SymInfoItr == SymbolInfos.end())
    return make_error<StringError>(ErrorMsgStem + ": symbol " + SymbolName +
                                       " not found",
                                   inconvertibleErrorCode());
  return SymInfoItr->second;
}

} // end namespace llvm

Triple getFirstFileTriple() {
  assert(!InputFiles.empty() && "InputFiles can not be empty");
  auto ObjBuffer =
      ExitOnErr(errorOrToExpected(MemoryBuffer::getFile(InputFiles.front())));
  auto Obj = ExitOnErr(
      object::ObjectFile::createObjectFile(ObjBuffer->getMemBufferRef()));
  return Obj->makeTriple();
}

Error sanitizeArguments(const Session &S) {
  if (EntryPointName.empty()) {
    if (S.TT.getObjectFormat() == Triple::MachO)
      EntryPointName = "_main";
    else
      EntryPointName = "main";
  }

  if (NoExec && !InputArgv.empty())
    outs() << "Warning: --args passed to -noexec run will be ignored.\n";

  // If -slab-address is passed, require -slab-allocate and -noexec
  if (SlabAddress != ~0ULL) {
    if (SlabAllocateSizeString == "" || !NoExec)
      return make_error<StringError>(
          "-slab-address requires -slab-allocate and -noexec",
          inconvertibleErrorCode());
  }

  return Error::success();
}

Error loadProcessSymbols(Session &S) {
  std::string ErrMsg;
  if (sys::DynamicLibrary::LoadLibraryPermanently(nullptr, &ErrMsg))
    return make_error<StringError>(std::move(ErrMsg), inconvertibleErrorCode());

  char GlobalPrefix = S.TT.getObjectFormat() == Triple::MachO ? '_' : '\0';
  auto InternedEntryPointName = S.ES.intern(EntryPointName);
  auto FilterMainEntryPoint = [InternedEntryPointName](SymbolStringPtr Name) {
    return Name != InternedEntryPointName;
  };
  S.MainJD->addGenerator(
      ExitOnErr(orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          GlobalPrefix, FilterMainEntryPoint)));

  return Error::success();
}

Error loadDylibs() {
  // FIXME: This should all be handled inside DynamicLibrary.
  for (const auto &Dylib : Dylibs) {
    if (!sys::fs::is_regular_file(Dylib))
      return make_error<StringError>("\"" + Dylib + "\" is not a regular file",
                                     inconvertibleErrorCode());
    std::string ErrMsg;
    if (sys::DynamicLibrary::LoadLibraryPermanently(Dylib.c_str(), &ErrMsg))
      return make_error<StringError>(ErrMsg, inconvertibleErrorCode());
  }

  return Error::success();
}

Error loadObjects(Session &S) {

  std::map<unsigned, JITDylib *> IdxToJLD;

  // First, set up JITDylibs.
  LLVM_DEBUG(dbgs() << "Creating JITDylibs...\n");
  {
    // Create a "main" JITLinkDylib.
    IdxToJLD[0] = S.MainJD;
    S.JDSearchOrder.push_back(S.MainJD);
    LLVM_DEBUG(dbgs() << "  0: " << S.MainJD->getName() << "\n");

    // Add any extra JITLinkDylibs from the command line.
    std::string JDNamePrefix("lib");
    for (auto JLDItr = JITLinkDylibs.begin(), JLDEnd = JITLinkDylibs.end();
         JLDItr != JLDEnd; ++JLDItr) {
      auto JD = S.ES.createJITDylib(JDNamePrefix + *JLDItr);
      if (!JD)
        return JD.takeError();
      unsigned JDIdx =
          JITLinkDylibs.getPosition(JLDItr - JITLinkDylibs.begin());
      IdxToJLD[JDIdx] = &*JD;
      S.JDSearchOrder.push_back(&*JD);
      LLVM_DEBUG(dbgs() << "  " << JDIdx << ": " << JD->getName() << "\n");
    }

    // Set every dylib to link against every other, in command line order.
    for (auto *JD : S.JDSearchOrder) {
      auto LookupFlags = JITDylibLookupFlags::MatchExportedSymbolsOnly;
      JITDylibSearchOrder O;
      for (auto *JD2 : S.JDSearchOrder) {
        if (JD2 == JD)
          continue;
        O.push_back(std::make_pair(JD2, LookupFlags));
      }
      JD->setSearchOrder(std::move(O));
    }
  }

  // Load each object into the corresponding JITDylib..
  LLVM_DEBUG(dbgs() << "Adding objects...\n");
  for (auto InputFileItr = InputFiles.begin(), InputFileEnd = InputFiles.end();
       InputFileItr != InputFileEnd; ++InputFileItr) {
    unsigned InputFileArgIdx =
        InputFiles.getPosition(InputFileItr - InputFiles.begin());
    StringRef InputFile = *InputFileItr;
    auto &JD = *std::prev(IdxToJLD.lower_bound(InputFileArgIdx))->second;
    LLVM_DEBUG(dbgs() << "  " << InputFileArgIdx << ": \"" << InputFile
                      << "\" to " << JD.getName() << "\n";);
    auto ObjBuffer =
        ExitOnErr(errorOrToExpected(MemoryBuffer::getFile(InputFile)));
    ExitOnErr(S.ObjLayer.add(JD, std::move(ObjBuffer)));
  }

  // Define absolute symbols.
  LLVM_DEBUG(dbgs() << "Defining absolute symbols...\n");
  for (auto AbsDefItr = AbsoluteDefs.begin(), AbsDefEnd = AbsoluteDefs.end();
       AbsDefItr != AbsDefEnd; ++AbsDefItr) {
    unsigned AbsDefArgIdx =
      AbsoluteDefs.getPosition(AbsDefItr - AbsoluteDefs.begin());
    auto &JD = *std::prev(IdxToJLD.lower_bound(AbsDefArgIdx))->second;

    StringRef AbsDefStmt = *AbsDefItr;
    size_t EqIdx = AbsDefStmt.find_first_of('=');
    if (EqIdx == StringRef::npos)
      return make_error<StringError>("Invalid absolute define \"" + AbsDefStmt +
                                     "\". Syntax: <name>=<addr>",
                                     inconvertibleErrorCode());
    StringRef Name = AbsDefStmt.substr(0, EqIdx).trim();
    StringRef AddrStr = AbsDefStmt.substr(EqIdx + 1).trim();

    uint64_t Addr;
    if (AddrStr.getAsInteger(0, Addr))
      return make_error<StringError>("Invalid address expression \"" + AddrStr +
                                     "\" in absolute define \"" + AbsDefStmt +
                                     "\"",
                                     inconvertibleErrorCode());
    JITEvaluatedSymbol AbsDef(Addr, JITSymbolFlags::Exported);
    if (auto Err = JD.define(absoluteSymbols({{S.ES.intern(Name), AbsDef}})))
      return Err;

    // Register the absolute symbol with the session symbol infos.
    S.SymbolInfos[Name] = { StringRef(), Addr };
  }

  LLVM_DEBUG({
    dbgs() << "Dylib search order is [ ";
    for (auto *JD : S.JDSearchOrder)
      dbgs() << JD->getName() << " ";
    dbgs() << "]\n";
  });

  return Error::success();
}

Error runChecks(Session &S) {

  auto TripleName = S.TT.str();
  std::string ErrorStr;
  const Target *TheTarget = TargetRegistry::lookupTarget("", S.TT, ErrorStr);
  if (!TheTarget)
    ExitOnErr(make_error<StringError>("Error accessing target '" + TripleName +
                                          "': " + ErrorStr,
                                      inconvertibleErrorCode()));

  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  if (!STI)
    ExitOnErr(
        make_error<StringError>("Unable to create subtarget for " + TripleName,
                                inconvertibleErrorCode()));

  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  if (!MRI)
    ExitOnErr(make_error<StringError>("Unable to create target register info "
                                      "for " +
                                          TripleName,
                                      inconvertibleErrorCode()));

  MCTargetOptions MCOptions;
  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  if (!MAI)
    ExitOnErr(make_error<StringError>("Unable to create target asm info " +
                                          TripleName,
                                      inconvertibleErrorCode()));

  MCContext Ctx(MAI.get(), MRI.get(), nullptr);

  std::unique_ptr<MCDisassembler> Disassembler(
      TheTarget->createMCDisassembler(*STI, Ctx));
  if (!Disassembler)
    ExitOnErr(make_error<StringError>("Unable to create disassembler for " +
                                          TripleName,
                                      inconvertibleErrorCode()));

  std::unique_ptr<MCInstrInfo> MII(TheTarget->createMCInstrInfo());

  std::unique_ptr<MCInstPrinter> InstPrinter(
      TheTarget->createMCInstPrinter(Triple(TripleName), 0, *MAI, *MII, *MRI));

  auto IsSymbolValid = [&S](StringRef Symbol) {
    return S.isSymbolRegistered(Symbol);
  };

  auto GetSymbolInfo = [&S](StringRef Symbol) {
    return S.findSymbolInfo(Symbol, "Can not get symbol info");
  };

  auto GetSectionInfo = [&S](StringRef FileName, StringRef SectionName) {
    return S.findSectionInfo(FileName, SectionName);
  };

  auto GetStubInfo = [&S](StringRef FileName, StringRef SectionName) {
    return S.findStubInfo(FileName, SectionName);
  };

  auto GetGOTInfo = [&S](StringRef FileName, StringRef SectionName) {
    return S.findGOTEntryInfo(FileName, SectionName);
  };

  RuntimeDyldChecker Checker(
      IsSymbolValid, GetSymbolInfo, GetSectionInfo, GetStubInfo, GetGOTInfo,
      S.TT.isLittleEndian() ? support::little : support::big,
      Disassembler.get(), InstPrinter.get(), dbgs());

  std::string CheckLineStart = "# " + CheckName + ":";
  for (auto &CheckFile : CheckFiles) {
    auto CheckerFileBuf =
        ExitOnErr(errorOrToExpected(MemoryBuffer::getFile(CheckFile)));
    if (!Checker.checkAllRulesInBuffer(CheckLineStart, &*CheckerFileBuf))
      ExitOnErr(make_error<StringError>(
          "Some checks in " + CheckFile + " failed", inconvertibleErrorCode()));
  }

  return Error::success();
}

static void dumpSessionStats(Session &S) {
  if (ShowSizes)
    outs() << "Total size of all blocks before pruning: " << S.SizeBeforePruning
           << "\nTotal size of all blocks after fixups: " << S.SizeAfterFixups
           << "\n";
}

static Expected<JITEvaluatedSymbol> getMainEntryPoint(Session &S) {
  return S.ES.lookup(S.JDSearchOrder, EntryPointName);
}

struct JITLinkTimers {
  TimerGroup JITLinkTG{"llvm-jitlink timers", "timers for llvm-jitlink phases"};
  Timer LoadObjectsTimer{"load", "time to load/add object files", JITLinkTG};
  Timer LinkTimer{"link", "time to link object files", JITLinkTG};
  Timer RunTimer{"run", "time to execute jitlink'd code", JITLinkTG};
};

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);

  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  cl::ParseCommandLineOptions(argc, argv, "llvm jitlink tool");
  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  /// If timers are enabled, create a JITLinkTimers instance.
  std::unique_ptr<JITLinkTimers> Timers =
      ShowTimes ? std::make_unique<JITLinkTimers>() : nullptr;

  auto S = ExitOnErr(Session::Create(getFirstFileTriple()));

  ExitOnErr(sanitizeArguments(*S));

  if (!NoProcessSymbols)
    ExitOnErr(loadProcessSymbols(*S));
  ExitOnErr(loadDylibs());

  {
    TimeRegion TR(Timers ? &Timers->LoadObjectsTimer : nullptr);
    ExitOnErr(loadObjects(*S));
  }

  if (ShowInitialExecutionSessionState)
    S->ES.dump(outs());

  JITEvaluatedSymbol EntryPoint = 0;
  {
    TimeRegion TR(Timers ? &Timers->LinkTimer : nullptr);
    EntryPoint = ExitOnErr(getMainEntryPoint(*S));
  }

  if (ShowAddrs)
    S->dumpSessionInfo(outs());

  ExitOnErr(runChecks(*S));

  dumpSessionStats(*S);

  if (NoExec)
    return 0;

  int Result = 0;
  {
    using MainTy = int (*)(int, char *[]);
    auto EntryFn = jitTargetAddressToFunction<MainTy>(EntryPoint.getAddress());
    TimeRegion TR(Timers ? &Timers->RunTimer : nullptr);
    Result = runAsMain(EntryFn, InputArgv, StringRef(InputFiles.front()));
  }

  return Result;
}
