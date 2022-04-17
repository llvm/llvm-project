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

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/ExecutionEngine/Orc/DebugObjectManagerPlugin.h"
#include "llvm/ExecutionEngine/Orc/DebuggerSupportPlugin.h"
#include "llvm/ExecutionEngine/Orc/ELFNixPlatform.h"
#include "llvm/ExecutionEngine/Orc/EPCDebugObjectRegistrar.h"
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/ExecutionEngine/Orc/EPCEHFrameRegistrar.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/ExecutionEngine/Orc/MachOPlatform.h"
#include "llvm/ExecutionEngine/Orc/ObjectFileInterface.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/RegisterEHFrames.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"

#include <cstring>
#include <list>
#include <string>

#ifdef LLVM_ON_UNIX
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif // LLVM_ON_UNIX

#define DEBUG_TYPE "llvm_jitlink"

using namespace llvm;
using namespace llvm::jitlink;
using namespace llvm::orc;

static cl::OptionCategory JITLinkCategory("JITLink Options");

static cl::list<std::string> InputFiles(cl::Positional, cl::OneOrMore,
                                        cl::desc("input files"),
                                        cl::cat(JITLinkCategory));

static cl::list<std::string>
    LibrarySearchPaths("L",
                       cl::desc("Add dir to the list of library search paths"),
                       cl::Prefix, cl::cat(JITLinkCategory));

static cl::list<std::string>
    Libraries("l",
              cl::desc("Link against library X in the library search paths"),
              cl::Prefix, cl::cat(JITLinkCategory));

static cl::list<std::string>
    LibrariesHidden("hidden-l",
                    cl::desc("Link against library X in the library search "
                             "paths with hidden visibility"),
                    cl::Prefix, cl::cat(JITLinkCategory));

static cl::list<std::string>
    LoadHidden("load_hidden",
               cl::desc("Link against library X with hidden visibility"),
               cl::cat(JITLinkCategory));

static cl::opt<bool> NoExec("noexec", cl::desc("Do not execute loaded code"),
                            cl::init(false), cl::cat(JITLinkCategory));

static cl::list<std::string>
    CheckFiles("check", cl::desc("File containing verifier checks"),
               cl::ZeroOrMore, cl::cat(JITLinkCategory));

static cl::opt<std::string>
    CheckName("check-name", cl::desc("Name of checks to match against"),
              cl::init("jitlink-check"), cl::cat(JITLinkCategory));

static cl::opt<std::string>
    EntryPointName("entry", cl::desc("Symbol to call as main entry point"),
                   cl::init(""), cl::cat(JITLinkCategory));

static cl::list<std::string> JITDylibs(
    "jd",
    cl::desc("Specifies the JITDylib to be used for any subsequent "
             "input file, -L<seacrh-path>, and -l<library> arguments"),
    cl::cat(JITLinkCategory));

static cl::list<std::string>
    Dylibs("preload",
           cl::desc("Pre-load dynamic libraries (e.g. language runtimes "
                    "required by the ORC runtime)"),
           cl::ZeroOrMore, cl::cat(JITLinkCategory));

static cl::list<std::string> InputArgv("args", cl::Positional,
                                       cl::desc("<program arguments>..."),
                                       cl::ZeroOrMore, cl::PositionalEatsArgs,
                                       cl::cat(JITLinkCategory));

static cl::opt<bool>
    DebuggerSupport("debugger-support",
                    cl::desc("Enable debugger suppport (default = !-noexec)"),
                    cl::init(true), cl::Hidden, cl::cat(JITLinkCategory));

static cl::opt<bool>
    NoProcessSymbols("no-process-syms",
                     cl::desc("Do not resolve to llvm-jitlink process symbols"),
                     cl::init(false), cl::cat(JITLinkCategory));

static cl::list<std::string> AbsoluteDefs(
    "abs",
    cl::desc("Inject absolute symbol definitions (syntax: <name>=<addr>)"),
    cl::ZeroOrMore, cl::cat(JITLinkCategory));

static cl::list<std::string>
    Aliases("alias", cl::desc("Inject symbol aliases (syntax: <name>=<addr>)"),
            cl::ZeroOrMore, cl::cat(JITLinkCategory));

static cl::list<std::string> TestHarnesses("harness", cl::Positional,
                                           cl::desc("Test harness files"),
                                           cl::ZeroOrMore,
                                           cl::PositionalEatsArgs,
                                           cl::cat(JITLinkCategory));

static cl::opt<bool> ShowInitialExecutionSessionState(
    "show-init-es",
    cl::desc("Print ExecutionSession state before resolving entry point"),
    cl::init(false), cl::cat(JITLinkCategory));

static cl::opt<bool> ShowEntryExecutionSessionState(
    "show-entry-es",
    cl::desc("Print ExecutionSession state after resolving entry point"),
    cl::init(false), cl::cat(JITLinkCategory));

static cl::opt<bool> ShowAddrs(
    "show-addrs",
    cl::desc("Print registered symbol, section, got and stub addresses"),
    cl::init(false), cl::cat(JITLinkCategory));

static cl::opt<bool> ShowLinkGraph(
    "show-graph",
    cl::desc("Print the link graph after fixups have been applied"),
    cl::init(false), cl::cat(JITLinkCategory));

static cl::opt<bool> ShowSizes(
    "show-sizes",
    cl::desc("Show sizes pre- and post-dead stripping, and allocations"),
    cl::init(false), cl::cat(JITLinkCategory));

static cl::opt<bool> ShowTimes("show-times",
                               cl::desc("Show times for llvm-jitlink phases"),
                               cl::init(false), cl::cat(JITLinkCategory));

static cl::opt<std::string> SlabAllocateSizeString(
    "slab-allocate",
    cl::desc("Allocate from a slab of the given size "
             "(allowable suffixes: Kb, Mb, Gb. default = "
             "Kb)"),
    cl::init(""), cl::cat(JITLinkCategory));

static cl::opt<uint64_t> SlabAddress(
    "slab-address",
    cl::desc("Set slab target address (requires -slab-allocate and -noexec)"),
    cl::init(~0ULL), cl::cat(JITLinkCategory));

static cl::opt<uint64_t> SlabPageSize(
    "slab-page-size",
    cl::desc("Set page size for slab (requires -slab-allocate and -noexec)"),
    cl::init(0), cl::cat(JITLinkCategory));

static cl::opt<bool> ShowRelocatedSectionContents(
    "show-relocated-section-contents",
    cl::desc("show section contents after fixups have been applied"),
    cl::init(false), cl::cat(JITLinkCategory));

static cl::opt<bool> PhonyExternals(
    "phony-externals",
    cl::desc("resolve all otherwise unresolved externals to null"),
    cl::init(false), cl::cat(JITLinkCategory));

static cl::opt<std::string> OutOfProcessExecutor(
    "oop-executor", cl::desc("Launch an out-of-process executor to run code"),
    cl::ValueOptional, cl::cat(JITLinkCategory));

static cl::opt<std::string> OutOfProcessExecutorConnect(
    "oop-executor-connect",
    cl::desc("Connect to an out-of-process executor via TCP"),
    cl::cat(JITLinkCategory));

static cl::opt<std::string>
    OrcRuntime("orc-runtime", cl::desc("Use ORC runtime from given path"),
               cl::init(""), cl::cat(JITLinkCategory));

static cl::opt<bool> AddSelfRelocations(
    "add-self-relocations",
    cl::desc("Add relocations to function pointers to the current function"),
    cl::init(false), cl::cat(JITLinkCategory));

static cl::opt<bool>
    ShowErrFailedToMaterialize("show-err-failed-to-materialize",
                               cl::desc("Show FailedToMaterialize errors"),
                               cl::init(false), cl::cat(JITLinkCategory));

static ExitOnError ExitOnErr;

static LLVM_ATTRIBUTE_USED void linkComponents() {
  errs() << (void *)&llvm_orc_registerEHFrameSectionWrapper
         << (void *)&llvm_orc_deregisterEHFrameSectionWrapper
         << (void *)&llvm_orc_registerJITLoaderGDBWrapper;
}

static bool UseTestResultOverride = false;
static int64_t TestResultOverride = 0;

extern "C" LLVM_ATTRIBUTE_USED void
llvm_jitlink_setTestResultOverride(int64_t Value) {
  TestResultOverride = Value;
  UseTestResultOverride = true;
}

static Error addSelfRelocations(LinkGraph &G);

namespace {

template <typename ErrT>

class ConditionalPrintErr {
public:
  ConditionalPrintErr(bool C) : C(C) {}
  void operator()(ErrT &EI) {
    if (C) {
      errs() << "llvm-jitlink error: ";
      EI.log(errs());
      errs() << "\n";
    }
  }

private:
  bool C;
};

Expected<std::unique_ptr<MemoryBuffer>> getFile(const Twine &FileName) {
  if (auto F = MemoryBuffer::getFile(FileName))
    return std::move(*F);
  else
    return createFileError(FileName, F.getError());
}

void reportLLVMJITLinkError(Error Err) {
  handleAllErrors(
      std::move(Err),
      ConditionalPrintErr<orc::FailedToMaterialize>(ShowErrFailedToMaterialize),
      ConditionalPrintErr<ErrorInfoBase>(true));
}

} // end anonymous namespace

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

static Error applyHarnessPromotions(Session &S, LinkGraph &G) {

  // If this graph is part of the test harness there's nothing to do.
  if (S.HarnessFiles.empty() || S.HarnessFiles.count(G.getName()))
    return Error::success();

  LLVM_DEBUG(dbgs() << "Applying promotions to graph " << G.getName() << "\n");

  // If this graph is part of the test then promote any symbols referenced by
  // the harness to default scope, remove all symbols that clash with harness
  // definitions.
  std::vector<Symbol *> DefinitionsToRemove;
  for (auto *Sym : G.defined_symbols()) {

    if (!Sym->hasName())
      continue;

    if (Sym->getLinkage() == Linkage::Weak) {
      if (!S.CanonicalWeakDefs.count(Sym->getName()) ||
          S.CanonicalWeakDefs[Sym->getName()] != G.getName()) {
        LLVM_DEBUG({
          dbgs() << "  Externalizing weak symbol " << Sym->getName() << "\n";
        });
        DefinitionsToRemove.push_back(Sym);
      } else {
        LLVM_DEBUG({
          dbgs() << "  Making weak symbol " << Sym->getName() << " strong\n";
        });
        if (S.HarnessExternals.count(Sym->getName()))
          Sym->setScope(Scope::Default);
        else
          Sym->setScope(Scope::Hidden);
        Sym->setLinkage(Linkage::Strong);
      }
    } else if (S.HarnessExternals.count(Sym->getName())) {
      LLVM_DEBUG(dbgs() << "  Promoting " << Sym->getName() << "\n");
      Sym->setScope(Scope::Default);
      Sym->setLive(true);
      continue;
    } else if (S.HarnessDefinitions.count(Sym->getName())) {
      LLVM_DEBUG(dbgs() << "  Externalizing " << Sym->getName() << "\n");
      DefinitionsToRemove.push_back(Sym);
    }
  }

  for (auto *Sym : DefinitionsToRemove)
    G.makeExternal(*Sym);

  return Error::success();
}

static uint64_t computeTotalBlockSizes(LinkGraph &G) {
  uint64_t TotalSize = 0;
  for (auto *B : G.blocks())
    TotalSize += B->getSize();
  return TotalSize;
}

static void dumpSectionContents(raw_ostream &OS, LinkGraph &G) {
  constexpr orc::ExecutorAddrDiff DumpWidth = 16;
  static_assert(isPowerOf2_64(DumpWidth), "DumpWidth must be a power of two");

  // Put sections in address order.
  std::vector<Section *> Sections;
  for (auto &S : G.sections())
    Sections.push_back(&S);

  llvm::sort(Sections, [](const Section *LHS, const Section *RHS) {
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

    orc::ExecutorAddr NextAddr(Syms.front()->getAddress().getValue() &
                               ~(DumpWidth - 1));
    for (auto *Sym : Syms) {
      bool IsZeroFill = Sym->getBlock().isZeroFill();
      auto SymStart = Sym->getAddress();
      auto SymSize = Sym->getSize();
      auto SymEnd = SymStart + SymSize;
      const uint8_t *SymData = IsZeroFill ? nullptr
                                          : reinterpret_cast<const uint8_t *>(
                                                Sym->getSymbolContent().data());

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
private:
  struct FinalizedAllocInfo {
    FinalizedAllocInfo(sys::MemoryBlock Mem,
                       std::vector<shared::WrapperFunctionCall> DeallocActions)
        : Mem(Mem), DeallocActions(std::move(DeallocActions)) {}
    sys::MemoryBlock Mem;
    std::vector<shared::WrapperFunctionCall> DeallocActions;
  };

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

  void allocate(const JITLinkDylib *JD, LinkGraph &G,
                OnAllocatedFunction OnAllocated) override {

    // Local class for allocation.
    class IPMMAlloc : public InFlightAlloc {
    public:
      IPMMAlloc(JITLinkSlabAllocator &Parent, BasicLayout BL,
                sys::MemoryBlock StandardSegs, sys::MemoryBlock FinalizeSegs)
          : Parent(Parent), BL(std::move(BL)),
            StandardSegs(std::move(StandardSegs)),
            FinalizeSegs(std::move(FinalizeSegs)) {}

      void finalize(OnFinalizedFunction OnFinalized) override {
        if (auto Err = applyProtections()) {
          OnFinalized(std::move(Err));
          return;
        }

        auto DeallocActions = runFinalizeActions(BL.graphAllocActions());
        if (!DeallocActions) {
          OnFinalized(DeallocActions.takeError());
          return;
        }

        if (auto Err = Parent.freeBlock(FinalizeSegs)) {
          OnFinalized(
              joinErrors(std::move(Err), runDeallocActions(*DeallocActions)));
          return;
        }

        OnFinalized(FinalizedAlloc(ExecutorAddr::fromPtr(
            new FinalizedAllocInfo(StandardSegs, std::move(*DeallocActions)))));
      }

      void abandon(OnAbandonedFunction OnAbandoned) override {
        OnAbandoned(joinErrors(Parent.freeBlock(StandardSegs),
                               Parent.freeBlock(FinalizeSegs)));
      }

    private:
      Error applyProtections() {
        for (auto &KV : BL.segments()) {
          const auto &Group = KV.first;
          auto &Seg = KV.second;

          auto Prot = toSysMemoryProtectionFlags(Group.getMemProt());

          uint64_t SegSize =
              alignTo(Seg.ContentSize + Seg.ZeroFillSize, Parent.PageSize);
          sys::MemoryBlock MB(Seg.WorkingMem, SegSize);
          if (auto EC = sys::Memory::protectMappedMemory(MB, Prot))
            return errorCodeToError(EC);
          if (Prot & sys::Memory::MF_EXEC)
            sys::Memory::InvalidateInstructionCache(MB.base(),
                                                    MB.allocatedSize());
        }
        return Error::success();
      }

      JITLinkSlabAllocator &Parent;
      BasicLayout BL;
      sys::MemoryBlock StandardSegs;
      sys::MemoryBlock FinalizeSegs;
    };

    BasicLayout BL(G);
    auto SegsSizes = BL.getContiguousPageBasedLayoutSizes(PageSize);

    if (!SegsSizes) {
      OnAllocated(SegsSizes.takeError());
      return;
    }

    char *AllocBase = nullptr;
    {
      std::lock_guard<std::mutex> Lock(SlabMutex);

      if (SegsSizes->total() > SlabRemaining.allocatedSize()) {
        OnAllocated(make_error<StringError>(
            "Slab allocator out of memory: request for " +
                formatv("{0:x}", SegsSizes->total()) +
                " bytes exceeds remaining capacity of " +
                formatv("{0:x}", SlabRemaining.allocatedSize()) + " bytes",
            inconvertibleErrorCode()));
        return;
      }

      AllocBase = reinterpret_cast<char *>(SlabRemaining.base());
      SlabRemaining =
          sys::MemoryBlock(AllocBase + SegsSizes->total(),
                           SlabRemaining.allocatedSize() - SegsSizes->total());
    }

    sys::MemoryBlock StandardSegs(AllocBase, SegsSizes->StandardSegs);
    sys::MemoryBlock FinalizeSegs(AllocBase + SegsSizes->StandardSegs,
                                  SegsSizes->FinalizeSegs);

    auto NextStandardSegAddr = ExecutorAddr::fromPtr(StandardSegs.base());
    auto NextFinalizeSegAddr = ExecutorAddr::fromPtr(FinalizeSegs.base());

    LLVM_DEBUG({
      dbgs() << "JITLinkSlabAllocator allocated:\n";
      if (SegsSizes->StandardSegs)
        dbgs() << formatv("  [ {0:x16} -- {1:x16} ]", NextStandardSegAddr,
                          NextStandardSegAddr + StandardSegs.allocatedSize())
               << " to stardard segs\n";
      else
        dbgs() << "  no standard segs\n";
      if (SegsSizes->FinalizeSegs)
        dbgs() << formatv("  [ {0:x16} -- {1:x16} ]", NextFinalizeSegAddr,
                          NextFinalizeSegAddr + FinalizeSegs.allocatedSize())
               << " to finalize segs\n";
      else
        dbgs() << "  no finalize segs\n";
    });

    for (auto &KV : BL.segments()) {
      auto &Group = KV.first;
      auto &Seg = KV.second;

      auto &SegAddr =
          (Group.getMemDeallocPolicy() == MemDeallocPolicy::Standard)
              ? NextStandardSegAddr
              : NextFinalizeSegAddr;

      LLVM_DEBUG({
        dbgs() << "  " << Group << " -> " << formatv("{0:x16}", SegAddr)
               << "\n";
      });
      Seg.WorkingMem = SegAddr.toPtr<char *>();
      Seg.Addr = SegAddr + SlabDelta;

      SegAddr += alignTo(Seg.ContentSize + Seg.ZeroFillSize, PageSize);

      // Zero out the zero-fill memory.
      if (Seg.ZeroFillSize != 0)
        memset(Seg.WorkingMem + Seg.ContentSize, 0, Seg.ZeroFillSize);
    }

    if (auto Err = BL.apply()) {
      OnAllocated(std::move(Err));
      return;
    }

    OnAllocated(std::unique_ptr<InProcessMemoryManager::InFlightAlloc>(
        new IPMMAlloc(*this, std::move(BL), std::move(StandardSegs),
                      std::move(FinalizeSegs))));
  }

  void deallocate(std::vector<FinalizedAlloc> FinalizedAllocs,
                  OnDeallocatedFunction OnDeallocated) override {
    Error Err = Error::success();
    for (auto &FA : FinalizedAllocs) {
      std::unique_ptr<FinalizedAllocInfo> FAI(
          FA.release().toPtr<FinalizedAllocInfo *>());

      // FIXME: Run dealloc actions.

      Err = joinErrors(std::move(Err), freeBlock(FAI->Mem));
    }
    OnDeallocated(std::move(Err));
  }

private:
  JITLinkSlabAllocator(uint64_t SlabSize, Error &Err) {
    ErrorAsOutParameter _(&Err);

    if (!SlabPageSize) {
      if (auto PageSizeOrErr = sys::Process::getPageSize())
        PageSize = *PageSizeOrErr;
      else {
        Err = PageSizeOrErr.takeError();
        return;
      }

      if (PageSize == 0) {
        Err = make_error<StringError>("Page size is zero",
                                      inconvertibleErrorCode());
        return;
      }
    } else
      PageSize = SlabPageSize;

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
      SlabDelta = ExecutorAddr(SlabAddress) -
                      ExecutorAddr::fromPtr(SlabRemaining.base());
  }

  Error freeBlock(sys::MemoryBlock MB) {
    // FIXME: Return memory to slab.
    return Error::success();
  }

  std::mutex SlabMutex;
  sys::MemoryBlock SlabRemaining;
  uint64_t PageSize = 0;
  int64_t SlabDelta = 0;
};

Expected<uint64_t> getSlabAllocSize(StringRef SizeString) {
  SizeString = SizeString.trim();

  uint64_t Units = 1024;

  if (SizeString.endswith_insensitive("kb"))
    SizeString = SizeString.drop_back(2).rtrim();
  else if (SizeString.endswith_insensitive("mb")) {
    Units = 1024 * 1024;
    SizeString = SizeString.drop_back(2).rtrim();
  } else if (SizeString.endswith_insensitive("gb")) {
    Units = 1024 * 1024 * 1024;
    SizeString = SizeString.drop_back(2).rtrim();
  }

  uint64_t SlabSize = 0;
  if (SizeString.getAsInteger(10, SlabSize))
    return make_error<StringError>("Invalid numeric format for slab size",
                                   inconvertibleErrorCode());

  return SlabSize * Units;
}

static std::unique_ptr<JITLinkMemoryManager> createMemoryManager() {
  if (!SlabAllocateSizeString.empty()) {
    auto SlabSize = ExitOnErr(getSlabAllocSize(SlabAllocateSizeString));
    return ExitOnErr(JITLinkSlabAllocator::Create(SlabSize));
  }
  return ExitOnErr(InProcessMemoryManager::Create());
}

static Expected<MaterializationUnit::Interface>
getTestObjectFileInterface(Session &S, MemoryBufferRef O) {

  // Get the standard interface for this object, but ignore the symbols field.
  // We'll handle that manually to include promotion.
  auto I = getObjectFileInterface(S.ES, O);
  if (!I)
    return I.takeError();
  I->SymbolFlags.clear();

  // If creating an object file was going to fail it would have happened above,
  // so we can 'cantFail' this.
  auto Obj = cantFail(object::ObjectFile::createObjectFile(O));

  // The init symbol must be included in the SymbolFlags map if present.
  if (I->InitSymbol)
    I->SymbolFlags[I->InitSymbol] =
        JITSymbolFlags::MaterializationSideEffectsOnly;

  for (auto &Sym : Obj->symbols()) {
    Expected<uint32_t> SymFlagsOrErr = Sym.getFlags();
    if (!SymFlagsOrErr)
      // TODO: Test this error.
      return SymFlagsOrErr.takeError();

    // Skip symbols not defined in this object file.
    if ((*SymFlagsOrErr & object::BasicSymbolRef::SF_Undefined))
      continue;

    auto Name = Sym.getName();
    if (!Name)
      return Name.takeError();

    // Skip symbols that have type SF_File.
    if (auto SymType = Sym.getType()) {
      if (*SymType == object::SymbolRef::ST_File)
        continue;
    } else
      return SymType.takeError();

    auto SymFlags = JITSymbolFlags::fromObjectSymbol(Sym);
    if (!SymFlags)
      return SymFlags.takeError();

    if (SymFlags->isWeak()) {
      // If this is a weak symbol that's not defined in the harness then we
      // need to either mark it as strong (if this is the first definition
      // that we've seen) or discard it.
      if (S.HarnessDefinitions.count(*Name) || S.CanonicalWeakDefs.count(*Name))
        continue;
      S.CanonicalWeakDefs[*Name] = O.getBufferIdentifier();
      *SymFlags &= ~JITSymbolFlags::Weak;
      if (!S.HarnessExternals.count(*Name))
        *SymFlags &= ~JITSymbolFlags::Exported;
    } else if (S.HarnessExternals.count(*Name)) {
      *SymFlags |= JITSymbolFlags::Exported;
    } else if (S.HarnessDefinitions.count(*Name) ||
               !(*SymFlagsOrErr & object::BasicSymbolRef::SF_Global))
      continue;

    auto InternedName = S.ES.intern(*Name);
    I->SymbolFlags[InternedName] = std::move(*SymFlags);
  }

  return I;
}

static Error loadProcessSymbols(Session &S) {
  auto FilterMainEntryPoint =
      [EPName = S.ES.intern(EntryPointName)](SymbolStringPtr Name) {
        return Name != EPName;
      };
  S.MainJD->addGenerator(
      ExitOnErr(orc::EPCDynamicLibrarySearchGenerator::GetForTargetProcess(
          S.ES, std::move(FilterMainEntryPoint))));

  return Error::success();
}

static Error loadDylibs(Session &S) {
  LLVM_DEBUG(dbgs() << "Loading dylibs...\n");
  for (const auto &Dylib : Dylibs) {
    LLVM_DEBUG(dbgs() << "  " << Dylib << "\n");
    auto G = orc::EPCDynamicLibrarySearchGenerator::Load(S.ES, Dylib.c_str());
    if (!G)
      return G.takeError();
    S.MainJD->addGenerator(std::move(*G));
  }

  return Error::success();
}

static Expected<std::unique_ptr<ExecutorProcessControl>> launchExecutor() {
#ifndef LLVM_ON_UNIX
  // FIXME: Add support for Windows.
  return make_error<StringError>("-" + OutOfProcessExecutor.ArgStr +
                                     " not supported on non-unix platforms",
                                 inconvertibleErrorCode());
#elif !LLVM_ENABLE_THREADS
  // Out of process mode using SimpleRemoteEPC depends on threads.
  return make_error<StringError>(
      "-" + OutOfProcessExecutor.ArgStr +
          " requires threads, but LLVM was built with "
          "LLVM_ENABLE_THREADS=Off",
      inconvertibleErrorCode());
#else

  constexpr int ReadEnd = 0;
  constexpr int WriteEnd = 1;

  // Pipe FDs.
  int ToExecutor[2];
  int FromExecutor[2];

  pid_t ChildPID;

  // Create pipes to/from the executor..
  if (pipe(ToExecutor) != 0 || pipe(FromExecutor) != 0)
    return make_error<StringError>("Unable to create pipe for executor",
                                   inconvertibleErrorCode());

  ChildPID = fork();

  if (ChildPID == 0) {
    // In the child...

    // Close the parent ends of the pipes
    close(ToExecutor[WriteEnd]);
    close(FromExecutor[ReadEnd]);

    // Execute the child process.
    std::unique_ptr<char[]> ExecutorPath, FDSpecifier;
    {
      ExecutorPath = std::make_unique<char[]>(OutOfProcessExecutor.size() + 1);
      strcpy(ExecutorPath.get(), OutOfProcessExecutor.data());

      std::string FDSpecifierStr("filedescs=");
      FDSpecifierStr += utostr(ToExecutor[ReadEnd]);
      FDSpecifierStr += ',';
      FDSpecifierStr += utostr(FromExecutor[WriteEnd]);
      FDSpecifier = std::make_unique<char[]>(FDSpecifierStr.size() + 1);
      strcpy(FDSpecifier.get(), FDSpecifierStr.c_str());
    }

    char *const Args[] = {ExecutorPath.get(), FDSpecifier.get(), nullptr};
    int RC = execvp(ExecutorPath.get(), Args);
    if (RC != 0) {
      errs() << "unable to launch out-of-process executor \""
             << ExecutorPath.get() << "\"\n";
      exit(1);
    }
  }
  // else we're the parent...

  // Close the child ends of the pipes
  close(ToExecutor[ReadEnd]);
  close(FromExecutor[WriteEnd]);

  return SimpleRemoteEPC::Create<FDSimpleRemoteEPCTransport>(
      std::make_unique<DynamicThreadPoolTaskDispatcher>(),
      SimpleRemoteEPC::Setup(), FromExecutor[ReadEnd], ToExecutor[WriteEnd]);
#endif
}

#if LLVM_ON_UNIX && LLVM_ENABLE_THREADS
static Error createTCPSocketError(Twine Details) {
  return make_error<StringError>(
      formatv("Failed to connect TCP socket '{0}': {1}",
              OutOfProcessExecutorConnect, Details),
      inconvertibleErrorCode());
}

static Expected<int> connectTCPSocket(std::string Host, std::string PortStr) {
  addrinfo *AI;
  addrinfo Hints{};
  Hints.ai_family = AF_INET;
  Hints.ai_socktype = SOCK_STREAM;
  Hints.ai_flags = AI_NUMERICSERV;

  if (int EC = getaddrinfo(Host.c_str(), PortStr.c_str(), &Hints, &AI))
    return createTCPSocketError("Address resolution failed (" +
                                StringRef(gai_strerror(EC)) + ")");

  // Cycle through the returned addrinfo structures and connect to the first
  // reachable endpoint.
  int SockFD;
  addrinfo *Server;
  for (Server = AI; Server != nullptr; Server = Server->ai_next) {
    // socket might fail, e.g. if the address family is not supported. Skip to
    // the next addrinfo structure in such a case.
    if ((SockFD = socket(AI->ai_family, AI->ai_socktype, AI->ai_protocol)) < 0)
      continue;

    // If connect returns null, we exit the loop with a working socket.
    if (connect(SockFD, Server->ai_addr, Server->ai_addrlen) == 0)
      break;

    close(SockFD);
  }
  freeaddrinfo(AI);

  // If we reached the end of the loop without connecting to a valid endpoint,
  // dump the last error that was logged in socket() or connect().
  if (Server == nullptr)
    return createTCPSocketError(std::strerror(errno));

  return SockFD;
}
#endif

static Expected<std::unique_ptr<ExecutorProcessControl>> connectToExecutor() {
#ifndef LLVM_ON_UNIX
  // FIXME: Add TCP support for Windows.
  return make_error<StringError>("-" + OutOfProcessExecutorConnect.ArgStr +
                                     " not supported on non-unix platforms",
                                 inconvertibleErrorCode());
#elif !LLVM_ENABLE_THREADS
  // Out of process mode using SimpleRemoteEPC depends on threads.
  return make_error<StringError>(
      "-" + OutOfProcessExecutorConnect.ArgStr +
          " requires threads, but LLVM was built with "
          "LLVM_ENABLE_THREADS=Off",
      inconvertibleErrorCode());
#else

  StringRef Host, PortStr;
  std::tie(Host, PortStr) = StringRef(OutOfProcessExecutorConnect).split(':');
  if (Host.empty())
    return createTCPSocketError("Host name for -" +
                                OutOfProcessExecutorConnect.ArgStr +
                                " can not be empty");
  if (PortStr.empty())
    return createTCPSocketError("Port number in -" +
                                OutOfProcessExecutorConnect.ArgStr +
                                " can not be empty");
  int Port = 0;
  if (PortStr.getAsInteger(10, Port))
    return createTCPSocketError("Port number '" + PortStr +
                                "' is not a valid integer");

  Expected<int> SockFD = connectTCPSocket(Host.str(), PortStr.str());
  if (!SockFD)
    return SockFD.takeError();

  return SimpleRemoteEPC::Create<FDSimpleRemoteEPCTransport>(
      std::make_unique<DynamicThreadPoolTaskDispatcher>(),
      SimpleRemoteEPC::Setup(), *SockFD, *SockFD);
#endif
}

class PhonyExternalsGenerator : public DefinitionGenerator {
public:
  Error tryToGenerate(LookupState &LS, LookupKind K, JITDylib &JD,
                      JITDylibLookupFlags JDLookupFlags,
                      const SymbolLookupSet &LookupSet) override {
    SymbolMap PhonySymbols;
    for (auto &KV : LookupSet)
      PhonySymbols[KV.first] = JITEvaluatedSymbol(0, JITSymbolFlags::Exported);
    return JD.define(absoluteSymbols(std::move(PhonySymbols)));
  }
};

Expected<std::unique_ptr<Session>> Session::Create(Triple TT) {

  std::unique_ptr<ExecutorProcessControl> EPC;
  if (OutOfProcessExecutor.getNumOccurrences()) {
    /// If -oop-executor is passed then launch the executor.
    if (auto REPC = launchExecutor())
      EPC = std::move(*REPC);
    else
      return REPC.takeError();
  } else if (OutOfProcessExecutorConnect.getNumOccurrences()) {
    /// If -oop-executor-connect is passed then connect to the executor.
    if (auto REPC = connectToExecutor())
      EPC = std::move(*REPC);
    else
      return REPC.takeError();
  } else {
    /// Otherwise use SelfExecutorProcessControl to target the current process.
    auto PageSize = sys::Process::getPageSize();
    if (!PageSize)
      return PageSize.takeError();
    EPC = std::make_unique<SelfExecutorProcessControl>(
        std::make_shared<SymbolStringPool>(),
        std::make_unique<InPlaceTaskDispatcher>(), std::move(TT), *PageSize,
        createMemoryManager());
  }

  Error Err = Error::success();
  std::unique_ptr<Session> S(new Session(std::move(EPC), Err));
  if (Err)
    return std::move(Err);
  return std::move(S);
}

Session::~Session() {
  if (auto Err = ES.endSession())
    ES.reportError(std::move(Err));
}

Session::Session(std::unique_ptr<ExecutorProcessControl> EPC, Error &Err)
    : ES(std::move(EPC)),
      ObjLayer(ES, ES.getExecutorProcessControl().getMemMgr()) {

  /// Local ObjectLinkingLayer::Plugin class to forward modifyPassConfig to the
  /// Session.
  class JITLinkSessionPlugin : public ObjectLinkingLayer::Plugin {
  public:
    JITLinkSessionPlugin(Session &S) : S(S) {}
    void modifyPassConfig(MaterializationResponsibility &MR, LinkGraph &G,
                          PassConfiguration &PassConfig) override {
      S.modifyPassConfig(G.getTargetTriple(), PassConfig);
    }

    Error notifyFailed(MaterializationResponsibility &MR) override {
      return Error::success();
    }
    Error notifyRemovingResources(ResourceKey K) override {
      return Error::success();
    }
    void notifyTransferringResources(ResourceKey DstKey,
                                     ResourceKey SrcKey) override {}

  private:
    Session &S;
  };

  ErrorAsOutParameter _(&Err);

  ES.setErrorReporter(reportLLVMJITLinkError);

  if (auto MainJDOrErr = ES.createJITDylib("main"))
    MainJD = &*MainJDOrErr;
  else {
    Err = MainJDOrErr.takeError();
    return;
  }

  if (!NoProcessSymbols)
    ExitOnErr(loadProcessSymbols(*this));
  ExitOnErr(loadDylibs(*this));

  auto &TT = ES.getExecutorProcessControl().getTargetTriple();

  if (DebuggerSupport && TT.isOSBinFormatMachO())
    ObjLayer.addPlugin(ExitOnErr(
        GDBJITDebugInfoRegistrationPlugin::Create(this->ES, *MainJD, TT)));

  // Set up the platform.
  if (TT.isOSBinFormatMachO() && !OrcRuntime.empty()) {
    if (auto P =
            MachOPlatform::Create(ES, ObjLayer, *MainJD, OrcRuntime.c_str()))
      ES.setPlatform(std::move(*P));
    else {
      Err = P.takeError();
      return;
    }
  } else if (TT.isOSBinFormatELF() && !OrcRuntime.empty()) {
    if (auto P =
            ELFNixPlatform::Create(ES, ObjLayer, *MainJD, OrcRuntime.c_str()))
      ES.setPlatform(std::move(*P));
    else {
      Err = P.takeError();
      return;
    }
  } else if (!TT.isOSWindows() && !TT.isOSBinFormatMachO()) {
    if (!NoExec)
      ObjLayer.addPlugin(std::make_unique<EHFrameRegistrationPlugin>(
          ES, ExitOnErr(EPCEHFrameRegistrar::Create(this->ES))));
    if (DebuggerSupport)
      ObjLayer.addPlugin(std::make_unique<DebugObjectManagerPlugin>(
          ES, ExitOnErr(createJITLoaderGDBRegistrar(this->ES))));
  }

  ObjLayer.addPlugin(std::make_unique<JITLinkSessionPlugin>(*this));

  // Process any harness files.
  for (auto &HarnessFile : TestHarnesses) {
    HarnessFiles.insert(HarnessFile);

    auto ObjBuffer = ExitOnErr(getFile(HarnessFile));

    auto ObjInterface =
        ExitOnErr(getObjectFileInterface(ES, ObjBuffer->getMemBufferRef()));

    for (auto &KV : ObjInterface.SymbolFlags)
      HarnessDefinitions.insert(*KV.first);

    auto Obj = ExitOnErr(
        object::ObjectFile::createObjectFile(ObjBuffer->getMemBufferRef()));

    for (auto &Sym : Obj->symbols()) {
      uint32_t SymFlags = ExitOnErr(Sym.getFlags());
      auto Name = ExitOnErr(Sym.getName());

      if (Name.empty())
        continue;

      if (SymFlags & object::BasicSymbolRef::SF_Undefined)
        HarnessExternals.insert(Name);
    }
  }

  // If a name is defined by some harness file then it's a definition, not an
  // external.
  for (auto &DefName : HarnessDefinitions)
    HarnessExternals.erase(DefName.getKey());
}

void Session::dumpSessionInfo(raw_ostream &OS) {
  OS << "Registered addresses:\n" << SymbolInfos << FileInfos;
}

void Session::modifyPassConfig(const Triple &TT,
                               PassConfiguration &PassConfig) {
  if (!CheckFiles.empty())
    PassConfig.PostFixupPasses.push_back([this](LinkGraph &G) {
      auto &EPC = ES.getExecutorProcessControl();
      if (EPC.getTargetTriple().getObjectFormat() == Triple::ELF)
        return registerELFGraphInfo(*this, G);

      if (EPC.getTargetTriple().getObjectFormat() == Triple::MachO)
        return registerMachOGraphInfo(*this, G);

      return make_error<StringError>("Unsupported object format for GOT/stub "
                                     "registration",
                                     inconvertibleErrorCode());
    });

  if (ShowLinkGraph)
    PassConfig.PostFixupPasses.push_back([](LinkGraph &G) -> Error {
      outs() << "Link graph \"" << G.getName() << "\" post-fixup:\n";
      G.dump(outs());
      return Error::success();
    });

  PassConfig.PrePrunePasses.push_back(
      [this](LinkGraph &G) { return applyHarnessPromotions(*this, G); });

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

  if (AddSelfRelocations)
    PassConfig.PostPrunePasses.push_back(addSelfRelocations);
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

static Triple getFirstFileTriple() {
  static Triple FirstTT = []() {
    assert(!InputFiles.empty() && "InputFiles can not be empty");
    for (auto InputFile : InputFiles) {
      auto ObjBuffer = ExitOnErr(getFile(InputFile));
      switch (identify_magic(ObjBuffer->getBuffer())) {
      case file_magic::elf_relocatable:
      case file_magic::macho_object:
      case file_magic::coff_object: {
        auto Obj = ExitOnErr(
            object::ObjectFile::createObjectFile(ObjBuffer->getMemBufferRef()));
        return Obj->makeTriple();
      }
      default:
        break;
      }
    }
    return Triple();
  }();

  return FirstTT;
}

static Error sanitizeArguments(const Triple &TT, const char *ArgV0) {

  // -noexec and --args should not be used together.
  if (NoExec && !InputArgv.empty())
    errs() << "Warning: --args passed to -noexec run will be ignored.\n";

  // Set the entry point name if not specified.
  if (EntryPointName.empty())
    EntryPointName = TT.getObjectFormat() == Triple::MachO ? "_main" : "main";

  // Disable debugger support by default in noexec tests.
  if (DebuggerSupport.getNumOccurrences() == 0 && NoExec)
    DebuggerSupport = false;

  // If -slab-allocate is passed, check that we're not trying to use it in
  // -oop-executor or -oop-executor-connect mode.
  //
  // FIXME: Remove once we enable remote slab allocation.
  if (SlabAllocateSizeString != "") {
    if (OutOfProcessExecutor.getNumOccurrences() ||
        OutOfProcessExecutorConnect.getNumOccurrences())
      return make_error<StringError>(
          "-slab-allocate cannot be used with -oop-executor or "
          "-oop-executor-connect",
          inconvertibleErrorCode());
  }

  // If -slab-address is passed, require -slab-allocate and -noexec
  if (SlabAddress != ~0ULL) {
    if (SlabAllocateSizeString == "" || !NoExec)
      return make_error<StringError>(
          "-slab-address requires -slab-allocate and -noexec",
          inconvertibleErrorCode());

    if (SlabPageSize == 0)
      errs() << "Warning: -slab-address used without -slab-page-size.\n";
  }

  if (SlabPageSize != 0) {
    // -slab-page-size requires slab alloc.
    if (SlabAllocateSizeString == "")
      return make_error<StringError>("-slab-page-size requires -slab-allocate",
                                     inconvertibleErrorCode());

    // Check -slab-page-size / -noexec interactions.
    if (!NoExec) {
      if (auto RealPageSize = sys::Process::getPageSize()) {
        if (SlabPageSize % *RealPageSize)
          return make_error<StringError>(
              "-slab-page-size must be a multiple of real page size for exec "
              "tests (did you mean to use -noexec ?)\n",
              inconvertibleErrorCode());
      } else {
        errs() << "Could not retrieve process page size:\n";
        logAllUnhandledErrors(RealPageSize.takeError(), errs(), "");
        errs() << "Executing with slab page size = "
               << formatv("{0:x}", SlabPageSize) << ".\n"
               << "Tool may crash if " << formatv("{0:x}", SlabPageSize)
               << " is not a multiple of the real process page size.\n"
               << "(did you mean to use -noexec ?)";
      }
    }
  }

  // Only one of -oop-executor and -oop-executor-connect can be used.
  if (!!OutOfProcessExecutor.getNumOccurrences() &&
      !!OutOfProcessExecutorConnect.getNumOccurrences())
    return make_error<StringError>(
        "Only one of -" + OutOfProcessExecutor.ArgStr + " and -" +
            OutOfProcessExecutorConnect.ArgStr + " can be specified",
        inconvertibleErrorCode());

  // If -oop-executor was used but no value was specified then use a sensible
  // default.
  if (!!OutOfProcessExecutor.getNumOccurrences() &&
      OutOfProcessExecutor.empty()) {
    SmallString<256> OOPExecutorPath(sys::fs::getMainExecutable(
        ArgV0, reinterpret_cast<void *>(&sanitizeArguments)));
    sys::path::remove_filename(OOPExecutorPath);
    sys::path::append(OOPExecutorPath, "llvm-jitlink-executor");
    OutOfProcessExecutor = OOPExecutorPath.str().str();
  }

  return Error::success();
}

static void addPhonyExternalsGenerator(Session &S) {
  S.MainJD->addGenerator(std::make_unique<PhonyExternalsGenerator>());
}

static Error createJITDylibs(Session &S,
                             std::map<unsigned, JITDylib *> &IdxToJD) {
  // First, set up JITDylibs.
  LLVM_DEBUG(dbgs() << "Creating JITDylibs...\n");
  {
    // Create a "main" JITLinkDylib.
    IdxToJD[0] = S.MainJD;
    S.JDSearchOrder.push_back({S.MainJD, JITDylibLookupFlags::MatchAllSymbols});
    LLVM_DEBUG(dbgs() << "  0: " << S.MainJD->getName() << "\n");

    // Add any extra JITDylibs from the command line.
    for (auto JDItr = JITDylibs.begin(), JDEnd = JITDylibs.end();
         JDItr != JDEnd; ++JDItr) {
      auto JD = S.ES.createJITDylib(*JDItr);
      if (!JD)
        return JD.takeError();
      unsigned JDIdx = JITDylibs.getPosition(JDItr - JITDylibs.begin());
      IdxToJD[JDIdx] = &*JD;
      S.JDSearchOrder.push_back({&*JD, JITDylibLookupFlags::MatchAllSymbols});
      LLVM_DEBUG(dbgs() << "  " << JDIdx << ": " << JD->getName() << "\n");
    }
  }

  LLVM_DEBUG({
    dbgs() << "Dylib search order is [ ";
    for (auto &KV : S.JDSearchOrder)
      dbgs() << KV.first->getName() << " ";
    dbgs() << "]\n";
  });

  return Error::success();
}

static Error addAbsoluteSymbols(Session &S,
                                const std::map<unsigned, JITDylib *> &IdxToJD) {
  // Define absolute symbols.
  LLVM_DEBUG(dbgs() << "Defining absolute symbols...\n");
  for (auto AbsDefItr = AbsoluteDefs.begin(), AbsDefEnd = AbsoluteDefs.end();
       AbsDefItr != AbsDefEnd; ++AbsDefItr) {
    unsigned AbsDefArgIdx =
      AbsoluteDefs.getPosition(AbsDefItr - AbsoluteDefs.begin());
    auto &JD = *std::prev(IdxToJD.lower_bound(AbsDefArgIdx))->second;

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
                                         "\" in absolute symbol definition \"" +
                                         AbsDefStmt + "\"",
                                     inconvertibleErrorCode());
    JITEvaluatedSymbol AbsDef(Addr, JITSymbolFlags::Exported);
    if (auto Err = JD.define(absoluteSymbols({{S.ES.intern(Name), AbsDef}})))
      return Err;

    // Register the absolute symbol with the session symbol infos.
    S.SymbolInfos[Name] = {ArrayRef<char>(), Addr};
  }

  return Error::success();
}

static Error addAliases(Session &S,
                        const std::map<unsigned, JITDylib *> &IdxToJD) {
  // Define absolute symbols.
  LLVM_DEBUG(dbgs() << "Defining aliases...\n");
  for (auto AliasItr = Aliases.begin(), AliasEnd = Aliases.end();
       AliasItr != AliasEnd; ++AliasItr) {
    unsigned AliasArgIdx = Aliases.getPosition(AliasItr - Aliases.begin());
    auto &JD = *std::prev(IdxToJD.lower_bound(AliasArgIdx))->second;

    StringRef AliasStmt = *AliasItr;
    size_t EqIdx = AliasStmt.find_first_of('=');
    if (EqIdx == StringRef::npos)
      return make_error<StringError>("Invalid alias definition \"" + AliasStmt +
                                         "\". Syntax: <name>=<addr>",
                                     inconvertibleErrorCode());
    StringRef Alias = AliasStmt.substr(0, EqIdx).trim();
    StringRef Aliasee = AliasStmt.substr(EqIdx + 1).trim();

    SymbolAliasMap SAM;
    SAM[S.ES.intern(Alias)] = {S.ES.intern(Aliasee), JITSymbolFlags::Exported};
    if (auto Err = JD.define(symbolAliases(std::move(SAM))))
      return Err;
  }

  return Error::success();
}

static Error addTestHarnesses(Session &S) {
  LLVM_DEBUG(dbgs() << "Adding test harness objects...\n");
  for (auto HarnessFile : TestHarnesses) {
    LLVM_DEBUG(dbgs() << "  " << HarnessFile << "\n");
    auto ObjBuffer = getFile(HarnessFile);
    if (!ObjBuffer)
      return ObjBuffer.takeError();
    if (auto Err = S.ObjLayer.add(*S.MainJD, std::move(*ObjBuffer)))
      return Err;
  }
  return Error::success();
}

static Error addObjects(Session &S,
                        const std::map<unsigned, JITDylib *> &IdxToJD) {

  // Load each object into the corresponding JITDylib..
  LLVM_DEBUG(dbgs() << "Adding objects...\n");
  for (auto InputFileItr = InputFiles.begin(), InputFileEnd = InputFiles.end();
       InputFileItr != InputFileEnd; ++InputFileItr) {
    unsigned InputFileArgIdx =
        InputFiles.getPosition(InputFileItr - InputFiles.begin());
    const std::string &InputFile = *InputFileItr;
    if (StringRef(InputFile).endswith(".a"))
      continue;
    auto &JD = *std::prev(IdxToJD.lower_bound(InputFileArgIdx))->second;
    LLVM_DEBUG(dbgs() << "  " << InputFileArgIdx << ": \"" << InputFile
                      << "\" to " << JD.getName() << "\n";);
    auto ObjBuffer = getFile(InputFile);
    if (!ObjBuffer)
      return ObjBuffer.takeError();

    if (S.HarnessFiles.empty()) {
      if (auto Err = S.ObjLayer.add(JD, std::move(*ObjBuffer)))
        return Err;
    } else {
      // We're in -harness mode. Use a custom interface for this
      // test object.
      auto ObjInterface =
          getTestObjectFileInterface(S, (*ObjBuffer)->getMemBufferRef());
      if (!ObjInterface)
        return ObjInterface.takeError();
      if (auto Err = S.ObjLayer.add(JD, std::move(*ObjBuffer),
                                    std::move(*ObjInterface)))
        return Err;
    }
  }

  return Error::success();
}

static Expected<MaterializationUnit::Interface>
getObjectFileInterfaceHidden(ExecutionSession &ES, MemoryBufferRef ObjBuffer) {
  auto I = getObjectFileInterface(ES, ObjBuffer);
  if (I) {
    for (auto &KV : I->SymbolFlags)
      KV.second &= ~JITSymbolFlags::Exported;
  }
  return I;
}

static Error addLibraries(Session &S,
                          const std::map<unsigned, JITDylib *> &IdxToJD) {

  // 1. Collect search paths for each JITDylib.
  DenseMap<const JITDylib *, SmallVector<StringRef, 2>> JDSearchPaths;

  for (auto LSPItr = LibrarySearchPaths.begin(),
            LSPEnd = LibrarySearchPaths.end();
       LSPItr != LSPEnd; ++LSPItr) {
    unsigned LibrarySearchPathIdx =
        LibrarySearchPaths.getPosition(LSPItr - LibrarySearchPaths.begin());
    auto &JD = *std::prev(IdxToJD.lower_bound(LibrarySearchPathIdx))->second;

    StringRef LibrarySearchPath = *LSPItr;
    if (sys::fs::get_file_type(LibrarySearchPath) !=
        sys::fs::file_type::directory_file)
      return make_error<StringError>("While linking " + JD.getName() + ", -L" +
                                         LibrarySearchPath +
                                         " does not point to a directory",
                                     inconvertibleErrorCode());

    JDSearchPaths[&JD].push_back(*LSPItr);
  }

  LLVM_DEBUG({
    if (!JDSearchPaths.empty())
      dbgs() << "Search paths:\n";
    for (auto &KV : JDSearchPaths) {
      dbgs() << "  " << KV.first->getName() << ": [";
      for (auto &LibSearchPath : KV.second)
        dbgs() << " \"" << LibSearchPath << "\"";
      dbgs() << " ]\n";
    }
  });

  // 2. Collect library loads
  struct LibraryLoad {
    StringRef LibName;
    bool IsPath = false;
    unsigned Position;
    StringRef *CandidateExtensions;
    enum { Standard, Hidden } Modifier;
  };
  std::vector<LibraryLoad> LibraryLoads;
  // Add archive files from the inputs to LibraryLoads.
  for (auto InputFileItr = InputFiles.begin(), InputFileEnd = InputFiles.end();
       InputFileItr != InputFileEnd; ++InputFileItr) {
    StringRef InputFile = *InputFileItr;
    if (!InputFile.endswith(".a"))
      continue;
    LibraryLoad LL;
    LL.LibName = InputFile;
    LL.IsPath = true;
    LL.Position = InputFiles.getPosition(InputFileItr - InputFiles.begin());
    LL.CandidateExtensions = nullptr;
    LL.Modifier = LibraryLoad::Standard;
    LibraryLoads.push_back(std::move(LL));
  }

  // Add -load_hidden arguments to LibraryLoads.
  for (auto LibItr = LoadHidden.begin(), LibEnd = LoadHidden.end();
       LibItr != LibEnd; ++LibItr) {
    LibraryLoad LL;
    LL.LibName = *LibItr;
    LL.IsPath = true;
    LL.Position = LoadHidden.getPosition(LibItr - LoadHidden.begin());
    LL.CandidateExtensions = nullptr;
    LL.Modifier = LibraryLoad::Hidden;
    LibraryLoads.push_back(std::move(LL));
  }
  StringRef StandardExtensions[] = {".so", ".dylib", ".a"};
  StringRef ArchiveExtensionsOnly[] = {".a"};

  // Add -lx arguments to LibraryLoads.
  for (auto LibItr = Libraries.begin(), LibEnd = Libraries.end();
       LibItr != LibEnd; ++LibItr) {
    LibraryLoad LL;
    LL.LibName = *LibItr;
    LL.Position = Libraries.getPosition(LibItr - Libraries.begin());
    LL.CandidateExtensions = StandardExtensions;
    LL.Modifier = LibraryLoad::Standard;
    LibraryLoads.push_back(std::move(LL));
  }

  // Add -hidden-lx arguments to LibraryLoads.
  for (auto LibHiddenItr = LibrariesHidden.begin(),
            LibHiddenEnd = LibrariesHidden.end();
       LibHiddenItr != LibHiddenEnd; ++LibHiddenItr) {
    LibraryLoad LL;
    LL.LibName = *LibHiddenItr;
    LL.Position =
        LibrariesHidden.getPosition(LibHiddenItr - LibrariesHidden.begin());
    LL.CandidateExtensions = ArchiveExtensionsOnly;
    LL.Modifier = LibraryLoad::Hidden;
    LibraryLoads.push_back(std::move(LL));
  }

  // If there are any load-<modified> options then turn on flag overrides
  // to avoid flag mismatch errors.
  if (!LibrariesHidden.empty() || !LoadHidden.empty())
    S.ObjLayer.setOverrideObjectFlagsWithResponsibilityFlags(true);

  // Sort library loads by position in the argument list.
  llvm::sort(LibraryLoads, [](const LibraryLoad &LHS, const LibraryLoad &RHS) {
    return LHS.Position < RHS.Position;
  });

  // 3. Process library loads.
  auto AddArchive = [&](const char *Path, const LibraryLoad &LL)
      -> Expected<std::unique_ptr<StaticLibraryDefinitionGenerator>> {
    unique_function<Expected<MaterializationUnit::Interface>(
        ExecutionSession & ES, MemoryBufferRef ObjBuffer)>
        GetObjFileInterface;
    switch (LL.Modifier) {
    case LibraryLoad::Standard:
      GetObjFileInterface = getObjectFileInterface;
      break;
    case LibraryLoad::Hidden:
      GetObjFileInterface = getObjectFileInterfaceHidden;
      break;
    }
    return StaticLibraryDefinitionGenerator::Load(
        S.ObjLayer, Path, S.ES.getExecutorProcessControl().getTargetTriple(),
        std::move(GetObjFileInterface));
  };

  for (auto &LL : LibraryLoads) {
    bool LibFound = false;
    auto &JD = *std::prev(IdxToJD.lower_bound(LL.Position))->second;

    // If this is the name of a JITDylib then link against that.
    if (auto *LJD = S.ES.getJITDylibByName(LL.LibName)) {
      JD.addToLinkOrder(*LJD);
      continue;
    }

    if (LL.IsPath) {
      auto G = AddArchive(LL.LibName.str().c_str(), LL);
      if (!G)
        return createFileError(LL.LibName, G.takeError());
      JD.addGenerator(std::move(*G));
      LLVM_DEBUG({
        dbgs() << "Adding generator for static library " << LL.LibName << " to "
               << JD.getName() << "\n";
      });
      continue;
    }

    // Otherwise look through the search paths.
    auto JDSearchPathsItr = JDSearchPaths.find(&JD);
    if (JDSearchPathsItr != JDSearchPaths.end()) {
      for (StringRef SearchPath : JDSearchPathsItr->second) {
        for (const char *LibExt : {".dylib", ".so", ".a"}) {
          SmallVector<char, 256> LibPath;
          LibPath.reserve(SearchPath.size() + strlen("lib") +
                          LL.LibName.size() + strlen(LibExt) +
                          2); // +2 for pathsep, null term.
          llvm::copy(SearchPath, std::back_inserter(LibPath));
          sys::path::append(LibPath, "lib" + LL.LibName + LibExt);
          LibPath.push_back('\0');

          // Skip missing or non-regular paths.
          if (sys::fs::get_file_type(LibPath.data()) !=
              sys::fs::file_type::regular_file) {
            continue;
          }

          file_magic Magic;
          if (auto EC = identify_magic(LibPath, Magic)) {
            // If there was an error loading the file then skip it.
            LLVM_DEBUG({
              dbgs() << "Library search found \"" << LibPath
                     << "\", but could not identify file type (" << EC.message()
                     << "). Skipping.\n";
            });
            continue;
          }

          // We identified the magic. Assume that we can load it -- we'll reset
          // in the default case.
          LibFound = true;
          switch (Magic) {
          case file_magic::elf_shared_object:
          case file_magic::macho_dynamically_linked_shared_lib: {
            // TODO: On first reference to LibPath this should create a JITDylib
            // with a generator and add it to JD's links-against list. Subsquent
            // references should use the JITDylib created on the first
            // reference.
            auto G =
                EPCDynamicLibrarySearchGenerator::Load(S.ES, LibPath.data());
            if (!G)
              return G.takeError();
            LLVM_DEBUG({
              dbgs() << "Adding generator for dynamic library "
                     << LibPath.data() << " to " << JD.getName() << "\n";
            });
            JD.addGenerator(std::move(*G));
            break;
          }
          case file_magic::archive:
          case file_magic::macho_universal_binary: {
            auto G = AddArchive(LibPath.data(), LL);
            if (!G)
              return G.takeError();
            JD.addGenerator(std::move(*G));
            LLVM_DEBUG({
              dbgs() << "Adding generator for static library " << LibPath.data()
                     << " to " << JD.getName() << "\n";
            });
            break;
          }
          default:
            // This file isn't a recognized library kind.
            LLVM_DEBUG({
              dbgs() << "Library search found \"" << LibPath
                     << "\", but file type is not supported. Skipping.\n";
            });
            LibFound = false;
            break;
          }
          if (LibFound)
            break;
        }
        if (LibFound)
          break;
      }
    }

    if (!LibFound)
      return make_error<StringError>("While linking " + JD.getName() +
                                         ", could not find library for -l" +
                                         LL.LibName,
                                     inconvertibleErrorCode());
  }

  return Error::success();
}

static Error addSessionInputs(Session &S) {
  std::map<unsigned, JITDylib *> IdxToJD;

  if (auto Err = createJITDylibs(S, IdxToJD))
    return Err;

  if (auto Err = addAbsoluteSymbols(S, IdxToJD))
    return Err;

  if (auto Err = addAliases(S, IdxToJD))
    return Err;

  if (!TestHarnesses.empty())
    if (auto Err = addTestHarnesses(S))
      return Err;

  if (auto Err = addObjects(S, IdxToJD))
    return Err;

  if (auto Err = addLibraries(S, IdxToJD))
    return Err;

  return Error::success();
}

namespace {
struct TargetInfo {
  const Target *TheTarget;
  std::unique_ptr<MCSubtargetInfo> STI;
  std::unique_ptr<MCRegisterInfo> MRI;
  std::unique_ptr<MCAsmInfo> MAI;
  std::unique_ptr<MCContext> Ctx;
  std::unique_ptr<MCDisassembler> Disassembler;
  std::unique_ptr<MCInstrInfo> MII;
  std::unique_ptr<MCInstrAnalysis> MIA;
  std::unique_ptr<MCInstPrinter> InstPrinter;
};
} // anonymous namespace

static TargetInfo getTargetInfo(const Triple &TT) {
  auto TripleName = TT.str();
  std::string ErrorStr;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, ErrorStr);
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

  auto Ctx = std::make_unique<MCContext>(Triple(TripleName), MAI.get(),
                                         MRI.get(), STI.get());

  std::unique_ptr<MCDisassembler> Disassembler(
      TheTarget->createMCDisassembler(*STI, *Ctx));
  if (!Disassembler)
    ExitOnErr(make_error<StringError>("Unable to create disassembler for " +
                                          TripleName,
                                      inconvertibleErrorCode()));

  std::unique_ptr<MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  if (!MII)
    ExitOnErr(make_error<StringError>("Unable to create instruction info for" +
                                          TripleName,
                                      inconvertibleErrorCode()));

  std::unique_ptr<MCInstrAnalysis> MIA(
      TheTarget->createMCInstrAnalysis(MII.get()));
  if (!MIA)
    ExitOnErr(make_error<StringError>(
        "Unable to create instruction analysis for" + TripleName,
        inconvertibleErrorCode()));

  std::unique_ptr<MCInstPrinter> InstPrinter(
      TheTarget->createMCInstPrinter(Triple(TripleName), 0, *MAI, *MII, *MRI));
  if (!InstPrinter)
    ExitOnErr(make_error<StringError>(
        "Unable to create instruction printer for" + TripleName,
        inconvertibleErrorCode()));
  return {TheTarget,      std::move(STI), std::move(MRI),
          std::move(MAI), std::move(Ctx), std::move(Disassembler),
          std::move(MII), std::move(MIA), std::move(InstPrinter)};
}

static Error runChecks(Session &S) {
  const auto &TT = S.ES.getExecutorProcessControl().getTargetTriple();

  if (CheckFiles.empty())
    return Error::success();

  LLVM_DEBUG(dbgs() << "Running checks...\n");

  auto TI = getTargetInfo(TT);

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
      TT.isLittleEndian() ? support::little : support::big,
      TI.Disassembler.get(), TI.InstPrinter.get(), dbgs());

  std::string CheckLineStart = "# " + CheckName + ":";
  for (auto &CheckFile : CheckFiles) {
    auto CheckerFileBuf = ExitOnErr(getFile(CheckFile));
    if (!Checker.checkAllRulesInBuffer(CheckLineStart, &*CheckerFileBuf))
      ExitOnErr(make_error<StringError>(
          "Some checks in " + CheckFile + " failed", inconvertibleErrorCode()));
  }

  return Error::success();
}

static Error addSelfRelocations(LinkGraph &G) {
  auto TI = getTargetInfo(G.getTargetTriple());
  for (auto *Sym : G.defined_symbols())
    if (Sym->isCallable())
      if (auto Err = addFunctionPointerRelocationsToCurrentSymbol(
              *Sym, G, *TI.Disassembler, *TI.MIA))
        return Err;
  return Error::success();
}

static void dumpSessionStats(Session &S) {
  if (!ShowSizes)
    return;
  if (!OrcRuntime.empty())
    outs() << "Note: Session stats include runtime and entry point lookup, but "
              "not JITDylib initialization/deinitialization.\n";
  if (ShowSizes)
    outs() << "  Total size of all blocks before pruning: "
           << S.SizeBeforePruning
           << "\n  Total size of all blocks after fixups: " << S.SizeAfterFixups
           << "\n";
}

static Expected<JITEvaluatedSymbol> getMainEntryPoint(Session &S) {
  return S.ES.lookup(S.JDSearchOrder, S.ES.intern(EntryPointName));
}

static Expected<JITEvaluatedSymbol> getOrcRuntimeEntryPoint(Session &S) {
  std::string RuntimeEntryPoint = "__orc_rt_run_program_wrapper";
  const auto &TT = S.ES.getExecutorProcessControl().getTargetTriple();
  if (TT.getObjectFormat() == Triple::MachO)
    RuntimeEntryPoint = '_' + RuntimeEntryPoint;
  return S.ES.lookup(S.JDSearchOrder, S.ES.intern(RuntimeEntryPoint));
}

static Expected<JITEvaluatedSymbol> getEntryPoint(Session &S) {
  JITEvaluatedSymbol EntryPoint;

  // Find the entry-point function unconditionally, since we want to force
  // it to be materialized to collect stats.
  if (auto EP = getMainEntryPoint(S))
    EntryPoint = *EP;
  else
    return EP.takeError();
  LLVM_DEBUG({
    dbgs() << "Using entry point \"" << EntryPointName
           << "\": " << formatv("{0:x16}", EntryPoint.getAddress()) << "\n";
  });

  // If we're running with the ORC runtime then replace the entry-point
  // with the __orc_rt_run_program symbol.
  if (!OrcRuntime.empty()) {
    if (auto EP = getOrcRuntimeEntryPoint(S))
      EntryPoint = *EP;
    else
      return EP.takeError();
    LLVM_DEBUG({
      dbgs() << "(called via __orc_rt_run_program_wrapper at "
             << formatv("{0:x16}", EntryPoint.getAddress()) << ")\n";
    });
  }

  return EntryPoint;
}

static Expected<int> runWithRuntime(Session &S, ExecutorAddr EntryPointAddr) {
  StringRef DemangledEntryPoint = EntryPointName;
  const auto &TT = S.ES.getExecutorProcessControl().getTargetTriple();
  if (TT.getObjectFormat() == Triple::MachO &&
      DemangledEntryPoint.front() == '_')
    DemangledEntryPoint = DemangledEntryPoint.drop_front();
  using llvm::orc::shared::SPSString;
  using SPSRunProgramSig =
      int64_t(SPSString, SPSString, shared::SPSSequence<SPSString>);
  int64_t Result;
  if (auto Err = S.ES.callSPSWrapper<SPSRunProgramSig>(
          EntryPointAddr, Result, S.MainJD->getName(), DemangledEntryPoint,
          static_cast<std::vector<std::string> &>(InputArgv)))
    return std::move(Err);
  return Result;
}

static Expected<int> runWithoutRuntime(Session &S,
                                       ExecutorAddr EntryPointAddr) {
  return S.ES.getExecutorProcessControl().runAsMain(EntryPointAddr, InputArgv);
}

namespace {
struct JITLinkTimers {
  TimerGroup JITLinkTG{"llvm-jitlink timers", "timers for llvm-jitlink phases"};
  Timer LoadObjectsTimer{"load", "time to load/add object files", JITLinkTG};
  Timer LinkTimer{"link", "time to link object files", JITLinkTG};
  Timer RunTimer{"run", "time to execute jitlink'd code", JITLinkTG};
};
} // namespace

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);

  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  cl::HideUnrelatedOptions({&JITLinkCategory, &getColorCategory()});
  cl::ParseCommandLineOptions(argc, argv, "llvm jitlink tool");
  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  /// If timers are enabled, create a JITLinkTimers instance.
  std::unique_ptr<JITLinkTimers> Timers =
      ShowTimes ? std::make_unique<JITLinkTimers>() : nullptr;

  ExitOnErr(sanitizeArguments(getFirstFileTriple(), argv[0]));

  auto S = ExitOnErr(Session::Create(getFirstFileTriple()));

  {
    TimeRegion TR(Timers ? &Timers->LoadObjectsTimer : nullptr);
    ExitOnErr(addSessionInputs(*S));
  }

  if (PhonyExternals)
    addPhonyExternalsGenerator(*S);

  if (ShowInitialExecutionSessionState)
    S->ES.dump(outs());

  Expected<JITEvaluatedSymbol> EntryPoint(nullptr);
  {
    ExpectedAsOutParameter<JITEvaluatedSymbol> _(&EntryPoint);
    TimeRegion TR(Timers ? &Timers->LinkTimer : nullptr);
    EntryPoint = getEntryPoint(*S);
  }

  // Print any reports regardless of whether we succeeded or failed.
  if (ShowEntryExecutionSessionState)
    S->ES.dump(outs());

  if (ShowAddrs)
    S->dumpSessionInfo(outs());

  dumpSessionStats(*S);

  if (!EntryPoint) {
    if (Timers)
      Timers->JITLinkTG.printAll(errs());
    reportLLVMJITLinkError(EntryPoint.takeError());
    exit(1);
  }

  ExitOnErr(runChecks(*S));

  if (NoExec)
    return 0;

  int Result = 0;
  {
    LLVM_DEBUG(dbgs() << "Running \"" << EntryPointName << "\"...\n");
    TimeRegion TR(Timers ? &Timers->RunTimer : nullptr);
    if (!OrcRuntime.empty())
      Result =
          ExitOnErr(runWithRuntime(*S, ExecutorAddr(EntryPoint->getAddress())));
    else
      Result = ExitOnErr(
          runWithoutRuntime(*S, ExecutorAddr(EntryPoint->getAddress())));
  }

  // Destroy the session.
  ExitOnErr(S->ES.endSession());
  S.reset();

  if (Timers)
    Timers->JITLinkTG.printAll(errs());

  // If the executing code set a test result override then use that.
  if (UseTestResultOverride)
    Result = TestResultOverride;

  return Result;
}
