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

#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include <list>
#include <string>

#define DEBUG_TYPE "llvm-jitlink"

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
    EntryPointName("entry", cl::desc("Symbol to call as main entry point"),
                   cl::init(""));

static cl::list<std::string> JITLinkDylibs(
    "jld", cl::desc("Specifies the JITDylib to be used for any subsequent "
                    "input file arguments"));

static cl::list<std::string>
    Dylibs("dlopen", cl::desc("Dynamic libraries to load before linking"),
           cl::ZeroOrMore);

static cl::opt<bool>
    NoProcessSymbols("no-process-syms",
                     cl::desc("Do not resolve to llvm-jitlink process symbols"),
                     cl::init(false));

static cl::list<std::string> AbsoluteDefs(
    "define-abs",
    cl::desc("Inject absolute symbol definitions (syntax: <name>=<addr>)"),
    cl::ZeroOrMore);

static cl::opt<bool> ShowAddrs(
    "show-addrs",
    cl::desc("Print registered symbol, section, got and stub addresses"),
    cl::init(false));

static cl::opt<bool> ShowAtomGraph(
    "show-graph",
    cl::desc("Print the atom graph after fixups have been applied"),
    cl::init(false));

static cl::opt<bool> ShowSizes(
    "show-sizes",
    cl::desc("Show sizes pre- and post-dead stripping, and allocations"),
    cl::init(false));

static cl::opt<bool> ShowRelocatedSectionContents(
    "show-relocated-section-contents",
    cl::desc("show section contents after fixups have been applied"),
    cl::init(false));

ExitOnError ExitOnErr;

namespace llvm {

static raw_ostream &
operator<<(raw_ostream &OS, const Session::MemoryRegionInfo &MRI) {
  return OS << "target addr = " << format("0x%016" PRIx64, MRI.TargetAddress)
            << ", content: " << (const void *)MRI.Content.data() << " -- "
            << (const void *)(MRI.Content.data() + MRI.Content.size()) << " ("
            << MRI.Content.size() << " bytes)";
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

static uint64_t computeTotalAtomSizes(AtomGraph &G) {
  uint64_t TotalSize = 0;
  for (auto *DA : G.defined_atoms())
    if (DA->isZeroFill())
      TotalSize += DA->getZeroFillSize();
    else
      TotalSize += DA->getContent().size();
  return TotalSize;
}

static void dumpSectionContents(raw_ostream &OS, AtomGraph &G) {
  constexpr JITTargetAddress DumpWidth = 16;
  static_assert(isPowerOf2_64(DumpWidth), "DumpWidth must be a power of two");

  // Put sections in address order.
  std::vector<Section *> Sections;
  for (auto &S : G.sections())
    Sections.push_back(&S);

  std::sort(Sections.begin(), Sections.end(),
            [](const Section *LHS, const Section *RHS) {
              if (LHS->atoms_empty() && RHS->atoms_empty())
                return false;
              if (LHS->atoms_empty())
                return false;
              if (RHS->atoms_empty())
                return true;
              return (*LHS->atoms().begin())->getAddress() <
                     (*RHS->atoms().begin())->getAddress();
            });

  for (auto *S : Sections) {
    OS << S->getName() << " content:";
    if (S->atoms_empty()) {
      OS << "\n  section empty\n";
      continue;
    }

    // Sort atoms into order, then render.
    std::vector<DefinedAtom *> Atoms(S->atoms().begin(), S->atoms().end());
    std::sort(Atoms.begin(), Atoms.end(),
              [](const DefinedAtom *LHS, const DefinedAtom *RHS) {
                return LHS->getAddress() < RHS->getAddress();
              });

    JITTargetAddress NextAddr = Atoms.front()->getAddress() & ~(DumpWidth - 1);
    for (auto *DA : Atoms) {
      bool IsZeroFill = DA->isZeroFill();
      JITTargetAddress AtomStart = DA->getAddress();
      JITTargetAddress AtomSize =
          IsZeroFill ? DA->getZeroFillSize() : DA->getContent().size();
      JITTargetAddress AtomEnd = AtomStart + AtomSize;
      const uint8_t *AtomData =
          IsZeroFill ? nullptr : DA->getContent().bytes_begin();

      // Pad any space before the atom starts.
      while (NextAddr != AtomStart) {
        if (NextAddr % DumpWidth == 0)
          OS << formatv("\n{0:x16}:", NextAddr);
        OS << "   ";
        ++NextAddr;
      }

      // Render the atom content.
      while (NextAddr != AtomEnd) {
        if (NextAddr % DumpWidth == 0)
          OS << formatv("\n{0:x16}:", NextAddr);
        if (IsZeroFill)
          OS << " 00";
        else
          OS << formatv(" {0:x-2}", AtomData[NextAddr - AtomStart]);
        ++NextAddr;
      }
    }
    OS << "\n";
  }
}

Session::Session(Triple TT)
    : ObjLayer(ES, MemMgr, ObjectLinkingLayer::NotifyLoadedFunction(),
               ObjectLinkingLayer::NotifyEmittedFunction(),
               [this](const Triple &TT, PassConfiguration &PassConfig) {
                 modifyPassConfig(TT, PassConfig);
               }),
      TT(std::move(TT)) {}

void Session::dumpSessionInfo(raw_ostream &OS) {
  OS << "Registered addresses:\n" << SymbolInfos << FileInfos;
}

void Session::modifyPassConfig(const Triple &FTT,
                               PassConfiguration &PassConfig) {
  if (!CheckFiles.empty())
    PassConfig.PostFixupPasses.push_back([this](AtomGraph &G) {
      if (TT.getObjectFormat() == Triple::MachO)
        return registerMachOStubsAndGOT(*this, G);
      return make_error<StringError>("Unsupported object format for GOT/stub "
                                     "registration",
                                     inconvertibleErrorCode());
    });

  if (ShowAtomGraph)
    PassConfig.PostFixupPasses.push_back([](AtomGraph &G) -> Error {
      outs() << "Atom graph post-fixup:\n";
      G.dump(outs());
      return Error::success();
    });


  if (ShowSizes) {
    PassConfig.PrePrunePasses.push_back([this](AtomGraph &G) -> Error {
        SizeBeforePruning += computeTotalAtomSizes(G);
        return Error::success();
      });
    PassConfig.PostFixupPasses.push_back([this](AtomGraph &G) -> Error {
        SizeAfterFixups += computeTotalAtomSizes(G);
        return Error::success();
      });
  }

  if (ShowRelocatedSectionContents)
    PassConfig.PostFixupPasses.push_back([](AtomGraph &G) -> Error {
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

void setEntryPointNameIfNotProvided(const Session &S) {
  if (EntryPointName.empty()) {
    if (S.TT.getObjectFormat() == Triple::MachO)
      EntryPointName = "_main";
    else
      EntryPointName = "main";
  }
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
  S.ES.getMainJITDylib().setGenerator(
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
    auto &MainJD = S.ES.getMainJITDylib();
    IdxToJLD[0] = &MainJD;
    S.JDSearchOrder.push_back(&MainJD);
    LLVM_DEBUG(dbgs() << "  0: " << MainJD.getName() << "\n");

    // Add any extra JITLinkDylibs from the command line.
    std::string JDNamePrefix("lib");
    for (auto JLDItr = JITLinkDylibs.begin(), JLDEnd = JITLinkDylibs.end();
         JLDItr != JLDEnd; ++JLDItr) {
      auto &JD = S.ES.createJITDylib(JDNamePrefix + *JLDItr);
      unsigned JDIdx =
          JITLinkDylibs.getPosition(JLDItr - JITLinkDylibs.begin());
      IdxToJLD[JDIdx] = &JD;
      S.JDSearchOrder.push_back(&JD);
      LLVM_DEBUG(dbgs() << "  " << JDIdx << ": " << JD.getName() << "\n");
    }

    // Set every dylib to link against every other, in command line order.
    for (auto *JD : S.JDSearchOrder) {
      JITDylibSearchList O;
      for (auto *JD2 : S.JDSearchOrder) {
        if (JD2 == JD)
          continue;
        O.push_back(std::make_pair(JD2, false));
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

  std::unique_ptr<MCAsmInfo> MAI(TheTarget->createMCAsmInfo(*MRI, TripleName));
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

  for (auto &CheckFile : CheckFiles) {
    auto CheckerFileBuf =
        ExitOnErr(errorOrToExpected(MemoryBuffer::getFile(CheckFile)));
    if (!Checker.checkAllRulesInBuffer("# jitlink-check:", &*CheckerFileBuf))
      ExitOnErr(make_error<StringError>(
          "Some checks in " + CheckFile + " failed", inconvertibleErrorCode()));
  }

  return Error::success();
}

static void dumpSessionStats(Session &S) {
  if (ShowSizes)
    outs() << "Total size of all atoms before pruning: " << S.SizeBeforePruning
           << "\nTotal size of all atoms after fixups: " << S.SizeAfterFixups
           << "\n";
}

static Expected<JITEvaluatedSymbol> getMainEntryPoint(Session &S) {
  return S.ES.lookup(S.JDSearchOrder, EntryPointName);
}

Expected<int> runEntryPoint(Session &S, JITEvaluatedSymbol EntryPoint) {
  assert(EntryPoint.getAddress() && "Entry point address should not be null");

  constexpr const char *JITProgramName = "<llvm-jitlink jit'd code>";
  auto PNStorage = llvm::make_unique<char[]>(strlen(JITProgramName) + 1);
  strcpy(PNStorage.get(), JITProgramName);

  std::vector<const char *> EntryPointArgs;
  EntryPointArgs.push_back(PNStorage.get());
  EntryPointArgs.push_back(nullptr);

  using MainTy = int (*)(int, const char *[]);
  MainTy EntryPointPtr = reinterpret_cast<MainTy>(EntryPoint.getAddress());

  return EntryPointPtr(EntryPointArgs.size() - 1, EntryPointArgs.data());
}

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);

  InitializeAllTargetInfos();
  InitializeAllTargetMCs();
  InitializeAllDisassemblers();

  cl::ParseCommandLineOptions(argc, argv, "llvm jitlink tool");
  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  Session S(getFirstFileTriple());

  setEntryPointNameIfNotProvided(S);

  if (!NoProcessSymbols)
    ExitOnErr(loadProcessSymbols(S));
  ExitOnErr(loadDylibs());

  ExitOnErr(loadObjects(S));

  auto EntryPoint = ExitOnErr(getMainEntryPoint(S));

  if (ShowAddrs)
    S.dumpSessionInfo(outs());

  ExitOnErr(runChecks(S));

  dumpSessionStats(S);

  if (NoExec)
    return 0;

  return ExitOnErr(runEntryPoint(S, EntryPoint));
}
