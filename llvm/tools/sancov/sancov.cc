//===-- sancov.cc --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a command-line tool for reading and analyzing sanitizer
// coverage.
//===----------------------------------------------------------------------===//
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SpecialCaseList.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <set>
#include <stdio.h>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

// --------- COMMAND LINE FLAGS ---------

enum ActionType {
  PrintAction,
  PrintCovPointsAction,
  CoveredFunctionsAction,
  NotCoveredFunctionsAction,
  HtmlReportAction,
  StatsAction
};

cl::opt<ActionType> Action(
    cl::desc("Action (required)"), cl::Required,
    cl::values(clEnumValN(PrintAction, "print", "Print coverage addresses"),
               clEnumValN(PrintCovPointsAction, "print-coverage-pcs",
                          "Print coverage instrumentation points addresses."),
               clEnumValN(CoveredFunctionsAction, "covered-functions",
                          "Print all covered funcions."),
               clEnumValN(NotCoveredFunctionsAction, "not-covered-functions",
                          "Print all not covered funcions."),
               clEnumValN(HtmlReportAction, "html-report",
                          "Print HTML coverage report."),
               clEnumValN(StatsAction, "print-coverage-stats",
                          "Print coverage statistics."),
               clEnumValEnd));

static cl::list<std::string>
    ClInputFiles(cl::Positional, cl::OneOrMore,
                 cl::desc("(<binary file>|<.sancov file>)..."));

static cl::opt<bool> ClDemangle("demangle", cl::init(true),
                                cl::desc("Print demangled function name."));

static cl::opt<std::string> ClStripPathPrefix(
    "strip_path_prefix", cl::init(""),
    cl::desc("Strip this prefix from file paths in reports."));

static cl::opt<std::string>
    ClBlacklist("blacklist", cl::init(""),
                cl::desc("Blacklist file (sanitizer blacklist format)."));

static cl::opt<bool> ClUseDefaultBlacklist(
    "use_default_blacklist", cl::init(true), cl::Hidden,
    cl::desc("Controls if default blacklist should be used."));

static const char *const DefaultBlacklistStr = "fun:__sanitizer_.*\n"
                                               "src:/usr/include/.*\n"
                                               "src:.*/libc\\+\\+/.*\n";

// --------- FORMAT SPECIFICATION ---------

struct FileHeader {
  uint32_t Bitness;
  uint32_t Magic;
};

static const uint32_t BinCoverageMagic = 0xC0BFFFFF;
static const uint32_t Bitness32 = 0xFFFFFF32;
static const uint32_t Bitness64 = 0xFFFFFF64;

// --------- ERROR HANDLING ---------

static void Fail(const llvm::Twine &E) {
  errs() << "Error: " << E << "\n";
  exit(1);
}

static void FailIfError(std::error_code Error) {
  if (!Error)
    return;
  errs() << "Error: " << Error.message() << "(" << Error.value() << ")\n";
  exit(1);
}

template <typename T> static void FailIfError(const ErrorOr<T> &E) {
  FailIfError(E.getError());
}

static void FailIfError(Error Err) {
  if (Err) {
    logAllUnhandledErrors(std::move(Err), errs(), "Error: ");
    exit(1);
  }
}

template <typename T> static void FailIfError(Expected<T> &E) {
  FailIfError(E.takeError());
}

static void FailIfNotEmpty(const llvm::Twine &E) {
  if (E.str().empty())
    return;
  Fail(E);
}

template <typename T>
static void FailIfEmpty(const std::unique_ptr<T> &Ptr,
                        const std::string &Message) {
  if (Ptr.get())
    return;
  Fail(Message);
}

// ---------

// Produces std::map<K, std::vector<E>> grouping input
// elements by FuncTy result.
template <class RangeTy, class FuncTy>
static inline auto group_by(const RangeTy &R, FuncTy F)
    -> std::map<typename std::decay<decltype(F(*R.begin()))>::type,
                std::vector<typename std::decay<decltype(*R.begin())>::type>> {
  std::map<typename std::decay<decltype(F(*R.begin()))>::type,
           std::vector<typename std::decay<decltype(*R.begin())>::type>>
      Result;
  for (const auto &E : R) {
    Result[F(E)].push_back(E);
  }
  return Result;
}

template <typename T>
static void readInts(const char *Start, const char *End,
                     std::set<uint64_t> *Ints) {
  const T *S = reinterpret_cast<const T *>(Start);
  const T *E = reinterpret_cast<const T *>(End);
  std::copy(S, E, std::inserter(*Ints, Ints->end()));
}

struct FileLoc {
  bool operator<(const FileLoc &RHS) const {
    return std::tie(FileName, Line) < std::tie(RHS.FileName, RHS.Line);
  }

  std::string FileName;
  uint32_t Line;
};

struct FileFn {
  bool operator<(const FileFn &RHS) const {
    return std::tie(FileName, FunctionName) <
           std::tie(RHS.FileName, RHS.FunctionName);
  }

  std::string FileName;
  std::string FunctionName;
};

struct FnLoc {
  bool operator<(const FnLoc &RHS) const {
    return std::tie(Loc, FunctionName) < std::tie(RHS.Loc, RHS.FunctionName);
  }

  FileLoc Loc;
  std::string FunctionName;
};

std::string stripPathPrefix(std::string Path) {
  if (ClStripPathPrefix.empty())
    return Path;
  size_t Pos = Path.find(ClStripPathPrefix);
  if (Pos == std::string::npos)
    return Path;
  return Path.substr(Pos + ClStripPathPrefix.size());
}

static std::unique_ptr<symbolize::LLVMSymbolizer> createSymbolizer() {
  symbolize::LLVMSymbolizer::Options SymbolizerOptions;
  SymbolizerOptions.Demangle = ClDemangle;
  SymbolizerOptions.UseSymbolTable = true;
  return std::unique_ptr<symbolize::LLVMSymbolizer>(
      new symbolize::LLVMSymbolizer(SymbolizerOptions));
}

// A DILineInfo with address.
struct AddrInfo : public DILineInfo {
  uint64_t Addr;

  AddrInfo(const DILineInfo &DI, uint64_t Addr) : DILineInfo(DI), Addr(Addr) {
    FileName = normalizeFilename(FileName);
  }

private:
  static std::string normalizeFilename(const std::string &FileName) {
    SmallString<256> S(FileName);
    sys::path::remove_dots(S, /* remove_dot_dot */ true);
    return S.str().str();
  }
};

class Blacklists {
public:
  Blacklists()
      : DefaultBlacklist(createDefaultBlacklist()),
        UserBlacklist(createUserBlacklist()) {}

  // AddrInfo contains normalized filename. It is important to check it rather
  // than DILineInfo.
  bool isBlacklisted(const AddrInfo &AI) {
    if (DefaultBlacklist && DefaultBlacklist->inSection("fun", AI.FunctionName))
      return true;
    if (DefaultBlacklist && DefaultBlacklist->inSection("src", AI.FileName))
      return true;
    if (UserBlacklist && UserBlacklist->inSection("fun", AI.FunctionName))
      return true;
    if (UserBlacklist && UserBlacklist->inSection("src", AI.FileName))
      return true;
    return false;
  }

private:
  static std::unique_ptr<SpecialCaseList> createDefaultBlacklist() {
    if (!ClUseDefaultBlacklist)
      return std::unique_ptr<SpecialCaseList>();
    std::unique_ptr<MemoryBuffer> MB =
        MemoryBuffer::getMemBuffer(DefaultBlacklistStr);
    std::string Error;
    auto Blacklist = SpecialCaseList::create(MB.get(), Error);
    FailIfNotEmpty(Error);
    return Blacklist;
  }

  static std::unique_ptr<SpecialCaseList> createUserBlacklist() {
    if (ClBlacklist.empty())
      return std::unique_ptr<SpecialCaseList>();

    return SpecialCaseList::createOrDie({{ClBlacklist}});
  }
  std::unique_ptr<SpecialCaseList> DefaultBlacklist;
  std::unique_ptr<SpecialCaseList> UserBlacklist;
};

// Collect all debug info for given addresses.
static std::vector<AddrInfo> getAddrInfo(const std::string &ObjectFile,
                                         const std::set<uint64_t> &Addrs,
                                         bool InlinedCode) {
  std::vector<AddrInfo> Result;
  auto Symbolizer(createSymbolizer());
  Blacklists B;

  for (auto Addr : Addrs) {
    auto LineInfo = Symbolizer->symbolizeCode(ObjectFile, Addr);
    FailIfError(LineInfo);
    auto LineAddrInfo = AddrInfo(*LineInfo, Addr);
    if (B.isBlacklisted(LineAddrInfo))
      continue;
    Result.push_back(LineAddrInfo);
    if (InlinedCode) {
      auto InliningInfo = Symbolizer->symbolizeInlinedCode(ObjectFile, Addr);
      FailIfError(InliningInfo);
      for (uint32_t I = 0; I < InliningInfo->getNumberOfFrames(); ++I) {
        auto FrameInfo = InliningInfo->getFrame(I);
        auto FrameAddrInfo = AddrInfo(FrameInfo, Addr);
        if (B.isBlacklisted(FrameAddrInfo))
          continue;
        Result.push_back(FrameAddrInfo);
      }
    }
  }

  return Result;
}

// Locate __sanitizer_cov* function addresses that are used for coverage
// reporting.
static std::set<uint64_t>
findSanitizerCovFunctions(const object::ObjectFile &O) {
  std::set<uint64_t> Result;

  for (const object::SymbolRef &Symbol : O.symbols()) {
    Expected<uint64_t> AddressOrErr = Symbol.getAddress();
    FailIfError(errorToErrorCode(AddressOrErr.takeError()));

    Expected<StringRef> NameOrErr = Symbol.getName();
    FailIfError(errorToErrorCode(NameOrErr.takeError()));
    StringRef Name = NameOrErr.get();

    if (Name == "__sanitizer_cov" || Name == "__sanitizer_cov_with_check" ||
        Name == "__sanitizer_cov_trace_func_enter") {
      if (!(Symbol.getFlags() & object::BasicSymbolRef::SF_Undefined))
        Result.insert(AddressOrErr.get());
    }
  }

  return Result;
}

// Locate addresses of all coverage points in a file. Coverage point
// is defined as the 'address of instruction following __sanitizer_cov
// call - 1'.
static void getObjectCoveragePoints(const object::ObjectFile &O,
                                    std::set<uint64_t> *Addrs) {
  Triple TheTriple("unknown-unknown-unknown");
  TheTriple.setArch(Triple::ArchType(O.getArch()));
  auto TripleName = TheTriple.getTriple();

  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
  FailIfNotEmpty(Error);

  std::unique_ptr<const MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, "", ""));
  FailIfEmpty(STI, "no subtarget info for target " + TripleName);

  std::unique_ptr<const MCRegisterInfo> MRI(
      TheTarget->createMCRegInfo(TripleName));
  FailIfEmpty(MRI, "no register info for target " + TripleName);

  std::unique_ptr<const MCAsmInfo> AsmInfo(
      TheTarget->createMCAsmInfo(*MRI, TripleName));
  FailIfEmpty(AsmInfo, "no asm info for target " + TripleName);

  std::unique_ptr<const MCObjectFileInfo> MOFI(new MCObjectFileInfo);
  MCContext Ctx(AsmInfo.get(), MRI.get(), MOFI.get());
  std::unique_ptr<MCDisassembler> DisAsm(
      TheTarget->createMCDisassembler(*STI, Ctx));
  FailIfEmpty(DisAsm, "no disassembler info for target " + TripleName);

  std::unique_ptr<const MCInstrInfo> MII(TheTarget->createMCInstrInfo());
  FailIfEmpty(MII, "no instruction info for target " + TripleName);

  std::unique_ptr<const MCInstrAnalysis> MIA(
      TheTarget->createMCInstrAnalysis(MII.get()));
  FailIfEmpty(MIA, "no instruction analysis info for target " + TripleName);

  auto SanCovAddrs = findSanitizerCovFunctions(O);
  if (SanCovAddrs.empty())
    Fail("__sanitizer_cov* functions not found");

  for (object::SectionRef Section : O.sections()) {
    if (Section.isVirtual() || !Section.isText()) // llvm-objdump does the same.
      continue;
    uint64_t SectionAddr = Section.getAddress();
    uint64_t SectSize = Section.getSize();
    if (!SectSize)
      continue;

    StringRef BytesStr;
    FailIfError(Section.getContents(BytesStr));
    ArrayRef<uint8_t> Bytes(reinterpret_cast<const uint8_t *>(BytesStr.data()),
                            BytesStr.size());

    for (uint64_t Index = 0, Size = 0; Index < Section.getSize();
         Index += Size) {
      MCInst Inst;
      if (!DisAsm->getInstruction(Inst, Size, Bytes.slice(Index),
                                  SectionAddr + Index, nulls(), nulls())) {
        if (Size == 0)
          Size = 1;
        continue;
      }
      uint64_t Addr = Index + SectionAddr;
      // Sanitizer coverage uses the address of the next instruction - 1.
      uint64_t CovPoint = Addr + Size - 1;
      uint64_t Target;
      if (MIA->isCall(Inst) &&
          MIA->evaluateBranch(Inst, SectionAddr + Index, Size, Target) &&
          SanCovAddrs.find(Target) != SanCovAddrs.end())
        Addrs->insert(CovPoint);
    }
  }
}

static void
visitObjectFiles(const object::Archive &A,
                 function_ref<void(const object::ObjectFile &)> Fn) {
  Error Err;
  for (auto &C : A.children(Err)) {
    Expected<std::unique_ptr<object::Binary>> ChildOrErr = C.getAsBinary();
    FailIfError(errorToErrorCode(ChildOrErr.takeError()));
    if (auto *O = dyn_cast<object::ObjectFile>(&*ChildOrErr.get()))
      Fn(*O);
    else
      FailIfError(object::object_error::invalid_file_type);
  }
  FailIfError(std::move(Err));
}

static void
visitObjectFiles(const std::string &FileName,
                 function_ref<void(const object::ObjectFile &)> Fn) {
  Expected<object::OwningBinary<object::Binary>> BinaryOrErr =
      object::createBinary(FileName);
  if (!BinaryOrErr)
    FailIfError(errorToErrorCode(BinaryOrErr.takeError()));

  object::Binary &Binary = *BinaryOrErr.get().getBinary();
  if (object::Archive *A = dyn_cast<object::Archive>(&Binary))
    visitObjectFiles(*A, Fn);
  else if (object::ObjectFile *O = dyn_cast<object::ObjectFile>(&Binary))
    Fn(*O);
  else
    FailIfError(object::object_error::invalid_file_type);
}

std::set<uint64_t> findSanitizerCovFunctions(const std::string &FileName) {
  std::set<uint64_t> Result;
  visitObjectFiles(FileName, [&](const object::ObjectFile &O) {
    auto Addrs = findSanitizerCovFunctions(O);
    Result.insert(Addrs.begin(), Addrs.end());
  });
  return Result;
}

// Locate addresses of all coverage points in a file. Coverage point
// is defined as the 'address of instruction following __sanitizer_cov
// call - 1'.
std::set<uint64_t> getCoveragePoints(const std::string &FileName) {
  std::set<uint64_t> Result;
  visitObjectFiles(FileName, [&](const object::ObjectFile &O) {
    getObjectCoveragePoints(O, &Result);
  });
  return Result;
}

static void printCovPoints(const std::string &ObjFile, raw_ostream &OS) {
  for (uint64_t Addr : getCoveragePoints(ObjFile)) {
    OS << "0x";
    OS.write_hex(Addr);
    OS << "\n";
  }
}

static std::string escapeHtml(const std::string &S) {
  std::string Result;
  Result.reserve(S.size());
  for (char Ch : S) {
    switch (Ch) {
    case '&':
      Result.append("&amp;");
      break;
    case '\'':
      Result.append("&apos;");
      break;
    case '"':
      Result.append("&quot;");
      break;
    case '<':
      Result.append("&lt;");
      break;
    case '>':
      Result.append("&gt;");
      break;
    default:
      Result.push_back(Ch);
      break;
    }
  }
  return Result;
}

// Adds leading zeroes wrapped in 'lz' style.
// Leading zeroes help locate 000% coverage.
static std::string formatHtmlPct(size_t Pct) {
  Pct = std::max(std::size_t{0}, std::min(std::size_t{100}, Pct));

  std::string Num = std::to_string(Pct);
  std::string Zeroes(3 - Num.size(), '0');
  if (!Zeroes.empty())
    Zeroes = "<span class='lz'>" + Zeroes + "</span>";

  return Zeroes + Num;
}

static std::string anchorName(const std::string &Anchor) {
  llvm::MD5 Hasher;
  llvm::MD5::MD5Result Hash;
  Hasher.update(Anchor);
  Hasher.final(Hash);

  SmallString<32> HexString;
  llvm::MD5::stringifyResult(Hash, HexString);
  return HexString.str().str();
}

static ErrorOr<bool> isCoverageFile(const std::string &FileName) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(FileName);
  if (!BufOrErr) {
    errs() << "Warning: " << BufOrErr.getError().message() << "("
           << BufOrErr.getError().value()
           << "), filename: " << llvm::sys::path::filename(FileName) << "\n";
    return BufOrErr.getError();
  }
  std::unique_ptr<MemoryBuffer> Buf = std::move(BufOrErr.get());
  if (Buf->getBufferSize() < 8) {
    return false;
  }
  const FileHeader *Header =
      reinterpret_cast<const FileHeader *>(Buf->getBufferStart());
  return Header->Magic == BinCoverageMagic;
}

struct CoverageStats {
  CoverageStats() : AllPoints(0), CovPoints(0), AllFns(0), CovFns(0) {}

  size_t AllPoints;
  size_t CovPoints;
  size_t AllFns;
  size_t CovFns;
};

static raw_ostream &operator<<(raw_ostream &OS, const CoverageStats &Stats) {
  OS << "all-edges: " << Stats.AllPoints << "\n";
  OS << "cov-edges: " << Stats.CovPoints << "\n";
  OS << "all-functions: " << Stats.AllFns << "\n";
  OS << "cov-functions: " << Stats.CovFns << "\n";
  return OS;
}

class CoverageData {
public:
  // Read single file coverage data.
  static ErrorOr<std::unique_ptr<CoverageData>>
  read(const std::string &FileName) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
        MemoryBuffer::getFile(FileName);
    if (!BufOrErr)
      return BufOrErr.getError();
    std::unique_ptr<MemoryBuffer> Buf = std::move(BufOrErr.get());
    if (Buf->getBufferSize() < 8) {
      errs() << "File too small (<8): " << Buf->getBufferSize();
      return make_error_code(errc::illegal_byte_sequence);
    }
    const FileHeader *Header =
        reinterpret_cast<const FileHeader *>(Buf->getBufferStart());

    if (Header->Magic != BinCoverageMagic) {
      errs() << "Wrong magic: " << Header->Magic;
      return make_error_code(errc::illegal_byte_sequence);
    }

    auto Addrs = llvm::make_unique<std::set<uint64_t>>();

    switch (Header->Bitness) {
    case Bitness64:
      readInts<uint64_t>(Buf->getBufferStart() + 8, Buf->getBufferEnd(),
                         Addrs.get());
      break;
    case Bitness32:
      readInts<uint32_t>(Buf->getBufferStart() + 8, Buf->getBufferEnd(),
                         Addrs.get());
      break;
    default:
      errs() << "Unsupported bitness: " << Header->Bitness;
      return make_error_code(errc::illegal_byte_sequence);
    }

    return std::unique_ptr<CoverageData>(new CoverageData(std::move(Addrs)));
  }

  // Merge multiple coverage data together.
  static std::unique_ptr<CoverageData>
  merge(const std::vector<std::unique_ptr<CoverageData>> &Covs) {
    auto Addrs = llvm::make_unique<std::set<uint64_t>>();

    for (const auto &Cov : Covs)
      Addrs->insert(Cov->Addrs->begin(), Cov->Addrs->end());

    return std::unique_ptr<CoverageData>(new CoverageData(std::move(Addrs)));
  }

  // Read list of files and merges their coverage info.
  static ErrorOr<std::unique_ptr<CoverageData>>
  readAndMerge(const std::vector<std::string> &FileNames) {
    std::vector<std::unique_ptr<CoverageData>> Covs;
    for (const auto &FileName : FileNames) {
      auto Cov = read(FileName);
      if (!Cov)
        return Cov.getError();
      Covs.push_back(std::move(Cov.get()));
    }
    return merge(Covs);
  }

  // Print coverage addresses.
  void printAddrs(raw_ostream &OS) {
    for (auto Addr : *Addrs) {
      OS << "0x";
      OS.write_hex(Addr);
      OS << "\n";
    }
  }

protected:
  explicit CoverageData(std::unique_ptr<std::set<uint64_t>> Addrs)
      : Addrs(std::move(Addrs)) {}

  friend class CoverageDataWithObjectFile;

  std::unique_ptr<std::set<uint64_t>> Addrs;
};

// Coverage data translated into source code line-level information.
// Fetches debug info in constructor and calculates various information per
// request.
class SourceCoverageData {
public:
  enum LineStatus {
    // coverage information for the line is not available.
    // default value in maps.
    UNKNOWN = 0,
    // the line is fully covered.
    COVERED = 1,
    // the line is fully uncovered.
    NOT_COVERED = 2,
    // some points in the line a covered, some are not.
    MIXED = 3
  };

  SourceCoverageData(std::string ObjectFile, const std::set<uint64_t> &Addrs)
      : AllCovPoints(getCoveragePoints(ObjectFile)) {
    if (!std::includes(AllCovPoints.begin(), AllCovPoints.end(), Addrs.begin(),
                       Addrs.end())) {
      Fail("Coverage points in binary and .sancov file do not match.");
    }

    AllAddrInfo = getAddrInfo(ObjectFile, AllCovPoints, true);
    CovAddrInfo = getAddrInfo(ObjectFile, Addrs, true);
  }

  // Compute number of coverage points hit/total in a file.
  // file_name -> <coverage, all_coverage>
  std::map<std::string, std::pair<size_t, size_t>> computeFileCoverage() {
    std::map<std::string, std::pair<size_t, size_t>> FileCoverage;
    auto AllCovPointsByFile =
        group_by(AllAddrInfo, [](const AddrInfo &AI) { return AI.FileName; });
    auto CovPointsByFile =
        group_by(CovAddrInfo, [](const AddrInfo &AI) { return AI.FileName; });

    for (const auto &P : AllCovPointsByFile) {
      const std::string &FileName = P.first;

      FileCoverage[FileName] =
          std::make_pair(CovPointsByFile[FileName].size(),
                         AllCovPointsByFile[FileName].size());
    }
    return FileCoverage;
  }

  // line_number -> line_status.
  typedef std::map<int, LineStatus> LineStatusMap;
  // file_name -> LineStatusMap
  typedef std::map<std::string, LineStatusMap> FileLineStatusMap;

  // fills in the {file_name -> {line_no -> status}} map.
  FileLineStatusMap computeLineStatusMap() {
    FileLineStatusMap StatusMap;

    auto AllLocs = group_by(AllAddrInfo, [](const AddrInfo &AI) {
      return FileLoc{AI.FileName, AI.Line};
    });
    auto CovLocs = group_by(CovAddrInfo, [](const AddrInfo &AI) {
      return FileLoc{AI.FileName, AI.Line};
    });

    for (const auto &P : AllLocs) {
      const FileLoc &Loc = P.first;
      auto I = CovLocs.find(Loc);

      if (I == CovLocs.end()) {
        StatusMap[Loc.FileName][Loc.Line] = NOT_COVERED;
      } else {
        StatusMap[Loc.FileName][Loc.Line] =
            (I->second.size() == P.second.size()) ? COVERED : MIXED;
      }
    }
    return StatusMap;
  }

  std::set<FileFn> computeAllFunctions() const {
    std::set<FileFn> Fns;
    for (const auto &AI : AllAddrInfo) {
      Fns.insert(FileFn{AI.FileName, AI.FunctionName});
    }
    return Fns;
  }

  std::set<FileFn> computeCoveredFunctions() const {
    std::set<FileFn> Fns;
    auto CovFns = group_by(CovAddrInfo, [](const AddrInfo &AI) {
      return FileFn{AI.FileName, AI.FunctionName};
    });

    for (const auto &P : CovFns) {
      Fns.insert(P.first);
    }
    return Fns;
  }

  std::set<FileFn> computeNotCoveredFunctions() const {
    std::set<FileFn> Fns;

    auto AllFns = group_by(AllAddrInfo, [](const AddrInfo &AI) {
      return FileFn{AI.FileName, AI.FunctionName};
    });
    auto CovFns = group_by(CovAddrInfo, [](const AddrInfo &AI) {
      return FileFn{AI.FileName, AI.FunctionName};
    });

    for (const auto &P : AllFns) {
      if (CovFns.find(P.first) == CovFns.end()) {
        Fns.insert(P.first);
      }
    }
    return Fns;
  }

  // Compute % coverage for each function.
  std::map<FileFn, int> computeFunctionsCoverage() const {
    std::map<FileFn, int> FnCoverage;
    auto AllFns = group_by(AllAddrInfo, [](const AddrInfo &AI) {
      return FileFn{AI.FileName, AI.FunctionName};
    });

    auto CovFns = group_by(CovAddrInfo, [](const AddrInfo &AI) {
      return FileFn{AI.FileName, AI.FunctionName};
    });

    for (const auto &P : AllFns) {
      FileFn F = P.first;
      FnCoverage[F] = CovFns[F].size() * 100 / P.second.size();
    }

    return FnCoverage;
  }

  typedef std::map<FileLoc, std::set<std::string>> FunctionLocs;
  // finds first line number in a file for each function.
  FunctionLocs resolveFunctions(const std::set<FileFn> &Fns) const {
    std::vector<AddrInfo> FnAddrs;
    for (const auto &AI : AllAddrInfo) {
      if (Fns.find(FileFn{AI.FileName, AI.FunctionName}) != Fns.end())
        FnAddrs.push_back(AI);
    }

    auto GroupedAddrs = group_by(FnAddrs, [](const AddrInfo &AI) {
      return FnLoc{FileLoc{AI.FileName, AI.Line}, AI.FunctionName};
    });

    FunctionLocs Result;
    std::string LastFileName;
    std::set<std::string> ProcessedFunctions;

    for (const auto &P : GroupedAddrs) {
      const FnLoc &Loc = P.first;
      std::string FileName = Loc.Loc.FileName;
      std::string FunctionName = Loc.FunctionName;

      if (LastFileName != FileName)
        ProcessedFunctions.clear();
      LastFileName = FileName;

      if (!ProcessedFunctions.insert(FunctionName).second)
        continue;

      auto FLoc = FileLoc{FileName, Loc.Loc.Line};
      Result[FLoc].insert(FunctionName);
    }
    return Result;
  }

  std::set<std::string> files() const {
    std::set<std::string> Files;
    for (const auto &AI : AllAddrInfo) {
      Files.insert(AI.FileName);
    }
    return Files;
  }

  void collectStats(CoverageStats *Stats) const {
    Stats->AllPoints += AllCovPoints.size();
    Stats->AllFns += computeAllFunctions().size();
    Stats->CovFns += computeCoveredFunctions().size();
  }

private:
  const std::set<uint64_t> AllCovPoints;

  std::vector<AddrInfo> AllAddrInfo;
  std::vector<AddrInfo> CovAddrInfo;
};

static void printFunctionLocs(const SourceCoverageData::FunctionLocs &FnLocs,
                              raw_ostream &OS) {
  for (const auto &Fns : FnLocs) {
    for (const auto &Fn : Fns.second) {
      OS << stripPathPrefix(Fns.first.FileName) << ":" << Fns.first.Line << " "
         << Fn << "\n";
    }
  }
}

// Holder for coverage data + filename of corresponding object file.
class CoverageDataWithObjectFile : public CoverageData {
public:
  static ErrorOr<std::unique_ptr<CoverageDataWithObjectFile>>
  readAndMerge(const std::string &ObjectFile,
               const std::vector<std::string> &FileNames) {
    auto MergedDataOrError = CoverageData::readAndMerge(FileNames);
    if (!MergedDataOrError)
      return MergedDataOrError.getError();
    return std::unique_ptr<CoverageDataWithObjectFile>(
        new CoverageDataWithObjectFile(ObjectFile,
                                       std::move(MergedDataOrError.get())));
  }

  std::string object_file() const { return ObjectFile; }

  // Print list of covered functions.
  // Line format: <file_name>:<line> <function_name>
  void printCoveredFunctions(raw_ostream &OS) const {
    SourceCoverageData SCovData(ObjectFile, *Addrs);
    auto CoveredFns = SCovData.computeCoveredFunctions();
    printFunctionLocs(SCovData.resolveFunctions(CoveredFns), OS);
  }

  // Print list of not covered functions.
  // Line format: <file_name>:<line> <function_name>
  void printNotCoveredFunctions(raw_ostream &OS) const {
    SourceCoverageData SCovData(ObjectFile, *Addrs);
    auto NotCoveredFns = SCovData.computeNotCoveredFunctions();
    printFunctionLocs(SCovData.resolveFunctions(NotCoveredFns), OS);
  }

  void printReport(raw_ostream &OS) const {
    SourceCoverageData SCovData(ObjectFile, *Addrs);
    auto LineStatusMap = SCovData.computeLineStatusMap();

    std::set<FileFn> AllFns = SCovData.computeAllFunctions();
    // file_loc -> set[function_name]
    auto AllFnsByLoc = SCovData.resolveFunctions(AllFns);
    auto FileCoverage = SCovData.computeFileCoverage();

    auto FnCoverage = SCovData.computeFunctionsCoverage();
    auto FnCoverageByFile =
        group_by(FnCoverage, [](const std::pair<FileFn, int> &FileFn) {
          return FileFn.first.FileName;
        });

    // TOC

    size_t NotCoveredFilesCount = 0;
    std::set<std::string> Files = SCovData.files();

    // Covered Files.
    OS << "<details open><summary>Touched Files</summary>\n";
    OS << "<table>\n";
    OS << "<tr><th>File</th><th>Coverage %</th>";
    OS << "<th>Hit (Total) Fns</th></tr>\n";
    for (const auto &FileName : Files) {
      std::pair<size_t, size_t> FC = FileCoverage[FileName];
      if (FC.first == 0) {
        NotCoveredFilesCount++;
        continue;
      }
      size_t CovPct = FC.second == 0 ? 100 : 100 * FC.first / FC.second;

      OS << "<tr><td><a href=\"#" << anchorName(FileName) << "\">"
         << stripPathPrefix(FileName) << "</a></td>"
         << "<td>" << formatHtmlPct(CovPct) << "%</td>"
         << "<td>" << FC.first << " (" << FC.second << ")"
         << "</tr>\n";
    }
    OS << "</table>\n";
    OS << "</details>\n";

    // Not covered files.
    if (NotCoveredFilesCount) {
      OS << "<details><summary>Not Touched Files</summary>\n";
      OS << "<table>\n";
      for (const auto &FileName : Files) {
        std::pair<size_t, size_t> FC = FileCoverage[FileName];
        if (FC.first == 0)
          OS << "<tr><td>" << stripPathPrefix(FileName) << "</td>\n";
      }
      OS << "</table>\n";
      OS << "</details>\n";
    } else {
      OS << "<p>Congratulations! All source files are touched.</p>\n";
    }

    // Source
    for (const auto &FileName : Files) {
      std::pair<size_t, size_t> FC = FileCoverage[FileName];
      if (FC.first == 0)
        continue;
      OS << "<a name=\"" << anchorName(FileName) << "\"></a>\n";
      OS << "<h2>" << stripPathPrefix(FileName) << "</h2>\n";
      OS << "<details open><summary>Function Coverage</summary>";
      OS << "<div class='fnlist'>\n";

      auto &FileFnCoverage = FnCoverageByFile[FileName];

      for (const auto &P : FileFnCoverage) {
        std::string FunctionName = P.first.FunctionName;

        OS << "<div class='fn' style='order: " << P.second << "'>";
        OS << "<span class='pct'>" << formatHtmlPct(P.second)
           << "%</span>&nbsp;";
        OS << "<span class='name'><a href=\"#"
           << anchorName(FileName + "::" + FunctionName) << "\">";
        OS << escapeHtml(FunctionName) << "</a></span>";
        OS << "</div>\n";
      }
      OS << "</div></details>\n";

      ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
          MemoryBuffer::getFile(FileName);
      if (!BufOrErr) {
        OS << "Error reading file: " << FileName << " : "
           << BufOrErr.getError().message() << "("
           << BufOrErr.getError().value() << ")\n";
        continue;
      }

      OS << "<pre>\n";
      const auto &LineStatuses = LineStatusMap[FileName];
      for (line_iterator I = line_iterator(*BufOrErr.get(), false);
           !I.is_at_eof(); ++I) {
        uint32_t Line = I.line_number();
        { // generate anchors (if any);
          FileLoc Loc = FileLoc{FileName, Line};
          auto It = AllFnsByLoc.find(Loc);
          if (It != AllFnsByLoc.end()) {
            for (const std::string &Fn : It->second) {
              OS << "<a name=\"" << anchorName(FileName + "::" + Fn)
                 << "\"></a>";
            };
          }
        }

        OS << "<span ";
        auto LIT = LineStatuses.find(I.line_number());
        auto Status = (LIT != LineStatuses.end()) ? LIT->second
                                                  : SourceCoverageData::UNKNOWN;
        switch (Status) {
        case SourceCoverageData::UNKNOWN:
          OS << "class=unknown";
          break;
        case SourceCoverageData::COVERED:
          OS << "class=covered";
          break;
        case SourceCoverageData::NOT_COVERED:
          OS << "class=notcovered";
          break;
        case SourceCoverageData::MIXED:
          OS << "class=mixed";
          break;
        }
        OS << ">";
        OS << escapeHtml(*I) << "</span>\n";
      }
      OS << "</pre>\n";
    }
  }

  void collectStats(CoverageStats *Stats) const {
    Stats->CovPoints += Addrs->size();

    SourceCoverageData SCovData(ObjectFile, *Addrs);
    SCovData.collectStats(Stats);
  }

private:
  CoverageDataWithObjectFile(std::string ObjectFile,
                             std::unique_ptr<CoverageData> Coverage)
      : CoverageData(std::move(Coverage->Addrs)),
        ObjectFile(std::move(ObjectFile)) {}
  const std::string ObjectFile;
};

// Multiple coverage files data organized by object file.
class CoverageDataSet {
public:
  static ErrorOr<std::unique_ptr<CoverageDataSet>>
  readCmdArguments(std::vector<std::string> FileNames) {
    // Short name => file name.
    std::map<std::string, std::string> ObjFiles;
    std::string FirstObjFile;
    std::set<std::string> CovFiles;

    // Partition input values into coverage/object files.
    for (const auto &FileName : FileNames) {
      auto ErrorOrIsCoverage = isCoverageFile(FileName);
      if (!ErrorOrIsCoverage)
        continue;
      if (ErrorOrIsCoverage.get()) {
        CovFiles.insert(FileName);
      } else {
        auto ShortFileName = llvm::sys::path::filename(FileName);
        if (ObjFiles.find(ShortFileName) != ObjFiles.end()) {
          Fail("Duplicate binary file with a short name: " + ShortFileName);
        }

        ObjFiles[ShortFileName] = FileName;
        if (FirstObjFile.empty())
          FirstObjFile = FileName;
      }
    }

    Regex SancovRegex("(.*)\\.[0-9]+\\.sancov");
    SmallVector<StringRef, 2> Components;

    // Object file => list of corresponding coverage file names.
    auto CoverageByObjFile = group_by(CovFiles, [&](std::string FileName) {
      auto ShortFileName = llvm::sys::path::filename(FileName);
      auto Ok = SancovRegex.match(ShortFileName, &Components);
      if (!Ok) {
        Fail("Can't match coverage file name against "
             "<module_name>.<pid>.sancov pattern: " +
             FileName);
      }

      auto Iter = ObjFiles.find(Components[1]);
      if (Iter == ObjFiles.end()) {
        Fail("Object file for coverage not found: " + FileName);
      }
      return Iter->second;
    });

    // Read coverage.
    std::vector<std::unique_ptr<CoverageDataWithObjectFile>> MergedCoverage;
    for (const auto &Pair : CoverageByObjFile) {
      if (findSanitizerCovFunctions(Pair.first).empty()) {
        for (const auto &FileName : Pair.second) {
          CovFiles.erase(FileName);
        }

        errs()
            << "Ignoring " << Pair.first
            << " and its coverage because  __sanitizer_cov* functions were not "
               "found.\n";
        continue;
      }

      auto DataOrError =
          CoverageDataWithObjectFile::readAndMerge(Pair.first, Pair.second);
      FailIfError(DataOrError);
      MergedCoverage.push_back(std::move(DataOrError.get()));
    }

    return std::unique_ptr<CoverageDataSet>(
        new CoverageDataSet(FirstObjFile, &MergedCoverage, CovFiles));
  }

  void printCoveredFunctions(raw_ostream &OS) const {
    for (const auto &Cov : Coverage) {
      Cov->printCoveredFunctions(OS);
    }
  }

  void printNotCoveredFunctions(raw_ostream &OS) const {
    for (const auto &Cov : Coverage) {
      Cov->printNotCoveredFunctions(OS);
    }
  }

  void printStats(raw_ostream &OS) const {
    CoverageStats Stats;
    for (const auto &Cov : Coverage) {
      Cov->collectStats(&Stats);
    }
    OS << Stats;
  }

  void printReport(raw_ostream &OS) const {
    auto Title =
        (llvm::sys::path::filename(MainObjFile) + " Coverage Report").str();

    OS << "<html>\n";
    OS << "<head>\n";

    // Stylesheet
    OS << "<style>\n";
    OS << ".covered { background: #7F7; }\n";
    OS << ".notcovered { background: #F77; }\n";
    OS << ".mixed { background: #FF7; }\n";
    OS << "summary { font-weight: bold; }\n";
    OS << "details > summary + * { margin-left: 1em; }\n";
    OS << ".fnlist { display: flex; flex-flow: column nowrap; }\n";
    OS << ".fn { display: flex; flex-flow: row nowrap; }\n";
    OS << ".pct { width: 3em; text-align: right; margin-right: 1em; }\n";
    OS << ".name { flex: 2; }\n";
    OS << ".lz { color: lightgray; }\n";
    OS << "</style>\n";
    OS << "<title>" << Title << "</title>\n";
    OS << "</head>\n";
    OS << "<body>\n";

    // Title
    OS << "<h1>" << Title << "</h1>\n";

    // Modules TOC.
    if (Coverage.size() > 1) {
      for (const auto &CovData : Coverage) {
        OS << "<li><a href=\"#module_" << anchorName(CovData->object_file())
           << "\">" << llvm::sys::path::filename(CovData->object_file())
           << "</a></li>\n";
      }
    }

    for (const auto &CovData : Coverage) {
      if (Coverage.size() > 1) {
        OS << "<h2>" << llvm::sys::path::filename(CovData->object_file())
           << "</h2>\n";
      }
      OS << "<a name=\"module_" << anchorName(CovData->object_file())
         << "\"></a>\n";
      CovData->printReport(OS);
    }

    // About
    OS << "<details><summary>About</summary>\n";
    OS << "Coverage files:<ul>";
    for (const auto &InputFile : CoverageFiles) {
      llvm::sys::fs::file_status Status;
      llvm::sys::fs::status(InputFile, Status);
      OS << "<li>" << stripPathPrefix(InputFile) << " ("
         << Status.getLastModificationTime().str() << ")</li>\n";
    }
    OS << "</ul></details>\n";

    OS << "</body>\n";
    OS << "</html>\n";
  }

  bool empty() const { return Coverage.empty(); }

private:
  explicit CoverageDataSet(
      const std::string &MainObjFile,
      std::vector<std::unique_ptr<CoverageDataWithObjectFile>> *Data,
      const std::set<std::string> &CoverageFiles)
      : MainObjFile(MainObjFile), CoverageFiles(CoverageFiles) {
    Data->swap(this->Coverage);
  }

  const std::string MainObjFile;
  std::vector<std::unique_ptr<CoverageDataWithObjectFile>> Coverage;
  const std::set<std::string> CoverageFiles;
};

} // namespace

int main(int argc, char **argv) {
  // Print stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();

  cl::ParseCommandLineOptions(argc, argv, "Sanitizer Coverage Processing Tool");

  // -print doesn't need object files.
  if (Action == PrintAction) {
    auto CovData = CoverageData::readAndMerge(ClInputFiles);
    FailIfError(CovData);
    CovData.get()->printAddrs(outs());
    return 0;
  } else if (Action == PrintCovPointsAction) {
    // -print-coverage-points doesn't need coverage files.
    for (const std::string &ObjFile : ClInputFiles) {
      printCovPoints(ObjFile, outs());
    }
    return 0;
  }

  auto CovDataSet = CoverageDataSet::readCmdArguments(ClInputFiles);
  FailIfError(CovDataSet);

  if (CovDataSet.get()->empty()) {
    Fail("No coverage files specified.");
  }

  switch (Action) {
  case CoveredFunctionsAction: {
    CovDataSet.get()->printCoveredFunctions(outs());
    return 0;
  }
  case NotCoveredFunctionsAction: {
    CovDataSet.get()->printNotCoveredFunctions(outs());
    return 0;
  }
  case HtmlReportAction: {
    CovDataSet.get()->printReport(outs());
    return 0;
  }
  case StatsAction: {
    CovDataSet.get()->printStats(outs());
    return 0;
  }
  case PrintAction:
  case PrintCovPointsAction:
    llvm_unreachable("unsupported action");
  }
}
