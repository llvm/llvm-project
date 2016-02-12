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
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
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
  CoveredFunctionsAction,
  NotCoveredFunctionsAction,
  HtmlReportAction
};

cl::opt<ActionType> Action(
    cl::desc("Action (required)"), cl::Required,
    cl::values(clEnumValN(PrintAction, "print", "Print coverage addresses"),
               clEnumValN(CoveredFunctionsAction, "covered-functions",
                          "Print all covered funcions."),
               clEnumValN(NotCoveredFunctionsAction, "not-covered-functions",
                          "Print all not covered funcions."),
               clEnumValN(HtmlReportAction, "html-report",
                          "Print HTML coverage report."),
               clEnumValEnd));

static cl::list<std::string>
    ClInputFiles(cl::Positional, cl::OneOrMore,
                 cl::desc("(<binary file>|<.sancov file>)..."));

static cl::opt<bool>
    ClDemangle("demangle", cl::init(true),
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

static const char *const DefaultBlacklist = "fun:__sanitizer_*";

// --------- FORMAT SPECIFICATION ---------

struct FileHeader {
  uint32_t Bitness;
  uint32_t Magic;
};

static const uint32_t BinCoverageMagic = 0xC0BFFFFF;
static const uint32_t Bitness32 = 0xFFFFFF32;
static const uint32_t Bitness64 = 0xFFFFFF64;

// ---------

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

struct FunctionLoc {
  bool operator<(const FunctionLoc &RHS) const {
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

// Compute [FileLoc -> FunctionName] map for given addresses.
static std::map<FileLoc, std::string>
computeFunctionsMap(std::string ObjectFile, const std::set<uint64_t> &Addrs) {
  std::map<FileLoc, std::string> Fns;

  auto Symbolizer(createSymbolizer());

  // Fill in Fns map.
  for (auto Addr : Addrs) {
    auto InliningInfo = Symbolizer->symbolizeInlinedCode(ObjectFile, Addr);
    FailIfError(InliningInfo);
    for (uint32_t I = 0; I < InliningInfo->getNumberOfFrames(); ++I) {
      auto FrameInfo = InliningInfo->getFrame(I);
      SmallString<256> FileName(FrameInfo.FileName);
      sys::path::remove_dots(FileName, /* remove_dot_dot */ true);
      FileLoc Loc = {FileName.str(), FrameInfo.Line};
      Fns[Loc] = FrameInfo.FunctionName;
    }
  }

  return Fns;
}

// Compute functions for given addresses. It keeps only the first
// occurence of a function within a file.
std::set<FunctionLoc> computeFunctionLocs(std::string ObjectFile,
                                          const std::set<uint64_t> &Addrs) {
  std::map<FileLoc, std::string> Fns = computeFunctionsMap(ObjectFile, Addrs);

  std::set<FunctionLoc> Result;
  std::string LastFileName;
  std::set<std::string> ProcessedFunctions;

  for (const auto &P : Fns) {
    std::string FileName = P.first.FileName;
    std::string FunctionName = P.second;

    if (LastFileName != FileName)
      ProcessedFunctions.clear();
    LastFileName = FileName;

    if (!ProcessedFunctions.insert(FunctionName).second)
      continue;

    Result.insert(FunctionLoc{P.first, P.second});
  }

  return Result;
}

// Locate __sanitizer_cov* function addresses that are used for coverage
// reporting.
static std::set<uint64_t>
findSanitizerCovFunctions(const object::ObjectFile &O) {
  std::set<uint64_t> Result;

  for (const object::SymbolRef &Symbol : O.symbols()) {
    ErrorOr<uint64_t> AddressOrErr = Symbol.getAddress();
    FailIfError(AddressOrErr);

    ErrorOr<StringRef> NameOrErr = Symbol.getName();
    FailIfError(NameOrErr);
    StringRef Name = NameOrErr.get();

    if (Name == "__sanitizer_cov" || Name == "__sanitizer_cov_with_check" ||
        Name == "__sanitizer_cov_trace_func_enter") {
      Result.insert(AddressOrErr.get());
    }
  }

  if (Result.empty())
    Fail("__sanitizer_cov* functions not found");

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

  for (const auto Section : O.sections()) {
    if (Section.isVirtual() || !Section.isText()) // llvm-objdump does the same.
      continue;
    uint64_t SectionAddr = Section.getAddress();
    uint64_t SectSize = Section.getSize();
    if (!SectSize)
      continue;

    StringRef SectionName;
    FailIfError(Section.getName(SectionName));

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
      uint64_t Target;
      if (MIA->isCall(Inst) &&
          MIA->evaluateBranch(Inst, SectionAddr + Index, Size, Target)) {
        if (SanCovAddrs.find(Target) != SanCovAddrs.end()) {
          // Sanitizer coverage uses the address of the next instruction - 1.
          Addrs->insert(Index + SectionAddr + Size - 1);
        }
      }
    }
  }
}

static void getArchiveCoveragePoints(const object::Archive &A,
                                     std::set<uint64_t> *Addrs) {
  for (auto &ErrorOrChild : A.children()) {
    FailIfError(ErrorOrChild);
    const object::Archive::Child &C = *ErrorOrChild;
    ErrorOr<std::unique_ptr<object::Binary>> ChildOrErr = C.getAsBinary();
    FailIfError(ChildOrErr);
    if (object::ObjectFile *O =
            dyn_cast<object::ObjectFile>(&*ChildOrErr.get()))
      getObjectCoveragePoints(*O, Addrs);
    else
      FailIfError(object::object_error::invalid_file_type);
  }
}

// Locate addresses of all coverage points in a file. Coverage point
// is defined as the 'address of instruction following __sanitizer_cov
// call - 1'.
std::set<uint64_t> getCoveragePoints(std::string FileName) {
  std::set<uint64_t> Result;

  ErrorOr<object::OwningBinary<object::Binary>> BinaryOrErr =
      object::createBinary(FileName);
  FailIfError(BinaryOrErr);

  object::Binary &Binary = *BinaryOrErr.get().getBinary();
  if (object::Archive *A = dyn_cast<object::Archive>(&Binary))
    getArchiveCoveragePoints(*A, &Result);
  else if (object::ObjectFile *O = dyn_cast<object::ObjectFile>(&Binary))
    getObjectCoveragePoints(*O, &Result);
  else
    FailIfError(object::object_error::invalid_file_type);

  return Result;
}

static std::unique_ptr<SpecialCaseList> createDefaultBlacklist() {
  if (!ClUseDefaultBlacklist) 
    return std::unique_ptr<SpecialCaseList>();
  std::unique_ptr<MemoryBuffer> MB =
      MemoryBuffer::getMemBuffer(DefaultBlacklist);
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

static void printFunctionLocs(const std::set<FunctionLoc> &FnLocs,
                              raw_ostream &OS) {
  std::unique_ptr<SpecialCaseList> DefaultBlacklist = createDefaultBlacklist();
  std::unique_ptr<SpecialCaseList> UserBlacklist = createUserBlacklist();

  for (const FunctionLoc &FnLoc : FnLocs) {
    if (DefaultBlacklist &&
        DefaultBlacklist->inSection("fun", FnLoc.FunctionName))
      continue;
    if (DefaultBlacklist &&
        DefaultBlacklist->inSection("src", FnLoc.Loc.FileName))
      continue;
    if (UserBlacklist && UserBlacklist->inSection("fun", FnLoc.FunctionName))
      continue;
    if (UserBlacklist && UserBlacklist->inSection("src", FnLoc.Loc.FileName))
      continue;

    OS << stripPathPrefix(FnLoc.Loc.FileName) << ":" << FnLoc.Loc.Line << " "
       << FnLoc.FunctionName << "\n";
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

// Computes a map file_name->{line_number}
static std::map<std::string, std::set<int>>
getFileLines(std::string ObjectFile, const std::set<uint64_t> &Addrs) {
  std::map<std::string, std::set<int>> FileLines;

  auto Symbolizer(createSymbolizer());

  // Fill in FileLines map.
  for (auto Addr : Addrs) {
    auto InliningInfo = Symbolizer->symbolizeInlinedCode(ObjectFile, Addr);
    FailIfError(InliningInfo);
    for (uint32_t I = 0; I < InliningInfo->getNumberOfFrames(); ++I) {
      auto FrameInfo = InliningInfo->getFrame(I);
      SmallString<256> FileName(FrameInfo.FileName);
      sys::path::remove_dots(FileName, /* remove_dot_dot */ true);
      FileLines[FileName.str()].insert(FrameInfo.Line);
    }
  }

  return FileLines;
}

static ErrorOr<bool> isCoverageFile(std::string FileName) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(FileName);
  if (!BufOrErr)
    return BufOrErr.getError();
  std::unique_ptr<MemoryBuffer> Buf = std::move(BufOrErr.get());
  if (Buf->getBufferSize() < 8) {
    return false;
  }
  const FileHeader *Header =
      reinterpret_cast<const FileHeader *>(Buf->getBufferStart());
  return Header->Magic == BinCoverageMagic;
}

class CoverageData {
 public:
  // Read single file coverage data.
   static ErrorOr<std::unique_ptr<CoverageData>> read(std::string FileName) {
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

  void printReport(std::string ObjectFile, raw_ostream &OS) {
    // file_name -> set of covered lines;
    std::map<std::string, std::set<int>> CoveredFileLines =
        getFileLines(ObjectFile, *Addrs);
    std::map<std::string, std::set<int>> CoveragePoints =
        getFileLines(ObjectFile, getCoveragePoints(ObjectFile));

    // TOC
    OS << "<ul>\n";
    for (auto It : CoveredFileLines) {
      auto FileName = It.first;
      OS << "<li><a href=\"#" << escapeHtml(FileName) << "\">"
         << stripPathPrefix(FileName) << "</a></li>\n";
    }
    OS << "</ul>\n";

    // Source
    for (auto It : CoveredFileLines) {
      auto FileName = It.first;
      auto Lines = It.second;
      auto CovLines = CoveragePoints[FileName];

      OS << "<a name=\"" << escapeHtml(FileName) << "\"></a>\n";
      OS << "<h2>" << stripPathPrefix(FileName) << "</h2>\n";
      ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
          MemoryBuffer::getFile(FileName);
      if (!BufOrErr) {
        OS << "Error reading file: " << FileName << " : "
           << BufOrErr.getError().message() << "("
           << BufOrErr.getError().value() << ")\n";
        continue;
      }

      OS << "<pre>\n";
      for (line_iterator I = line_iterator(*BufOrErr.get(), false);
           !I.is_at_eof(); ++I) {
        OS << "<span ";
        if (Lines.find(I.line_number()) != Lines.end())
          OS << "class=covered";
        else if (CovLines.find(I.line_number()) != CovLines.end())
          OS << "class=notcovered";
        OS << ">";
        OS << escapeHtml(*I) << "</span>\n";
      }
      OS << "</pre>\n";
    }
  }

  // Print list of covered functions.
  // Line format: <file_name>:<line> <function_name>
  void printCoveredFunctions(std::string ObjectFile, raw_ostream &OS) {
    printFunctionLocs(computeFunctionLocs(ObjectFile, *Addrs), OS);
  }

  // Print list of not covered functions.
  // Line format: <file_name>:<line> <function_name>
  void printNotCoveredFunctions(std::string ObjectFile, raw_ostream &OS) {
    std::set<FunctionLoc> AllFns =
        computeFunctionLocs(ObjectFile, getCoveragePoints(ObjectFile));
    std::set<FunctionLoc> CoveredFns = computeFunctionLocs(ObjectFile, *Addrs);

    std::set<FunctionLoc> NotCoveredFns;
    std::set_difference(AllFns.begin(), AllFns.end(), CoveredFns.begin(),
                        CoveredFns.end(),
                        std::inserter(NotCoveredFns, NotCoveredFns.end()));
    printFunctionLocs(NotCoveredFns, OS);
  }

private:
  explicit CoverageData(std::unique_ptr<std::set<uint64_t>> Addrs)
      : Addrs(std::move(Addrs)) {}

  std::unique_ptr<std::set<uint64_t>> Addrs;
};

// Holder for coverage data + filename of corresponding object file.
class CoverageDataWithObjectFile {
public:
  static ErrorOr<std::unique_ptr<CoverageDataWithObjectFile>>
  readAndMerge(std::string ObjectFile,
               const std::vector<std::string> &FileNames) {
    auto MergedDataOrError = CoverageData::readAndMerge(FileNames);
    if (!MergedDataOrError)
      return MergedDataOrError.getError();
    return std::unique_ptr<CoverageDataWithObjectFile>(
        new CoverageDataWithObjectFile(ObjectFile,
                                       std::move(MergedDataOrError.get())));
  }

  std::string object_file() const { return ObjectFile; }

  void printCoveredFunctions(raw_ostream &OS) const {
    Coverage->printCoveredFunctions(ObjectFile, OS);
  }

  void printNotCoveredFunctions(raw_ostream &OS) const {
    Coverage->printNotCoveredFunctions(ObjectFile, OS);
  }

  void printReport(raw_ostream &OS) const {
    Coverage->printReport(ObjectFile, OS);
  }

private:
  CoverageDataWithObjectFile(std::string ObjectFile,
                             std::unique_ptr<CoverageData> Coverage)
      : ObjectFile(std::move(ObjectFile)), Coverage(std::move(Coverage)) {}

  const std::string ObjectFile;
  const std::unique_ptr<CoverageData> Coverage;
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
      FailIfError(ErrorOrIsCoverage);
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

    // Object file => list of corresponding coverage files.
    std::map<std::string, std::vector<std::string>> CoverageByObjFile;
    Regex SancovRegex("(.*)\\.[0-9]+\\.sancov");
    SmallVector<StringRef, 2> Components;

    // Group coverage files by object file.
    for (const auto &FileName : CovFiles) {
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
      auto ObjectFile = Iter->second;
      CoverageByObjFile[ObjectFile].push_back(FileName);
    }

    // Read coverage.
    std::vector<std::unique_ptr<CoverageDataWithObjectFile>> MergedCoverage;
    for (const auto &Pair : CoverageByObjFile) {
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

  void printReport(raw_ostream &OS) const {
    std::string Title = stripPathPrefix(MainObjFile) + " Coverage Report";

    OS << "<html>\n";
    OS << "<head>\n";

    // Stylesheet
    OS << "<style>\n";
    OS << ".covered { background: #7F7; }\n";
    OS << ".notcovered { background: #F77; }\n";
    OS << "</style>\n";
    OS << "<title>" << Title << "</title>\n";
    OS << "</head>\n";
    OS << "<body>\n";

    // Title
    OS << "<h1>" << Title << "</h1>\n";
    OS << "<p>Coverage files: ";
    for (auto InputFile : CoverageFiles) {
      llvm::sys::fs::file_status Status;
      llvm::sys::fs::status(InputFile, Status);
      OS << stripPathPrefix(InputFile) << " ("
         << Status.getLastModificationTime().str() << ") ";
    }
    OS << "</p>\n";

    // Modules TOC.
    if (Coverage.size() > 1) {
      for (const auto &CovData : Coverage) {
        OS << "<li><a href=\"#module_" << escapeHtml(CovData->object_file())
           << "\">" << llvm::sys::path::filename(CovData->object_file())
           << "</a></li>\n";
      }
    }

    for (const auto &CovData : Coverage) {
      if (Coverage.size() > 1) {
        OS << "<h2>" << llvm::sys::path::filename(CovData->object_file())
           << "</h2>\n";
      }
      OS << "<a name=\"module_" << escapeHtml(CovData->object_file())
         << "\"></a>\n";
      CovData->printReport(OS);
    }

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
  sys::PrintStackTraceOnErrorSignal();
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
  case PrintAction:
    llvm_unreachable("unsupported action");
  }
}
