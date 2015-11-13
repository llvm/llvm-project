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
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <set>
#include <stdio.h>
#include <vector>

using namespace llvm;

namespace {

// --------- COMMAND LINE FLAGS ---------

enum ActionType { PrintAction, CoveredFunctionsAction };

cl::opt<ActionType> Action(
    cl::desc("Action (required)"), cl::Required,
    cl::values(clEnumValN(PrintAction, "print", "Print coverage addresses"),
               clEnumValN(CoveredFunctionsAction, "covered_functions",
                          "Print all covered funcions."),
               clEnumValEnd));

static cl::list<std::string> ClInputFiles(cl::Positional, cl::OneOrMore,
                                          cl::desc("<filenames...>"));

static cl::opt<std::string>
    ClBinaryName("obj", cl::Required,
                 cl::desc("Path to object file to be symbolized"));

static cl::opt<bool>
    ClDemangle("demangle", cl::init(true),
        cl::desc("Print demangled function name."));

// --------- FORMAT SPECIFICATION ---------

struct FileHeader {
  uint32_t Bitness;
  uint32_t Magic;
};

static const uint32_t BinCoverageMagic = 0xC0BFFFFF;
static const uint32_t Bitness32 = 0xFFFFFF32;
static const uint32_t Bitness64 = 0xFFFFFF64;

// ---------

template <typename T> static void FailIfError(const ErrorOr<T> &E) {
  if (E)
    return;

  auto Error = E.getError();
  errs() << "Error: " << Error.message() << "(" << Error.value() << ")\n";
  exit(-2);
}

template <typename T>
static void readInts(const char *Start, const char *End,
                     std::vector<uint64_t> *V) {
  const T *S = reinterpret_cast<const T *>(Start);
  const T *E = reinterpret_cast<const T *>(End);
  V->reserve(E - S);
  std::copy(S, E, std::back_inserter(*V));
}

static std::string CommonPrefix(std::string A, std::string B) {
  if (A.size() > B.size())
    return std::string(B.begin(),
                       std::mismatch(B.begin(), B.end(), A.begin()).first);
  else
    return std::string(A.begin(),
                       std::mismatch(A.begin(), A.end(), B.begin()).first);
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

    auto Addrs = llvm::make_unique<std::vector<uint64_t>>();

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
    std::set<uint64_t> Addrs;

    for (const auto &Cov : Covs)
      Addrs.insert(Cov->Addrs->begin(), Cov->Addrs->end());

    auto AddrsVector = llvm::make_unique<std::vector<uint64_t>>(
        Addrs.begin(), Addrs.end());
    return std::unique_ptr<CoverageData>(
        new CoverageData(std::move(AddrsVector)));
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
  void printAddrs(raw_ostream &out) {
    for (auto Addr : *Addrs) {
      out << "0x";
      out.write_hex(Addr);
      out << "\n";
    }
  }

  // Print list of covered functions.
  // Line format: <file_name>:<line> <function_name>
  void printCoveredFunctions(raw_ostream &out) {
    if (Addrs->empty())
      return;
    symbolize::LLVMSymbolizer::Options SymbolizerOptions;
    SymbolizerOptions.Demangle = ClDemangle;
    symbolize::LLVMSymbolizer Symbolizer(SymbolizerOptions);

    struct FileLoc {
      std::string FileName;
      uint32_t Line;
      bool operator<(const FileLoc &Rhs) const {
        return std::tie(FileName, Line) < std::tie(Rhs.FileName, Rhs.Line);
      }
    };

    // FileLoc -> FunctionName
    std::map<FileLoc, std::string> Fns;

    // Fill in Fns map.
    for (auto Addr : *Addrs) {
      auto InliningInfo = Symbolizer.symbolizeInlinedCode(ClBinaryName, Addr);
      FailIfError(InliningInfo);
      for (uint32_t i = 0; i < InliningInfo->getNumberOfFrames(); ++i) {
        auto FrameInfo = InliningInfo->getFrame(i);
        SmallString<256> FileName(FrameInfo.FileName);
        sys::path::remove_dots(FileName, /* remove_dot_dot */ true);
        FileLoc Loc = { FileName.str(), FrameInfo.Line };
        Fns[Loc] = FrameInfo.FunctionName;
      }
    }

    // Compute file names common prefix.
    std::string FilePrefix = Fns.begin()->first.FileName;
    for (const auto &P : Fns)
      FilePrefix = CommonPrefix(FilePrefix, P.first.FileName);

    // Print first function occurence in a file.
    {
      std::string LastFileName;
      std::set<std::string> ProcessedFunctions;

      for (const auto &P : Fns) {
        std::string FileName = P.first.FileName;
        std::string FunctionName = P.second;
        uint32_t Line = P.first.Line;

        if (LastFileName != FileName)
          ProcessedFunctions.clear();
        LastFileName = FileName;

        if (!ProcessedFunctions.insert(FunctionName).second)
          continue;

        // Don't strip prefix if we only have a single file.
        if (FileName.size() > FilePrefix.size())
          FileName = FileName.substr(FilePrefix.size());

        out << FileName << ":" << Line << " " << FunctionName << "\n";
      }
    }
  }

 private:
  explicit CoverageData(std::unique_ptr<std::vector<uint64_t>> Addrs)
      : Addrs(std::move(Addrs)) {}

  std::unique_ptr<std::vector<uint64_t>> Addrs;
};
} // namespace

int main(int argc, char **argv) {
  // Print stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "Sanitizer Coverage Processing Tool");

  auto CovData = CoverageData::readAndMerge(ClInputFiles);
  FailIfError(CovData);

  switch (Action) {
  case PrintAction: {
    CovData.get()->printAddrs(outs());
    return 0;
  }
  case CoveredFunctionsAction: {
    CovData.get()->printCoveredFunctions(outs());
    return 0;
  }
  }
}
