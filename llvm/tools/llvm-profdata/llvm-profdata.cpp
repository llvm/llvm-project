//===- llvm-profdata.cpp - LLVM profile data tool -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// llvm-profdata merges .profdata files.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/ProfileData/ProfileCommon.h"
#include "llvm/ProfileData/SampleProfReader.h"
#include "llvm/ProfileData/SampleProfWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;

enum ProfileFormat { PF_None = 0, PF_Text, PF_Binary, PF_GCC };

static void exitWithError(const Twine &Message, StringRef Whence = "",
                          StringRef Hint = "") {
  errs() << "error: ";
  if (!Whence.empty())
    errs() << Whence << ": ";
  errs() << Message << "\n";
  if (!Hint.empty())
    errs() << Hint << "\n";
  ::exit(1);
}

static void exitWithError(Error E, StringRef Whence = "") {
  if (E.isA<InstrProfError>()) {
    handleAllErrors(std::move(E), [&](const InstrProfError &IPE) {
      instrprof_error instrError = IPE.get();
      StringRef Hint = "";
      if (instrError == instrprof_error::unrecognized_format) {
        // Hint for common error of forgetting -sample for sample profiles.
        Hint = "Perhaps you forgot to use the -sample option?";
      }
      exitWithError(IPE.message(), Whence, Hint);
    });
  }

  exitWithError(toString(std::move(E)), Whence);
}

static void exitWithErrorCode(std::error_code EC, StringRef Whence = "") {
  exitWithError(EC.message(), Whence);
}

namespace {
enum ProfileKinds { instr, sample };
}

static void handleMergeWriterError(Error E, StringRef WhenceFile = "",
                                   StringRef WhenceFunction = "",
                                   bool ShowHint = true) {
  if (!WhenceFile.empty())
    errs() << WhenceFile << ": ";
  if (!WhenceFunction.empty())
    errs() << WhenceFunction << ": ";

  auto IPE = instrprof_error::success;
  E = handleErrors(std::move(E),
                   [&IPE](std::unique_ptr<InstrProfError> E) -> Error {
                     IPE = E->get();
                     return Error(std::move(E));
                   });
  errs() << toString(std::move(E)) << "\n";

  if (ShowHint) {
    StringRef Hint = "";
    if (IPE != instrprof_error::success) {
      switch (IPE) {
      case instrprof_error::hash_mismatch:
      case instrprof_error::count_mismatch:
      case instrprof_error::value_site_count_mismatch:
        Hint = "Make sure that all profile data to be merged is generated "
               "from the same binary.";
        break;
      default:
        break;
      }
    }

    if (!Hint.empty())
      errs() << Hint << "\n";
  }
}

struct WeightedFile {
  std::string Filename;
  uint64_t Weight;
};
typedef SmallVector<WeightedFile, 5> WeightedFileVector;

/// Keep track of merged data and reported errors.
struct WriterContext {
  std::mutex Lock;
  InstrProfWriter Writer;
  Error Err;
  StringRef ErrWhence;
  std::mutex &ErrLock;
  SmallSet<instrprof_error, 4> &WriterErrorCodes;

  WriterContext(bool IsSparse, std::mutex &ErrLock,
                SmallSet<instrprof_error, 4> &WriterErrorCodes)
      : Lock(), Writer(IsSparse), Err(Error::success()), ErrWhence(""),
        ErrLock(ErrLock), WriterErrorCodes(WriterErrorCodes) {}
};

/// Load an input into a writer context.
static void loadInput(const WeightedFile &Input, WriterContext *WC) {
  std::unique_lock<std::mutex> CtxGuard{WC->Lock};

  // If there's a pending hard error, don't do more work.
  if (WC->Err)
    return;

  WC->ErrWhence = Input.Filename;

  auto ReaderOrErr = InstrProfReader::create(Input.Filename);
  if (Error E = ReaderOrErr.takeError()) {
    // Skip the empty profiles by returning sliently.
    instrprof_error IPE = InstrProfError::take(std::move(E));
    if (IPE != instrprof_error::empty_raw_profile)
      WC->Err = make_error<InstrProfError>(IPE);
    return;
  }

  auto Reader = std::move(ReaderOrErr.get());
  bool IsIRProfile = Reader->isIRLevelProfile();
  if (WC->Writer.setIsIRLevelProfile(IsIRProfile)) {
    WC->Err = make_error<StringError>(
        "Merge IR generated profile with Clang generated profile.",
        std::error_code());
    return;
  }

  for (auto &I : *Reader) {
    const StringRef FuncName = I.Name;
    bool Reported = false;
    WC->Writer.addRecord(std::move(I), Input.Weight, [&](Error E) {
      if (Reported) {
        consumeError(std::move(E));
        return;
      }
      Reported = true;
      // Only show hint the first time an error occurs.
      instrprof_error IPE = InstrProfError::take(std::move(E));
      std::unique_lock<std::mutex> ErrGuard{WC->ErrLock};
      bool firstTime = WC->WriterErrorCodes.insert(IPE).second;
      handleMergeWriterError(make_error<InstrProfError>(IPE), Input.Filename,
                             FuncName, firstTime);
    });
  }
  if (Reader->hasError())
    WC->Err = Reader->getError();
}

/// Merge the \p Src writer context into \p Dst.
static void mergeWriterContexts(WriterContext *Dst, WriterContext *Src) {
  bool Reported = false;
  Dst->Writer.mergeRecordsFromWriter(std::move(Src->Writer), [&](Error E) {
    if (Reported) {
      consumeError(std::move(E));
      return;
    }
    Reported = true;
    Dst->Err = std::move(E);
  });
}

static void mergeInstrProfile(const WeightedFileVector &Inputs,
                              StringRef OutputFilename,
                              ProfileFormat OutputFormat, bool OutputSparse,
                              unsigned NumThreads) {
  if (OutputFilename.compare("-") == 0)
    exitWithError("Cannot write indexed profdata format to stdout.");

  if (OutputFormat != PF_Binary && OutputFormat != PF_Text)
    exitWithError("Unknown format is specified.");

  std::error_code EC;
  raw_fd_ostream Output(OutputFilename.data(), EC, sys::fs::F_None);
  if (EC)
    exitWithErrorCode(EC, OutputFilename);

  std::mutex ErrorLock;
  SmallSet<instrprof_error, 4> WriterErrorCodes;

  // If NumThreads is not specified, auto-detect a good default.
  if (NumThreads == 0)
    NumThreads = std::max(1U, std::min(std::thread::hardware_concurrency(),
                                       unsigned(Inputs.size() / 2)));

  // Initialize the writer contexts.
  SmallVector<std::unique_ptr<WriterContext>, 4> Contexts;
  for (unsigned I = 0; I < NumThreads; ++I)
    Contexts.emplace_back(llvm::make_unique<WriterContext>(
        OutputSparse, ErrorLock, WriterErrorCodes));

  if (NumThreads == 1) {
    for (const auto &Input : Inputs)
      loadInput(Input, Contexts[0].get());
  } else {
    ThreadPool Pool(NumThreads);

    // Load the inputs in parallel (N/NumThreads serial steps).
    unsigned Ctx = 0;
    for (const auto &Input : Inputs) {
      Pool.async(loadInput, Input, Contexts[Ctx].get());
      Ctx = (Ctx + 1) % NumThreads;
    }
    Pool.wait();

    // Merge the writer contexts together (~ lg(NumThreads) serial steps).
    unsigned Mid = Contexts.size() / 2;
    unsigned End = Contexts.size();
    assert(Mid > 0 && "Expected more than one context");
    do {
      for (unsigned I = 0; I < Mid; ++I)
        Pool.async(mergeWriterContexts, Contexts[I].get(),
                   Contexts[I + Mid].get());
      Pool.wait();
      if (End & 1) {
        Pool.async(mergeWriterContexts, Contexts[0].get(),
                   Contexts[End - 1].get());
        Pool.wait();
      }
      End = Mid;
      Mid /= 2;
    } while (Mid > 0);
  }

  // Handle deferred hard errors encountered during merging.
  for (std::unique_ptr<WriterContext> &WC : Contexts)
    if (WC->Err)
      exitWithError(std::move(WC->Err), WC->ErrWhence);

  InstrProfWriter &Writer = Contexts[0]->Writer;
  if (OutputFormat == PF_Text) {
    if (Error E = Writer.writeText(Output))
      exitWithError(std::move(E));
  } else {
    Writer.write(Output);
  }
}

static sampleprof::SampleProfileFormat FormatMap[] = {
    sampleprof::SPF_None, sampleprof::SPF_Text, sampleprof::SPF_Binary,
    sampleprof::SPF_GCC};

static void mergeSampleProfile(const WeightedFileVector &Inputs,
                               StringRef OutputFilename,
                               ProfileFormat OutputFormat) {
  using namespace sampleprof;
  auto WriterOrErr =
      SampleProfileWriter::create(OutputFilename, FormatMap[OutputFormat]);
  if (std::error_code EC = WriterOrErr.getError())
    exitWithErrorCode(EC, OutputFilename);

  auto Writer = std::move(WriterOrErr.get());
  StringMap<FunctionSamples> ProfileMap;
  SmallVector<std::unique_ptr<sampleprof::SampleProfileReader>, 5> Readers;
  LLVMContext Context;
  for (const auto &Input : Inputs) {
    auto ReaderOrErr = SampleProfileReader::create(Input.Filename, Context);
    if (std::error_code EC = ReaderOrErr.getError())
      exitWithErrorCode(EC, Input.Filename);

    // We need to keep the readers around until after all the files are
    // read so that we do not lose the function names stored in each
    // reader's memory. The function names are needed to write out the
    // merged profile map.
    Readers.push_back(std::move(ReaderOrErr.get()));
    const auto Reader = Readers.back().get();
    if (std::error_code EC = Reader->read())
      exitWithErrorCode(EC, Input.Filename);

    StringMap<FunctionSamples> &Profiles = Reader->getProfiles();
    for (StringMap<FunctionSamples>::iterator I = Profiles.begin(),
                                              E = Profiles.end();
         I != E; ++I) {
      StringRef FName = I->first();
      FunctionSamples &Samples = I->second;
      sampleprof_error Result = ProfileMap[FName].merge(Samples, Input.Weight);
      if (Result != sampleprof_error::success) {
        std::error_code EC = make_error_code(Result);
        handleMergeWriterError(errorCodeToError(EC), Input.Filename, FName);
      }
    }
  }
  Writer->write(ProfileMap);
}

static WeightedFile parseWeightedFile(const StringRef &WeightedFilename) {
  StringRef WeightStr, FileName;
  std::tie(WeightStr, FileName) = WeightedFilename.split(',');

  uint64_t Weight;
  if (WeightStr.getAsInteger(10, Weight) || Weight < 1)
    exitWithError("Input weight must be a positive integer.");

  return {FileName, Weight};
}

static std::unique_ptr<MemoryBuffer>
getInputFilenamesFileBuf(const StringRef &InputFilenamesFile) {
  if (InputFilenamesFile == "")
    return {};

  auto BufOrError = MemoryBuffer::getFileOrSTDIN(InputFilenamesFile);
  if (!BufOrError)
    exitWithErrorCode(BufOrError.getError(), InputFilenamesFile);

  return std::move(*BufOrError);
}

static void addWeightedInput(WeightedFileVector &WNI, const WeightedFile &WF) {
  StringRef Filename = WF.Filename;
  uint64_t Weight = WF.Weight;

  // If it's STDIN just pass it on.
  if (Filename == "-") {
    WNI.push_back({Filename, Weight});
    return;
  }

  llvm::sys::fs::file_status Status;
  llvm::sys::fs::status(Filename, Status);
  if (!llvm::sys::fs::exists(Status))
    exitWithErrorCode(make_error_code(errc::no_such_file_or_directory),
                      Filename);
  // If it's a source file, collect it.
  if (llvm::sys::fs::is_regular_file(Status)) {
    WNI.push_back({Filename, Weight});
    return;
  }

  if (llvm::sys::fs::is_directory(Status)) {
    std::error_code EC;
    for (llvm::sys::fs::recursive_directory_iterator F(Filename, EC), E;
         F != E && !EC; F.increment(EC)) {
      if (llvm::sys::fs::is_regular_file(F->path())) {
        addWeightedInput(WNI, {F->path(), Weight});
      }
    }
    if (EC)
      exitWithErrorCode(EC, Filename);
  }
}

static void parseInputFilenamesFile(MemoryBuffer *Buffer,
                                    WeightedFileVector &WFV) {
  if (!Buffer)
    return;

  SmallVector<StringRef, 8> Entries;
  StringRef Data = Buffer->getBuffer();
  Data.split(Entries, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  for (const StringRef &FileWeightEntry : Entries) {
    StringRef SanitizedEntry = FileWeightEntry.trim(" \t\v\f\r");
    // Skip comments.
    if (SanitizedEntry.startswith("#"))
      continue;
    // If there's no comma, it's an unweighted profile.
    else if (SanitizedEntry.find(',') == StringRef::npos)
      addWeightedInput(WFV, {SanitizedEntry, 1});
    else
      addWeightedInput(WFV, parseWeightedFile(SanitizedEntry));
  }
}

static int merge_main(int argc, const char *argv[]) {
  cl::list<std::string> InputFilenames(cl::Positional,
                                       cl::desc("<filename...>"));
  cl::list<std::string> WeightedInputFilenames("weighted-input",
                                               cl::desc("<weight>,<filename>"));
  cl::opt<std::string> InputFilenamesFile(
      "input-files", cl::init(""),
      cl::desc("Path to file containing newline-separated "
               "[<weight>,]<filename> entries"));
  cl::alias InputFilenamesFileA("f", cl::desc("Alias for --input-files"),
                                cl::aliasopt(InputFilenamesFile));
  cl::opt<bool> DumpInputFileList(
      "dump-input-file-list", cl::init(false), cl::Hidden,
      cl::desc("Dump the list of input files and their weights, then exit"));
  cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                      cl::init("-"), cl::Required,
                                      cl::desc("Output file"));
  cl::alias OutputFilenameA("o", cl::desc("Alias for --output"),
                            cl::aliasopt(OutputFilename));
  cl::opt<ProfileKinds> ProfileKind(
      cl::desc("Profile kind:"), cl::init(instr),
      cl::values(clEnumVal(instr, "Instrumentation profile (default)"),
                 clEnumVal(sample, "Sample profile")));
  cl::opt<ProfileFormat> OutputFormat(
      cl::desc("Format of output profile"), cl::init(PF_Binary),
      cl::values(clEnumValN(PF_Binary, "binary", "Binary encoding (default)"),
                 clEnumValN(PF_Text, "text", "Text encoding"),
                 clEnumValN(PF_GCC, "gcc",
                            "GCC encoding (only meaningful for -sample)")));
  cl::opt<bool> OutputSparse("sparse", cl::init(false),
      cl::desc("Generate a sparse profile (only meaningful for -instr)"));
  cl::opt<unsigned> NumThreads(
      "num-threads", cl::init(0),
      cl::desc("Number of merge threads to use (default: autodetect)"));
  cl::alias NumThreadsA("j", cl::desc("Alias for --num-threads"),
                        cl::aliasopt(NumThreads));

  cl::ParseCommandLineOptions(argc, argv, "LLVM profile data merger\n");

  WeightedFileVector WeightedInputs;
  for (StringRef Filename : InputFilenames)
    addWeightedInput(WeightedInputs, {Filename, 1});
  for (StringRef WeightedFilename : WeightedInputFilenames)
    addWeightedInput(WeightedInputs, parseWeightedFile(WeightedFilename));

  // Make sure that the file buffer stays alive for the duration of the
  // weighted input vector's lifetime.
  auto Buffer = getInputFilenamesFileBuf(InputFilenamesFile);
  parseInputFilenamesFile(Buffer.get(), WeightedInputs);

  if (WeightedInputs.empty())
    exitWithError("No input files specified. See " +
                  sys::path::filename(argv[0]) + " -help");

  if (DumpInputFileList) {
    for (auto &WF : WeightedInputs)
      outs() << WF.Weight << "," << WF.Filename << "\n";
    return 0;
  }

  if (ProfileKind == instr)
    mergeInstrProfile(WeightedInputs, OutputFilename, OutputFormat,
                      OutputSparse, NumThreads);
  else
    mergeSampleProfile(WeightedInputs, OutputFilename, OutputFormat);

  return 0;
}

typedef struct ValueSitesStats {
  ValueSitesStats()
      : TotalNumValueSites(0), TotalNumValueSitesWithValueProfile(0),
        TotalNumValues(0) {}
  uint64_t TotalNumValueSites;
  uint64_t TotalNumValueSitesWithValueProfile;
  uint64_t TotalNumValues;
  std::vector<unsigned> ValueSitesHistogram;
} ValueSitesStats;

static void traverseAllValueSites(const InstrProfRecord &Func, uint32_t VK,
                                  ValueSitesStats &Stats, raw_fd_ostream &OS,
                                  InstrProfSymtab *Symtab) {
  uint32_t NS = Func.getNumValueSites(VK);
  Stats.TotalNumValueSites += NS;
  for (size_t I = 0; I < NS; ++I) {
    uint32_t NV = Func.getNumValueDataForSite(VK, I);
    std::unique_ptr<InstrProfValueData[]> VD = Func.getValueForSite(VK, I);
    Stats.TotalNumValues += NV;
    if (NV) {
      Stats.TotalNumValueSitesWithValueProfile++;
      if (NV > Stats.ValueSitesHistogram.size())
        Stats.ValueSitesHistogram.resize(NV, 0);
      Stats.ValueSitesHistogram[NV - 1]++;
    }
    for (uint32_t V = 0; V < NV; V++) {
      OS << "\t[ " << I << ", ";
      if (Symtab == nullptr)
        OS << VD[V].Value;
      else
        OS << Symtab->getFuncName(VD[V].Value);
      OS << ", " << VD[V].Count << " ]\n";
    }
  }
}

static void showValueSitesStats(raw_fd_ostream &OS, uint32_t VK,
                                ValueSitesStats &Stats) {
  OS << "  Total number of sites: " << Stats.TotalNumValueSites << "\n";
  OS << "  Total number of sites with values: "
     << Stats.TotalNumValueSitesWithValueProfile << "\n";
  OS << "  Total number of profiled values: " << Stats.TotalNumValues << "\n";

  OS << "  Value sites histogram:\n\tNumTargets, SiteCount\n";
  for (unsigned I = 0; I < Stats.ValueSitesHistogram.size(); I++) {
    if (Stats.ValueSitesHistogram[I] > 0)
      OS << "\t" << I + 1 << ", " << Stats.ValueSitesHistogram[I] << "\n";
  }
}

static int showInstrProfile(const std::string &Filename, bool ShowCounts,
                            uint32_t TopN, bool ShowIndirectCallTargets,
                            bool ShowMemOPSizes, bool ShowDetailedSummary,
                            std::vector<uint32_t> DetailedSummaryCutoffs,
                            bool ShowAllFunctions,
                            const std::string &ShowFunction, bool TextFormat,
                            raw_fd_ostream &OS) {
  auto ReaderOrErr = InstrProfReader::create(Filename);
  std::vector<uint32_t> Cutoffs = std::move(DetailedSummaryCutoffs);
  if (ShowDetailedSummary && Cutoffs.empty()) {
    Cutoffs = {800000, 900000, 950000, 990000, 999000, 999900, 999990};
  }
  InstrProfSummaryBuilder Builder(std::move(Cutoffs));
  if (Error E = ReaderOrErr.takeError())
    exitWithError(std::move(E), Filename);

  auto Reader = std::move(ReaderOrErr.get());
  bool IsIRInstr = Reader->isIRLevelProfile();
  size_t ShownFunctions = 0;
  int NumVPKind = IPVK_Last - IPVK_First + 1;
  std::vector<ValueSitesStats> VPStats(NumVPKind);

  auto MinCmp = [](const std::pair<std::string, uint64_t> &v1,
                   const std::pair<std::string, uint64_t> &v2) {
    return v1.second > v2.second;
  };

  std::priority_queue<std::pair<std::string, uint64_t>,
                      std::vector<std::pair<std::string, uint64_t>>,
                      decltype(MinCmp)>
      HottestFuncs(MinCmp);

  for (const auto &Func : *Reader) {
    bool Show =
        ShowAllFunctions || (!ShowFunction.empty() &&
                             Func.Name.find(ShowFunction) != Func.Name.npos);

    bool doTextFormatDump = (Show && ShowCounts && TextFormat);

    if (doTextFormatDump) {
      InstrProfSymtab &Symtab = Reader->getSymtab();
      InstrProfWriter::writeRecordInText(Func.Name, Func.Hash, Func, Symtab,
                                         OS);
      continue;
    }

    assert(Func.Counts.size() > 0 && "function missing entry counter");
    Builder.addRecord(Func);

    if (TopN) {
      uint64_t FuncMax = 0;
      for (size_t I = 0, E = Func.Counts.size(); I < E; ++I)
        FuncMax = std::max(FuncMax, Func.Counts[I]);

      if (HottestFuncs.size() == TopN) {
        if (HottestFuncs.top().second < FuncMax) {
          HottestFuncs.pop();
          HottestFuncs.emplace(std::make_pair(std::string(Func.Name), FuncMax));
        }
      } else
        HottestFuncs.emplace(std::make_pair(std::string(Func.Name), FuncMax));
    }

    if (Show) {

      if (!ShownFunctions)
        OS << "Counters:\n";

      ++ShownFunctions;

      OS << "  " << Func.Name << ":\n"
         << "    Hash: " << format("0x%016" PRIx64, Func.Hash) << "\n"
         << "    Counters: " << Func.Counts.size() << "\n";
      if (!IsIRInstr)
        OS << "    Function count: " << Func.Counts[0] << "\n";

      if (ShowIndirectCallTargets)
        OS << "    Indirect Call Site Count: "
           << Func.getNumValueSites(IPVK_IndirectCallTarget) << "\n";

      uint32_t NumMemOPCalls = Func.getNumValueSites(IPVK_MemOPSize);
      if (ShowMemOPSizes && NumMemOPCalls > 0)
        OS << "    Number of Memory Intrinsics Calls: " << NumMemOPCalls
           << "\n";

      if (ShowCounts) {
        OS << "    Block counts: [";
        size_t Start = (IsIRInstr ? 0 : 1);
        for (size_t I = Start, E = Func.Counts.size(); I < E; ++I) {
          OS << (I == Start ? "" : ", ") << Func.Counts[I];
        }
        OS << "]\n";
      }

      if (ShowIndirectCallTargets) {
        OS << "    Indirect Target Results:\n";
        traverseAllValueSites(Func, IPVK_IndirectCallTarget,
                              VPStats[IPVK_IndirectCallTarget], OS,
                              &(Reader->getSymtab()));
      }

      if (ShowMemOPSizes && NumMemOPCalls > 0) {
        OS << "    Memory Intrinsic Size Results:\n";
        traverseAllValueSites(Func, IPVK_MemOPSize, VPStats[IPVK_MemOPSize], OS,
                              nullptr);
      }
    }
  }
  if (Reader->hasError())
    exitWithError(Reader->getError(), Filename);

  if (ShowCounts && TextFormat)
    return 0;
  std::unique_ptr<ProfileSummary> PS(Builder.getSummary());
  if (ShowAllFunctions || !ShowFunction.empty())
    OS << "Functions shown: " << ShownFunctions << "\n";
  OS << "Total functions: " << PS->getNumFunctions() << "\n";
  OS << "Maximum function count: " << PS->getMaxFunctionCount() << "\n";
  OS << "Maximum internal block count: " << PS->getMaxInternalCount() << "\n";

  if (TopN) {
    std::vector<std::pair<std::string, uint64_t>> SortedHottestFuncs;
    while (!HottestFuncs.empty()) {
      SortedHottestFuncs.emplace_back(HottestFuncs.top());
      HottestFuncs.pop();
    }
    OS << "Top " << TopN
       << " functions with the largest internal block counts: \n";
    for (auto &hotfunc : llvm::reverse(SortedHottestFuncs))
      OS << "  " << hotfunc.first << ", max count = " << hotfunc.second << "\n";
  }

  if (ShownFunctions && ShowIndirectCallTargets) {
    OS << "Statistics for indirect call sites profile:\n";
    showValueSitesStats(OS, IPVK_IndirectCallTarget,
                        VPStats[IPVK_IndirectCallTarget]);
  }

  if (ShownFunctions && ShowMemOPSizes) {
    OS << "Statistics for memory intrinsic calls sizes profile:\n";
    showValueSitesStats(OS, IPVK_MemOPSize, VPStats[IPVK_MemOPSize]);
  }

  if (ShowDetailedSummary) {
    OS << "Detailed summary:\n";
    OS << "Total number of blocks: " << PS->getNumCounts() << "\n";
    OS << "Total count: " << PS->getTotalCount() << "\n";
    for (auto Entry : PS->getDetailedSummary()) {
      OS << Entry.NumCounts << " blocks with count >= " << Entry.MinCount
         << " account for "
         << format("%0.6g", (float)Entry.Cutoff / ProfileSummary::Scale * 100)
         << " percentage of the total counts.\n";
    }
  }
  return 0;
}

static int showSampleProfile(const std::string &Filename, bool ShowCounts,
                             bool ShowAllFunctions,
                             const std::string &ShowFunction,
                             raw_fd_ostream &OS) {
  using namespace sampleprof;
  LLVMContext Context;
  auto ReaderOrErr = SampleProfileReader::create(Filename, Context);
  if (std::error_code EC = ReaderOrErr.getError())
    exitWithErrorCode(EC, Filename);

  auto Reader = std::move(ReaderOrErr.get());
  if (std::error_code EC = Reader->read())
    exitWithErrorCode(EC, Filename);

  if (ShowAllFunctions || ShowFunction.empty())
    Reader->dump(OS);
  else
    Reader->dumpFunctionProfile(ShowFunction, OS);

  return 0;
}

static int show_main(int argc, const char *argv[]) {
  cl::opt<std::string> Filename(cl::Positional, cl::Required,
                                cl::desc("<profdata-file>"));

  cl::opt<bool> ShowCounts("counts", cl::init(false),
                           cl::desc("Show counter values for shown functions"));
  cl::opt<bool> TextFormat(
      "text", cl::init(false),
      cl::desc("Show instr profile data in text dump format"));
  cl::opt<bool> ShowIndirectCallTargets(
      "ic-targets", cl::init(false),
      cl::desc("Show indirect call site target values for shown functions"));
  cl::opt<bool> ShowMemOPSizes(
      "memop-sizes", cl::init(false),
      cl::desc("Show the profiled sizes of the memory intrinsic calls "
               "for shown functions"));
  cl::opt<bool> ShowDetailedSummary("detailed-summary", cl::init(false),
                                    cl::desc("Show detailed profile summary"));
  cl::list<uint32_t> DetailedSummaryCutoffs(
      cl::CommaSeparated, "detailed-summary-cutoffs",
      cl::desc(
          "Cutoff percentages (times 10000) for generating detailed summary"),
      cl::value_desc("800000,901000,999999"));
  cl::opt<bool> ShowAllFunctions("all-functions", cl::init(false),
                                 cl::desc("Details for every function"));
  cl::opt<std::string> ShowFunction("function",
                                    cl::desc("Details for matching functions"));

  cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                      cl::init("-"), cl::desc("Output file"));
  cl::alias OutputFilenameA("o", cl::desc("Alias for --output"),
                            cl::aliasopt(OutputFilename));
  cl::opt<ProfileKinds> ProfileKind(
      cl::desc("Profile kind:"), cl::init(instr),
      cl::values(clEnumVal(instr, "Instrumentation profile (default)"),
                 clEnumVal(sample, "Sample profile")));
  cl::opt<uint32_t> TopNFunctions(
      "topn", cl::init(0),
      cl::desc("Show the list of functions with the largest internal counts"));

  cl::ParseCommandLineOptions(argc, argv, "LLVM profile data summary\n");

  if (OutputFilename.empty())
    OutputFilename = "-";

  std::error_code EC;
  raw_fd_ostream OS(OutputFilename.data(), EC, sys::fs::F_Text);
  if (EC)
    exitWithErrorCode(EC, OutputFilename);

  if (ShowAllFunctions && !ShowFunction.empty())
    errs() << "warning: -function argument ignored: showing all functions\n";

  std::vector<uint32_t> Cutoffs(DetailedSummaryCutoffs.begin(),
                                DetailedSummaryCutoffs.end());
  if (ProfileKind == instr)
    return showInstrProfile(Filename, ShowCounts, TopNFunctions,
                            ShowIndirectCallTargets, ShowMemOPSizes,
                            ShowDetailedSummary, DetailedSummaryCutoffs,
                            ShowAllFunctions, ShowFunction, TextFormat, OS);
  else
    return showSampleProfile(Filename, ShowCounts, ShowAllFunctions,
                             ShowFunction, OS);
}

int main(int argc, const char *argv[]) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  StringRef ProgName(sys::path::filename(argv[0]));
  if (argc > 1) {
    int (*func)(int, const char *[]) = nullptr;

    if (strcmp(argv[1], "merge") == 0)
      func = merge_main;
    else if (strcmp(argv[1], "show") == 0)
      func = show_main;

    if (func) {
      std::string Invocation(ProgName.str() + " " + argv[1]);
      argv[1] = Invocation.c_str();
      return func(argc - 1, argv + 1);
    }

    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "-help") == 0 ||
        strcmp(argv[1], "--help") == 0) {

      errs() << "OVERVIEW: LLVM profile data tools\n\n"
             << "USAGE: " << ProgName << " <command> [args...]\n"
             << "USAGE: " << ProgName << " <command> -help\n\n"
             << "See each individual command --help for more details.\n"
             << "Available commands: merge, show\n";
      return 0;
    }
  }

  if (argc < 2)
    errs() << ProgName << ": No command specified!\n";
  else
    errs() << ProgName << ": Unknown command!\n";

  errs() << "USAGE: " << ProgName << " <merge|show> [args...]\n";
  return 1;
}
