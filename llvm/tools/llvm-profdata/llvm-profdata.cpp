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
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <tuple>

using namespace llvm;

enum ProfileFormat { PF_None = 0, PF_Text, PF_Binary, PF_GCC };

///// Profile summary computation ////
// The 'show' command displays richer summary of the profile data. The profile
// summary is one or more (Cutoff, MinBlockCount, NumBlocks) triplets. Given a
// target execution count percentile, we compute the minimum number of blocks
// needed to reach this target and the minimum execution count of these blocks.
struct ProfileSummaryEntry {
  uint32_t Cutoff;        ///< The required percentile of total execution count.
  uint64_t MinBlockCount; ///< The minimum execution count for this percentile.
  uint64_t NumBlocks;     ///< Number of blocks >= the minumum execution count.
};

class ProfileSummary {
  // We keep track of the number of times a count appears in the profile and
  // keep the map sorted in the descending order of counts.
  std::map<uint64_t, uint32_t, std::greater<uint64_t>> CountFrequencies;
  std::vector<ProfileSummaryEntry> DetailedSummary;
  std::vector<uint32_t> DetailedSummaryCutoffs;
  // Sum of all counts.
  uint64_t TotalCount;
  uint64_t MaxBlockCount, MaxFunctionCount;
  uint32_t NumBlocks, NumFunctions;
  void addCount(uint64_t Count);
  void computeDetailedSummary();

public:
  static const int Scale = 1000000;
  ProfileSummary(std::vector<uint32_t> Cutoffs)
      : DetailedSummaryCutoffs(Cutoffs), TotalCount(0), MaxBlockCount(0),
        MaxFunctionCount(0), NumBlocks(0), NumFunctions(0) {}
  void addRecord(const InstrProfRecord &);
  std::vector<ProfileSummaryEntry> &getDetailedSummary();
  uint32_t getNumBlocks() { return NumBlocks; }
  uint64_t getTotalCount() { return TotalCount; }
  uint32_t getNumFunctions() { return NumFunctions; }
  uint64_t getMaxFunctionCount() { return MaxFunctionCount; }
  uint64_t getMaxBlockCount() { return MaxBlockCount; }
};

// This is called when a count is seen in the profile.
void ProfileSummary::addCount(uint64_t Count) {
  TotalCount += Count;
  if (Count > MaxBlockCount)
    MaxBlockCount = Count;
  NumBlocks++;
  CountFrequencies[Count]++;
}

void ProfileSummary::addRecord(const InstrProfRecord &R) {
  NumFunctions++;
  if (R.Counts[0] > MaxFunctionCount)
    MaxFunctionCount = R.Counts[0];

  for (size_t I = 1, E = R.Counts.size(); I < E; ++I)
    addCount(R.Counts[I]);
}

// The argument to this method is a vector of cutoff percentages and the return
// value is a vector of (Cutoff, MinBlockCount, NumBlocks) triplets.
void ProfileSummary::computeDetailedSummary() {
  if (DetailedSummaryCutoffs.empty())
    return;
  auto Iter = CountFrequencies.begin();
  auto End = CountFrequencies.end();
  std::sort(DetailedSummaryCutoffs.begin(), DetailedSummaryCutoffs.end());

  uint32_t BlocksSeen = 0;
  uint64_t CurrSum = 0, Count;

  for (uint32_t Cutoff : DetailedSummaryCutoffs) {
    assert(Cutoff <= 999999);
    APInt Temp(128, TotalCount);
    APInt N(128, Cutoff);
    APInt D(128, ProfileSummary::Scale);
    Temp *= N;
    Temp = Temp.sdiv(D);
    uint64_t DesiredCount = Temp.getZExtValue();
    dbgs() << "Cutoff = " << Cutoff << "\n";
    dbgs() << "DesiredCount = " << DesiredCount << "\n";
    assert(DesiredCount <= TotalCount);
    while (CurrSum < DesiredCount && Iter != End) {
      Count = Iter->first;
      uint32_t Freq = Iter->second;
      CurrSum += (Count * Freq);
      BlocksSeen += Freq;
      Iter++;
    }
    assert(CurrSum >= DesiredCount);
    ProfileSummaryEntry PSE = {Cutoff, Count, BlocksSeen};
    DetailedSummary.push_back(PSE);
  }
  return;
}

std::vector<ProfileSummaryEntry> &ProfileSummary::getDetailedSummary() {
  if (!DetailedSummaryCutoffs.empty() && DetailedSummary.empty())
    computeDetailedSummary();
  return DetailedSummary;
}

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

static void exitWithErrorCode(const std::error_code &Error,
                              StringRef Whence = "") {
  if (Error.category() == instrprof_category()) {
    instrprof_error instrError = static_cast<instrprof_error>(Error.value());
    if (instrError == instrprof_error::unrecognized_format) {
      // Hint for common error of forgetting -sample for sample profiles.
      exitWithError(Error.message(), Whence,
                    "Perhaps you forgot to use the -sample option?");
    }
  }
  exitWithError(Error.message(), Whence);
}

namespace {
enum ProfileKinds { instr, sample };
}

static void handleMergeWriterError(std::error_code &Error,
                                   StringRef WhenceFile = "",
                                   StringRef WhenceFunction = "",
                                   bool ShowHint = true) {
  if (!WhenceFile.empty())
    errs() << WhenceFile << ": ";
  if (!WhenceFunction.empty())
    errs() << WhenceFunction << ": ";
  errs() << Error.message() << "\n";

  if (ShowHint) {
    StringRef Hint = "";
    if (Error.category() == instrprof_category()) {
      instrprof_error instrError = static_cast<instrprof_error>(Error.value());
      switch (instrError) {
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
  StringRef Filename;
  uint64_t Weight;

  WeightedFile() {}

  WeightedFile(StringRef F, uint64_t W) : Filename{F}, Weight{W} {}
};
typedef SmallVector<WeightedFile, 5> WeightedFileVector;

static void mergeInstrProfile(const WeightedFileVector &Inputs,
                              StringRef OutputFilename,
                              ProfileFormat OutputFormat) {
  if (OutputFilename.compare("-") == 0)
    exitWithError("Cannot write indexed profdata format to stdout.");

  if (OutputFormat != PF_Binary && OutputFormat != PF_Text)
    exitWithError("Unknown format is specified.");

  std::error_code EC;
  raw_fd_ostream Output(OutputFilename.data(), EC, sys::fs::F_None);
  if (EC)
    exitWithErrorCode(EC, OutputFilename);

  InstrProfWriter Writer;
  SmallSet<std::error_code, 4> WriterErrorCodes;
  for (const auto &Input : Inputs) {
    auto ReaderOrErr = InstrProfReader::create(Input.Filename);
    if (std::error_code ec = ReaderOrErr.getError())
      exitWithErrorCode(ec, Input.Filename);

    auto Reader = std::move(ReaderOrErr.get());
    for (auto &I : *Reader) {
      if (std::error_code EC = Writer.addRecord(std::move(I), Input.Weight)) {
        // Only show hint the first time an error occurs.
        bool firstTime = WriterErrorCodes.insert(EC).second;
        handleMergeWriterError(EC, Input.Filename, I.Name, firstTime);
      }
    }
    if (Reader->hasError())
      exitWithErrorCode(Reader->getError(), Input.Filename);
  }
  if (OutputFormat == PF_Text)
    Writer.writeText(Output);
  else
    Writer.write(Output);
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
  for (const auto &Input : Inputs) {
    auto ReaderOrErr =
        SampleProfileReader::create(Input.Filename, getGlobalContext());
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
        handleMergeWriterError(EC, Input.Filename, FName);
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

  if (!sys::fs::exists(FileName))
    exitWithErrorCode(make_error_code(errc::no_such_file_or_directory),
                      FileName);

  return WeightedFile(FileName, Weight);
}

static int merge_main(int argc, const char *argv[]) {
  cl::list<std::string> InputFilenames(cl::Positional,
                                       cl::desc("<filename...>"));
  cl::list<std::string> WeightedInputFilenames("weighted-input",
                                               cl::desc("<weight>,<filename>"));
  cl::opt<std::string> OutputFilename("output", cl::value_desc("output"),
                                      cl::init("-"), cl::Required,
                                      cl::desc("Output file"));
  cl::alias OutputFilenameA("o", cl::desc("Alias for --output"),
                            cl::aliasopt(OutputFilename));
  cl::opt<ProfileKinds> ProfileKind(
      cl::desc("Profile kind:"), cl::init(instr),
      cl::values(clEnumVal(instr, "Instrumentation profile (default)"),
                 clEnumVal(sample, "Sample profile"), clEnumValEnd));

  cl::opt<ProfileFormat> OutputFormat(
      cl::desc("Format of output profile"), cl::init(PF_Binary),
      cl::values(clEnumValN(PF_Binary, "binary", "Binary encoding (default)"),
                 clEnumValN(PF_Text, "text", "Text encoding"),
                 clEnumValN(PF_GCC, "gcc",
                            "GCC encoding (only meaningful for -sample)"),
                 clEnumValEnd));

  cl::ParseCommandLineOptions(argc, argv, "LLVM profile data merger\n");

  if (InputFilenames.empty() && WeightedInputFilenames.empty())
    exitWithError("No input files specified. See " +
                  sys::path::filename(argv[0]) + " -help");

  WeightedFileVector WeightedInputs;
  for (StringRef Filename : InputFilenames)
    WeightedInputs.push_back(WeightedFile(Filename, 1));
  for (StringRef WeightedFilename : WeightedInputFilenames)
    WeightedInputs.push_back(parseWeightedFile(WeightedFilename));

  if (ProfileKind == instr)
    mergeInstrProfile(WeightedInputs, OutputFilename, OutputFormat);
  else
    mergeSampleProfile(WeightedInputs, OutputFilename, OutputFormat);

  return 0;
}

static int showInstrProfile(std::string Filename, bool ShowCounts,
                            bool ShowIndirectCallTargets,
                            bool ShowDetailedSummary,
                            std::vector<uint32_t> DetailedSummaryCutoffs,
                            bool ShowAllFunctions, std::string ShowFunction,
                            bool TextFormat, raw_fd_ostream &OS) {
  auto ReaderOrErr = InstrProfReader::create(Filename);
  std::vector<uint32_t> Cutoffs(DetailedSummaryCutoffs);
  if (ShowDetailedSummary && DetailedSummaryCutoffs.empty()) {
    Cutoffs = {800000, 900000, 950000, 990000, 999000, 999900, 999990};
  }
  ProfileSummary PS(Cutoffs);
  if (std::error_code EC = ReaderOrErr.getError())
    exitWithErrorCode(EC, Filename);

  auto Reader = std::move(ReaderOrErr.get());
  size_t ShownFunctions = 0;
  for (const auto &Func : *Reader) {
    bool Show =
        ShowAllFunctions || (!ShowFunction.empty() &&
                             Func.Name.find(ShowFunction) != Func.Name.npos);

    bool doTextFormatDump = (Show && ShowCounts && TextFormat);

    if (doTextFormatDump) {
      InstrProfSymtab &Symtab = Reader->getSymtab();
      InstrProfWriter::writeRecordInText(Func, Symtab, OS);
      continue;
    }

    assert(Func.Counts.size() > 0 && "function missing entry counter");
    PS.addRecord(Func);

    if (Show) {

      if (!ShownFunctions)
        OS << "Counters:\n";

      ++ShownFunctions;

      OS << "  " << Func.Name << ":\n"
         << "    Hash: " << format("0x%016" PRIx64, Func.Hash) << "\n"
         << "    Counters: " << Func.Counts.size() << "\n"
         << "    Function count: " << Func.Counts[0] << "\n";

      if (ShowIndirectCallTargets)
        OS << "    Indirect Call Site Count: "
           << Func.getNumValueSites(IPVK_IndirectCallTarget) << "\n";

      if (ShowCounts) {
        OS << "    Block counts: [";
        for (size_t I = 1, E = Func.Counts.size(); I < E; ++I) {
          OS << (I == 1 ? "" : ", ") << Func.Counts[I];
        }
        OS << "]\n";
      }

      if (ShowIndirectCallTargets) {
        InstrProfSymtab &Symtab = Reader->getSymtab();
        uint32_t NS = Func.getNumValueSites(IPVK_IndirectCallTarget);
        OS << "    Indirect Target Results: \n";
        for (size_t I = 0; I < NS; ++I) {
          uint32_t NV = Func.getNumValueDataForSite(IPVK_IndirectCallTarget, I);
          std::unique_ptr<InstrProfValueData[]> VD =
              Func.getValueForSite(IPVK_IndirectCallTarget, I);
          for (uint32_t V = 0; V < NV; V++) {
            OS << "\t[ " << I << ", ";
            OS << Symtab.getFuncName(VD[V].Value) << ", " << VD[V].Count
               << " ]\n";
          }
        }
      }
    }
  }

  if (Reader->hasError())
    exitWithErrorCode(Reader->getError(), Filename);

  if (ShowCounts && TextFormat)
    return 0;

  if (ShowAllFunctions || !ShowFunction.empty())
    OS << "Functions shown: " << ShownFunctions << "\n";
  OS << "Total functions: " << PS.getNumFunctions() << "\n";
  OS << "Maximum function count: " << PS.getMaxFunctionCount() << "\n";
  OS << "Maximum internal block count: " << PS.getMaxBlockCount() << "\n";

  if (ShowDetailedSummary) {
    OS << "Detailed summary:\n";
    OS << "Total number of blocks: " << PS.getNumBlocks() << "\n";
    OS << "Total count: " << PS.getTotalCount() << "\n";
    for (auto Entry : PS.getDetailedSummary()) {
      OS << Entry.NumBlocks << " blocks with count >= " << Entry.MinBlockCount
         << " account for "
         << format("%0.6g", (float)Entry.Cutoff / ProfileSummary::Scale * 100)
         << " percentage of the total counts.\n";
    }
  }
  return 0;
}

static int showSampleProfile(std::string Filename, bool ShowCounts,
                             bool ShowAllFunctions, std::string ShowFunction,
                             raw_fd_ostream &OS) {
  using namespace sampleprof;
  auto ReaderOrErr = SampleProfileReader::create(Filename, getGlobalContext());
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
                 clEnumVal(sample, "Sample profile"), clEnumValEnd));

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
    return showInstrProfile(Filename, ShowCounts, ShowIndirectCallTargets,
                            ShowDetailedSummary, DetailedSummaryCutoffs,
                            ShowAllFunctions, ShowFunction, TextFormat, OS);
  else
    return showSampleProfile(Filename, ShowCounts, ShowAllFunctions,
                             ShowFunction, OS);
}

int main(int argc, const char *argv[]) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
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
