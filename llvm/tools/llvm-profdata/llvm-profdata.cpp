//===- llvm-profdata.cpp - LLVM profile data tool -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;

enum ProfileFormat {
  PF_None = 0,
  PF_Text,
  PF_Compact_Binary,
  PF_Ext_Binary,
  PF_GCC,
  PF_Binary
};

static void warn(Twine Message, std::string Whence = "",
                 std::string Hint = "") {
  WithColor::warning();
  if (!Whence.empty())
    errs() << Whence << ": ";
  errs() << Message << "\n";
  if (!Hint.empty())
    WithColor::note() << Hint << "\n";
}

static void exitWithError(Twine Message, std::string Whence = "",
                          std::string Hint = "") {
  WithColor::error();
  if (!Whence.empty())
    errs() << Whence << ": ";
  errs() << Message << "\n";
  if (!Hint.empty())
    WithColor::note() << Hint << "\n";
  ::exit(1);
}

static void exitWithError(Error E, StringRef Whence = "") {
  if (E.isA<InstrProfError>()) {
    handleAllErrors(std::move(E), [&](const InstrProfError &IPE) {
      instrprof_error instrError = IPE.get();
      StringRef Hint = "";
      if (instrError == instrprof_error::unrecognized_format) {
        // Hint for common error of forgetting --sample for sample profiles.
        Hint = "Perhaps you forgot to use the --sample option?";
      }
      exitWithError(IPE.message(), std::string(Whence), std::string(Hint));
    });
  }

  exitWithError(toString(std::move(E)), std::string(Whence));
}

static void exitWithErrorCode(std::error_code EC, StringRef Whence = "") {
  exitWithError(EC.message(), std::string(Whence));
}

namespace {
enum ProfileKinds { instr, sample };
enum FailureMode { failIfAnyAreInvalid, failIfAllAreInvalid };
}

static void warnOrExitGivenError(FailureMode FailMode, std::error_code EC,
                                 StringRef Whence = "") {
  if (FailMode == failIfAnyAreInvalid)
    exitWithErrorCode(EC, Whence);
  else
    warn(EC.message(), std::string(Whence));
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

namespace {
/// A remapper from original symbol names to new symbol names based on a file
/// containing a list of mappings from old name to new name.
class SymbolRemapper {
  std::unique_ptr<MemoryBuffer> File;
  DenseMap<StringRef, StringRef> RemappingTable;

public:
  /// Build a SymbolRemapper from a file containing a list of old/new symbols.
  static std::unique_ptr<SymbolRemapper> create(StringRef InputFile) {
    auto BufOrError = MemoryBuffer::getFileOrSTDIN(InputFile);
    if (!BufOrError)
      exitWithErrorCode(BufOrError.getError(), InputFile);

    auto Remapper = std::make_unique<SymbolRemapper>();
    Remapper->File = std::move(BufOrError.get());

    for (line_iterator LineIt(*Remapper->File, /*SkipBlanks=*/true, '#');
         !LineIt.is_at_eof(); ++LineIt) {
      std::pair<StringRef, StringRef> Parts = LineIt->split(' ');
      if (Parts.first.empty() || Parts.second.empty() ||
          Parts.second.count(' ')) {
        exitWithError("unexpected line in remapping file",
                      (InputFile + ":" + Twine(LineIt.line_number())).str(),
                      "expected 'old_symbol new_symbol'");
      }
      Remapper->RemappingTable.insert(Parts);
    }
    return Remapper;
  }

  /// Attempt to map the given old symbol into a new symbol.
  ///
  /// \return The new symbol, or \p Name if no such symbol was found.
  StringRef operator()(StringRef Name) {
    StringRef New = RemappingTable.lookup(Name);
    return New.empty() ? Name : New;
  }
};
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
  std::vector<std::pair<Error, std::string>> Errors;
  std::mutex &ErrLock;
  SmallSet<instrprof_error, 4> &WriterErrorCodes;

  WriterContext(bool IsSparse, std::mutex &ErrLock,
                SmallSet<instrprof_error, 4> &WriterErrorCodes)
      : Lock(), Writer(IsSparse), Errors(), ErrLock(ErrLock),
        WriterErrorCodes(WriterErrorCodes) {}
};

/// Computer the overlap b/w profile BaseFilename and TestFileName,
/// and store the program level result to Overlap.
static void overlapInput(const std::string &BaseFilename,
                         const std::string &TestFilename, WriterContext *WC,
                         OverlapStats &Overlap,
                         const OverlapFuncFilters &FuncFilter,
                         raw_fd_ostream &OS, bool IsCS) {
  auto ReaderOrErr = InstrProfReader::create(TestFilename);
  if (Error E = ReaderOrErr.takeError()) {
    // Skip the empty profiles by returning sliently.
    instrprof_error IPE = InstrProfError::take(std::move(E));
    if (IPE != instrprof_error::empty_raw_profile)
      WC->Errors.emplace_back(make_error<InstrProfError>(IPE), TestFilename);
    return;
  }

  auto Reader = std::move(ReaderOrErr.get());
  for (auto &I : *Reader) {
    OverlapStats FuncOverlap(OverlapStats::FunctionLevel);
    FuncOverlap.setFuncInfo(I.Name, I.Hash);

    WC->Writer.overlapRecord(std::move(I), Overlap, FuncOverlap, FuncFilter);
    FuncOverlap.dump(OS);
  }
}

/// Load an input into a writer context.
static void loadInput(const WeightedFile &Input, SymbolRemapper *Remapper,
                      WriterContext *WC) {
  std::unique_lock<std::mutex> CtxGuard{WC->Lock};

  // Copy the filename, because llvm::ThreadPool copied the input "const
  // WeightedFile &" by value, making a reference to the filename within it
  // invalid outside of this packaged task.
  std::string Filename = Input.Filename;

  auto ReaderOrErr = InstrProfReader::create(Input.Filename);
  if (Error E = ReaderOrErr.takeError()) {
    // Skip the empty profiles by returning sliently.
    instrprof_error IPE = InstrProfError::take(std::move(E));
    if (IPE != instrprof_error::empty_raw_profile)
      WC->Errors.emplace_back(make_error<InstrProfError>(IPE), Filename);
    return;
  }

  auto Reader = std::move(ReaderOrErr.get());
  bool IsIRProfile = Reader->isIRLevelProfile();
  bool HasCSIRProfile = Reader->hasCSIRLevelProfile();
  if (WC->Writer.setIsIRLevelProfile(IsIRProfile, HasCSIRProfile)) {
    WC->Errors.emplace_back(
        make_error<StringError>(
            "Merge IR generated profile with Clang generated profile.",
            std::error_code()),
        Filename);
    return;
  }

  for (auto &I : *Reader) {
    if (Remapper)
      I.Name = (*Remapper)(I.Name);
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
    if (Error E = Reader->getError())
      WC->Errors.emplace_back(std::move(E), Filename);
}

/// Merge the \p Src writer context into \p Dst.
static void mergeWriterContexts(WriterContext *Dst, WriterContext *Src) {
  for (auto &ErrorPair : Src->Errors)
    Dst->Errors.push_back(std::move(ErrorPair));
  Src->Errors.clear();

  Dst->Writer.mergeRecordsFromWriter(std::move(Src->Writer), [&](Error E) {
    instrprof_error IPE = InstrProfError::take(std::move(E));
    std::unique_lock<std::mutex> ErrGuard{Dst->ErrLock};
    bool firstTime = Dst->WriterErrorCodes.insert(IPE).second;
    if (firstTime)
      warn(toString(make_error<InstrProfError>(IPE)));
  });
}

static void mergeInstrProfile(const WeightedFileVector &Inputs,
                              SymbolRemapper *Remapper,
                              StringRef OutputFilename,
                              ProfileFormat OutputFormat, bool OutputSparse,
                              unsigned NumThreads, FailureMode FailMode) {
  if (OutputFilename.compare("-") == 0)
    exitWithError("Cannot write indexed profdata format to stdout.");

  if (OutputFormat != PF_Binary && OutputFormat != PF_Compact_Binary &&
      OutputFormat != PF_Ext_Binary && OutputFormat != PF_Text)
    exitWithError("Unknown format is specified.");

  std::mutex ErrorLock;
  SmallSet<instrprof_error, 4> WriterErrorCodes;

  // If NumThreads is not specified, auto-detect a good default.
  if (NumThreads == 0)
    NumThreads = std::min(hardware_concurrency().compute_thread_count(),
                          unsigned((Inputs.size() + 1) / 2));
  // FIXME: There's a bug here, where setting NumThreads = Inputs.size() fails
  // the merge_empty_profile.test because the InstrProfWriter.ProfileKind isn't
  // merged, thus the emitted file ends up with a PF_Unknown kind.

  // Initialize the writer contexts.
  SmallVector<std::unique_ptr<WriterContext>, 4> Contexts;
  for (unsigned I = 0; I < NumThreads; ++I)
    Contexts.emplace_back(std::make_unique<WriterContext>(
        OutputSparse, ErrorLock, WriterErrorCodes));

  if (NumThreads == 1) {
    for (const auto &Input : Inputs)
      loadInput(Input, Remapper, Contexts[0].get());
  } else {
    ThreadPool Pool(hardware_concurrency(NumThreads));

    // Load the inputs in parallel (N/NumThreads serial steps).
    unsigned Ctx = 0;
    for (const auto &Input : Inputs) {
      Pool.async(loadInput, Input, Remapper, Contexts[Ctx].get());
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

  // Handle deferred errors encountered during merging. If the number of errors
  // is equal to the number of inputs the merge failed.
  unsigned NumErrors = 0;
  for (std::unique_ptr<WriterContext> &WC : Contexts) {
    for (auto &ErrorPair : WC->Errors) {
      ++NumErrors;
      warn(toString(std::move(ErrorPair.first)), ErrorPair.second);
    }
  }
  if (NumErrors == Inputs.size() ||
      (NumErrors > 0 && FailMode == failIfAnyAreInvalid))
    exitWithError("No profiles could be merged.");

  std::error_code EC;
  raw_fd_ostream Output(OutputFilename.data(), EC, sys::fs::OF_None);
  if (EC)
    exitWithErrorCode(EC, OutputFilename);

  InstrProfWriter &Writer = Contexts[0]->Writer;
  if (OutputFormat == PF_Text) {
    if (Error E = Writer.writeText(Output))
      exitWithError(std::move(E));
  } else {
    Writer.write(Output);
  }
}

/// Make a copy of the given function samples with all symbol names remapped
/// by the provided symbol remapper.
static sampleprof::FunctionSamples
remapSamples(const sampleprof::FunctionSamples &Samples,
             SymbolRemapper &Remapper, sampleprof_error &Error) {
  sampleprof::FunctionSamples Result;
  Result.setName(Remapper(Samples.getName()));
  Result.addTotalSamples(Samples.getTotalSamples());
  Result.addHeadSamples(Samples.getHeadSamples());
  for (const auto &BodySample : Samples.getBodySamples()) {
    Result.addBodySamples(BodySample.first.LineOffset,
                          BodySample.first.Discriminator,
                          BodySample.second.getSamples());
    for (const auto &Target : BodySample.second.getCallTargets()) {
      Result.addCalledTargetSamples(BodySample.first.LineOffset,
                                    BodySample.first.Discriminator,
                                    Remapper(Target.first()), Target.second);
    }
  }
  for (const auto &CallsiteSamples : Samples.getCallsiteSamples()) {
    sampleprof::FunctionSamplesMap &Target =
        Result.functionSamplesAt(CallsiteSamples.first);
    for (const auto &Callsite : CallsiteSamples.second) {
      sampleprof::FunctionSamples Remapped =
          remapSamples(Callsite.second, Remapper, Error);
      MergeResult(Error,
                  Target[std::string(Remapped.getName())].merge(Remapped));
    }
  }
  return Result;
}

static sampleprof::SampleProfileFormat FormatMap[] = {
    sampleprof::SPF_None,
    sampleprof::SPF_Text,
    sampleprof::SPF_Compact_Binary,
    sampleprof::SPF_Ext_Binary,
    sampleprof::SPF_GCC,
    sampleprof::SPF_Binary};

static std::unique_ptr<MemoryBuffer>
getInputFileBuf(const StringRef &InputFile) {
  if (InputFile == "")
    return {};

  auto BufOrError = MemoryBuffer::getFileOrSTDIN(InputFile);
  if (!BufOrError)
    exitWithErrorCode(BufOrError.getError(), InputFile);

  return std::move(*BufOrError);
}

static void populateProfileSymbolList(MemoryBuffer *Buffer,
                                      sampleprof::ProfileSymbolList &PSL) {
  if (!Buffer)
    return;

  SmallVector<StringRef, 32> SymbolVec;
  StringRef Data = Buffer->getBuffer();
  Data.split(SymbolVec, '\n', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  for (StringRef symbol : SymbolVec)
    PSL.add(symbol);
}

static void handleExtBinaryWriter(sampleprof::SampleProfileWriter &Writer,
                                  ProfileFormat OutputFormat,
                                  MemoryBuffer *Buffer,
                                  sampleprof::ProfileSymbolList &WriterList,
                                  bool CompressAllSections, bool UseMD5) {
  populateProfileSymbolList(Buffer, WriterList);
  if (WriterList.size() > 0 && OutputFormat != PF_Ext_Binary)
    warn("Profile Symbol list is not empty but the output format is not "
         "ExtBinary format. The list will be lost in the output. ");

  Writer.setProfileSymbolList(&WriterList);

  if (CompressAllSections) {
    if (OutputFormat != PF_Ext_Binary)
      warn("-compress-all-section is ignored. Specify -extbinary to enable it");
    else
      Writer.setToCompressAllSections();
  }
  if (UseMD5) {
    if (OutputFormat != PF_Ext_Binary)
      warn("-use-md5 is ignored. Specify -extbinary to enable it");
    else
      Writer.setUseMD5();
  }
}

static void
mergeSampleProfile(const WeightedFileVector &Inputs, SymbolRemapper *Remapper,
                   StringRef OutputFilename, ProfileFormat OutputFormat,
                   StringRef ProfileSymbolListFile, bool CompressAllSections,
                   bool UseMD5, FailureMode FailMode) {
  using namespace sampleprof;
  StringMap<FunctionSamples> ProfileMap;
  SmallVector<std::unique_ptr<sampleprof::SampleProfileReader>, 5> Readers;
  LLVMContext Context;
  sampleprof::ProfileSymbolList WriterList;
  for (const auto &Input : Inputs) {
    auto ReaderOrErr = SampleProfileReader::create(Input.Filename, Context);
    if (std::error_code EC = ReaderOrErr.getError()) {
      warnOrExitGivenError(FailMode, EC, Input.Filename);
      continue;
    }

    // We need to keep the readers around until after all the files are
    // read so that we do not lose the function names stored in each
    // reader's memory. The function names are needed to write out the
    // merged profile map.
    Readers.push_back(std::move(ReaderOrErr.get()));
    const auto Reader = Readers.back().get();
    if (std::error_code EC = Reader->read()) {
      warnOrExitGivenError(FailMode, EC, Input.Filename);
      Readers.pop_back();
      continue;
    }

    StringMap<FunctionSamples> &Profiles = Reader->getProfiles();
    for (StringMap<FunctionSamples>::iterator I = Profiles.begin(),
                                              E = Profiles.end();
         I != E; ++I) {
      sampleprof_error Result = sampleprof_error::success;
      FunctionSamples Remapped =
          Remapper ? remapSamples(I->second, *Remapper, Result)
                   : FunctionSamples();
      FunctionSamples &Samples = Remapper ? Remapped : I->second;
      StringRef FName = Samples.getName();
      MergeResult(Result, ProfileMap[FName].merge(Samples, Input.Weight));
      if (Result != sampleprof_error::success) {
        std::error_code EC = make_error_code(Result);
        handleMergeWriterError(errorCodeToError(EC), Input.Filename, FName);
      }
    }

    std::unique_ptr<sampleprof::ProfileSymbolList> ReaderList =
        Reader->getProfileSymbolList();
    if (ReaderList)
      WriterList.merge(*ReaderList);
  }
  auto WriterOrErr =
      SampleProfileWriter::create(OutputFilename, FormatMap[OutputFormat]);
  if (std::error_code EC = WriterOrErr.getError())
    exitWithErrorCode(EC, OutputFilename);

  auto Writer = std::move(WriterOrErr.get());
  // WriterList will have StringRef refering to string in Buffer.
  // Make sure Buffer lives as long as WriterList.
  auto Buffer = getInputFileBuf(ProfileSymbolListFile);
  handleExtBinaryWriter(*Writer, OutputFormat, Buffer.get(), WriterList,
                        CompressAllSections, UseMD5);
  Writer->write(ProfileMap);
}

static WeightedFile parseWeightedFile(const StringRef &WeightedFilename) {
  StringRef WeightStr, FileName;
  std::tie(WeightStr, FileName) = WeightedFilename.split(',');

  uint64_t Weight;
  if (WeightStr.getAsInteger(10, Weight) || Weight < 1)
    exitWithError("Input weight must be a positive integer.");

  return {std::string(FileName), Weight};
}

static void addWeightedInput(WeightedFileVector &WNI, const WeightedFile &WF) {
  StringRef Filename = WF.Filename;
  uint64_t Weight = WF.Weight;

  // If it's STDIN just pass it on.
  if (Filename == "-") {
    WNI.push_back({std::string(Filename), Weight});
    return;
  }

  llvm::sys::fs::file_status Status;
  llvm::sys::fs::status(Filename, Status);
  if (!llvm::sys::fs::exists(Status))
    exitWithErrorCode(make_error_code(errc::no_such_file_or_directory),
                      Filename);
  // If it's a source file, collect it.
  if (llvm::sys::fs::is_regular_file(Status)) {
    WNI.push_back({std::string(Filename), Weight});
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
      addWeightedInput(WFV, {std::string(SanitizedEntry), 1});
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
  cl::opt<std::string> RemappingFile("remapping-file", cl::value_desc("file"),
                                     cl::desc("Symbol remapping file"));
  cl::alias RemappingFileA("r", cl::desc("Alias for --remapping-file"),
                           cl::aliasopt(RemappingFile));
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
      cl::values(
          clEnumValN(PF_Binary, "binary", "Binary encoding (default)"),
          clEnumValN(PF_Compact_Binary, "compbinary",
                     "Compact binary encoding"),
          clEnumValN(PF_Ext_Binary, "extbinary", "Extensible binary encoding"),
          clEnumValN(PF_Text, "text", "Text encoding"),
          clEnumValN(PF_GCC, "gcc",
                     "GCC encoding (only meaningful for -sample)")));
  cl::opt<FailureMode> FailureMode(
      "failure-mode", cl::init(failIfAnyAreInvalid), cl::desc("Failure mode:"),
      cl::values(clEnumValN(failIfAnyAreInvalid, "any",
                            "Fail if any profile is invalid."),
                 clEnumValN(failIfAllAreInvalid, "all",
                            "Fail only if all profiles are invalid.")));
  cl::opt<bool> OutputSparse("sparse", cl::init(false),
      cl::desc("Generate a sparse profile (only meaningful for -instr)"));
  cl::opt<unsigned> NumThreads(
      "num-threads", cl::init(0),
      cl::desc("Number of merge threads to use (default: autodetect)"));
  cl::alias NumThreadsA("j", cl::desc("Alias for --num-threads"),
                        cl::aliasopt(NumThreads));
  cl::opt<std::string> ProfileSymbolListFile(
      "prof-sym-list", cl::init(""),
      cl::desc("Path to file containing the list of function symbols "
               "used to populate profile symbol list"));
  cl::opt<bool> CompressAllSections(
      "compress-all-sections", cl::init(false), cl::Hidden,
      cl::desc("Compress all sections when writing the profile (only "
               "meaningful for -extbinary)"));
  cl::opt<bool> UseMD5(
      "use-md5", cl::init(false), cl::Hidden,
      cl::desc("Choose to use MD5 to represent string in name table (only "
               "meaningful for -extbinary)"));

  cl::ParseCommandLineOptions(argc, argv, "LLVM profile data merger\n");

  WeightedFileVector WeightedInputs;
  for (StringRef Filename : InputFilenames)
    addWeightedInput(WeightedInputs, {std::string(Filename), 1});
  for (StringRef WeightedFilename : WeightedInputFilenames)
    addWeightedInput(WeightedInputs, parseWeightedFile(WeightedFilename));

  // Make sure that the file buffer stays alive for the duration of the
  // weighted input vector's lifetime.
  auto Buffer = getInputFileBuf(InputFilenamesFile);
  parseInputFilenamesFile(Buffer.get(), WeightedInputs);

  if (WeightedInputs.empty())
    exitWithError("No input files specified. See " +
                  sys::path::filename(argv[0]) + " -help");

  if (DumpInputFileList) {
    for (auto &WF : WeightedInputs)
      outs() << WF.Weight << "," << WF.Filename << "\n";
    return 0;
  }

  std::unique_ptr<SymbolRemapper> Remapper;
  if (!RemappingFile.empty())
    Remapper = SymbolRemapper::create(RemappingFile);

  if (ProfileKind == instr)
    mergeInstrProfile(WeightedInputs, Remapper.get(), OutputFilename,
                      OutputFormat, OutputSparse, NumThreads, FailureMode);
  else
    mergeSampleProfile(WeightedInputs, Remapper.get(), OutputFilename,
                       OutputFormat, ProfileSymbolListFile, CompressAllSections,
                       UseMD5, FailureMode);

  return 0;
}

/// Computer the overlap b/w profile BaseFilename and profile TestFilename.
static void overlapInstrProfile(const std::string &BaseFilename,
                                const std::string &TestFilename,
                                const OverlapFuncFilters &FuncFilter,
                                raw_fd_ostream &OS, bool IsCS) {
  std::mutex ErrorLock;
  SmallSet<instrprof_error, 4> WriterErrorCodes;
  WriterContext Context(false, ErrorLock, WriterErrorCodes);
  WeightedFile WeightedInput{BaseFilename, 1};
  OverlapStats Overlap;
  Error E = Overlap.accumulateCounts(BaseFilename, TestFilename, IsCS);
  if (E)
    exitWithError(std::move(E), "Error in getting profile count sums");
  if (Overlap.Base.CountSum < 1.0f) {
    OS << "Sum of edge counts for profile " << BaseFilename << " is 0.\n";
    exit(0);
  }
  if (Overlap.Test.CountSum < 1.0f) {
    OS << "Sum of edge counts for profile " << TestFilename << " is 0.\n";
    exit(0);
  }
  loadInput(WeightedInput, nullptr, &Context);
  overlapInput(BaseFilename, TestFilename, &Context, Overlap, FuncFilter, OS,
               IsCS);
  Overlap.dump(OS);
}

static int overlap_main(int argc, const char *argv[]) {
  cl::opt<std::string> BaseFilename(cl::Positional, cl::Required,
                                    cl::desc("<base profile file>"));
  cl::opt<std::string> TestFilename(cl::Positional, cl::Required,
                                    cl::desc("<test profile file>"));
  cl::opt<std::string> Output("output", cl::value_desc("output"), cl::init("-"),
                              cl::desc("Output file"));
  cl::alias OutputA("o", cl::desc("Alias for --output"), cl::aliasopt(Output));
  cl::opt<bool> IsCS("cs", cl::init(false),
                     cl::desc("For context sensitive counts"));
  cl::opt<unsigned long long> ValueCutoff(
      "value-cutoff", cl::init(-1),
      cl::desc(
          "Function level overlap information for every function in test "
          "profile with max count value greater then the parameter value"));
  cl::opt<std::string> FuncNameFilter(
      "function",
      cl::desc("Function level overlap information for matching functions"));
  cl::ParseCommandLineOptions(argc, argv, "LLVM profile data overlap tool\n");

  std::error_code EC;
  raw_fd_ostream OS(Output.data(), EC, sys::fs::OF_Text);
  if (EC)
    exitWithErrorCode(EC, Output);

  overlapInstrProfile(BaseFilename, TestFilename,
                      OverlapFuncFilters{ValueCutoff, FuncNameFilter}, OS,
                      IsCS);

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

    uint64_t SiteSum = 0;
    for (uint32_t V = 0; V < NV; V++)
      SiteSum += VD[V].Count;
    if (SiteSum == 0)
      SiteSum = 1;

    for (uint32_t V = 0; V < NV; V++) {
      OS << "\t[ " << format("%2u", I) << ", ";
      if (Symtab == nullptr)
        OS << format("%4" PRIu64, VD[V].Value);
      else
        OS << Symtab->getFuncName(VD[V].Value);
      OS << ", " << format("%10" PRId64, VD[V].Count) << " ] ("
         << format("%.2f%%", (VD[V].Count * 100.0 / SiteSum)) << ")\n";
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
                            bool ShowAllFunctions, bool ShowCS,
                            uint64_t ValueCutoff, bool OnlyListBelow,
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
  size_t BelowCutoffFunctions = 0;
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

  if (!TextFormat && OnlyListBelow) {
    OS << "The list of functions with the maximum counter less than "
       << ValueCutoff << ":\n";
  }

  // Add marker so that IR-level instrumentation round-trips properly.
  if (TextFormat && IsIRInstr)
    OS << ":ir\n";

  for (const auto &Func : *Reader) {
    if (Reader->isIRLevelProfile()) {
      bool FuncIsCS = NamedInstrProfRecord::hasCSFlagInHash(Func.Hash);
      if (FuncIsCS != ShowCS)
        continue;
    }
    bool Show =
        ShowAllFunctions || (!ShowFunction.empty() &&
                             Func.Name.find(ShowFunction) != Func.Name.npos);

    bool doTextFormatDump = (Show && TextFormat);

    if (doTextFormatDump) {
      InstrProfSymtab &Symtab = Reader->getSymtab();
      InstrProfWriter::writeRecordInText(Func.Name, Func.Hash, Func, Symtab,
                                         OS);
      continue;
    }

    assert(Func.Counts.size() > 0 && "function missing entry counter");
    Builder.addRecord(Func);

    uint64_t FuncMax = 0;
    uint64_t FuncSum = 0;
    for (size_t I = 0, E = Func.Counts.size(); I < E; ++I) {
      FuncMax = std::max(FuncMax, Func.Counts[I]);
      FuncSum += Func.Counts[I];
    }

    if (FuncMax < ValueCutoff) {
      ++BelowCutoffFunctions;
      if (OnlyListBelow) {
        OS << "  " << Func.Name << ": (Max = " << FuncMax
           << " Sum = " << FuncSum << ")\n";
      }
      continue;
    } else if (OnlyListBelow)
      continue;

    if (TopN) {
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

  if (TextFormat)
    return 0;
  std::unique_ptr<ProfileSummary> PS(Builder.getSummary());
  OS << "Instrumentation level: "
     << (Reader->isIRLevelProfile() ? "IR" : "Front-end") << "\n";
  if (ShowAllFunctions || !ShowFunction.empty())
    OS << "Functions shown: " << ShownFunctions << "\n";
  OS << "Total functions: " << PS->getNumFunctions() << "\n";
  if (ValueCutoff > 0) {
    OS << "Number of functions with maximum count (< " << ValueCutoff
       << "): " << BelowCutoffFunctions << "\n";
    OS << "Number of functions with maximum count (>= " << ValueCutoff
       << "): " << PS->getNumFunctions() - BelowCutoffFunctions << "\n";
  }
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

static void showSectionInfo(sampleprof::SampleProfileReader *Reader,
                            raw_fd_ostream &OS) {
  if (!Reader->dumpSectionInfo(OS)) {
    WithColor::warning() << "-show-sec-info-only is only supported for "
                         << "sample profile in extbinary format and is "
                         << "ignored for other formats.\n";
    return;
  }
}

static int showSampleProfile(const std::string &Filename, bool ShowCounts,
                             bool ShowAllFunctions,
                             const std::string &ShowFunction,
                             bool ShowProfileSymbolList,
                             bool ShowSectionInfoOnly, raw_fd_ostream &OS) {
  using namespace sampleprof;
  LLVMContext Context;
  auto ReaderOrErr = SampleProfileReader::create(Filename, Context);
  if (std::error_code EC = ReaderOrErr.getError())
    exitWithErrorCode(EC, Filename);

  auto Reader = std::move(ReaderOrErr.get());

  if (ShowSectionInfoOnly) {
    showSectionInfo(Reader.get(), OS);
    return 0;
  }

  if (std::error_code EC = Reader->read())
    exitWithErrorCode(EC, Filename);

  if (ShowAllFunctions || ShowFunction.empty())
    Reader->dump(OS);
  else
    Reader->dumpFunctionProfile(ShowFunction, OS);

  if (ShowProfileSymbolList) {
    std::unique_ptr<sampleprof::ProfileSymbolList> ReaderList =
        Reader->getProfileSymbolList();
    ReaderList->dump(OS);
  }

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
  cl::opt<bool> ShowCS("showcs", cl::init(false),
                       cl::desc("Show context sensitive counts"));
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
  cl::opt<uint32_t> ValueCutoff(
      "value-cutoff", cl::init(0),
      cl::desc("Set the count value cutoff. Functions with the maximum count "
               "less than this value will not be printed out. (Default is 0)"));
  cl::opt<bool> OnlyListBelow(
      "list-below-cutoff", cl::init(false),
      cl::desc("Only output names of functions whose max count values are "
               "below the cutoff value"));
  cl::opt<bool> ShowProfileSymbolList(
      "show-prof-sym-list", cl::init(false),
      cl::desc("Show profile symbol list if it exists in the profile. "));
  cl::opt<bool> ShowSectionInfoOnly(
      "show-sec-info-only", cl::init(false),
      cl::desc("Show the information of each section in the sample profile. "
               "The flag is only usable when the sample profile is in "
               "extbinary format"));

  cl::ParseCommandLineOptions(argc, argv, "LLVM profile data summary\n");

  if (OutputFilename.empty())
    OutputFilename = "-";

  if (!Filename.compare(OutputFilename)) {
    errs() << sys::path::filename(argv[0])
           << ": Input file name cannot be the same as the output file name!\n";
    return 1;
  }

  std::error_code EC;
  raw_fd_ostream OS(OutputFilename.data(), EC, sys::fs::OF_Text);
  if (EC)
    exitWithErrorCode(EC, OutputFilename);

  if (ShowAllFunctions && !ShowFunction.empty())
    WithColor::warning() << "-function argument ignored: showing all functions\n";

  if (ProfileKind == instr)
    return showInstrProfile(Filename, ShowCounts, TopNFunctions,
                            ShowIndirectCallTargets, ShowMemOPSizes,
                            ShowDetailedSummary, DetailedSummaryCutoffs,
                            ShowAllFunctions, ShowCS, ValueCutoff,
                            OnlyListBelow, ShowFunction, TextFormat, OS);
  else
    return showSampleProfile(Filename, ShowCounts, ShowAllFunctions,
                             ShowFunction, ShowProfileSymbolList,
                             ShowSectionInfoOnly, OS);
}

int main(int argc, const char *argv[]) {
  InitLLVM X(argc, argv);

  StringRef ProgName(sys::path::filename(argv[0]));
  if (argc > 1) {
    int (*func)(int, const char *[]) = nullptr;

    if (strcmp(argv[1], "merge") == 0)
      func = merge_main;
    else if (strcmp(argv[1], "show") == 0)
      func = show_main;
    else if (strcmp(argv[1], "overlap") == 0)
      func = overlap_main;

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
             << "Available commands: merge, show, overlap\n";
      return 0;
    }
  }

  if (argc < 2)
    errs() << ProgName << ": No command specified!\n";
  else
    errs() << ProgName << ": Unknown command!\n";

  errs() << "USAGE: " << ProgName << " <merge|show|overlap> [args...]\n";
  return 1;
}
