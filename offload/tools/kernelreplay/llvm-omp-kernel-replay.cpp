//===- llvm-omp-kernel-replay.cpp - Replay OpenMP offload kernel ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility to replay the execution of recorded OpenMP
// offload kernels.
//
//===----------------------------------------------------------------------===//

#include "Shared/Utils.h"
#include "omptarget.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>

using namespace llvm;

#define TOOL_NAME "llvm-omp-kernel-replay"
#define TOOL_PREFIX "[" TOOL_NAME "]"

cl::OptionCategory ReplayOptions(TOOL_NAME " Options");

/// The filename to read the JSON kernel description.
static cl::opt<std::string> JsonFilename(cl::Positional,
                                         cl::desc("<input kernel JSON file>"),
                                         cl::Required);

static cl::opt<bool> VerifyOpt(
    "verify",
    cl::desc("Verify device memory after replaying against the record output."),
    cl::init(false), cl::cat(ReplayOptions));

static cl::opt<bool> SaveOutputOpt(
    "save-output",
    cl::desc("Save the device memory output of the replayed kernel execution."),
    cl::init(false), cl::cat(ReplayOptions));

static cl::opt<uint32_t> NumTeamsOpt("num-teams",
                                     cl::desc("Set the number of teams."),
                                     cl::init(0), cl::cat(ReplayOptions));

static cl::opt<uint32_t> NumThreadsOpt("num-threads",
                                       cl::desc("Set the number of threads."),
                                       cl::init(0), cl::cat(ReplayOptions));

static cl::opt<int32_t> DeviceIdOpt("device-id", cl::desc("Set the device id."),
                                    cl::init(-1), cl::cat(ReplayOptions));

static cl::opt<uint32_t>
    RepetitionsOpt("repetitions",
                   cl::desc("Set the number of replay repetitions."),
                   cl::init(1), cl::cat(ReplayOptions));

static cl::opt<bool>
    IgnoreLimitsOpt("ignore-limits",
                    cl::desc("Ignore thread and team limits (unrecommended)."),
                    cl::init(false), cl::cat(ReplayOptions));

static cl::opt<bool>
    LoadBitcodeOpt("load-bitcode",
                   cl::desc("Load the recorded IR bitcode image file."),
                   cl::init(false), cl::cat(ReplayOptions));

static cl::opt<std::string> BatchOpt(
    "batch",
    cl::desc("Replay every launch configuration listed in this file within a "
             "single process. Each line holds '<num-teams> <num-threads>' (0 "
             "keeps the recorded value); blank lines and lines starting with "
             "'#' are ignored. The recorded inputs are restored before each "
             "replay."),
    cl::init(""), cl::cat(ReplayOptions));

template <typename... ArgsTy>
Error createErr(const char *ErrFmt, ArgsTy &&...Args) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), ErrFmt,
                                 std::forward<ArgsTy>(Args)...);
}

template <typename T>
Error getInteger(const json::Object *Obj, StringRef Key, T &Result) {
  auto OptInt = Obj->getInteger(Key);
  if (!OptInt)
    return createErr("failed to read JSON integer %s", Key.data());
  Result = static_cast<T>(*OptInt);
  return Error::success();
}

Error getPointer(const json::Object *Obj, StringRef Key, void *&Result) {
  auto OptInt = Obj->getInteger(Key);
  if (!OptInt)
    return createErr("failed to read JSON integer %s", Key.data());
  Result = reinterpret_cast<void *>(*OptInt);
  return Error::success();
}

Error getString(const json::Object *Obj, StringRef Key, StringRef &Result) {
  auto OptStr = Obj->getString(Key);
  if (!OptStr)
    return createErr("failed to read JSON string %s", Key.data());
  Result = *OptStr;
  return Error::success();
}

template <typename Func>
Error processIntegerArray(const json::Object *Obj, StringRef Key,
                          Func ProcessFunc) {
  auto Array = Obj->getArray(Key);
  if (!Array)
    return createErr("failed to read JSON array %s", Key.data());

  for (const auto &Val : *Array) {
    if (auto OptInt = Val.getAsInteger())
      ProcessFunc(*OptInt);
    else
      return createErr("failed to read an integer from JSON array %s",
                       Key.data());
  }
  return Error::success();
}

/// A launch configuration to replay. Zero keeps the recorded value.
struct LaunchConfigTy {
  uint32_t NumTeams = 0;
  uint32_t NumThreads = 0;
};

/// Read the launch configurations of a batch replay from \p Filename.
static Error readBatchFile(const std::string &Filename,
                           SmallVectorImpl<LaunchConfigTy> &Configs) {
  auto BatchBufferOrErr = MemoryBuffer::getFile(Filename, /*IsText=*/true);
  if (std::error_code EC = BatchBufferOrErr.getError())
    return createErr("failed to read the batch file %s: %s", Filename.c_str(),
                     EC.message().c_str());

  SmallVector<StringRef> Lines;
  (*BatchBufferOrErr)->getBuffer().split(Lines, '\n');
  for (auto [LineNo, RawLine] : enumerate(Lines)) {
    StringRef Line = RawLine.trim();
    if (Line.empty() || Line.starts_with("#"))
      continue;

    SmallVector<StringRef, 2> Fields;
    SplitString(Line, Fields);
    LaunchConfigTy Config;
    if (Fields.size() != 2 || Fields[0].getAsInteger(10, Config.NumTeams) ||
        Fields[1].getAsInteger(10, Config.NumThreads))
      return createErr("invalid batch file line %zu: expected '<num-teams> "
                       "<num-threads>'",
                       LineNo + 1);
    Configs.push_back(Config);
  }
  if (Configs.empty())
    return createErr("the batch file %s has no launch configurations",
                     Filename.c_str());
  return Error::success();
}

/// Check that a launch configuration respects the recorded teams and threads
/// limits (min and max), unless they are zero or ignored.
static Error checkLimits(const LaunchConfigTy &Config,
                         ArrayRef<uint32_t> TeamsLimits,
                         ArrayRef<uint32_t> ThreadsLimits) {
  if (IgnoreLimitsOpt)
    return Error::success();
  if (TeamsLimits[0] > 0 &&
      (Config.NumTeams < TeamsLimits[0] || Config.NumTeams > TeamsLimits[1]))
    return createErr("number of teams (%" PRIu32
                     ") is out of the allowed limits (min,max: %" PRIu32
                     ",%" PRIu32 ")",
                     Config.NumTeams, TeamsLimits[0], TeamsLimits[1]);
  if (ThreadsLimits[0] > 0 && (Config.NumThreads < ThreadsLimits[0] ||
                               Config.NumThreads > ThreadsLimits[1]))
    return createErr("number of threads (%" PRIu32
                     ") is out of the allowed limits (min,max: %" PRIu32
                     ",%" PRIu32 ")",
                     Config.NumThreads, ThreadsLimits[0], ThreadsLimits[1]);
  return Error::success();
}

/// Verify that the replay output is the same as the record output.
Error verifyReplayOutput(StringRef RecordOutputFilename,
                         StringRef ReplayOutputFilename) {
  // Load the record output file.
  auto RecordOutputBufferOrErr =
      MemoryBuffer::getFile(RecordOutputFilename,
                            /*isText=*/false,
                            /*RequiresNullTerminator=*/false);
  if (!RecordOutputBufferOrErr)
    return createErr("failed to read the kernel record output file");

  // Load the replay output file.
  auto ReplayOutputBufferOrErr =
      MemoryBuffer::getFile(ReplayOutputFilename,
                            /*isText=*/false,
                            /*RequiresNullTerminator=*/false);
  if (!ReplayOutputBufferOrErr)
    return createErr("failed to read the kernel replay output file");

  // Compare record and replay outputs to verify they match. If they are both
  // empty, the verification is successful.
  StringRef RecordOutput = RecordOutputBufferOrErr.get()->getBuffer();
  StringRef ReplayOutput = ReplayOutputBufferOrErr.get()->getBuffer();
  if (RecordOutput != ReplayOutput)
    return createErr("replay device memory failed to verify");

  // Sucessfully verified.
  return Error::success();
}

/// Replay the kernel and return whether verification occurred.
Error replayKernel() {
  if (RepetitionsOpt == 0)
    return createErr("invalid number of repetitions");

  // Load the kernel descriptor JSON file.
  auto KernelDescrBufferOrErr =
      MemoryBuffer::getFile(JsonFilename, /*isText=*/true,
                            /*RequiresNullTerminator=*/true);
  if (!KernelDescrBufferOrErr)
    return createErr("failed read the kernel info JSON file");

  // Parse the JSON file.
  auto JsonDescrOrErr = json::parse(KernelDescrBufferOrErr.get()->getBuffer());
  if (!JsonDescrOrErr)
    return JsonDescrOrErr.takeError();

  auto JsonObj = JsonDescrOrErr->getAsObject();
  if (!JsonObj)
    return createErr("invalid JSON file");

  // Retrieve the values from the JSON file.
  uint32_t NumTeams, NumThreads, SharedMemorySize, DeviceId, NumArgs;
  if (auto Err = getInteger(JsonObj, "NumTeams", NumTeams))
    return Err;
  if (auto Err = getInteger(JsonObj, "NumThreads", NumThreads))
    return Err;
  if (auto Err = getInteger(JsonObj, "SharedMemorySize", SharedMemorySize))
    return Err;
  if (auto Err = getInteger(JsonObj, "DeviceId", DeviceId))
    return Err;
  if (auto Err = getInteger(JsonObj, "NumArgs", NumArgs))
    return Err;

  uint64_t LoopTripCount, VAllocSize;
  if (auto Err = getInteger(JsonObj, "VAllocSize", VAllocSize))
    return Err;
  if (auto Err = getInteger(JsonObj, "LoopTripCount", LoopTripCount))
    return Err;

  void *VAllocAddr;
  if (auto Err = getPointer(JsonObj, "VAllocAddr", VAllocAddr))
    return Err;

  StringRef KernelName;
  if (auto Err = getString(JsonObj, "Name", KernelName))
    return Err;

  // Select the launch configurations to replay: the ones in the batch file or
  // the one from the command line. Zero values keep the recorded ones.
  const bool IsBatch = !BatchOpt.empty();
  if (IsBatch && (NumTeamsOpt > 0 || NumThreadsOpt > 0))
    return createErr("--batch cannot be combined with --num-teams or "
                     "--num-threads");

  SmallVector<LaunchConfigTy> Configs;
  if (IsBatch) {
    if (auto Err = readBatchFile(BatchOpt, Configs))
      return Err;
  } else {
    Configs.push_back({NumTeamsOpt, NumThreadsOpt});
  }
  for (auto &Config : Configs) {
    Config.NumTeams = Config.NumTeams > 0 ? Config.NumTeams : NumTeams;
    Config.NumThreads = Config.NumThreads > 0 ? Config.NumThreads : NumThreads;
  }

  // If needed, adjust the device identifier.
  DeviceId = DeviceIdOpt >= 0 ? DeviceIdOpt : DeviceId;

  // Retrieve the teams and threads limits (min and max).
  SmallVector<uint32_t> TeamsLimits;
  auto Err = processIntegerArray(JsonObj, "TeamsLimits", [&](uint64_t Val) {
    TeamsLimits.push_back(static_cast<uint32_t>(Val));
  });
  if (Err)
    return Err;

  SmallVector<uint32_t> ThreadsLimits;
  Err = processIntegerArray(JsonObj, "ThreadsLimits", [&](uint64_t Val) {
    ThreadsLimits.push_back(static_cast<uint32_t>(Val));
  });
  if (Err)
    return Err;

  // Check that a minimum and maximum have been exported.
  if (TeamsLimits.size() != 2 || ThreadsLimits.size() != 2)
    return createErr("TeamsLimits and ThreadsLimits must have a min and max");

  // Check that the minimum and maximum are specified or both are zero.
  if (bool(TeamsLimits[0]) != bool(TeamsLimits[1]))
    return createErr("TeamsLimits min and max are inconsistent");
  if (bool(ThreadsLimits[0]) != bool(ThreadsLimits[1]))
    return createErr("ThreadsLimits min and max are inconsistent");

  // If the limits were specified, verify the selected values are valid. A batch
  // replay checks each configuration before replaying it instead.
  if (!IsBatch)
    if (auto Err = checkLimits(Configs[0], TeamsLimits, ThreadsLimits))
      return Err;

  // Retrieve the arguments of the kernel.
  SmallVector<void *> TgtArgs;
  Err = processIntegerArray(JsonObj, "ArgPtrs", [&](uint64_t Val) {
    TgtArgs.push_back(reinterpret_cast<void *>(Val));
  });
  if (Err)
    return Err;

  SmallVector<ptrdiff_t> TgtArgOffsets;
  Err = processIntegerArray(JsonObj, "ArgOffsets", [&](uint64_t Val) {
    TgtArgOffsets.push_back(static_cast<ptrdiff_t>(Val));
  });
  if (Err)
    return Err;

  // Keep the filepath and directory for future use.
  auto Filepath = std::filesystem::path(JsonFilename.getValue());
  auto Directory = Filepath.parent_path();

  // Load the recorded globals file.
  Filepath.replace_extension("globals");
  auto GlobalsBufferOrErr =
      MemoryBuffer::getFile(Filepath.c_str(), /*isText=*/false,
                            /*RequiresNullTerminator=*/false);
  if (!GlobalsBufferOrErr)
    return createErr("failed to read the globals file");
  auto GlobalsBuffer = std::move(GlobalsBufferOrErr.get());

  const void *BufferPtr = const_cast<char *>(GlobalsBuffer->getBufferStart());
  uint32_t NumGlobals = *((const uint32_t *)(BufferPtr));
  BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint32_t));

  SmallVector<llvm::offloading::EntryTy> OffloadEntries(
      NumGlobals + 1, {0x0, 0x1, object::OffloadKind::OFK_OpenMP, 0, nullptr,
                       nullptr, 0, 0, nullptr});

  // The first offload entry corresponds to the kernel function.
  OffloadEntries[0].SymbolName = const_cast<char *>(KernelName.data());
  // Use a unique identifier.
  OffloadEntries[0].Address = (void *)0x1;

  // The rest of entries correspond to the recorded global variables.
  for (uint32_t I = 0; I < NumGlobals; ++I) {
    auto &Global = OffloadEntries[I + 1];

    // Use a unique identifier.
    Global.Address = static_cast<char *>(OffloadEntries[0].Address) + I + 1;

    // Setup the offload entry using the information from the file.
    uint32_t NameSize = *((const uint32_t *)(BufferPtr));
    BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint32_t));
    uint64_t Size = *((const uint64_t *)(BufferPtr));
    BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint64_t));
    Global.Size = Size;
    Global.SymbolName =
        const_cast<char *>(static_cast<const char *>(BufferPtr));
    BufferPtr = utils::advancePtr(BufferPtr, NameSize);
    Global.AuxAddr = const_cast<void *>(BufferPtr);
    BufferPtr = utils::advancePtr(BufferPtr, Size);
  }

  // Load the device image file.
  Filepath.replace_extension(LoadBitcodeOpt ? "bc" : "image");
  auto ImageBufferOrErr =
      MemoryBuffer::getFile(Filepath.c_str(), /*isText=*/false,
                            /*RequiresNullTerminator=*/false);
  if (!ImageBufferOrErr)
    return createErr("failed to read the kernel image file");
  auto ImageBuffer = std::move(ImageBufferOrErr.get());

  // Prepare the device image and binary descriptor.
  __tgt_device_image DeviceImage;
  DeviceImage.ImageStart = const_cast<char *>(ImageBuffer->getBufferStart());
  DeviceImage.ImageEnd = const_cast<char *>(ImageBuffer->getBufferEnd());
  DeviceImage.EntriesBegin = &OffloadEntries[0];
  DeviceImage.EntriesEnd = &OffloadEntries[OffloadEntries.size() - 1] + 1;

  __tgt_bin_desc Desc;
  Desc.NumDeviceImages = 1;
  Desc.HostEntriesBegin = &OffloadEntries[0];
  Desc.HostEntriesEnd = &OffloadEntries[OffloadEntries.size() - 1] + 1;
  Desc.DeviceImages = &DeviceImage;

  // Register the image and the offload entries.
  __tgt_register_lib(&Desc);

  int Rc = __tgt_activate_record_replay(
      DeviceId, VAllocSize, VAllocAddr, /*IsRecord=*/false,
      VerifyOpt || SaveOutputOpt,
      /*EmitReport=*/false, Directory.c_str());
  if (Rc != OMP_TGT_SUCCESS)
    return createErr("failed to activate record replay");

  // Load the record input file.
  Filepath.replace_extension("record_input");
  auto RecordInputBufferOrErr =
      MemoryBuffer::getFile(Filepath.c_str(), /*isText=*/false,
                            /*RequiresNullTerminator=*/false);
  if (!RecordInputBufferOrErr)
    return createErr("failed to read the kernel record input file");
  auto RecordInputBuffer = std::move(RecordInputBufferOrErr.get());

  Filepath.replace_extension("record_output");
  const std::string RecordOutputFilename = Filepath.string();

  // The device allocation is kept in the outcome and reused by every replay;
  // each replay restores the recorded inputs and globals before launching.
  KernelReplayOutcomeTy Outcome;
  uint32_t NumFailed = 0;

  for (size_t Idx = 0; Idx < Configs.size(); ++Idx) {
    const LaunchConfigTy &Config = Configs[Idx];

    // In a batch, report a configuration error and continue with the next one.
    auto HandleConfigError = [&](Error Err) -> Error {
      if (!IsBatch)
        return Err;
      errs() << TOOL_PREFIX << " Error: configuration " << Idx << ": "
             << toString(std::move(Err)) << "\n";
      outs() << "||ReplayError|| index: " << Idx << "\n";
      outs().flush();
      ++NumFailed;
      return Error::success();
    };

    if (IsBatch) {
      outs() << "||ReplayConfig|| index: " << Idx
             << " num_teams: " << Config.NumTeams
             << " num_threads: " << Config.NumThreads << "\n";
      if (auto Err = checkLimits(Config, TeamsLimits, ThreadsLimits)) {
        if (auto Err2 = HandleConfigError(std::move(Err)))
          return Err2;
        continue;
      }
    }

    // Perform the kernel replay for each repetition.
    SmallVector<uint64_t> TimesNs;
    Outcome.OutputFilepath.clear();
    for (uint32_t R = 1; R <= RepetitionsOpt; ++R) {
      Rc = __tgt_target_kernel_replay(
          /*Loc=*/nullptr, DeviceId, OffloadEntries[0].Address,
          const_cast<char *>(RecordInputBuffer->getBufferStart()),
          Outcome.ReplayDeviceAlloc, RecordInputBuffer->getBufferSize(),
          NumGlobals ? &OffloadEntries[1] : nullptr, NumGlobals, TgtArgs.data(),
          TgtArgOffsets.data(), NumArgs, Config.NumTeams, Config.NumThreads,
          SharedMemorySize, LoopTripCount, &Outcome);
      // A failed launch may leave the device in an unknown state, so it stops
      // a batch too.
      if (Rc != OMP_TGT_SUCCESS) {
        if (IsBatch)
          return createErr("failed to replay kernel (configuration %zu)", Idx);
        return createErr("failed to replay kernel");
      }

      TimesNs.push_back(Outcome.KernelReplayTimeNs);
      outs() << TOOL_PREFIX << " Replay time (" << R
             << "): " << Outcome.KernelReplayTimeNs << " ns\n";
    }

    if (IsBatch) {
      // Report the median replay time of the configuration.
      llvm::sort(TimesNs);
      size_t N = TimesNs.size();
      double MedianNs =
          N % 2 ? TimesNs[N / 2] : 0.5 * (TimesNs[N / 2 - 1] + TimesNs[N / 2]);
      outs() << "||ReplayTiming|| duration_seconds: "
             << format("%.9g", MedianNs / 1.0e9) << "\n";
    }

    // Verify the replay output if requested.
    if (VerifyOpt) {
      Error Err = Outcome.OutputFilepath.empty()
                      ? createErr("replay output file was not generated")
                      : verifyReplayOutput(RecordOutputFilename,
                                           Outcome.OutputFilepath.c_str());
      if (Err) {
        if (auto Err2 = HandleConfigError(std::move(Err)))
          return Err2;
        continue;
      }

      // The verification was successful.
      outs() << TOOL_PREFIX << " Replay done, device memory verified\n";
    } else {
      outs() << TOOL_PREFIX << " Replay done, verification skipped\n";
    }
    outs().flush();
  }

  if (NumFailed)
    return createErr("%" PRIu32 " of %zu batch configurations failed",
                     NumFailed, Configs.size());
  return Error::success();
}

int main(int Argc, char **Argv) {
  cl::HideUnrelatedOptions(ReplayOptions);
  cl::ParseCommandLineOptions(Argc, Argv, TOOL_NAME "\n");

  if (auto Err = replayKernel()) {
    errs() << TOOL_PREFIX << " Error: " << llvm::toString(std::move(Err))
           << "\n";
    return 1;
  }
  return 0;
}
