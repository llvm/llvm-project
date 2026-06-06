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

#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
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

  // Compare record and replay outputs to verify they match.
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

  // If needed, adjust number of teams and threads, and the device identifier.
  NumTeams = NumTeamsOpt > 0 ? NumTeamsOpt : NumTeams;
  NumThreads = NumThreadsOpt > 0 ? NumThreadsOpt : NumThreads;
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

  if (TeamsLimits.size() != 2 || ThreadsLimits.size() != 2)
    return createErr("TeamsLimits and ThreadsLimits must have a min and max");

  // If the limits were specified, verify the selected values are valid.
  if (TeamsLimits[0] > 0 &&
      (NumTeams < TeamsLimits[0] || NumTeams > TeamsLimits[1]))
    return createErr("number of teams is out of the allowed limits");
  if (ThreadsLimits[0] > 0 &&
      (NumThreads < ThreadsLimits[0] || NumThreads > ThreadsLimits[1]))
    return createErr("number of threads is out of the allowed limits");

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
  Filepath.replace_extension("image");
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

  KernelReplayOutcomeTy Outcome;

  // Perform the kernel replay and verification (if needed) for each repetition.
  for (uint32_t R = 1; R <= RepetitionsOpt; ++R) {
    Rc = __tgt_target_kernel_replay(
        /*Loc=*/nullptr, DeviceId, OffloadEntries[0].Address,
        const_cast<char *>(RecordInputBuffer->getBufferStart()),
        R > 0 ? Outcome.ReplayDeviceAlloc : nullptr,
        RecordInputBuffer->getBufferSize(),
        NumGlobals ? &OffloadEntries[1] : nullptr, NumGlobals, TgtArgs.data(),
        TgtArgOffsets.data(), NumArgs, NumTeams, NumThreads, SharedMemorySize,
        LoopTripCount, &Outcome);
    if (Rc != OMP_TGT_SUCCESS)
      return createErr("failed to replay kernel");

    outs() << TOOL_PREFIX << " Replay time (" << R
           << "): " << Outcome.KernelReplayTimeNs << " ns\n";
  }

  // Verify the replay output if requested.
  if (VerifyOpt) {
    if (Outcome.OutputFilepath.empty())
      return createErr("replay output file was not generated");

    Filepath.replace_extension("record_output");
    if (auto Err = verifyReplayOutput(Filepath.c_str(),
                                      Outcome.OutputFilepath.c_str()))
      return Err;

    // The verification was successful.
    outs() << TOOL_PREFIX << " Replay done, device memory verified\n";
  } else {
    outs() << TOOL_PREFIX << " Replay done, verification skipped\n";
  }
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
