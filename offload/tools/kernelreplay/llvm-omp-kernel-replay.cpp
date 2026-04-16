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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>

using namespace llvm;

cl::OptionCategory ReplayOptions("llvm-omp-kernel-replay Options");

// InputFilename - The filename to read the json description of the kernel.
static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input kernel json file>"),
                                          cl::Required);

static cl::opt<bool> VerifyOpt(
    "verify",
    cl::desc(
        "Verify device memory post execution against the original output."),
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

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions(ReplayOptions);
  cl::ParseCommandLineOptions(argc, argv, "llvm-omp-kernel-replay\n");

  ErrorOr<std::unique_ptr<MemoryBuffer>> KernelInfoMB =
      MemoryBuffer::getFile(InputFilename, /*isText=*/true,
                            /*RequiresNullTerminator=*/true);
  if (!KernelInfoMB)
    reportFatalUsageError("Error reading the kernel info json file");
  Expected<json::Value> JsonKernelInfo =
      json::parse(KernelInfoMB.get()->getBuffer());
  if (auto Err = JsonKernelInfo.takeError())
    reportFatalUsageError("Cannot parse the kernel info json file");

  auto NumTeamsJson =
      JsonKernelInfo->getAsObject()->getInteger("NumTeamsClause");
  uint32_t NumTeams = (NumTeamsOpt > 0 ? NumTeamsOpt : NumTeamsJson.value());
  auto NumThreadsJson =
      JsonKernelInfo->getAsObject()->getInteger("ThreadLimitClause");
  uint32_t NumThreads =
      (NumThreadsOpt > 0 ? NumThreadsOpt : NumThreadsJson.value());
  uint32_t SharedMemorySize =
      JsonKernelInfo->getAsObject()->getInteger("SharedMemorySize").value();
  // TODO: Print a warning if number of teams/threads is explicitly set in the
  // kernel info but overridden through command line options.
  auto LoopTripCount =
      JsonKernelInfo->getAsObject()->getInteger("LoopTripCount");
  auto KernelFunc = JsonKernelInfo->getAsObject()->getString("Name");
  std::string KernelName = KernelFunc.value().str();

  SmallVector<void *> TgtArgs;
  SmallVector<ptrdiff_t> TgtArgOffsets;
  auto NumArgs = JsonKernelInfo->getAsObject()->getInteger("NumArgs");
  auto *TgtArgsArray = JsonKernelInfo->getAsObject()->getArray("ArgPtrs");
  for (auto It : *TgtArgsArray)
    TgtArgs.push_back(reinterpret_cast<void *>(It.getAsInteger().value()));
  auto *TgtArgOffsetsArray =
      JsonKernelInfo->getAsObject()->getArray("ArgOffsets");
  for (auto It : *TgtArgOffsetsArray)
    TgtArgOffsets.push_back(static_cast<ptrdiff_t>(It.getAsInteger().value()));

  void *VAllocAddr = reinterpret_cast<void *>(
      JsonKernelInfo->getAsObject()->getInteger("VAllocAddr").value());
  uint64_t VAllocSize =
      JsonKernelInfo->getAsObject()->getInteger("VAllocSize").value();

  auto Filepath = std::filesystem::path(InputFilename.getValue());
  auto Directory = Filepath.parent_path();

  Filepath.replace_extension("globals");
  ErrorOr<std::unique_ptr<MemoryBuffer>> GlobalsMB =
      MemoryBuffer::getFile(Filepath.string(), /*isText=*/false,
                            /*RequiresNullTerminator=*/false);

  if (!GlobalsMB)
    reportFatalUsageError("Error reading the globals file");

  // On AMD for currently unknown reasons we cannot copy memory mapped data to
  // device. This is a work-around.
  uint8_t *RecordedGlobals = new uint8_t[GlobalsMB.get()->getBufferSize()];
  std::memcpy(RecordedGlobals,
              const_cast<char *>(GlobalsMB.get()->getBuffer().data()),
              GlobalsMB.get()->getBufferSize());

  void *BufferPtr = (void *)RecordedGlobals;
  uint32_t NumGlobals = *((uint32_t *)(BufferPtr));
  BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint32_t));

  llvm::SmallVector<llvm::offloading::EntryTy> OffloadEntries(
      NumGlobals + 1, {0x0, 0x1, object::OffloadKind::OFK_OpenMP, 0, nullptr,
                       nullptr, 0, 0, nullptr});

  OffloadEntries[0].SymbolName = const_cast<char *>(KernelName.c_str());
  // Anything non-zero works to uniquely identify the kernel.
  OffloadEntries[0].Address = (void *)0x1;

  for (uint32_t I = 0; I < NumGlobals; ++I) {
    auto &Global = OffloadEntries[I + 1];

    // Use a unique identifier.
    Global.Address = static_cast<char *>(OffloadEntries[0].Address) + I + 1;

    uint32_t NameSize = *((uint32_t *)(BufferPtr));
    BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint32_t));
    uint64_t Size = *((uint64_t *)(BufferPtr));
    BufferPtr = utils::advancePtr(BufferPtr, sizeof(uint64_t));
    Global.Size = Size;
    Global.SymbolName = (char *)BufferPtr;
    BufferPtr = utils::advancePtr(BufferPtr, NameSize);
    Global.AuxAddr = BufferPtr;
    BufferPtr = utils::advancePtr(BufferPtr, Size);
  }

  Filepath.replace_extension("image");
  ErrorOr<std::unique_ptr<MemoryBuffer>> ImageMB =
      MemoryBuffer::getFile(Filepath.string(), /*isText=*/false,
                            /*RequiresNullTerminator=*/false);
  if (!ImageMB)
    reportFatalUsageError("Error reading the kernel image file");

  __tgt_device_image DeviceImage;
  DeviceImage.ImageStart = const_cast<char *>(ImageMB.get()->getBufferStart());
  DeviceImage.ImageEnd = const_cast<char *>(ImageMB.get()->getBufferEnd());
  DeviceImage.EntriesBegin = &OffloadEntries[0];
  DeviceImage.EntriesEnd = &OffloadEntries[OffloadEntries.size() - 1] + 1;

  __tgt_bin_desc Desc;
  Desc.NumDeviceImages = 1;
  Desc.HostEntriesBegin = &OffloadEntries[0];
  Desc.HostEntriesEnd = &OffloadEntries[OffloadEntries.size() - 1] + 1;
  Desc.DeviceImages = &DeviceImage;

  auto DeviceIdJson = JsonKernelInfo->getAsObject()->getInteger("DeviceId");
  // TODO: Print warning if the user overrides the device id in the json file.
  int32_t DeviceId = (DeviceIdOpt > -1 ? DeviceIdOpt : DeviceIdJson.value());

  // TODO: do we need requires?
  //__tgt_register_requires(/*Flags=*/1);

  __tgt_register_lib(&Desc);

  int Rc = __tgt_activate_record_replay(
      DeviceId, VAllocSize, VAllocAddr, /*IsRecord=*/false, VerifyOpt,
      /*EmitReport=*/false, Directory.c_str());
  if (Rc != OMP_TGT_SUCCESS)
    reportFatalUsageError("Error activating record replay");

  Filepath.replace_extension("record_input");
  ErrorOr<std::unique_ptr<MemoryBuffer>> DeviceMemoryMB =
      MemoryBuffer::getFile(Filepath.string(), /*isText=*/false,
                            /*RequiresNullTerminator=*/false);

  if (!DeviceMemoryMB)
    reportFatalUsageError("Error reading the kernel record input file");

  // On AMD for currently unknown reasons we cannot copy memory mapped data to
  // device. This is a work-around.
  uint8_t *RecordedData = new uint8_t[DeviceMemoryMB.get()->getBufferSize()];
  std::memcpy(RecordedData,
              const_cast<char *>(DeviceMemoryMB.get()->getBuffer().data()),
              DeviceMemoryMB.get()->getBufferSize());

  Rc = __tgt_target_kernel_replay(
      /*Loc=*/nullptr, DeviceId, OffloadEntries[0].Address,
      (char *)RecordedData, DeviceMemoryMB.get()->getBufferSize(),
      NumGlobals ? &OffloadEntries[1] : nullptr, NumGlobals, TgtArgs.data(),
      TgtArgOffsets.data(), NumArgs.value(), NumTeams, NumThreads,
      SharedMemorySize, LoopTripCount.value());
  if (Rc != OMP_TGT_SUCCESS)
    reportFatalUsageError("Error replaying kernel");

  int ErrorDetected = 0;
  if (VerifyOpt) {
    Filepath.replace_extension("record_output");
    ErrorOr<std::unique_ptr<MemoryBuffer>> OriginalOutputMB =
        MemoryBuffer::getFile(Filepath.string(),
                              /*isText=*/false,
                              /*RequiresNullTerminator=*/false);
    if (!OriginalOutputMB)
      reportFatalUsageError(
          "Error reading the kernel record output file. Make sure "
          "LIBOMPTARGET_RECORD_OUTPUT is set when recording");

    Filepath.replace_extension("replay_output");
    ErrorOr<std::unique_ptr<MemoryBuffer>> ReplayOutputMB =
        MemoryBuffer::getFile(Filepath.string(),
                              /*isText=*/false,
                              /*RequiresNullTerminator=*/false);
    if (!ReplayOutputMB)
      reportFatalUsageError("Error reading the kernel replay output file");

    StringRef OriginalOutput = OriginalOutputMB.get()->getBuffer();
    StringRef ReplayOutput = ReplayOutputMB.get()->getBuffer();
    if (OriginalOutput == ReplayOutput) {
      outs() << "[llvm-omp-kernel-replay] Replay device memory verified!\n";
    } else {
      ErrorDetected = 1;
      outs() << "[llvm-omp-kernel-replay] Replay device memory failed to "
                "verify!\n";
    }
  }

  delete[] RecordedData;

  return ErrorDetected;
}
