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

#include "omptargetplugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdlib>

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

static cl::opt<unsigned> NumTeamsOpt("num-teams",
                                     cl::desc("Set the number of teams."),
                                     cl::init(0), cl::cat(ReplayOptions));

static cl::opt<unsigned> NumThreadsOpt("num-threads",
                                       cl::desc("Set the number of threads."),
                                       cl::init(0), cl::cat(ReplayOptions));

static cl::opt<int32_t> DeviceIdOpt("device-id", cl::desc("Set the device id."),
                                    cl::init(-1), cl::cat(ReplayOptions));

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions(ReplayOptions);
  cl::ParseCommandLineOptions(argc, argv, "llvm-omp-kernel-replay\n");

  ErrorOr<std::unique_ptr<MemoryBuffer>> KernelInfoMB =
      MemoryBuffer::getFile(InputFilename, /* isText */ true,
                            /* RequiresNullTerminator */ true);
  if (!KernelInfoMB)
    report_fatal_error("Error reading the kernel info json file");
  Expected<json::Value> JsonKernelInfo =
      json::parse(KernelInfoMB.get()->getBuffer());
  if (auto Err = JsonKernelInfo.takeError())
    report_fatal_error("Cannot parse the kernel info json file");

  auto NumTeamsJson =
      JsonKernelInfo->getAsObject()->getInteger("NumTeamsClause");
  unsigned NumTeams = (NumTeamsOpt > 0 ? NumTeamsOpt : NumTeamsJson.value());
  auto NumThreadsJson =
      JsonKernelInfo->getAsObject()->getInteger("ThreadLimitClause");
  unsigned NumThreads =
      (NumThreadsOpt > 0 ? NumThreadsOpt : NumThreadsJson.value());
  // TODO: Print a warning if number of teams/threads is explicitly set in the
  // kernel info but overriden through command line options.
  auto LoopTripCount =
      JsonKernelInfo->getAsObject()->getInteger("LoopTripCount");
  auto KernelFunc = JsonKernelInfo->getAsObject()->getString("Name");

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

  __tgt_offload_entry KernelEntry = {nullptr, nullptr, 0, 0, 0};
  std::string KernelEntryName = KernelFunc.value().str();
  KernelEntry.name = const_cast<char *>(KernelEntryName.c_str());
  // Anything non-zero works to uniquely identify the kernel.
  KernelEntry.addr = (void *)0x1;

  ErrorOr<std::unique_ptr<MemoryBuffer>> ImageMB =
      MemoryBuffer::getFile(KernelEntryName + ".image", /* isText */ false,
                            /* RequiresNullTerminator */ false);
  if (!ImageMB)
    report_fatal_error("Error reading the kernel image.");

  __tgt_device_image DeviceImage;
  DeviceImage.ImageStart = const_cast<char *>(ImageMB.get()->getBufferStart());
  DeviceImage.ImageEnd = const_cast<char *>(ImageMB.get()->getBufferEnd());
  DeviceImage.EntriesBegin = &KernelEntry;
  DeviceImage.EntriesEnd = &KernelEntry + 1;

  __tgt_bin_desc Desc;
  Desc.NumDeviceImages = 1;
  Desc.HostEntriesBegin = &KernelEntry;
  Desc.HostEntriesEnd = &KernelEntry + 1;
  Desc.DeviceImages = &DeviceImage;

  ErrorOr<std::unique_ptr<MemoryBuffer>> DeviceMemoryMB =
      MemoryBuffer::getFile(KernelEntryName + ".memory", /* isText */ false,
                            /* RequiresNullTerminator */ false);
  if (!DeviceMemoryMB)
    report_fatal_error("Error reading the kernel input device memory.");

  setenv("LIBOMPTARGET_REPLAY", "1", 1);
  if (VerifyOpt || SaveOutputOpt)
    setenv("LIBOMPTARGET_RR_SAVE_OUTPUT", "1", 1);

  auto DeviceMemorySizeJson =
      JsonKernelInfo->getAsObject()->getInteger("DeviceMemorySize");
  // Set device memory size to the ceiling of GB granularity.
  uint64_t DeviceMemorySize =
      std::ceil(DeviceMemorySizeJson.value() / (1024.0 * 1024.0 * 1024.0));

  setenv("LIBOMPTARGET_RR_DEVMEM_SIZE",
         std::to_string(DeviceMemorySize).c_str(), 1);

  auto DeviceIdJson = JsonKernelInfo->getAsObject()->getInteger("DeviceId");
  // TODO: Print warning if the user overrides the device id in the json file.
  int32_t DeviceId = (DeviceIdOpt > -1 ? DeviceIdOpt : DeviceIdJson.value());

  // TODO: do we need requires?
  //__tgt_register_requires(/* Flags */1);

  __tgt_init_all_rtls();

  __tgt_register_lib(&Desc);

  __tgt_target_kernel_replay(
      /* Loc */ nullptr, DeviceId, KernelEntry.addr,
      const_cast<char *>(DeviceMemoryMB.get()->getBuffer().data()),
      DeviceMemoryMB.get()->getBufferSize(), TgtArgs.data(),
      TgtArgOffsets.data(), NumArgs.value(), NumTeams, NumThreads,
      LoopTripCount.value());

  if (VerifyOpt) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> OriginalOutputMB =
        MemoryBuffer::getFile(KernelEntryName + ".original.output",
                              /* isText */ false,
                              /* RequiresNullTerminator */ false);
    if (!OriginalOutputMB)
      report_fatal_error("Error reading the kernel original output file, make "
                         "sure LIBOMPTARGET_SAVE_OUTPUT is set when recording");
    ErrorOr<std::unique_ptr<MemoryBuffer>> ReplayOutputMB =
        MemoryBuffer::getFile(KernelEntryName + ".replay.output",
                              /* isText */ false,
                              /* RequiresNullTerminator */ false);
    if (!ReplayOutputMB)
      report_fatal_error("Error reading the kernel replay output file");

    StringRef OriginalOutput = OriginalOutputMB.get()->getBuffer();
    StringRef ReplayOutput = ReplayOutputMB.get()->getBuffer();
    if (OriginalOutput == ReplayOutput)
      outs() << "[llvm-omp-kernel-replay] Replay device memory verified!\n";
    else
      outs() << "[llvm-omp-kernel-replay] Replay device memory failed to "
                "verify!\n";
  }
  // TODO: calling unregister lib causes plugin deinit error for nextgen
  // plugins.
  //__tgt_unregister_lib(&Desc);

  return 0;
}
