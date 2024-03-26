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

#include "omptarget.h"

#include "Shared/PluginAPI.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdint>
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
      MemoryBuffer::getFile(InputFilename, /*isText=*/true,
                            /*RequiresNullTerminator=*/true);
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

  void *BAllocStart = reinterpret_cast<void *>(
      JsonKernelInfo->getAsObject()->getInteger("BumpAllocVAStart").value());

  __tgt_offload_entry KernelEntry = {nullptr, nullptr, 0, 0, 0};
  std::string KernelEntryName = KernelFunc.value().str();
  KernelEntry.name = const_cast<char *>(KernelEntryName.c_str());
  // Anything non-zero works to uniquely identify the kernel.
  KernelEntry.addr = (void *)0x1;

  ErrorOr<std::unique_ptr<MemoryBuffer>> ImageMB =
      MemoryBuffer::getFile(KernelEntryName + ".image", /*isText=*/false,
                            /*RequiresNullTerminator=*/false);
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

  auto DeviceMemorySizeJson =
      JsonKernelInfo->getAsObject()->getInteger("DeviceMemorySize");
  // Set device memory size to the ceiling of GB granularity.
  uint64_t DeviceMemorySize = std::ceil(DeviceMemorySizeJson.value());

  auto DeviceIdJson = JsonKernelInfo->getAsObject()->getInteger("DeviceId");
  // TODO: Print warning if the user overrides the device id in the json file.
  int32_t DeviceId = (DeviceIdOpt > -1 ? DeviceIdOpt : DeviceIdJson.value());

  // TODO: do we need requires?
  //__tgt_register_requires(/*Flags=*/1);

  __tgt_register_lib(&Desc);

  uint64_t ReqPtrArgOffset = 0;
  int Rc = __tgt_activate_record_replay(DeviceId, DeviceMemorySize, BAllocStart,
                                        false, VerifyOpt, ReqPtrArgOffset);

  if (Rc != OMP_TGT_SUCCESS) {
    report_fatal_error("Cannot activate record replay\n");
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> DeviceMemoryMB =
      MemoryBuffer::getFile(KernelEntryName + ".memory", /*isText=*/false,
                            /*RequiresNullTerminator=*/false);

  if (!DeviceMemoryMB)
    report_fatal_error("Error reading the kernel input device memory.");

  // On AMD for currently unknown reasons we cannot copy memory mapped data to
  // device. This is a work-around.
  uint8_t *recored_data = new uint8_t[DeviceMemoryMB.get()->getBufferSize()];
  std::memcpy(recored_data,
              const_cast<char *>(DeviceMemoryMB.get()->getBuffer().data()),
              DeviceMemoryMB.get()->getBufferSize());

  // If necessary, adjust pointer arguments.
  if (ReqPtrArgOffset) {
    for (auto *&Arg : TgtArgs) {
      auto ArgInt = uintptr_t(Arg);
      // Try to find pointer arguments.
      if (ArgInt < uintptr_t(BAllocStart) ||
          ArgInt >= uintptr_t(BAllocStart) + DeviceMemorySize)
        continue;
      Arg = reinterpret_cast<void *>(ArgInt - ReqPtrArgOffset);
    }
  }

  __tgt_target_kernel_replay(
      /*Loc=*/nullptr, DeviceId, KernelEntry.addr, (char *)recored_data,
      DeviceMemoryMB.get()->getBufferSize(), TgtArgs.data(),
      TgtArgOffsets.data(), NumArgs.value(), NumTeams, NumThreads,
      LoopTripCount.value());

  if (VerifyOpt) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> OriginalOutputMB =
        MemoryBuffer::getFile(KernelEntryName + ".original.output",
                              /*isText=*/false,
                              /*RequiresNullTerminator=*/false);
    if (!OriginalOutputMB)
      report_fatal_error("Error reading the kernel original output file, make "
                         "sure LIBOMPTARGET_SAVE_OUTPUT is set when recording");
    ErrorOr<std::unique_ptr<MemoryBuffer>> ReplayOutputMB =
        MemoryBuffer::getFile(KernelEntryName + ".replay.output",
                              /*isText=*/false,
                              /*RequiresNullTerminator=*/false);
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

  delete[] recored_data;

  return 0;
}
