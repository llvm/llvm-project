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

#include "Shared/RecordReplayDiff.h"
#include "Shared/RecordReplayPointers.h"
#include "Shared/Utils.h"
#include "omptarget.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Frontend/Offloading/Utility.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>

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
    cl::desc("Save the device memory output of the replayed kernel execution, "
             "as a diff against the record input."),
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

static cl::opt<bool> RelocateOpt(
    "relocate",
    cl::desc("Place the replay memory at any device address instead of the "
             "recorded one, allocating only the memory the kernel uses. The "
             "replay falls back to the recorded address if the recording has "
             "pointers that cannot be relocated."),
    cl::init(false), cl::cat(ReplayOptions));

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

/// Read the header of an output snapshot diff.
static Error readDiffHeader(StringRef Buffer, StringRef Kind,
                            RecordReplayDiffHeaderTy &Header) {
  if (Buffer.size() < sizeof(Header))
    return createErr("the kernel %s output file is truncated", Kind.data());
  std::memcpy(&Header, Buffer.data(), sizeof(Header));
  if (std::memcmp(Header.Magic, RecordReplayDiffMagic, sizeof(Header.Magic)))
    return createErr("the kernel %s output file is not a memory diff",
                     Kind.data());
  return Error::success();
}

/// The relocation of the replay memory to another device address.
struct RelocationTy {
  /// The sorted offsets of the pointers into the replay memory.
  SmallVector<uint64_t> PointerOffsets;
  /// The replay memory address and size.
  uint64_t Address = 0;
  uint64_t Size = 0;
  /// The replay memory address minus the recorded one.
  uint64_t Delta = 0;
  /// The relocated input snapshot.
  StringRef Input;

  /// Whether \p Value points into the replay memory, or just past its end.
  bool pointsToMemory(uint64_t Value) const {
    return Value >= Address && Value <= Address + Size;
  }
};

/// The contents of a pointers file.
struct PointersFileTy {
  RecordReplayPointersHeaderTy Header;
  SmallVector<uint64_t> PointerOffsets;
};

/// Read the pointers file that describes how to relocate the replay memory.
static Expected<PointersFileTy> readPointersFile(StringRef Filename) {
  auto BufferOrErr = MemoryBuffer::getFile(Filename, /*IsText=*/false,
                                           /*RequiresNullTerminator=*/false);
  if (!BufferOrErr)
    return createErr("failed to read the pointers file");

  StringRef Buffer = (*BufferOrErr)->getBuffer();
  PointersFileTy File;
  if (Buffer.size() < sizeof(File.Header))
    return createErr("the pointers file is truncated");
  std::memcpy(&File.Header, Buffer.data(), sizeof(File.Header));
  if (std::memcmp(File.Header.Magic, RecordReplayPointersMagic,
                  sizeof(File.Header.Magic)))
    return createErr("the pointers file has an invalid format");
  if (Buffer.size() - sizeof(File.Header) !=
      File.Header.NumPointers * sizeof(uint64_t))
    return createErr("the pointers file is truncated");

  File.PointerOffsets.resize(File.Header.NumPointers);
  std::memcpy(File.PointerOffsets.data(), Buffer.data() + sizeof(File.Header),
              File.Header.NumPointers * sizeof(uint64_t));
  return File;
}

/// Undo the relocation of the pointers stored in the regions of the replay
/// output diff \p Output, so it can be compared with the record output.
static Error unrelocateReplayOutput(MutableArrayRef<char> Output,
                                    const RelocationTy &Relocation) {
  constexpr uint64_t PtrSize = sizeof(uint64_t);
  uint64_t Pos = sizeof(RecordReplayDiffHeaderTy);
  while (Pos < Output.size()) {
    RecordReplayDiffRegionTy Region;
    if (Output.size() - Pos < sizeof(Region))
      return createErr("the kernel replay output file is truncated");
    std::memcpy(&Region, Output.data() + Pos, sizeof(Region));
    Pos += sizeof(Region);
    if (Output.size() - Pos < Region.Size)
      return createErr("the kernel replay output file is truncated");

    // Adjust each pointer that overlaps the region. Its bytes outside the
    // region are unchanged, so they are taken from the relocated input.
    uint64_t RegionEnd = Region.Offset + Region.Size;
    auto It =
        llvm::lower_bound(Relocation.PointerOffsets,
                          Region.Offset - std::min(Region.Offset, PtrSize - 1));
    for (; It != Relocation.PointerOffsets.end() && *It < RegionEnd; ++It) {
      uint64_t PtrOffset = *It;
      if (PtrOffset + PtrSize > Relocation.Input.size())
        continue;

      char Bytes[PtrSize];
      for (uint64_t I = 0; I < PtrSize; ++I) {
        uint64_t Offset = PtrOffset + I;
        Bytes[I] = (Offset >= Region.Offset && Offset < RegionEnd)
                       ? Output[Pos + Offset - Region.Offset]
                       : Relocation.Input[Offset];
      }
      uint64_t Value;
      std::memcpy(&Value, Bytes, PtrSize);
      if (!Relocation.pointsToMemory(Value))
        continue;
      Value -= Relocation.Delta;
      std::memcpy(Bytes, &Value, PtrSize);

      for (uint64_t I = 0; I < PtrSize; ++I) {
        uint64_t Offset = PtrOffset + I;
        if (Offset >= Region.Offset && Offset < RegionEnd)
          Output[Pos + Offset - Region.Offset] = Bytes[I];
      }
    }
    Pos += Region.Size;
  }
  return Error::success();
}

/// Verify that the replay output is the same as the record output. Both are
/// diffs against the record input, so they match if and only if the device
/// memory after the kernel execution matches. If the replay memory was
/// relocated, the pointers in the replay output are first adjusted back to the
/// recorded address.
Error verifyReplayOutput(StringRef RecordOutputFilename,
                         StringRef ReplayOutputFilename,
                         const RelocationTy *Relocation) {
  // Load the record output file.
  auto RecordOutputBufferOrErr =
      MemoryBuffer::getFile(RecordOutputFilename,
                            /*isText=*/false,
                            /*RequiresNullTerminator=*/false);
  if (!RecordOutputBufferOrErr)
    return createErr("failed to read the kernel record output file");

  // Load the replay output file, as a writable copy to undo the relocation.
  auto ReplayOutputBufferOrErr =
      WritableMemoryBuffer::getFile(ReplayOutputFilename, /*IsVolatile=*/false);
  if (!ReplayOutputBufferOrErr)
    return createErr("failed to read the kernel replay output file");

  StringRef RecordOutput = RecordOutputBufferOrErr.get()->getBuffer();
  MutableArrayRef<char> ReplayOutputData =
      ReplayOutputBufferOrErr.get()->getBuffer();
  StringRef ReplayOutput(ReplayOutputData.data(), ReplayOutputData.size());
  RecordReplayDiffHeaderTy RecordHeader, ReplayHeader;
  if (auto Err = readDiffHeader(RecordOutput, "record", RecordHeader))
    return Err;
  if (auto Err = readDiffHeader(ReplayOutput, "replay", ReplayHeader))
    return Err;
  if (RecordHeader.InputSize != ReplayHeader.InputSize)
    return createErr("record and replay outputs have different input sizes "
                     "(%" PRIu64 " and %" PRIu64 ")",
                     RecordHeader.InputSize, ReplayHeader.InputSize);
  if (RecordHeader.OutputSize != ReplayHeader.OutputSize)
    return createErr("record and replay outputs have different sizes (%" PRIu64
                     " and %" PRIu64 ")",
                     RecordHeader.OutputSize, ReplayHeader.OutputSize);

  if (Relocation)
    if (auto Err = unrelocateReplayOutput(ReplayOutputData, *Relocation))
      return Err;

  // Compare record and replay outputs to verify they match.
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

  // Check that a minimum and maximum have been exported.
  if (TeamsLimits.size() != 2 || ThreadsLimits.size() != 2)
    return createErr("TeamsLimits and ThreadsLimits must have a min and max");

  // Check that the minimum and maximum are specified or both are zero.
  if (bool(TeamsLimits[0]) != bool(TeamsLimits[1]))
    return createErr("TeamsLimits min and max are inconsistent");
  if (bool(ThreadsLimits[0]) != bool(ThreadsLimits[1]))
    return createErr("ThreadsLimits min and max are inconsistent");

  // If the limits were specified, verify the selected values are valid.
  if (!IgnoreLimitsOpt && TeamsLimits[0] > 0 &&
      (NumTeams < TeamsLimits[0] || NumTeams > TeamsLimits[1]))
    return createErr("number of teams (%" PRIu32
                     ") is out of the allowed limits (min,max: %" PRIu32
                     ",%" PRIu32 ")",
                     NumTeams, TeamsLimits[0], TeamsLimits[1]);
  if (!IgnoreLimitsOpt && ThreadsLimits[0] > 0 &&
      (NumThreads < ThreadsLimits[0] || NumThreads > ThreadsLimits[1]))
    return createErr("number of threads (%" PRIu32
                     ") is out of the allowed limits (min,max: %" PRIu32
                     ",%" PRIu32 ")",
                     NumThreads, ThreadsLimits[0], ThreadsLimits[1]);

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

  // Decide whether the replay memory can be placed at another address. That
  // requires knowing every pointer into the recorded memory and, to verify the
  // replay, into the output.
  std::optional<PointersFileTy> Pointers;
  if (RelocateOpt) {
    Filepath.replace_extension("pointers");
    std::string Reason;
    if (!std::filesystem::exists(Filepath)) {
      Reason = "the recording has no pointers file";
    } else {
      auto PointersOrErr = readPointersFile(Filepath.string());
      if (!PointersOrErr)
        return PointersOrErr.takeError();
      const RecordReplayPointersHeaderTy &Header = PointersOrErr->Header;
      if (Header.NumUnknownInputPointers)
        Reason = std::to_string(Header.NumUnknownInputPointers) +
                 " unknown pointers in the recorded memory";
      else if (VerifyOpt && Header.NumUnknownOutputPointers &&
               Header.NumUnknownOutputPointers != RecordReplayUnknownCount)
        Reason = std::to_string(Header.NumUnknownOutputPointers) +
                 " unknown pointers in the recorded output";
      else
        Pointers = std::move(*PointersOrErr);
    }
    if (!Pointers)
      outs() << TOOL_PREFIX << " Replaying at the recorded address: " << Reason
             << "\n";
  }

  // Load the record input file. A relocated input needs a writable copy.
  Filepath.replace_extension("record_input");
  std::unique_ptr<MemoryBuffer> RecordInputBuffer;
  if (Pointers) {
    auto BufferOrErr =
        WritableMemoryBuffer::getFile(Filepath.c_str(), /*IsVolatile=*/false);
    if (BufferOrErr)
      RecordInputBuffer = std::move(*BufferOrErr);
  } else {
    auto BufferOrErr = MemoryBuffer::getFile(Filepath.c_str(), /*IsText=*/false,
                                             /*RequiresNullTerminator=*/false);
    if (BufferOrErr)
      RecordInputBuffer = std::move(*BufferOrErr);
  }
  if (!RecordInputBuffer)
    return createErr("failed to read the kernel record input file");
  uint64_t InputSize = RecordInputBuffer->getBufferSize();

  // A relocated replay memory only needs to hold the input snapshot.
  int Rc = __tgt_activate_record_replay(
      DeviceId, Pointers ? std::max<uint64_t>(InputSize, 1) : VAllocSize,
      Pointers ? nullptr : VAllocAddr, /*IsRecord=*/false,
      VerifyOpt || SaveOutputOpt,
      /*EmitReport=*/false, Directory.c_str());
  if (Rc != OMP_TGT_SUCCESS)
    return createErr("failed to activate record replay");

  KernelReplayOutcomeTy Outcome;

  // Relocate the pointers in the input snapshot and the kernel arguments to
  // the replay memory. The first allocation is placed at its start, and it is
  // reused by every replay.
  std::optional<RelocationTy> Relocation;
  if (Pointers && InputSize) {
    void *Address = omp_target_alloc(InputSize, DeviceId);
    if (!Address)
      return createErr("failed to allocate the replay memory");
    Outcome.ReplayDeviceAlloc = Address;

    uint64_t RecordAddress = reinterpret_cast<uintptr_t>(VAllocAddr);
    Relocation.emplace();
    Relocation->PointerOffsets = std::move(Pointers->PointerOffsets);
    Relocation->Address = reinterpret_cast<uintptr_t>(Address);
    Relocation->Size = InputSize;
    Relocation->Delta = Relocation->Address - RecordAddress;
    Relocation->Input = RecordInputBuffer->getBuffer();

    char *Input = const_cast<char *>(RecordInputBuffer->getBufferStart());
    for (uint64_t Offset : Relocation->PointerOffsets) {
      if (Offset + sizeof(uint64_t) > InputSize)
        return createErr("the pointers file has an invalid offset");
      uint64_t Value;
      std::memcpy(&Value, Input + Offset, sizeof(Value));
      Value += Relocation->Delta;
      std::memcpy(Input + Offset, &Value, sizeof(Value));
    }
    for (void *&Arg : TgtArgs) {
      uint64_t Value = reinterpret_cast<uintptr_t>(Arg);
      if (Value >= RecordAddress && Value <= RecordAddress + InputSize)
        Arg = reinterpret_cast<void *>(Value + Relocation->Delta);
    }

    outs() << TOOL_PREFIX << " Relocated the replay memory from " << VAllocAddr
           << " to " << Address << " (" << Relocation->PointerOffsets.size()
           << " pointers)\n";
  }

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
    if (auto Err =
            verifyReplayOutput(Filepath.c_str(), Outcome.OutputFilepath.c_str(),
                               Relocation ? &*Relocation : nullptr))
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
