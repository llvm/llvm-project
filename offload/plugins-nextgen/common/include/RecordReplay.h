//===- RecordReplay.h - Target independent kernel record replay interface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_RECORDREPLAY_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_RECORDREPLAY_H

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <unordered_set>

#include "Shared/APITypes.h"
#include "Shared/EnvironmentVar.h"
#include "Shared/Utils.h"

#include "OffloadError.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StableHashing.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

struct GenericKernelTy;
struct GenericDeviceTy;

struct RecordReplayTy {
protected:
  struct InstanceTy;

public:
  /// Describes the state of the record replay mechanism.
  enum StatusTy { Deactivated = 0, Recording, Replaying };

  /// Describes the format of the recording and replaying.
  enum FormatTy { Native = 0 };

  /// Describes the file types that can be recorded.
  enum FileTy {
    PrologueSnapshot = 0,
    EpilogueSnapshot,
    Descriptor,
    Globals,
    Program
  };

  struct HandleTy {
    const InstanceTy *Instance = nullptr;
    bool Active = false;
  };

protected:
  /// Address and size of record replay memory space.
  void *StartAddr = nullptr;
  uint64_t TotalSize = 0;
  uint64_t CurrentSize = 0;

  /// Status of the record or replay.
  StatusTy Status;

  /// The path where to store all recorded files.
  std::filesystem::path OutputDirectory;

  /// Whether a memory snapshot should be recorded a kernel execution.
  bool SaveOutput;

  /// Whether a report should be emitted afther the recording.
  bool EmitReport;

  /// Reference to the corresponding device.
  GenericDeviceTy &Device;

  /// The information for a global.
  struct GlobalEntryTy {
    std::string Name;
    uint64_t Size;
    void *Addr;
  };

  /// List of all globals mapped to the device.
  SmallVector<GlobalEntryTy> GlobalEntries;

  /// Mutex that protects dynamic allocations and globals.
  std::mutex AllocationLock;

  // An instance of a kernel record replay.
  struct InstanceTy {
    /// Reference to the kernel.
    const GenericKernelTy &Kernel;

    /// The launch configuration parameters.
    uint32_t NumTeams = 0;
    uint32_t NumThreads = 0;
    uint32_t SharedMemorySize = 0;

    /// The hashes representing the kernel and the launch configuration.
    size_t KernelHash = 0;
    size_t LaunchConfigHash = 0;

    /// When replaying, the entity requesting the replay may also need further
    /// information about the the kernel's replay, such as the snapshot file.
    KernelReplayOutcomeTy *ReplayOutcome = nullptr;

    /// The number of occurrences during the execution.
    mutable size_t Occurrences = 0;

    InstanceTy(const GenericKernelTy &Kernel, uint32_t NumTeams,
               uint32_t NumThreads, uint32_t SharedMemorySize,
               KernelReplayOutcomeTy *ReplayOutcome);

    bool operator==(const InstanceTy &Other) const {
      return (KernelHash == Other.KernelHash &&
              LaunchConfigHash == Other.LaunchConfigHash &&
              NumTeams == Other.NumTeams && NumThreads == Other.NumThreads &&
              SharedMemorySize == Other.SharedMemorySize);
    }
  };

  struct InstanceHasher {
    std::size_t operator()(const InstanceTy &I) const {
      llvm::stable_hash H =
          llvm::stable_hash_combine(I.KernelHash, I.LaunchConfigHash);
      return static_cast<std::size_t>(H);
    }
  };

  /// Tracker of record replay instances.
  std::unordered_set<InstanceTy, InstanceHasher> Instances;
  std::mutex InstancesLock;

public:
  RecordReplayTy(StatusTy Status, StringRef OutputDirectoryStr, bool SaveOutput,
                 bool EmitReport, GenericDeviceTy &Device)
      : Status(Status), SaveOutput(SaveOutput), EmitReport(EmitReport),
        Device(Device) {
    if (OutputDirectoryStr == "")
      OutputDirectory = std::filesystem::current_path();
    else
      OutputDirectory = OutputDirectoryStr.data();
  }

  virtual ~RecordReplayTy() = default;

  /// Initialize kernel record replay for the corresponding device.
  Error init(uint64_t MemSize, void *VAddr);
  Error deinit();

  /// Emit the information of each registered instance.
  Error emitInstanceReport();

  bool isRecording() const { return Status == StatusTy::Recording; }
  bool isReplaying() const { return Status == StatusTy::Replaying; }
  bool isRecordingOrReplaying() const { return isRecording() || isReplaying(); }
  bool shouldRecordPrologue() const { return isRecording(); }
  bool shouldRecordEpilogue() const {
    return isRecordingOrReplaying() && SaveOutput;
  }

  /// Add information about a global.
  void addGlobal(const char *Name, uint64_t Size, void *Addr) {
    std::lock_guard<std::mutex> Lock(AllocationLock);
    GlobalEntries.emplace_back(GlobalEntryTy{Name, Size, Addr});
  }

  /// Register a record replay instance, record the prologue data if necessary,
  /// and return the instance's handle. The prologue is the phase just before
  /// executing the kernel. This phase can include the recording of memory
  /// snapshot, the record descriptor and the globals. When replaying, only the
  /// instance is registered.
  Expected<HandleTy>
  recordPrologue(const GenericKernelTy &Kernel, const KernelArgsTy &KernelArgs,
                 const KernelExtraArgsTy *KernelExtraArgs,
                 const KernelLaunchParamsTy &LaunchParams, uint32_t NumTeams[3],
                 uint32_t NumThreads[3], uint32_t SharedMemorySize);

  /// Record the epilogue if necessary, which can include the memory snapshot
  /// when recording or replaying.
  Error recordEpilogue(const GenericKernelTy &Kernel, HandleTy Handle);

  /// Get a string with the filename.
  SmallString<128> getFilename(const InstanceTy &Instance, FileTy FileType,
                               bool IncludeDirectory = true) {
    return getFilenameImpl(Instance, FileType, IncludeDirectory);
  }

  /// Allocates device memory from the record replay space.
  Expected<void *> allocate(uint64_t Size);
  Error deallocate(void *Ptr);

private:
  /// Register an instance and return a reference and whether it was registered
  /// as a new instance.
  std::pair<const InstanceTy &, bool>
  registerInstance(const GenericKernelTy &Kernel, uint32_t NumTeams,
                   uint32_t NumThreads, uint32_t SharedMemorySize,
                   KernelReplayOutcomeTy *ReplayOutcome);

  /// Record the prologue data.
  virtual Error
  recordPrologueImpl(const GenericKernelTy &Kernel, const InstanceTy &Instance,
                     const KernelArgsTy &KernelArgs,
                     const KernelLaunchParamsTy &LaunchParams) = 0;

  /// Record the epilogue data.
  virtual Error recordEpilogueImpl(const GenericKernelTy &Kernel,
                                   const InstanceTy &Instance) = 0;

  /// Record the descriptor of the kernel.
  virtual Error recordDescImpl(const GenericKernelTy &Kernel,
                               const InstanceTy &Instance,
                               const KernelArgsTy &KernelArgs,
                               const KernelLaunchParamsTy &LaunchParams) = 0;

  /// Get a string with the filename.
  virtual SmallString<128> getFilenameImpl(const InstanceTy &Instance,
                                           FileTy FileType,
                                           bool IncludeDirectory) = 0;
};

/// The native kernel record replay support.
struct NativeRecordReplayTy : public RecordReplayTy {
  NativeRecordReplayTy(StatusTy Status, StringRef OutputDirectoryStr,
                       bool SaveOutput, bool EmitReport,
                       GenericDeviceTy &Device)
      : RecordReplayTy(Status, OutputDirectoryStr, SaveOutput, EmitReport,
                       Device) {}

private:
  Error recordPrologueImpl(const GenericKernelTy &Kernel,
                           const InstanceTy &Instance,
                           const KernelArgsTy &KernelArgs,
                           const KernelLaunchParamsTy &LaunchParams) override;
  Error recordEpilogueImpl(const GenericKernelTy &Kernel,
                           const InstanceTy &Instance) override;
  Error recordDescImpl(const GenericKernelTy &Kernel,
                       const InstanceTy &Instance,
                       const KernelArgsTy &KernelArgs,
                       const KernelLaunchParamsTy &LaunchParams) override;

  /// Get a string with the filename.
  SmallString<128> getFilenameImpl(const InstanceTy &Instance, FileTy FileType,
                                   bool IncludeDirectory) override;

  /// Get the extension for a recording file type.
  StringRef getExtension(FileTy FileType);

  /// Record a memory snapshot to a file.
  Error recordSnapshot(StringRef Filename);

  /// Record the globals to a file.
  Error recordGlobals(StringRef Filename);

  /// Record the device image to a file.
  Error recordImage(const GenericKernelTy &Kernel, StringRef Filename);
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OPENMP_LIBOMPTARGET_PLUGINS_COMMON_RECORDREPLAY_H
