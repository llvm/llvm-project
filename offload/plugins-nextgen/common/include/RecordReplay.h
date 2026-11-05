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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <unordered_set>

#include "Shared/APITypes.h"
#include "Shared/EnvironmentVar.h"
#include "Shared/RecordReplayBlocks.h"
#include "Shared/RecordReplayPointers.h"
#include "Shared/Utils.h"

#include "OffloadError.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StableHashing.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

struct GenericDeviceTy;

/// The subset of kernel information needed to record and replay a launch.
class KernelInfoTy {
public:
  KernelInfoTy() = default;
  KernelInfoTy(StringRef Name, MemoryBufferRef Image, MemoryBufferRef IRImage)
      : Name(Name), Image(Image), IRImage(IRImage) {}

  StringRef getName() const { return Name; }
  MemoryBufferRef getImage() const { return Image; }
  MemoryBufferRef getIRImage() const { return IRImage; }
  bool hasIRImage() const { return IRImage.getBufferSize() != 0; }

private:
  /// The kernel name. Must outlive the call it is passed to.
  StringRef Name;
  /// The image the kernel was loaded from.
  MemoryBufferRef Image;
  /// The IR image associated with the kernel's image, if any.
  MemoryBufferRef IRImage;
};

struct KernelRecordArgsTy {
  /// Number of kernel arguments in \p Args.
  uint32_t NumArgs = 0;
  /// Array of \p NumArgs pointers, each pointing at one argument's value,
  /// with any offsets already resolved.
  void **Args = nullptr;
  /// Tripcount for the teams / distribute loop, 0 otherwise.
  uint64_t Tripcount = 0;
  /// Number of blocks originally requested by the program for the first
  /// dimension (e.g., num_teams clause), or 0 if none was requested.
  uint32_t RequestedNumBlocks = 0;
  /// Maximum number of threads per block that this kernel may use.
  uint32_t MaxNumThreads = 0;
  /// Effective number of blocks for this launch (for x,y,z dimension).
  uint32_t NumBlocks[3] = {0, 0, 0};
  /// Effective number of threads for this launch (for x,y,z dimension).
  uint32_t NumThreads[3] = {0, 0, 0};
  /// Amount of dynamic cgroup memory this launch requests.
  uint32_t DynCGroupMem = 0;
  /// Set by the caller when replaying a previously recorded kernel launch, so
  /// the outcome can be reported back; null for a normal launch.
  KernelReplayOutcomeTy *ReplayOutcome = nullptr;
};

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
    Program,
    IRImage,
    Pointers
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

  /// Whether a report should be emitted after the recording.
  bool EmitReport;

  /// The name of the file where to emit the record report.
  std::string ReportFilename;

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

  /// The offsets, within the record replay memory, of the locations where the
  /// runtime stored a device pointer while recording.
  SmallVector<uint64_t> PointerLocations;

  /// Mutex that protects dynamic allocations, globals and pointer locations.
  std::mutex AllocationLock;

  /// Whether the input snapshots are recorded into the block store rather than
  /// as full copies (LIBOMPTARGET_RECORD_BLOCK_STORE).
  bool UseBlockStore = true;

  /// Whether the record output snapshots are stored compressed
  /// (LIBOMPTARGET_RECORD_COMPRESS_OUTPUT).
  bool CompressOutput = true;

  /// The block store file shared by the input snapshots, its generation and
  /// size, the stored block of each distinct block content, by its 128-bit
  /// hash, and the descriptor of the locked lock file of the directory.
  std::unique_ptr<raw_fd_ostream> BlockStore;
  uint64_t BlockStoreGeneration = 0;
  uint64_t BlockStoreSize = 0;
  int BlockLockFD = -1;
  DenseMap<std::pair<uint64_t, uint64_t>, RecordReplayBlockTy> StoredBlocks;
  std::mutex BlockStoreLock;

  // An instance of a kernel record replay.
  struct InstanceTy {
    /// The kernel name.
    std::string KernelName;

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

    /// The begin and end time points of the kernel execution.
    using ClockTy = std::chrono::steady_clock;
    mutable std::chrono::time_point<ClockTy> BeginTime, EndTime;

    /// The number of occurrences during the execution.
    mutable size_t Occurrences = 0;

    /// The device memory just before the kernel execution. It is kept until
    /// the epilogue to store the output snapshot as a diff against it.
    mutable std::unique_ptr<MemoryBuffer> InputSnapshot;

    /// When recording, the sorted offsets of the logged pointer locations that
    /// point into the record replay memory before the kernel execution, and the
    /// number of unknown pointers in the input and the output.
    mutable SmallVector<uint64_t, 0> PointerOffsets;
    mutable uint64_t NumUnknownInputPointers = 0;
    mutable uint64_t NumUnknownOutputPointers = RecordReplayUnknownCount;

    InstanceTy(const KernelInfoTy &Kernel, uint32_t NumTeams,
               uint32_t NumThreads, uint32_t SharedMemorySize,
               KernelReplayOutcomeTy *ReplayOutcome);

    bool operator==(const InstanceTy &Other) const {
      return (KernelHash == Other.KernelHash &&
              LaunchConfigHash == Other.LaunchConfigHash &&
              NumTeams == Other.NumTeams && NumThreads == Other.NumThreads &&
              SharedMemorySize == Other.SharedMemorySize);
    }

    /// Record the begin and ending of the kernel execution.
    void recordBeginTime() const { BeginTime = ClockTy::now(); }
    void recordEndTime() const { EndTime = ClockTy::now(); }

    /// Get the kernel execution time in nanoseconds.
    uint64_t getRecordedTimeNs() const {
      using DurationNsTy = std::chrono::duration<uint64_t, std::nano>;
      return std::chrono::duration_cast<DurationNsTy>(EndTime - BeginTime)
          .count();
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
  SmallVector<const InstanceTy *> OrderedInstances;
  std::mutex InstancesLock;

public:
  RecordReplayTy(StatusTy Status, StringRef OutputDirectoryStr, bool SaveOutput,
                 bool EmitReport, StringRef ReportFilename,
                 GenericDeviceTy &Device)
      : Status(Status), SaveOutput(SaveOutput), EmitReport(EmitReport),
        ReportFilename(ReportFilename.str()), Device(Device) {
    if (OutputDirectoryStr == "")
      OutputDirectory = std::filesystem::current_path();
    else
      OutputDirectory = OutputDirectoryStr.data();

    BoolEnvar OMPX_RecordBlockStore("LIBOMPTARGET_RECORD_BLOCK_STORE", true);
    UseBlockStore = OMPX_RecordBlockStore;
    BoolEnvar OMPX_RecordCompressOutput("LIBOMPTARGET_RECORD_COMPRESS_OUTPUT",
                                        true);
    CompressOutput = OMPX_RecordCompressOutput;
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
  bool shouldRecordPrologue(const InstanceTy &Instance) const {
    return isRecording() || shouldRecordEpilogue(Instance);
  }
  bool shouldRecordEpilogue(const InstanceTy &Instance) const {
    return (isRecordingOrReplaying() && SaveOutput) ||
           shouldVerifyOutput(Instance);
  }
  /// Whether the replay of \p Instance compares the device memory after the
  /// kernel execution with an expected output.
  bool shouldVerifyOutput(const InstanceTy &Instance) const {
    return isReplaying() && Instance.ReplayOutcome &&
           Instance.ReplayOutcome->ExpectedOutput;
  }

  /// Add information about a global.
  void addGlobal(const char *Name, uint64_t Size, void *Addr) {
    std::lock_guard<std::mutex> Lock(AllocationLock);
    GlobalEntries.emplace_back(GlobalEntryTy{Name, Size, Addr});
  }

  /// Log that the runtime stored a device pointer at \p TgtPtrAddr, which is
  /// needed to relocate the replay. Only locations within the record replay
  /// memory are logged, and only when recording.
  void addPointerLocation(void *TgtPtrAddr);

  /// Register a record replay instance, record the prologue data if necessary,
  /// and return the instance's handle. The prologue is the phase just before
  /// executing the kernel. This phase can include the recording of memory
  /// snapshot, the record descriptor and the globals. When replaying, only the
  /// instance is registered.
  Expected<HandleTy> recordPrologue(const KernelInfoTy &Kernel,
                                    const KernelRecordArgsTy &RecordArgs);

  /// Record the epilogue if necessary, which can include the memory snapshot
  /// when recording or replaying.
  Error recordEpilogue(const KernelInfoTy &Kernel, HandleTy Handle);

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
  registerInstance(const KernelInfoTy &Kernel, uint32_t NumTeams,
                   uint32_t NumThreads, uint32_t SharedMemorySize,
                   KernelReplayOutcomeTy *ReplayOutcome);

  /// Unregister an instance once it has been replayed. Instances during
  /// recording cannot be unregistered. Accessing the instance beyond this point
  /// is invalid.
  Error unregisterInstance(const InstanceTy &Instance);

  /// Populate the replay outcome struct to forward some replay information.
  void populateReplayOutcome(const InstanceTy &Instance,
                             KernelReplayOutcomeTy &Outcome);

protected:
  /// Whether \p Value is an address within the first \p MemorySize bytes of
  /// the record replay memory, i.e., the memory in use, or just past them.
  /// Checking only the memory in use, rather than the whole reservation, makes
  /// it unlikely that other data, such as two 32-bit integers, looks like a
  /// pointer.
  bool pointsToMemory(uint64_t Value, uint64_t MemorySize) const {
    uint64_t Start = reinterpret_cast<uintptr_t>(StartAddr);
    return Value >= Start && Value <= Start + MemorySize;
  }

  /// Count the 8-byte aligned values in \p Data that point into the first
  /// \p MemorySize bytes of the record replay memory and whose location is not
  /// in the sorted \p Known list. \p Offset is the offset of \p Data within
  /// the record replay memory.
  uint64_t countUnknownPointers(const char *Data, uint64_t Size,
                                uint64_t Offset, ArrayRef<uint64_t> Known,
                                uint64_t MemorySize) const;

private:
  /// Record the prologue data.
  virtual Error recordPrologueImpl(const KernelInfoTy &Kernel,
                                   const InstanceTy &Instance) = 0;

  /// Record the epilogue data.
  virtual Error recordEpilogueImpl(const KernelInfoTy &Kernel,
                                   const InstanceTy &Instance) = 0;

  /// Record the descriptor of the kernel.
  virtual Error recordDescImpl(const KernelInfoTy &Kernel,
                               const InstanceTy &Instance,
                               const KernelRecordArgsTy &RecordArgs) = 0;

  /// Record the pointers needed to relocate the replay.
  virtual Error recordPointersImpl(const InstanceTy &Instance) = 0;

  /// Get a string with the filename.
  virtual SmallString<128> getFilenameImpl(const InstanceTy &Instance,
                                           FileTy FileType,
                                           bool IncludeDirectory) = 0;
};

/// The native kernel record replay support.
struct NativeRecordReplayTy : public RecordReplayTy {
  NativeRecordReplayTy(StatusTy Status, StringRef OutputDirectoryStr,
                       bool SaveOutput, bool EmitReport,
                       StringRef ReportFilename, GenericDeviceTy &Device)
      : RecordReplayTy(Status, OutputDirectoryStr, SaveOutput, EmitReport,
                       ReportFilename, Device) {}

private:
  Error recordPrologueImpl(const KernelInfoTy &Kernel,
                           const InstanceTy &Instance) override;
  Error recordEpilogueImpl(const KernelInfoTy &Kernel,
                           const InstanceTy &Instance) override;
  Error recordDescImpl(const KernelInfoTy &Kernel, const InstanceTy &Instance,
                       const KernelRecordArgsTy &RecordArgs) override;
  Error recordPointersImpl(const InstanceTy &Instance) override;

  /// Get a string with the filename.
  SmallString<128> getFilenameImpl(const InstanceTy &Instance, FileTy FileType,
                                   bool IncludeDirectory) override;

  /// Get the extension for a recording file type.
  StringRef getExtension(FileTy FileType);

  /// Retrieve the device memory currently in use, or null if there is none.
  Expected<std::unique_ptr<WritableMemoryBuffer>> retrieveSnapshot();

  /// Record a memory snapshot to a file.
  Error recordSnapshot(StringRef Filename, const MemoryBuffer *Snapshot);

  /// Open the latest block store of the output directory, and index the
  /// blocks it holds.
  Error openBlockStore();

  /// Record a memory snapshot to a file as a list of blocks, adding the blocks
  /// not stored yet to the block store.
  Error recordSnapshotBlocks(StringRef Filename, const MemoryBuffer *Snapshot);

  /// Retrieve the first \p Size bytes of the device memory in chunks, to bound
  /// the host memory usage, and call \p Callback with each chunk and its
  /// offset until it returns false.
  Error retrieveSnapshotChunks(
      uint64_t Size,
      function_ref<bool(const char *Chunk, uint64_t Offset, uint64_t Size)>
          Callback);

  /// Write the device memory currently in use to \p OS, as a diff against the
  /// input snapshot of \p Instance.
  Error writeSnapshotDiff(raw_pwrite_stream &OS, const InstanceTy &Instance);

  /// Record the device memory currently in use to a file, as a diff against
  /// the input snapshot of \p Instance, compressed if \p Compress is set and
  /// compression makes it smaller.
  Error recordSnapshotDiff(StringRef Filename, const InstanceTy &Instance,
                           bool Compress);

  /// Compare the device memory currently in use with the expected output of
  /// the replay of \p Instance, and store the result in its replay outcome.
  Error verifySnapshot(const InstanceTy &Instance);

  /// Record the globals to a file, and count the unknown pointers in them.
  Error recordGlobals(StringRef Filename, uint64_t &NumUnknownPointers);

  /// Find the logged pointer locations that point into the record replay
  /// memory in the input \p Snapshot, and count the unknown pointers.
  void findInputPointers(const InstanceTy &Instance,
                         const MemoryBuffer *Snapshot);

  /// Record the device image to a file.
  Error recordImage(const KernelInfoTy &Kernel, StringRef Filename,
                    StringRef IRImageFilename);
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OPENMP_LIBOMPTARGET_PLUGINS_COMMON_RECORDREPLAY_H
