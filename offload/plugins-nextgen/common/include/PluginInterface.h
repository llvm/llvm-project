//===- PluginInterface.h - Target independent plugin device interface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_PLUGININTERFACE_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_PLUGININTERFACE_H

#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <map>
#include <shared_mutex>
#include <variant>
#include <vector>

#include "ExclusiveAccess.h"
#include "OpenMP/InteropAPI.h"
#include "Shared/APITypes.h"
#include "Shared/Debug.h"
#include "Shared/Environment.h"
#include "Shared/EnvironmentVar.h"
#include "Shared/Requirements.h"
#include "Shared/Utils.h"

#include "GlobalHandler.h"
#include "JIT.h"
#include "MemoryManager.h"
#include "OffloadError.h"
#include "RPC.h"
#include "omptarget.h"

#ifdef OMPT_SUPPORT
#include "omp-tools.h"
#endif

#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
namespace omp {
namespace target {

namespace plugin {

struct GenericPluginTy;
struct GenericKernelTy;
struct GenericDeviceTy;
struct RecordReplayTy;
template <typename ResourceRef> class GenericDeviceResourceManagerTy;

namespace Plugin {
/// Create a success error. This is the same as calling Error::success(), but
/// it is recommended to use this one for consistency with Plugin::error() and
/// Plugin::check().
static inline Error success() { return Error::success(); }

/// Create an Offload error.
template <typename... ArgsTy>
static Error error(error::ErrorCode Code, const char *ErrFmt, ArgsTy... Args) {
  return error::createOffloadError(Code, ErrFmt, Args...);
}

inline Error error(error::ErrorCode Code, const char *S) {
  return make_error<error::OffloadError>(Code, S);
}

inline Error error(error::ErrorCode Code, Error &&OtherError,
                   const char *Context) {
  return error::createOffloadError(Code, std::move(OtherError), Context);
}

/// Check the plugin-specific error code and return an error or success
/// accordingly. In case of an error, create a string error with the error
/// description. The ErrFmt should follow the format:
///     "Error in <function name>[<optional info>]: %s"
/// The last format specifier "%s" is mandatory and will be used to place the
/// error code's description. Notice this function should be only called from
/// the plugin-specific code.
/// TODO: Refactor this, must be defined individually by each plugin.
template <typename... ArgsTy>
static Error check(int32_t ErrorCode, const char *ErrFmt, ArgsTy... Args);
} // namespace Plugin

/// Class that wraps the __tgt_async_info to simply its usage. In case the
/// object is constructed without a valid __tgt_async_info, the object will use
/// an internal one and will synchronize the current thread with the pending
/// operations when calling AsyncInfoWrapperTy::finalize(). This latter function
/// must be called before destroying the wrapper object.
struct AsyncInfoWrapperTy {
  AsyncInfoWrapperTy(GenericDeviceTy &Device, __tgt_async_info *AsyncInfoPtr);

  ~AsyncInfoWrapperTy() {
    assert(!AsyncInfoPtr && "AsyncInfoWrapperTy not finalized");
  }

  /// Get the raw __tgt_async_info pointer.
  operator __tgt_async_info *() const { return AsyncInfoPtr; }

  /// Indicate whether there is queue.
  bool hasQueue() const { return (AsyncInfoPtr->Queue != nullptr); }

  /// Get the queue.
  template <typename Ty> Ty getQueueAs() {
    static_assert(sizeof(Ty) == sizeof(AsyncInfoPtr->Queue),
                  "Queue is not of the same size as target type");
    return static_cast<Ty>(AsyncInfoPtr->Queue);
  }

  /// Set the queue.
  template <typename Ty> void setQueueAs(Ty Queue) {
    static_assert(sizeof(Ty) == sizeof(AsyncInfoPtr->Queue),
                  "Queue is not of the same size as target type");
    assert(!AsyncInfoPtr->Queue && "Overwriting queue");
    AsyncInfoPtr->Queue = Queue;
  }

  /// Get the queue, using the provided resource manager to initialise it if it
  /// doesn't exist.
  template <typename Ty, typename RMTy>
  Expected<Ty>
  getOrInitQueue(GenericDeviceResourceManagerTy<RMTy> &ResourceManager) {
    std::lock_guard<std::mutex> Lock(AsyncInfoPtr->Mutex);
    if (!AsyncInfoPtr->Queue) {
      if (auto Err = ResourceManager.getResource(
              *reinterpret_cast<Ty *>(&AsyncInfoPtr->Queue)))
        return Err;
    }
    return getQueueAs<Ty>();
  }

  /// Synchronize with the __tgt_async_info's pending operations if it's the
  /// internal async info. The error associated to the asynchronous operations
  /// issued in this queue must be provided in \p Err. This function will update
  /// the error parameter with the result of the synchronization if it was
  /// actually executed. This function must be called before destroying the
  /// object and only once.
  void finalize(Error &Err);

  /// Register \p Ptr as an associated allocation that is freed after
  /// finalization.
  void freeAllocationAfterSynchronization(void *Ptr) {
    std::lock_guard<std::mutex> AllocationGuard(AsyncInfoPtr->Mutex);
    AsyncInfoPtr->AssociatedAllocations.push_back(Ptr);
  }

private:
  GenericDeviceTy &Device;
  __tgt_async_info LocalAsyncInfo;
  __tgt_async_info *AsyncInfoPtr;
};

enum class DeviceInfo {
#define OFFLOAD_DEVINFO(Name, _, Value) Name = Value,
#include "OffloadInfo.inc"
#undef OFFLOAD_DEVINFO
};

/// Tree node for device information
///
/// This information is either printed or used by liboffload to extract certain
/// device queries. Each property has an optional key, an optional value
/// and optional children. The children can be used to store additional
/// information (such as x, y and z components of ranges).
struct InfoTreeNode {
  static constexpr uint64_t IndentSize = 4;

  std::string Key;
  using VariantType = std::variant<uint64_t, std::string, bool, std::monostate>;
  VariantType Value;
  std::string Units;
  // Need to specify a default value number of elements here as `InfoTreeNode`'s
  // size is unknown. This is a vector (rather than a Key->Value map) since:
  // * The keys need to be owned and thus `std::string`s
  // * The order of keys is important
  // * The same key can appear multiple times
  std::unique_ptr<llvm::SmallVector<InfoTreeNode, 8>> Children;

  llvm::DenseMap<DeviceInfo, size_t> DeviceInfoMap;

  InfoTreeNode() : InfoTreeNode("", std::monostate{}, "") {}
  InfoTreeNode(std::string Key, VariantType Value, std::string Units)
      : Key(std::move(Key)), Value(Value), Units(std::move(Units)) {}

  /// Add a new info entry as a child of this node. The entry requires at least
  /// a key string in \p Key. The value in \p Value is optional and can be any
  /// type that is representable as a string. The units in \p Units is optional
  /// and must be a string. Providing a device info key allows liboffload to
  /// use that value for an appropriate olGetDeviceInfo query
  template <typename T = std::monostate>
  InfoTreeNode *add(std::string Key, T Value = T(),
                    std::string Units = std::string(),
                    std::optional<DeviceInfo> DeviceInfoKey = std::nullopt) {
    assert(!Key.empty() && "Invalid info key");

    if (!Children)
      Children = std::make_unique<llvm::SmallVector<InfoTreeNode, 8>>();

    VariantType ValueVariant;
    if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, std::monostate>)
      ValueVariant = Value;
    else if constexpr (std::is_arithmetic_v<T>)
      ValueVariant = static_cast<uint64_t>(Value);
    else
      ValueVariant = std::string{Value};

    auto Ptr =
        &Children->emplace_back(std::move(Key), ValueVariant, std::move(Units));

    if (DeviceInfoKey)
      DeviceInfoMap[*DeviceInfoKey] = Children->size() - 1;

    return Ptr;
  }

  std::optional<InfoTreeNode *> get(StringRef Key) {
    if (!Children)
      return std::nullopt;

    auto It = std::find_if(Children->begin(), Children->end(),
                           [&](auto &V) { return V.Key == Key; });
    if (It == Children->end())
      return std::nullopt;
    return It;
  }

  std::optional<InfoTreeNode *> get(DeviceInfo Info) {
    auto Result = DeviceInfoMap.find(Info);
    if (Result != DeviceInfoMap.end())
      return &(*Children)[Result->second];
    return std::nullopt;
  }

  /// Print all info entries in the tree
  void print() const {
    // Fake an additional indent so that values are offset from the keys
    doPrint(0, maxKeySize(1));
  }

private:
  void doPrint(int Level, uint64_t MaxKeySize) const {
    if (Key.size()) {
      // Compute the indentations for the current entry.
      uint64_t KeyIndentSize = Level * IndentSize;
      uint64_t ValIndentSize =
          MaxKeySize - (Key.size() + KeyIndentSize) + IndentSize;

      llvm::outs() << std::string(KeyIndentSize, ' ') << Key
                   << std::string(ValIndentSize, ' ');
      std::visit(
          [](auto &&V) {
            using T = std::decay_t<decltype(V)>;
            if constexpr (std::is_same_v<T, std::string>)
              llvm::outs() << V;
            else if constexpr (std::is_same_v<T, bool>)
              llvm::outs() << (V ? "Yes" : "No");
            else if constexpr (std::is_same_v<T, uint64_t>)
              llvm::outs() << V;
            else if constexpr (std::is_same_v<T, std::monostate>) {
              // Do nothing
            } else
              static_assert(false, "doPrint visit not exhaustive");
          },
          Value);
      llvm::outs() << (Units.empty() ? "" : " ") << Units << "\n";
    }

    // Print children
    if (Children)
      for (const auto &Entry : *Children)
        Entry.doPrint(Level + 1, MaxKeySize);
  }

  // Recursively calculates the maximum width of each key, including indentation
  uint64_t maxKeySize(int Level) const {
    uint64_t MaxKeySize = 0;

    if (Children)
      for (const auto &Entry : *Children) {
        uint64_t KeySize = Entry.Key.size() + Level * IndentSize;
        MaxKeySize = std::max(MaxKeySize, KeySize);
        MaxKeySize = std::max(MaxKeySize, Entry.maxKeySize(Level + 1));
      }

    return MaxKeySize;
  }
};

/// Class wrapping a __tgt_device_image and its offload entry table on a
/// specific device. This class is responsible for storing and managing
/// the offload entries for an image on a device.
class DeviceImageTy {
  /// Image identifier within the corresponding device. Notice that this id is
  /// not unique between different device; they may overlap.
  int32_t ImageId;

  /// The managed image data.
  std::unique_ptr<MemoryBuffer> Image;

  /// Reference to the device this image is loaded on.
  GenericDeviceTy &Device;

public:
  virtual ~DeviceImageTy() = default;

  DeviceImageTy(int32_t Id, GenericDeviceTy &Device,
                std::unique_ptr<MemoryBuffer> &&Image)
      : ImageId(Id), Image(std::move(Image)), Device(Device) {}

  /// Get the image identifier within the device.
  int32_t getId() const { return ImageId; }

  /// Get the device that this image is loaded onto.
  GenericDeviceTy &getDevice() const { return Device; }

  /// Get the image starting address.
  const void *getStart() const { return Image->getBufferStart(); }

  /// Get the image size.
  size_t getSize() const { return Image->getBufferSize(); }

  /// Get a memory buffer reference to the whole image.
  MemoryBufferRef getMemoryBuffer() const {
    return MemoryBufferRef(StringRef((const char *)getStart(), getSize()),
                           "Image");
  }
};

/// Class implementing common functionalities of offload kernels. Each plugin
/// should define the specific kernel class, derive from this generic one, and
/// implement the necessary virtual function members.
struct GenericKernelTy {
  /// Construct a kernel with a name and a execution mode.
  GenericKernelTy(const char *Name)
      : Name(Name), PreferredNumThreads(0), MaxNumThreads(0) {}

  virtual ~GenericKernelTy() {}

  /// Initialize the kernel object from a specific device.
  Error init(GenericDeviceTy &GenericDevice, DeviceImageTy &Image);
  virtual Error initImpl(GenericDeviceTy &GenericDevice,
                         DeviceImageTy &Image) = 0;

  /// Launch the kernel on the specific device. The device must be the same
  /// one used to initialize the kernel.
  Error launch(GenericDeviceTy &GenericDevice, void **ArgPtrs,
               ptrdiff_t *ArgOffsets, KernelArgsTy &KernelArgs,
               AsyncInfoWrapperTy &AsyncInfoWrapper) const;
  virtual Error launchImpl(GenericDeviceTy &GenericDevice,
                           uint32_t NumThreads[3], uint32_t NumBlocks[3],
                           KernelArgsTy &KernelArgs,
                           KernelLaunchParamsTy LaunchParams,
                           AsyncInfoWrapperTy &AsyncInfoWrapper) const = 0;

  virtual Expected<uint64_t> maxGroupSize(GenericDeviceTy &GenericDevice,
                                          uint64_t DynamicMemSize) const = 0;

  /// Get the kernel name.
  const char *getName() const { return Name.c_str(); }

  /// Get the kernel image.
  DeviceImageTy &getImage() const {
    assert(ImagePtr && "Kernel is not initialized!");
    return *ImagePtr;
  }

  /// Return the kernel environment object for kernel \p Name.
  const KernelEnvironmentTy &getKernelEnvironmentForKernel() {
    return KernelEnvironment;
  }

  /// Return a device pointer to a new kernel launch environment.
  Expected<KernelLaunchEnvironmentTy *>
  getKernelLaunchEnvironment(GenericDeviceTy &GenericDevice, uint32_t Version,
                             AsyncInfoWrapperTy &AsyncInfo) const;

  /// Indicate whether an execution mode is valid.
  static bool isValidExecutionMode(OMPTgtExecModeFlags ExecutionMode) {
    switch (ExecutionMode) {
    case OMP_TGT_EXEC_MODE_BARE:
    case OMP_TGT_EXEC_MODE_SPMD:
    case OMP_TGT_EXEC_MODE_GENERIC:
    case OMP_TGT_EXEC_MODE_GENERIC_SPMD:
    case OMP_TGT_EXEC_MODE_SPMD_NO_LOOP:
      return true;
    }
    return false;
  }

protected:
  /// Get the execution mode name of the kernel.
  const char *getExecutionModeName() const {
    switch (KernelEnvironment.Configuration.ExecMode) {
    case OMP_TGT_EXEC_MODE_BARE:
      return "BARE";
    case OMP_TGT_EXEC_MODE_SPMD:
      return "SPMD";
    case OMP_TGT_EXEC_MODE_GENERIC:
      return "Generic";
    case OMP_TGT_EXEC_MODE_GENERIC_SPMD:
      return "Generic-SPMD";
    case OMP_TGT_EXEC_MODE_SPMD_NO_LOOP:
      return "SPMD-No-Loop";
    }
    llvm_unreachable("Unknown execution mode!");
  }

  /// Prints generic kernel launch information.
  Error printLaunchInfo(GenericDeviceTy &GenericDevice,
                        KernelArgsTy &KernelArgs, uint32_t NumThreads[3],
                        uint32_t NumBlocks[3]) const;

  /// Prints plugin-specific kernel launch information after generic kernel
  /// launch information
  virtual Error printLaunchInfoDetails(GenericDeviceTy &GenericDevice,
                                       KernelArgsTy &KernelArgs,
                                       uint32_t NumThreads[3],
                                       uint32_t NumBlocks[3]) const;

private:
  /// Prepare the arguments before launching the kernel.
  KernelLaunchParamsTy
  prepareArgs(GenericDeviceTy &GenericDevice, void **ArgPtrs,
              ptrdiff_t *ArgOffsets, uint32_t &NumArgs,
              llvm::SmallVectorImpl<void *> &Args,
              llvm::SmallVectorImpl<void *> &Ptrs,
              KernelLaunchEnvironmentTy *KernelLaunchEnvironment) const;

  /// Get the number of threads and blocks for the kernel based on the
  /// user-defined threads and block clauses.
  uint32_t getNumThreads(GenericDeviceTy &GenericDevice,
                         uint32_t ThreadLimitClause[3]) const;

  /// The number of threads \p NumThreads can be adjusted by this method.
  /// \p IsNumThreadsFromUser is true is \p NumThreads is defined by user via
  /// thread_limit clause.
  uint32_t getNumBlocks(GenericDeviceTy &GenericDevice,
                        uint32_t BlockLimitClause[3], uint64_t LoopTripCount,
                        uint32_t &NumThreads, bool IsNumThreadsFromUser) const;

  /// Indicate if the kernel works in Generic SPMD, Generic, No-Loop
  /// or SPMD mode.
  bool isGenericSPMDMode() const {
    return KernelEnvironment.Configuration.ExecMode ==
           OMP_TGT_EXEC_MODE_GENERIC_SPMD;
  }
  bool isGenericMode() const {
    return KernelEnvironment.Configuration.ExecMode ==
           OMP_TGT_EXEC_MODE_GENERIC;
  }
  bool isSPMDMode() const {
    return KernelEnvironment.Configuration.ExecMode == OMP_TGT_EXEC_MODE_SPMD;
  }
  bool isBareMode() const {
    return KernelEnvironment.Configuration.ExecMode == OMP_TGT_EXEC_MODE_BARE;
  }
  bool isNoLoopMode() const {
    return KernelEnvironment.Configuration.ExecMode ==
           OMP_TGT_EXEC_MODE_SPMD_NO_LOOP;
  }

  /// The kernel name.
  std::string Name;

  /// The image that contains this kernel.
  DeviceImageTy *ImagePtr = nullptr;

protected:
  /// The preferred number of threads to run the kernel.
  uint32_t PreferredNumThreads;

  /// The maximum number of threads which the kernel could leverage.
  uint32_t MaxNumThreads;

  /// The kernel environment, including execution flags.
  KernelEnvironmentTy KernelEnvironment;

  /// The prototype kernel launch environment.
  KernelLaunchEnvironmentTy KernelLaunchEnvironment;
};

/// Information about an allocation, when it has been allocated, and when/if it
/// has been deallocated, for error reporting purposes.
struct AllocationTraceInfoTy {

  /// The stack trace of the allocation itself.
  std::string AllocationTrace;

  /// The stack trace of the deallocation, or empty.
  std::string DeallocationTrace;

  /// The allocated device pointer.
  void *DevicePtr = nullptr;

  /// The corresponding host pointer (can be null).
  void *HostPtr = nullptr;

  /// The size of the allocation.
  uint64_t Size = 0;

  /// The kind of the allocation.
  TargetAllocTy Kind = TargetAllocTy::TARGET_ALLOC_DEFAULT;

  /// Information about the last allocation at this address, if any.
  AllocationTraceInfoTy *LastAllocationInfo = nullptr;

  /// Lock to keep accesses race free.
  std::mutex Lock;
};

/// Information about an allocation, when it has been allocated, and when/if it
/// has been deallocated, for error reporting purposes.
struct KernelTraceInfoTy {

  /// The launched kernel.
  GenericKernelTy *Kernel;

  /// The stack trace of the launch itself.
  std::string LaunchTrace;

  /// The async info the kernel was launched in.
  __tgt_async_info *AsyncInfo;
};

struct KernelTraceInfoRecordTy {
  KernelTraceInfoRecordTy() { KTIs.fill({}); }

  /// Return the (maximal) record size.
  auto size() const { return KTIs.size(); }

  /// Create a new kernel trace info and add it into the record.
  void emplace(GenericKernelTy *Kernel, const std::string &&StackTrace,
               __tgt_async_info *AsyncInfo) {
    KTIs[Idx] = {Kernel, std::move(StackTrace), AsyncInfo};
    Idx = (Idx + 1) % size();
  }

  /// Return the \p I'th last kernel trace info.
  auto getKernelTraceInfo(int32_t I) const {
    // Note that kernel trace infos "grow forward", so lookup is backwards.
    return KTIs[(Idx - I - 1 + size()) % size()];
  }

private:
  std::array<KernelTraceInfoTy, 8> KTIs;
  unsigned Idx = 0;
};

/// Class representing a map of host pinned allocations. We track these pinned
/// allocations, so memory transfers involving these buffers can be optimized.
class PinnedAllocationMapTy {

  /// Struct representing a map entry.
  struct EntryTy {
    /// The host pointer of the pinned allocation.
    void *HstPtr;

    /// The pointer that devices' driver should use to transfer data from/to the
    /// pinned allocation. In most plugins, this pointer will be the same as the
    /// host pointer above.
    void *DevAccessiblePtr;

    /// The size of the pinned allocation.
    size_t Size;

    /// Indicate whether the allocation was locked from outside the plugin, for
    /// instance, from the application. The externally locked allocations are
    /// not unlocked by the plugin when unregistering the last user.
    bool ExternallyLocked;

    /// The number of references to the pinned allocation. The allocation should
    /// remain pinned and registered to the map until the number of references
    /// becomes zero.
    mutable size_t References;

    /// Create an entry with the host and device accessible pointers, the buffer
    /// size, and a boolean indicating whether the buffer was locked externally.
    EntryTy(void *HstPtr, void *DevAccessiblePtr, size_t Size,
            bool ExternallyLocked)
        : HstPtr(HstPtr), DevAccessiblePtr(DevAccessiblePtr), Size(Size),
          ExternallyLocked(ExternallyLocked), References(1) {}

    /// Utility constructor used for std::set searches.
    EntryTy(void *HstPtr)
        : HstPtr(HstPtr), DevAccessiblePtr(nullptr), Size(0),
          ExternallyLocked(false), References(0) {}
  };

  /// Comparator of mep entries. Use the host pointer to enforce an order
  /// between entries.
  struct EntryCmpTy {
    bool operator()(const EntryTy &Left, const EntryTy &Right) const {
      return Left.HstPtr < Right.HstPtr;
    }
  };

  typedef std::set<EntryTy, EntryCmpTy> PinnedAllocSetTy;

  /// The map of host pinned allocations.
  PinnedAllocSetTy Allocs;

  /// The mutex to protect accesses to the map.
  mutable std::shared_mutex Mutex;

  /// Reference to the corresponding device.
  GenericDeviceTy &Device;

  /// Indicate whether mapped host buffers should be locked automatically.
  bool LockMappedBuffers;

  /// Indicate whether failures when locking mapped buffers should be ignored.
  bool IgnoreLockMappedFailures;

  /// Find an allocation that intersects with \p HstPtr pointer. Assume the
  /// map's mutex is acquired.
  const EntryTy *findIntersecting(const void *HstPtr) const {
    if (Allocs.empty())
      return nullptr;

    // Search the first allocation with starting address that is not less than
    // the buffer address.
    auto It = Allocs.lower_bound({const_cast<void *>(HstPtr)});

    // Direct match of starting addresses.
    if (It != Allocs.end() && It->HstPtr == HstPtr)
      return &(*It);

    // Not direct match but may be a previous pinned allocation in the map which
    // contains the buffer. Return false if there is no such a previous
    // allocation.
    if (It == Allocs.begin())
      return nullptr;

    // Move to the previous pinned allocation.
    --It;

    // The buffer is not contained in the pinned allocation.
    if (utils::advancePtr(It->HstPtr, It->Size) > HstPtr)
      return &(*It);

    // None found.
    return nullptr;
  }

  /// Insert an entry to the map representing a locked buffer. The number of
  /// references is set to one.
  Error insertEntry(void *HstPtr, void *DevAccessiblePtr, size_t Size,
                    bool ExternallyLocked = false);

  /// Erase an existing entry from the map.
  Error eraseEntry(const EntryTy &Entry);

  /// Register a new user into an entry that represents a locked buffer. Check
  /// also that the registered buffer with \p HstPtr address and \p Size is
  /// actually contained into the entry.
  Error registerEntryUse(const EntryTy &Entry, void *HstPtr, size_t Size);

  /// Unregister a user from the entry and return whether it is the last user.
  /// If it is the last user, the entry will have to be removed from the map
  /// and unlock the entry's host buffer (if necessary).
  Expected<bool> unregisterEntryUse(const EntryTy &Entry);

  /// Indicate whether the first range A fully contains the second range B.
  static bool contains(void *PtrA, size_t SizeA, void *PtrB, size_t SizeB) {
    void *EndA = utils::advancePtr(PtrA, SizeA);
    void *EndB = utils::advancePtr(PtrB, SizeB);
    return (PtrB >= PtrA && EndB <= EndA);
  }

  /// Indicate whether the first range A intersects with the second range B.
  static bool intersects(void *PtrA, size_t SizeA, void *PtrB, size_t SizeB) {
    void *EndA = utils::advancePtr(PtrA, SizeA);
    void *EndB = utils::advancePtr(PtrB, SizeB);
    return (PtrA < EndB && PtrB < EndA);
  }

public:
  /// Create the map of pinned allocations corresponding to a specific device.
  PinnedAllocationMapTy(GenericDeviceTy &Device) : Device(Device) {

    // Envar that indicates whether mapped host buffers should be locked
    // automatically. The possible values are boolean (on/off) and a special:
    //   off:       Mapped host buffers are not locked.
    //   on:        Mapped host buffers are locked in a best-effort approach.
    //              Failure to lock the buffers are silent.
    //   mandatory: Mapped host buffers are always locked and failures to lock
    //              a buffer results in a fatal error.
    StringEnvar OMPX_LockMappedBuffers("LIBOMPTARGET_LOCK_MAPPED_HOST_BUFFERS",
                                       "off");

    bool Enabled;
    if (StringParser::parse(OMPX_LockMappedBuffers.get().data(), Enabled)) {
      // Parsed as a boolean value. Enable the feature if necessary.
      LockMappedBuffers = Enabled;
      IgnoreLockMappedFailures = true;
    } else if (OMPX_LockMappedBuffers.get() == "mandatory") {
      // Enable the feature and failures are fatal.
      LockMappedBuffers = true;
      IgnoreLockMappedFailures = false;
    } else {
      // Disable by default.
      DP("Invalid value LIBOMPTARGET_LOCK_MAPPED_HOST_BUFFERS=%s\n",
         OMPX_LockMappedBuffers.get().data());
      LockMappedBuffers = false;
    }
  }

  /// Register a buffer that was recently allocated as a locked host buffer.
  /// None of the already registered pinned allocations should intersect with
  /// this new one. The registration requires the host pointer in \p HstPtr,
  /// the device accessible pointer in \p DevAccessiblePtr, and the size of the
  /// allocation in \p Size. The allocation must be unregistered using the
  /// unregisterHostBuffer function.
  Error registerHostBuffer(void *HstPtr, void *DevAccessiblePtr, size_t Size);

  /// Unregister a host pinned allocation passing the host pointer which was
  /// previously registered using the registerHostBuffer function. When calling
  /// this function, the pinned allocation cannot have any other user and will
  /// not be unlocked by this function.
  Error unregisterHostBuffer(void *HstPtr);

  /// Lock the host buffer at \p HstPtr or register a new user if it intersects
  /// with an already existing one. A partial overlapping with extension is not
  /// allowed. The function returns the device accessible pointer of the pinned
  /// buffer. The buffer must be unlocked using the unlockHostBuffer function.
  Expected<void *> lockHostBuffer(void *HstPtr, size_t Size);

  /// Unlock the host buffer at \p HstPtr or unregister a user if other users
  /// are still using the pinned allocation. If this was the last user, the
  /// pinned allocation is removed from the map and the memory is unlocked.
  Error unlockHostBuffer(void *HstPtr);

  /// Lock or register a host buffer that was recently mapped by libomptarget.
  /// This behavior is applied if LIBOMPTARGET_LOCK_MAPPED_HOST_BUFFERS is
  /// enabled. Even if not enabled, externally locked buffers are registered
  /// in order to optimize their transfers.
  Error lockMappedHostBuffer(void *HstPtr, size_t Size);

  /// Unlock or unregister a host buffer that was unmapped by libomptarget.
  Error unlockUnmappedHostBuffer(void *HstPtr);

  /// Return the device accessible pointer associated to the host pinned
  /// allocation which the \p HstPtr belongs, if any. Return null in case the
  /// \p HstPtr does not belong to any host pinned allocation. The device
  /// accessible pointer is the one that devices should use for data transfers
  /// that involve a host pinned buffer.
  void *getDeviceAccessiblePtrFromPinnedBuffer(const void *HstPtr) const {
    std::shared_lock<std::shared_mutex> Lock(Mutex);

    // Find the intersecting allocation if any.
    const EntryTy *Entry = findIntersecting(HstPtr);
    if (!Entry)
      return nullptr;

    return utils::advancePtr(Entry->DevAccessiblePtr,
                             utils::getPtrDiff(HstPtr, Entry->HstPtr));
  }

  /// Check whether a buffer belongs to a registered host pinned allocation.
  bool isHostPinnedBuffer(const void *HstPtr) const {
    std::shared_lock<std::shared_mutex> Lock(Mutex);

    // Return whether there is an intersecting allocation.
    return (findIntersecting(const_cast<void *>(HstPtr)) != nullptr);
  }
};

/// Class implementing common functionalities of offload devices. Each plugin
/// should define the specific device class, derive from this generic one, and
/// implement the necessary virtual function members.
struct GenericDeviceTy : public DeviceAllocatorTy {
  /// Construct a device with its device id within the plugin, the number of
  /// devices in the plugin and the grid values for that kind of device.
  GenericDeviceTy(GenericPluginTy &Plugin, int32_t DeviceId, int32_t NumDevices,
                  const llvm::omp::GV &GridValues);

  /// Get the device identifier within the corresponding plugin. Notice that
  /// this id is not unique between different plugins; they may overlap.
  int32_t getDeviceId() const { return DeviceId; }

  /// Set the context of the device if needed, before calling device-specific
  /// functions. Plugins may implement this function as a no-op if not needed.
  virtual Error setContext() = 0;

  /// Initialize the device. After this call, the device should be already
  /// working and ready to accept queries or modifications.
  Error init(GenericPluginTy &Plugin);
  virtual Error initImpl(GenericPluginTy &Plugin) = 0;

  /// Deinitialize the device and free all its resources. After this call, the
  /// device is no longer considered ready, so no queries or modifications are
  /// allowed.
  Error deinit(GenericPluginTy &Plugin);
  virtual Error deinitImpl() = 0;

  /// Load the binary image into the device and return the target table.
  Expected<DeviceImageTy *> loadBinary(GenericPluginTy &Plugin,
                                       StringRef TgtImage);
  virtual Expected<DeviceImageTy *>
  loadBinaryImpl(std::unique_ptr<MemoryBuffer> &&TgtImage, int32_t ImageId) = 0;

  /// Unload a previously loaded Image from the device
  Error unloadBinary(DeviceImageTy *Image);
  virtual Error unloadBinaryImpl(DeviceImageTy *Image) = 0;

  /// Setup the global device memory pool, if the plugin requires one.
  Error setupDeviceMemoryPool(GenericPluginTy &Plugin, DeviceImageTy &Image,
                              uint64_t PoolSize);

  // Setup the RPC server for this device if needed. This may not run on some
  // plugins like the CPU targets. By default, it will not be executed so it is
  // up to the target to override this using the shouldSetupRPCServer function.
  Error setupRPCServer(GenericPluginTy &Plugin, DeviceImageTy &Image);

  /// Synchronize the current thread with the pending operations on the
  /// __tgt_async_info structure. If ReleaseQueue is false, then the
  // underlying queue will not be released. In this case, additional
  // work may be submitted to the queue whilst a synchronize is running.
  Error synchronize(__tgt_async_info *AsyncInfo, bool ReleaseQueue = true);
  virtual Error synchronizeImpl(__tgt_async_info &AsyncInfo,
                                bool ReleaseQueue) = 0;

  /// Invokes any global constructors on the device if present and is required
  /// by the target.
  virtual Error callGlobalConstructors(GenericPluginTy &Plugin,
                                       DeviceImageTy &Image) {
    return Error::success();
  }

  /// Invokes any global destructors on the device if present and is required
  /// by the target.
  virtual Error callGlobalDestructors(GenericPluginTy &Plugin,
                                      DeviceImageTy &Image) {
    return Error::success();
  }

  /// Query for the completion of the pending operations on the __tgt_async_info
  /// structure in a non-blocking manner.
  Error queryAsync(__tgt_async_info *AsyncInfo);
  virtual Error queryAsyncImpl(__tgt_async_info &AsyncInfo) = 0;

  /// Check whether the architecture supports VA management
  virtual bool supportVAManagement() const { return false; }

  /// Get the total device memory size
  virtual Error getDeviceMemorySize(uint64_t &DSize);

  /// Allocates \p RSize bytes (rounded up to page size) and hints the driver to
  /// map it to \p VAddr. The obtained address is stored in \p Addr. At return
  /// \p RSize contains the actual size which can be equal or larger than the
  /// requested size.
  virtual Error memoryVAMap(void **Addr, void *VAddr, size_t *RSize);

  /// De-allocates device memory and unmaps the virtual address \p VAddr
  virtual Error memoryVAUnMap(void *VAddr, size_t Size);

  /// Allocate data on the device or involving the device.
  Expected<void *> dataAlloc(int64_t Size, void *HostPtr, TargetAllocTy Kind);

  /// Deallocate data from the device or involving the device.
  Error dataDelete(void *TgtPtr, TargetAllocTy Kind);

  /// Pin host memory to optimize transfers and return the device accessible
  /// pointer that devices should use for memory transfers involving the host
  /// pinned allocation.
  Expected<void *> dataLock(void *HstPtr, int64_t Size) {
    return PinnedAllocs.lockHostBuffer(HstPtr, Size);
  }

  /// Unpin a host memory buffer that was previously pinned.
  Error dataUnlock(void *HstPtr) {
    return PinnedAllocs.unlockHostBuffer(HstPtr);
  }

  /// Lock the host buffer \p HstPtr with \p Size bytes with the vendor-specific
  /// API and return the device accessible pointer.
  virtual Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) = 0;

  /// Unlock a previously locked host buffer starting at \p HstPtr.
  virtual Error dataUnlockImpl(void *HstPtr) = 0;

  /// Mark the host buffer with address \p HstPtr and \p Size bytes as a mapped
  /// buffer. This means that libomptarget created a new mapping of that host
  /// buffer (e.g., because a user OpenMP target map) and the buffer may be used
  /// as source/destination of memory transfers. We can use this information to
  /// lock the host buffer and optimize its memory transfers.
  Error notifyDataMapped(void *HstPtr, int64_t Size) {
    return PinnedAllocs.lockMappedHostBuffer(HstPtr, Size);
  }

  /// Mark the host buffer with address \p HstPtr as unmapped. This means that
  /// libomptarget removed an existing mapping. If the plugin locked the buffer
  /// in notifyDataMapped, this function should unlock it.
  Error notifyDataUnmapped(void *HstPtr) {
    return PinnedAllocs.unlockUnmappedHostBuffer(HstPtr);
  }

  /// Check whether the host buffer with address \p HstPtr is pinned by the
  /// underlying vendor-specific runtime (if any). Retrieve the host pointer,
  /// the device accessible pointer and the size of the original pinned buffer.
  virtual Expected<bool> isPinnedPtrImpl(void *HstPtr, void *&BaseHstPtr,
                                         void *&BaseDevAccessiblePtr,
                                         size_t &BaseSize) const = 0;

  /// Submit data to the device (host to device transfer).
  Error dataSubmit(void *TgtPtr, const void *HstPtr, int64_t Size,
                   __tgt_async_info *AsyncInfo);
  virtual Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                               AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  /// Retrieve data from the device (device to host transfer).
  Error dataRetrieve(void *HstPtr, const void *TgtPtr, int64_t Size,
                     __tgt_async_info *AsyncInfo);
  virtual Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                                 AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  /// Instert a data fence between previous data operations and the following
  /// operations if necessary for the device
  virtual Error dataFence(__tgt_async_info *AsyncInfo) = 0;

  /// Exchange data between devices (device to device transfer). Calling this
  /// function is only valid if GenericPlugin::isDataExchangable() passing the
  /// two devices returns true.
  Error dataExchange(const void *SrcPtr, GenericDeviceTy &DstDev, void *DstPtr,
                     int64_t Size, __tgt_async_info *AsyncInfo);
  virtual Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstDev,
                                 void *DstPtr, int64_t Size,
                                 AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  /// Fill data on the device with a pattern from the host
  Error dataFill(void *TgtPtr, const void *PatternPtr, int64_t PatternSize,
                 int64_t Size, __tgt_async_info *AsyncInfo);
  virtual Error dataFillImpl(void *TgtPtr, const void *PatternPtr,
                             int64_t PatternSize, int64_t Size,
                             AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  /// Run the kernel associated with \p EntryPtr
  Error launchKernel(void *EntryPtr, void **ArgPtrs, ptrdiff_t *ArgOffsets,
                     KernelArgsTy &KernelArgs, __tgt_async_info *AsyncInfo);

  /// Initialize a __tgt_async_info structure.
  Error initAsyncInfo(__tgt_async_info **AsyncInfoPtr);
  virtual Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  /// Enqueue a host call to AsyncInfo
  Error enqueueHostCall(void (*Callback)(void *), void *UserData,
                        __tgt_async_info *AsyncInfo);
  virtual Error enqueueHostCallImpl(void (*Callback)(void *), void *UserData,
                                    AsyncInfoWrapperTy &AsyncInfo) = 0;

  /// Create an event.
  Error createEvent(void **EventPtrStorage);
  virtual Error createEventImpl(void **EventPtrStorage) = 0;

  /// Destroy an event.
  Error destroyEvent(void *Event);
  virtual Error destroyEventImpl(void *EventPtr) = 0;

  /// Start the recording of the event.
  Error recordEvent(void *Event, __tgt_async_info *AsyncInfo);
  virtual Error recordEventImpl(void *EventPtr,
                                AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  /// Wait for an event to finish. Notice this wait is asynchronous if the
  /// __tgt_async_info is not nullptr.
  Error waitEvent(void *Event, __tgt_async_info *AsyncInfo);
  virtual Error waitEventImpl(void *EventPtr,
                              AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  /// Check if the event enqueued to AsyncInfo is complete
  Expected<bool> isEventComplete(void *Event, __tgt_async_info *AsyncInfo);
  virtual Expected<bool>
  isEventCompleteImpl(void *EventPtr, AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  /// Synchronize the current thread with the event.
  Error syncEvent(void *EventPtr);
  virtual Error syncEventImpl(void *EventPtr) = 0;

  /// Print information about the device.
  Error printInfo();
  virtual Expected<InfoTreeNode> obtainInfoImpl() = 0;

  /// Return true if the device has work that is either queued or currently
  /// running
  ///
  /// Devices which cannot report this information should always return true
  Expected<bool> hasPendingWork(__tgt_async_info *AsyncInfo);
  virtual Expected<bool>
  hasPendingWorkImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  /// Getters of the grid values.
  uint32_t getWarpSize() const { return GridValues.GV_Warp_Size; }
  uint32_t getThreadLimit() const { return GridValues.GV_Max_WG_Size; }
  uint32_t getBlockLimit() const { return GridValues.GV_Max_Teams; }
  uint32_t getDefaultNumThreads() const {
    return GridValues.GV_Default_WG_Size;
  }
  uint32_t getDefaultNumBlocks() const {
    return GridValues.GV_Default_Num_Teams;
  }
  uint32_t getDebugKind() const { return OMPX_DebugKind; }
  uint32_t getDynamicMemorySize() const { return OMPX_SharedMemorySize; }
  virtual uint64_t getClockFrequency() const { return CLOCKS_PER_SEC; }

  /// Get target compute unit kind (e.g., sm_80, or gfx908).
  virtual std::string getComputeUnitKind() const { return "unknown"; }

  /// Post processing after jit backend. The ownership of \p MB will be taken.
  virtual Expected<std::unique_ptr<MemoryBuffer>>
  doJITPostProcessing(std::unique_ptr<MemoryBuffer> MB) const {
    return std::move(MB);
  }

  /// The minimum number of threads we use for a low-trip count combined loop.
  /// Instead of using more threads we increase the outer (block/team)
  /// parallelism.
  /// @see OMPX_MinThreadsForLowTripCount
  virtual uint32_t getMinThreadsForLowTripCountLoop() {
    return OMPX_MinThreadsForLowTripCount;
  }

  /// Whether or not to reuse blocks for high trip count loops.
  /// @see OMPX_ReuseBlocksForHighTripCount
  bool getReuseBlocksForHighTripCount() {
    return OMPX_ReuseBlocksForHighTripCount;
  }

  /// Get the total amount of hardware parallelism supported by the target
  /// device. This is the total amount of warps or wavefronts that can be
  /// resident on the device simultaneously.
  virtual uint64_t getHardwareParallelism() const { return 0; }

  /// Get the RPC server running on this device.
  RPCServerTy *getRPCServer() const { return RPCServer; }

  /// The number of parallel RPC ports to use on the device. In general, this
  /// should be roughly equivalent to the amount of hardware parallelism the
  /// device can support. This is because GPUs in general do not have forward
  /// progress guarantees, so we minimize thread level dependencies by
  /// allocating enough space such that each device thread can have a port. This
  /// is likely overly pessimistic in the average case, but guarantees no
  /// deadlocks at the cost of memory. This must be overloaded by targets
  /// expecting to use the RPC server.
  virtual uint64_t requestedRPCPortCount() const {
    assert(!shouldSetupRPCServer() && "Default implementation cannot be used");
    return 0;
  }

  virtual Error getDeviceStackSize(uint64_t &V) = 0;

  /// Returns true if current plugin architecture is an APU
  /// and unified_shared_memory was not requested by the program.
  bool useAutoZeroCopy();
  virtual bool useAutoZeroCopyImpl() { return false; }

  virtual Expected<omp_interop_val_t *>
  createInterop(int32_t InteropType, interop_spec_t &InteropSpec) {
    return nullptr;
  }

  virtual Error releaseInterop(omp_interop_val_t *Interop) {
    return Plugin::success();
  }

  virtual interop_spec_t selectInteropPreference(int32_t InteropType,
                                                 int32_t NumPrefers,
                                                 interop_spec_t *Prefers) {
    return interop_spec_t{tgt_fr_none, {false, 0}, 0};
  }

  /// Allocate and construct a kernel object.
  virtual Expected<GenericKernelTy &> constructKernel(const char *Name) = 0;

  /// Reference to the underlying plugin that created this device.
  GenericPluginTy &Plugin;

  /// Map to record when allocations have been performed, and when they have
  /// been deallocated, both for error reporting purposes.
  ProtectedObj<DenseMap<void *, AllocationTraceInfoTy *>> AllocationTraces;

  /// Return the allocation trace info for a device pointer, that is the
  /// allocation into which this device pointer points to (or pointed into).
  AllocationTraceInfoTy *getAllocationTraceInfoForAddr(void *DevicePtr) {
    auto AllocationTraceMap = AllocationTraces.getExclusiveAccessor();
    for (auto &It : *AllocationTraceMap) {
      if (It.first <= DevicePtr &&
          utils::advancePtr(It.first, It.second->Size) > DevicePtr)
        return It.second;
    }
    return nullptr;
  }

  /// Return the allocation trace info for a device pointer, that is the
  /// allocation into which this device pointer points to (or pointed into).
  AllocationTraceInfoTy *
  getClosestAllocationTraceInfoForAddr(void *DevicePtr, uintptr_t &Distance) {
    Distance = 0;
    if (auto *ATI = getAllocationTraceInfoForAddr(DevicePtr)) {
      return ATI;
    }

    AllocationTraceInfoTy *ATI = nullptr;
    uintptr_t DevicePtrI = uintptr_t(DevicePtr);
    auto AllocationTraceMap = AllocationTraces.getExclusiveAccessor();
    for (auto &It : *AllocationTraceMap) {
      uintptr_t Begin = uintptr_t(It.second->DevicePtr);
      uintptr_t End = Begin + It.second->Size - 1;
      uintptr_t ItDistance = std::min(Begin - DevicePtrI, DevicePtrI - End);
      if (ATI && ItDistance > Distance)
        continue;
      ATI = It.second;
      Distance = ItDistance;
    }
    return ATI;
  }

  /// Map to record kernel have been launchedl, for error reporting purposes.
  ProtectedObj<KernelTraceInfoRecordTy> KernelLaunchTraces;

  /// Environment variable to determine if stack traces for kernel launches are
  /// tracked.
  UInt32Envar OMPX_TrackNumKernelLaunches =
      UInt32Envar("OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES", 0);

  /// Environment variable to determine if stack traces for allocations and
  /// deallocations are tracked.
  BoolEnvar OMPX_TrackAllocationTraces =
      BoolEnvar("OFFLOAD_TRACK_ALLOCATION_TRACES", false);

  /// Array of images loaded into the device. Images are automatically
  /// deallocated by the allocator.
  llvm::SmallVector<DeviceImageTy *> LoadedImages;

private:
  /// Get and set the stack size and heap size for the device. If not used, the
  /// plugin can implement the setters as no-op and setting the output
  /// value to zero for the getters.
  virtual Error setDeviceStackSize(uint64_t V) = 0;
  virtual Error getDeviceHeapSize(uint64_t &V) = 0;
  virtual Error setDeviceHeapSize(uint64_t V) = 0;

  /// Indicate whether the device should setup the global device memory pool. If
  /// false is return the value on the device will be uninitialized.
  virtual bool shouldSetupDeviceMemoryPool() const { return true; }

  /// Indicate whether or not the device should setup the RPC server. This is
  /// only necessary for unhosted targets like the GPU.
  virtual bool shouldSetupRPCServer() const { return false; }

  /// Pointer to the memory manager or nullptr if not available.
  MemoryManagerTy *MemoryManager;

  /// Per device setting of MemoryManager's Threshold
  virtual size_t getMemoryManagerSizeThreshold() { return 0; }

  /// Environment variables defined by the OpenMP standard.
  Int32Envar OMP_TeamLimit;
  Int32Envar OMP_NumTeams;
  Int32Envar OMP_TeamsThreadLimit;

  /// Environment variables defined by the LLVM OpenMP implementation.
  Int32Envar OMPX_DebugKind;
  UInt32Envar OMPX_SharedMemorySize;
  UInt64Envar OMPX_TargetStackSize;
  UInt64Envar OMPX_TargetHeapSize;

  /// Environment flag to set the minimum number of threads we use for a
  /// low-trip count combined loop. Instead of using more threads we increase
  /// the outer (block/team) parallelism.
  UInt32Envar OMPX_MinThreadsForLowTripCount =
      UInt32Envar("LIBOMPTARGET_MIN_THREADS_FOR_LOW_TRIP_COUNT", 32);

  BoolEnvar OMPX_ReuseBlocksForHighTripCount =
      BoolEnvar("LIBOMPTARGET_REUSE_BLOCKS_FOR_HIGH_TRIP_COUNT", true);

protected:
  /// Environment variables defined by the LLVM OpenMP implementation
  /// regarding the initial number of streams and events.
  UInt32Envar OMPX_InitialNumStreams;
  UInt32Envar OMPX_InitialNumEvents;

  /// The identifier of the device within the plugin. Notice this is not a
  /// global device id and is not the device id visible to the OpenMP user.
  const int32_t DeviceId;

  /// The default grid values used for this device.
  llvm::omp::GV GridValues;

  /// Enumeration used for representing the current state between two devices
  /// two devices (both under the same plugin) for the peer access between them.
  /// The states can be a) PENDING when the state has not been queried and needs
  /// to be queried, b) AVAILABLE when the peer access is available to be used,
  /// and c) UNAVAILABLE if the system does not allow it.
  enum class PeerAccessState : uint8_t { AVAILABLE, UNAVAILABLE, PENDING };

  /// Array of peer access states with the rest of devices. This means that if
  /// the device I has a matrix PeerAccesses with PeerAccesses == AVAILABLE,
  /// the device I can access device J's memory directly. However, notice this
  /// does not mean that device J can access device I's memory directly.
  llvm::SmallVector<PeerAccessState> PeerAccesses;
  std::mutex PeerAccessesLock;

  /// Map of host pinned allocations used for optimize device transfers.
  PinnedAllocationMapTy PinnedAllocs;

  /// A pointer to an RPC server instance attached to this device if present.
  /// This is used to run the RPC server during task synchronization.
  RPCServerTy *RPCServer;

#ifdef OMPT_SUPPORT
  /// OMPT callback functions
#define defineOmptCallback(Name, Type, Code) Name##_t Name##_fn = nullptr;
  FOREACH_OMPT_DEVICE_EVENT(defineOmptCallback)
#undef defineOmptCallback

  /// Internal representation for OMPT device (initialize & finalize)
  std::atomic<bool> OmptInitialized;
#endif

private:
  DeviceMemoryPoolTy DeviceMemoryPool = {nullptr, 0};
  DeviceMemoryPoolTrackingTy DeviceMemoryPoolTracking = {0, 0, ~0U, 0};
};

/// Class implementing common functionalities of offload plugins. Each plugin
/// should define the specific plugin class, derive from this generic one, and
/// implement the necessary virtual function members.
struct GenericPluginTy {

  /// Construct a plugin instance.
  GenericPluginTy(Triple::ArchType TA)
      : GlobalHandler(nullptr), JIT(TA), RPCServer(nullptr),
        RecordReplay(nullptr) {}

  virtual ~GenericPluginTy() {}

  /// Initialize the plugin.
  Error init();

  /// Initialize the plugin and return the number of available devices.
  virtual Expected<int32_t> initImpl() = 0;

  /// Deinitialize the plugin and release the resources.
  Error deinit();
  virtual Error deinitImpl() = 0;

  /// Create a new device for the underlying plugin.
  virtual GenericDeviceTy *createDevice(GenericPluginTy &Plugin,
                                        int32_t DeviceID,
                                        int32_t NumDevices) = 0;

  /// Create a new global handler for the underlying plugin.
  virtual GenericGlobalHandlerTy *createGlobalHandler() = 0;

  /// Get the reference to the device with a certain device id.
  GenericDeviceTy &getDevice(int32_t DeviceId) {
    assert(isValidDeviceId(DeviceId) && "Invalid device id");
    assert(Devices[DeviceId] && "Device is uninitialized");

    return *Devices[DeviceId];
  }

  /// Get the number of active devices.
  int32_t getNumDevices() const { return NumDevices; }

  /// Get the plugin-specific device identifier.
  int32_t getUserId(int32_t DeviceId) const {
    assert(UserDeviceIds.contains(DeviceId) && "No user-id registered");
    return UserDeviceIds.at(DeviceId);
  }

  /// Get the ELF code to recognize the binary image of this plugin.
  virtual uint16_t getMagicElfBits() const = 0;

  /// Get the target triple of this plugin.
  virtual Triple::ArchType getTripleArch() const = 0;

  /// Get the constant name identifier for this plugin.
  virtual const char *getName() const = 0;

  /// Allocate a structure using the internal allocator.
  template <typename Ty> Ty *allocate() {
    return reinterpret_cast<Ty *>(Allocator.Allocate(sizeof(Ty), alignof(Ty)));
  }

  template <typename Ty> void free(Ty *Mem) { Allocator.Deallocate(Mem); }

  /// Get the reference to the global handler of this plugin.
  GenericGlobalHandlerTy &getGlobalHandler() {
    assert(GlobalHandler && "Global handler not initialized");
    return *GlobalHandler;
  }

  /// Get the reference to the JIT used for all devices connected to this
  /// plugin.
  JITEngine &getJIT() { return JIT; }

  /// Get a reference to the RPC server used to provide host services.
  RPCServerTy &getRPCServer() {
    assert(RPCServer && "RPC server not initialized");
    return *RPCServer;
  }

  /// Get a reference to the record and replay interface for the plugin.
  RecordReplayTy &getRecordReplay() {
    assert(RecordReplay && "RR interface not initialized");
    return *RecordReplay;
  }

  /// Initialize a device within the plugin.
  Error initDevice(int32_t DeviceId);

  /// Deinitialize a device within the plugin and release its resources.
  Error deinitDevice(int32_t DeviceId);

  /// Indicate whether data can be exchanged directly between two devices under
  /// this same plugin. If this function returns true, it's safe to call the
  /// GenericDeviceTy::exchangeData() function on the source device.
  virtual bool isDataExchangable(int32_t SrcDeviceId, int32_t DstDeviceId) {
    return isValidDeviceId(SrcDeviceId) && isValidDeviceId(DstDeviceId);
  }

  /// Top level interface to verify if a given ELF image can be executed on a
  /// given target. Returns true if the \p Image is compatible with the plugin.
  Expected<bool> checkELFImage(StringRef Image) const;

  /// Return true if the \p Image can be compiled to run on the platform's
  /// target architecture.
  Expected<bool> checkBitcodeImage(StringRef Image) const;

  /// Indicate if an image is compatible with the plugin devices. Notice that
  /// this function may be called before actually initializing the devices. So
  /// we could not move this function into GenericDeviceTy.
  virtual Expected<bool> isELFCompatible(uint32_t DeviceID,
                                         StringRef Image) const = 0;

  virtual Error flushQueueImpl(omp_interop_val_t *Interop) {
    return Plugin::success();
  }

  virtual Error syncBarrierImpl(omp_interop_val_t *Interop) {
    return Plugin::error(error::ErrorCode::UNSUPPORTED,
                         "sync_barrier not supported");
  }

  virtual Error asyncBarrierImpl(omp_interop_val_t *Interop) {
    return Plugin::error(error::ErrorCode::UNSUPPORTED,
                         "async_barrier not supported");
  }

protected:
  /// Indicate whether a device id is valid.
  bool isValidDeviceId(int32_t DeviceId) const {
    return (DeviceId >= 0 && DeviceId < getNumDevices());
  }

public:
  // TODO: This plugin interface needs to be cleaned up.

  /// Returns non-zero if the plugin runtime has been initialized.
  int32_t is_initialized() const;

  /// Returns non-zero if the \p Image is compatible with the plugin. This
  /// function does not require the plugin to be initialized before use.
  int32_t isPluginCompatible(StringRef Image);

  /// Returns non-zero if the \p Image is compatible with the device.
  int32_t isDeviceCompatible(int32_t DeviceId, StringRef Image);

  /// Returns non-zero if the plugin device has been initialized.
  int32_t is_device_initialized(int32_t DeviceId) const;

  /// Initialize the device inside of the plugin.
  int32_t init_device(int32_t DeviceId);

  /// Return the number of devices this plugin can support.
  int32_t number_of_devices();

  /// Returns non-zero if the data can be exchanged between the two devices.
  int32_t is_data_exchangable(int32_t SrcDeviceId, int32_t DstDeviceId);

  /// Initializes the record and replay mechanism inside the plugin.
  int32_t initialize_record_replay(int32_t DeviceId, int64_t MemorySize,
                                   void *VAddr, bool isRecord, bool SaveOutput,
                                   uint64_t &ReqPtrArgOffset);

  /// Loads the associated binary into the plugin and returns a handle to it.
  int32_t load_binary(int32_t DeviceId, __tgt_device_image *TgtImage,
                      __tgt_device_binary *Binary);

  /// Allocates memory that is accessively to the given device.
  void *data_alloc(int32_t DeviceId, int64_t Size, void *HostPtr, int32_t Kind);

  /// Deallocates memory on the given device.
  int32_t data_delete(int32_t DeviceId, void *TgtPtr, int32_t Kind);

  /// Locks / pins host memory using the plugin runtime.
  int32_t data_lock(int32_t DeviceId, void *Ptr, int64_t Size,
                    void **LockedPtr);

  /// Unlocks / unpins host memory using the plugin runtime.
  int32_t data_unlock(int32_t DeviceId, void *Ptr);

  /// Notify the runtime about a new mapping that has been created outside.
  int32_t data_notify_mapped(int32_t DeviceId, void *HstPtr, int64_t Size);

  /// Notify t he runtime about a mapping that has been deleted.
  int32_t data_notify_unmapped(int32_t DeviceId, void *HstPtr);

  /// Copy data to the given device.
  int32_t data_submit(int32_t DeviceId, void *TgtPtr, void *HstPtr,
                      int64_t Size);

  /// Copy data to the given device asynchronously.
  int32_t data_submit_async(int32_t DeviceId, void *TgtPtr, void *HstPtr,
                            int64_t Size, __tgt_async_info *AsyncInfoPtr);

  /// Copy data from the given device.
  int32_t data_retrieve(int32_t DeviceId, void *HstPtr, void *TgtPtr,
                        int64_t Size);

  /// Copy data from the given device asynchronously.
  int32_t data_retrieve_async(int32_t DeviceId, void *HstPtr, void *TgtPtr,
                              int64_t Size, __tgt_async_info *AsyncInfoPtr);

  /// Exchange memory addresses between two devices.
  int32_t data_exchange(int32_t SrcDeviceId, void *SrcPtr, int32_t DstDeviceId,
                        void *DstPtr, int64_t Size);

  /// Exchange memory addresses between two devices asynchronously.
  int32_t data_exchange_async(int32_t SrcDeviceId, void *SrcPtr,
                              int DstDeviceId, void *DstPtr, int64_t Size,
                              __tgt_async_info *AsyncInfo);

  /// Places a fence between previous data movements and following data
  /// movements if necessary on the device
  int32_t data_fence(int32_t DeviceId, __tgt_async_info *AsyncInfo);

  /// Begin executing a kernel on the given device.
  int32_t launch_kernel(int32_t DeviceId, void *TgtEntryPtr, void **TgtArgs,
                        ptrdiff_t *TgtOffsets, KernelArgsTy *KernelArgs,
                        __tgt_async_info *AsyncInfoPtr);

  /// Synchronize an asyncrhonous queue with the plugin runtime.
  int32_t synchronize(int32_t DeviceId, __tgt_async_info *AsyncInfoPtr);

  /// Query the current state of an asynchronous queue.
  int32_t query_async(int32_t DeviceId, __tgt_async_info *AsyncInfoPtr);

  /// Prints information about the given devices supported by the plugin.
  void print_device_info(int32_t DeviceId);

  /// Creates an event in the given plugin if supported.
  int32_t create_event(int32_t DeviceId, void **EventPtr);

  /// Records an event that has occurred.
  int32_t record_event(int32_t DeviceId, void *EventPtr,
                       __tgt_async_info *AsyncInfoPtr);

  /// Wait until an event has occurred.
  int32_t wait_event(int32_t DeviceId, void *EventPtr,
                     __tgt_async_info *AsyncInfoPtr);

  /// Synchronize execution until an event is done.
  int32_t sync_event(int32_t DeviceId, void *EventPtr);

  /// Remove the event from the plugin.
  int32_t destroy_event(int32_t DeviceId, void *EventPtr);

  /// Remove the event from the plugin.
  void set_info_flag(uint32_t NewInfoLevel);

  /// Creates an asynchronous queue for the given plugin.
  int32_t init_async_info(int32_t DeviceId, __tgt_async_info **AsyncInfoPtr);

  /// Sets the offset into the devices for use by OMPT.
  int32_t set_device_identifier(int32_t UserId, int32_t DeviceId);

  /// Returns if the plugin can support automatic copy.
  int32_t use_auto_zero_copy(int32_t DeviceId);

  /// Look up a global symbol in the given binary.
  int32_t get_global(__tgt_device_binary Binary, uint64_t Size,
                     const char *Name, void **DevicePtr);

  /// Look up a kernel function in the given binary.
  int32_t get_function(__tgt_device_binary Binary, const char *Name,
                       void **KernelPtr);

  /// Return the interop specification that the plugin supports
  /// It might not be one of the user specified ones.
  interop_spec_t select_interop_preference(int32_t ID, int32_t InteropType,
                                           int32_t NumPrefers,
                                           interop_spec_t *Prefers) {
    auto &Device = getDevice(ID);
    return Device.selectInteropPreference(InteropType, NumPrefers, Prefers);
  }

  /// Create OpenMP interop with the given interop context
  omp_interop_val_t *create_interop(int32_t ID, int32_t InteropContext,
                                    interop_spec_t *InteropSpec);

  /// Release OpenMP interop object
  int32_t release_interop(int32_t ID, omp_interop_val_t *Interop);

  /// Flush the queue associated with the interop object if necessary
  int32_t flush_queue(omp_interop_val_t *Interop);

  /// Perform a host synchronization with the queue associated with the interop
  /// object and wait for it to complete.
  int32_t sync_barrier(omp_interop_val_t *Interop);

  /// Queue an asynchronous barrier in the queue associated with the interop
  /// object and return immediately.
  int32_t async_barrier(omp_interop_val_t *Interop);

private:
  /// Indicates if the platform runtime has been fully initialized.
  bool Initialized = false;

  /// Number of devices available for the plugin.
  int32_t NumDevices = 0;

  /// Map of plugin device identifiers to the user device identifier.
  llvm::DenseMap<int32_t, int32_t> UserDeviceIds;

  /// Array of pointers to the devices. Initially, they are all set to nullptr.
  /// Once a device is initialized, the pointer is stored in the position given
  /// by its device id. A position with nullptr means that the corresponding
  /// device was not initialized yet.
  llvm::SmallVector<GenericDeviceTy *> Devices;

  /// Pointer to the global handler for this plugin.
  GenericGlobalHandlerTy *GlobalHandler;

  /// Internal allocator for different structures.
  BumpPtrAllocator Allocator;

  /// The JIT engine shared by all devices connected to this plugin.
  JITEngine JIT;

  /// The interface between the plugin and the GPU for host services.
  RPCServerTy *RPCServer;

  /// The interface between the plugin and the GPU for host services.
  RecordReplayTy *RecordReplay;
};

/// Auxiliary interface class for GenericDeviceResourceManagerTy. This class
/// acts as a reference to a device resource, such as a stream, and requires
/// some basic functions to be implemented. The derived class should define an
/// empty constructor that creates an empty and invalid resource reference. Do
/// not create a new resource on the ctor, but on the create() function instead.
///
/// The derived class should also define the type HandleTy as the underlying
/// resource handle type. For instance, in a CUDA stream it would be:
///   using HandleTy = CUstream;
struct GenericDeviceResourceRef {
  /// Create a new resource and stores a reference.
  virtual Error create(GenericDeviceTy &Device) = 0;

  /// Destroy and release the resources pointed by the reference.
  virtual Error destroy(GenericDeviceTy &Device) = 0;

protected:
  ~GenericDeviceResourceRef() = default;
};

/// Class that implements a resource pool belonging to a device. This class
/// operates with references to the actual resources. These reference must
/// derive from the GenericDeviceResourceRef class and implement the create
/// and destroy virtual functions.
template <typename ResourceRef> class GenericDeviceResourceManagerTy {
  using ResourcePoolTy = GenericDeviceResourceManagerTy<ResourceRef>;
  using ResourceHandleTy = typename ResourceRef::HandleTy;

public:
  /// Create an empty resource pool for a specific device.
  GenericDeviceResourceManagerTy(GenericDeviceTy &Device)
      : Device(Device), NextAvailable(0) {}

  /// Destroy the resource pool. At this point, the deinit() function should
  /// already have been executed so the resource pool should be empty.
  virtual ~GenericDeviceResourceManagerTy() {
    assert(ResourcePool.empty() && "Resource pool not empty");
  }

  /// Initialize the resource pool.
  Error init(uint32_t InitialSize) {
    assert(ResourcePool.empty() && "Resource pool already initialized");
    return ResourcePoolTy::resizeResourcePool(InitialSize);
  }

  /// Deinitialize the resource pool and delete all resources. This function
  /// must be called before the destructor.
  virtual Error deinit() {
    if (NextAvailable)
      DP("Missing %d resources to be returned\n", NextAvailable);

    // TODO: This prevents a bug on libomptarget to make the plugins fail. There
    // may be some resources not returned. Do not destroy these ones.
    if (auto Err = ResourcePoolTy::resizeResourcePool(NextAvailable))
      return Err;

    ResourcePool.clear();

    return Plugin::success();
  }

  /// Get a resource from the pool or create new ones. If the function
  /// succeeds, the handle to the resource is saved in \p Handle.
  virtual Error getResource(ResourceHandleTy &Handle) {
    // Get a resource with an empty resource processor.
    return getResourcesImpl(1, &Handle,
                            [](ResourceHandleTy) { return Plugin::success(); });
  }

  /// Get multiple resources from the pool or create new ones. If the function
  /// succeeds, the handles to the resources are saved in \p Handles.
  virtual Error getResources(uint32_t Num, ResourceHandleTy *Handles) {
    // Get resources with an empty resource processor.
    return getResourcesImpl(Num, Handles,
                            [](ResourceHandleTy) { return Plugin::success(); });
  }

  /// Return resource to the pool.
  virtual Error returnResource(ResourceHandleTy Handle) {
    // Return a resource with an empty resource processor.
    return returnResourceImpl(
        Handle, [](ResourceHandleTy) { return Plugin::success(); });
  }

protected:
  /// Get multiple resources from the pool or create new ones. If the function
  /// succeeds, the handles to the resources are saved in \p Handles. Also
  /// process each of the obtained resources with \p Processor.
  template <typename FuncTy>
  Error getResourcesImpl(uint32_t Num, ResourceHandleTy *Handles,
                         FuncTy Processor) {
    const std::lock_guard<std::mutex> Lock(Mutex);

    assert(NextAvailable <= ResourcePool.size() &&
           "Resource pool is corrupted");

    if (NextAvailable + Num > ResourcePool.size())
      // Double the resource pool or resize it to provide the requested ones.
      if (auto Err = ResourcePoolTy::resizeResourcePool(
              std::max(NextAvailable * 2, NextAvailable + Num)))
        return Err;

    // Save the handles in the output array parameter.
    for (uint32_t r = 0; r < Num; ++r)
      Handles[r] = ResourcePool[NextAvailable + r];

    // Process all obtained resources.
    for (uint32_t r = 0; r < Num; ++r)
      if (auto Err = Processor(Handles[r]))
        return Err;

    NextAvailable += Num;

    return Plugin::success();
  }

  /// Return resource to the pool and process the resource with \p Processor.
  template <typename FuncTy>
  Error returnResourceImpl(ResourceHandleTy Handle, FuncTy Processor) {
    const std::lock_guard<std::mutex> Lock(Mutex);

    // Process the returned resource.
    if (auto Err = Processor(Handle))
      return Err;

    assert(NextAvailable > 0 && "Resource pool is corrupted");
    ResourcePool[--NextAvailable] = Handle;

    return Plugin::success();
  }

protected:
  /// The resources between \p OldSize and \p NewSize need to be created or
  /// destroyed. The mutex is locked when this function is called.
  Error resizeResourcePoolImpl(uint32_t OldSize, uint32_t NewSize) {
    assert(OldSize != NewSize && "Resizing to the same size");

    if (auto Err = Device.setContext())
      return Err;

    if (OldSize < NewSize) {
      // Create new resources.
      for (uint32_t I = OldSize; I < NewSize; ++I) {
        if (auto Err = ResourcePool[I].create(Device))
          return Err;
      }
    } else {
      // Destroy the obsolete resources.
      for (uint32_t I = NewSize; I < OldSize; ++I) {
        if (auto Err = ResourcePool[I].destroy(Device))
          return Err;
      }
    }
    return Plugin::success();
  }

  /// Increase or decrease the number of resources. This function should
  /// be called with the mutex acquired.
  Error resizeResourcePool(uint32_t NewSize) {
    uint32_t OldSize = ResourcePool.size();

    // Nothing to do.
    if (OldSize == NewSize)
      return Plugin::success();

    if (OldSize < NewSize) {
      // Increase the number of resources.
      ResourcePool.resize(NewSize);
      return ResourcePoolTy::resizeResourcePoolImpl(OldSize, NewSize);
    }

    // Decrease the number of resources otherwise.
    auto Err = ResourcePoolTy::resizeResourcePoolImpl(OldSize, NewSize);
    ResourcePool.resize(NewSize);

    return Err;
  }

  /// The device to which the resources belong
  GenericDeviceTy &Device;

  /// Mutex for the resource pool.
  std::mutex Mutex;

  /// The next available resource in the pool.
  uint32_t NextAvailable;

  /// The actual resource pool.
  std::deque<ResourceRef> ResourcePool;
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OPENMP_LIBOMPTARGET_PLUGINS_COMMON_PLUGININTERFACE_H
