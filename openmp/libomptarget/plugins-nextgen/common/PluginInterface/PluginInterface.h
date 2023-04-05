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
#include <list>
#include <map>
#include <shared_mutex>
#include <vector>

#include "Debug.h"
#include "DeviceEnvironment.h"
#include "GlobalHandler.h"
#include "JIT.h"
#include "MemoryManager.h"
#include "Utilities.h"
#include "omptarget.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
namespace omp {
namespace target {

namespace plugin {

struct GenericPluginTy;
struct GenericKernelTy;
struct GenericDeviceTy;

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

  /// Get a reference to the underlying plugin-specific queue type.
  template <typename Ty> Ty &getQueueAs() const {
    static_assert(sizeof(Ty) == sizeof(AsyncInfoPtr->Queue),
                  "Queue is not of the same size as target type");
    return reinterpret_cast<Ty &>(AsyncInfoPtr->Queue);
  }

  /// Indicate whether there is queue.
  bool hasQueue() const { return (AsyncInfoPtr->Queue != nullptr); }

  // Get a reference to the error associated with the asycnhronous operations
  // related to the async info wrapper.
  Error &getError() { return Err; }

  /// Synchronize with the __tgt_async_info's pending operations if it's the
  /// internal async info and return the error associated with the async
  /// operations. This function must be called before destroying the object.
  Error finalize();

private:
  Error Err;
  GenericDeviceTy &Device;
  __tgt_async_info LocalAsyncInfo;
  __tgt_async_info *AsyncInfoPtr;
};

/// Class wrapping a __tgt_device_image and its offload entry table on a
/// specific device. This class is responsible for storing and managing
/// the offload entries for an image on a device.
class DeviceImageTy {

  /// Class representing the offload entry table. The class stores the
  /// __tgt_target_table and a map to search in the table faster.
  struct OffloadEntryTableTy {
    /// Add new entry to the table.
    void addEntry(const __tgt_offload_entry &Entry) {
      Entries.push_back(Entry);
      TTTablePtr.EntriesBegin = &Entries[0];
      TTTablePtr.EntriesEnd = TTTablePtr.EntriesBegin + Entries.size();
    }

    /// Get the raw pointer to the __tgt_target_table.
    operator __tgt_target_table *() {
      if (Entries.empty())
        return nullptr;
      return &TTTablePtr;
    }

  private:
    __tgt_target_table TTTablePtr;
    llvm::SmallVector<__tgt_offload_entry> Entries;
  };

  /// Image identifier within the corresponding device. Notice that this id is
  /// not unique between different device; they may overlap.
  int32_t ImageId;

  /// The pointer to the raw __tgt_device_image.
  const __tgt_device_image *TgtImage;
  const __tgt_device_image *TgtImageBitcode;

  /// Table of offload entries.
  OffloadEntryTableTy OffloadEntryTable;

public:
  DeviceImageTy(int32_t Id, const __tgt_device_image *Image)
      : ImageId(Id), TgtImage(Image), TgtImageBitcode(nullptr) {
    assert(TgtImage && "Invalid target image");
  }

  /// Get the image identifier within the device.
  int32_t getId() const { return ImageId; }

  /// Get the pointer to the raw __tgt_device_image.
  const __tgt_device_image *getTgtImage() const { return TgtImage; }

  void setTgtImageBitcode(const __tgt_device_image *TgtImageBitcode) {
    this->TgtImageBitcode = TgtImageBitcode;
  }

  const __tgt_device_image *getTgtImageBitcode() const {
    return TgtImageBitcode;
  }

  /// Get the image starting address.
  void *getStart() const { return TgtImage->ImageStart; }

  /// Get the image size.
  size_t getSize() const {
    return getPtrDiff(TgtImage->ImageEnd, TgtImage->ImageStart);
  }

  /// Get a memory buffer reference to the whole image.
  MemoryBufferRef getMemoryBuffer() const {
    return MemoryBufferRef(StringRef((const char *)getStart(), getSize()),
                           "Image");
  }

  /// Get a reference to the offload entry table for the image.
  OffloadEntryTableTy &getOffloadEntryTable() { return OffloadEntryTable; }
};

/// Class implementing common functionalities of offload kernels. Each plugin
/// should define the specific kernel class, derive from this generic one, and
/// implement the necessary virtual function members.
struct GenericKernelTy {
  /// Construct a kernel with a name and a execution mode.
  GenericKernelTy(const char *Name, OMPTgtExecModeFlags ExecutionMode)
      : Name(Name), ExecutionMode(ExecutionMode),
        PreferredNumThreads(0), MaxNumThreads(0) {}

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
  virtual Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads,
                           uint64_t NumBlocks,
                           KernelArgsTy &KernelArgs, void *Args,
                           AsyncInfoWrapperTy &AsyncInfoWrapper) const = 0;

  /// Get the kernel name.
  const char *getName() const { return Name; }

  /// Indicate whether an execution mode is valid.
  static bool isValidExecutionMode(OMPTgtExecModeFlags ExecutionMode) {
    switch (ExecutionMode) {
    case OMP_TGT_EXEC_MODE_SPMD:
    case OMP_TGT_EXEC_MODE_GENERIC:
    case OMP_TGT_EXEC_MODE_GENERIC_SPMD:
      return true;
    }
    return false;
  }

protected:
  /// Get the execution mode name of the kernel.
  const char *getExecutionModeName() const {
    switch (ExecutionMode) {
    case OMP_TGT_EXEC_MODE_SPMD:
      return "SPMD";
    case OMP_TGT_EXEC_MODE_GENERIC:
      return "Generic";
    case OMP_TGT_EXEC_MODE_GENERIC_SPMD:
      return "Generic-SPMD";
    }
    llvm_unreachable("Unknown execution mode!");
  }

  /// Prints generic kernel launch information.
  Error printLaunchInfo(GenericDeviceTy &GenericDevice,
                        KernelArgsTy &KernelArgs, uint32_t NumThreads,
                        uint64_t NumBlocks) const;

  /// Prints plugin-specific kernel launch information after generic kernel
  /// launch information
  virtual Error printLaunchInfoDetails(GenericDeviceTy &GenericDevice,
                                       KernelArgsTy &KernelArgs,
                                       uint32_t NumThreads,
                                       uint64_t NumBlocks) const;

private:
  /// Prepare the arguments before launching the kernel.
  void *prepareArgs(GenericDeviceTy &GenericDevice, void **ArgPtrs,
                    ptrdiff_t *ArgOffsets, int32_t NumArgs,
                    llvm::SmallVectorImpl<void *> &Args,
                    llvm::SmallVectorImpl<void *> &Ptrs) const;

  /// Get the default number of threads and blocks for the kernel.
  virtual uint32_t getDefaultNumThreads(GenericDeviceTy &Device) const = 0;
  virtual uint32_t getDefaultNumBlocks(GenericDeviceTy &Device) const = 0;

  /// Get the number of threads and blocks for the kernel based on the
  /// user-defined threads and block clauses.
  uint32_t getNumThreads(GenericDeviceTy &GenericDevice,
                         uint32_t ThreadLimitClause[3]) const;
  uint64_t getNumBlocks(GenericDeviceTy &GenericDevice,
                        uint32_t BlockLimitClause[3], uint64_t LoopTripCount,
                        uint32_t NumThreads) const;

  /// Indicate if the kernel works in Generic SPMD, Generic or SPMD mode.
  bool isGenericSPMDMode() const {
    return ExecutionMode == OMP_TGT_EXEC_MODE_GENERIC_SPMD;
  }
  bool isGenericMode() const {
    return ExecutionMode == OMP_TGT_EXEC_MODE_GENERIC;
  }
  bool isSPMDMode() const { return ExecutionMode == OMP_TGT_EXEC_MODE_SPMD; }

  /// The kernel name.
  const char *Name;

  /// The execution flags of the kernel.
  OMPTgtExecModeFlags ExecutionMode;

protected:
  /// The preferred number of threads to run the kernel.
  uint32_t PreferredNumThreads;

  /// The maximum number of threads which the kernel could leverage.
  uint32_t MaxNumThreads;
};

/// Class representing a map of host pinned allocations. We track these pinned
/// allocations, so memory tranfers invloving these buffers can be optimized.
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

    /// Create an entry with the host and device acessible pointers, the buffer
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

  /// Indicate whether failures when locking mapped buffers should be ingored.
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
    if (advanceVoidPtr(It->HstPtr, It->Size) > HstPtr)
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
    void *EndA = advanceVoidPtr(PtrA, SizeA);
    void *EndB = advanceVoidPtr(PtrB, SizeB);
    return (PtrB >= PtrA && EndB <= EndA);
  }

  /// Indicate whether the first range A intersects with the second range B.
  static bool intersects(void *PtrA, size_t SizeA, void *PtrB, size_t SizeB) {
    void *EndA = advanceVoidPtr(PtrA, SizeA);
    void *EndB = advanceVoidPtr(PtrB, SizeB);
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

    return advanceVoidPtr(Entry->DevAccessiblePtr,
                          getPtrDiff(HstPtr, Entry->HstPtr));
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
  GenericDeviceTy(int32_t DeviceId, int32_t NumDevices,
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
  Error deinit();
  virtual Error deinitImpl() = 0;

  /// Load the binary image into the device and return the target table.
  Expected<__tgt_target_table *> loadBinary(GenericPluginTy &Plugin,
                                            const __tgt_device_image *TgtImage);
  virtual Expected<DeviceImageTy *>
  loadBinaryImpl(const __tgt_device_image *TgtImage, int32_t ImageId) = 0;

  /// Setup the device environment if needed. Notice this setup may not be run
  /// on some plugins. By default, it will be executed, but plugins can change
  /// this behavior by overriding the shouldSetupDeviceEnvironment function.
  Error setupDeviceEnvironment(GenericPluginTy &Plugin, DeviceImageTy &Image);

  /// Register the offload entries for a specific image on the device.
  Error registerOffloadEntries(DeviceImageTy &Image);

  /// Synchronize the current thread with the pending operations on the
  /// __tgt_async_info structure.
  Error synchronize(__tgt_async_info *AsyncInfo);
  virtual Error synchronizeImpl(__tgt_async_info &AsyncInfo) = 0;

  /// Query for the completion of the pending operations on the __tgt_async_info
  /// structure in a non-blocking manner.
  Error queryAsync(__tgt_async_info *AsyncInfo);
  virtual Error queryAsyncImpl(__tgt_async_info &AsyncInfo) = 0;

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

  /// Exchange data between devices (device to device transfer). Calling this
  /// function is only valid if GenericPlugin::isDataExchangable() passing the
  /// two devices returns true.
  Error dataExchange(const void *SrcPtr, GenericDeviceTy &DstDev, void *DstPtr,
                     int64_t Size, __tgt_async_info *AsyncInfo);
  virtual Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstDev,
                                 void *DstPtr, int64_t Size,
                                 AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  /// Run the kernel associated with \p EntryPtr
  Error launchKernel(void *EntryPtr, void **ArgPtrs, ptrdiff_t *ArgOffsets,
                     KernelArgsTy &KernelArgs, __tgt_async_info *AsyncInfo);

  /// Initialize a __tgt_async_info structure. Related to interop features.
  Error initAsyncInfo(__tgt_async_info **AsyncInfoPtr);
  virtual Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) = 0;

  /// Initialize a __tgt_device_info structure. Related to interop features.
  Error initDeviceInfo(__tgt_device_info *DeviceInfo);
  virtual Error initDeviceInfoImpl(__tgt_device_info *DeviceInfo) = 0;

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

  /// Synchronize the current thread with the event.
  Error syncEvent(void *EventPtr);
  virtual Error syncEventImpl(void *EventPtr) = 0;

  /// Print information about the device.
  Error printInfo();
  virtual Error printInfoImpl() = 0;

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
  uint32_t getDynamicMemorySize() const { return OMPX_SharedMemorySize; }

  /// Get target compute unit kind (e.g., sm_80, or gfx908).
  virtual std::string getComputeUnitKind() const { return "unknown"; }

  /// Post processing after jit backend. The ownership of \p MB will be taken.
  virtual Expected<std::unique_ptr<MemoryBuffer>>
  doJITPostProcessing(std::unique_ptr<MemoryBuffer> MB) const {
    return std::move(MB);
  }

private:
  /// Register offload entry for global variable.
  Error registerGlobalOffloadEntry(DeviceImageTy &DeviceImage,
                                   const __tgt_offload_entry &GlobalEntry,
                                   __tgt_offload_entry &DeviceEntry);

  /// Register offload entry for kernel function.
  Error registerKernelOffloadEntry(DeviceImageTy &DeviceImage,
                                   const __tgt_offload_entry &KernelEntry,
                                   __tgt_offload_entry &DeviceEntry);

  /// Allocate and construct a kernel object.
  virtual Expected<GenericKernelTy *>
  constructKernelEntry(const __tgt_offload_entry &KernelEntry,
                       DeviceImageTy &Image) = 0;

  /// Get and set the stack size and heap size for the device. If not used, the
  /// plugin can implement the setters as no-op and setting the output
  /// value to zero for the getters.
  virtual Error getDeviceStackSize(uint64_t &V) = 0;
  virtual Error setDeviceStackSize(uint64_t V) = 0;
  virtual Error getDeviceHeapSize(uint64_t &V) = 0;
  virtual Error setDeviceHeapSize(uint64_t V) = 0;

  /// Indicate whether the device should setup the device environment. Notice
  /// that returning false in this function will change the behavior of the
  /// setupDeviceEnvironment() function.
  virtual bool shouldSetupDeviceEnvironment() const { return true; }

  /// Pointer to the memory manager or nullptr if not available.
  MemoryManagerTy *MemoryManager;

  /// Environment variables defined by the OpenMP standard.
  Int32Envar OMP_TeamLimit;
  Int32Envar OMP_NumTeams;
  Int32Envar OMP_TeamsThreadLimit;

  /// Environment variables defined by the LLVM OpenMP implementation.
  Int32Envar OMPX_DebugKind;
  UInt32Envar OMPX_SharedMemorySize;
  UInt64Envar OMPX_TargetStackSize;
  UInt64Envar OMPX_TargetHeapSize;

protected:
  /// Return the execution mode used for kernel \p Name.
  Expected<OMPTgtExecModeFlags> getExecutionModeForKernel(StringRef Name,
                                                          DeviceImageTy &Image);

  /// Environment variables defined by the LLVM OpenMP implementation
  /// regarding the initial number of streams and events.
  UInt32Envar OMPX_InitialNumStreams;
  UInt32Envar OMPX_InitialNumEvents;

  /// Array of images loaded into the device. Images are automatically
  /// deallocated by the allocator.
  llvm::SmallVector<DeviceImageTy *> LoadedImages;

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
  /// the device I has a matrix PeerAccesses with PeerAccesses[J] == AVAILABLE,
  /// the device I can access device J's memory directly. However, notice this
  /// does not mean that device J can access device I's memory directly.
  llvm::SmallVector<PeerAccessState> PeerAccesses;
  std::mutex PeerAccessesLock;

  /// Map of host pinned allocations used for optimize device transfers.
  PinnedAllocationMapTy PinnedAllocs;
};

/// Class implementing common functionalities of offload plugins. Each plugin
/// should define the specific plugin class, derive from this generic one, and
/// implement the necessary virtual function members.
struct GenericPluginTy {

  /// Construct a plugin instance.
  GenericPluginTy(Triple::ArchType TA)
      : RequiresFlags(OMP_REQ_UNDEFINED), GlobalHandler(nullptr), JIT(TA) {}

  virtual ~GenericPluginTy() {}

  /// Initialize the plugin.
  Error init();

  /// Initialize the plugin and return the number of available devices.
  virtual Expected<int32_t> initImpl() = 0;

  /// Deinitialize the plugin and release the resources.
  Error deinit();
  virtual Error deinitImpl() = 0;

  /// Get the reference to the device with a certain device id.
  GenericDeviceTy &getDevice(int32_t DeviceId) {
    assert(isValidDeviceId(DeviceId) && "Invalid device id");
    assert(Devices[DeviceId] && "Device is unitialized");

    return *Devices[DeviceId];
  }

  /// Get the number of active devices.
  int32_t getNumDevices() const { return NumDevices; }

  /// Get the ELF code to recognize the binary image of this plugin.
  virtual uint16_t getMagicElfBits() const = 0;

  /// Get the target triple of this plugin.
  virtual Triple::ArchType getTripleArch() const = 0;

  /// Allocate a structure using the internal allocator.
  template <typename Ty> Ty *allocate() {
    return reinterpret_cast<Ty *>(Allocator.Allocate(sizeof(Ty), alignof(Ty)));
  }

  /// Get the reference to the global handler of this plugin.
  GenericGlobalHandlerTy &getGlobalHandler() {
    assert(GlobalHandler && "Global handler not initialized");
    return *GlobalHandler;
  }

  /// Get the reference to the JIT used for all devices connected to this
  /// plugin.
  JITEngine &getJIT() { return JIT; }

  /// Get the OpenMP requires flags set for this plugin.
  int64_t getRequiresFlags() const { return RequiresFlags; }

  /// Set the OpenMP requires flags for this plugin.
  void setRequiresFlag(int64_t Flags) { RequiresFlags = Flags; }

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

  /// Indicate if an image is compatible with the plugin devices. Notice that
  /// this function may be called before actually initializing the devices. So
  /// we could not move this function into GenericDeviceTy.
  virtual Expected<bool> isImageCompatible(__tgt_image_info *Info) const = 0;

  /// Indicate whether the plugin supports empty images.
  virtual bool supportsEmptyImages() const { return false; }

protected:
  /// Indicate whether a device id is valid.
  bool isValidDeviceId(int32_t DeviceId) const {
    return (DeviceId >= 0 && DeviceId < getNumDevices());
  }

private:
  /// Number of devices available for the plugin.
  int32_t NumDevices;

  /// Array of pointers to the devices. Initially, they are all set to nullptr.
  /// Once a device is initialized, the pointer is stored in the position given
  /// by its device id. A position with nullptr means that the corresponding
  /// device was not initialized yet.
  llvm::SmallVector<GenericDeviceTy *> Devices;

  /// OpenMP requires flags.
  int64_t RequiresFlags;

  /// Pointer to the global handler for this plugin.
  GenericGlobalHandlerTy *GlobalHandler;

  /// Internal allocator for different structures.
  BumpPtrAllocator Allocator;

  /// The JIT engine shared by all devices connected to this plugin.
  JITEngine JIT;
};

/// Class for simplifying the getter operation of the plugin. Anywhere on the
/// code, the current plugin can be retrieved by Plugin::get(). The class also
/// declares functions to create plugin-specific object instances. The check(),
/// createPlugin(), createDevice() and createGlobalHandler() functions should be
/// defined by each plugin implementation.
class Plugin {
  // Reference to the plugin instance.
  static GenericPluginTy *SpecificPlugin;

  Plugin() {
    if (auto Err = init())
      REPORT("Failed to initialize plugin: %s\n",
             toString(std::move(Err)).data());
  }

  ~Plugin() {
    if (auto Err = deinit())
      REPORT("Failed to deinitialize plugin: %s\n",
             toString(std::move(Err)).data());
  }

  Plugin(const Plugin &) = delete;
  void operator=(const Plugin &) = delete;

  /// Create and intialize the plugin instance.
  static Error init() {
    assert(!SpecificPlugin && "Plugin already created");

    // Create the specific plugin.
    SpecificPlugin = createPlugin();
    assert(SpecificPlugin && "Plugin was not created");

    // Initialize the plugin.
    return SpecificPlugin->init();
  }

  // Deinitialize and destroy the plugin instance.
  static Error deinit() {
    assert(SpecificPlugin && "Plugin no longer valid");

    // Deinitialize the plugin.
    if (auto Err = SpecificPlugin->deinit())
      return Err;

    // Delete the plugin instance.
    delete SpecificPlugin;

    // Invalidate the plugin reference.
    SpecificPlugin = nullptr;

    return Plugin::success();
  }

public:
  /// Initialize the plugin if needed. The plugin could have been initialized by
  /// a previous call to Plugin::get().
  static Error initIfNeeded() {
    // Trigger the initialization if needed.
    get();

    return Error::success();
  }

  // Deinitialize the plugin if needed. The plugin could have been deinitialized
  // because the plugin library was exiting.
  static Error deinitIfNeeded() {
    // Do nothing. The plugin is deinitialized automatically.
    return Plugin::success();
  }

  /// Get a reference (or create if it was not created) to the plugin instance.
  static GenericPluginTy &get() {
    // This static variable will initialize the underlying plugin instance in
    // case there was no previous explicit initialization. The initialization is
    // thread safe.
    static Plugin Plugin;

    assert(SpecificPlugin && "Plugin is not active");
    return *SpecificPlugin;
  }

  /// Get a reference to the plugin with a specific plugin-specific type.
  template <typename Ty> static Ty &get() { return static_cast<Ty &>(get()); }

  /// Indicate whether the plugin is active.
  static bool isActive() { return SpecificPlugin != nullptr; }

  /// Create a success error. This is the same as calling Error::success(), but
  /// it is recommended to use this one for consistency with Plugin::error() and
  /// Plugin::check().
  static Error success() { return Error::success(); }

  /// Create a string error.
  template <typename... ArgsTy>
  static Error error(const char *ErrFmt, ArgsTy... Args) {
    return createStringError(inconvertibleErrorCode(), ErrFmt, Args...);
  }

  /// Check the plugin-specific error code and return an error or success
  /// accordingly. In case of an error, create a string error with the error
  /// description. The ErrFmt should follow the format:
  ///     "Error in <function name>[<optional info>]: %s"
  /// The last format specifier "%s" is mandatory and will be used to place the
  /// error code's description. Notice this function should be only called from
  /// the plugin-specific code.
  template <typename... ArgsTy>
  static Error check(int32_t ErrorCode, const char *ErrFmt, ArgsTy... Args);

  /// Create a plugin instance.
  static GenericPluginTy *createPlugin();

  /// Create a plugin-specific device.
  static GenericDeviceTy *createDevice(int32_t DeviceId, int32_t NumDevices);

  /// Create a plugin-specific global handler.
  static GenericGlobalHandlerTy *createGlobalHandler();
};

/// Auxiliary interface class for GenericDeviceResourceManagerTy. This class
/// acts as a reference to a device resource, such as a stream, and requires
/// some basic functions to be implemented. The derived class should define an
/// empty constructor that creates an empty and invalid resource reference. Do
/// not create a new resource on the ctor, but on the create() function instead.
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
  Error deinit() {
    if (NextAvailable)
      DP("Missing %d resources to be returned\n", NextAvailable);

    // TODO: This prevents a bug on libomptarget to make the plugins fail. There
    // may be some resources not returned. Do not destroy these ones.
    if (auto Err = ResourcePoolTy::resizeResourcePool(NextAvailable))
      return Err;

    ResourcePool.clear();

    return Plugin::success();
  }

  /// Get resource from the pool or create new resources.
  ResourceRef getResource() {
    const std::lock_guard<std::mutex> Lock(Mutex);

    assert(NextAvailable <= ResourcePool.size() &&
           "Resource pool is corrupted");

    if (NextAvailable == ResourcePool.size()) {
      // By default we double the resource pool every time.
      if (auto Err = ResourcePoolTy::resizeResourcePool(NextAvailable * 2)) {
        REPORT("Failure to resize the resource pool: %s",
               toString(std::move(Err)).data());
        // Return an empty reference.
        return ResourceRef();
      }
    }
    return ResourcePool[NextAvailable++];
  }

  /// Return resource to the pool.
  void returnResource(ResourceRef Resource) {
    const std::lock_guard<std::mutex> Lock(Mutex);

    assert(NextAvailable > 0 && "Resource pool is corrupted");
    ResourcePool[--NextAvailable] = Resource;
  }

private:
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
