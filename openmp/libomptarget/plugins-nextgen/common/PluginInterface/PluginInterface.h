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
#include <vector>

#include "Debug.h"
#include "DeviceEnvironment.h"
#include "GlobalHandler.h"
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
/// operations on object destruction.
struct AsyncInfoWrapperTy {
  AsyncInfoWrapperTy(Error &Err, GenericDeviceTy &Device,
                     __tgt_async_info *AsyncInfoPtr)
      : Err(Err), ErrOutParam(&Err), Device(Device),
        AsyncInfoPtr(AsyncInfoPtr ? AsyncInfoPtr : &LocalAsyncInfo) {}

  /// Synchronize with the __tgt_async_info's pending operations if it's the
  /// internal one.
  ~AsyncInfoWrapperTy();

  /// Get the raw __tgt_async_info pointer.
  operator __tgt_async_info *() const { return AsyncInfoPtr; }

  /// Get a reference to the underlying plugin-specific queue type.
  template <typename Ty> Ty &getQueueAs() const {
    static_assert(sizeof(Ty) == sizeof(AsyncInfoPtr->Queue),
                  "Queue is not of the same size as target type");
    return reinterpret_cast<Ty &>(AsyncInfoPtr->Queue);
  }

private:
  Error &Err;
  ErrorAsOutParameter ErrOutParam;
  GenericDeviceTy &Device;
  __tgt_async_info LocalAsyncInfo;
  __tgt_async_info *const AsyncInfoPtr;
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

  /// Table of offload entries.
  OffloadEntryTableTy OffloadEntryTable;

public:
  DeviceImageTy(int32_t Id, const __tgt_device_image *Image)
      : ImageId(Id), TgtImage(Image) {
    assert(TgtImage && "Invalid target image");
  }

  /// Get the image identifier within the device.
  int32_t getId() const { return ImageId; }

  /// Get the pointer to the raw __tgt_device_image.
  const __tgt_device_image *getTgtImage() const { return TgtImage; }

  /// Get the image starting address.
  void *getStart() const { return TgtImage->ImageStart; }

  /// Get the image size.
  size_t getSize() const {
    return ((char *)TgtImage->ImageEnd) - ((char *)TgtImage->ImageStart);
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
      : Name(Name), ExecutionMode(ExecutionMode), DynamicMemorySize(0),
        PreferredNumThreads(0), MaxNumThreads(0) {}

  virtual ~GenericKernelTy() {}

  /// Initialize the kernel object from a specific device.
  Error init(GenericDeviceTy &GenericDevice, DeviceImageTy &Image);
  virtual Error initImpl(GenericDeviceTy &GenericDevice,
                         DeviceImageTy &Image) = 0;

  /// Launch the kernel on the specific device. The device must be the same
  /// one used to initialize the kernel.
  Error launch(GenericDeviceTy &GenericDevice, void **ArgPtrs,
               ptrdiff_t *ArgOffsets, int32_t NumArgs, uint64_t NumTeamsClause,
               uint32_t ThreadLimitClause, uint64_t LoopTripCount,
               AsyncInfoWrapperTy &AsyncInfoWrapper) const;
  virtual Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads,
                           uint64_t NumBlocks, uint32_t DynamicMemorySize,
                           int32_t NumKernelArgs, void *KernelArgs,
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

private:
  /// Prepare the arguments before launching the kernel.
  void *prepareArgs(GenericDeviceTy &GenericDevice, void **ArgPtrs,
                    ptrdiff_t *ArgOffsets, int32_t NumArgs,
                    llvm::SmallVectorImpl<void *> &Args,
                    llvm::SmallVectorImpl<void *> &Ptrs,
                    AsyncInfoWrapperTy &AsyncInfoWrapper) const;

  /// Get the default number of threads and blocks for the kernel.
  virtual uint32_t getDefaultNumThreads(GenericDeviceTy &Device) const = 0;
  virtual uint64_t getDefaultNumBlocks(GenericDeviceTy &Device) const = 0;

  /// Get the number of threads and blocks for the kernel based on the
  /// user-defined threads and block clauses.
  uint32_t getNumThreads(GenericDeviceTy &GenericDevice,
                         uint32_t ThreadLimitClause) const;
  uint64_t getNumBlocks(GenericDeviceTy &GenericDevice,
                        uint64_t BlockLimitClause, uint64_t LoopTripCount,
                        uint32_t NumThreads) const;

  /// Indicate if the kernel works in Generic SPMD, Generic or SPMD mode.
  bool isGenericSPMDMode() const {
    return ExecutionMode == OMP_TGT_EXEC_MODE_GENERIC_SPMD;
  }
  bool isGenericMode() const {
    return ExecutionMode == OMP_TGT_EXEC_MODE_GENERIC;
  }
  bool isSPMDMode() const { return ExecutionMode == OMP_TGT_EXEC_MODE_SPMD; }

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

  /// The kernel name.
  const char *Name;

  /// The execution flags of the kernel.
  OMPTgtExecModeFlags ExecutionMode;

protected:
  /// The dynamic memory size reserved for executing the kernel.
  uint32_t DynamicMemorySize;

  /// The preferred number of threads to run the kernel.
  uint32_t PreferredNumThreads;

  /// The maximum number of threads which the kernel could leverage.
  uint32_t MaxNumThreads;
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

  /// Allocate data on the device or involving the device.
  Expected<void *> dataAlloc(int64_t Size, void *HostPtr, TargetAllocTy Kind);

  /// Deallocate data from the device or involving the device.
  Error dataDelete(void *TgtPtr, TargetAllocTy Kind);

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

  /// Run the target region with multiple teams.
  Error runTargetTeamRegion(void *EntryPtr, void **ArgPtrs,
                            ptrdiff_t *ArgOffsets, int32_t NumArgs,
                            uint64_t NumTeamsClause, uint32_t ThreadLimitClause,
                            uint64_t LoopTripCount,
                            __tgt_async_info *AsyncInfo);

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
  uint64_t getBlockLimit() const { return GridValues.GV_Max_Teams; }
  uint32_t getDefaultNumThreads() const {
    return GridValues.GV_Default_WG_Size;
  }
  uint64_t getDefaultNumBlocks() const {
    // TODO: Introduce a default num blocks value.
    return GridValues.GV_Default_WG_Size;
  }
  uint32_t getDynamicMemorySize() const { return OMPX_SharedMemorySize; }

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

  /// Environment variables defined by the OpenMP standard.
  Int32Envar OMP_TeamLimit;
  Int32Envar OMP_NumTeams;
  Int32Envar OMP_TeamsThreadLimit;

  /// Environment variables defined by the LLVM OpenMP implementation.
  Int32Envar OMPX_DebugKind;
  UInt32Envar OMPX_SharedMemorySize;
  UInt64Envar OMPX_TargetStackSize;
  UInt64Envar OMPX_TargetHeapSize;

  /// Pointer to the memory manager or nullptr if not available.
  MemoryManagerTy *MemoryManager;

protected:
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
};

/// Class implementing common functionalities of offload plugins. Each plugin
/// should define the specific plugin class, derive from this generic one, and
/// implement the necessary virtual function members.
struct GenericPluginTy {

  /// Construct a plugin instance. The number of active instances should be
  /// always be either zero or one.
  GenericPluginTy() : RequiresFlags(OMP_REQ_UNDEFINED), GlobalHandler(nullptr) {
    ++NumActiveInstances;
  }

  /// Destroy the plugin instance and release all its resources. Also decrease
  /// the number of instances.
  virtual ~GenericPluginTy() {
    // There is no global handler if no device is available.
    if (GlobalHandler)
      delete GlobalHandler;

    // Deinitialize all active devices.
    for (int32_t DeviceId = 0; DeviceId < NumDevices; ++DeviceId) {
      if (Devices[DeviceId]) {
        if (auto Err = deinitDevice(DeviceId))
          REPORT("Failure to deinitialize device %d: %s\n", DeviceId,
                 toString(std::move(Err)).data());
      }
      assert(!Devices[DeviceId] && "Device was not deinitialized");
    }

    --NumActiveInstances;
  }

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

  /// Allocate a structure using the internal allocator.
  template <typename Ty> Ty *allocate() {
    return reinterpret_cast<Ty *>(Allocator.Allocate(sizeof(Ty), alignof(Ty)));
  }

  /// Get the reference to the global handler of this plugin.
  GenericGlobalHandlerTy &getGlobalHandler() {
    assert(GlobalHandler && "Global handler not initialized");
    return *GlobalHandler;
  }

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

  /// Indicate whether there is any active plugin instance.
  static bool hasAnyActiveInstance() {
    assert(NumActiveInstances <= 1 && "Invalid number of instances");
    return (NumActiveInstances > 0);
  }

protected:
  /// Initialize the plugin and prepare for initializing its devices.
  void init(int NumDevices, GenericGlobalHandlerTy *GlobalHandler) {
    this->NumDevices = NumDevices;
    this->GlobalHandler = GlobalHandler;

    assert(Devices.size() == 0 && "Plugin already intialized");

    Devices.resize(NumDevices, nullptr);
  }

  /// Create a new device with a specific device id.
  virtual GenericDeviceTy &createDevice(int32_t DeviceId) = 0;

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

  /// Indicates the number of active plugin instances. Actually, we should only
  /// have one active instance per plugin library. But we use a counter for
  /// simplicity.
  static uint32_t NumActiveInstances;
};

/// Class for simplifying the getter operation of the plugin. Anywhere on the
/// code, the current plugin can be retrieved by Plugin::get(). The init(),
/// deinit(), get() and check() functions should be defined by each plugin
/// implementation.
class Plugin {
  /// Avoid instances of this class.
  Plugin() {}
  Plugin(const Plugin &) = delete;
  void operator=(const Plugin &) = delete;

public:
  /// Initialize the plugin if it was not initialized yet.
  static Error init();

  /// Deinitialize the plugin if it was not deinitialized yet.
  static Error deinit();

  /// Get a reference (or create if it was not created) to the plugin instance.
  static GenericPluginTy &get();

  /// Get a reference to the plugin with a specific plugin-specific type.
  template <typename Ty> static Ty &get() { return static_cast<Ty &>(get()); }

  /// Indicate if the plugin is currently active. Actually, we check if there is
  /// any active instances.
  static bool isActive() { return GenericPluginTy::hasAnyActiveInstance(); }

  /// Create a success error.
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
};

/// Auxiliary interface class for GenericDeviceResourcePoolTy. This class acts
/// as a reference to a device resource, such as a stream, and requires some
/// basic functions to be implemented. The derived class should define an empty
/// constructor that creates an empty and invalid resource reference. Do not
/// create a new resource on the ctor, but on the create() function instead.
struct GenericDeviceResourceRef {
  /// Create a new resource and stores a reference.
  virtual Error create() = 0;

  /// Destroy and release the resources pointed by the reference.
  virtual Error destroy() = 0;
};

/// Class that implements a resource pool belonging to a device. This class
/// operates with references to the actual resources. These reference must
/// derive from the GenericDeviceResourceRef class and implement the create
/// and destroy virtual functions.
template <typename ResourceRef> class GenericDeviceResourcePoolTy {
  using ResourcePoolTy = GenericDeviceResourcePoolTy<ResourceRef>;

public:
  /// Create an empty resource pool for a specific device.
  GenericDeviceResourcePoolTy(GenericDeviceTy &Device)
      : Device(Device), NextAvailable(0) {}

  /// Destroy the resource pool. At this point, the deinit() function should
  /// already have been executed so the resource pool should be empty.
  virtual ~GenericDeviceResourcePoolTy() {
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

protected:
  /// Get resource from the pool or create new resources.
  ResourceRef getResource() {
    const std::lock_guard<std::mutex> Lock(Mutex);
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
        if (auto Err = ResourcePool[I].create())
          return Err;
      }
    } else {
      // Destroy the obsolete resources.
      for (uint32_t I = NewSize; I < OldSize; ++I) {
        if (auto Err = ResourcePool[I].destroy())
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

    if (OldSize > NewSize) {
      // Decrease the number of resources.
      auto Err = ResourcePoolTy::resizeResourcePoolImpl(OldSize, NewSize);
      ResourcePool.resize(NewSize);
      return Err;
    }

    // Increase the number of resources otherwise.
    ResourcePool.resize(NewSize);
    return ResourcePoolTy::resizeResourcePoolImpl(OldSize, NewSize);
  }

  /// The device to which the resources belong
  GenericDeviceTy &Device;

  /// Mutex for the resource pool.
  std::mutex Mutex;

  /// The next available resource in the pool.
  uint32_t NextAvailable;

protected:
  /// The actual resource pool.
  std::deque<ResourceRef> ResourcePool;
};

/// Class implementing a common stream manager. This class can be directly used
/// by the specific plugins if necessary. The StreamRef type should derive from
/// the GenericDeviceResourceRef. Look at its description to know the details of
/// their requirements.
template <typename StreamRef>
class GenericStreamManagerTy : public GenericDeviceResourcePoolTy<StreamRef> {
  using ResourcePoolTy = GenericDeviceResourcePoolTy<StreamRef>;

public:
  /// Create a stream manager with space for an initial number of streams. No
  /// stream will be created until the init() function is called.
  GenericStreamManagerTy(GenericDeviceTy &Device, uint32_t DefNumStreams = 32)
      : ResourcePoolTy(Device),
        InitialNumStreams("LIBOMPTARGET_NUM_INITIAL_STREAMS", DefNumStreams) {}

  /// Initialize the stream pool and their resources with the initial number of
  /// streams.
  Error init() { return ResourcePoolTy::init(InitialNumStreams.get()); }

  /// Get an available stream or create new.
  StreamRef getStream() { return ResourcePoolTy::getResource(); }

  /// Return idle stream.
  void returnStream(StreamRef Stream) {
    ResourcePoolTy::returnResource(Stream);
  }

private:
  /// The initial stream pool size, potentially defined by an envar.
  UInt32Envar InitialNumStreams;
};

/// Class implementing a common event manager. This class can be directly used
/// by the specific plugins if necessary. The EventRef type should derive from
/// the GenericDeviceResourceRef. Look at its description to know the details of
/// their requirements.
template <typename EventRef>
struct GenericEventManagerTy : public GenericDeviceResourcePoolTy<EventRef> {
  using ResourcePoolTy = GenericDeviceResourcePoolTy<EventRef>;

public:
  /// Create an event manager with space for an initial number of events. No
  /// event will be created until the init() function is called.
  GenericEventManagerTy(GenericDeviceTy &Device, uint32_t DefNumEvents = 32)
      : ResourcePoolTy(Device),
        InitialNumEvents("LIBOMPTARGET_NUM_INITIAL_EVENTS", DefNumEvents) {}

  /// Initialize the event pool and their resources with the initial number of
  /// events.
  Error init() { return ResourcePoolTy::init(InitialNumEvents.get()); }

  /// Get an available event or create new.
  EventRef getEvent() { return ResourcePoolTy::getResource(); }

  /// Return an idle event.
  void returnEvent(EventRef Event) { ResourcePoolTy::returnResource(Event); }

private:
  /// The initial event pool size, potentially defined by an envar.
  UInt32Envar InitialNumEvents;
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

#endif // OPENMP_LIBOMPTARGET_PLUGINS_COMMON_PLUGININTERFACE_H
