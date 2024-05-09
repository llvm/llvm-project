//===------RTLs/mpi/src/rtl.cpp - Target RTLs Implementation - C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL NextGen for MPI applications
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <optional>
#include <string>

#include "Shared/Debug.h"
#include "Utils/ELF.h"

#include "EventSystem.h"
#include "GlobalHandler.h"
#include "OpenMP/OMPT/Callback.h"
#include "PluginInterface.h"
#include "omptarget.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Frontend/OpenMP/OMPDeviceConstants.h"
#include "llvm/Support/Error.h"

#if !defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__) ||           \
    !defined(__ORDER_BIG_ENDIAN__)
#error "Missing preprocessor definitions for endianness detection."
#endif

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define LITTLEENDIAN_CPU
#elif defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define BIGENDIAN_CPU
#endif

namespace llvm::omp::target::plugin {

/// Forward declarations for all specialized data structures.
struct MPIPluginTy;
struct MPIDeviceTy;
struct MPIDeviceImageTy;
struct MPIKernelTy;
class MPIGlobalHandlerTy;

// TODO: Should this be defined inside the EventSystem?
using MPIEventQueue = std::list<EventTy>;
using MPIEventQueuePtr = MPIEventQueue *;

/// Class implementing the MPI device images properties.
struct MPIDeviceImageTy : public DeviceImageTy {
  /// Create the MPI image with the id and the target image pointer.
  MPIDeviceImageTy(int32_t ImageId, GenericDeviceTy &Device,
                   const __tgt_device_image *TgtImage)
      : DeviceImageTy(ImageId, Device, TgtImage), DeviceImageAddrs(getSize()) {}

  llvm::SmallVector<void *> DeviceImageAddrs;
};

class MPIGlobalHandlerTy final : public GenericGlobalHandlerTy {
public:
  Error getGlobalMetadataFromDevice(GenericDeviceTy &GenericDevice,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
    const char *GlobalName = DeviceGlobal.getName().data();
    MPIDeviceImageTy &MPIImage = static_cast<MPIDeviceImageTy &>(Image);

    if (GlobalName == nullptr) {
      return Plugin::error("Failed to get name for global %p", &DeviceGlobal);
    }

    void *EntryAddress = nullptr;

    __tgt_offload_entry *Begin = MPIImage.getTgtImage()->EntriesBegin;
    __tgt_offload_entry *End = MPIImage.getTgtImage()->EntriesEnd;

    int I = 0;
    for (auto &Entry = Begin; Entry < End; ++Entry) {
      if (!strcmp(Entry->name, GlobalName)) {
        EntryAddress = MPIImage.DeviceImageAddrs[I];
        break;
      }
      I++;
    }

    if (EntryAddress == nullptr) {
      return Plugin::error("Failed to find global %s", GlobalName);
    }

    // Save the pointer to the symbol.
    DeviceGlobal.setPtr(EntryAddress);

    return Plugin::success();
  }
};

struct MPIKernelTy : public GenericKernelTy {
  /// Construct the kernel with a name and an execution mode.
  MPIKernelTy(const char *Name, EventSystemTy &EventSystem)
      : GenericKernelTy(Name), Func(nullptr), EventSystem(EventSystem) {}

  /// Initialize the kernel.
  Error initImpl(GenericDeviceTy &Device, DeviceImageTy &Image) override {
    // Functions have zero size.
    GlobalTy Global(getName(), 0);

    // Get the metadata (address) of the kernel function.
    GenericGlobalHandlerTy &GHandler = Device.Plugin.getGlobalHandler();
    if (auto Err = GHandler.getGlobalMetadataFromDevice(Device, Image, Global))
      return Err;

    // Check that the function pointer is valid.
    if (!Global.getPtr())
      return Plugin::error("Invalid function for kernel %s", getName());

    // Save the function pointer.
    Func = (void (*)())Global.getPtr();

    // TODO: Check which settings are appropriate for the mpi plugin
    // for now we are using the Elf64 plugin configuration
    KernelEnvironment.Configuration.ExecMode = OMP_TGT_EXEC_MODE_GENERIC;
    KernelEnvironment.Configuration.MayUseNestedParallelism = /* Unknown */ 2;
    KernelEnvironment.Configuration.UseGenericStateMachine = /* Unknown */ 2;

    // Set the maximum number of threads to a single.
    MaxNumThreads = 1;
    return Plugin::success();
  }

  /// Launch the kernel.
  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads,
                   uint64_t NumBlocks, KernelArgsTy &KernelArgs, void *Args,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override;

private:
  /// The kernel function to execute.
  void (*Func)(void);
  EventSystemTy &EventSystem;
};

/// MPI resource reference and queue. These are the objects handled by the
/// MPIQueue Manager for the MPI plugin.
template <typename ResourceTy>
struct MPIResourceRef final : public GenericDeviceResourceRef {

  /// The underlying handler type for the resource.
  using HandleTy = ResourceTy *;

  /// Create a empty reference to an invalid resource.
  MPIResourceRef() : Resource(nullptr) {}

  /// Create a reference to an existing resource.
  MPIResourceRef(HandleTy Queue) : Resource(Queue) {}

  /// Create a new resource and save the reference.
  Error create(GenericDeviceTy &Device) override {
    if (Resource)
      return Plugin::error("Recreating an existing resource");

    Resource = new ResourceTy;
    if (!Resource)
      return Plugin::error("Failed to allocated a new resource");

    return Plugin::success();
  }

  /// Destroy the resource and invalidate the reference.
  Error destroy(GenericDeviceTy &Device) override {
    if (!Resource)
      return Plugin::error("Destroying an invalid resource");

    delete Resource;
    Resource = nullptr;

    return Plugin::success();
  }

  operator HandleTy() const { return Resource; }

private:
  HandleTy Resource;
};

/// Class implementing the device functionalities for remote x86_64 processes.
struct MPIDeviceTy : public GenericDeviceTy {
  /// Create a MPI Device with a device id and the default MPI grid values.
  MPIDeviceTy(GenericPluginTy &Plugin, int32_t DeviceId, int32_t NumDevices,
              EventSystemTy &EventSystem)
      : GenericDeviceTy(Plugin, DeviceId, NumDevices, MPIGridValues),
        MPIEventQueueManager(*this), MPIEventManager(*this),
        EventSystem(EventSystem) {}

  /// Initialize the device, its resources and get its properties.
  Error initImpl(GenericPluginTy &Plugin) override {
    if (auto Err = MPIEventQueueManager.init(OMPX_InitialNumStreams))
      return Err;

    if (auto Err = MPIEventManager.init(OMPX_InitialNumEvents))
      return Err;

    return Plugin::success();
  }

  /// Deinitizalize the device and release its resources.
  Error deinitImpl() override {
    if (auto Err = MPIEventQueueManager.deinit())
      return Err;

    if (auto Err = MPIEventManager.deinit())
      return Err;

    return Plugin::success();
  }

  Error setContext() override { return Plugin::success(); }

  /// Load the binary image into the device and allocate an image object.
  Expected<DeviceImageTy *> loadBinaryImpl(const __tgt_device_image *TgtImage,
                                           int32_t ImageId) override {

    // Allocate and initialize the image object.
    MPIDeviceImageTy *Image = Plugin.allocate<MPIDeviceImageTy>();
    new (Image) MPIDeviceImageTy(ImageId, *this, TgtImage);

    auto Event = EventSystem.createEvent(OriginEvents::loadBinary, DeviceId,
                                         TgtImage, &(Image->DeviceImageAddrs));

    if (Event.empty()) {
      return Plugin::error("Failed to create loadBinary event for image %p",
                           TgtImage);
    }

    Event.wait();

    if (auto Error = Event.getError(); Error) {
      return Plugin::error("Event failed during loadBinary. %s\n",
                           toString(std::move(Error)).c_str());
    }

    return Image;
  }

  /// Allocate memory on the device or related to the device.
  void *allocate(size_t Size, void *, TargetAllocTy Kind) override {
    if (Size == 0)
      return nullptr;

    void *BufferAddress = nullptr;
    std::optional<Error> Err = std::nullopt;
    EventTy Event{nullptr};

    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
    case TARGET_ALLOC_DEVICE_NON_BLOCKING:
      Event = EventSystem.createEvent(OriginEvents::allocateBuffer, DeviceId,
                                      Size, &BufferAddress);

      if (Event.empty()) {
        Err = Plugin::error("Failed to create alloc event with size %z", Size);
        break;
      }

      Event.wait();
      Err = Event.getError();
      break;
    case TARGET_ALLOC_HOST:
      BufferAddress = memAllocHost(Size);
      Err = Plugin::check(BufferAddress == nullptr,
                          "Failed to allocate host memory");
      break;
    case TARGET_ALLOC_SHARED:
      Err = Plugin::error("Incompatible memory type %d", Kind);
      break;
    }

    if (*Err) {
      REPORT("Failed to allocate memory: %s\n",
             toString(std::move(*Err)).c_str());
      return nullptr;
    }

    return BufferAddress;
  }

  /// Deallocate memory on the device or related to the device.
  int free(void *TgtPtr, TargetAllocTy Kind) override {
    if (TgtPtr == nullptr)
      return OFFLOAD_SUCCESS;

    std::optional<Error> Err = std::nullopt;
    EventTy Event{nullptr};

    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
    case TARGET_ALLOC_DEVICE_NON_BLOCKING:
      Event =
          EventSystem.createEvent(OriginEvents::deleteBuffer, DeviceId, TgtPtr);

      if (Event.empty()) {
        Err = Plugin::error("Failed to create delete event");
        break;
      }

      Event.wait();
      Err = Event.getError();
      break;
    case TARGET_ALLOC_HOST:
      Err = Plugin::check(memFreeHost(TgtPtr), "Failed to free host memory");
      break;
    case TARGET_ALLOC_SHARED:
      Err = createStringError(inconvertibleErrorCode(),
                              "Incompatible memory type %d", Kind);
      break;
    }

    if (*Err) {
      REPORT("Failed to free memory: %s\n", toString(std::move(*Err)).c_str());
      return OFFLOAD_FAIL;
    }

    return OFFLOAD_SUCCESS;
  }

  /// Submit data to the device (host to device transfer).
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    MPIEventQueuePtr Queue = nullptr;
    if (auto Err = getQueue(AsyncInfoWrapper, Queue))
      return Err;

    // Copy HstData to a buffer with event-managed lifetime.
    void *SubmitBuffer = std::malloc(Size);
    std::memcpy(SubmitBuffer, HstPtr, Size);
    EventDataHandleTy DataHandle(SubmitBuffer, &std::free);

    auto Event = EventSystem.createEvent(OriginEvents::submit, DeviceId,
                                         DataHandle, TgtPtr, Size);

    if (Event.empty())
      return Plugin::error("Failed to create submit event");

    Queue->push_back(Event);

    return Plugin::success();
  }

  /// Retrieve data from the device (device to host transfer).
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    MPIEventQueuePtr Queue = nullptr;
    if (auto Err = getQueue(AsyncInfoWrapper, Queue))
      return Err;

    auto Event = EventSystem.createEvent(OriginEvents::retrieve, DeviceId,
                                         HstPtr, TgtPtr, Size);

    if (Event.empty())
      return Plugin::error("Failed to create retrieve event");

    Queue->push_back(Event);

    return Plugin::success();
  }

  /// Exchange data between two devices directly. In the MPI plugin, this
  /// function will create an event for the host to tell the devices about the
  /// exchange. Then, the devices will do the transfer themselves and let the
  /// host know when it's done.
  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstDev,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    MPIEventQueuePtr Queue = nullptr;
    if (auto Err = getQueue(AsyncInfoWrapper, Queue))
      return Err;

    auto Event = EventSystem.createExchangeEvent(
        DeviceId, SrcPtr, DstDev.getDeviceId(), DstPtr, Size);

    if (Event.empty())
      return Plugin::error("Failed to create exchange event");

    Queue->push_back(Event);

    return Plugin::success();
  }

  /// Allocate and construct a MPI kernel.
  Expected<GenericKernelTy &> constructKernel(const char *Name) override {
    // Allocate and construct the kernel.
    MPIKernelTy *MPIKernel = Plugin.allocate<MPIKernelTy>();

    if (!MPIKernel)
      return Plugin::error("Failed to allocate memory for MPI kernel");

    new (MPIKernel) MPIKernelTy(Name, EventSystem);

    return *MPIKernel;
  }

  /// Create an event.
  Error createEventImpl(void **EventStoragePtr) override {
    if (!EventStoragePtr)
      return Plugin::error("Received invalid event storage pointer");

    EventTy **NewEvent = reinterpret_cast<EventTy **>(EventStoragePtr);
    auto Err = MPIEventManager.getResource(*NewEvent);
    if (Err)
      return Plugin::error("Could not allocate a new synchronization event");

    return Plugin::success();
  }

  /// Destroy a previously created event.
  Error destroyEventImpl(void *Event) override {
    if (!Event)
      return Plugin::error("Received invalid event pointer");

    return MPIEventManager.returnResource(reinterpret_cast<EventTy *>(Event));
  }

  /// Record the event.
  Error recordEventImpl(void *Event,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    if (!Event)
      return Plugin::error("Received invalid event pointer");

    MPIEventQueuePtr Queue = nullptr;
    if (auto Err = getQueue(AsyncInfoWrapper, Queue))
      return Err;

    if (Queue->empty())
      return Plugin::success();

    auto &RecordedEvent = *reinterpret_cast<EventTy *>(Event);
    RecordedEvent = Queue->back();

    return Plugin::success();
  }

  /// Make the queue wait on the event.
  Error waitEventImpl(void *Event,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    if (!Event)
      return Plugin::error("Received invalid event pointer");

    auto &RecordedEvent = *reinterpret_cast<EventTy *>(Event);
    auto SyncEvent = OriginEvents::sync(RecordedEvent);

    MPIEventQueuePtr Queue = nullptr;
    if (auto Err = getQueue(AsyncInfoWrapper, Queue))
      return Err;

    Queue->push_back(SyncEvent);

    return Plugin::success();
  }

  /// Synchronize the current thread with the event
  Error syncEventImpl(void *Event) override {
    if (!Event)
      return Plugin::error("Received invalid event pointer");

    auto &RecordedEvent = *reinterpret_cast<EventTy *>(Event);
    auto SyncEvent = OriginEvents::sync(RecordedEvent);

    SyncEvent.wait();

    return SyncEvent.getError();
  }

  /// Synchronize current thread with the pending operations on the async info.
  Error synchronizeImpl(__tgt_async_info &AsyncInfo) override {
    auto *Queue = reinterpret_cast<MPIEventQueue *>(AsyncInfo.Queue);

    for (auto &Event : *Queue) {
      Event.wait();

      if (auto Error = Event.getError(); Error)
        return Plugin::error("Event failed during synchronization. %s\n",
                             toString(std::move(Error)).c_str());
    }

    // Once the queue is synchronized, return it to the pool and reset the
    // AsyncInfo. This is to make sure that the synchronization only works
    // for its own tasks.
    AsyncInfo.Queue = nullptr;
    return MPIEventQueueManager.returnResource(Queue);
  }

  /// Query for the completion of the pending operations on the async info.
  Error queryAsyncImpl(__tgt_async_info &AsyncInfo) override {
    auto *Queue = reinterpret_cast<MPIEventQueue *>(AsyncInfo.Queue);

    // Returns success when there are pending operations in AsyncInfo, moving
    // forward through the events on the queue until it is fully completed.
    while (!Queue->empty()) {
      auto &Event = Queue->front();

      Event.resume();

      if (!Event.done())
        return Plugin::success();

      if (auto Error = Event.getError(); Error)
        return Plugin::error("Event failed during query. %s\n",
                             toString(std::move(Error)).c_str());

      Queue->pop_front();
    }

    // Once the queue is synchronized, return it to the pool and reset the
    // AsyncInfo. This is to make sure that the synchronization only works
    // for its own tasks.
    AsyncInfo.Queue = nullptr;
    return MPIEventQueueManager.returnResource(Queue);
  }

  Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) override {
    return HstPtr;
  }

  /// Indicate that the buffer is not pinned.
  Expected<bool> isPinnedPtrImpl(void *HstPtr, void *&BaseHstPtr,
                                 void *&BaseDevAccessiblePtr,
                                 size_t &BaseSize) const override {
    return false;
  }

  Error dataUnlockImpl(void *HstPtr) override { return Plugin::success(); }

  /// This plugin should not setup the device environment or memory pool.
  virtual bool shouldSetupDeviceEnvironment() const override { return false; };
  virtual bool shouldSetupDeviceMemoryPool() const override { return false; };

  /// Device memory limits are currently not applicable to the MPI plugin.
  Error getDeviceStackSize(uint64_t &Value) override {
    Value = 0;
    return Plugin::success();
  }

  Error setDeviceStackSize(uint64_t Value) override {
    return Plugin::success();
  }

  Error getDeviceHeapSize(uint64_t &Value) override {
    Value = 0;
    return Plugin::success();
  }

  Error setDeviceHeapSize(uint64_t Value) override { return Plugin::success(); }

  /// Device interoperability. Not supported by MPI right now.
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::error("initAsyncInfoImpl not supported");
  }

  /// This plugin does not support interoperability.
  Error initDeviceInfoImpl(__tgt_device_info *DeviceInfo) override {
    return Plugin::error("initDeviceInfoImpl not supported");
  }

  /// Print information about the device.
  Error obtainInfoImpl(InfoQueueTy &Info) override {
    // TODO: Add more information about the device.
    Info.add("MPI plugin");
    Info.add("MPI OpenMP Device Number", DeviceId);

    return Plugin::success();
  }

  Error getQueue(AsyncInfoWrapperTy &AsyncInfoWrapper,
                 MPIEventQueuePtr &Queue) {
    Queue = AsyncInfoWrapper.getQueueAs<MPIEventQueuePtr>();
    if (!Queue) {
      // There was no queue; get a new one.
      if (auto Err = MPIEventQueueManager.getResource(Queue))
        return Err;

      // Modify the AsyncInfoWrapper to hold the new queue.
      AsyncInfoWrapper.setQueueAs<MPIEventQueuePtr>(Queue);
    }
    return Plugin::success();
  }

private:
  using MPIEventQueueManagerTy =
      GenericDeviceResourceManagerTy<MPIResourceRef<MPIEventQueue>>;
  using MPIEventManagerTy =
      GenericDeviceResourceManagerTy<MPIResourceRef<EventTy>>;

  MPIEventQueueManagerTy MPIEventQueueManager;
  MPIEventManagerTy MPIEventManager;
  EventSystemTy &EventSystem;

  /// Grid values for the MPI plugin.
  static constexpr GV MPIGridValues = {
      1, // GV_Slot_Size
      1, // GV_Warp_Size
      1, // GV_Max_Teams
      1, // GV_Default_Num_Teams
      1, // GV_SimpleBufferSize
      1, // GV_Max_WG_Size
      1, // GV_Default_WG_Size
  };
};

Error MPIKernelTy::launchImpl(GenericDeviceTy &GenericDevice,
                              uint32_t NumThreads, uint64_t NumBlocks,
                              KernelArgsTy &KernelArgs, void *Args,
                              AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  MPIDeviceTy &MPIDevice = static_cast<MPIDeviceTy &>(GenericDevice);
  MPIEventQueuePtr Queue = nullptr;
  if (auto Err = MPIDevice.getQueue(AsyncInfoWrapper, Queue))
    return Err;

  uint32_t NumArgs = KernelArgs.NumArgs;

  // Copy explicit Args to a buffer with event-managed lifetime.
  // This is necessary because host addresses are not accessible on the MPI
  // device and the Args buffer lifetime is not compatible with the lifetime of
  // the Execute Event
  void *TgtArgs = std::malloc(sizeof(void *) * NumArgs);
  std::memcpy(TgtArgs, *static_cast<void **>(Args), sizeof(void *) * NumArgs);
  EventDataHandleTy DataHandle(TgtArgs, &std::free);

  auto Event = EventSystem.createEvent(OriginEvents::execute,
                                       GenericDevice.getDeviceId(), DataHandle,
                                       NumArgs, (void *)Func);
  if (Event.empty())
    return Plugin::error("Failed to create execute event");

  Queue->push_back(Event);

  return Plugin::success();
}

/// Class implementing the MPI plugin.
struct MPIPluginTy : GenericPluginTy {
  MPIPluginTy() : GenericPluginTy(getTripleArch()) {}

  /// This class should not be copied.
  MPIPluginTy(const MPIPluginTy &) = delete;
  MPIPluginTy(MPIPluginTy &&) = delete;

  /// Initialize the plugin and return the number of devices.
  Expected<int32_t> initImpl() override {
    EventSystem.initialize();
    return EventSystem.getNumWorkers();
  }

  /// Deinitialize the plugin.
  Error deinitImpl() override {
    EventSystem.deinitialize();
    return Plugin::success();
  }

  /// Creates a MPI device.
  GenericDeviceTy *createDevice(GenericPluginTy &Plugin, int32_t DeviceId,
                                int32_t NumDevices) override {
    return new MPIDeviceTy(Plugin, DeviceId, NumDevices, EventSystem);
  }

  /// Creates a MPI global handler.
  GenericGlobalHandlerTy *createGlobalHandler() override {
    return new MPIGlobalHandlerTy();
  }

  /// Get the ELF code to recognize the compatible binary images.
  uint16_t getMagicElfBits() const override {
    return utils::elf::getTargetMachine();
  }

  /// All images (ELF-compatible) should be compatible with this plugin.
  Expected<bool> isELFCompatible(StringRef) const override { return true; }

  Triple::ArchType getTripleArch() const override {
#if defined(__x86_64__)
    return llvm::Triple::x86_64;
#elif defined(__s390x__)
    return llvm::Triple::systemz;
#elif defined(__aarch64__)
#ifdef LITTLEENDIAN_CPU
    return llvm::Triple::aarch64;
#else
    return llvm::Triple::aarch64_be;
#endif
#elif defined(__powerpc64__)
#ifdef LITTLEENDIAN_CPU
    return llvm::Triple::ppc64le;
#else
    return llvm::Triple::ppc64;
#endif
#else
    return llvm::Triple::UnknownArch;
#endif
  }

  const char *getName() const override { return GETNAME(TARGET_NAME); }

private:
  EventSystemTy EventSystem;
};

template <typename... ArgsTy>
static Error Plugin::check(int32_t ErrorCode, const char *ErrFmt,
                           ArgsTy... Args) {
  if (ErrorCode == OFFLOAD_SUCCESS)
    return Error::success();

  return createStringError<ArgsTy..., const char *>(
      inconvertibleErrorCode(), ErrFmt, Args...,
      std::to_string(ErrorCode).data());
}

} // namespace llvm::omp::target::plugin

extern "C" {
llvm::omp::target::plugin::GenericPluginTy *createPlugin_mpi() {
  return new llvm::omp::target::plugin::MPIPluginTy();
}
}