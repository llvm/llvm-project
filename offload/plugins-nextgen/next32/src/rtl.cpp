//===-RTLs/next32/src/rtl.cpp - Target RTLs Implementation - C++ --------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL NextGen for Next32 machine
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstddef>
#include <string>
#include <unordered_map>

#include <sys/mman.h>

#include "Shared/Debug.h"
#include "Shared/Environment.h"

#include "GlobalHandler.h"
#include "PluginInterface.h"
#include "omptarget.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"

#include "nsapi/internal/function.h"

// The number of devices in this plugin.
#define NUM_DEVICES 1

// The ELF ID should be defined at compile-time by the build system.
#ifndef TARGET_ELF_ID
#define TARGET_ELF_ID 0
#endif

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// Forward declarations for all specialized data structures.
struct Next32KernelTy;
struct Next32DeviceTy;
struct Next32PluginTy;

/// Class implementing kernel functionalities for Next32.
struct Next32KernelTy : public GenericKernelTy {
  /// Construct the kernel with a name, execution mode and a function.
  Next32KernelTy(const char *Name)
      : GenericKernelTy(Name), FuncHandle(NSAPI_INVALID_FUNCTION_HANDLE) {}

  /// Initialize the kernel.
  Error initImpl(GenericDeviceTy &Device, DeviceImageTy &Image) override {
    GlobalTy Func(getName(), 0);

    // Get the metadata of the kernel function.
    GenericGlobalHandlerTy &GHandler = Device.Plugin.getGlobalHandler();
    if (auto Err = GHandler.getGlobalMetadataFromDevice(Device, Image, Func))
      return Err;

    FuncHandle = static_cast<llns_function_handle>(Func.getPtr());

    KernelEnvironment.Configuration.ExecMode = OMP_TGT_EXEC_MODE_GENERIC;
    KernelEnvironment.Configuration.MayUseNestedParallelism = /* Unknown */ 2;
    KernelEnvironment.Configuration.UseGenericStateMachine = /* Unknown */ 2;

    // Set the maximum number of threads to a single.
    MaxNumThreads = 1;
    return Plugin::success();
  }

  /// Launch the kernel using NSAPI.
  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads,
                   uint64_t NumBlocks, KernelArgsTy &KernelArgs,
                   KernelLaunchParamsTy LaunchParams,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override {
    bool Status =
        llns_execute_function(FuncHandle, reinterpret_cast<void **>(LaunchParams.Ptrs),
                              KernelArgs.NumArgs, nullptr, 0);

    if (!Status)
      return Plugin::error("Failed to execute function: %s", getName());

    return Plugin::success();
  }

private:
  /// The handle to the kernel function to execute.
  [[maybe_unused]] llns_function_handle FuncHandle;
};

/// Class implementing the Next32 device images properties.
struct Next32DeviceImageTy : public DeviceImageTy {
  /// Create the Next32 image with the id and the target image pointer.
  Next32DeviceImageTy(int32_t ImageId, GenericDeviceTy &Device,
                      const __tgt_device_image *TgtImage)
      : DeviceImageTy(ImageId, Device, TgtImage) {}
};

/// Page sizes available on the system.
static constexpr size_t PageSizes[] = {
    1LLU << 12, // 4KB
    1LLU << 14, // 16KB
    1LLU << 16, // 64KB
    1LLU << 18, // 256KB
    1LLU << 20, // 1MB
    1LLU << 22, // 4MB
    1LLU << 24, // 16MB
    1LLU << 26, // 64MB
    1LLU << 28, // 256MB
    1LLU << 30, // 1GB
    1LLU << 32, // 4GB
    1LLU << 34, // 16GB
    1LLU << 36, // 64GB
};

/// Round up the allocation size to the nearest page size.
static size_t roundUpToPageSize(size_t Size) {
  for (size_t PageSize : PageSizes) {
    if (Size <= PageSize)
      return PageSize;
  }
  return Size;
}

/// Class managing the memory allocations. We use mmap/munmap to
/// allocate/deallocate device memory, so we need to keep track of the size of
/// each allocation. We also only allocate memory in page-sized chunks, so we
/// need to round up allocation sizes to the nearest page size.
struct Next32AllocManager {
  /// Use mmap to allocate memory in page-sized chunks.
  void *allocate(size_t Size) {
    // Allocate memory using mmap.
    void *MemAlloc = mmap(nullptr, Size, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (MemAlloc == MAP_FAILED)
      return nullptr;

    // Keep track of the size of the allocation.
    AllocSizes[MemAlloc] = Size;

    return MemAlloc;
  }

  /// Free memory. Use munmap to deallocate memory.
  int free(void *TgtPtr) {
    // Find the size of the allocation.
    auto Iter = AllocSizes.find(TgtPtr);
    if (Iter == AllocSizes.end())
      return -1;

    // Deallocate memory using munmap.
    size_t Size = Iter->second;
    int Res = munmap(TgtPtr, Size);
    if (Res == EINVAL)
      return Res;

    // Remove the allocation from the map.
    AllocSizes.erase(Iter);

    return 0;
  }

private:
  /// Keep track of the size of each allocation.
  DenseMap<void *, size_t> AllocSizes;
};

/// Class implementing the device functionalities for Next32.
struct Next32DeviceTy : public GenericDeviceTy {
  /// Create the device with a specific id.
  Next32DeviceTy(GenericPluginTy &Plugin, int32_t DeviceId, int32_t NumDevices)
      : GenericDeviceTy(Plugin, DeviceId, NumDevices, Next32GridValues) {}

  /// Initialize the device, which is a no-op.
  Error initImpl(GenericPluginTy &Plugin) override { return Plugin::success(); }

  /// Deinitialize the device, which is a no-op.
  Error deinitImpl() override { return Plugin::success(); }

  /// Construct the kernel for a specific image on the device.
  Expected<GenericKernelTy &> constructKernel(const char *Name) override {
    // Allocate and create the kernel.
    Next32KernelTy *Next32Kernel = Plugin.allocate<Next32KernelTy>();
    if (!Next32Kernel)
      return Plugin::error("Failed to allocate memory for Next32 kernel");

    new (Next32Kernel) Next32KernelTy(Name);

    return *Next32Kernel;
  }

  /// Set the current context to this device, which is a no-op.
  Error setContext() override { return Plugin::success(); }

  /// Create a new DeviceImage object. The actual image is loaded externally.
  Expected<DeviceImageTy *> loadBinaryImpl(const __tgt_device_image *TgtImage,
                                           int32_t ImageId) override {
    // Allocate and initialize the image object.
    Next32DeviceImageTy *Image = Plugin.allocate<Next32DeviceImageTy>();
    new (Image) Next32DeviceImageTy(ImageId, *this, TgtImage);

    return Image;
  }

  /// We need to round up the allocation size before we get to the
  /// MemoryManager, since it also keeps track of the size of each allocation.
  Expected<void *> dataAlloc(int64_t Size, void *HostPtr,
                             TargetAllocTy Kind) override {
    // Round up the size to the nearest page size.
    size_t PageSize = roundUpToPageSize(Size);

    return GenericDeviceTy::dataAlloc(PageSize, HostPtr, Kind);
  }

  /// Allocate memory. Use std::malloc in all cases.
  void *allocate(size_t Size, void *, TargetAllocTy Kind) override {
    if (Size == 0)
      return nullptr;

    void *MemAlloc = nullptr;
    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
    case TARGET_ALLOC_DEVICE_NON_BLOCKING:
      MemAlloc = AllocManager.allocate(Size);
      break;
    case TARGET_ALLOC_HOST:
    case TARGET_ALLOC_SHARED:
      MemAlloc = std::malloc(Size);
      break;
    }
    return MemAlloc;
  }

  /// Free the memory. Use std::free in all cases.
  int free(void *TgtPtr, TargetAllocTy Kind) override {
    int Retval = 0;

    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
    case TARGET_ALLOC_DEVICE_NON_BLOCKING:
      Retval = AllocManager.free(TgtPtr);
      break;
    case TARGET_ALLOC_HOST:
    case TARGET_ALLOC_SHARED:
      std::free(TgtPtr);
      break;
    }

    return Retval ? OFFLOAD_FAIL : OFFLOAD_SUCCESS;
  }

  /// This plugin does nothing to lock buffers. Do not return an error, just
  /// return the same pointer as the device pointer.
  Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) override {
    return HstPtr;
  }

  /// Nothing to do when unlocking the buffer.
  Error dataUnlockImpl(void *HstPtr) override { return Plugin::success(); }

  /// Indicate that the buffer is not pinned.
  Expected<bool> isPinnedPtrImpl(void *HstPtr, void *&BaseHstPtr,
                                 void *&BaseDevAccessiblePtr,
                                 size_t &BaseSize) const override {
    return false;
  }

  /// Submit data to the device (host to device transfer).
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    /// TODO: Eventually use NSAPI memory migration interface.
    std::memcpy(TgtPtr, HstPtr, Size);
    return Plugin::success();
  }

  /// Retrieve data from the device (device to host transfer).
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    /// TODO: Eventually use NSAPI memory migration interface.
    std::memcpy(HstPtr, TgtPtr, Size);
    return Plugin::success();
  }

  /// Exchange data between two devices within the plugin. This function is not
  /// supported in this plugin.
  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstGenericDevice,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // This function should never be called because the function
    // Next32PluginTy::isDataExchangable() returns false.
    return Plugin::error("dataExchangeImpl not supported");
  }

  /// All functions are already synchronous. No need to do anything on this
  /// synchronization function.
  Error synchronizeImpl(__tgt_async_info &AsyncInfo) override {
    return Plugin::success();
  }

  /// All functions are already synchronous. No need to do anything on this
  /// query function.
  Error queryAsyncImpl(__tgt_async_info &AsyncInfo) override {
    return Plugin::success();
  }

  /// This plugin does not support interoperability
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::error("initAsyncInfoImpl not supported");
  }

  /// This plugin does not support interoperability
  Error initDeviceInfoImpl(__tgt_device_info *DeviceInfo) override {
    return Plugin::error("initDeviceInfoImpl not supported");
  }

  /// This plugin does not support the event API. Do nothing without failing.
  Error createEventImpl(void **EventPtrStorage) override {
    *EventPtrStorage = nullptr;
    return Plugin::success();
  }
  Error destroyEventImpl(void *EventPtr) override { return Plugin::success(); }
  Error recordEventImpl(void *EventPtr,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::success();
  }
  Error waitEventImpl(void *EventPtr,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::success();
  }
  Error syncEventImpl(void *EventPtr) override { return Plugin::success(); }

  /// Print information about the device.
  Error obtainInfoImpl(InfoQueueTy &Info) override {
    Info.add("Device Type", "NextSilicon");
    return Plugin::success();
  }

  /// This plugin should not setup the device environment.
  bool shouldSetupDeviceEnvironment() const override { return false; };

  /// Getters and setters for stack size and heap size not relevant.
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

  /// Next32 plugin uses an external compilation process which expects the
  /// image to remain in bitcode format.
  Expected<bool> shouldUseBitcodeImage() const override { return true; }

private:
  /// Generic grid values (unused by this plugin).
  static constexpr GV Next32GridValues = {
      1, // GV_Slot_Size
      1, // GV_Warp_Size
      1, // GV_Max_Teams
      1, // GV_SimpleBufferSize
      1, // GV_Max_WG_Size
      1, // GV_Default_WG_Size
  };

  /// Allocation manager for this device.
  Next32AllocManager AllocManager;
};

class Next32GlobalHandlerTy final : public GenericGlobalHandlerTy {
public:
  Error getGlobalMetadataFromDevice(GenericDeviceTy &GenericDevice,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
    // Get the function handle from the NSAPI.
    llns_function_handle FuncHandle =
        llns_get_function_handle_by_name(DeviceGlobal.getName().c_str());

    // This is currently considered an error, but might not necessarily be once
    // we integrate the OMP target flow with the default flow in nextutils.
    if (FuncHandle == NSAPI_INVALID_FUNCTION_HANDLE)
      return Plugin::error("Unable to find function handle for function: %s",
                           DeviceGlobal.getName().c_str());

    // Save the handle.
    DeviceGlobal.setPtr(const_cast<void *>(FuncHandle));

    return Plugin::success();
  }
};

/// Class implementing the plugin functionalities for Next32.
struct Next32PluginTy final : public GenericPluginTy {
  /// Create the Next32 plugin.
  Next32PluginTy() : GenericPluginTy(getTripleArch()) {}

  /// This class should not be copied.
  Next32PluginTy(const Next32PluginTy &) = delete;
  Next32PluginTy(Next32PluginTy &&) = delete;

  /// Initialize the plugin and return the number of devices.
  Expected<int32_t> initImpl() override { return NUM_DEVICES; }

  /// Deinitialize the plugin.
  Error deinitImpl() override { return Plugin::success(); }

  /// Creates a Next32 device.
  GenericDeviceTy *createDevice(GenericPluginTy &Plugin, int32_t DeviceId,
                                int32_t NumDevices) override {
    return new Next32DeviceTy(Plugin, DeviceId, NumDevices);
  }

  /// Creates a global handler.
  GenericGlobalHandlerTy *createGlobalHandler() override {
    return new Next32GlobalHandlerTy();
  }

  /// Get the ELF code to recognize the compatible binary images.
  uint16_t getMagicElfBits() const override { return ELF::TARGET_ELF_ID; }

  /// This plugin does not support exchanging data between two devices.
  bool isDataExchangable(int32_t SrcDeviceId, int32_t DstDeviceId) override {
    return false;
  }

  /// All images (ELF-compatible) should be compatible with this plugin.
  Expected<bool> isELFCompatible(uint32_t DeviceId,
                                 StringRef Image) const override {
    return true;
  }

  Triple::ArchType getTripleArch() const override {
    return llvm::Triple(LIBOMPTARGET_NEXTGEN_GENERIC_PLUGIN_TRIPLE).getArch();
  }

  const char *getName() const override { return GETNAME(TARGET_NAME); }
};

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

extern "C" {
llvm::omp::target::plugin::GenericPluginTy *createPlugin_next32() {
  return new llvm::omp::target::plugin::Next32PluginTy();
}
}