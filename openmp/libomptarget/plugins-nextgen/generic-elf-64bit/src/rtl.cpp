//===-RTLs/generic-64bit/src/rtl.cpp - Target RTLs Implementation - C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL NextGen for generic 64-bit machine
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstddef>
#include <ffi.h>
#include <string>
#include <unordered_map>

#include "Debug.h"
#include "DeviceEnvironment.h"
#include "GlobalHandler.h"
#include "PluginInterface.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/DynamicLibrary.h"

// The number of devices in this plugin.
#define NUM_DEVICES 4

// The ELF ID should be defined at compile-time by the build system.
#ifndef TARGET_ELF_ID
#define TARGET_ELF_ID 0
#endif

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// Forward declarations for all specialized data structures.
struct GenELF64KernelTy;
struct GenELF64DeviceTy;
struct GenELF64PluginTy;

using llvm::sys::DynamicLibrary;

/// Class implementing kernel functionalities for GenELF64.
struct GenELF64KernelTy : public GenericKernelTy {
  /// Construct the kernel with a name, execution mode and a function.
  GenELF64KernelTy(const char *Name, OMPTgtExecModeFlags ExecutionMode,
                   void (*Func)(void))
      : GenericKernelTy(Name, ExecutionMode), Func(Func) {}

  /// Initialize the kernel.
  Error initImpl(GenericDeviceTy &GenericDevice,
                 DeviceImageTy &Image) override {
    // Set the maximum number of threads to a single.
    MaxNumThreads = 1;
    return Plugin::success();
  }

  /// Launch the kernel using the libffi.
  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads,
                   uint64_t NumBlocks, uint32_t DynamicMemorySize,
                   int32_t NumKernelArgs, void *KernelArgs,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override {
    // Create a vector of ffi_types, one per argument.
    SmallVector<ffi_type *, 16> ArgTypes(NumKernelArgs, &ffi_type_pointer);
    ffi_type **ArgTypesPtr = (ArgTypes.size()) ? &ArgTypes[0] : nullptr;

    // Prepare the cif structure before running the kernel function.
    ffi_cif Cif;
    ffi_status Status = ffi_prep_cif(&Cif, FFI_DEFAULT_ABI, NumKernelArgs,
                                     &ffi_type_void, ArgTypesPtr);
    if (Status != FFI_OK)
      return Plugin::error("Error in ffi_prep_cif: %d", Status);

    // Call the kernel function through libffi.
    long Return;
    ffi_call(&Cif, Func, &Return, (void **)KernelArgs);

    return Plugin::success();
  }

  /// Get the default number of blocks and threads for the kernel.
  uint64_t getDefaultNumBlocks(GenericDeviceTy &) const override { return 1; }
  uint32_t getDefaultNumThreads(GenericDeviceTy &) const override { return 1; }

private:
  /// The kernel function to execute.
  void (*Func)(void);
};

/// Class implementing the GenELF64 device images properties.
struct GenELF64DeviceImageTy : public DeviceImageTy {
  /// Create the GenELF64 image with the id and the target image pointer.
  GenELF64DeviceImageTy(int32_t ImageId, const __tgt_device_image *TgtImage)
      : DeviceImageTy(ImageId, TgtImage), DynLib() {}

  /// Getter and setter for the dynamic library.
  DynamicLibrary &getDynamicLibrary() { return DynLib; }
  void setDynamicLibrary(const DynamicLibrary &Lib) { DynLib = Lib; }

private:
  /// The dynamic library that loaded the image.
  DynamicLibrary DynLib;
};

/// Class implementing the device functionalities for GenELF64.
struct GenELF64DeviceTy : public GenericDeviceTy {
  /// Create the device with a specific id.
  GenELF64DeviceTy(int32_t DeviceId, int32_t NumDevices)
      : GenericDeviceTy(DeviceId, NumDevices, GenELF64GridValues) {}

  ~GenELF64DeviceTy() {}

  /// Initialize the device, which is a no-op
  Error initImpl(GenericPluginTy &Plugin) override { return Plugin::success(); }

  /// Deinitialize the device, which is a no-op
  Error deinitImpl() override { return Plugin::success(); }

  /// Construct the kernel for a specific image on the device.
  Expected<GenericKernelTy *>
  constructKernelEntry(const __tgt_offload_entry &KernelEntry,
                       DeviceImageTy &Image) override {
    GlobalTy Func(KernelEntry);

    // Get the metadata (address) of the kernel function.
    GenericGlobalHandlerTy &GHandler = Plugin::get().getGlobalHandler();
    if (auto Err = GHandler.getGlobalMetadataFromDevice(*this, Image, Func))
      return std::move(Err);

    // Allocate and create the kernel.
    GenELF64KernelTy *GenELF64Kernel =
        Plugin::get().allocate<GenELF64KernelTy>();
    new (GenELF64Kernel) GenELF64KernelTy(
        KernelEntry.name, OMP_TGT_EXEC_MODE_GENERIC, (void (*)())Func.getPtr());

    return GenELF64Kernel;
  }

  /// Set the current context to this device, which is a no-op.
  Error setContext() override { return Plugin::success(); }

  /// Load the binary image into the device and allocate an image object.
  Expected<DeviceImageTy *> loadBinaryImpl(const __tgt_device_image *TgtImage,
                                           int32_t ImageId) override {
    // Allocate and initialize the image object.
    GenELF64DeviceImageTy *Image =
        Plugin::get().allocate<GenELF64DeviceImageTy>();
    new (Image) GenELF64DeviceImageTy(ImageId, TgtImage);

    // Create a temporary file.
    char TmpFileName[] = "/tmp/tmpfile_XXXXXX";
    int TmpFileFd = mkstemp(TmpFileName);
    if (TmpFileFd == -1)
      return Plugin::error("Failed to create tmpfile for loading target image");

    // Open the temporary file.
    FILE *TmpFile = fdopen(TmpFileFd, "wb");
    if (!TmpFile)
      return Plugin::error("Failed to open tmpfile %s for loading target image",
                           TmpFileName);

    // Write the image into the temporary file.
    size_t Written = fwrite(Image->getStart(), Image->getSize(), 1, TmpFile);
    if (Written != 1)
      return Plugin::error("Failed to write target image to tmpfile %s",
                           TmpFileName);

    // Close the temporary file.
    int Ret = fclose(TmpFile);
    if (Ret)
      return Plugin::error("Failed to close tmpfile %s with the target image",
                           TmpFileName);

    // Load the temporary file as a dynamic library.
    std::string ErrMsg;
    DynamicLibrary DynLib =
        DynamicLibrary::getPermanentLibrary(TmpFileName, &ErrMsg);

    // Check if the loaded library is valid.
    if (!DynLib.isValid())
      return Plugin::error("Failed to load target image: %s", ErrMsg.c_str());

    // Save a reference of the image's dynamic library.
    Image->setDynamicLibrary(DynLib);

    return Image;
  }

  /// Allocate memory. Use std::malloc in all cases.
  void *allocate(size_t Size, void *, TargetAllocTy Kind) override {
    if (Size == 0)
      return nullptr;

    void *MemAlloc = nullptr;
    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
    case TARGET_ALLOC_HOST:
    case TARGET_ALLOC_SHARED:
      MemAlloc = std::malloc(Size);
      break;
    }
    return MemAlloc;
  }

  /// Free the memory. Use std::free in all cases.
  int free(void *TgtPtr, TargetAllocTy Kind) override {
    std::free(TgtPtr);
    return OFFLOAD_SUCCESS;
  }

  /// Submit data to the device (host to device transfer).
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    std::memcpy(TgtPtr, HstPtr, Size);
    return Plugin::success();
  }

  /// Retrieve data from the device (device to host transfer).
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    std::memcpy(HstPtr, TgtPtr, Size);
    return Plugin::success();
  }

  /// Exchange data between two devices within the plugin. This function is not
  /// supported in this plugin.
  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstGenericDevice,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // This function should never be called because the function
    // GenELF64PluginTy::isDataExchangable() returns false.
    return Plugin::error("dataExchangeImpl not supported");
  }

  /// All functions are already synchronous. No need to do anything on this
  /// synchronization function.
  Error synchronizeImpl(__tgt_async_info &AsyncInfo) override {
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
  Error printInfoImpl() override {
    printf("    This is a generic-elf-64bit device\n");
    return Plugin::success();
  }

  /// This plugin should not setup the device environment.
  virtual bool shouldSetupDeviceEnvironment() const override { return false; };

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

private:
  /// Grid values for Generic ELF64 plugins.
  static constexpr GV GenELF64GridValues = {
      1, // GV_Slot_Size
      1, // GV_Warp_Size
      1, // GV_Max_Teams
      1, // GV_SimpleBufferSize
      1, // GV_Max_WG_Size
      1, // GV_Default_WG_Size
  };
};

class GenELF64GlobalHandlerTy final : public GenericGlobalHandlerTy {
public:
  Error getGlobalMetadataFromDevice(GenericDeviceTy &GenericDevice,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
    const char *GlobalName = DeviceGlobal.getName().data();
    GenELF64DeviceImageTy &GenELF64Image =
        static_cast<GenELF64DeviceImageTy &>(Image);

    // Get dynamic library that has loaded the device image.
    DynamicLibrary &DynLib = GenELF64Image.getDynamicLibrary();

    // Get the address of the symbol.
    void *Addr = DynLib.getAddressOfSymbol(GlobalName);
    if (Addr == nullptr) {
      return Plugin::error("Failed to load global '%s'", GlobalName);
    }

    // Save the pointer to the symbol.
    DeviceGlobal.setPtr(Addr);

    return Plugin::success();
  }
};

/// Class implementing the plugin functionalities for GenELF64.
struct GenELF64PluginTy final : public GenericPluginTy {
  /// Create the GenELF64 plugin.
  GenELF64PluginTy() : GenericPluginTy() {}

  /// This class should not be copied.
  GenELF64PluginTy(const GenELF64PluginTy &) = delete;
  GenELF64PluginTy(GenELF64PluginTy &&) = delete;

  /// Initialize the plugin and return the number of devices.
  Expected<int32_t> initImpl() override { return NUM_DEVICES; }

  /// Deinitialize the plugin.
  Error deinitImpl() override { return Plugin::success(); }

  /// Get the ELF code to recognize the compatible binary images.
  uint16_t getMagicElfBits() const override { return TARGET_ELF_ID; }

  /// This plugin does not support exchanging data between two devices.
  bool isDataExchangable(int32_t SrcDeviceId, int32_t DstDeviceId) override {
    return false;
  }

  /// All images (ELF-compatible) should be compatible with this plugin.
  Expected<bool> isImageCompatible(__tgt_image_info *Info) const override {
    return true;
  }
};

GenericPluginTy *Plugin::createPlugin() { return new GenELF64PluginTy(); }

GenericDeviceTy *Plugin::createDevice(int32_t DeviceId, int32_t NumDevices) {
  return new GenELF64DeviceTy(DeviceId, NumDevices);
}

GenericGlobalHandlerTy *Plugin::createGlobalHandler() {
  return new GenELF64GlobalHandlerTy();
}

template <typename... ArgsTy>
Error Plugin::check(int32_t Code, const char *ErrMsg, ArgsTy... Args) {
  if (Code == 0)
    return Error::success();

  return createStringError<ArgsTy..., const char *>(
      inconvertibleErrorCode(), ErrMsg, Args..., std::to_string(Code).data());
}

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm
