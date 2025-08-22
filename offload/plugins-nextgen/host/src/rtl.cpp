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

#include "Shared/Debug.h"
#include "Shared/Environment.h"
#include "Utils/ELF.h"

#include "GlobalHandler.h"
#include "OpenMP/OMPT/Callback.h"
#include "PluginInterface.h"
#include "omptarget.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPDeviceConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/DynamicLibrary.h"

#if !defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__) ||           \
    !defined(__ORDER_BIG_ENDIAN__)
#error "Missing preprocessor definitions for endianness detection."
#endif

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define LITTLEENDIAN_CPU
#elif defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define BIGENDIAN_CPU
#endif

// The number of devices in this plugin.
#define NUM_DEVICES 4

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// Forward declarations for all specialized data structures.
struct GenELF64KernelTy;
struct GenELF64DeviceTy;
struct GenELF64PluginTy;

using llvm::sys::DynamicLibrary;
using namespace error;

/// Class implementing kernel functionalities for GenELF64.
struct GenELF64KernelTy : public GenericKernelTy {
  /// Construct the kernel with a name and an execution mode.
  GenELF64KernelTy(const char *Name) : GenericKernelTy(Name), Func(nullptr) {}

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
      return Plugin::error(ErrorCode::INVALID_BINARY,
                           "invalid function for kernel %s", getName());

    // Save the function pointer.
    Func = (void (*)())Global.getPtr();

    KernelEnvironment.Configuration.ExecMode = OMP_TGT_EXEC_MODE_GENERIC;
    KernelEnvironment.Configuration.MayUseNestedParallelism = /*Unknown=*/2;
    KernelEnvironment.Configuration.UseGenericStateMachine = /*Unknown=*/2;

    // Set the maximum number of threads to a single.
    MaxNumThreads = 1;
    return Plugin::success();
  }

  /// Launch the kernel using the libffi.
  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads[3],
                   uint32_t NumBlocks[3], KernelArgsTy &KernelArgs,
                   KernelLaunchParamsTy LaunchParams,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override {
    // Create a vector of ffi_types, one per argument.
    SmallVector<ffi_type *, 16> ArgTypes(KernelArgs.NumArgs, &ffi_type_pointer);
    ffi_type **ArgTypesPtr = (ArgTypes.size()) ? &ArgTypes[0] : nullptr;

    // Prepare the cif structure before running the kernel function.
    ffi_cif Cif;
    ffi_status Status = ffi_prep_cif(&Cif, FFI_DEFAULT_ABI, KernelArgs.NumArgs,
                                     &ffi_type_void, ArgTypesPtr);
    if (Status != FFI_OK)
      return Plugin::error(ErrorCode::UNKNOWN, "error in ffi_prep_cif: %d",
                           Status);

    // Call the kernel function through libffi.
    long Return;
    ffi_call(&Cif, Func, &Return, (void **)LaunchParams.Ptrs);

    return Plugin::success();
  }

  /// Return maximum block size for maximum occupancy
  Expected<uint64_t> maxGroupSize(GenericDeviceTy &Device,
                                  uint64_t DynamicMemSize) const override {
    return Plugin::error(
        ErrorCode::UNSUPPORTED,
        "occupancy calculations are not implemented for the host device");
  }

private:
  /// The kernel function to execute.
  void (*Func)(void);
};

/// Class implementing the GenELF64 device images properties.
struct GenELF64DeviceImageTy : public DeviceImageTy {
  /// Create the GenELF64 image with the id and the target image pointer.
  GenELF64DeviceImageTy(int32_t ImageId, GenericDeviceTy &Device,
                        const __tgt_device_image *TgtImage)
      : DeviceImageTy(ImageId, Device, TgtImage), DynLib() {}

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
  GenELF64DeviceTy(GenericPluginTy &Plugin, int32_t DeviceId,
                   int32_t NumDevices)
      : GenericDeviceTy(Plugin, DeviceId, NumDevices, GenELF64GridValues) {}

  ~GenELF64DeviceTy() {}

  /// Initialize the device, which is a no-op
  Error initImpl(GenericPluginTy &Plugin) override { return Plugin::success(); }

  /// Unload the binary image
  ///
  /// TODO: This currently does nothing, and should be implemented as part of
  /// broader memory handling logic for this plugin
  Error unloadBinaryImpl(DeviceImageTy *Image) override {
    auto Elf = reinterpret_cast<GenELF64DeviceImageTy *>(Image);
    DynamicLibrary::closeLibrary(Elf->getDynamicLibrary());
    Plugin.free(Elf);
    return Plugin::success();
  }

  /// Deinitialize the device, which is a no-op
  Error deinitImpl() override { return Plugin::success(); }

  /// See GenericDeviceTy::getComputeUnitKind().
  std::string getComputeUnitKind() const override { return "generic-64bit"; }

  /// Construct the kernel for a specific image on the device.
  Expected<GenericKernelTy &> constructKernel(const char *Name) override {
    // Allocate and construct the kernel.
    GenELF64KernelTy *GenELF64Kernel = Plugin.allocate<GenELF64KernelTy>();
    if (!GenELF64Kernel)
      return Plugin::error(ErrorCode::OUT_OF_RESOURCES,
                           "failed to allocate memory for GenELF64 kernel");

    new (GenELF64Kernel) GenELF64KernelTy(Name);

    return *GenELF64Kernel;
  }

  /// Set the current context to this device, which is a no-op.
  Error setContext() override { return Plugin::success(); }

  /// Load the binary image into the device and allocate an image object.
  Expected<DeviceImageTy *> loadBinaryImpl(const __tgt_device_image *TgtImage,
                                           int32_t ImageId) override {
    // Allocate and initialize the image object.
    GenELF64DeviceImageTy *Image = Plugin.allocate<GenELF64DeviceImageTy>();
    new (Image) GenELF64DeviceImageTy(ImageId, *this, TgtImage);

    // Create a temporary file.
    char TmpFileName[] = "/tmp/tmpfile_XXXXXX";
    int TmpFileFd = mkstemp(TmpFileName);
    if (TmpFileFd == -1)
      return Plugin::error(ErrorCode::HOST_IO,
                           "failed to create tmpfile for loading target image");

    // Open the temporary file.
    FILE *TmpFile = fdopen(TmpFileFd, "wb");
    if (!TmpFile)
      return Plugin::error(ErrorCode::HOST_IO,
                           "failed to open tmpfile %s for loading target image",
                           TmpFileName);

    // Write the image into the temporary file.
    size_t Written = fwrite(Image->getStart(), Image->getSize(), 1, TmpFile);
    if (Written != 1)
      return Plugin::error(ErrorCode::HOST_IO,
                           "failed to write target image to tmpfile %s",
                           TmpFileName);

    // Close the temporary file.
    int Ret = fclose(TmpFile);
    if (Ret)
      return Plugin::error(ErrorCode::HOST_IO,
                           "failed to close tmpfile %s with the target image",
                           TmpFileName);

    // Load the temporary file as a dynamic library.
    std::string ErrMsg;
    DynamicLibrary DynLib = DynamicLibrary::getLibrary(TmpFileName, &ErrMsg);

    // Check if the loaded library is valid.
    if (!DynLib.isValid())
      return Plugin::error(ErrorCode::INVALID_BINARY,
                           "failed to load target image: %s", ErrMsg.c_str());

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
    case TARGET_ALLOC_DEVICE_NON_BLOCKING:
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
    return Plugin::error(ErrorCode::UNSUPPORTED,
                         "dataExchangeImpl not supported");
  }

  /// Insert a data fence between previous data operations and the following
  /// operations. This is a no-op for Host devices as operations inserted into
  /// a queue are in-order.
  Error dataFence(__tgt_async_info *Async) override {
    return Plugin::success();
  }

  /// All functions are already synchronous. No need to do anything on this
  /// synchronization function.
  Error synchronizeImpl(__tgt_async_info &AsyncInfo,
                        bool ReleaseQueue) override {
    return Plugin::success();
  }

  /// All functions are already synchronous. No need to do anything on this
  /// query function.
  Error queryAsyncImpl(__tgt_async_info &AsyncInfo) override {
    return Plugin::success();
  }

  /// This plugin does not support interoperability
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::error(ErrorCode::UNSUPPORTED,
                         "initAsyncInfoImpl not supported");
  }

  /// This plugin does not support interoperability
  Error initDeviceInfoImpl(__tgt_device_info *DeviceInfo) override {
    return Plugin::error(ErrorCode::UNSUPPORTED,
                         "initDeviceInfoImpl not supported");
  }

  Error enqueueHostCallImpl(void (*Callback)(void *), void *UserData,
                            AsyncInfoWrapperTy &AsyncInfo) override {
    Callback(UserData);
    return Plugin::success();
  };

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
  Expected<bool> hasPendingWorkImpl(AsyncInfoWrapperTy &AsyncInfo) override {
    return true;
  }
  Expected<bool> isEventCompleteImpl(void *Event,
                                     AsyncInfoWrapperTy &AsyncInfo) override {
    return true;
  }
  Error syncEventImpl(void *EventPtr) override { return Plugin::success(); }

  /// Print information about the device.
  Expected<InfoTreeNode> obtainInfoImpl() override {
    InfoTreeNode Info;
    Info.add("Device Type", "Generic-elf-64bit");
    return Info;
  }

  /// This plugin should not setup the device environment or memory pool.
  virtual bool shouldSetupDeviceEnvironment() const override { return false; };
  virtual bool shouldSetupDeviceMemoryPool() const override { return false; };

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
      1, // GV_Default_Num_Teams
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
      return Plugin::error(ErrorCode::NOT_FOUND, "failed to load global '%s'",
                           GlobalName);
    }

    // Save the pointer to the symbol.
    DeviceGlobal.setPtr(Addr);

    return Plugin::success();
  }
};

/// Class implementing the plugin functionalities for GenELF64.
struct GenELF64PluginTy final : public GenericPluginTy {
  /// Create the GenELF64 plugin.
  GenELF64PluginTy() : GenericPluginTy(getTripleArch()) {}

  /// This class should not be copied.
  GenELF64PluginTy(const GenELF64PluginTy &) = delete;
  GenELF64PluginTy(GenELF64PluginTy &&) = delete;

  /// Initialize the plugin and return the number of devices.
  Expected<int32_t> initImpl() override {
#ifdef USES_DYNAMIC_FFI
    if (auto Err = Plugin::check(ffi_init(), "failed to initialize libffi"))
      return std::move(Err);
#endif

    return NUM_DEVICES;
  }

  /// Deinitialize the plugin.
  Error deinitImpl() override { return Plugin::success(); }

  /// Creates a generic ELF device.
  GenericDeviceTy *createDevice(GenericPluginTy &Plugin, int32_t DeviceId,
                                int32_t NumDevices) override {
    return new GenELF64DeviceTy(Plugin, DeviceId, NumDevices);
  }

  /// Creates a generic global handler.
  GenericGlobalHandlerTy *createGlobalHandler() override {
    return new GenELF64GlobalHandlerTy();
  }

  /// Get the ELF code to recognize the compatible binary images.
  uint16_t getMagicElfBits() const override {
    return utils::elf::getTargetMachine();
  }

  /// This plugin does not support exchanging data between two devices.
  bool isDataExchangable(int32_t SrcDeviceId, int32_t DstDeviceId) override {
    return false;
  }

  /// All images (ELF-compatible) should be compatible with this plugin.
  Expected<bool> isELFCompatible(uint32_t, StringRef) const override {
    return true;
  }

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
#elif defined(__riscv) && (__riscv_xlen == 64)
    return llvm::Triple::riscv64;
#elif defined(__loongarch__) && (__loongarch_grlen == 64)
    return llvm::Triple::loongarch64;
#else
    return llvm::Triple::UnknownArch;
#endif
  }

  const char *getName() const override { return GETNAME(TARGET_NAME); }
};

template <typename... ArgsTy>
static Error Plugin::check(int32_t Code, const char *ErrMsg, ArgsTy... Args) {
  if (Code == 0)
    return Plugin::success();

  return Plugin::error(ErrorCode::UNKNOWN, ErrMsg, Args...,
                       std::to_string(Code).data());
}

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

extern "C" {
llvm::omp::target::plugin::GenericPluginTy *createPlugin_host() {
  return new llvm::omp::target::plugin::GenELF64PluginTy();
}
}
