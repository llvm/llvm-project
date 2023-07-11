//===----RTLs/cuda/src/rtl.cpp - Target RTLs Implementation ------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTL NextGen for CUDA machine
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstddef>
#include <cuda.h>
#include <string>
#include <unordered_map>

#include "Debug.h"
#include "DeviceEnvironment.h"
#include "GlobalHandler.h"
#include "OmptCallback.h"
#include "PluginInterface.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// Forward declarations for all specialized data structures.
struct CUDAKernelTy;
struct CUDADeviceTy;
struct CUDAPluginTy;

/// Class implementing the CUDA kernel functionalities which derives from the
/// generic kernel class.
struct CUDAKernelTy : public GenericKernelTy {
  /// Create a CUDA kernel with a name, an execution mode, and the kernel
  /// function.
  CUDAKernelTy(const char *Name, OMPTgtExecModeFlags ExecutionMode,
               CUfunction Func)
      : GenericKernelTy(Name, ExecutionMode), Func(Func) {}

  /// Initialize the CUDA kernel
  Error initImpl(GenericDeviceTy &GenericDevice,
                 DeviceImageTy &Image) override {
    int MaxThreads;
    CUresult Res = cuFuncGetAttribute(
        &MaxThreads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, Func);
    if (auto Err = Plugin::check(Res, "Error in cuFuncGetAttribute: %s"))
      return Err;

    /// Set the maximum number of threads for the CUDA kernel.
    MaxNumThreads = std::min(MaxNumThreads, (uint32_t)MaxThreads);

    return Plugin::success();
  }

  /// Launch the CUDA kernel function
  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads,
                   uint64_t NumBlocks, KernelArgsTy &KernelArgs, void *Args,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override;

  /// The default number of blocks is common to the whole device.
  uint32_t getDefaultNumBlocks(GenericDeviceTy &GenericDevice) const override {
    return GenericDevice.getDefaultNumBlocks();
  }

  /// The default number of threads is common to the whole device.
  uint32_t getDefaultNumThreads(GenericDeviceTy &GenericDevice) const override {
    return GenericDevice.getDefaultNumThreads();
  }

private:
  /// The CUDA kernel function to execute.
  CUfunction Func;
};

/// Class wrapping a CUDA stream reference. These are the objects handled by the
/// Stream Manager for the CUDA plugin.
class CUDAStreamRef final : public GenericDeviceResourceRef {
  /// The reference to the CUDA stream.
  CUstream Stream;

public:
  /// Create an empty reference to an invalid stream.
  CUDAStreamRef() : Stream(nullptr) {}

  /// Create a reference to an existing stream.
  CUDAStreamRef(CUstream Stream) : Stream(Stream) {}

  /// Create a new stream and save the reference. The reference must be empty
  /// before calling to this function.
  Error create(GenericDeviceTy &Device) override {
    if (Stream)
      return Plugin::error("Creating an existing stream");

    CUresult Res = cuStreamCreate(&Stream, CU_STREAM_NON_BLOCKING);
    if (auto Err = Plugin::check(Res, "Error in cuStreamCreate: %s"))
      return Err;

    return Plugin::success();
  }

  /// Destroy the referenced stream and invalidate the reference. The reference
  /// must be to a valid stream before calling to this function.
  Error destroy(GenericDeviceTy &Device) override {
    if (!Stream)
      return Plugin::error("Destroying an invalid stream");

    CUresult Res = cuStreamDestroy(Stream);
    if (auto Err = Plugin::check(Res, "Error in cuStreamDestroy: %s"))
      return Err;

    Stream = nullptr;
    return Plugin::success();
  }

  /// Get the underlying CUstream.
  operator CUstream() const { return Stream; }
};

/// Class wrapping a CUDA event reference. These are the objects handled by the
/// Event Manager for the CUDA plugin.
class CUDAEventRef final : public GenericDeviceResourceRef {
  CUevent Event;

public:
  /// Create an empty reference to an invalid event.
  CUDAEventRef() : Event(nullptr) {}

  /// Create a reference to an existing event.
  CUDAEventRef(CUevent Event) : Event(Event) {}

  /// Create a new event and save the reference. The reference must be empty
  /// before calling to this function.
  Error create(GenericDeviceTy &Device) override {
    if (Event)
      return Plugin::error("Creating an existing event");

    CUresult Res = cuEventCreate(&Event, CU_EVENT_DEFAULT);
    if (auto Err = Plugin::check(Res, "Error in cuEventCreate: %s"))
      return Err;

    return Plugin::success();
  }

  /// Destroy the referenced event and invalidate the reference. The reference
  /// must be to a valid event before calling to this function.
  Error destroy(GenericDeviceTy &Device) override {
    if (!Event)
      return Plugin::error("Destroying an invalid event");

    CUresult Res = cuEventDestroy(Event);
    if (auto Err = Plugin::check(Res, "Error in cuEventDestroy: %s"))
      return Err;

    Event = nullptr;
    return Plugin::success();
  }

  /// Get the underlying CUevent.
  operator CUevent() const { return Event; }
};

/// Class implementing the CUDA device images properties.
struct CUDADeviceImageTy : public DeviceImageTy {
  /// Create the CUDA image with the id and the target image pointer.
  CUDADeviceImageTy(int32_t ImageId, const __tgt_device_image *TgtImage)
      : DeviceImageTy(ImageId, TgtImage), Module(nullptr) {}

  /// Load the image as a CUDA module.
  Error loadModule() {
    assert(!Module && "Module already loaded");

    CUresult Res = cuModuleLoadDataEx(&Module, getStart(), 0, nullptr, nullptr);
    if (auto Err = Plugin::check(Res, "Error in cuModuleLoadDataEx: %s"))
      return Err;

    return Plugin::success();
  }

  /// Unload the CUDA module corresponding to the image.
  Error unloadModule() {
    assert(Module && "Module not loaded");

    CUresult Res = cuModuleUnload(Module);
    if (auto Err = Plugin::check(Res, "Error in cuModuleUnload: %s"))
      return Err;

    Module = nullptr;

    return Plugin::success();
  }

  /// Getter of the CUDA module.
  CUmodule getModule() const { return Module; }

private:
  /// The CUDA module that loaded the image.
  CUmodule Module;
};

/// Class implementing the CUDA device functionalities which derives from the
/// generic device class.
struct CUDADeviceTy : public GenericDeviceTy {
  // Create a CUDA device with a device id and the default CUDA grid values.
  CUDADeviceTy(int32_t DeviceId, int32_t NumDevices)
      : GenericDeviceTy(DeviceId, NumDevices, NVPTXGridValues),
        CUDAStreamManager(*this), CUDAEventManager(*this) {}

  ~CUDADeviceTy() {}

  /// Initialize the device, its resources and get its properties.
  Error initImpl(GenericPluginTy &Plugin) override {
    CUresult Res = cuDeviceGet(&Device, DeviceId);
    if (auto Err = Plugin::check(Res, "Error in cuDeviceGet: %s"))
      return Err;

    // Query the current flags of the primary context and set its flags if
    // it is inactive.
    unsigned int FormerPrimaryCtxFlags = 0;
    int FormerPrimaryCtxIsActive = 0;
    Res = cuDevicePrimaryCtxGetState(Device, &FormerPrimaryCtxFlags,
                                     &FormerPrimaryCtxIsActive);
    if (auto Err =
            Plugin::check(Res, "Error in cuDevicePrimaryCtxGetState: %s"))
      return Err;

    if (FormerPrimaryCtxIsActive) {
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
           "The primary context is active, no change to its flags\n");
      if ((FormerPrimaryCtxFlags & CU_CTX_SCHED_MASK) !=
          CU_CTX_SCHED_BLOCKING_SYNC)
        INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
             "Warning: The current flags are not CU_CTX_SCHED_BLOCKING_SYNC\n");
    } else {
      INFO(OMP_INFOTYPE_PLUGIN_KERNEL, DeviceId,
           "The primary context is inactive, set its flags to "
           "CU_CTX_SCHED_BLOCKING_SYNC\n");
      Res = cuDevicePrimaryCtxSetFlags(Device, CU_CTX_SCHED_BLOCKING_SYNC);
      if (auto Err =
              Plugin::check(Res, "Error in cuDevicePrimaryCtxSetFlags: %s"))
        return Err;
    }

    // Retain the per device primary context and save it to use whenever this
    // device is selected.
    Res = cuDevicePrimaryCtxRetain(&Context, Device);
    if (auto Err = Plugin::check(Res, "Error in cuDevicePrimaryCtxRetain: %s"))
      return Err;

    if (auto Err = setContext())
      return Err;

    // Initialize stream pool.
    if (auto Err = CUDAStreamManager.init(OMPX_InitialNumStreams))
      return Err;

    // Initialize event pool.
    if (auto Err = CUDAEventManager.init(OMPX_InitialNumEvents))
      return Err;

    // Query attributes to determine number of threads/block and blocks/grid.
    if (auto Err = getDeviceAttr(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                                 GridValues.GV_Max_Teams))
      return Err;

    if (auto Err = getDeviceAttr(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                                 GridValues.GV_Max_WG_Size))
      return Err;

    if (auto Err = getDeviceAttr(CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                 GridValues.GV_Warp_Size))
      return Err;

    if (auto Err = getDeviceAttr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                 ComputeCapability.Major))
      return Err;

    if (auto Err = getDeviceAttr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                 ComputeCapability.Minor))
      return Err;

    return Plugin::success();
  }

  /// Deinitialize the device and release its resources.
  Error deinitImpl() override {
    if (Context) {
      if (auto Err = setContext())
        return Err;
    }

    // Deinitialize the stream manager.
    if (auto Err = CUDAStreamManager.deinit())
      return Err;

    if (auto Err = CUDAEventManager.deinit())
      return Err;

    // Close modules if necessary.
    if (!LoadedImages.empty()) {
      assert(Context && "Invalid CUDA context");

      // Each image has its own module.
      for (DeviceImageTy *Image : LoadedImages) {
        CUDADeviceImageTy &CUDAImage = static_cast<CUDADeviceImageTy &>(*Image);

        // Unload the module of the image.
        if (auto Err = CUDAImage.unloadModule())
          return Err;
      }
    }

    if (Context) {
      CUresult Res = cuDevicePrimaryCtxRelease(Device);
      if (auto Err =
              Plugin::check(Res, "Error in cuDevicePrimaryCtxRelease: %s"))
        return Err;
    }

    // Invalidate context and device references.
    Context = nullptr;
    Device = CU_DEVICE_INVALID;

    return Plugin::success();
  }

  /// Allocate and construct a CUDA kernel.
  Expected<GenericKernelTy *>
  constructKernelEntry(const __tgt_offload_entry &KernelEntry,
                       DeviceImageTy &Image) override {
    CUDADeviceImageTy &CUDAImage = static_cast<CUDADeviceImageTy &>(Image);

    // Retrieve the function pointer of the kernel.
    CUfunction Func;
    CUresult Res =
        cuModuleGetFunction(&Func, CUDAImage.getModule(), KernelEntry.name);
    if (auto Err = Plugin::check(Res, "Error in cuModuleGetFunction('%s'): %s",
                                 KernelEntry.name))
      return std::move(Err);

    DP("Entry point " DPxMOD " maps to %s (" DPxMOD ")\n", DPxPTR(&KernelEntry),
       KernelEntry.name, DPxPTR(Func));

    Expected<OMPTgtExecModeFlags> ExecModeOrErr =
        getExecutionModeForKernel(KernelEntry.name, Image);
    if (!ExecModeOrErr)
      return ExecModeOrErr.takeError();

    // Allocate and initialize the CUDA kernel.
    CUDAKernelTy *CUDAKernel = Plugin::get().allocate<CUDAKernelTy>();
    new (CUDAKernel) CUDAKernelTy(KernelEntry.name, ExecModeOrErr.get(), Func);

    return CUDAKernel;
  }

  /// Set the current context to this device's context.
  Error setContext() override {
    CUresult Res = cuCtxSetCurrent(Context);
    return Plugin::check(Res, "Error in cuCtxSetCurrent: %s");
  }

  /// We want to set up the RPC server for host services to the GPU if it is
  /// availible.
  bool shouldSetupRPCServer() const override {
    return libomptargetSupportsRPC();
  }

  /// Get the stream of the asynchronous info sructure or get a new one.
  CUstream getStream(AsyncInfoWrapperTy &AsyncInfoWrapper) {
    CUstream &Stream = AsyncInfoWrapper.getQueueAs<CUstream>();
    if (!Stream)
      Stream = CUDAStreamManager.getResource();
    return Stream;
  }

  /// Getters of CUDA references.
  CUcontext getCUDAContext() const { return Context; }
  CUdevice getCUDADevice() const { return Device; }

  /// Load the binary image into the device and allocate an image object.
  Expected<DeviceImageTy *> loadBinaryImpl(const __tgt_device_image *TgtImage,
                                           int32_t ImageId) override {
    if (auto Err = setContext())
      return std::move(Err);

    // Allocate and initialize the image object.
    CUDADeviceImageTy *CUDAImage = Plugin::get().allocate<CUDADeviceImageTy>();
    new (CUDAImage) CUDADeviceImageTy(ImageId, TgtImage);

    // Load the CUDA module.
    if (auto Err = CUDAImage->loadModule())
      return std::move(Err);

    return CUDAImage;
  }

  /// Allocate memory on the device or related to the device.
  void *allocate(size_t Size, void *, TargetAllocTy Kind) override {
    if (Size == 0)
      return nullptr;

    if (auto Err = setContext()) {
      REPORT("Failure to alloc memory: %s\n", toString(std::move(Err)).data());
      return nullptr;
    }

    void *MemAlloc = nullptr;
    CUdeviceptr DevicePtr;
    CUresult Res;

    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
      Res = cuMemAlloc(&DevicePtr, Size);
      MemAlloc = (void *)DevicePtr;
      break;
    case TARGET_ALLOC_HOST:
      Res = cuMemAllocHost(&MemAlloc, Size);
      break;
    case TARGET_ALLOC_SHARED:
      Res = cuMemAllocManaged(&DevicePtr, Size, CU_MEM_ATTACH_GLOBAL);
      MemAlloc = (void *)DevicePtr;
      break;
    }

    if (auto Err =
            Plugin::check(Res, "Error in cuMemAlloc[Host|Managed]: %s")) {
      REPORT("Failure to alloc memory: %s\n", toString(std::move(Err)).data());
      return nullptr;
    }
    return MemAlloc;
  }

  /// Deallocate memory on the device or related to the device.
  int free(void *TgtPtr, TargetAllocTy Kind) override {
    if (TgtPtr == nullptr)
      return OFFLOAD_SUCCESS;

    if (auto Err = setContext()) {
      REPORT("Failure to free memory: %s\n", toString(std::move(Err)).data());
      return OFFLOAD_FAIL;
    }

    CUresult Res;
    switch (Kind) {
    case TARGET_ALLOC_DEFAULT:
    case TARGET_ALLOC_DEVICE:
    case TARGET_ALLOC_SHARED:
      Res = cuMemFree((CUdeviceptr)TgtPtr);
      break;
    case TARGET_ALLOC_HOST:
      Res = cuMemFreeHost(TgtPtr);
      break;
    }

    if (auto Err = Plugin::check(Res, "Error in cuMemFree[Host]: %s")) {
      REPORT("Failure to free memory: %s\n", toString(std::move(Err)).data());
      return OFFLOAD_FAIL;
    }
    return OFFLOAD_SUCCESS;
  }

  /// Synchronize current thread with the pending operations on the async info.
  Error synchronizeImpl(__tgt_async_info &AsyncInfo) override {
    CUstream Stream = reinterpret_cast<CUstream>(AsyncInfo.Queue);
    CUresult Res;
    // If we have an RPC server running on this device we will continuously
    // query it for work rather than blocking.
    if (!getRPCServer()) {
      Res = cuStreamSynchronize(Stream);
    } else {
      do {
        Res = cuStreamQuery(Stream);
        if (auto Err = getRPCServer()->runServer(*this))
          return Err;
      } while (Res == CUDA_ERROR_NOT_READY);
    }

    // Once the stream is synchronized, return it to stream pool and reset
    // AsyncInfo. This is to make sure the synchronization only works for its
    // own tasks.
    CUDAStreamManager.returnResource(Stream);
    AsyncInfo.Queue = nullptr;

    return Plugin::check(Res, "Error in cuStreamSynchronize: %s");
  }

  /// Query for the completion of the pending operations on the async info.
  Error queryAsyncImpl(__tgt_async_info &AsyncInfo) override {
    CUstream Stream = reinterpret_cast<CUstream>(AsyncInfo.Queue);
    CUresult Res = cuStreamQuery(Stream);

    // Not ready streams must be considered as successful operations.
    if (Res == CUDA_ERROR_NOT_READY)
      return Plugin::success();

    // Once the stream is synchronized and the operations completed (or an error
    // occurs), return it to stream pool and reset AsyncInfo. This is to make
    // sure the synchronization only works for its own tasks.
    CUDAStreamManager.returnResource(Stream);
    AsyncInfo.Queue = nullptr;

    return Plugin::check(Res, "Error in cuStreamQuery: %s");
  }

  Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) override {
    // TODO: Register the buffer as CUDA host memory.
    return HstPtr;
  }

  Error dataUnlockImpl(void *HstPtr) override { return Plugin::success(); }

  Expected<bool> isPinnedPtrImpl(void *HstPtr, void *&BaseHstPtr,
                                 void *&BaseDevAccessiblePtr,
                                 size_t &BaseSize) const override {
    // TODO: Implement pinning feature for CUDA.
    return false;
  }

  /// Submit data to the device (host to device transfer).
  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    if (auto Err = setContext())
      return Err;

    CUstream Stream = getStream(AsyncInfoWrapper);
    if (!Stream)
      return Plugin::error("Failure to get stream");

    CUresult Res = cuMemcpyHtoDAsync((CUdeviceptr)TgtPtr, HstPtr, Size, Stream);
    return Plugin::check(Res, "Error in cuMemcpyHtoDAsync: %s");
  }

  /// Retrieve data from the device (device to host transfer).
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    if (auto Err = setContext())
      return Err;

    CUstream Stream = getStream(AsyncInfoWrapper);
    if (!Stream)
      return Plugin::error("Failure to get stream");

    CUresult Res = cuMemcpyDtoHAsync(HstPtr, (CUdeviceptr)TgtPtr, Size, Stream);
    return Plugin::check(Res, "Error in cuMemcpyDtoHAsync: %s");
  }

  /// Exchange data between two devices directly. We may use peer access if
  /// the CUDA devices and driver allow them.
  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstGenericDevice,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override;

  /// Initialize the async info for interoperability purposes.
  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    if (auto Err = setContext())
      return Err;

    if (!getStream(AsyncInfoWrapper))
      return Plugin::error("Failure to get stream");

    return Plugin::success();
  }

  /// Initialize the device info for interoperability purposes.
  Error initDeviceInfoImpl(__tgt_device_info *DeviceInfo) override {
    assert(Context && "Context is null");
    assert(Device != CU_DEVICE_INVALID && "Invalid CUDA device");

    if (auto Err = setContext())
      return Err;

    if (!DeviceInfo->Context)
      DeviceInfo->Context = Context;

    if (!DeviceInfo->Device)
      DeviceInfo->Device = reinterpret_cast<void *>(Device);

    return Plugin::success();
  }

  /// Create an event.
  Error createEventImpl(void **EventPtrStorage) override {
    CUevent *Event = reinterpret_cast<CUevent *>(EventPtrStorage);
    *Event = CUDAEventManager.getResource();
    return Plugin::success();
  }

  /// Destroy a previously created event.
  Error destroyEventImpl(void *EventPtr) override {
    CUevent Event = reinterpret_cast<CUevent>(EventPtr);
    CUDAEventManager.returnResource(Event);
    return Plugin::success();
  }

  /// Record the event.
  Error recordEventImpl(void *EventPtr,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    CUevent Event = reinterpret_cast<CUevent>(EventPtr);

    CUstream Stream = getStream(AsyncInfoWrapper);
    if (!Stream)
      return Plugin::error("Failure to get stream");

    CUresult Res = cuEventRecord(Event, Stream);
    return Plugin::check(Res, "Error in cuEventRecord: %s");
  }

  /// Make the stream wait on the event.
  Error waitEventImpl(void *EventPtr,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    CUevent Event = reinterpret_cast<CUevent>(EventPtr);

    CUstream Stream = getStream(AsyncInfoWrapper);
    if (!Stream)
      return Plugin::error("Failure to get stream");

    // Do not use CU_EVENT_WAIT_DEFAULT here as it is only available from
    // specific CUDA version, and defined as 0x0. In previous version, per CUDA
    // API document, that argument has to be 0x0.
    CUresult Res = cuStreamWaitEvent(Stream, Event, 0);
    return Plugin::check(Res, "Error in cuStreamWaitEvent: %s");
  }

  /// Synchronize the current thread with the event.
  Error syncEventImpl(void *EventPtr) override {
    CUevent Event = reinterpret_cast<CUevent>(EventPtr);
    CUresult Res = cuEventSynchronize(Event);
    return Plugin::check(Res, "Error in cuEventSynchronize: %s");
  }

  /// Print information about the device.
  Error obtainInfoImpl(InfoQueueTy &Info) override {
    char TmpChar[1000];
    const char *TmpCharPtr;
    size_t TmpSt;
    int TmpInt;

    CUresult Res = cuDriverGetVersion(&TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("CUDA Driver Version", TmpInt);

    Info.add("CUDA OpenMP Device Number", DeviceId);

    Res = cuDeviceGetName(TmpChar, 1000, Device);
    if (Res == CUDA_SUCCESS)
      Info.add("Device Name", TmpChar);

    Res = cuDeviceTotalMem(&TmpSt, Device);
    if (Res == CUDA_SUCCESS)
      Info.add("Global Memory Size", TmpSt, "bytes");

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Number of Multiprocessors", TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_GPU_OVERLAP, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Concurrent Copy and Execution", (bool)TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Total Constant Memory", TmpInt, "bytes");

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                           TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Max Shared Memory per Block", TmpInt, "bytes");

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Registers per Block", TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_WARP_SIZE, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Warp Size", TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Maximum Threads per Block", TmpInt);

    Info.add("Maximum Block Dimensions", "");
    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add<InfoLevel2>("x", TmpInt);
    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add<InfoLevel2>("y", TmpInt);
    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add<InfoLevel2>("z", TmpInt);

    Info.add("Maximum Grid Dimensions", "");
    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add<InfoLevel2>("x", TmpInt);
    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add<InfoLevel2>("y", TmpInt);
    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add<InfoLevel2>("z", TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MAX_PITCH, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Maximum Memory Pitch", TmpInt, "bytes");

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Texture Alignment", TmpInt, "bytes");

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Clock Rate", TmpInt, "kHz");

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Execution Timeout", (bool)TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_INTEGRATED, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Integrated Device", (bool)TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Can Map Host Memory", (bool)TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, TmpInt);
    if (Res == CUDA_SUCCESS) {
      if (TmpInt == CU_COMPUTEMODE_DEFAULT)
        TmpCharPtr = "Default";
      else if (TmpInt == CU_COMPUTEMODE_PROHIBITED)
        TmpCharPtr = "Prohibited";
      else if (TmpInt == CU_COMPUTEMODE_EXCLUSIVE_PROCESS)
        TmpCharPtr = "Exclusive process";
      else
        TmpCharPtr = "Unknown";
      Info.add("Compute Mode", TmpCharPtr);
    }

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Concurrent Kernels", (bool)TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_ECC_ENABLED, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("ECC Enabled", (bool)TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Memory Clock Rate", TmpInt, "kHz");

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Memory Bus Width", TmpInt, "bits");

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("L2 Cache Size", TmpInt, "bytes");

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                           TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Max Threads Per SMP", TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Async Engines", TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Unified Addressing", (bool)TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Managed Memory", (bool)TmpInt);

    Res =
        getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Concurrent Managed Memory", (bool)TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED,
                           TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Preemption Supported", (bool)TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Cooperative Launch", (bool)TmpInt);

    Res = getDeviceAttrRaw(CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD, TmpInt);
    if (Res == CUDA_SUCCESS)
      Info.add("Multi-Device Boars", (bool)TmpInt);

    Info.add("Compute Capabilities", ComputeCapability.str());

    return Plugin::success();
  }

  /// Getters and setters for stack and heap sizes.
  Error getDeviceStackSize(uint64_t &Value) override {
    return getCtxLimit(CU_LIMIT_STACK_SIZE, Value);
  }
  Error setDeviceStackSize(uint64_t Value) override {
    return setCtxLimit(CU_LIMIT_STACK_SIZE, Value);
  }
  Error getDeviceHeapSize(uint64_t &Value) override {
    return getCtxLimit(CU_LIMIT_MALLOC_HEAP_SIZE, Value);
  }
  Error setDeviceHeapSize(uint64_t Value) override {
    return setCtxLimit(CU_LIMIT_MALLOC_HEAP_SIZE, Value);
  }

  /// CUDA-specific functions for getting and setting context limits.
  Error setCtxLimit(CUlimit Kind, uint64_t Value) {
    CUresult Res = cuCtxSetLimit(Kind, Value);
    return Plugin::check(Res, "Error in cuCtxSetLimit: %s");
  }
  Error getCtxLimit(CUlimit Kind, uint64_t &Value) {
    CUresult Res = cuCtxGetLimit(&Value, Kind);
    return Plugin::check(Res, "Error in cuCtxGetLimit: %s");
  }

  /// CUDA-specific function to get device attributes.
  Error getDeviceAttr(uint32_t Kind, uint32_t &Value) {
    // TODO: Warn if the new value is larger than the old.
    CUresult Res =
        cuDeviceGetAttribute((int *)&Value, (CUdevice_attribute)Kind, Device);
    return Plugin::check(Res, "Error in cuDeviceGetAttribute: %s");
  }

  CUresult getDeviceAttrRaw(uint32_t Kind, int &Value) {
    return cuDeviceGetAttribute(&Value, (CUdevice_attribute)Kind, Device);
  }

  /// See GenericDeviceTy::getComputeUnitKind().
  std::string getComputeUnitKind() const override {
    return ComputeCapability.str();
  }

  /// Returns the clock frequency for the given NVPTX device.
  uint64_t getClockFrequency() const override { return 1000000000; }

private:
  using CUDAStreamManagerTy = GenericDeviceResourceManagerTy<CUDAStreamRef>;
  using CUDAEventManagerTy = GenericDeviceResourceManagerTy<CUDAEventRef>;

  /// Stream manager for CUDA streams.
  CUDAStreamManagerTy CUDAStreamManager;

  /// Event manager for CUDA events.
  CUDAEventManagerTy CUDAEventManager;

  /// The device's context. This context should be set before performing
  /// operations on the device.
  CUcontext Context = nullptr;

  /// The CUDA device handler.
  CUdevice Device = CU_DEVICE_INVALID;

  /// The compute capability of the corresponding CUDA device.
  struct ComputeCapabilityTy {
    uint32_t Major;
    uint32_t Minor;
    std::string str() const {
      return "sm_" + std::to_string(Major * 10 + Minor);
    }
  } ComputeCapability;
};

Error CUDAKernelTy::launchImpl(GenericDeviceTy &GenericDevice,
                               uint32_t NumThreads, uint64_t NumBlocks,
                               KernelArgsTy &KernelArgs, void *Args,
                               AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  CUDADeviceTy &CUDADevice = static_cast<CUDADeviceTy &>(GenericDevice);

  CUstream Stream = CUDADevice.getStream(AsyncInfoWrapper);
  if (!Stream)
    return Plugin::error("Failure to get stream");

  uint32_t MaxDynCGroupMem =
      std::max(KernelArgs.DynCGroupMem, GenericDevice.getDynamicMemorySize());

  CUresult Res =
      cuLaunchKernel(Func, NumBlocks, /* gridDimY */ 1,
                     /* gridDimZ */ 1, NumThreads,
                     /* blockDimY */ 1, /* blockDimZ */ 1, MaxDynCGroupMem,
                     Stream, (void **)Args, nullptr);
  return Plugin::check(Res, "Error in cuLaunchKernel for '%s': %s", getName());
}

/// Class implementing the CUDA-specific functionalities of the global handler.
class CUDAGlobalHandlerTy final : public GenericGlobalHandlerTy {
public:
  /// Get the metadata of a global from the device. The name and size of the
  /// global is read from DeviceGlobal and the address of the global is written
  /// to DeviceGlobal.
  Error getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
    CUDADeviceImageTy &CUDAImage = static_cast<CUDADeviceImageTy &>(Image);

    const char *GlobalName = DeviceGlobal.getName().data();

    size_t CUSize;
    CUdeviceptr CUPtr;
    CUresult Res =
        cuModuleGetGlobal(&CUPtr, &CUSize, CUDAImage.getModule(), GlobalName);
    if (auto Err = Plugin::check(Res, "Error in cuModuleGetGlobal for '%s': %s",
                                 GlobalName))
      return Err;

    if (CUSize != DeviceGlobal.getSize())
      return Plugin::error(
          "Failed to load global '%s' due to size mismatch (%zu != %zu)",
          GlobalName, CUSize, (size_t)DeviceGlobal.getSize());

    DeviceGlobal.setPtr(reinterpret_cast<void *>(CUPtr));
    return Plugin::success();
  }
};

/// Class implementing the CUDA-specific functionalities of the plugin.
struct CUDAPluginTy final : public GenericPluginTy {
  /// Create a CUDA plugin.
  CUDAPluginTy() : GenericPluginTy(getTripleArch()) {}

  /// This class should not be copied.
  CUDAPluginTy(const CUDAPluginTy &) = delete;
  CUDAPluginTy(CUDAPluginTy &&) = delete;

  /// Initialize the plugin and return the number of devices.
  Expected<int32_t> initImpl() override {
    CUresult Res = cuInit(0);
    if (Res == CUDA_ERROR_INVALID_HANDLE) {
      // Cannot call cuGetErrorString if dlsym failed.
      DP("Failed to load CUDA shared library\n");
      return 0;
    }

#ifdef OMPT_SUPPORT
    ompt::connectLibrary();
#endif

    if (Res == CUDA_ERROR_NO_DEVICE) {
      // Do not initialize if there are no devices.
      DP("There are no devices supporting CUDA.\n");
      return 0;
    }

    if (auto Err = Plugin::check(Res, "Error in cuInit: %s"))
      return std::move(Err);

    // Get the number of devices.
    int NumDevices;
    Res = cuDeviceGetCount(&NumDevices);
    if (auto Err = Plugin::check(Res, "Error in cuDeviceGetCount: %s"))
      return std::move(Err);

    // Do not initialize if there are no devices.
    if (NumDevices == 0)
      DP("There are no devices supporting CUDA.\n");

    return NumDevices;
  }

  /// Deinitialize the plugin.
  Error deinitImpl() override { return Plugin::success(); }

  /// Get the ELF code for recognizing the compatible image binary.
  uint16_t getMagicElfBits() const override { return ELF::EM_CUDA; }

  Triple::ArchType getTripleArch() const override {
    // TODO: I think we can drop the support for 32-bit NVPTX devices.
    return Triple::nvptx64;
  }

  /// Check whether the image is compatible with the available CUDA devices.
  Expected<bool> isImageCompatible(__tgt_image_info *Info) const override {
    for (int32_t DevId = 0; DevId < getNumDevices(); ++DevId) {
      CUdevice Device;
      CUresult Res = cuDeviceGet(&Device, DevId);
      if (auto Err = Plugin::check(Res, "Error in cuDeviceGet: %s"))
        return std::move(Err);

      int32_t Major, Minor;
      Res = cuDeviceGetAttribute(
          &Major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, Device);
      if (auto Err = Plugin::check(Res, "Error in cuDeviceGetAttribute: %s"))
        return std::move(Err);

      Res = cuDeviceGetAttribute(
          &Minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, Device);
      if (auto Err = Plugin::check(Res, "Error in cuDeviceGetAttribute: %s"))
        return std::move(Err);

      StringRef ArchStr(Info->Arch);
      StringRef PrefixStr("sm_");
      if (!ArchStr.startswith(PrefixStr))
        return Plugin::error("Unrecognized image arch %s", ArchStr.data());

      int32_t ImageMajor = ArchStr[PrefixStr.size() + 0] - '0';
      int32_t ImageMinor = ArchStr[PrefixStr.size() + 1] - '0';

      // A cubin generated for a certain compute capability is supported to run
      // on any GPU with the same major revision and same or higher minor
      // revision.
      if (Major != ImageMajor || Minor < ImageMinor)
        return false;
    }
    return true;
  }
};

Error CUDADeviceTy::dataExchangeImpl(const void *SrcPtr,
                                     GenericDeviceTy &DstGenericDevice,
                                     void *DstPtr, int64_t Size,
                                     AsyncInfoWrapperTy &AsyncInfoWrapper) {
  if (auto Err = setContext())
    return Err;

  CUDADeviceTy &DstDevice = static_cast<CUDADeviceTy &>(DstGenericDevice);

  CUresult Res;
  int32_t DstDeviceId = DstDevice.DeviceId;
  CUdeviceptr CUSrcPtr = (CUdeviceptr)SrcPtr;
  CUdeviceptr CUDstPtr = (CUdeviceptr)DstPtr;

  int CanAccessPeer = 0;
  if (DeviceId != DstDeviceId) {
    // Make sure the lock is released before performing the copies.
    std::lock_guard<std::mutex> Lock(PeerAccessesLock);

    switch (PeerAccesses[DstDeviceId]) {
    case PeerAccessState::AVAILABLE:
      CanAccessPeer = 1;
      break;
    case PeerAccessState::UNAVAILABLE:
      CanAccessPeer = 0;
      break;
    case PeerAccessState::PENDING:
      // Check whether the source device can access the destination device.
      Res = cuDeviceCanAccessPeer(&CanAccessPeer, Device, DstDevice.Device);
      if (auto Err = Plugin::check(Res, "Error in cuDeviceCanAccessPeer: %s"))
        return Err;

      if (CanAccessPeer) {
        Res = cuCtxEnablePeerAccess(DstDevice.Context, 0);
        if (Res == CUDA_ERROR_TOO_MANY_PEERS) {
          // Resources may be exhausted due to many P2P links.
          CanAccessPeer = 0;
          DP("Too many P2P so fall back to D2D memcpy");
        } else if (auto Err =
                       Plugin::check(Res, "Error in cuCtxEnablePeerAccess: %s"))
          return Err;
      }
      PeerAccesses[DstDeviceId] = (CanAccessPeer)
                                      ? PeerAccessState::AVAILABLE
                                      : PeerAccessState::UNAVAILABLE;
    }
  }

  CUstream Stream = getStream(AsyncInfoWrapper);
  if (!Stream)
    return Plugin::error("Failure to get stream");

  if (CanAccessPeer) {
    // TODO: Should we fallback to D2D if peer access fails?
    Res = cuMemcpyPeerAsync(CUDstPtr, Context, CUSrcPtr, DstDevice.Context,
                            Size, Stream);
    return Plugin::check(Res, "Error in cuMemcpyPeerAsync: %s");
  }

  // Fallback to D2D copy.
  Res = cuMemcpyDtoDAsync(CUDstPtr, CUSrcPtr, Size, Stream);
  return Plugin::check(Res, "Error in cuMemcpyDtoDAsync: %s");
}

GenericPluginTy *Plugin::createPlugin() { return new CUDAPluginTy(); }

GenericDeviceTy *Plugin::createDevice(int32_t DeviceId, int32_t NumDevices) {
  return new CUDADeviceTy(DeviceId, NumDevices);
}

GenericGlobalHandlerTy *Plugin::createGlobalHandler() {
  return new CUDAGlobalHandlerTy();
}

template <typename... ArgsTy>
Error Plugin::check(int32_t Code, const char *ErrFmt, ArgsTy... Args) {
  CUresult ResultCode = static_cast<CUresult>(Code);
  if (ResultCode == CUDA_SUCCESS)
    return Error::success();

  const char *Desc = "Unknown error";
  CUresult Ret = cuGetErrorString(ResultCode, &Desc);
  if (Ret != CUDA_SUCCESS)
    REPORT("Unrecognized " GETNAME(TARGET_NAME) " error code %d\n", Code);

  return createStringError<ArgsTy..., const char *>(inconvertibleErrorCode(),
                                                    ErrFmt, Args..., Desc);
}

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm
