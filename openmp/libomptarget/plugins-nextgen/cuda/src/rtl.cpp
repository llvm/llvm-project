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

#include "Shared/Debug.h"
#include "Shared/Environment.h"

#include "GlobalHandler.h"
#include "OpenMP/OMPT/Callback.h"
#include "PluginInterface.h"
#include "Utils/ELF.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// Forward declarations for all specialized data structures.
struct CUDAKernelTy;
struct CUDADeviceTy;
struct CUDAPluginTy;

#if (defined(CUDA_VERSION) && (CUDA_VERSION < 11000))
/// Forward declarations for all Virtual Memory Management
/// related data structures and functions. This is necessary
/// for older cuda versions.
typedef void *CUmemGenericAllocationHandle;
typedef void *CUmemAllocationProp;
typedef void *CUmemAccessDesc;
typedef void *CUmemAllocationGranularity_flags;
CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                             CUdeviceptr addr, unsigned long long flags) {}
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle,
                  unsigned long long flags) {}
CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop,
                     unsigned long long flags) {}
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc *desc, size_t count) {}
CUresult
cuMemGetAllocationGranularity(size_t *granularity,
                              const CUmemAllocationProp *prop,
                              CUmemAllocationGranularity_flags option) {}
#endif

#if (defined(CUDA_VERSION) && (CUDA_VERSION < 11020))
// Forward declarations of asynchronous memory management functions. This is
// necessary for older versions of CUDA.
CUresult cuMemAllocAsync(CUdeviceptr *ptr, size_t, CUstream) { *ptr = 0; }

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {}
#endif

/// Class implementing the CUDA device images properties.
struct CUDADeviceImageTy : public DeviceImageTy {
  /// Create the CUDA image with the id and the target image pointer.
  CUDADeviceImageTy(int32_t ImageId, GenericDeviceTy &Device,
                    const __tgt_device_image *TgtImage)
      : DeviceImageTy(ImageId, Device, TgtImage), Module(nullptr) {}

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

/// Class implementing the CUDA kernel functionalities which derives from the
/// generic kernel class.
struct CUDAKernelTy : public GenericKernelTy {
  /// Create a CUDA kernel with a name and an execution mode.
  CUDAKernelTy(const char *Name) : GenericKernelTy(Name), Func(nullptr) {}

  /// Initialize the CUDA kernel.
  Error initImpl(GenericDeviceTy &GenericDevice,
                 DeviceImageTy &Image) override {
    CUresult Res;
    CUDADeviceImageTy &CUDAImage = static_cast<CUDADeviceImageTy &>(Image);

    // Retrieve the function pointer of the kernel.
    Res = cuModuleGetFunction(&Func, CUDAImage.getModule(), getName());
    if (auto Err = Plugin::check(Res, "Error in cuModuleGetFunction('%s'): %s",
                                 getName()))
      return Err;

    // Check that the function pointer is valid.
    if (!Func)
      return Plugin::error("Invalid function for kernel %s", getName());

    int MaxThreads;
    Res = cuFuncGetAttribute(&MaxThreads,
                             CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, Func);
    if (auto Err = Plugin::check(Res, "Error in cuFuncGetAttribute: %s"))
      return Err;

    // The maximum number of threads cannot exceed the maximum of the kernel.
    MaxNumThreads = std::min(MaxNumThreads, (uint32_t)MaxThreads);

    return Plugin::success();
  }

  /// Launch the CUDA kernel function.
  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads,
                   uint64_t NumBlocks, KernelArgsTy &KernelArgs, void *Args,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override;

private:
  /// The CUDA kernel function to execute.
  CUfunction Func;
};

/// Class wrapping a CUDA stream reference. These are the objects handled by the
/// Stream Manager for the CUDA plugin.
struct CUDAStreamRef final : public GenericDeviceResourceRef {
  /// The underlying handle type for streams.
  using HandleTy = CUstream;

  /// Create an empty reference to an invalid stream.
  CUDAStreamRef() : Stream(nullptr) {}

  /// Create a reference to an existing stream.
  CUDAStreamRef(HandleTy Stream) : Stream(Stream) {}

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

  /// Get the underlying CUDA stream.
  operator HandleTy() const { return Stream; }

private:
  /// The reference to the CUDA stream.
  HandleTy Stream;
};

/// Class wrapping a CUDA event reference. These are the objects handled by the
/// Event Manager for the CUDA plugin.
struct CUDAEventRef final : public GenericDeviceResourceRef {
  /// The underlying handle type for events.
  using HandleTy = CUevent;

  /// Create an empty reference to an invalid event.
  CUDAEventRef() : Event(nullptr) {}

  /// Create a reference to an existing event.
  CUDAEventRef(HandleTy Event) : Event(Event) {}

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
  operator HandleTy() const { return Event; }

private:
  /// The reference to the CUDA event.
  HandleTy Event;
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

    uint32_t NumMuliprocessors = 0;
    uint32_t MaxThreadsPerSM = 0;
    uint32_t WarpSize = 0;
    if (auto Err = getDeviceAttr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                 NumMuliprocessors))
      return Err;
    if (auto Err =
            getDeviceAttr(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                          MaxThreadsPerSM))
      return Err;
    if (auto Err = getDeviceAttr(CU_DEVICE_ATTRIBUTE_WARP_SIZE, WarpSize))
      return Err;
    HardwareParallelism = NumMuliprocessors * (MaxThreadsPerSM / WarpSize);

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

  virtual Error callGlobalConstructors(GenericPluginTy &Plugin,
                                       DeviceImageTy &Image) override {
    // Check for the presense of global destructors at initialization time. This
    // is required when the image may be deallocated before destructors are run.
    GenericGlobalHandlerTy &Handler = Plugin.getGlobalHandler();
    if (Handler.isSymbolInImage(*this, Image, "nvptx$device$fini"))
      Image.setPendingGlobalDtors();

    return callGlobalCtorDtorCommon(Plugin, Image, /*IsCtor=*/true);
  }

  virtual Error callGlobalDestructors(GenericPluginTy &Plugin,
                                      DeviceImageTy &Image) override {
    if (Image.hasPendingGlobalDtors())
      return callGlobalCtorDtorCommon(Plugin, Image, /*IsCtor=*/false);
    return Plugin::success();
  }

  Expected<std::unique_ptr<MemoryBuffer>>
  doJITPostProcessing(std::unique_ptr<MemoryBuffer> MB) const override {
    // TODO: We should be able to use the 'nvidia-ptxjitcompiler' interface to
    //       avoid the call to 'ptxas'.
    SmallString<128> PTXInputFilePath;
    std::error_code EC = sys::fs::createTemporaryFile("nvptx-pre-link-jit", "s",
                                                      PTXInputFilePath);
    if (EC)
      return Plugin::error("Failed to create temporary file for ptxas");

    // Write the file's contents to the output file.
    Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
        FileOutputBuffer::create(PTXInputFilePath, MB->getBuffer().size());
    if (!OutputOrErr)
      return OutputOrErr.takeError();
    std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
    llvm::copy(MB->getBuffer(), Output->getBufferStart());
    if (Error E = Output->commit())
      return std::move(E);

    SmallString<128> PTXOutputFilePath;
    EC = sys::fs::createTemporaryFile("nvptx-post-link-jit", "cubin",
                                      PTXOutputFilePath);
    if (EC)
      return Plugin::error("Failed to create temporary file for ptxas");

    // Try to find `ptxas` in the path to compile the PTX to a binary.
    const auto ErrorOrPath = sys::findProgramByName("ptxas");
    if (!ErrorOrPath)
      return Plugin::error("Failed to find 'ptxas' on the PATH.");

    std::string Arch = getComputeUnitKind();
    StringRef Args[] = {*ErrorOrPath,
                        "-m64",
                        "-O2",
                        "--gpu-name",
                        Arch,
                        "--output-file",
                        PTXOutputFilePath,
                        PTXInputFilePath};

    std::string ErrMsg;
    if (sys::ExecuteAndWait(*ErrorOrPath, Args, std::nullopt, {}, 0, 0,
                            &ErrMsg))
      return Plugin::error("Running 'ptxas' failed: %s\n", ErrMsg.c_str());

    auto BufferOrErr = MemoryBuffer::getFileOrSTDIN(PTXOutputFilePath.data());
    if (!BufferOrErr)
      return Plugin::error("Failed to open temporary file for ptxas");

    // Clean up the temporary files afterwards.
    if (sys::fs::remove(PTXOutputFilePath))
      return Plugin::error("Failed to remove temporary file for ptxas");
    if (sys::fs::remove(PTXInputFilePath))
      return Plugin::error("Failed to remove temporary file for ptxas");

    return std::move(*BufferOrErr);
  }

  /// Allocate and construct a CUDA kernel.
  Expected<GenericKernelTy &> constructKernel(const char *Name) override {
    // Allocate and construct the CUDA kernel.
    CUDAKernelTy *CUDAKernel = Plugin::get().allocate<CUDAKernelTy>();
    if (!CUDAKernel)
      return Plugin::error("Failed to allocate memory for CUDA kernel");

    new (CUDAKernel) CUDAKernelTy(Name);

    return *CUDAKernel;
  }

  /// Set the current context to this device's context.
  Error setContext() override {
    CUresult Res = cuCtxSetCurrent(Context);
    return Plugin::check(Res, "Error in cuCtxSetCurrent: %s");
  }

  /// NVIDIA returns the product of the SM count and the number of warps that
  /// fit if the maximum number of threads were scheduled on each SM.
  uint64_t getHardwareParallelism() const override {
    return HardwareParallelism;
  }

  /// We want to set up the RPC server for host services to the GPU if it is
  /// availible.
  bool shouldSetupRPCServer() const override {
    return libomptargetSupportsRPC();
  }

  /// The RPC interface should have enough space for all availible parallelism.
  uint64_t requestedRPCPortCount() const override {
    return getHardwareParallelism();
  }

  /// Get the stream of the asynchronous info sructure or get a new one.
  Error getStream(AsyncInfoWrapperTy &AsyncInfoWrapper, CUstream &Stream) {
    // Get the stream (if any) from the async info.
    Stream = AsyncInfoWrapper.getQueueAs<CUstream>();
    if (!Stream) {
      // There was no stream; get an idle one.
      if (auto Err = CUDAStreamManager.getResource(Stream))
        return Err;

      // Modify the async info's stream.
      AsyncInfoWrapper.setQueueAs<CUstream>(Stream);
    }
    return Plugin::success();
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
    new (CUDAImage) CUDADeviceImageTy(ImageId, *this, TgtImage);

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
    case TARGET_ALLOC_DEVICE_NON_BLOCKING: {
      CUstream Stream;
      if ((Res = cuStreamCreate(&Stream, CU_STREAM_NON_BLOCKING)))
        break;
      if ((Res = cuMemAllocAsync(&DevicePtr, Size, Stream)))
        break;
      cuStreamSynchronize(Stream);
      Res = cuStreamDestroy(Stream);
      MemAlloc = (void *)DevicePtr;
    }
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
    case TARGET_ALLOC_DEVICE_NON_BLOCKING: {
      CUstream Stream;
      if ((Res = cuStreamCreate(&Stream, CU_STREAM_NON_BLOCKING)))
        break;
      cuMemFreeAsync(reinterpret_cast<CUdeviceptr>(TgtPtr), Stream);
      cuStreamSynchronize(Stream);
      if ((Res = cuStreamDestroy(Stream)))
        break;
    }
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
    AsyncInfo.Queue = nullptr;
    if (auto Err = CUDAStreamManager.returnResource(Stream))
      return Err;

    return Plugin::check(Res, "Error in cuStreamSynchronize: %s");
  }

  /// CUDA support VA management
  bool supportVAManagement() const override {
#if (defined(CUDA_VERSION) && (CUDA_VERSION >= 11000))
    return true;
#else
    return false;
#endif
  }

  /// Allocates \p RSize bytes (rounded up to page size) and hints the cuda
  /// driver to map it to \p VAddr. The obtained address is stored in \p Addr.
  /// At return \p RSize contains the actual size
  Error memoryVAMap(void **Addr, void *VAddr, size_t *RSize) override {
    CUdeviceptr DVAddr = reinterpret_cast<CUdeviceptr>(VAddr);
    auto IHandle = DeviceMMaps.find(DVAddr);
    size_t Size = *RSize;

    if (Size == 0)
      return Plugin::error("Memory Map Size must be larger than 0");

    // Check if we have already mapped this address
    if (IHandle != DeviceMMaps.end())
      return Plugin::error("Address already memory mapped");

    CUmemAllocationProp Prop = {};
    size_t Granularity = 0;

    size_t Free, Total;
    CUresult Res = cuMemGetInfo(&Free, &Total);
    if (auto Err = Plugin::check(Res, "Error in cuMemGetInfo: %s"))
      return Err;

    if (Size >= Free) {
      *Addr = nullptr;
      return Plugin::error(
          "Canot map memory size larger than the available device memory");
    }

    // currently NVidia only supports pinned device types
    Prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    Prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;

    Prop.location.id = DeviceId;
    cuMemGetAllocationGranularity(&Granularity, &Prop,
                                  CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    if (auto Err =
            Plugin::check(Res, "Error in cuMemGetAllocationGranularity: %s"))
      return Err;

    if (Granularity == 0)
      return Plugin::error("Wrong device Page size");

    // Ceil to page size.
    Size = roundUp(Size, Granularity);

    // Create a handler of our allocation
    CUmemGenericAllocationHandle AHandle;
    Res = cuMemCreate(&AHandle, Size, &Prop, 0);
    if (auto Err = Plugin::check(Res, "Error in cuMemCreate: %s"))
      return Err;

    CUdeviceptr DevPtr = 0;
    Res = cuMemAddressReserve(&DevPtr, Size, 0, DVAddr, 0);
    if (auto Err = Plugin::check(Res, "Error in cuMemAddressReserve: %s"))
      return Err;

    Res = cuMemMap(DevPtr, Size, 0, AHandle, 0);
    if (auto Err = Plugin::check(Res, "Error in cuMemMap: %s"))
      return Err;

    CUmemAccessDesc ADesc = {};
    ADesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ADesc.location.id = DeviceId;
    ADesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Sets address
    Res = cuMemSetAccess(DevPtr, Size, &ADesc, 1);
    if (auto Err = Plugin::check(Res, "Error in cuMemSetAccess: %s"))
      return Err;

    *Addr = reinterpret_cast<void *>(DevPtr);
    *RSize = Size;
    DeviceMMaps.insert({DevPtr, AHandle});
    return Plugin::success();
  }

  /// De-allocates device memory and Unmaps the Virtual Addr
  Error memoryVAUnMap(void *VAddr, size_t Size) override {
    CUdeviceptr DVAddr = reinterpret_cast<CUdeviceptr>(VAddr);
    auto IHandle = DeviceMMaps.find(DVAddr);
    // Mapping does not exist
    if (IHandle == DeviceMMaps.end()) {
      return Plugin::error("Addr is not MemoryMapped");
    }

    if (IHandle == DeviceMMaps.end())
      return Plugin::error("Addr is not MemoryMapped");

    CUmemGenericAllocationHandle &AllocHandle = IHandle->second;

    CUresult Res = cuMemUnmap(DVAddr, Size);
    if (auto Err = Plugin::check(Res, "Error in cuMemUnmap: %s"))
      return Err;

    Res = cuMemRelease(AllocHandle);
    if (auto Err = Plugin::check(Res, "Error in cuMemRelease: %s"))
      return Err;

    Res = cuMemAddressFree(DVAddr, Size);
    if (auto Err = Plugin::check(Res, "Error in cuMemAddressFree: %s"))
      return Err;

    DeviceMMaps.erase(IHandle);
    return Plugin::success();
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
    AsyncInfo.Queue = nullptr;
    if (auto Err = CUDAStreamManager.returnResource(Stream))
      return Err;

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

    CUstream Stream;
    if (auto Err = getStream(AsyncInfoWrapper, Stream))
      return Err;

    CUresult Res = cuMemcpyHtoDAsync((CUdeviceptr)TgtPtr, HstPtr, Size, Stream);
    return Plugin::check(Res, "Error in cuMemcpyHtoDAsync: %s");
  }

  /// Retrieve data from the device (device to host transfer).
  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    if (auto Err = setContext())
      return Err;

    CUstream Stream;
    if (auto Err = getStream(AsyncInfoWrapper, Stream))
      return Err;

    // If there is already pending work on the stream it could be waiting for
    // someone to check the RPC server.
    if (auto *RPCServer = getRPCServer()) {
      CUresult Res = cuStreamQuery(Stream);
      while (Res == CUDA_ERROR_NOT_READY) {
        if (auto Err = RPCServer->runServer(*this))
          return Err;
        Res = cuStreamQuery(Stream);
      }
    }

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

    CUstream Stream;
    if (auto Err = getStream(AsyncInfoWrapper, Stream))
      return Err;

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
    return CUDAEventManager.getResource(*Event);
  }

  /// Destroy a previously created event.
  Error destroyEventImpl(void *EventPtr) override {
    CUevent Event = reinterpret_cast<CUevent>(EventPtr);
    return CUDAEventManager.returnResource(Event);
  }

  /// Record the event.
  Error recordEventImpl(void *EventPtr,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    CUevent Event = reinterpret_cast<CUevent>(EventPtr);

    CUstream Stream;
    if (auto Err = getStream(AsyncInfoWrapper, Stream))
      return Err;

    CUresult Res = cuEventRecord(Event, Stream);
    return Plugin::check(Res, "Error in cuEventRecord: %s");
  }

  /// Make the stream wait on the event.
  Error waitEventImpl(void *EventPtr,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    CUevent Event = reinterpret_cast<CUevent>(EventPtr);

    CUstream Stream;
    if (auto Err = getStream(AsyncInfoWrapper, Stream))
      return Err;

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

  virtual bool shouldSetupDeviceMemoryPool() const override {
    /// We use the CUDA malloc for now.
    return false;
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
  Error getDeviceMemorySize(uint64_t &Value) override {
    CUresult Res = cuDeviceTotalMem(&Value, Device);
    return Plugin::check(Res, "Error in getDeviceMemorySize %s");
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

  Error callGlobalCtorDtorCommon(GenericPluginTy &Plugin, DeviceImageTy &Image,
                                 bool IsCtor) {
    const char *KernelName = IsCtor ? "nvptx$device$init" : "nvptx$device$fini";
    // Perform a quick check for the named kernel in the image. The kernel
    // should be created by the 'nvptx-lower-ctor-dtor' pass.
    GenericGlobalHandlerTy &Handler = Plugin.getGlobalHandler();
    if (IsCtor && !Handler.isSymbolInImage(*this, Image, KernelName))
      return Plugin::success();

    // The Nvidia backend cannot handle creating the ctor / dtor array
    // automatically so we must create it ourselves. The backend will emit
    // several globals that contain function pointers we can call. These are
    // prefixed with a known name due to Nvidia's lack of section support.
    auto ELFObjOrErr = Handler.getELFObjectFile(Image);
    if (!ELFObjOrErr)
      return ELFObjOrErr.takeError();

    // Search for all symbols that contain a constructor or destructor.
    SmallVector<std::pair<StringRef, uint16_t>> Funcs;
    for (ELFSymbolRef Sym : ELFObjOrErr->symbols()) {
      auto NameOrErr = Sym.getName();
      if (!NameOrErr)
        return NameOrErr.takeError();

      if (!NameOrErr->starts_with(IsCtor ? "__init_array_object_"
                                         : "__fini_array_object_"))
        continue;

      uint16_t Priority;
      if (NameOrErr->rsplit('_').second.getAsInteger(10, Priority))
        return Plugin::error("Invalid priority for constructor or destructor");

      Funcs.emplace_back(*NameOrErr, Priority);
    }

    // Sort the created array to be in priority order.
    llvm::sort(Funcs, [=](auto X, auto Y) { return X.second < Y.second; });

    // Allocate a buffer to store all of the known constructor / destructor
    // functions in so we can iterate them on the device.
    void *Buffer =
        allocate(Funcs.size() * sizeof(void *), nullptr, TARGET_ALLOC_DEVICE);
    if (!Buffer)
      return Plugin::error("Failed to allocate memory for global buffer");

    auto *GlobalPtrStart = reinterpret_cast<uintptr_t *>(Buffer);
    auto *GlobalPtrStop = reinterpret_cast<uintptr_t *>(Buffer) + Funcs.size();

    SmallVector<void *> FunctionPtrs(Funcs.size());
    std::size_t Idx = 0;
    for (auto [Name, Priority] : Funcs) {
      GlobalTy FunctionAddr(Name.str(), sizeof(void *), &FunctionPtrs[Idx++]);
      if (auto Err = Handler.readGlobalFromDevice(*this, Image, FunctionAddr))
        return Err;
    }

    // Copy the local buffer to the device.
    if (auto Err = dataSubmit(GlobalPtrStart, FunctionPtrs.data(),
                              FunctionPtrs.size() * sizeof(void *), nullptr))
      return Err;

    // Copy the created buffer to the appropriate symbols so the kernel can
    // iterate through them.
    GlobalTy StartGlobal(IsCtor ? "__init_array_start" : "__fini_array_start",
                         sizeof(void *), &GlobalPtrStart);
    if (auto Err = Handler.writeGlobalToDevice(*this, Image, StartGlobal))
      return Err;

    GlobalTy StopGlobal(IsCtor ? "__init_array_end" : "__fini_array_end",
                        sizeof(void *), &GlobalPtrStop);
    if (auto Err = Handler.writeGlobalToDevice(*this, Image, StopGlobal))
      return Err;

    CUDAKernelTy CUDAKernel(KernelName);

    if (auto Err = CUDAKernel.init(*this, Image))
      return Err;

    AsyncInfoWrapperTy AsyncInfoWrapper(*this, nullptr);

    KernelArgsTy KernelArgs = {};
    if (auto Err = CUDAKernel.launchImpl(*this, /*NumThread=*/1u,
                                         /*NumBlocks=*/1ul, KernelArgs, nullptr,
                                         AsyncInfoWrapper))
      return Err;

    Error Err = Plugin::success();
    AsyncInfoWrapper.finalize(Err);

    if (free(Buffer, TARGET_ALLOC_DEVICE) != OFFLOAD_SUCCESS)
      return Plugin::error("Failed to free memory for global buffer");

    return Err;
  }

  /// Stream manager for CUDA streams.
  CUDAStreamManagerTy CUDAStreamManager;

  /// Event manager for CUDA events.
  CUDAEventManagerTy CUDAEventManager;

  /// The device's context. This context should be set before performing
  /// operations on the device.
  CUcontext Context = nullptr;

  /// The CUDA device handler.
  CUdevice Device = CU_DEVICE_INVALID;

  /// The memory mapped addresses and their handles
  std::unordered_map<CUdeviceptr, CUmemGenericAllocationHandle> DeviceMMaps;

  /// The compute capability of the corresponding CUDA device.
  struct ComputeCapabilityTy {
    uint32_t Major;
    uint32_t Minor;
    std::string str() const {
      return "sm_" + std::to_string(Major * 10 + Minor);
    }
  } ComputeCapability;

  /// The maximum number of warps that can be resident on all the SMs
  /// simultaneously.
  uint32_t HardwareParallelism = 0;
};

Error CUDAKernelTy::launchImpl(GenericDeviceTy &GenericDevice,
                               uint32_t NumThreads, uint64_t NumBlocks,
                               KernelArgsTy &KernelArgs, void *Args,
                               AsyncInfoWrapperTy &AsyncInfoWrapper) const {
  CUDADeviceTy &CUDADevice = static_cast<CUDADeviceTy &>(GenericDevice);

  CUstream Stream;
  if (auto Err = CUDADevice.getStream(AsyncInfoWrapper, Stream))
    return Err;

  uint32_t MaxDynCGroupMem =
      std::max(KernelArgs.DynCGroupMem, GenericDevice.getDynamicMemorySize());

  CUresult Res =
      cuLaunchKernel(Func, NumBlocks, /*gridDimY=*/1,
                     /*gridDimZ=*/1, NumThreads,
                     /*blockDimY=*/1, /*blockDimZ=*/1, MaxDynCGroupMem, Stream,
                     (void **)Args, nullptr);
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
  Expected<bool> isELFCompatible(StringRef Image) const override {
    auto ElfOrErr =
        ELF64LEObjectFile::create(MemoryBufferRef(Image, /*Identifier=*/""),
                                  /*InitContent=*/false);
    if (!ElfOrErr)
      return ElfOrErr.takeError();

    // Get the numeric value for the image's `sm_` value.
    auto SM = ElfOrErr->getPlatformFlags() & ELF::EF_CUDA_SM;

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

      int32_t ImageMajor = SM / 10;
      int32_t ImageMinor = SM % 10;

      // A cubin generated for a certain compute capability is supported to
      // run on any GPU with the same major revision and same or higher minor
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

  CUstream Stream;
  if (auto Err = getStream(AsyncInfoWrapper, Stream))
    return Err;

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
