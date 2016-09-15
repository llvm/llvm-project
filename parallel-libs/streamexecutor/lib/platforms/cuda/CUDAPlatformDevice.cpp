//===-- CUDAPlatformDevice.cpp - CUDAPlatformDevice implementation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of CUDAPlatformDevice.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/platforms/cuda/CUDAPlatformDevice.h"
#include "streamexecutor/PlatformDevice.h"

#include "cuda.h"

namespace streamexecutor {
namespace cuda {

static void *offset(const void *Base, size_t Offset) {
  return const_cast<char *>(static_cast<const char *>(Base) + Offset);
}

Error CUresultToError(int CUResult, const llvm::Twine &Message) {
  CUresult Result = static_cast<CUresult>(CUResult);
  if (Result) {
    const char *ErrorName;
    if (cuGetErrorName(Result, &ErrorName))
      ErrorName = "UNKNOWN ERROR NAME";
    const char *ErrorString;
    if (cuGetErrorString(Result, &ErrorString))
      ErrorString = "UNKNOWN ERROR DESCRIPTION";
    return make_error("CUDA driver error: '" + Message + "', error code = " +
                      llvm::Twine(static_cast<int>(Result)) + ", name = " +
                      ErrorName + ", description = '" + ErrorString + "'");
  } else
    return Error::success();
}

std::string CUDAPlatformDevice::getName() const {
  static std::string CachedName = [](int DeviceIndex) {
    static constexpr size_t MAX_DRIVER_NAME_BYTES = 1024;
    std::string Name = "CUDA device " + std::to_string(DeviceIndex);
    char NameFromDriver[MAX_DRIVER_NAME_BYTES];
    if (!cuDeviceGetName(NameFromDriver, MAX_DRIVER_NAME_BYTES - 1,
                         DeviceIndex)) {
      NameFromDriver[MAX_DRIVER_NAME_BYTES - 1] = '\0';
      Name.append(": ").append(NameFromDriver);
    }
    return Name;
  }(DeviceIndex);
  return CachedName;
}

Expected<CUDAPlatformDevice> CUDAPlatformDevice::create(size_t DeviceIndex) {
  CUdevice DeviceHandle;
  if (CUresult Result = cuDeviceGet(&DeviceHandle, DeviceIndex))
    return CUresultToError(Result, "cuDeviceGet");

  CUcontext ContextHandle;
  if (CUresult Result = cuDevicePrimaryCtxRetain(&ContextHandle, DeviceHandle))
    return CUresultToError(Result, "cuDevicePrimaryCtxRetain");

  if (CUresult Result = cuCtxSetCurrent(ContextHandle))
    return CUresultToError(Result, "cuCtxSetCurrent");

  return CUDAPlatformDevice(DeviceIndex);
}

CUDAPlatformDevice::CUDAPlatformDevice(CUDAPlatformDevice &&Other) noexcept
    : DeviceIndex(Other.DeviceIndex) {
  Other.DeviceIndex = -1;
}

CUDAPlatformDevice &CUDAPlatformDevice::
operator=(CUDAPlatformDevice &&Other) noexcept {
  DeviceIndex = Other.DeviceIndex;
  Other.DeviceIndex = -1;
  return *this;
}

CUDAPlatformDevice::~CUDAPlatformDevice() {
  CUresult Result = cuDevicePrimaryCtxRelease(DeviceIndex);
  (void)Result;
  // TODO(jhen): Log error.
}

Expected<const void *>
CUDAPlatformDevice::createKernel(const MultiKernelLoaderSpec &Spec) {
  // TODO(jhen): Maybe first check loaded modules?
  if (!Spec.hasCUDAPTXInMemory())
    return make_error("no CUDA code available to create kernel");

  CUdevice Device = static_cast<int>(DeviceIndex);
  int ComputeCapabilityMajor = 0;
  int ComputeCapabilityMinor = 0;
  if (CUresult Result = cuDeviceGetAttribute(
          &ComputeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
          Device))
    return CUresultToError(
        Result,
        "cuDeviceGetAttribute CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR");
  if (CUresult Result = cuDeviceGetAttribute(
          &ComputeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
          Device))
    return CUresultToError(
        Result,
        "cuDeviceGetAttribute CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR");
  const char *Code = Spec.getCUDAPTXInMemory().getCode(ComputeCapabilityMajor,
                                                       ComputeCapabilityMinor);

  if (!Code)
    return make_error("no suitable CUDA source found for compute capability " +
                      llvm::Twine(ComputeCapabilityMajor) + "." +
                      llvm::Twine(ComputeCapabilityMinor));

  CUmodule Module;
  if (CUresult Result = cuModuleLoadData(&Module, Code))
    return CUresultToError(Result, "cuModuleLoadData");

  CUfunction Function;
  if (CUresult Result =
          cuModuleGetFunction(&Function, Module, Spec.getKernelName().c_str()))
    return CUresultToError(Result, "cuModuleGetFunction");

  // TODO(jhen): Should I save this function pointer in case someone asks for
  // it again?

  // TODO(jhen): Should I save the module pointer so I can unload it when I
  // destroy this device?

  return static_cast<const void *>(Function);
}

Error CUDAPlatformDevice::destroyKernel(const void *Handle) {
  // TODO(jhen): Maybe keep track of kernels for each module and unload the
  // module after they are all destroyed.
  return Error::success();
}

Expected<const void *> CUDAPlatformDevice::createStream() {
  CUstream Stream;
  if (CUresult Result = cuStreamCreate(&Stream, CU_STREAM_DEFAULT))
    return CUresultToError(Result, "cuStreamCreate");
  return Stream;
}

Error CUDAPlatformDevice::destroyStream(const void *Handle) {
  return CUresultToError(
      cuStreamDestroy(static_cast<CUstream>(const_cast<void *>(Handle))),
      "cuStreamDestroy");
}

Error CUDAPlatformDevice::launch(
    const void *PlatformStreamHandle, BlockDimensions BlockSize,
    GridDimensions GridSize, const void *PKernelHandle,
    const PackedKernelArgumentArrayBase &ArgumentArray) {
  CUfunction Function =
      reinterpret_cast<CUfunction>(const_cast<void *>(PKernelHandle));
  CUstream Stream =
      reinterpret_cast<CUstream>(const_cast<void *>(PlatformStreamHandle));
  // TODO(jhen): Deal with shared memory arguments.
  unsigned SharedMemoryBytes = 0;
  void **ArgumentAddresses = const_cast<void **>(ArgumentArray.getAddresses());
  return CUresultToError(cuLaunchKernel(Function, GridSize.X, GridSize.Y,
                                        GridSize.Z, BlockSize.X, BlockSize.Y,
                                        BlockSize.Z, SharedMemoryBytes, Stream,
                                        ArgumentAddresses, nullptr),
                         "cuLaunchKernel");
}

Error CUDAPlatformDevice::copyD2H(const void *PlatformStreamHandle,
                                  const void *DeviceSrcHandle,
                                  size_t SrcByteOffset, void *HostDst,
                                  size_t DstByteOffset, size_t ByteCount) {
  return CUresultToError(
      cuMemcpyDtoHAsync(
          offset(HostDst, DstByteOffset),
          reinterpret_cast<CUdeviceptr>(offset(DeviceSrcHandle, SrcByteOffset)),
          ByteCount,
          static_cast<CUstream>(const_cast<void *>(PlatformStreamHandle))),
      "cuMemcpyDtoHAsync");
}

Error CUDAPlatformDevice::copyH2D(const void *PlatformStreamHandle,
                                  const void *HostSrc, size_t SrcByteOffset,
                                  const void *DeviceDstHandle,
                                  size_t DstByteOffset, size_t ByteCount) {
  return CUresultToError(
      cuMemcpyHtoDAsync(
          reinterpret_cast<CUdeviceptr>(offset(DeviceDstHandle, DstByteOffset)),
          offset(HostSrc, SrcByteOffset), ByteCount,
          static_cast<CUstream>(const_cast<void *>(PlatformStreamHandle))),
      "cuMemcpyHtoDAsync");
}

Error CUDAPlatformDevice::copyD2D(const void *PlatformStreamHandle,
                                  const void *DeviceSrcHandle,
                                  size_t SrcByteOffset,
                                  const void *DeviceDstHandle,
                                  size_t DstByteOffset, size_t ByteCount) {
  return CUresultToError(
      cuMemcpyDtoDAsync(
          reinterpret_cast<CUdeviceptr>(offset(DeviceDstHandle, DstByteOffset)),
          reinterpret_cast<CUdeviceptr>(offset(DeviceSrcHandle, SrcByteOffset)),
          ByteCount,
          static_cast<CUstream>(const_cast<void *>(PlatformStreamHandle))),
      "cuMemcpyDtoDAsync");
}

Error CUDAPlatformDevice::blockHostUntilDone(const void *PlatformStreamHandle) {
  return CUresultToError(cuStreamSynchronize(static_cast<CUstream>(
                             const_cast<void *>(PlatformStreamHandle))),
                         "cuStreamSynchronize");
}

Expected<void *> CUDAPlatformDevice::allocateDeviceMemory(size_t ByteCount) {
  CUdeviceptr Pointer;
  if (CUresult Result = cuMemAlloc(&Pointer, ByteCount))
    return CUresultToError(Result, "cuMemAlloc");
  return reinterpret_cast<void *>(Pointer);
}

Error CUDAPlatformDevice::freeDeviceMemory(const void *Handle) {
  return CUresultToError(cuMemFree(reinterpret_cast<CUdeviceptr>(Handle)),
                         "cuMemFree");
}

Error CUDAPlatformDevice::registerHostMemory(void *Memory, size_t ByteCount) {
  return CUresultToError(cuMemHostRegister(Memory, ByteCount, 0u),
                         "cuMemHostRegister");
}

Error CUDAPlatformDevice::unregisterHostMemory(const void *Memory) {
  return CUresultToError(cuMemHostUnregister(const_cast<void *>(Memory)),
                         "cuMemHostUnregister");
}

Error CUDAPlatformDevice::synchronousCopyD2H(const void *DeviceSrcHandle,
                                             size_t SrcByteOffset,
                                             void *HostDst,
                                             size_t DstByteOffset,
                                             size_t ByteCount) {
  return CUresultToError(cuMemcpyDtoH(offset(HostDst, DstByteOffset),
                                      reinterpret_cast<CUdeviceptr>(offset(
                                          DeviceSrcHandle, SrcByteOffset)),
                                      ByteCount),
                         "cuMemcpyDtoH");
}

Error CUDAPlatformDevice::synchronousCopyH2D(const void *HostSrc,
                                             size_t SrcByteOffset,
                                             const void *DeviceDstHandle,
                                             size_t DstByteOffset,
                                             size_t ByteCount) {
  return CUresultToError(
      cuMemcpyHtoD(
          reinterpret_cast<CUdeviceptr>(offset(DeviceDstHandle, DstByteOffset)),
          offset(HostSrc, SrcByteOffset), ByteCount),
      "cuMemcpyHtoD");
}

Error CUDAPlatformDevice::synchronousCopyD2D(const void *DeviceDstHandle,
                                             size_t DstByteOffset,
                                             const void *DeviceSrcHandle,
                                             size_t SrcByteOffset,
                                             size_t ByteCount) {
  return CUresultToError(
      cuMemcpyDtoD(
          reinterpret_cast<CUdeviceptr>(offset(DeviceDstHandle, DstByteOffset)),
          reinterpret_cast<CUdeviceptr>(offset(DeviceSrcHandle, SrcByteOffset)),
          ByteCount),
      "cuMemcpyDtoD");
}

} // namespace cuda
} // namespace streamexecutor
