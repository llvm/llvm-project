//===--- cuda_acxxel.cpp - CUDA implementation of the Acxxel API ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// This file defines the standard CUDA implementation of the Acxxel API.
///
//===----------------------------------------------------------------------===//

#include "acxxel.h"

#include "cuda.h"
#include "cuda_runtime.h"

#include <array>
#include <cassert>
#include <sstream>
#include <vector>

namespace acxxel {

namespace {

static std::string getCUErrorMessage(CUresult Result) {
  if (!Result)
    return "success";
  const char *ErrorName = "UNKNOWN_ERROR_NAME";
  const char *ErrorDescription = "UNKNOWN_ERROR_DESCRIPTION";
  cuGetErrorName(Result, &ErrorName);
  cuGetErrorString(Result, &ErrorDescription);
  std::ostringstream OutStream;
  OutStream << "CUDA driver error: code = " << Result
            << ", name = " << ErrorName
            << ", description = " << ErrorDescription;
  return OutStream.str();
}

static Status getCUError(CUresult Result, const std::string &Message) {
  if (!Result)
    return Status();
  std::ostringstream OutStream;
  OutStream << getCUErrorMessage(Result) << ", message = " << Message;
  return Status(OutStream.str());
}

static std::string getCUDAErrorMessage(cudaError_t E) {
  if (!E)
    return "success";
  std::ostringstream OutStream;
  OutStream << "CUDA runtime error: code = " << E
            << ", name = " << cudaGetErrorName(E)
            << ", description = " << cudaGetErrorString(E);
  return OutStream.str();
}

static Status getCUDAError(cudaError_t E, const std::string &Message) {
  if (!E)
    return Status();
  std::ostringstream OutStream;
  OutStream << getCUDAErrorMessage(E) << ", message = " << Message;
  return Status(OutStream.str());
}

static void logCUWarning(CUresult Result, const std::string &Message) {
  if (Result) {
    std::ostringstream OutStream;
    OutStream << Message << ": " << getCUErrorMessage(Result);
    logWarning(OutStream.str());
  }
}

/// A CUDA Platform implementation.
class CUDAPlatform : public Platform {
public:
  ~CUDAPlatform() override = default;

  static Expected<CUDAPlatform> create();

  Expected<int> getDeviceCount() override;

  Expected<Stream> createStream(int DeviceIndex) override;

  Status streamSync(void *Stream) override;

  Status streamWaitOnEvent(void *Stream, void *Event) override;

  Expected<Event> createEvent(int DeviceIndex) override;

protected:
  Expected<void *> rawMallocD(ptrdiff_t ByteCount, int DeviceIndex) override;
  HandleDestructor getDeviceMemoryHandleDestructor() override;
  void *getDeviceMemorySpanHandle(void *BaseHandle, size_t ByteSize,
                                  size_t ByteOffset) override;
  virtual void rawDestroyDeviceMemorySpanHandle(void *Handle) override;

  Expected<void *> rawGetDeviceSymbolAddress(const void *Symbol,
                                             int DeviceIndex) override;
  Expected<ptrdiff_t> rawGetDeviceSymbolSize(const void *Symbol,
                                             int DeviceIndex) override;

  Status rawRegisterHostMem(const void *Memory, ptrdiff_t ByteCount) override;
  HandleDestructor getUnregisterHostMemoryHandleDestructor() override;

  Expected<void *> rawMallocRegisteredH(ptrdiff_t ByteCount) override;
  HandleDestructor getFreeHostMemoryHandleDestructor() override;

  Status asyncCopyDToD(const void *DeviceSrc, ptrdiff_t DeviceSrcByteOffset,
                       void *DeviceDst, ptrdiff_t DeviceDstByteOffset,
                       ptrdiff_t ByteCount, void *Stream) override;
  Status asyncCopyDToH(const void *DeviceSrc, ptrdiff_t DeviceSrcByteOffset,
                       void *HostDst, ptrdiff_t ByteCount,
                       void *Stream) override;
  Status asyncCopyHToD(const void *HostSrc, void *DeviceDst,
                       ptrdiff_t DeviceDstByteOffset, ptrdiff_t ByteCount,
                       void *Stream) override;

  Status asyncMemsetD(void *DeviceDst, ptrdiff_t ByteOffset,
                      ptrdiff_t ByteCount, char ByteValue,
                      void *Stream) override;

  Status addStreamCallback(Stream &Stream, StreamCallback Callback) override;

  Expected<Program> createProgramFromSource(Span<const char> Source,
                                            int DeviceIndex) override;

  Status enqueueEvent(void *Event, void *Stream) override;
  bool eventIsDone(void *Event) override;
  Status eventSync(void *Event) override;
  Expected<float> getSecondsBetweenEvents(void *StartEvent,
                                          void *EndEvent) override;

  Expected<void *> rawCreateKernel(void *Program,
                                   const std::string &Name) override;
  HandleDestructor getKernelHandleDestructor() override;

  Status rawEnqueueKernelLaunch(void *Stream, void *Kernel,
                                KernelLaunchDimensions LaunchDimensions,
                                Span<void *> Arguments,
                                Span<size_t> ArgumentSizes,
                                size_t SharedMemoryBytes) override;

private:
  explicit CUDAPlatform(const std::vector<CUcontext> &Contexts)
      : TheContexts(Contexts) {}

  Status setContext(int DeviceIndex) {
    if (DeviceIndex < 0 ||
        static_cast<size_t>(DeviceIndex) >= TheContexts.size())
      return Status("invalid deivce index " + std::to_string(DeviceIndex));
    return getCUError(cuCtxSetCurrent(TheContexts[DeviceIndex]),
                      "cuCtxSetCurrent");
  }

  // Vector of contexts for each device.
  std::vector<CUcontext> TheContexts;
};

Expected<CUDAPlatform> CUDAPlatform::create() {
  std::vector<CUcontext> Contexts;
  if (CUresult Result = cuInit(0))
    return getCUError(Result, "cuInit");

  int DeviceCount = 0;
  if (CUresult Result = cuDeviceGetCount(&DeviceCount))
    return getCUError(Result, "cuDeviceGetCount");

  for (int I = 0; I < DeviceCount; ++I) {
    CUdevice Device;
    if (CUresult Result = cuDeviceGet(&Device, I))
      return getCUError(Result, "cuDeviceGet");
    CUcontext Context;
    if (CUresult Result = cuDevicePrimaryCtxRetain(&Context, Device))
      return getCUError(Result, "cuDevicePrimaryCtxRetain");
    if (CUresult Result = cuCtxSetCurrent(Context))
      return getCUError(Result, "cuCtxSetCurrent");
    Contexts.emplace_back(Context);
  }

  return CUDAPlatform(Contexts);
}

Expected<int> CUDAPlatform::getDeviceCount() {
  int Count = 0;
  if (CUresult Result = cuDeviceGetCount(&Count))
    return getCUError(Result, "cuDeviceGetCount");
  return Count;
}

static void cudaDestroyStream(void *H) {
  logCUWarning(cuStreamDestroy(static_cast<CUstream_st *>(H)),
               "cuStreamDestroy");
}

Expected<Stream> CUDAPlatform::createStream(int DeviceIndex) {
  Status S = setContext(DeviceIndex);
  if (S.isError())
    return S;
  unsigned int Flags = CU_STREAM_DEFAULT;
  CUstream Handle;
  if (CUresult Result = cuStreamCreate(&Handle, Flags))
    return getCUError(Result, "cuStreamCreate");
  return constructStream(this, DeviceIndex, Handle, cudaDestroyStream);
}

Status CUDAPlatform::streamSync(void *Stream) {
  return getCUError(cuStreamSynchronize(static_cast<CUstream_st *>(Stream)),
                    "cuStreamSynchronize");
}

Status CUDAPlatform::streamWaitOnEvent(void *Stream, void *Event) {
  // CUDA docs says flags must be 0.
  unsigned int Flags = 0u;
  return getCUError(cuStreamWaitEvent(static_cast<CUstream_st *>(Stream),
                                      static_cast<CUevent_st *>(Event), Flags),
                    "cuStreamWaitEvent");
}

static void cudaDestroyEvent(void *H) {
  logCUWarning(cuEventDestroy(static_cast<CUevent_st *>(H)), "cuEventDestroy");
}

Expected<Event> CUDAPlatform::createEvent(int DeviceIndex) {
  Status S = setContext(DeviceIndex);
  if (S.isError())
    return S;
  unsigned int Flags = CU_EVENT_DEFAULT;
  CUevent Handle;
  if (CUresult Result = cuEventCreate(&Handle, Flags))
    return getCUError(Result, "cuEventCreate");
  return constructEvent(this, DeviceIndex, Handle, cudaDestroyEvent);
}

Status CUDAPlatform::enqueueEvent(void *Event, void *Stream) {
  return getCUError(cuEventRecord(static_cast<CUevent_st *>(Event),
                                  static_cast<CUstream_st *>(Stream)),
                    "cuEventRecord");
}

bool CUDAPlatform::eventIsDone(void *Event) {
  return cuEventQuery(static_cast<CUevent_st *>(Event)) != CUDA_ERROR_NOT_READY;
}

Status CUDAPlatform::eventSync(void *Event) {
  return getCUError(cuEventSynchronize(static_cast<CUevent_st *>(Event)),
                    "cuEventSynchronize");
}

Expected<float> CUDAPlatform::getSecondsBetweenEvents(void *StartEvent,
                                                      void *EndEvent) {
  float Milliseconds;
  if (CUresult Result = cuEventElapsedTime(
          &Milliseconds, static_cast<CUevent_st *>(StartEvent),
          static_cast<CUevent_st *>(EndEvent)))
    return getCUError(Result, "cuEventElapsedTime");
  return Milliseconds * 1e-6;
}

Expected<void *> CUDAPlatform::rawMallocD(ptrdiff_t ByteCount,
                                          int DeviceIndex) {
  Status S = setContext(DeviceIndex);
  if (S.isError())
    return S;
  if (!ByteCount)
    return nullptr;
  CUdeviceptr Pointer;
  if (CUresult Result = cuMemAlloc(&Pointer, ByteCount))
    return getCUError(Result, "cuMemAlloc");
  return reinterpret_cast<void *>(Pointer);
}

static void cudaDestroyDeviceMemory(void *H) {
  logCUWarning(cuMemFree(reinterpret_cast<CUdeviceptr>(H)), "cuMemFree");
}

HandleDestructor CUDAPlatform::getDeviceMemoryHandleDestructor() {
  return cudaDestroyDeviceMemory;
}

void *CUDAPlatform::getDeviceMemorySpanHandle(void *BaseHandle, size_t,
                                              size_t ByteOffset) {
  return static_cast<char *>(BaseHandle) + ByteOffset;
}

void CUDAPlatform::rawDestroyDeviceMemorySpanHandle(void *) {
  // Do nothing for this platform.
}

Expected<void *> CUDAPlatform::rawGetDeviceSymbolAddress(const void *Symbol,
                                                         int DeviceIndex) {
  Status S = setContext(DeviceIndex);
  if (S.isError())
    return S;
  void *Address;
  if (cudaError_t Status = cudaGetSymbolAddress(&Address, Symbol))
    return getCUDAError(Status, "cudaGetSymbolAddress");
  return Address;
}

Expected<ptrdiff_t> CUDAPlatform::rawGetDeviceSymbolSize(const void *Symbol,
                                                         int DeviceIndex) {
  Status S = setContext(DeviceIndex);
  if (S.isError())
    return S;
  size_t Size;
  if (cudaError_t Status = cudaGetSymbolSize(&Size, Symbol))
    return getCUDAError(Status, "cudaGetSymbolSize");
  return Size;
}

static const void *offsetVoidPtr(const void *Ptr, ptrdiff_t ByteOffset) {
  return static_cast<const void *>(static_cast<const char *>(Ptr) + ByteOffset);
}

static void *offsetVoidPtr(void *Ptr, ptrdiff_t ByteOffset) {
  return static_cast<void *>(static_cast<char *>(Ptr) + ByteOffset);
}

Status CUDAPlatform::rawRegisterHostMem(const void *Memory,
                                        ptrdiff_t ByteCount) {
  unsigned int Flags = 0;
  return getCUError(
      cuMemHostRegister(const_cast<void *>(Memory), ByteCount, Flags),
      "cuMemHostRegiser");
}

static void cudaUnregisterHostMemoryHandleDestructor(void *H) {
  logCUWarning(cuMemHostUnregister(H), "cuMemHostUnregister");
}

HandleDestructor CUDAPlatform::getUnregisterHostMemoryHandleDestructor() {
  return cudaUnregisterHostMemoryHandleDestructor;
}

Expected<void *> CUDAPlatform::rawMallocRegisteredH(ptrdiff_t ByteCount) {
  unsigned int Flags = 0;
  void *Memory;
  if (CUresult Result = cuMemHostAlloc(&Memory, ByteCount, Flags))
    return getCUError(Result, "cuMemHostAlloc");
  return Memory;
}

static void cudaFreeHostMemoryHandleDestructor(void *H) {
  logCUWarning(cuMemFreeHost(H), "cuMemFreeHost");
}

HandleDestructor CUDAPlatform::getFreeHostMemoryHandleDestructor() {
  return cudaFreeHostMemoryHandleDestructor;
}

Status CUDAPlatform::asyncCopyDToD(const void *DeviceSrc,
                                   ptrdiff_t DeviceSrcByteOffset,
                                   void *DeviceDst,
                                   ptrdiff_t DeviceDstByteOffset,
                                   ptrdiff_t ByteCount, void *Stream) {
  return getCUError(
      cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(
                            offsetVoidPtr(DeviceDst, DeviceDstByteOffset)),
                        reinterpret_cast<CUdeviceptr>(
                            offsetVoidPtr(DeviceSrc, DeviceSrcByteOffset)),
                        ByteCount, static_cast<CUstream_st *>(Stream)),
      "cuMemcpyDtoDAsync");
}

Status CUDAPlatform::asyncCopyDToH(const void *DeviceSrc,
                                   ptrdiff_t DeviceSrcByteOffset, void *HostDst,
                                   ptrdiff_t ByteCount, void *Stream) {
  return getCUError(
      cuMemcpyDtoHAsync(HostDst, reinterpret_cast<CUdeviceptr>(offsetVoidPtr(
                                     DeviceSrc, DeviceSrcByteOffset)),
                        ByteCount, static_cast<CUstream_st *>(Stream)),
      "cuMemcpyDtoHAsync");
}

Status CUDAPlatform::asyncCopyHToD(const void *HostSrc, void *DeviceDst,
                                   ptrdiff_t DeviceDstByteOffset,
                                   ptrdiff_t ByteCount, void *Stream) {
  return getCUError(
      cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(
                            offsetVoidPtr(DeviceDst, DeviceDstByteOffset)),
                        HostSrc, ByteCount, static_cast<CUstream_st *>(Stream)),
      "cuMemcpyHtoDAsync");
}

Status CUDAPlatform::asyncMemsetD(void *DeviceDst, ptrdiff_t ByteOffset,
                                  ptrdiff_t ByteCount, char ByteValue,
                                  void *Stream) {
  return getCUError(
      cuMemsetD8Async(
          reinterpret_cast<CUdeviceptr>(offsetVoidPtr(DeviceDst, ByteOffset)),
          ByteValue, ByteCount, static_cast<CUstream_st *>(Stream)),
      "cuMemsetD8Async");
}

struct StreamCallbackUserData {
  StreamCallbackUserData(Stream &Stream, StreamCallback Function)
      : TheStream(Stream), TheFunction(std::move(Function)) {}

  Stream &TheStream;
  StreamCallback TheFunction;
};

static void CUDA_CB cuStreamCallbackShim(CUstream HStream, CUresult Status,
                                         void *UserData) {
  std::unique_ptr<StreamCallbackUserData> Data(
      static_cast<StreamCallbackUserData *>(UserData));
  Stream &TheStream = Data->TheStream;
  assert(static_cast<CUstream_st *>(TheStream) == HStream);
  Data->TheFunction(TheStream,
                    getCUError(Status, "stream callback error state"));
}

Status CUDAPlatform::addStreamCallback(Stream &Stream,
                                       StreamCallback Callback) {
  // CUDA docs say flags must always be 0 here.
  unsigned int Flags = 0u;
  std::unique_ptr<StreamCallbackUserData> UserData(
      new StreamCallbackUserData(Stream, std::move(Callback)));
  return getCUError(cuStreamAddCallback(Stream, cuStreamCallbackShim,
                                        UserData.release(), Flags),
                    "cuStreamAddCallback");
}

static void cudaDestroyProgram(void *H) {
  logCUWarning(cuModuleUnload(static_cast<CUmod_st *>(H)), "cuModuleUnload");
}

Expected<Program> CUDAPlatform::createProgramFromSource(Span<const char> Source,
                                                        int DeviceIndex) {
  Status S = setContext(DeviceIndex);
  if (S.isError())
    return S;
  CUmodule Module;
  constexpr int LogBufferSizeBytes = 1024;
  char InfoLogBuffer[LogBufferSizeBytes];
  char ErrorLogBuffer[LogBufferSizeBytes];
  constexpr size_t OptionsCount = 4;
  std::array<CUjit_option, OptionsCount> OptionNames = {
      {CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
       CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES}};
  std::array<void *, OptionsCount> OptionValues = {
      {InfoLogBuffer, const_cast<int *>(&LogBufferSizeBytes), ErrorLogBuffer,
       const_cast<int *>(&LogBufferSizeBytes)}};
  if (CUresult Result =
          cuModuleLoadDataEx(&Module, Source.data(), OptionsCount,
                             OptionNames.data(), OptionValues.data())) {
    InfoLogBuffer[LogBufferSizeBytes - 1] = '\0';
    ErrorLogBuffer[LogBufferSizeBytes - 1] = '\0';
    std::ostringstream OutStream;
    OutStream << "Error creating program from source: "
              << getCUErrorMessage(Result)
              << "\nINFO MESSAGES\n================\n"
              << InfoLogBuffer << "\nERROR MESSAGES\n==================\n"
              << ErrorLogBuffer;
    return Status(OutStream.str());
  }
  return constructProgram(this, Module, cudaDestroyProgram);
}

Expected<void *> CUDAPlatform::rawCreateKernel(void *Program,
                                               const std::string &Name) {
  CUmodule Module = static_cast<CUmodule>(Program);
  CUfunction Kernel;
  if (CUresult Result = cuModuleGetFunction(&Kernel, Module, Name.c_str()))
    return getCUError(Result, "cuModuleGetFunction");
  return Kernel;
}

static void cudaDestroyKernel(void *) {
  // Do nothing.
}

HandleDestructor CUDAPlatform::getKernelHandleDestructor() {
  return cudaDestroyKernel;
}

Status CUDAPlatform::rawEnqueueKernelLaunch(
    void *Stream, void *Kernel, KernelLaunchDimensions LaunchDimensions,
    Span<void *> Arguments, Span<size_t>, size_t SharedMemoryBytes) {
  return getCUError(
      cuLaunchKernel(static_cast<CUfunction>(Kernel), LaunchDimensions.GridX,
                     LaunchDimensions.GridY, LaunchDimensions.GridZ,
                     LaunchDimensions.BlockX, LaunchDimensions.BlockY,
                     LaunchDimensions.BlockZ, SharedMemoryBytes,
                     static_cast<CUstream>(Stream), Arguments.data(), nullptr),
      "cuLaunchKernel");
}

} // namespace

namespace cuda {

/// Gets the CUDAPlatform instance and returns it as an unowned pointer to a
/// Platform.
Expected<Platform *> getPlatform() {
  static auto MaybePlatform = []() -> Expected<CUDAPlatform *> {
    Expected<CUDAPlatform> CreationResult = CUDAPlatform::create();
    if (CreationResult.isError())
      return CreationResult.getError();
    else
      return new CUDAPlatform(CreationResult.takeValue());
  }();
  return MaybePlatform;
}

} // namespace cuda

} // namespace acxxel
