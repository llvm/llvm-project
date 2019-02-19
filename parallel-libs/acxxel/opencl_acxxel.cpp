//===--- opencl_acxxel.cpp - OpenCL implementation of the Acxxel API ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines the standard OpenCL implementation of the Acxxel API.
///
//===----------------------------------------------------------------------===//

#include "acxxel.h"

#include "CL/cl.h"

#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

namespace acxxel {

namespace {

/// An ID containing the platform ID and the device ID within the platform.
struct FullDeviceID {
  cl_platform_id PlatformID;
  cl_device_id DeviceID;

  FullDeviceID(cl_platform_id PlatformID, cl_device_id DeviceID)
      : PlatformID(PlatformID), DeviceID(DeviceID) {}
};

static std::string getOpenCLErrorMessage(cl_int Result) {
  if (!Result)
    return "success";
  std::ostringstream OutStream;
  OutStream << "OpenCL error: code = " << Result;
  return OutStream.str();
}

static Status getOpenCLError(cl_int Result, const std::string &Message) {
  if (!Result)
    return Status();
  std::ostringstream OutStream;
  OutStream << getOpenCLErrorMessage(Result) << ", message = " << Message;
  return Status(OutStream.str());
}

static void logOpenCLWarning(cl_int Result, const std::string &Message) {
  if (Result) {
    std::ostringstream OutStream;
    OutStream << Message << ": " << getOpenCLErrorMessage(Result);
    logWarning(OutStream.str());
  }
}

class OpenCLPlatform : public Platform {
public:
  ~OpenCLPlatform() override = default;

  static Expected<OpenCLPlatform> create();

  Expected<int> getDeviceCount() override;

  Expected<Stream> createStream(int DeviceIndex) override;

  Expected<Event> createEvent(int DeviceIndex) override;

  Expected<Program> createProgramFromSource(Span<const char> Source,
                                            int DeviceIndex) override;

protected:
  Status streamSync(void *Stream) override;

  Status streamWaitOnEvent(void *Stream, void *Event) override;

  Expected<void *> rawMallocD(ptrdiff_t ByteCount, int DeviceIndex) override;
  HandleDestructor getDeviceMemoryHandleDestructor() override;
  void *getDeviceMemorySpanHandle(void *BaseHandle, size_t ByteSize,
                                  size_t ByteOffset) override;
  void rawDestroyDeviceMemorySpanHandle(void *Handle) override;

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
  OpenCLPlatform(std::vector<FullDeviceID> &&FullDeviceIDs,
                 std::vector<cl_context> &&Contexts,
                 std::vector<cl_command_queue> &&CommandQueues)
      : FullDeviceIDs(std::move(FullDeviceIDs)), Contexts(std::move(Contexts)),
        CommandQueues(std::move(CommandQueues)) {}

  std::vector<FullDeviceID> FullDeviceIDs;
  std::vector<cl_context> Contexts;
  std::vector<cl_command_queue> CommandQueues;
};

Expected<OpenCLPlatform> OpenCLPlatform::create() {
  constexpr cl_uint MaxNumEntries = 100;
  cl_platform_id Platforms[MaxNumEntries];
  cl_uint NumPlatforms;
  if (cl_int Result = clGetPlatformIDs(MaxNumEntries, Platforms, &NumPlatforms))
    return getOpenCLError(Result, "clGetPlatformIDs");

  std::vector<FullDeviceID> FullDeviceIDs;
  for (cl_uint PlatformIndex = 0; PlatformIndex < NumPlatforms;
       ++PlatformIndex) {
    cl_uint NumDevices;
    cl_device_id Devices[MaxNumEntries];
    if (cl_int Result =
            clGetDeviceIDs(Platforms[PlatformIndex], CL_DEVICE_TYPE_ALL,
                           MaxNumEntries, Devices, &NumDevices))
      return getOpenCLError(Result, "clGetDeviceIDs");
    for (cl_uint DeviceIndex = 0; DeviceIndex < NumDevices; ++DeviceIndex)
      FullDeviceIDs.emplace_back(Platforms[PlatformIndex],
                                 Devices[DeviceIndex]);
  }

  if (FullDeviceIDs.empty())
    return Status("No OpenCL device available on this system.");

  std::vector<cl_context> Contexts(FullDeviceIDs.size());
  std::vector<cl_command_queue> CommandQueues(FullDeviceIDs.size());
  for (size_t I = 0; I < FullDeviceIDs.size(); ++I) {
    cl_int CreateContextResult;
    Contexts[I] = clCreateContext(nullptr, 1, &FullDeviceIDs[I].DeviceID,
                                  nullptr, nullptr, &CreateContextResult);
    if (CreateContextResult)
      return getOpenCLError(CreateContextResult, "clCreateContext");

    cl_int CreateCommandQueueResult;
    CommandQueues[I] = clCreateCommandQueue(
        Contexts[I], FullDeviceIDs[I].DeviceID, CL_QUEUE_PROFILING_ENABLE,
        &CreateCommandQueueResult);
    if (CreateCommandQueueResult)
      return getOpenCLError(CreateCommandQueueResult, "clCreateCommandQueue");
  }

  return OpenCLPlatform(std::move(FullDeviceIDs), std::move(Contexts),
                        std::move(CommandQueues));
}

Expected<int> OpenCLPlatform::getDeviceCount() { return FullDeviceIDs.size(); }

static void openCLDestroyStream(void *H) {
  logOpenCLWarning(clReleaseCommandQueue(static_cast<cl_command_queue>(H)),
                   "clReleaseCommandQueue");
}

Expected<Stream> OpenCLPlatform::createStream(int DeviceIndex) {
  cl_int Result;
  cl_command_queue Queue = clCreateCommandQueue(
      Contexts[DeviceIndex], FullDeviceIDs[DeviceIndex].DeviceID,
      CL_QUEUE_PROFILING_ENABLE, &Result);
  if (Result)
    return getOpenCLError(Result, "clCreateCommandQueue");
  return constructStream(this, DeviceIndex, Queue, openCLDestroyStream);
}

static void openCLEventDestroy(void *H) {
  cl_event *CLEvent = static_cast<cl_event *>(H);
  logOpenCLWarning(clReleaseEvent(*CLEvent), "clReleaseEvent");
  delete CLEvent;
}

Status OpenCLPlatform::streamSync(void *Stream) {
  return getOpenCLError(clFinish(static_cast<cl_command_queue>(Stream)),
                        "clFinish");
}

Status OpenCLPlatform::streamWaitOnEvent(void *Stream, void *Event) {
  cl_event *CLEvent = static_cast<cl_event *>(Event);
  return getOpenCLError(
      clEnqueueBarrierWithWaitList(static_cast<cl_command_queue>(Stream), 1,
                                   CLEvent, nullptr),
      "clEnqueueMarkerWithWaitList");
}

Expected<Event> OpenCLPlatform::createEvent(int DeviceIndex) {
  cl_int Result;
  cl_event Event = clCreateUserEvent(Contexts[DeviceIndex], &Result);
  if (Result)
    return getOpenCLError(Result, "clCreateUserEvent");
  if (cl_int Result = clSetUserEventStatus(Event, CL_COMPLETE))
    return getOpenCLError(Result, "clSetUserEventStatus");
  return constructEvent(this, DeviceIndex, new cl_event(Event),
                        openCLEventDestroy);
}

static void openCLDestroyProgram(void *H) {
  logOpenCLWarning(clReleaseProgram(static_cast<cl_program>(H)),
                   "clReleaseProgram");
}

Expected<Program>
OpenCLPlatform::createProgramFromSource(Span<const char> Source,
                                        int DeviceIndex) {
  cl_int Error;
  const char *CSource = Source.data();
  size_t SourceSize = Source.size();
  cl_program Program = clCreateProgramWithSource(Contexts[DeviceIndex], 1,
                                                 &CSource, &SourceSize, &Error);
  if (Error)
    return getOpenCLError(Error, "clCreateProgramWithSource");
  cl_device_id DeviceID = FullDeviceIDs[DeviceIndex].DeviceID;
  if (cl_int Error =
          clBuildProgram(Program, 1, &DeviceID, nullptr, nullptr, nullptr))
    return getOpenCLError(Error, "clBuildProgram");
  return constructProgram(this, Program, openCLDestroyProgram);
}

Expected<void *> OpenCLPlatform::rawMallocD(ptrdiff_t ByteCount,
                                            int DeviceIndex) {
  cl_int Result;
  cl_mem Memory = clCreateBuffer(Contexts[DeviceIndex], CL_MEM_READ_WRITE,
                                 ByteCount, nullptr, &Result);
  if (Result)
    return getOpenCLError(Result, "clCreateBuffer");
  return reinterpret_cast<void *>(Memory);
}

static void openCLDestroyDeviceMemory(void *H) {
  logOpenCLWarning(clReleaseMemObject(static_cast<cl_mem>(H)),
                   "clReleaseMemObject");
}

HandleDestructor OpenCLPlatform::getDeviceMemoryHandleDestructor() {
  return openCLDestroyDeviceMemory;
}

void *OpenCLPlatform::getDeviceMemorySpanHandle(void *BaseHandle,
                                                size_t ByteSize,
                                                size_t ByteOffset) {
  cl_int Error;
  cl_buffer_region Region;
  Region.origin = ByteOffset;
  Region.size = ByteSize;
  cl_mem SubBuffer =
      clCreateSubBuffer(static_cast<cl_mem>(BaseHandle), 0,
                        CL_BUFFER_CREATE_TYPE_REGION, &Region, &Error);
  logOpenCLWarning(Error, "clCreateSubBuffer");
  if (Error)
    return nullptr;
  return SubBuffer;
}

void OpenCLPlatform::rawDestroyDeviceMemorySpanHandle(void *Handle) {
  openCLDestroyDeviceMemory(Handle);
}

Expected<void *>
OpenCLPlatform::rawGetDeviceSymbolAddress(const void * /*Symbol*/,
                                          int /*DeviceIndex*/) {
  // This doesn't seem to have any equivalent in OpenCL.
  return Status("not implemented");
}

Expected<ptrdiff_t>
OpenCLPlatform::rawGetDeviceSymbolSize(const void * /*Symbol*/,
                                       int /*DeviceIndex*/) {
  // This doesn't seem to have any equivalent in OpenCL.
  return Status("not implemented");
}

static void noOpHandleDestructor(void *) {}

Status OpenCLPlatform::rawRegisterHostMem(const void * /*Memory*/,
                                          ptrdiff_t /*ByteCount*/) {
  // TODO(jhen): Do we want to do something to pin the memory here?
  return Status();
}

HandleDestructor OpenCLPlatform::getUnregisterHostMemoryHandleDestructor() {
  // TODO(jhen): Do we want to unpin the memory here?
  return noOpHandleDestructor;
}

Expected<void *> OpenCLPlatform::rawMallocRegisteredH(ptrdiff_t ByteCount) {
  // TODO(jhen): Do we want to do something to pin the memory here?
  return std::malloc(ByteCount);
}

static void freeMemoryHandleDestructor(void *Memory) {
  // TODO(jhen): Do we want to unpin the memory here?
  std::free(Memory);
}

HandleDestructor OpenCLPlatform::getFreeHostMemoryHandleDestructor() {
  return freeMemoryHandleDestructor;
}

Status OpenCLPlatform::asyncCopyDToD(const void *DeviceSrc,
                                     ptrdiff_t DeviceSrcByteOffset,
                                     void *DeviceDst,
                                     ptrdiff_t DeviceDstByteOffset,
                                     ptrdiff_t ByteCount, void *Stream) {
  return getOpenCLError(
      clEnqueueCopyBuffer(static_cast<cl_command_queue>(Stream),
                          static_cast<cl_mem>(const_cast<void *>(DeviceSrc)),
                          static_cast<cl_mem>(DeviceDst), DeviceSrcByteOffset,
                          DeviceDstByteOffset, ByteCount, 0, nullptr, nullptr),
      "clEnqueueCopyBuffer");
}

Status OpenCLPlatform::asyncCopyDToH(const void *DeviceSrc,
                                     ptrdiff_t DeviceSrcByteOffset,
                                     void *HostDst, ptrdiff_t ByteCount,
                                     void *Stream) {
  return getOpenCLError(
      clEnqueueReadBuffer(static_cast<cl_command_queue>(Stream),
                          static_cast<cl_mem>(const_cast<void *>(DeviceSrc)),
                          CL_TRUE, DeviceSrcByteOffset, ByteCount, HostDst, 0,
                          nullptr, nullptr),
      "clEnqueueReadBuffer");
}

Status OpenCLPlatform::asyncCopyHToD(const void *HostSrc, void *DeviceDst,
                                     ptrdiff_t DeviceDstByteOffset,
                                     ptrdiff_t ByteCount, void *Stream) {
  return getOpenCLError(
      clEnqueueWriteBuffer(static_cast<cl_command_queue>(Stream),
                           static_cast<cl_mem>(DeviceDst), CL_TRUE,
                           DeviceDstByteOffset, ByteCount, HostSrc, 0, nullptr,
                           nullptr),
      "clEnqueueWriteBuffer");
}

Status OpenCLPlatform::asyncMemsetD(void *DeviceDst, ptrdiff_t ByteOffset,
                                    ptrdiff_t ByteCount, char ByteValue,
                                    void *Stream) {
  return getOpenCLError(
      clEnqueueFillBuffer(static_cast<cl_command_queue>(Stream),
                          static_cast<cl_mem>(DeviceDst), &ByteValue, 1,
                          ByteOffset, ByteCount, 0, nullptr, nullptr),
      "clEnqueueFillBuffer");
}

struct StreamCallbackUserData {
  StreamCallbackUserData(Stream &TheStream, StreamCallback Function,
                         cl_event EndEvent)
      : TheStream(TheStream), TheFunction(std::move(Function)),
        EndEvent(EndEvent) {}

  Stream &TheStream;
  StreamCallback TheFunction;
  cl_event EndEvent;
};

// A function with the right signature to pass to clSetEventCallback.
void CL_CALLBACK openCLStreamCallbackShim(cl_event,
                                          cl_int EventCommandExecStatus,
                                          void *UserData) {
  std::unique_ptr<StreamCallbackUserData> Data(
      static_cast<StreamCallbackUserData *>(UserData));
  Data->TheFunction(
      Data->TheStream,
      getOpenCLError(EventCommandExecStatus, "stream callback error state"));
  if (cl_int Result = clSetUserEventStatus(Data->EndEvent, CL_COMPLETE))
    logOpenCLWarning(Result, "clSetUserEventStatus");
  if (cl_int Result = clReleaseEvent(Data->EndEvent))
    logOpenCLWarning(Result, "clReleaseEvent");
}

Status OpenCLPlatform::addStreamCallback(Stream &TheStream,
                                         StreamCallback Callback) {
  cl_int Result;
  cl_event StartEvent =
      clCreateUserEvent(Contexts[TheStream.getDeviceIndex()], &Result);
  if (Result)
    return getOpenCLError(Result, "clCreateUserEvent");
  cl_event EndEvent =
      clCreateUserEvent(Contexts[TheStream.getDeviceIndex()], &Result);
  if (Result)
    return getOpenCLError(Result, "clCreateUserEvent");
  cl_event StartBarrierEvent;
  if (cl_int Result = clEnqueueBarrierWithWaitList(
          static_cast<cl_command_queue>(getStreamHandle(TheStream)), 1,
          &StartEvent, &StartBarrierEvent))
    return getOpenCLError(Result, "clEnqueueBarrierWithWaitList");

  if (cl_int Result = clEnqueueBarrierWithWaitList(
          static_cast<cl_command_queue>(getStreamHandle(TheStream)), 1,
          &EndEvent, nullptr))
    return getOpenCLError(Result, "clEnqueueBarrierWithWaitList");

  std::unique_ptr<StreamCallbackUserData> UserData(
      new StreamCallbackUserData(TheStream, std::move(Callback), EndEvent));
  if (cl_int Result =
          clSetEventCallback(StartBarrierEvent, CL_RUNNING,
                             openCLStreamCallbackShim, UserData.release()))
    return getOpenCLError(Result, "clSetEventCallback");

  if (cl_int Result = clSetUserEventStatus(StartEvent, CL_COMPLETE))
    return getOpenCLError(Result, "clSetUserEventStatus");

  if (cl_int Result = clReleaseEvent(StartBarrierEvent))
    return getOpenCLError(Result, "clReleaseEvent");

  return getOpenCLError(clReleaseEvent(StartEvent), "clReleaseEvent");
}

Status OpenCLPlatform::enqueueEvent(void *Event, void *Stream) {
  cl_event *CLEvent = static_cast<cl_event *>(Event);
  cl_event OldEvent = *CLEvent;
  cl_event NewEvent;
  if (cl_int Result = clEnqueueMarkerWithWaitList(
          static_cast<cl_command_queue>(Stream), 0, nullptr, &NewEvent))
    return getOpenCLError(Result, "clEnqueueMarkerWithWaitList");
  *CLEvent = NewEvent;
  return getOpenCLError(clReleaseEvent(OldEvent), "clReleaseEvent");
}

bool OpenCLPlatform::eventIsDone(void *Event) {
  cl_event *CLEvent = static_cast<cl_event *>(Event);
  cl_int EventStatus;
  logOpenCLWarning(clGetEventInfo(*CLEvent, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                  sizeof(EventStatus), &EventStatus, nullptr),
                   "clGetEventInfo");
  return EventStatus == CL_COMPLETE || EventStatus < 0;
}

Status OpenCLPlatform::eventSync(void *Event) {
  cl_event *CLEvent = static_cast<cl_event *>(Event);
  return getOpenCLError(clWaitForEvents(1, CLEvent), "clWaitForEvents");
}

Expected<float> OpenCLPlatform::getSecondsBetweenEvents(void *StartEvent,
                                                        void *EndEvent) {
  cl_event *CLStartEvent = static_cast<cl_event *>(StartEvent);
  cl_event *CLEndEvent = static_cast<cl_event *>(EndEvent);

  cl_profiling_info ParamName = CL_PROFILING_COMMAND_END;
  cl_ulong StartNanoseconds;
  cl_ulong EndNanoseconds;
  if (cl_int Result =
          clGetEventProfilingInfo(*CLStartEvent, ParamName, sizeof(cl_ulong),
                                  &StartNanoseconds, nullptr))
    return getOpenCLError(Result, "clGetEventProfilingInfo");
  if (cl_int Result = clGetEventProfilingInfo(
          *CLEndEvent, ParamName, sizeof(cl_ulong), &EndNanoseconds, nullptr))
    return getOpenCLError(Result, "clGetEventProfilingInfo");
  return (EndNanoseconds - StartNanoseconds) * 1e-12;
}

Expected<void *> OpenCLPlatform::rawCreateKernel(void *Program,
                                                 const std::string &Name) {

  cl_int Error;
  cl_kernel Kernel =
      clCreateKernel(static_cast<cl_program>(Program), Name.c_str(), &Error);
  if (Error)
    return getOpenCLError(Error, "clCreateKernel");
  return Kernel;
}

static void openCLDestroyKernel(void *H) {
  logOpenCLWarning(clReleaseKernel(static_cast<cl_kernel>(H)),
                   "clReleaseKernel");
}

HandleDestructor OpenCLPlatform::getKernelHandleDestructor() {
  return openCLDestroyKernel;
}

Status OpenCLPlatform::rawEnqueueKernelLaunch(
    void *Stream, void *Kernel, KernelLaunchDimensions LaunchDimensions,
    Span<void *> Arguments, Span<size_t> ArgumentSizes,
    size_t SharedMemoryBytes) {
  if (SharedMemoryBytes != 0)
    return Status("OpenCL kernel launches only accept zero for the shared "
                  "memory byte size");
  cl_kernel TheKernel = static_cast<cl_kernel>(Kernel);
  for (int I = 0; I < Arguments.size(); ++I)
    if (cl_int Error =
            clSetKernelArg(TheKernel, I, ArgumentSizes[I], Arguments[I]))
      return getOpenCLError(Error, "clSetKernelArg");
  size_t LocalWorkSize[] = {LaunchDimensions.BlockX, LaunchDimensions.BlockY,
                            LaunchDimensions.BlockZ};
  size_t GlobalWorkSize[] = {LaunchDimensions.BlockX * LaunchDimensions.GridX,
                             LaunchDimensions.BlockY * LaunchDimensions.GridY,
                             LaunchDimensions.BlockZ * LaunchDimensions.GridZ};
  return getOpenCLError(
      clEnqueueNDRangeKernel(static_cast<cl_command_queue>(Stream), TheKernel,
                             3, nullptr, GlobalWorkSize, LocalWorkSize, 0,
                             nullptr, nullptr),
      "clEnqueueNDRangeKernel");
}

} // namespace

namespace opencl {

/// Gets an OpenCLPlatform instance and returns it as an unowned pointer to a
/// Platform.
Expected<Platform *> getPlatform() {
  static auto MaybePlatform = []() -> Expected<OpenCLPlatform *> {
    Expected<OpenCLPlatform> CreationResult = OpenCLPlatform::create();
    if (CreationResult.isError())
      return CreationResult.getError();
    else
      return new OpenCLPlatform(CreationResult.takeValue());
  }();
  return MaybePlatform;
}

} // namespace opencl

} // namespace acxxel
