//===--- acxxel.h - The Acxxel API ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// \mainpage Welcome to Acxxel
///
/// \section Introduction
///
/// \b Acxxel is a library providing a modern C++ interface for managing
/// accelerator devices such as GPUs. Acxxel handles operations such as
/// allocating device memory, copying data to and from device memory, creating
/// and managing device events, and creating and managing device streams.
///
/// \subsection ExampleUsage Example Usage
///
/// Below is some example code to show you the basics of Acxxel.
///
/// \snippet examples/simple_example.cu Example simple saxpy
///
/// The above code could be compiled with either `clang` or `nvcc`. Compare this
/// with the standard CUDA runtime library code to perform these same
/// operations:
///
/// \snippet examples/simple_example.cu Example CUDA simple saxpy
///
/// Notice that the CUDA runtime calls are not type safe. For example, if you
/// change the type of the inputs from `float` to `double`, you have to remember
/// to change the size calculation. If you forget, you will get garbage output
/// data. In the Acxxel example, you would instead get a helpful compile-time
/// error that wouldn't let you forget to change the types inside the function.
///
/// The Acxxel example also automatically uses the right sizes for memory
/// copies, so you don't have to worry about computing the sizes yourself.
///
/// The CUDA runtime interface makes it easy to get the source and destination
/// mixed up in a call to `cudaMemcpy`. If you pass the pointers in the wrong
/// order or pass the wrong enum value for the direction parameter, you won't
/// find out until runtime (if you remembered to check the error return value of
/// `cudaMemcpy`). In Acxxel there is no verbose direction enum because the name
/// of the function says which way the copy goes, and mixing up the order of
/// source and destination is a compile-time error.
///
/// The CUDA runtime interface makes you clean up your device memory by calling
/// `cudaFree` for each call to `cudaMalloc`. In Acxxel, you don't have to worry
/// about that because the memory cleans itself up when it goes out of scope.
///
/// \subsection AcxxelFeatures Acxxel Features
///
/// Acxxel provides many nice features compared to the C-like interfaces, such
/// as the CUDA runtime API, which are normally used for the host code in
/// applications using accelerators.
///
/// \subsubsection TypeSafety Type safety
///
/// Most errors involving mixing up types, sources and destinations, or host and
/// device memory result in helpful compile-time errors.
///
/// \subsubsection NoCopySizes No need to specify sizes for memory copies
///
/// When the arguments to copy functions such as acxxel::Platform::copyHToD know
/// their sizes (e.g std::array, std::vector, and C-style arrays), there is no
/// need to specify the amount of memory to copy; Acxxel will just copy the
/// whole thing. Of course the copy functions also have overloads that accept an
/// element count for those times when you don't want to copy everything.
///
/// \subsubsection MemoryCleanup Automatic memory cleanup
///
/// Device memory allocated with acxxel::Platform::mallocD is automatically
/// freed when it goes out of scope.
///
/// \subsubsection NiceErrorHandling Error handling
///
/// Operations that would normally return values return acxxel::Expected obects
/// in Acxxel. These `Expected` objects contain either a value or an error
/// message explaining why the value is not present. This reminds the user to
/// check for errors, but also allows them to opt-out easily be calling the
/// acxxel::Expected::getValue or acxxel::Expected::takeValue methods. The
/// `getValue` method returns a reference to the value, leaving the `Expected`
/// instance as the value owner, whereas the `takeValue` method moves the value
/// out of the `Expected` object and transfers ownership to the caller.
///
/// \subsubsection PlatformIndependence Platform independence
///
/// Acxxel code works not only with CUDA, but also with any other platform that
/// can support its interface. For example, Acxxel supports OpenCL. The
/// acxxel::getCUDAPlatform and acxxel::getOpenCLPlatform functions are provided
/// to allow easy access to the built-in CUDA and OpenCL platforms. Other
/// platforms can be created by implementing the acxxel::Platform interface, and
/// instances of those classes can be created directly.
///
/// \subsubsection CUDAInterop Seamless interoperation with CUDA
///
/// Acxxel functions as a modern replacement for the standard CUDA runtime
/// library and interoperates seamlessly with kernel calls.

#ifndef ACXXEL_ACXXEL_H
#define ACXXEL_ACXXEL_H

#include "span.h"
#include "status.h"

#include <functional>
#include <memory>
#include <string>
#include <type_traits>

#if defined(__clang__) || defined(__GNUC__)
#define ACXXEL_WARN_UNUSED_RESULT __attribute__((warn_unused_result))
#else
#define ACXXEL_WARN_UNUSED_RESULT
#endif

/// This type is declared here to provide smooth interoperability with the CUDA
/// triple-chevron kernel launch syntax.
///
/// A acxxel::Stream instance will be implicitly convertible to a CUstream_st*,
/// which is the type expected for the stream argument in the triple-chevron
/// CUDA kernel launch. This means that a acxxel::Stream can be passed without
/// explicit casting as the fourth argument to a triple-chevron CUDA kernel
/// launch.
struct CUstream_st; // NOLINT

namespace acxxel {

class Event;
class Platform;
class Stream;

template <typename T> class DeviceMemory;

template <typename T> class DeviceMemorySpan;

template <typename T> class AsyncHostMemory;

template <typename T> class AsyncHostMemorySpan;

template <typename T> class OwnedAsyncHostMemory;

/// Function type used to destroy opaque handles given out by the platform.
using HandleDestructor = void (*)(void *);

/// Functor type for enqueuing host callbacks on a stream.
using StreamCallback = std::function<void(Stream &, const Status &)>;

struct KernelLaunchDimensions {
  // Intentionally implicit
  KernelLaunchDimensions(unsigned int BlockX = 1, unsigned int BlockY = 1,
                         unsigned int BlockZ = 1, unsigned int GridX = 1,
                         unsigned int GridY = 1, unsigned int GridZ = 1)
      : BlockX(BlockX), BlockY(BlockY), BlockZ(BlockZ), GridX(GridX),
        GridY(GridY), GridZ(GridZ) {}

  unsigned int BlockX;
  unsigned int BlockY;
  unsigned int BlockZ;
  unsigned int GridX;
  unsigned int GridY;
  unsigned int GridZ;
};

/// Logs a warning message.
void logWarning(const std::string &Message);

/// Gets a pointer to the standard CUDA platform.
Expected<Platform *> getCUDAPlatform();

/// Gets a pointer to the standard OpenCL platform.
Expected<Platform *> getOpenCLPlatform();

/// A function that can be executed on the device.
///
/// A Kernel is created from a Program by calling Program::createKernel, and a
/// kernel is enqueued into a Stream by calling Stream::asyncKernelLaunch.
class Kernel {
public:
  Kernel(const Kernel &) = delete;
  Kernel &operator=(const Kernel &) = delete;
  Kernel(Kernel &&) noexcept;
  Kernel &operator=(Kernel &&That) noexcept;
  ~Kernel() = default;

private:
  // Only a Program can make a kernel.
  friend class Program;
  Kernel(Platform *APlatform, void *AHandle, HandleDestructor Destructor)
      : ThePlatform(APlatform), TheHandle(AHandle, Destructor) {}

  // Let stream get raw handle for kernel launches.
  friend class Stream;

  Platform *ThePlatform;
  std::unique_ptr<void, HandleDestructor> TheHandle;
};

/// A program loaded on a device.
///
/// A program can be created by calling Platform::createProgramFromSource, and a
/// Kernel can be created from a program by running Program::createKernel.
///
/// A program can contain any number of kernels, and a program only needs to be
/// loaded once in order to use all its kernels.
class Program {
public:
  Program(const Program &) = delete;
  Program &operator=(const Program &) = delete;
  Program(Program &&) noexcept;
  Program &operator=(Program &&That) noexcept;
  ~Program() = default;

  Expected<Kernel> createKernel(const std::string &Name);

private:
  // Only a platform can make a program.
  friend class Platform;
  Program(Platform *APlatform, void *AHandle, HandleDestructor Destructor)
      : ThePlatform(APlatform), TheHandle(AHandle, Destructor) {}

  Platform *ThePlatform;
  std::unique_ptr<void, HandleDestructor> TheHandle;
};

/// A stream of computation.
///
/// All operations enqueued on a Stream are serialized, but operations enqueued
/// on different Streams may run concurrently.
///
/// Each Stream is associated with a specific, fixed device.
class Stream {
public:
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  Stream(Stream &&) noexcept;
  Stream &operator=(Stream &&) noexcept;
  ~Stream() = default;

  /// Gets the index of the device on which this Stream operates.
  int getDeviceIndex() { return TheDeviceIndex; }

  /// Blocks the host until the Stream is done executing all previously enqueued
  /// work.
  ///
  /// Returns a Status for any errors emitted by the asynchronous work on the
  /// Stream, or by any error in the synchronization process itself. Clears the
  /// Status state of the stream.
  Status sync() ACXXEL_WARN_UNUSED_RESULT;

  /// Makes all future work submitted to this stream wait until the event
  /// reports completion.
  ///
  /// This is useful because the event argument may be recorded on a different
  /// stream, so this method allows for synchronization between streams without
  /// synchronizing all streams.
  ///
  /// Returns a Status for any errors emitted by the asynchronous work on the
  /// Stream, or by any error in the synchronization process itself. Clears the
  /// Status state of the stream.
  Status waitOnEvent(Event &Event) ACXXEL_WARN_UNUSED_RESULT;

  /// Adds a host callback function to the stream.
  ///
  /// The callback will be called on the host after all previously enqueued work
  /// on the stream is complete, and no work enqueued after the callback will
  /// begin until after the callback has finished.
  Stream &addCallback(std::function<void(Stream &, const Status &)> Callback);

  /// \name Asynchronous device memory copies.
  ///
  /// These functions enqueue asynchronous memory copy operations into the
  /// stream. Only async host memory is allowed for host arguments to these
  /// functions. Async host memory can be created from normal host memory by
  /// registering it with Platform::registerHostMem. AsyncHostMemory can also be
  /// allocated directly by calling Platform::newAsyncHostMem.
  ///
  /// For all these functions, DeviceSrcTy must be convertible to
  /// DeviceMemorySpan<const T>, DeviceDstTy must be convertible to
  /// DeviceMemorySpan<T>, HostSrcTy must be convertible to
  /// AsyncHostMemorySpan<const T> and HostDstTy must be convertible to
  /// AsyncHostMemorySpan<T>. Additionally, the T types must match for the
  /// destination and source.
  /// \{

  /// Copies from device memory to device memory.
  template <typename DeviceSrcTy, typename DeviceDstTy>
  Stream &asyncCopyDToD(DeviceSrcTy &&DeviceSrc, DeviceDstTy &&DeviceDst);

  /// Copies from device memory to device memory with a given element count.
  template <typename DeviceSrcTy, typename DeviceDstTy>
  Stream &asyncCopyDToD(DeviceSrcTy &&DeviceSrc, DeviceDstTy &&DeviceDst,
                        ptrdiff_t ElementCount);

  /// Copies from device memory to host memory.
  template <typename DeviceSrcTy, typename HostDstTy>
  Stream &asyncCopyDToH(DeviceSrcTy &&DeviceSrc, HostDstTy &&HostDst);

  /// Copies from device memory to host memory with a given element count.
  template <typename DeviceSrcTy, typename HostDstTy>
  Stream &asyncCopyDToH(DeviceSrcTy &&DeviceSrc, HostDstTy &&HostDst,
                        ptrdiff_t ElementCount);

  /// Copies from host memory to device memory.
  template <typename HostSrcTy, typename DeviceDstTy>
  Stream &asyncCopyHToD(HostSrcTy &&HostSrc, DeviceDstTy &&DeviceDst);

  /// Copies from host memory to device memory with a given element count.
  template <typename HostSrcTy, typename DeviceDstTy>
  Stream &asyncCopyHToD(HostSrcTy &&HostSrc, DeviceDstTy &DeviceDst,
                        ptrdiff_t ElementCount);

  /// \}

  /// \name Stream-synchronous device memory copies
  ///
  /// These functions block the host until the copy and all previously-enqueued
  /// work on the stream has completed.
  ///
  /// For all these functions, DeviceSrcTy must be convertible to
  /// DeviceMemorySpan<const T>, DeviceDstTy must be convertible to
  /// DeviceMemorySpan<T>, HostSrcTy must be convertible to Span<const T> and
  /// HostDstTy must be convertible to Span<T>. Additionally, the T types must
  /// match for the destination and source.
  /// \{

  template <typename DeviceSrcTy, typename DeviceDstTy>
  Stream &syncCopyDToD(DeviceSrcTy &&DeviceSrc, DeviceDstTy &&DeviceDst);

  template <typename DeviceSrcTy, typename DeviceDstTy>
  Stream &syncCopyDToD(DeviceSrcTy &&DeviceSrc, DeviceDstTy &&DeviceDst,
                       ptrdiff_t ElementCount);

  template <typename DeviceSrcTy, typename HostDstTy>
  Stream &syncCopyDToH(DeviceSrcTy &&DeviceSrc, HostDstTy &&HostDst);

  template <typename DeviceSrcTy, typename HostDstTy>
  Stream &syncCopyDToH(DeviceSrcTy &&DeviceSrc, HostDstTy &&HostDst,
                       ptrdiff_t ElementCount);

  template <typename HostSrcTy, typename DeviceDstTy>
  Stream &syncCopyHToD(HostSrcTy &&HostSrc, DeviceDstTy &&DeviceDst);

  template <typename HostSrcTy, typename DeviceDstTy>
  Stream &syncCopyHToD(HostSrcTy &&HostSrc, DeviceDstTy &DeviceDst,
                       ptrdiff_t ElementCount);

  /// \}

  /// Enqueues an operation in the stream to set the bytes of a given device
  /// memory region to a given value.
  ///
  /// DeviceDstTy must be convertible to DeviceMemorySpan<T> for non-const T.
  template <typename DeviceDstTy>
  Stream &asyncMemsetD(DeviceDstTy &&DeviceDst, char ByteValue);

  /// Enqueues a kernel launch operation on this stream.
  Stream &asyncKernelLaunch(const Kernel &TheKernel,
                            KernelLaunchDimensions LaunchDimensions,
                            Span<void *> Arguments, Span<size_t> ArgumentSizes,
                            size_t SharedMemoryBytes = 0);

  /// Enqueues an event in the stream.
  Stream &enqueueEvent(Event &E);

  // Allows implicit conversion to (CUstream_st *). This makes triple-chevron
  // kernel calls look nicer because you can just pass a acxxel::Stream
  // directly.
  operator CUstream_st *() {
    return static_cast<CUstream_st *>(TheHandle.get());
  }

  /// Gets the current status for the Stream and clears the Stream's status.
  Status takeStatus() ACXXEL_WARN_UNUSED_RESULT {
    Status OldStatus = TheStatus;
    TheStatus = Status();
    return OldStatus;
  }

private:
  // Only a platform can make a stream.
  friend class Platform;
  Stream(Platform *APlatform, int DeviceIndex, void *AHandle,
         HandleDestructor Destructor)
      : ThePlatform(APlatform), TheDeviceIndex(DeviceIndex),
        TheHandle(AHandle, Destructor) {}

  const Status &setStatus(const Status &S) {
    if (S.isError() && !TheStatus.isError()) {
      TheStatus = S;
    }
    return S;
  }

  Status takeStatusOr(const Status &S) {
    if (TheStatus.isError()) {
      Status OldStatus = TheStatus;
      TheStatus = Status();
      return OldStatus;
    }
    return S;
  }

  // The platform that created the stream.
  Platform *ThePlatform;

  // The index of the device on which the stream operates.
  int TheDeviceIndex;

  // A handle to the platform-specific handle implementation.
  std::unique_ptr<void, HandleDestructor> TheHandle;
  Status TheStatus;
};

/// A user-created event on a device.
///
/// This is useful for setting synchronization points in a Stream. The host can
/// synchronize with a Stream without using events, but that requires all the
/// work in the Stream to be finished in order for the host to be notified.
/// Events provide more flexibility by allowing the host to be notified when a
/// single Event in the Stream is finished, rather than all the work in the
/// Stream.
class Event {
public:
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
  Event(Event &&) noexcept;
  Event &operator=(Event &&That) noexcept;
  ~Event() = default;

  /// Checks to see if the event is done running.
  bool isDone();

  /// Blocks the host until the event is done.
  Status sync();

  /// Gets the time elapsed between the previous event's execution and this
  /// event's execution.
  Expected<float> getSecondsSince(const Event &Previous);

private:
  // Only a platform can make an event.
  friend class Platform;
  Event(Platform *APlatform, int DeviceIndex, void *AHandle,
        HandleDestructor Destructor)
      : ThePlatform(APlatform), TheDeviceIndex(DeviceIndex),
        TheHandle(AHandle, Destructor) {}

  Platform *ThePlatform;

  // The index of the device on which the event can be enqueued.
  int TheDeviceIndex;

  std::unique_ptr<void, HandleDestructor> TheHandle;
};

/// An accelerator platform.
///
/// This is the base class for all platforms such as CUDA and OpenCL. It
/// contains many virtual methods that must be overridden by each platform
/// implementation.
///
/// It also has some template wrapper functions that take care of type checking
/// and then forward their arguments on to raw virtual functions that are
/// implemented by each specific platform.
class Platform {
public:
  virtual ~Platform(){};

  /// Gets the number of devices for this platform in this system.
  virtual Expected<int> getDeviceCount() = 0;

  /// Creates a stream on the given device for the platform.
  virtual Expected<Stream> createStream(int DeviceIndex = 0) = 0;

  /// Creates an event on the given device for the platform.
  virtual Expected<Event> createEvent(int DeviceIndex = 0) = 0;

  /// Allocates owned device memory.
  ///
  /// \warning This function only allocates space in device memory, it does not
  /// call the constructor of T.
  template <typename T>
  Expected<DeviceMemory<T>> mallocD(ptrdiff_t ElementCount,
                                    int DeviceIndex = 0) {
    Expected<void *> MaybePointer =
        rawMallocD(ElementCount * sizeof(T), DeviceIndex);
    if (MaybePointer.isError())
      return MaybePointer.getError();
    return DeviceMemory<T>(this, MaybePointer.getValue(), ElementCount,
                           this->getDeviceMemoryHandleDestructor());
  }

  /// Creates a DeviceMemorySpan for a device symbol.
  ///
  /// This function is present to support __device__ variables in CUDA. Given a
  /// pointer to a __device__ variable, this function returns a DeviceMemorySpan
  /// referencing the device memory that stores that __device__ variable.
  template <typename ElementType>
  Expected<DeviceMemorySpan<ElementType>> getSymbolMemory(ElementType *Symbol,
                                                          int DeviceIndex = 0) {
    Expected<void *> MaybeAddress =
        rawGetDeviceSymbolAddress(Symbol, DeviceIndex);
    if (MaybeAddress.isError())
      return MaybeAddress.getError();
    ElementType *Address = static_cast<ElementType *>(MaybeAddress.getValue());
    Expected<ptrdiff_t> MaybeSize = rawGetDeviceSymbolSize(Symbol, DeviceIndex);
    if (MaybeSize.isError())
      return MaybeSize.getError();
    ptrdiff_t Size = MaybeSize.getValue();
    return DeviceMemorySpan<ElementType>(this, Address,
                                         Size / sizeof(ElementType), 0);
  }

  /// \name Host memory registration functions.
  /// \{

  template <typename T>
  Expected<AsyncHostMemory<const T>> registerHostMem(Span<const T> Memory) {
    Status S = rawRegisterHostMem(Memory.data(), Memory.size() * sizeof(T));
    if (S.isError())
      return S;
    return AsyncHostMemory<const T>(
        Memory.data(), Memory.size(),
        this->getUnregisterHostMemoryHandleDestructor());
  }

  template <typename T>
  Expected<AsyncHostMemory<T>> registerHostMem(Span<T> Memory) {
    Status S = rawRegisterHostMem(Memory.data(), Memory.size() * sizeof(T));
    if (S.isError())
      return S;
    return AsyncHostMemory<T>(Memory.data(), Memory.size(),
                              this->getUnregisterHostMemoryHandleDestructor());
  }

  template <typename T, size_t N>
  Expected<AsyncHostMemory<T>> registerHostMem(T (&Array)[N]) {
    Span<T> Span(Array);
    Status S = rawRegisterHostMem(Span.data(), Span.size() * sizeof(T));
    if (S.isError())
      return S;
    return AsyncHostMemory<T>(Span.data(), Span.size(),
                              this->getUnregisterHostMemoryHandleDestructor());
  }

  /// Registers memory stored in a container with a data() member function and
  /// which can be converted to a Span<T*>.
  template <typename Container>
  auto registerHostMem(Container &Cont) -> Expected<AsyncHostMemory<
      typename std::remove_reference<decltype(*Cont.data())>::type>> {
    using ValueType =
        typename std::remove_reference<decltype(*Cont.data())>::type;
    Span<ValueType> Span(Cont);
    Status S = rawRegisterHostMem(Span.data(), Span.size() * sizeof(ValueType));
    if (S.isError())
      return S;
    return AsyncHostMemory<ValueType>(
        Span.data(), Span.size(),
        this->getUnregisterHostMemoryHandleDestructor());
  }

  /// Allocates an owned, registered array of objects on the host.
  ///
  /// Default constructs each element in the resulting array.
  template <typename T>
  Expected<OwnedAsyncHostMemory<T>> newAsyncHostMem(ptrdiff_t ElementCount) {
    Expected<void *> MaybeMemory =
        rawMallocRegisteredH(ElementCount * sizeof(T));
    if (MaybeMemory.isError())
      return MaybeMemory.getError();
    T *Memory = static_cast<T *>(MaybeMemory.getValue());
    for (ptrdiff_t I = 0; I < ElementCount; ++I)
      new (Memory + I) T;
    return OwnedAsyncHostMemory<T>(Memory, ElementCount,
                                   this->getFreeHostMemoryHandleDestructor());
  }

  /// \}

  virtual Expected<Program> createProgramFromSource(Span<const char> Source,
                                                    int DeviceIndex = 0) = 0;

protected:
  friend class Stream;
  friend class Event;
  friend class Program;
  template <typename T> friend class DeviceMemorySpan;

  void *getStreamHandle(Stream &Stream) { return Stream.TheHandle.get(); }
  void *getEventHandle(Event &Event) { return Event.TheHandle.get(); }

  // Pass along access to Stream constructor to subclasses.
  Stream constructStream(Platform *APlatform, int DeviceIndex, void *AHandle,
                         HandleDestructor Destructor) {
    return Stream(APlatform, DeviceIndex, AHandle, Destructor);
  }

  // Pass along access to Event constructor to subclasses.
  Event constructEvent(Platform *APlatform, int DeviceIndex, void *AHandle,
                       HandleDestructor Destructor) {
    return Event(APlatform, DeviceIndex, AHandle, Destructor);
  }

  // Pass along access to Program constructor to subclasses.
  Program constructProgram(Platform *APlatform, void *AHandle,
                           HandleDestructor Destructor) {
    return Program(APlatform, AHandle, Destructor);
  }

  virtual Status streamSync(void *Stream) = 0;
  virtual Status streamWaitOnEvent(void *Stream, void *Event) = 0;

  virtual Status enqueueEvent(void *Event, void *Stream) = 0;
  virtual bool eventIsDone(void *Event) = 0;
  virtual Status eventSync(void *Event) = 0;
  virtual Expected<float> getSecondsBetweenEvents(void *StartEvent,
                                                  void *EndEvent) = 0;

  virtual Expected<void *> rawMallocD(ptrdiff_t ByteCount, int DeviceIndex) = 0;
  virtual HandleDestructor getDeviceMemoryHandleDestructor() = 0;
  virtual void *getDeviceMemorySpanHandle(void *BaseHandle, size_t ByteSize,
                                          size_t ByteOffset) = 0;
  virtual void rawDestroyDeviceMemorySpanHandle(void *Handle) = 0;

  virtual Expected<void *> rawGetDeviceSymbolAddress(const void *Symbol,
                                                     int DeviceIndex) = 0;
  virtual Expected<ptrdiff_t> rawGetDeviceSymbolSize(const void *Symbol,
                                                     int DeviceIndex) = 0;

  virtual Status rawRegisterHostMem(const void *Memory,
                                    ptrdiff_t ByteCount) = 0;
  virtual HandleDestructor getUnregisterHostMemoryHandleDestructor() = 0;

  virtual Expected<void *> rawMallocRegisteredH(ptrdiff_t ByteCount) = 0;
  virtual HandleDestructor getFreeHostMemoryHandleDestructor() = 0;

  virtual Status asyncCopyDToD(const void *DeviceSrc,
                               ptrdiff_t DeviceSrcByteOffset, void *DeviceDst,
                               ptrdiff_t DeviceDstByteOffset,
                               ptrdiff_t ByteCount, void *Stream) = 0;
  virtual Status asyncCopyDToH(const void *DeviceSrc,
                               ptrdiff_t DeviceSrcByteOffset, void *HostDst,
                               ptrdiff_t ByteCount, void *Stream) = 0;
  virtual Status asyncCopyHToD(const void *HostSrc, void *DeviceDst,
                               ptrdiff_t DeviceDstByteOffset,
                               ptrdiff_t ByteCount, void *Stream) = 0;

  virtual Status asyncMemsetD(void *DeviceDst, ptrdiff_t ByteOffset,
                              ptrdiff_t ByteCount, char ByteValue,
                              void *Stream) = 0;

  virtual Status addStreamCallback(Stream &Stream, StreamCallback Callback) = 0;

  virtual Expected<void *> rawCreateKernel(void *Program,
                                           const std::string &Name) = 0;
  virtual HandleDestructor getKernelHandleDestructor() = 0;

  virtual Status rawEnqueueKernelLaunch(void *Stream, void *Kernel,
                                        KernelLaunchDimensions LaunchDimensions,
                                        Span<void *> Arguments,
                                        Span<size_t> ArgumentSizes,
                                        size_t SharedMemoryBytes) = 0;
};

// Implementation of templated Stream functions.

template <typename DeviceSrcTy, typename DeviceDstTy>
Stream &Stream::asyncCopyDToD(DeviceSrcTy &&DeviceSrc,
                              DeviceDstTy &&DeviceDst) {
  using SrcElementTy =
      typename std::remove_reference<DeviceSrcTy>::type::value_type;
  using DstElementTy =
      typename std::remove_reference<DeviceDstTy>::type::value_type;
  static_assert(std::is_same<SrcElementTy, DstElementTy>::value,
                "asyncCopyDToD cannot copy between arrays of different types");
  DeviceMemorySpan<const SrcElementTy> DeviceSrcSpan(DeviceSrc);
  DeviceMemorySpan<DstElementTy> DeviceDstSpan(DeviceDst);
  if (DeviceSrcSpan.size() != DeviceDstSpan.size()) {
    setStatus(Status("asyncCopyDToD source element count " +
                     std::to_string(DeviceSrcSpan.size()) +
                     " does not equal destination element count " +
                     std::to_string(DeviceDstSpan.size())));
    return *this;
  }
  setStatus(ThePlatform->asyncCopyDToD(
      DeviceSrcSpan.baseHandle(), DeviceSrcSpan.byte_offset(),
      DeviceDstSpan.baseHandle(), DeviceDstSpan.byte_offset(),
      DeviceSrcSpan.byte_size(), TheHandle.get()));
  return *this;
}

template <typename DeviceSrcTy, typename DeviceDstTy>
Stream &Stream::asyncCopyDToD(DeviceSrcTy &&DeviceSrc, DeviceDstTy &&DeviceDst,
                              ptrdiff_t ElementCount) {
  using SrcElementTy =
      typename std::remove_reference<DeviceSrcTy>::type::value_type;
  using DstElementTy =
      typename std::remove_reference<DeviceDstTy>::type::value_type;
  static_assert(std::is_same<SrcElementTy, DstElementTy>::value,
                "asyncCopyDToD cannot copy between arrays of different types");
  DeviceMemorySpan<const SrcElementTy> DeviceSrcSpan(DeviceSrc);
  DeviceMemorySpan<DstElementTy> DeviceDstSpan(DeviceDst);
  if (DeviceSrcSpan.size() < ElementCount) {
    setStatus(Status("asyncCopyDToD source element count " +
                     std::to_string(DeviceSrcSpan.size()) +
                     " is less than requested element count " +
                     std::to_string(ElementCount)));
    return *this;
  }
  if (DeviceDstSpan.size() < ElementCount) {
    setStatus(Status("asyncCopyDToD destination element count " +
                     std::to_string(DeviceDst.size()) +
                     " is less than requested element count " +
                     std::to_string(ElementCount)));
    return *this;
  }
  setStatus(ThePlatform->asyncCopyDToD(
      DeviceSrcSpan.baseHandle(), DeviceSrcSpan.byte_offset(),
      DeviceDstSpan.baseHandle(), DeviceDstSpan.byte_offset(),
      ElementCount * sizeof(SrcElementTy), TheHandle.get()));
  return *this;
}

template <typename DeviceSrcTy, typename HostDstTy>
Stream &Stream::asyncCopyDToH(DeviceSrcTy &&DeviceSrc, HostDstTy &&HostDst) {
  using SrcElementTy =
      typename std::remove_reference<DeviceSrcTy>::type::value_type;
  DeviceMemorySpan<const SrcElementTy> DeviceSrcSpan(DeviceSrc);
  AsyncHostMemorySpan<SrcElementTy> HostDstSpan(HostDst);
  if (DeviceSrcSpan.size() != HostDstSpan.size()) {
    setStatus(Status("asyncCopyDToH source element count " +
                     std::to_string(DeviceSrcSpan.size()) +
                     " does not equal destination element count " +
                     std::to_string(HostDstSpan.size())));
    return *this;
  }
  setStatus(ThePlatform->asyncCopyDToH(
      DeviceSrcSpan.baseHandle(), DeviceSrcSpan.byte_offset(),
      HostDstSpan.data(), DeviceSrcSpan.byte_size(), TheHandle.get()));
  return *this;
}

template <typename DeviceSrcTy, typename HostDstTy>
Stream &Stream::asyncCopyDToH(DeviceSrcTy &&DeviceSrc, HostDstTy &&HostDst,
                              ptrdiff_t ElementCount) {
  using SrcElementTy =
      typename std::remove_reference<DeviceSrcTy>::type::value_type;
  DeviceMemorySpan<const SrcElementTy> DeviceSrcSpan(DeviceSrc);
  AsyncHostMemorySpan<SrcElementTy> HostDstSpan(HostDst);
  if (DeviceSrcSpan.size() < ElementCount) {
    setStatus(Status("asyncCopyDToH source element count " +
                     std::to_string(DeviceSrcSpan.size()) +
                     " is less than requested element count " +
                     std::to_string(ElementCount)));
    return *this;
  }
  if (HostDstSpan.size() < ElementCount) {
    setStatus(Status("asyncCopyDToH destination element count " +
                     std::to_string(HostDstSpan.size()) +
                     " is less than requested element count " +
                     std::to_string(ElementCount)));
    return *this;
  }
  setStatus(ThePlatform->asyncCopyDToH(
      DeviceSrcSpan.baseHandle(), DeviceSrcSpan.byte_offset(),
      HostDstSpan.data(), ElementCount * sizeof(SrcElementTy),
      TheHandle.get()));
  return *this;
}

template <typename HostSrcTy, typename DeviceDstTy>
Stream &Stream::asyncCopyHToD(HostSrcTy &&HostSrc, DeviceDstTy &&DeviceDst) {
  using DstElementTy =
      typename std::remove_reference<DeviceDstTy>::type::value_type;
  AsyncHostMemorySpan<const DstElementTy> HostSrcSpan(HostSrc);
  DeviceMemorySpan<DstElementTy> DeviceDstSpan(DeviceDst);
  if (HostSrcSpan.size() != DeviceDstSpan.size()) {
    setStatus(Status("asyncCopyHToD source element count " +
                     std::to_string(HostSrcSpan.size()) +
                     " does not equal destination element count " +
                     std::to_string(DeviceDstSpan.size())));
    return *this;
  }
  setStatus(ThePlatform->asyncCopyHToD(
      HostSrcSpan.data(), DeviceDstSpan.baseHandle(),
      DeviceDstSpan.byte_offset(), HostSrcSpan.byte_size(), TheHandle.get()));
  return *this;
}

template <typename HostSrcTy, typename DeviceDstTy>
Stream &Stream::asyncCopyHToD(HostSrcTy &&HostSrc, DeviceDstTy &DeviceDst,
                              ptrdiff_t ElementCount) {
  using DstElementTy =
      typename std::remove_reference<DeviceDstTy>::type::value_type;
  AsyncHostMemorySpan<const DstElementTy> HostSrcSpan(HostSrc);
  DeviceMemorySpan<DstElementTy> DeviceDstSpan(DeviceDst);
  if (HostSrcSpan.size() < ElementCount) {
    setStatus(Status("copyHToD source element count " +
                     std::to_string(HostSrcSpan.size()) +
                     " is less than requested element count " +
                     std::to_string(ElementCount)));
    return *this;
  }
  if (DeviceDstSpan.size() < ElementCount) {
    setStatus(Status("copyHToD destination element count " +
                     std::to_string(DeviceDstSpan.size()) +
                     " is less than requested element count " +
                     std::to_string(ElementCount)));
    return *this;
  }
  setStatus(ThePlatform->asyncCopyHToD(
      HostSrcSpan.data(), DeviceDstSpan.baseHandle(),
      DeviceDstSpan.byte_offset(), ElementCount * sizeof(DstElementTy),
      TheHandle.get()));
  return *this;
}

template <typename DeviceDstTy>
Stream &Stream::asyncMemsetD(DeviceDstTy &&DeviceDst, char ByteValue) {
  using DstElementTy =
      typename std::remove_reference<DeviceDstTy>::type::value_type;
  DeviceMemorySpan<DstElementTy> DeviceDstSpan(DeviceDst);
  setStatus(ThePlatform->asyncMemsetD(
      DeviceDstSpan.baseHandle(), DeviceDstSpan.byte_offset(),
      DeviceDstSpan.byte_size(), ByteValue, TheHandle.get()));
  return *this;
}

template <typename DeviceSrcTy, typename DeviceDstTy>
Stream &Stream::syncCopyDToD(DeviceSrcTy &&DeviceSrc, DeviceDstTy &&DeviceDst) {
  using SrcElementTy =
      typename std::remove_reference<DeviceSrcTy>::type::value_type;
  using DstElementTy =
      typename std::remove_reference<DeviceDstTy>::type::value_type;
  static_assert(std::is_same<SrcElementTy, DstElementTy>::value,
                "copyDToD cannot copy between arrays of different types");
  DeviceMemorySpan<const SrcElementTy> DeviceSrcSpan(DeviceSrc);
  DeviceMemorySpan<DstElementTy> DeviceDstSpan(DeviceDst);
  if (DeviceSrcSpan.size() != DeviceDstSpan.size()) {
    setStatus(Status("copyDToD source element count " +
                     std::to_string(DeviceSrcSpan.size()) +
                     " does not equal destination element count " +
                     std::to_string(DeviceDstSpan.size())));
    return *this;
  }
  if (setStatus(ThePlatform->asyncCopyDToD(
                    DeviceSrcSpan.baseHandle(), DeviceSrcSpan.byte_offset(),
                    DeviceDstSpan.baseHandle(), DeviceDstSpan.byte_offset(),
                    DeviceSrcSpan.byte_size(), TheHandle.get()))
          .isError()) {
    return *this;
  }
  setStatus(sync());
  return *this;
}

template <typename DeviceSrcTy, typename DeviceDstTy>
Stream &Stream::syncCopyDToD(DeviceSrcTy &&DeviceSrc, DeviceDstTy &&DeviceDst,
                             ptrdiff_t ElementCount) {
  using SrcElementTy =
      typename std::remove_reference<DeviceSrcTy>::type::value_type;
  using DstElementTy =
      typename std::remove_reference<DeviceDstTy>::type::value_type;
  static_assert(std::is_same<SrcElementTy, DstElementTy>::value,
                "copyDToD cannot copy between arrays of different types");
  DeviceMemorySpan<const SrcElementTy> DeviceSrcSpan(DeviceSrc);
  DeviceMemorySpan<DstElementTy> DeviceDstSpan(DeviceDst);
  if (DeviceSrcSpan.size() < ElementCount) {
    setStatus(Status("copyDToD source element count " +
                     std::to_string(DeviceSrcSpan.size()) +
                     " is less than requested element count " +
                     std::to_string(ElementCount)));
    return *this;
  }
  if (DeviceDstSpan.size() < ElementCount) {
    setStatus(Status("copyDToD destination element count " +
                     std::to_string(DeviceDst.size()) +
                     " is less than requested element count " +
                     std::to_string(ElementCount)));
    return *this;
  }
  if (setStatus(ThePlatform->asyncCopyDToD(
                    DeviceSrcSpan.baseHandle(), DeviceSrcSpan.byte_offset(),
                    DeviceDstSpan.baseHandle(), DeviceDstSpan.byte_offset(),
                    ElementCount * sizeof(SrcElementTy), TheHandle.get()))
          .isError()) {
    return *this;
  }
  setStatus(sync());
  return *this;
}

template <typename DeviceSrcTy, typename HostDstTy>
Stream &Stream::syncCopyDToH(DeviceSrcTy &&DeviceSrc, HostDstTy &&HostDst) {
  using SrcElementTy =
      typename std::remove_reference<DeviceSrcTy>::type::value_type;
  DeviceMemorySpan<const SrcElementTy> DeviceSrcSpan(DeviceSrc);
  Span<SrcElementTy> HostDstSpan(HostDst);
  if (DeviceSrcSpan.size() != HostDstSpan.size()) {
    setStatus(Status("copyDToH source element count " +
                     std::to_string(DeviceSrcSpan.size()) +
                     " does not equal destination element count " +
                     std::to_string(HostDstSpan.size())));
    return *this;
  }
  if (setStatus(ThePlatform->asyncCopyDToH(
                    DeviceSrcSpan.baseHandle(), DeviceSrcSpan.byte_offset(),
                    HostDstSpan.data(), DeviceSrcSpan.byte_size(),
                    TheHandle.get()))
          .isError()) {
    return *this;
  }
  setStatus(sync());
  return *this;
}

template <typename DeviceSrcTy, typename HostDstTy>
Stream &Stream::syncCopyDToH(DeviceSrcTy &&DeviceSrc, HostDstTy &&HostDst,
                             ptrdiff_t ElementCount) {
  using SrcElementTy =
      typename std::remove_reference<DeviceSrcTy>::type::value_type;
  DeviceMemorySpan<const SrcElementTy> DeviceSrcSpan(DeviceSrc);
  Span<SrcElementTy> HostDstSpan(HostDst);
  if (DeviceSrcSpan.size() < ElementCount) {
    setStatus(Status("copyDToH source element count " +
                     std::to_string(DeviceSrcSpan.size()) +
                     " is less than requested element count " +
                     std::to_string(ElementCount)));
    return *this;
  }
  if (HostDstSpan.size() < ElementCount) {
    setStatus(Status("copyDToH destination element count " +
                     std::to_string(HostDstSpan.size()) +
                     " is less than requested element count " +
                     std::to_string(ElementCount)));
    return *this;
  }
  if (setStatus(ThePlatform->asyncCopyDToH(
                    DeviceSrcSpan.baseHandle(), DeviceSrcSpan.byte_offset(),
                    HostDstSpan.data(), ElementCount * sizeof(SrcElementTy),
                    TheHandle.get()))
          .isError()) {
    return *this;
  }
  setStatus(sync());
  return *this;
}

template <typename HostSrcTy, typename DeviceDstTy>
Stream &Stream::syncCopyHToD(HostSrcTy &&HostSrc, DeviceDstTy &&DeviceDst) {
  using DstElementTy =
      typename std::remove_reference<DeviceDstTy>::type::value_type;
  Span<const DstElementTy> HostSrcSpan(HostSrc);
  DeviceMemorySpan<DstElementTy> DeviceDstSpan(DeviceDst);
  if (HostSrcSpan.size() != DeviceDstSpan.size()) {
    setStatus(Status("copyHToD source element count " +
                     std::to_string(HostSrcSpan.size()) +
                     " does not equal destination element count " +
                     std::to_string(DeviceDstSpan.size())));
    return *this;
  }
  if (setStatus(ThePlatform->asyncCopyHToD(
                    HostSrcSpan.data(), DeviceDstSpan.baseHandle(),
                    DeviceDstSpan.byte_offset(), DeviceDstSpan.byte_size(),
                    TheHandle.get()))
          .isError()) {
    return *this;
  }
  setStatus(sync());
  return *this;
}

template <typename HostSrcTy, typename DeviceDstTy>
Stream &Stream::syncCopyHToD(HostSrcTy &&HostSrc, DeviceDstTy &DeviceDst,
                             ptrdiff_t ElementCount) {
  using DstElementTy =
      typename std::remove_reference<DeviceDstTy>::type::value_type;
  Span<const DstElementTy> HostSrcSpan(HostSrc);
  DeviceMemorySpan<DstElementTy> DeviceDstSpan(DeviceDst);
  if (HostSrcSpan.size() < ElementCount) {
    setStatus(Status("copyHToD source element count " +
                     std::to_string(HostSrcSpan.size()) +
                     " is less than requested element count " +
                     std::to_string(ElementCount)));
    return *this;
  }
  if (DeviceDstSpan.size() < ElementCount) {
    setStatus(Status("copyHToD destination element count " +
                     std::to_string(DeviceDstSpan.size()) +
                     " is less than requested element count " +
                     std::to_string(ElementCount)));
    return *this;
  }
  if (setStatus(ThePlatform->asyncCopyHToD(
                    HostSrcSpan.data(), DeviceDstSpan.baseHandle(),
                    DeviceDstSpan.byte_offset(),
                    ElementCount * sizeof(DstElementTy), TheHandle.get()))
          .isError()) {
    return *this;
  }
  setStatus(sync());
  return *this;
}

/// Owned device memory.
///
/// Device memory that frees itself when it goes out of scope.
template <typename ElementType> class DeviceMemory {
public:
  using element_type = ElementType;
  using index_type = std::ptrdiff_t;
  using value_type = typename std::remove_const<element_type>::type;

  DeviceMemory(const DeviceMemory &) = delete;
  DeviceMemory &operator=(const DeviceMemory &) = delete;
  DeviceMemory(DeviceMemory &&) noexcept;
  DeviceMemory &operator=(DeviceMemory &&) noexcept;
  ~DeviceMemory() = default;

  /// Gets the raw base handle for the underlying platform implementation.
  void *handle() const { return ThePointer.get(); }

  index_type length() const { return TheSize; }
  index_type size() const { return TheSize; }
  index_type byte_size() const { // NOLINT
    return TheSize * sizeof(element_type);
  }
  bool empty() const { return TheSize == 0; }

  // These conversion operators are useful for making triple-chevron kernel
  // launches more concise.
  operator element_type *() {
    return static_cast<element_type *>(ThePointer.get());
  }
  operator const element_type *() const { return ThePointer.get(); }

  /// Converts a const object to a DeviceMemorySpan of const elements.
  DeviceMemorySpan<const element_type> asSpan() const {
    return DeviceMemorySpan<const element_type>(
        ThePlatform, static_cast<const element_type *>(ThePointer.get()),
        TheSize, 0);
  }

  /// Converts an object to a DeviceMemorySpan.
  DeviceMemorySpan<element_type> asSpan() {
    return DeviceMemorySpan<element_type>(
        ThePlatform, static_cast<element_type *>(ThePointer.get()), TheSize, 0);
  }

private:
  friend class Platform;
  template <typename T> friend class DeviceMemorySpan;

  DeviceMemory(Platform *ThePlatform, void *Pointer, index_type ElementCount,
               HandleDestructor Destructor)
      : ThePlatform(ThePlatform), ThePointer(Pointer, Destructor),
        TheSize(ElementCount) {}

  Platform *ThePlatform;
  std::unique_ptr<void, HandleDestructor> ThePointer;
  ptrdiff_t TheSize;
};

template <typename T>
DeviceMemory<T>::DeviceMemory(DeviceMemory &&) noexcept = default;
template <typename T>
DeviceMemory<T> &DeviceMemory<T>::operator=(DeviceMemory &&) noexcept = default;

/// View into device memory.
///
/// Like a Span, but for device memory rather than host memory.
template <typename ElementType> class DeviceMemorySpan {
public:
  /// \name constants and types
  /// \{
  using element_type = ElementType;
  using index_type = std::ptrdiff_t;
  using pointer = element_type *;
  using reference = element_type &;
  using iterator = element_type *;
  using const_iterator = const element_type *;
  using value_type = typename std::remove_const<element_type>::type;
  /// \}

  DeviceMemorySpan()
      : ThePlatform(nullptr), TheHandle(nullptr), TheSize(0), TheOffset(0),
        TheSpanHandle(nullptr) {}

  // Intentionally implicit.
  template <typename OtherElementType>
  DeviceMemorySpan(DeviceMemorySpan<OtherElementType> &ASpan)
      : ThePlatform(ASpan.ThePlatform),
        TheHandle(static_cast<pointer>(ASpan.baseHandle())),
        TheSize(ASpan.size()), TheOffset(ASpan.offset()),
        TheSpanHandle(nullptr) {}

  // Intentionally implicit.
  template <typename OtherElementType>
  DeviceMemorySpan(DeviceMemorySpan<OtherElementType> &&ASpan)
      : ThePlatform(ASpan.ThePlatform),
        TheHandle(static_cast<pointer>(ASpan.baseHandle())),
        TheSize(ASpan.size()), TheOffset(ASpan.offset()),
        TheSpanHandle(nullptr) {}

  // Intentionally implicit.
  template <typename OtherElementType>
  DeviceMemorySpan(DeviceMemory<OtherElementType> &Memory)
      : ThePlatform(Memory.ThePlatform),
        TheHandle(static_cast<value_type *>(Memory.handle())),
        TheSize(Memory.size()), TheOffset(0), TheSpanHandle(nullptr) {}

  ~DeviceMemorySpan() {
    if (TheSpanHandle) {
      ThePlatform->rawDestroyDeviceMemorySpanHandle(
          const_cast<value_type *>(TheSpanHandle));
    }
  }

  /// \name observers
  /// \{
  index_type length() const { return TheSize; }
  index_type size() const { return TheSize; }
  index_type byte_size() const { // NOLINT
    return TheSize * sizeof(element_type);
  }
  index_type offset() const { return TheOffset; }
  index_type byte_offset() const { // NOLINT
    return TheOffset * sizeof(element_type);
  }
  bool empty() const { return TheSize == 0; }
  /// \}

  void *baseHandle() const {
    return static_cast<void *>(const_cast<value_type *>(TheHandle));
  }

  /// Casts to a host memory pointer.
  ///
  /// This is only guaranteed to make sense for the CUDA platform, where device
  /// pointers can be stored and manipulated much like host pointers. This makes
  /// it easy to do triple-chevron kernel launches in CUDA because
  /// DeviceMemorySpan values can be passed to parameters expecting regular
  /// pointers.
  ///
  /// If the CUDA platform is using unified memory, it may also be possible to
  /// dereference this pointer on the host.
  ///
  /// For platforms other than CUDA, this may return a garbage pointer.
  operator element_type *() const {
    if (!TheSpanHandle)
      TheSpanHandle = ThePlatform->getDeviceMemorySpanHandle(
          TheHandle, TheSize * sizeof(element_type),
          TheOffset * sizeof(element_type));
    return TheSpanHandle;
  }

  DeviceMemorySpan<element_type> first(index_type Count) const {
    bool Valid = Count >= 0 && Count <= TheSize;
    if (!Valid)
      std::terminate();
    return DeviceMemorySpan<element_type>(ThePlatform, TheHandle, Count,
                                          TheOffset);
  }

  DeviceMemorySpan<element_type> last(index_type Count) const {
    bool Valid = Count >= 0 && Count <= TheSize;
    if (!Valid)
      std::terminate();
    return DeviceMemorySpan<element_type>(ThePlatform, TheHandle, Count,
                                          TheOffset + TheSize - Count);
  }

  DeviceMemorySpan<element_type>
  subspan(index_type Offset, index_type Count = dynamic_extent) const {
    bool Valid =
        (Offset == 0 || (Offset > 0 && Offset <= TheSize)) &&
        (Count == dynamic_extent || (Count >= 0 && Offset + Count <= TheSize));
    if (!Valid)
      std::terminate();
    return DeviceMemorySpan<element_type>(ThePlatform, TheHandle, Count,
                                          TheOffset + Offset);
  }

private:
  template <typename T> friend class DeviceMemory;
  template <typename T> friend class DeviceMemorySpan;
  friend class Platform;

  DeviceMemorySpan(Platform *ThePlatform, pointer AHandle, index_type Size,
                   index_type Offset)
      : ThePlatform(ThePlatform), TheHandle(AHandle), TheSize(Size),
        TheOffset(Offset), TheSpanHandle(nullptr) {}

  Platform *ThePlatform;
  pointer TheHandle;
  index_type TheSize;
  index_type TheOffset;
  pointer TheSpanHandle;
};

/// Asynchronous host memory.
///
/// This memory is pinned or otherwise registered in the host memory space to
/// allow for asynchronous copies between it and device memory.
///
/// This memory unpins/unregisters itself when it goes out of scope, but does
/// not free itself.
template <typename ElementType> class AsyncHostMemory {
public:
  using value_type = ElementType;
  using remove_const_type = typename std::remove_const<ElementType>::type;

  AsyncHostMemory(const AsyncHostMemory &) = delete;
  AsyncHostMemory &operator=(const AsyncHostMemory &) = delete;
  AsyncHostMemory(AsyncHostMemory &&) noexcept;
  AsyncHostMemory &operator=(AsyncHostMemory &&) noexcept;
  ~AsyncHostMemory() = default;

  template <typename OtherElementType>
  AsyncHostMemory(AsyncHostMemory<OtherElementType> &&Other)
      : ThePointer(std::move(Other.ThePointer)),
        TheElementCount(Other.TheElementCount) {
    static_assert(
        std::is_assignable<ElementType *, OtherElementType *>::value,
        "cannot assign OtherElementType pointer to ElementType pointer type");
  }

  ElementType *data() const {
    return const_cast<ElementType *>(
        static_cast<remove_const_type *>(ThePointer.get()));
  }
  ptrdiff_t size() const { return TheElementCount; }

private:
  template <typename U> friend class AsyncHostMemory;
  friend class Platform;
  AsyncHostMemory(ElementType *Pointer, ptrdiff_t ElementCount,
                  HandleDestructor Destructor)
      : ThePointer(
            static_cast<void *>(const_cast<remove_const_type *>(Pointer)),
            Destructor),
        TheElementCount(ElementCount) {}

  std::unique_ptr<void, HandleDestructor> ThePointer;
  ptrdiff_t TheElementCount;
};

template <typename T>
AsyncHostMemory<T>::AsyncHostMemory(AsyncHostMemory &&) noexcept = default;
template <typename T>
AsyncHostMemory<T> &AsyncHostMemory<T>::
operator=(AsyncHostMemory &&) noexcept = default;

/// Owned registered host memory.
///
/// Like AsyncHostMemory, but this memory also frees itself in addition to
/// unpinning/unregistering itself when it goes out of scope.
template <typename ElementType> class OwnedAsyncHostMemory {
public:
  using remove_const_type = typename std::remove_const<ElementType>::type;

  OwnedAsyncHostMemory(const OwnedAsyncHostMemory &) = delete;
  OwnedAsyncHostMemory &operator=(const OwnedAsyncHostMemory &) = delete;
  OwnedAsyncHostMemory(OwnedAsyncHostMemory &&) noexcept;
  OwnedAsyncHostMemory &operator=(OwnedAsyncHostMemory &&) noexcept;

  ~OwnedAsyncHostMemory() {
    if (ThePointer.get()) {
      // We use placement new to construct these objects, so we have to call the
      // destructors explicitly.
      for (ptrdiff_t I = 0; I < TheElementCount; ++I)
        static_cast<ElementType *>(ThePointer.get())[I].~ElementType();
    }
  }

  ElementType *get() const {
    return const_cast<ElementType *>(
        static_cast<remove_const_type *>(ThePointer.get()));
  }

  ElementType &operator[](ptrdiff_t I) const {
    assert(I >= 0 && I < TheElementCount);
    return get()[I];
  }

private:
  template <typename T> friend class AsyncHostMemorySpan;

  friend class Platform;

  OwnedAsyncHostMemory(void *Memory, ptrdiff_t ElementCount,
                       HandleDestructor Destructor)
      : ThePointer(Memory, Destructor), TheElementCount(ElementCount) {}

  std::unique_ptr<void, HandleDestructor> ThePointer;
  ptrdiff_t TheElementCount;
};

template <typename T>
OwnedAsyncHostMemory<T>::OwnedAsyncHostMemory(
    OwnedAsyncHostMemory &&) noexcept = default;
template <typename T>
OwnedAsyncHostMemory<T> &OwnedAsyncHostMemory<T>::
operator=(OwnedAsyncHostMemory &&) noexcept = default;

/// View into registered host memory.
///
/// Like Span but for registered host memory.
template <typename ElementType> class AsyncHostMemorySpan {
public:
  /// \name constants and types
  /// \{
  using element_type = ElementType;
  using index_type = std::ptrdiff_t;
  using pointer = element_type *;
  using reference = element_type &;
  using iterator = element_type *;
  using const_iterator = const element_type *;
  using value_type = typename std::remove_const<element_type>::type;
  /// \}

  AsyncHostMemorySpan() : TheSpan() {}

  // Intentionally implicit.
  template <typename OtherElementType>
  AsyncHostMemorySpan(AsyncHostMemory<OtherElementType> &Memory)
      : TheSpan(Memory.data(), Memory.size()) {}

  // Intentionally implicit.
  template <typename OtherElementType>
  AsyncHostMemorySpan(OwnedAsyncHostMemory<OtherElementType> &Owned)
      : TheSpan(Owned.get(), Owned.TheElementCount) {}

  // Intentionally implicit.
  template <typename OtherElementType>
  AsyncHostMemorySpan(AsyncHostMemorySpan<OtherElementType> &ASpan)
      : TheSpan(ASpan) {}

  // Intentionally implicit.
  template <typename OtherElementType>
  AsyncHostMemorySpan(AsyncHostMemorySpan<OtherElementType> &&Span)
      : TheSpan(Span) {}

  /// \name observers
  /// \{
  index_type length() const { return TheSpan.length(); }
  index_type size() const { return TheSpan.size(); }
  index_type byte_size() const { // NOLINT
    return TheSpan.size() * sizeof(element_type);
  }
  bool empty() const { return TheSpan.empty(); }
  /// \}

  pointer data() const noexcept { return TheSpan.data(); }
  operator element_type *() const { return TheSpan.data(); }

  AsyncHostMemorySpan<element_type> first(index_type Count) const {
    return AsyncHostMemorySpan<element_type>(TheSpan.first(Count));
  }

  AsyncHostMemorySpan<element_type> last(index_type Count) const {
    return AsyncHostMemorySpan<element_type>(TheSpan.last(Count));
  }

  AsyncHostMemorySpan<element_type>
  subspan(index_type Offset, index_type Count = dynamic_extent) const {
    return AsyncHostMemorySpan<element_type>(TheSpan.subspan(Offset, Count));
  }

private:
  template <typename T> friend class AsyncHostMemory;

  explicit AsyncHostMemorySpan(Span<ElementType> ArraySpan)
      : TheSpan(ArraySpan) {}

  Span<ElementType> TheSpan;
};

} // namespace acxxel

#endif // ACXXEL_ACXXEL_H
