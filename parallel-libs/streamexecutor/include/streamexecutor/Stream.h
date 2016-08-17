//===-- Stream.h - A stream of execution ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// A Stream instance represents a queue of sequential, host-asynchronous work
/// to be performed on a device.
///
/// To enqueue work on a device, first create a Executor instance for a
/// given device and then use that Executor to create a Stream instance.
/// The Stream instance will perform its work on the device managed by the
/// Executor that created it.
///
/// The various "then" methods of the Stream object, such as thenMemcpyH2D and
/// thenLaunch, may be used to enqueue work on the Stream, and the
/// blockHostUntilDone() method may be used to block the host code until the
/// Stream has completed all its work.
///
/// Multiple Stream instances can be created for the same Executor. This
/// allows several independent streams of computation to be performed
/// simultaneously on a single device.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_STREAM_H
#define STREAMEXECUTOR_STREAM_H

#include <cassert>
#include <memory>
#include <string>

#include "streamexecutor/DeviceMemory.h"
#include "streamexecutor/Kernel.h"
#include "streamexecutor/LaunchDimensions.h"
#include "streamexecutor/PackedKernelArgumentArray.h"
#include "streamexecutor/PlatformInterfaces.h"
#include "streamexecutor/Utils/Error.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/RWMutex.h"

namespace streamexecutor {

/// Represents a stream of dependent computations on a device.
///
/// The operations within a stream execute sequentially and asynchronously until
/// blockHostUntilDone() is invoked, which synchronously joins host code with
/// the execution of the stream.
///
/// If any given operation fails when entraining work for the stream, isOK()
/// will indicate that an error has occurred and getStatus() will get the first
/// error that occurred on the stream. There is no way to clear the error state
/// of a stream once it is in an error state.
class Stream {
public:
  explicit Stream(std::unique_ptr<PlatformStreamHandle> PStream);

  ~Stream();

  /// Returns whether any error has occurred while entraining work on this
  /// stream.
  bool isOK() const {
    llvm::sys::ScopedReader ReaderLock(ErrorMessageMutex);
    return !ErrorMessage;
  }

  /// Returns the status created by the first error that occurred while
  /// entraining work on this stream.
  Error getStatus() const {
    llvm::sys::ScopedReader ReaderLock(ErrorMessageMutex);
    if (ErrorMessage)
      return make_error(*ErrorMessage);
    else
      return Error::success();
  };

  /// Entrains onto the stream of operations a kernel launch with the given
  /// arguments.
  ///
  /// These arguments can be device memory types like GlobalDeviceMemory<T> and
  /// SharedDeviceMemory<T>, or they can be primitive types such as int. The
  /// allowable argument types are determined by the template parameters to the
  /// TypedKernel argument.
  template <typename... ParameterTs>
  Stream &thenLaunch(BlockDimensions BlockSize, GridDimensions GridSize,
                     const TypedKernel<ParameterTs...> &Kernel,
                     const ParameterTs &... Arguments) {
    auto ArgumentArray =
        make_kernel_argument_pack<ParameterTs...>(Arguments...);
    setError(PExecutor->launch(ThePlatformStream.get(), BlockSize, GridSize,
                               Kernel, ArgumentArray));
    return *this;
  }

  /// Entrain onto the stream a memcpy of a given number of elements from a
  /// device source to a host destination.
  ///
  /// HostDst must be a pointer to host memory allocated by
  /// Executor::allocateHostMemory or otherwise allocated and then
  /// registered with Executor::registerHostMemory.
  template <typename T>
  Stream &thenMemcpyD2H(const GlobalDeviceMemory<T> &DeviceSrc,
                        llvm::MutableArrayRef<T> HostDst, size_t ElementCount) {
    if (ElementCount > DeviceSrc.getElementCount())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", from device memory array of size " +
               llvm::Twine(DeviceSrc.getElementCount()));
    else if (ElementCount > HostDst.size())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", to host array of size " + llvm::Twine(HostDst.size()));
    else
      setError(PExecutor->memcpyD2H(ThePlatformStream.get(), DeviceSrc,
                                    HostDst.data(), ElementCount * sizeof(T)));
    return *this;
  }

  /// Same as thenMemcpyD2H above, but copies the entire source to the
  /// destination.
  template <typename T>
  Stream &thenMemcpyD2H(const GlobalDeviceMemory<T> &DeviceSrc,
                        llvm::MutableArrayRef<T> HostDst) {
    return thenMemcpyD2H(DeviceSrc, HostDst, DeviceSrc.getElementCount());
  }

  /// Entrain onto the stream a memcpy of a given number of elements from a host
  /// source to a device destination.
  ///
  /// HostSrc must be a pointer to host memory allocated by
  /// Executor::allocateHostMemory or otherwise allocated and then
  /// registered with Executor::registerHostMemory.
  template <typename T>
  Stream &thenMemcpyH2D(llvm::ArrayRef<T> HostSrc,
                        GlobalDeviceMemory<T> *DeviceDst, size_t ElementCount) {
    if (ElementCount > HostSrc.size())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", from host array of size " + llvm::Twine(HostSrc.size()));
    else if (ElementCount > DeviceDst->getElementCount())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", to device memory array of size " +
               llvm::Twine(DeviceDst->getElementCount()));
    else
      setError(PExecutor->memcpyH2D(ThePlatformStream.get(), HostSrc.data(),
                                    DeviceDst, ElementCount * sizeof(T)));
    return *this;
  }

  /// Same as thenMemcpyH2D above, but copies the entire source to the
  /// destination.
  template <typename T>
  Stream &thenMemcpyH2D(llvm::ArrayRef<T> HostSrc,
                        GlobalDeviceMemory<T> *DeviceDst) {
    return thenMemcpyH2D(HostSrc, DeviceDst, HostSrc.size());
  }

  /// Entrain onto the stream a memcpy of a given number of elements from a
  /// device source to a device destination.
  template <typename T>
  Stream &thenMemcpyD2D(const GlobalDeviceMemory<T> &DeviceSrc,
                        GlobalDeviceMemory<T> *DeviceDst, size_t ElementCount) {
    if (ElementCount > DeviceSrc.getElementCount())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", from device memory array of size " +
               llvm::Twine(DeviceSrc.getElementCount()));
    else if (ElementCount > DeviceDst->getElementCount())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", to device memory array of size " +
               llvm::Twine(DeviceDst->getElementCount()));
    else
      setError(PExecutor->memcpyD2D(ThePlatformStream.get(), DeviceSrc,
                                    DeviceDst, ElementCount * sizeof(T)));
    return *this;
  }

  /// Same as thenMemcpyD2D above, but copies the entire source to the
  /// destination.
  template <typename T>
  Stream &thenMemcpyD2D(const GlobalDeviceMemory<T> &DeviceSrc,
                        GlobalDeviceMemory<T> *DeviceDst) {
    return thenMemcpyD2D(DeviceSrc, DeviceDst, DeviceSrc.getElementCount());
  }

  /// Blocks the host code, waiting for the operations entrained on the stream
  /// (enqueued up to this point in program execution) to complete.
  ///
  /// Returns true if there are no errors on the stream.
  bool blockHostUntilDone() {
    Error E = PExecutor->blockHostUntilDone(ThePlatformStream.get());
    bool returnValue = static_cast<bool>(E);
    setError(std::move(E));
    return returnValue;
  }

private:
  /// Sets the error state from an Error object.
  ///
  /// Does not overwrite the error if it is already set.
  void setError(Error &&E) {
    if (E) {
      llvm::sys::ScopedWriter WriterLock(ErrorMessageMutex);
      if (!ErrorMessage)
        ErrorMessage = consumeAndGetMessage(std::move(E));
    }
  }

  /// Sets the error state from an error message.
  ///
  /// Does not overwrite the error if it is already set.
  void setError(llvm::Twine Message) {
    llvm::sys::ScopedWriter WriterLock(ErrorMessageMutex);
    if (!ErrorMessage)
      ErrorMessage = Message.str();
  }

  /// The PlatformExecutor that supports the operations of this stream.
  PlatformExecutor *PExecutor;

  /// The platform-specific stream handle for this instance.
  std::unique_ptr<PlatformStreamHandle> ThePlatformStream;

  /// Mutex that guards the error state flags.
  ///
  /// Mutable so that it can be obtained via const reader lock.
  mutable llvm::sys::RWMutex ErrorMessageMutex;

  /// First error message for an operation in this stream or empty if there have
  /// been no errors.
  llvm::Optional<std::string> ErrorMessage;

  Stream(const Stream &) = delete;
  void operator=(const Stream &) = delete;
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_STREAM_H
