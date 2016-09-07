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
/// To enqueue work on a device, first create a Device instance then use that
/// Device to create a Stream instance. The Stream instance will perform its
/// work on the device managed by the Device object that created it.
///
/// The various "then" methods of the Stream object, such as thenCopyH2D and
/// thenLaunch, may be used to enqueue work on the Stream, and the
/// blockHostUntilDone() method may be used to block the host code until the
/// Stream has completed all its work.
///
/// Multiple Stream instances can be created for the same Device. This allows
/// several independent streams of computation to be performed simultaneously on
/// a single device.
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
#include "streamexecutor/PlatformDevice.h"
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
  Stream(PlatformDevice *D, const void *PlatformStreamHandle);

  Stream(const Stream &Other) = delete;
  Stream &operator=(const Stream &Other) = delete;

  Stream(Stream &&Other);
  Stream &operator=(Stream &&Other);

  ~Stream();

  /// Returns whether any error has occurred while entraining work on this
  /// stream.
  bool isOK() const {
    llvm::sys::ScopedReader ReaderLock(*ErrorMessageMutex);
    return !ErrorMessage;
  }

  /// Returns the status created by the first error that occurred while
  /// entraining work on this stream.
  Error getStatus() const {
    llvm::sys::ScopedReader ReaderLock(*ErrorMessageMutex);
    if (ErrorMessage)
      return make_error(*ErrorMessage);
    else
      return Error::success();
  }

  // Blocks the calling host thread until all work enqueued on this Stream
  // completes.
  //
  // Returns the result of getStatus() after the Stream work completes.
  Error blockHostUntilDone() {
    setError(PDevice->blockHostUntilDone(PlatformStreamHandle));
    return getStatus();
  }

  /// Entrains onto the stream of operations a kernel launch with the given
  /// arguments.
  ///
  /// These arguments can be device memory types like GlobalDeviceMemory<T> and
  /// SharedDeviceMemory<T>, or they can be primitive types such as int. The
  /// allowable argument types are determined by the template parameters to the
  /// Kernel argument.
  template <typename... ParameterTs>
  Stream &thenLaunch(BlockDimensions BlockSize, GridDimensions GridSize,
                     const Kernel<ParameterTs...> &K,
                     const ParameterTs &... Arguments) {
    auto ArgumentArray =
        make_kernel_argument_pack<ParameterTs...>(Arguments...);
    setError(PDevice->launch(PlatformStreamHandle, BlockSize, GridSize,
                             K.getPlatformHandle(), ArgumentArray));
    return *this;
  }

  /// \name Device memory copying functions
  ///
  /// These methods enqueue a device memory copy operation on the stream and
  /// return without waiting for the operation to complete.
  ///
  /// Any host memory used as a source or destination for one of these
  /// operations must be allocated with Device::allocateHostMemory or registered
  /// with Device::registerHostMemory. Otherwise, the enqueuing operation may
  /// block until the copy operation is fully complete.
  ///
  /// The arguments and bounds checking for these methods match the API of the
  /// \ref DeviceHostSyncCopyGroup
  /// "host-synchronous device memory copying functions" of Device.
  ///@{

  template <typename T>
  Stream &thenCopyD2H(GlobalDeviceMemorySlice<T> Src,
                      llvm::MutableArrayRef<T> Dst, size_t ElementCount) {
    if (ElementCount > Src.getElementCount())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", from a device array of element count " +
               llvm::Twine(Src.getElementCount()));
    else if (ElementCount > Dst.size())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", to a host array of element count " + llvm::Twine(Dst.size()));
    else
      setError(PDevice->copyD2H(PlatformStreamHandle,
                                Src.getBaseMemory().getHandle(),
                                Src.getElementOffset() * sizeof(T), Dst.data(),
                                0, ElementCount * sizeof(T)));
    return *this;
  }

  template <typename T>
  Stream &thenCopyD2H(GlobalDeviceMemorySlice<T> Src,
                      llvm::MutableArrayRef<T> Dst) {
    if (Src.getElementCount() != Dst.size())
      setError("array size mismatch for D2H, device source has element count " +
               llvm::Twine(Src.getElementCount()) +
               " but host destination has element count " +
               llvm::Twine(Dst.size()));
    else
      thenCopyD2H(Src, Dst, Src.getElementCount());
    return *this;
  }

  template <typename T>
  Stream &thenCopyD2H(GlobalDeviceMemorySlice<T> Src, T *Dst,
                      size_t ElementCount) {
    thenCopyD2H(Src, llvm::MutableArrayRef<T>(Dst, ElementCount), ElementCount);
    return *this;
  }

  template <typename T>
  Stream &thenCopyD2H(const GlobalDeviceMemory<T> &Src,
                      llvm::MutableArrayRef<T> Dst, size_t ElementCount) {
    thenCopyD2H(Src.asSlice(), Dst, ElementCount);
    return *this;
  }

  template <typename T>
  Stream &thenCopyD2H(const GlobalDeviceMemory<T> &Src,
                      llvm::MutableArrayRef<T> Dst) {
    thenCopyD2H(Src.asSlice(), Dst);
    return *this;
  }

  template <typename T>
  Stream &thenCopyD2H(const GlobalDeviceMemory<T> &Src, T *Dst,
                      size_t ElementCount) {
    thenCopyD2H(Src.asSlice(), Dst, ElementCount);
    return *this;
  }

  template <typename T>
  Stream &thenCopyH2D(llvm::ArrayRef<T> Src, GlobalDeviceMemorySlice<T> Dst,
                      size_t ElementCount) {
    if (ElementCount > Src.size())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", from a host array of element count " +
               llvm::Twine(Src.size()));
    else if (ElementCount > Dst.getElementCount())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", to a device array of element count " +
               llvm::Twine(Dst.getElementCount()));
    else
      setError(PDevice->copyH2D(
          PlatformStreamHandle, Src.data(), 0, Dst.getBaseMemory().getHandle(),
          Dst.getElementOffset() * sizeof(T), ElementCount * sizeof(T)));
    return *this;
  }

  template <typename T>
  Stream &thenCopyH2D(llvm::ArrayRef<T> Src, GlobalDeviceMemorySlice<T> Dst) {
    if (Src.size() != Dst.getElementCount())
      setError("array size mismatch for H2D, host source has element count " +
               llvm::Twine(Src.size()) +
               " but device destination has element count " +
               llvm::Twine(Dst.getElementCount()));
    else
      thenCopyH2D(Src, Dst, Dst.getElementCount());
    return *this;
  }

  template <typename T>
  Stream &thenCopyH2D(T *Src, GlobalDeviceMemorySlice<T> Dst,
                      size_t ElementCount) {
    thenCopyH2D(llvm::ArrayRef<T>(Src, ElementCount), Dst, ElementCount);
    return *this;
  }

  template <typename T>
  Stream &thenCopyH2D(llvm::ArrayRef<T> Src, GlobalDeviceMemory<T> &Dst,
                      size_t ElementCount) {
    thenCopyH2D(Src, Dst.asSlice(), ElementCount);
    return *this;
  }

  template <typename T>
  Stream &thenCopyH2D(llvm::ArrayRef<T> Src, GlobalDeviceMemory<T> &Dst) {
    thenCopyH2D(Src, Dst.asSlice());
    return *this;
  }

  template <typename T>
  Stream &thenCopyH2D(T *Src, GlobalDeviceMemory<T> &Dst, size_t ElementCount) {
    thenCopyH2D(Src, Dst.asSlice(), ElementCount);
    return *this;
  }

  template <typename T>
  Stream &thenCopyD2D(GlobalDeviceMemorySlice<T> Src,
                      GlobalDeviceMemorySlice<T> Dst, size_t ElementCount) {
    if (ElementCount > Src.getElementCount())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", from a device array of element count " +
               llvm::Twine(Src.getElementCount()));
    else if (ElementCount > Dst.getElementCount())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", to a device array of element count " +
               llvm::Twine(Dst.getElementCount()));
    else
      setError(PDevice->copyD2D(
          PlatformStreamHandle, Src.getBaseMemory().getHandle(),
          Src.getElementOffset() * sizeof(T), Dst.getBaseMemory().getHandle(),
          Dst.getElementOffset() * sizeof(T), ElementCount * sizeof(T)));
    return *this;
  }

  template <typename T>
  Stream &thenCopyD2D(GlobalDeviceMemorySlice<T> Src,
                      GlobalDeviceMemorySlice<T> Dst) {
    if (Src.getElementCount() != Dst.getElementCount())
      setError("array size mismatch for D2D, device source has element count " +
               llvm::Twine(Src.getElementCount()) +
               " but device destination has element count " +
               llvm::Twine(Dst.getElementCount()));
    else
      thenCopyD2D(Src, Dst, Src.getElementCount());
    return *this;
  }

  template <typename T>
  Stream &thenCopyD2D(const GlobalDeviceMemory<T> &Src,
                      GlobalDeviceMemorySlice<T> Dst, size_t ElementCount) {
    thenCopyD2D(Src.asSlice(), Dst, ElementCount);
    return *this;
  }

  template <typename T>
  Stream &thenCopyD2D(const GlobalDeviceMemory<T> &Src,
                      GlobalDeviceMemorySlice<T> Dst) {
    thenCopyD2D(Src.asSlice(), Dst);
    return *this;
  }

  template <typename T>
  Stream &thenCopyD2D(GlobalDeviceMemorySlice<T> Src,
                      GlobalDeviceMemory<T> &Dst, size_t ElementCount) {
    thenCopyD2D(Src, Dst.asSlice(), ElementCount);
    return *this;
  }

  template <typename T>
  Stream &thenCopyD2D(GlobalDeviceMemorySlice<T> Src,
                      GlobalDeviceMemory<T> &Dst) {
    thenCopyD2D(Src, Dst.asSlice());
    return *this;
  }

  template <typename T>
  Stream &thenCopyD2D(const GlobalDeviceMemory<T> &Src,
                      GlobalDeviceMemory<T> &Dst, size_t ElementCount) {
    thenCopyD2D(Src.asSlice(), Dst.asSlice(), ElementCount);
    return *this;
  }

  template <typename T>
  Stream &thenCopyD2D(const GlobalDeviceMemory<T> &Src,
                      GlobalDeviceMemory<T> &Dst) {
    thenCopyD2D(Src.asSlice(), Dst.asSlice());
    return *this;
  }

  ///@} End device memory copying functions

private:
  /// Sets the error state from an Error object.
  ///
  /// Does not overwrite the error if it is already set.
  void setError(Error &&E) {
    if (E) {
      llvm::sys::ScopedWriter WriterLock(*ErrorMessageMutex);
      if (!ErrorMessage)
        ErrorMessage = consumeAndGetMessage(std::move(E));
    }
  }

  /// Sets the error state from an error message.
  ///
  /// Does not overwrite the error if it is already set.
  void setError(llvm::Twine Message) {
    llvm::sys::ScopedWriter WriterLock(*ErrorMessageMutex);
    if (!ErrorMessage)
      ErrorMessage = Message.str();
  }

  /// The PlatformDevice that supports the operations of this stream.
  PlatformDevice *PDevice;

  /// The platform-specific stream handle for this instance.
  const void *PlatformStreamHandle;

  /// Mutex that guards the error state flags.
  std::unique_ptr<llvm::sys::RWMutex> ErrorMessageMutex;

  /// First error message for an operation in this stream or empty if there have
  /// been no errors.
  llvm::Optional<std::string> ErrorMessage;
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_STREAM_H
