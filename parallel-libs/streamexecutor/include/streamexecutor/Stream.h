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
#include <type_traits>

#include "streamexecutor/DeviceMemory.h"
#include "streamexecutor/Error.h"
#include "streamexecutor/HostMemory.h"
#include "streamexecutor/Kernel.h"
#include "streamexecutor/LaunchDimensions.h"
#include "streamexecutor/PackedKernelArgumentArray.h"
#include "streamexecutor/PlatformDevice.h"

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
  /// The arguments and bounds checking for these methods match the API of the
  /// \ref DeviceHostSyncCopyGroup
  /// "host-synchronous device memory copying functions" of Device.
  ///
  /// The template types SrcTy and DstTy must match the following constraints:
  ///   * Must define typename ElementTy (the type of element stored in the
  ///   memory);
  ///   * ElementTy for the source argument must be the same as ElementTy for
  ///     the destination argument;
  ///   * Must be convertible to the correct slice type:
  ///     * GlobalDeviceMemorySlice<ElementTy> for device memory arguments,
  ///     * RegisteredHostMemorySlice<ElementTy> for host memory source
  ///       arguments,
  ///     * MutableRegisteredHostMemorySlice<ElementT> for host memory
  ///       destination arguments.
  ///@{

  // D2H

  template <typename SrcTy, typename DstTy>
  Stream &thenCopyD2H(SrcTy &&Src, DstTy &&Dst, size_t ElementCount) {
    using SrcElemTy = typename std::remove_reference<SrcTy>::type::ElementTy;
    using DstElemTy = typename std::remove_reference<DstTy>::type::ElementTy;
    static_assert(std::is_same<SrcElemTy, DstElemTy>::value,
                  "src/dst element type mismatch for thenCopyD2H");
    GlobalDeviceMemorySlice<SrcElemTy> SrcSlice(Src);
    MutableRegisteredHostMemorySlice<DstElemTy> DstSlice(Dst);
    if (ElementCount > Src.getElementCount())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", from a device array of element count " +
               llvm::Twine(SrcSlice.getElementCount()));
    else if (ElementCount > DstSlice.getElementCount())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", to a host array of element count " +
               llvm::Twine(DstSlice.getElementCount()));
    else
      setError(PDevice->copyD2H(
          PlatformStreamHandle, SrcSlice.getBaseMemory().getHandle(),
          SrcSlice.getElementOffset() * sizeof(SrcElemTy),
          DstSlice.getPointer(), 0, ElementCount * sizeof(DstElemTy)));
    return *this;
  }

  template <typename SrcTy, typename DstTy>
  Stream &thenCopyD2H(SrcTy &&Src, DstTy &&Dst) {
    using SrcElemTy = typename std::remove_reference<SrcTy>::type::ElementTy;
    using DstElemTy = typename std::remove_reference<DstTy>::type::ElementTy;
    static_assert(std::is_same<SrcElemTy, DstElemTy>::value,
                  "src/dst element type mismatch for thenCopyD2H");
    GlobalDeviceMemorySlice<SrcElemTy> SrcSlice(Src);
    MutableRegisteredHostMemorySlice<DstElemTy> DstSlice(Dst);
    if (SrcSlice.getElementCount() != DstSlice.getElementCount())
      setError("array size mismatch for D2H, device source has element count " +
               llvm::Twine(SrcSlice.getElementCount()) +
               " but host destination has element count " +
               llvm::Twine(DstSlice.getElementCount()));
    else
      thenCopyD2H(SrcSlice, DstSlice, SrcSlice.getElementCount());
    return *this;
  }

  // H2D

  template <typename SrcTy, typename DstTy>
  Stream &thenCopyH2D(SrcTy &&Src, DstTy &&Dst, size_t ElementCount) {
    using SrcElemTy = typename std::remove_reference<SrcTy>::type::ElementTy;
    using DstElemTy = typename std::remove_reference<DstTy>::type::ElementTy;
    static_assert(std::is_same<SrcElemTy, DstElemTy>::value,
                  "src/dst element type mismatch for thenCopyH2D");
    RegisteredHostMemorySlice<SrcElemTy> SrcSlice(Src);
    GlobalDeviceMemorySlice<DstElemTy> DstSlice(Dst);
    if (ElementCount > SrcSlice.getElementCount())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", from a host array of element count " +
               llvm::Twine(SrcSlice.getElementCount()));
    else if (ElementCount > DstSlice.getElementCount())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", to a device array of element count " +
               llvm::Twine(DstSlice.getElementCount()));
    else
      setError(PDevice->copyH2D(PlatformStreamHandle, SrcSlice.getPointer(), 0,
                                DstSlice.getBaseMemory().getHandle(),
                                DstSlice.getElementOffset() * sizeof(DstElemTy),
                                ElementCount * sizeof(SrcElemTy)));
    return *this;
  }

  template <typename SrcTy, typename DstTy>
  Stream &thenCopyH2D(SrcTy &&Src, DstTy &&Dst) {
    using SrcElemTy = typename std::remove_reference<SrcTy>::type::ElementTy;
    using DstElemTy = typename std::remove_reference<DstTy>::type::ElementTy;
    static_assert(std::is_same<SrcElemTy, DstElemTy>::value,
                  "src/dst element type mismatch for thenCopyH2D");
    RegisteredHostMemorySlice<SrcElemTy> SrcSlice(Src);
    GlobalDeviceMemorySlice<DstElemTy> DstSlice(Dst);
    if (SrcSlice.getElementCount() != DstSlice.getElementCount())
      setError("array size mismatch for H2D, host source has element count " +
               llvm::Twine(SrcSlice.getElementCount()) +
               " but device destination has element count " +
               llvm::Twine(DstSlice.getElementCount()));
    else
      thenCopyH2D(SrcSlice, DstSlice, DstSlice.getElementCount());
    return *this;
  }

  // D2D

  template <typename SrcTy, typename DstTy>
  Stream &thenCopyD2D(SrcTy &&Src, DstTy &&Dst, size_t ElementCount) {
    using SrcElemTy = typename std::remove_reference<SrcTy>::type::ElementTy;
    using DstElemTy = typename std::remove_reference<DstTy>::type::ElementTy;
    static_assert(std::is_same<SrcElemTy, DstElemTy>::value,
                  "src/dst element type mismatch for thenCopyD2D");
    GlobalDeviceMemorySlice<SrcElemTy> SrcSlice(Src);
    GlobalDeviceMemorySlice<DstElemTy> DstSlice(Dst);
    if (ElementCount > SrcSlice.getElementCount())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", from a device array of element count " +
               llvm::Twine(SrcSlice.getElementCount()));
    else if (ElementCount > DstSlice.getElementCount())
      setError("copying too many elements, " + llvm::Twine(ElementCount) +
               ", to a device array of element count " +
               llvm::Twine(DstSlice.getElementCount()));
    else
      setError(PDevice->copyD2D(PlatformStreamHandle,
                                SrcSlice.getBaseMemory().getHandle(),
                                SrcSlice.getElementOffset() * sizeof(SrcElemTy),
                                DstSlice.getBaseMemory().getHandle(),
                                DstSlice.getElementOffset() * sizeof(DstElemTy),
                                ElementCount * sizeof(SrcElemTy)));
    return *this;
  }

  template <typename SrcTy, typename DstTy>
  Stream &thenCopyD2D(SrcTy &&Src, DstTy &&Dst) {
    using SrcElemTy = typename std::remove_reference<SrcTy>::type::ElementTy;
    using DstElemTy = typename std::remove_reference<DstTy>::type::ElementTy;
    static_assert(std::is_same<SrcElemTy, DstElemTy>::value,
                  "src/dst element type mismatch for thenCopyD2D");
    GlobalDeviceMemorySlice<SrcElemTy> SrcSlice(Src);
    GlobalDeviceMemorySlice<DstElemTy> DstSlice(Dst);
    if (SrcSlice.getElementCount() != DstSlice.getElementCount())
      setError("array size mismatch for D2D, device source has element count " +
               llvm::Twine(SrcSlice.getElementCount()) +
               " but device destination has element count " +
               llvm::Twine(DstSlice.getElementCount()));
    else
      thenCopyD2D(SrcSlice, DstSlice, SrcSlice.getElementCount());
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
