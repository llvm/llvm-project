//===-- HostMemory.h - Types for registered host memory ---------*- C++ -*-===//
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
/// This file defines types that represent registered host memory buffers. Host
/// memory must be registered to participate in asynchronous copies to or from
/// device memory.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_HOSTMEMORY_H
#define STREAMEXECUTOR_HOSTMEMORY_H

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "llvm/ADT/ArrayRef.h"

namespace streamexecutor {

class Device;
template <typename ElemT> class RegisteredHostMemory;

/// A mutable slice of registered host memory.
///
/// The memory is registered in the sense of
/// streamexecutor::Device::registerHostMemory.
///
/// Holds a reference to an underlying registered host memory buffer. Must not
/// be used after the underlying buffer is freed or unregistered.
template <typename ElemT> class MutableRegisteredHostMemorySlice {
public:
  using ElementTy = ElemT;

  MutableRegisteredHostMemorySlice(RegisteredHostMemory<ElemT> &Registered)
      : MutableArrayRef(Registered.getPointer(), Registered.getElementCount()) {
  }

  ElemT *getPointer() const { return MutableArrayRef.data(); }
  size_t getElementCount() const { return MutableArrayRef.size(); }

  /// Chops off the first N elements of the slice.
  MutableRegisteredHostMemorySlice slice(size_t N) const {
    return MutableRegisteredHostMemorySlice(MutableArrayRef.slice(N));
  }

  /// Chops off the first N elements of the slice and keeps the next M elements.
  MutableRegisteredHostMemorySlice slice(size_t N, size_t M) const {
    return MutableRegisteredHostMemorySlice(MutableArrayRef.slice(N, M));
  }

  /// Chops off the last N elements of the slice.
  MutableRegisteredHostMemorySlice drop_back(size_t N) const {
    return MutableRegisteredHostMemorySlice(MutableArrayRef.drop_back(N));
  }

private:
  MutableRegisteredHostMemorySlice(llvm::MutableArrayRef<ElemT> MutableArrayRef)
      : MutableArrayRef(MutableArrayRef) {}

  llvm::MutableArrayRef<ElemT> MutableArrayRef;
};

/// An immutable slice of registered host memory.
///
/// The memory is registered in the sense of
/// streamexecutor::Device::registerHostMemory.
///
/// Holds a reference to an underlying registered host memory buffer. Must not
/// be used after the underlying buffer is freed or unregistered.
template <typename ElemT> class RegisteredHostMemorySlice {
public:
  using ElementTy = ElemT;

  RegisteredHostMemorySlice(const RegisteredHostMemory<ElemT> &Registered)
      : ArrayRef(Registered.getPointer(), Registered.getElementCount()) {}

  RegisteredHostMemorySlice(
      MutableRegisteredHostMemorySlice<ElemT> MutableSlice)
      : ArrayRef(MutableSlice.getPointer(), MutableSlice.getElementCount()) {}

  const ElemT *getPointer() const { return ArrayRef.data(); }
  size_t getElementCount() const { return ArrayRef.size(); }

  /// Chops off the first N elements of the slice.
  RegisteredHostMemorySlice slice(size_t N) const {
    return RegisteredHostMemorySlice(ArrayRef.slice(N));
  }

  /// Chops off the first N elements of the slice and keeps the next M elements.
  RegisteredHostMemorySlice slice(size_t N, size_t M) const {
    return RegisteredHostMemorySlice(ArrayRef.slice(N, M));
  }

  /// Chops off the last N elements of the slice.
  RegisteredHostMemorySlice drop_back(size_t N) const {
    return RegisteredHostMemorySlice(ArrayRef.drop_back(N));
  }

private:
  llvm::ArrayRef<ElemT> ArrayRef;
};

namespace internal {

/// Helper function to unregister host memory.
///
/// This is a thin wrapper around streamexecutor::Device::unregisterHostMemory.
/// It is defined so this operation can be performed from the destructor of the
/// template class RegisteredHostMemory without including Device.h in this
/// header and creating a header inclusion cycle.
void destroyRegisteredHostMemoryInternals(Device *TheDevice, void *Pointer);

} // namespace internal

/// Registered host memory that knows how to unregister itself upon destruction.
///
/// The memory is registered in the sense of
/// streamexecutor::Device::registerHostMemory.
///
/// ElemT is the type of element stored in the host buffer.
template <typename ElemT> class RegisteredHostMemory {
public:
  using ElementTy = ElemT;

  RegisteredHostMemory(Device *TheDevice, ElemT *Pointer, size_t ElementCount)
      : TheDevice(TheDevice), Pointer(Pointer), ElementCount(ElementCount) {
    assert(TheDevice != nullptr && "cannot construct a "
                                   "RegisteredHostMemoryBase with a null "
                                   "platform device");
  }

  RegisteredHostMemory(const RegisteredHostMemory &) = delete;
  RegisteredHostMemory &operator=(const RegisteredHostMemory &) = delete;

  RegisteredHostMemory(RegisteredHostMemory &&Other)
      : TheDevice(Other.TheDevice), Pointer(Other.Pointer),
        ElementCount(Other.ElementCount) {
    Other.TheDevice = nullptr;
    Other.Pointer = nullptr;
  }

  RegisteredHostMemory &operator=(RegisteredHostMemory &&Other) {
    TheDevice = Other.TheDevice;
    Pointer = Other.Pointer;
    ElementCount = Other.ElementCount;
    Other.TheDevice = nullptr;
    Other.Pointer = nullptr;
  }

  ~RegisteredHostMemory() {
    internal::destroyRegisteredHostMemoryInternals(TheDevice, Pointer);
  }

  ElemT *getPointer() { return static_cast<ElemT *>(Pointer); }
  const ElemT *getPointer() const { return static_cast<ElemT *>(Pointer); }
  size_t getElementCount() const { return ElementCount; }

  /// Creates an immutable slice for the entire contents of this memory.
  RegisteredHostMemorySlice<ElemT> asSlice() const {
    return RegisteredHostMemorySlice<ElemT>(*this);
  }

  /// Creates a mutable slice for the entire contents of this memory.
  MutableRegisteredHostMemorySlice<ElemT> asSlice() {
    return MutableRegisteredHostMemorySlice<ElemT>(*this);
  }

private:
  Device *TheDevice;
  void *Pointer;
  size_t ElementCount;
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_HOSTMEMORY_H
