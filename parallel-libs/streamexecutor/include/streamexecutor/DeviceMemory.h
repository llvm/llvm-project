//===-- DeviceMemory.h - Types representing device memory -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines types that represent device memory buffers. Two memory
/// spaces are represented here: global and shared. Host code can have a handle
/// to device global memory, and that handle can be used to copy data to and
/// from the device. Host code cannot have a handle to device shared memory
/// because that memory only exists during the execution of a kernel.
///
/// GlobalDeviceMemory<T> is a handle to an array of elements of type T in
/// global device memory. It is similar to a pair of a std::unique_ptr<T> and an
/// element count to tell how many elements of type T fit in the memory pointed
/// to by that T*.
///
/// SharedDeviceMemory<T> is just the size in elements of an array of elements
/// of type T in device shared memory. No resources are actually attached to
/// this class, it is just like a memo to the device to allocate space in shared
/// memory.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_DEVICEMEMORY_H
#define STREAMEXECUTOR_DEVICEMEMORY_H

#include <cassert>
#include <cstddef>

#include "streamexecutor/Error.h"

namespace streamexecutor {

class Device;

template <typename ElemT> class GlobalDeviceMemory;

/// Reference to a slice of device memory.
///
/// Contains a base memory handle, an element count offset into that base
/// memory, and an element count for the size of the slice.
template <typename ElemT> class GlobalDeviceMemorySlice {
public:
  using ElementTy = ElemT;

  /// Intentionally implicit so GlobalDeviceMemory<T> can be passed to functions
  /// expecting GlobalDeviceMemorySlice<T> arguments.
  GlobalDeviceMemorySlice(const GlobalDeviceMemory<ElemT> &Memory)
      : BaseMemory(Memory), ElementOffset(0),
        ElementCount(Memory.getElementCount()) {}

  GlobalDeviceMemorySlice(const GlobalDeviceMemory<ElemT> &BaseMemory,
                          size_t ElementOffset, size_t ElementCount)
      : BaseMemory(BaseMemory), ElementOffset(ElementOffset),
        ElementCount(ElementCount) {
    assert(ElementOffset + ElementCount <= BaseMemory.getElementCount() &&
           "slicing past the end of a GlobalDeviceMemory buffer");
  }

  /// Gets the GlobalDeviceMemory backing this slice.
  const GlobalDeviceMemory<ElemT> &getBaseMemory() const { return BaseMemory; }

  /// Gets the offset of this slice from the base memory.
  ///
  /// The offset is measured in elements, not bytes.
  size_t getElementOffset() const { return ElementOffset; }

  /// Gets the number of elements in this slice.
  size_t getElementCount() const { return ElementCount; }

  /// Returns the number of bytes that can fit in this slice.
  size_t getByteCount() const { return ElementCount * sizeof(ElemT); }

  /// Creates a slice of the memory with the first DropCount elements removed.
  LLVM_ATTRIBUTE_UNUSED_RESULT
  GlobalDeviceMemorySlice<ElemT> slice(size_t DropCount) const {
    assert(DropCount <= ElementCount &&
           "dropping more than the size of a slice");
    return GlobalDeviceMemorySlice<ElemT>(BaseMemory, ElementOffset + DropCount,
                                          ElementCount - DropCount);
  }

  /// Creates a slice of the memory with the last DropCount elements removed.
  LLVM_ATTRIBUTE_UNUSED_RESULT
  GlobalDeviceMemorySlice<ElemT> drop_back(size_t DropCount) const {
    assert(DropCount <= ElementCount &&
           "dropping more than the size of a slice");
    return GlobalDeviceMemorySlice<ElemT>(BaseMemory, ElementOffset,
                                          ElementCount - DropCount);
  }

  /// Creates a slice of the memory that chops off the first DropCount elements
  /// and keeps the next TakeCount elements.
  LLVM_ATTRIBUTE_UNUSED_RESULT
  GlobalDeviceMemorySlice<ElemT> slice(size_t DropCount,
                                       size_t TakeCount) const {
    assert(DropCount + TakeCount <= ElementCount &&
           "sub-slice operation overruns slice bounds");
    return GlobalDeviceMemorySlice<ElemT>(BaseMemory, ElementOffset + DropCount,
                                          TakeCount);
  }

private:
  const GlobalDeviceMemory<ElemT> &BaseMemory;
  size_t ElementOffset;
  size_t ElementCount;
};

/// Wrapper around a generic global device memory allocation.
///
/// This class represents a buffer of untyped bytes in the global memory space
/// of a device. See GlobalDeviceMemory<T> for the corresponding type that
/// includes type information for the elements in its buffer.
///
/// This is effectively a pair consisting of an opaque handle and a buffer size
/// in bytes. The opaque handle is a platform-dependent handle to the actual
/// memory that is allocated on the device.
///
/// In some cases, such as in the CUDA platform, the opaque handle may actually
/// be a pointer in the virtual address space and it may be valid to perform
/// arithmetic on it to obtain other device pointers, but this is not the case
/// in general.
///
/// For example, in the OpenCL platform, the handle is a pointer to a _cl_mem
/// handle object which really is completely opaque to the user.
class GlobalDeviceMemoryBase {
public:
  /// Returns an opaque handle to the underlying memory.
  const void *getHandle() const { return Handle; }

  // Cannot copy because the handle must be owned by a single object.
  GlobalDeviceMemoryBase(const GlobalDeviceMemoryBase &) = delete;
  GlobalDeviceMemoryBase &operator=(const GlobalDeviceMemoryBase &) = delete;

protected:
  /// Creates a GlobalDeviceMemoryBase from a handle and a byte count.
  GlobalDeviceMemoryBase(Device *D, const void *Handle, size_t ByteCount)
      : TheDevice(D), Handle(Handle), ByteCount(ByteCount) {}

  /// Transfer ownership of the underlying handle.
  GlobalDeviceMemoryBase(GlobalDeviceMemoryBase &&Other)
      : TheDevice(Other.TheDevice), Handle(Other.Handle),
        ByteCount(Other.ByteCount) {
    Other.TheDevice = nullptr;
    Other.Handle = nullptr;
    Other.ByteCount = 0;
  }

  GlobalDeviceMemoryBase &operator=(GlobalDeviceMemoryBase &&Other) {
    TheDevice = Other.TheDevice;
    Handle = Other.Handle;
    ByteCount = Other.ByteCount;
    Other.TheDevice = nullptr;
    Other.Handle = nullptr;
    Other.ByteCount = 0;
    return *this;
  }

  ~GlobalDeviceMemoryBase();

  Device *TheDevice;  // Pointer to the device on which this memory lives.
  const void *Handle; // Platform-dependent value representing allocated memory.
  size_t ByteCount;   // Size in bytes of this allocation.
};

/// Typed wrapper around the "void *"-like GlobalDeviceMemoryBase class.
///
/// For example, GlobalDeviceMemory<int> is a simple wrapper around
/// GlobalDeviceMemoryBase that represents a buffer of integers stored in global
/// device memory.
template <typename ElemT>
class GlobalDeviceMemory : public GlobalDeviceMemoryBase {
public:
  using ElementTy = ElemT;

  GlobalDeviceMemory(GlobalDeviceMemory &&Other) = default;
  GlobalDeviceMemory &operator=(GlobalDeviceMemory &&Other) = default;

  /// Returns the number of elements of type ElemT that constitute this
  /// allocation.
  size_t getElementCount() const { return ByteCount / sizeof(ElemT); }

  /// Returns the number of bytes that can fit in this memory buffer.
  size_t getByteCount() const { return ByteCount; }

  /// Converts this memory object into a slice.
  GlobalDeviceMemorySlice<ElemT> asSlice() const {
    return GlobalDeviceMemorySlice<ElemT>(*this);
  }

private:
  GlobalDeviceMemory(const GlobalDeviceMemory &) = delete;
  GlobalDeviceMemory &operator=(const GlobalDeviceMemory &) = delete;

  // Only a Device can create a GlobalDeviceMemory instance.
  friend Device;
  GlobalDeviceMemory(Device *D, const void *Handle, size_t ElementCount)
      : GlobalDeviceMemoryBase(D, Handle, ElementCount * sizeof(ElemT)) {}
};

/// A class to represent the size of a dynamic shared memory buffer of elements
/// of type T on a device.
///
/// Shared memory buffers exist only on the device and cannot be manipulated
/// from the host, so instances of this class do not have an opaque handle, only
/// a size.
///
/// This type of memory is called "local" memory in OpenCL and "shared" memory
/// in CUDA, and both platforms follow the rule that the host code only knows
/// the size of these buffers and does not have a handle to them.
///
/// The treatment of shared memory in StreamExecutor matches the way it is done
/// in OpenCL, where a kernel takes any number of shared memory sizes as kernel
/// function arguments.
///
/// In CUDA only one shared memory size argument is allowed per kernel call.
/// StreamExecutor handles this by allowing CUDA kernel signatures that take
/// multiple SharedDeviceMemory arguments, and simply adding together all the
/// shared memory sizes to get the final shared memory size that is used to
/// launch the kernel.
template <typename ElemT> class SharedDeviceMemory {
public:
  /// Creates a typed area of shared device memory with a given number of
  /// elements.
  static SharedDeviceMemory<ElemT> makeFromElementCount(size_t ElementCount) {
    return SharedDeviceMemory(ElementCount);
  }

  /// Copyable because it is just an array size.
  SharedDeviceMemory(const SharedDeviceMemory &) = default;

  /// Copy-assignable because it is just an array size.
  SharedDeviceMemory &operator=(const SharedDeviceMemory &) = default;

  /// Returns the number of elements of type ElemT that can fit in this memory
  /// buffer.
  size_t getElementCount() const { return ElementCount; }

  /// Returns the number of bytes that can fit in this memory buffer.
  size_t getByteCount() const { return ElementCount * sizeof(ElemT); }

  /// Returns whether this is a single-element memory buffer.
  bool isScalar() const { return getElementCount() == 1; }

private:
  /// Constructs a SharedDeviceMemory instance from an element count.
  ///
  /// This constructor is not public because there is a potential for confusion
  /// between the size of the buffer in bytes and the size of the buffer in
  /// elements.
  ///
  /// The static method makeFromElementCount is provided for users of this class
  /// because its name makes the meaning of the size parameter clear.
  explicit SharedDeviceMemory(size_t ElementCount)
      : ElementCount(ElementCount) {}

  size_t ElementCount;
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_DEVICEMEMORY_H
