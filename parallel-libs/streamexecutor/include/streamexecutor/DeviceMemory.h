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
/// GlobalDeviceMemoryBase is similar to a pair consisting of a void* pointer
/// and a byte count to tell how much memory is pointed to by that void*.
///
/// GlobalDeviceMemory<T> is a subclass of GlobalDeviceMemoryBase which keeps
/// track of the type of element to be stored in the device memory. It is
/// similar to a pair of a T* pointer and an element count to tell how many
/// elements of type T fit in the memory pointed to by that T*.
///
/// SharedDeviceMemoryBase is just the size in bytes of a shared memory buffer.
///
/// SharedDeviceMemory<T> is a subclass of SharedDeviceMemoryBase which knows
/// how many elements of type T it can hold.
///
/// These classes are useful for keeping track of which memory space a buffer
/// lives in, and the typed subclasses are useful for type-checking.
///
/// The typed subclass will be used by user code, and the untyped base classes
/// will be used for type-unsafe operations inside of StreamExecutor.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_DEVICEMEMORY_H
#define STREAMEXECUTOR_DEVICEMEMORY_H

#include <cassert>
#include <cstddef>

namespace streamexecutor {

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
///
/// The only fully platform-generic operations on handles are using them to
/// create new GlobalDeviceMemoryBase objects, and comparing them to each other
/// for equality.
class GlobalDeviceMemoryBase {
public:
  /// Creates a GlobalDeviceMemoryBase from an optional handle and an optional
  /// byte count.
  explicit GlobalDeviceMemoryBase(const void *Handle = nullptr,
                                  size_t ByteCount = 0)
      : Handle(Handle), ByteCount(ByteCount) {}

  /// Copyable like a pointer.
  GlobalDeviceMemoryBase(const GlobalDeviceMemoryBase &) = default;

  /// Copy-assignable like a pointer.
  GlobalDeviceMemoryBase &operator=(const GlobalDeviceMemoryBase &) = default;

  /// Returns the size, in bytes, for the backing memory.
  size_t getByteCount() const { return ByteCount; }

  /// Gets the internal handle.
  ///
  /// Warning: note that the pointer returned is not necessarily directly to
  /// device virtual address space, but is platform-dependent.
  const void *getHandle() const { return Handle; }

private:
  const void *Handle; // Platform-dependent value representing allocated memory.
  size_t ByteCount;   // Size in bytes of this allocation.
};

template <typename ElemT> class GlobalDeviceMemory;

/// Reference to a slice of device memory.
///
/// Contains a base memory handle, an element count offset into that base
/// memory, and an element count for the size of the slice.
template <typename ElemT> class GlobalDeviceMemorySlice {
public:
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
  GlobalDeviceMemory<ElemT> getBaseMemory() const { return BaseMemory; }

  /// Gets the offset of this slice from the base memory.
  ///
  /// The offset is measured in elements, not bytes.
  size_t getElementOffset() const { return ElementOffset; }

  /// Gets the number of elements in this slice.
  size_t getElementCount() const { return ElementCount; }

  /// Creates a slice of the memory with the first DropCount elements removed.
  GlobalDeviceMemorySlice<ElemT> drop_front(size_t DropCount) const {
    assert(DropCount <= ElementCount &&
           "dropping more than the size of a slice");
    return GlobalDeviceMemorySlice<ElemT>(BaseMemory, ElementOffset + DropCount,
                                          ElementCount - DropCount);
  }

  /// Creates a slice of the memory with the last DropCount elements removed.
  GlobalDeviceMemorySlice<ElemT> drop_back(size_t DropCount) const {
    assert(DropCount <= ElementCount &&
           "dropping more than the size of a slice");
    return GlobalDeviceMemorySlice<ElemT>(BaseMemory, ElementOffset,
                                          ElementCount - DropCount);
  }

  /// Creates a slice of the memory that chops off the first DropCount elements
  /// and keeps the next TakeCount elements.
  GlobalDeviceMemorySlice<ElemT> slice(size_t DropCount,
                                       size_t TakeCount) const {
    assert(DropCount + TakeCount <= ElementCount &&
           "sub-slice operation overruns slice bounds");
    return GlobalDeviceMemorySlice<ElemT>(BaseMemory, ElementOffset + DropCount,
                                          TakeCount);
  }

private:
  GlobalDeviceMemory<ElemT> BaseMemory;
  size_t ElementOffset;
  size_t ElementCount;
};

/// Typed wrapper around the "void *"-like GlobalDeviceMemoryBase class.
///
/// For example, GlobalDeviceMemory<int> is a simple wrapper around
/// GlobalDeviceMemoryBase that represents a buffer of integers stored in global
/// device memory.
template <typename ElemT>
class GlobalDeviceMemory : public GlobalDeviceMemoryBase {
public:
  /// Creates a typed area of GlobalDeviceMemory with a given opaque handle and
  /// the given element count.
  static GlobalDeviceMemory<ElemT> makeFromElementCount(const void *Handle,
                                                        size_t ElementCount) {
    return GlobalDeviceMemory<ElemT>(Handle, ElementCount);
  }

  /// Creates a typed device memory region from an untyped device memory region.
  ///
  /// This effectively amounts to a cast from a void* to an ElemT*, but it also
  /// manages the difference in the size measurements when
  /// GlobalDeviceMemoryBase is measured in bytes and GlobalDeviceMemory is
  /// measured in elements.
  explicit GlobalDeviceMemory(const GlobalDeviceMemoryBase &Other)
      : GlobalDeviceMemoryBase(Other.getHandle(), Other.getByteCount()) {}

  /// Copyable like a pointer.
  GlobalDeviceMemory(const GlobalDeviceMemory &) = default;

  /// Copy-assignable like a pointer.
  GlobalDeviceMemory &operator=(const GlobalDeviceMemory &) = default;

  /// Returns the number of elements of type ElemT that constitute this
  /// allocation.
  size_t getElementCount() const { return getByteCount() / sizeof(ElemT); }

  /// Converts this memory object into a slice.
  GlobalDeviceMemorySlice<ElemT> asSlice() {
    return GlobalDeviceMemorySlice<ElemT>(*this);
  }

private:
  /// Constructs a GlobalDeviceMemory instance from an opaque handle and an
  /// element count.
  ///
  /// This constructor is not public because there is a potential for confusion
  /// between the size of the buffer in bytes and the size of the buffer in
  /// elements.
  ///
  /// The static method makeFromElementCount is provided for users of this class
  /// because its name makes the meaning of the size parameter clear.
  GlobalDeviceMemory(const void *Handle, size_t ElementCount)
      : GlobalDeviceMemoryBase(Handle, ElementCount * sizeof(ElemT)) {}
};

/// A class to represent the size of a dynamic shared memory buffer on a device.
///
/// This class maintains no information about the types to be stored in the
/// buffer. For the typed version of this class see SharedDeviceMemory<ElemT>.
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
class SharedDeviceMemoryBase {
public:
  /// Creates an untyped shared memory array from a byte count.
  SharedDeviceMemoryBase(size_t ByteCount) : ByteCount(ByteCount) {}

  /// Copyable because it is just an array size.
  SharedDeviceMemoryBase(const SharedDeviceMemoryBase &) = default;

  /// Copy-assignable because it is just an array size.
  SharedDeviceMemoryBase &operator=(const SharedDeviceMemoryBase &) = default;

  /// Gets the byte count.
  size_t getByteCount() const { return ByteCount; }

private:
  size_t ByteCount;
};

/// Typed wrapper around the untyped SharedDeviceMemoryBase class.
///
/// For example, SharedDeviceMemory<int> is a wrapper around
/// SharedDeviceMemoryBase that represents a buffer of integers stored in shared
/// device memory.
template <typename ElemT>
class SharedDeviceMemory : public SharedDeviceMemoryBase {
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

  /// Returns the number of elements of type ElemT that can fit this memory
  /// buffer.
  size_t getElementCount() const { return getByteCount() / sizeof(ElemT); }

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
      : SharedDeviceMemoryBase(ElementCount * sizeof(ElemT)) {}
};

} // namespace streamexecutor

#endif // STREAMEXECUTOR_DEVICEMEMORY_H
