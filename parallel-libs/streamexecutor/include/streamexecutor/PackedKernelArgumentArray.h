//===-- PackedKernelArgumentArray.h - Packed kernel arg types ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// The types in this file are designed to deal with the fact that device memory
/// kernel arguments are treated differently from other arguments during kernel
/// argument packing.
///
/// GlobalDeviceMemory<T> arguments are passed to a kernel by passing their
/// opaque handle. SharedDeviceMemory<T> arguments have no associated address,
/// only a size, so the size is the only information that gets passed to the
/// kernel launch.
///
/// The KernelArgumentType enum is used to keep track of the type of each
/// argument.
///
/// The PackedKernelArgumentArray class uses template metaprogramming to convert
/// each argument to a PackedKernelArgument with minimal runtime overhead.
///
/// The design of the PackedKernelArgumentArray class has a few idiosyncrasies
/// due to the fact that parameter packing has been identified as
/// performance-critical in some applications. The packed argument data is
/// stored as a struct of arrays rather than an array of structs because CUDA
/// kernel launches in the CUDA driver API take an array of argument addresses.
/// Having created the array of argument addresses here, no further work will
/// need to be done in the CUDA driver layer to unpack and repack the addresses.
///
/// The shared memory argument count is maintained separately because in the
/// common case where it is zero, the CUDA layer doesn't have to loop through
/// the argument array and sum up all the shared memory sizes. This is another
/// performance optimization that shows up as a quirk in this class interface.
///
/// The platform-interface kernel launch function will take the following
/// arguments, which are provided by this interface:
///   * argument count,
///   * array of argument address,
///   * array of argument sizes,
///   * array of argument types, and
///   * shared pointer count.
/// This information should be enough to allow any platform to launch the kernel
/// efficiently, although it is probably more information than is needed for any
/// specific platform.
///
//===----------------------------------------------------------------------===//

#ifndef STREAMEXECUTOR_PACKEDKERNELARGUMENTARRAY_H
#define STREAMEXECUTOR_PACKEDKERNELARGUMENTARRAY_H

#include <array>

#include "streamexecutor/DeviceMemory.h"

namespace streamexecutor {

enum class KernelArgumentType {
  VALUE,                /// Non-device-memory argument.
  GLOBAL_DEVICE_MEMORY, /// Non-shared device memory argument.
  SHARED_DEVICE_MEMORY  /// Shared device memory argument.
};

/// An array of packed kernel arguments.
template <typename... ParameterTs> class PackedKernelArgumentArray {
public:
  /// Constructs an instance by packing the specified arguments.
  PackedKernelArgumentArray(const ParameterTs &... Arguments)
      : SharedCount(0u) {
    PackArguments(0, Arguments...);
  }

  /// Gets the number of packed arguments.
  size_t getArgumentCount() const { return sizeof...(ParameterTs); }

  /// Gets the address of the argument at the given index.
  const void *getAddress(size_t Index) const { return Addresses[Index]; }

  /// Gets the size of the argument at the given index.
  size_t getSize(size_t Index) const { return Sizes[Index]; }

  /// Gets the type of the argument at the given index.
  KernelArgumentType getType(size_t Index) const { return Types[Index]; }

  /// Gets a pointer to the address array.
  const void *const *getAddresses() const { return Addresses.data(); }

  /// Gets a pointer to the sizes array.
  const size_t *getSizes() const { return Sizes.data(); }

  /// Gets a pointer to the types array.
  const KernelArgumentType *getTypes() const { return Types.data(); }

  /// Gets the number of shared device memory arguments.
  size_t getSharedCount() const { return SharedCount; }

private:
  // Base case for PackArguments when there are no arguments to pack.
  void PackArguments(size_t) {}

  // Induction step for PackArguments.
  template <typename T, typename... RemainingParameterTs>
  void PackArguments(size_t Index, const T &Argument,
                     const RemainingParameterTs &... RemainingArguments) {
    PackOneArgument(Index, Argument);
    PackArguments(Index + 1, RemainingArguments...);
  }

  // Pack a normal, non-device-memory argument.
  template <typename T> void PackOneArgument(size_t Index, const T &Argument) {
    Addresses[Index] = &Argument;
    Sizes[Index] = sizeof(T);
    Types[Index] = KernelArgumentType::VALUE;
  }

  // Pack a GlobalDeviceMemoryBase argument.
  void PackOneArgument(size_t Index, const GlobalDeviceMemoryBase &Argument) {
    Addresses[Index] = Argument.getHandle();
    Sizes[Index] = sizeof(void *);
    Types[Index] = KernelArgumentType::GLOBAL_DEVICE_MEMORY;
  }

  // Pack a GlobalDeviceMemoryBase pointer argument.
  void PackOneArgument(size_t Index, GlobalDeviceMemoryBase *Argument) {
    Addresses[Index] = Argument->getHandle();
    Sizes[Index] = sizeof(void *);
    Types[Index] = KernelArgumentType::GLOBAL_DEVICE_MEMORY;
  }

  // Pack a const GlobalDeviceMemoryBase pointer argument.
  void PackOneArgument(size_t Index, const GlobalDeviceMemoryBase *Argument) {
    Addresses[Index] = Argument->getHandle();
    Sizes[Index] = sizeof(void *);
    Types[Index] = KernelArgumentType::GLOBAL_DEVICE_MEMORY;
  }

  // Pack a GlobalDeviceMemory<T> argument.
  template <typename T>
  void PackOneArgument(size_t Index, const GlobalDeviceMemory<T> &Argument) {
    Addresses[Index] = Argument.getHandle();
    Sizes[Index] = sizeof(void *);
    Types[Index] = KernelArgumentType::GLOBAL_DEVICE_MEMORY;
  }

  // Pack a GlobalDeviceMemory<T> pointer argument.
  template <typename T>
  void PackOneArgument(size_t Index, GlobalDeviceMemory<T> *Argument) {
    Addresses[Index] = Argument->getHandle();
    Sizes[Index] = sizeof(void *);
    Types[Index] = KernelArgumentType::GLOBAL_DEVICE_MEMORY;
  }

  // Pack a const GlobalDeviceMemory<T> pointer argument.
  template <typename T>
  void PackOneArgument(size_t Index, const GlobalDeviceMemory<T> *Argument) {
    Addresses[Index] = Argument->getHandle();
    Sizes[Index] = sizeof(void *);
    Types[Index] = KernelArgumentType::GLOBAL_DEVICE_MEMORY;
  }

  // Pack a SharedDeviceMemoryBase argument.
  void PackOneArgument(size_t Index, const SharedDeviceMemoryBase &Argument) {
    ++SharedCount;
    Addresses[Index] = nullptr;
    Sizes[Index] = Argument.getByteCount();
    Types[Index] = KernelArgumentType::SHARED_DEVICE_MEMORY;
  }

  // Pack a SharedDeviceMemoryBase pointer argument.
  void PackOneArgument(size_t Index, SharedDeviceMemoryBase *Argument) {
    ++SharedCount;
    Addresses[Index] = nullptr;
    Sizes[Index] = Argument->getByteCount();
    Types[Index] = KernelArgumentType::SHARED_DEVICE_MEMORY;
  }

  // Pack a const SharedDeviceMemoryBase pointer argument.
  void PackOneArgument(size_t Index, const SharedDeviceMemoryBase *Argument) {
    ++SharedCount;
    Addresses[Index] = nullptr;
    Sizes[Index] = Argument->getByteCount();
    Types[Index] = KernelArgumentType::SHARED_DEVICE_MEMORY;
  }

  // Pack a SharedDeviceMemory argument.
  template <typename T>
  void PackOneArgument(size_t Index, const SharedDeviceMemory<T> &Argument) {
    ++SharedCount;
    Addresses[Index] = nullptr;
    Sizes[Index] = Argument.getByteCount();
    Types[Index] = KernelArgumentType::SHARED_DEVICE_MEMORY;
  }

  // Pack a SharedDeviceMemory pointer argument.
  template <typename T>
  void PackOneArgument(size_t Index, SharedDeviceMemory<T> *Argument) {
    ++SharedCount;
    Addresses[Index] = nullptr;
    Sizes[Index] = Argument->getByteCount();
    Types[Index] = KernelArgumentType::SHARED_DEVICE_MEMORY;
  }

  // Pack a const SharedDeviceMemory pointer argument.
  template <typename T>
  void PackOneArgument(size_t Index, const SharedDeviceMemory<T> *Argument) {
    ++SharedCount;
    Addresses[Index] = nullptr;
    Sizes[Index] = Argument->getByteCount();
    Types[Index] = KernelArgumentType::SHARED_DEVICE_MEMORY;
  }

  std::array<const void *, sizeof...(ParameterTs)> Addresses;
  std::array<size_t, sizeof...(ParameterTs)> Sizes;
  std::array<KernelArgumentType, sizeof...(ParameterTs)> Types;
  size_t SharedCount;
};

// Utility template function to call the PackedKernelArgumentArray constructor
// with the template arguments matching the types of the arguments passed to
// this function.
template <typename... ParameterTs>
PackedKernelArgumentArray<ParameterTs...>
make_kernel_argument_pack(const ParameterTs &... Arguments) {
  return PackedKernelArgumentArray<ParameterTs...>(Arguments...);
}

} // namespace streamexecutor

#endif // STREAMEXECUTOR_PACKEDKERNELARGUMENTARRAY_H
