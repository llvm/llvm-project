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
/// The PackedKernelArgumentArrayBase class has no template parameters, so it
/// does not benefit from compile-time type checking. However, since it has no
/// template parameters, it can be passed as an argument to virtual functions,
/// and this allows it to be passed to functions that use virtual function
/// overloading to handle platform-specific kernel launching.
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

/// An array of packed kernel arguments without compile-time type information.
///
/// This un-templated base class is useful because packed kernel arguments must
/// at some point be passed to a virtual function that performs
/// platform-specific kernel launches. Such a virtual function cannot be
/// templated to handle all specializations of the
/// PackedKernelArgumentArray<...> class template, so, instead, references to
/// PackedKernelArgumentArray<...> are passed as references to this base class.
class PackedKernelArgumentArrayBase {
public:
  virtual ~PackedKernelArgumentArrayBase();

  /// Gets the number of packed arguments.
  size_t getArgumentCount() const { return ArgumentCount; }

  /// Gets the address of the argument at the given index.
  const void *getAddress(size_t Index) const { return AddressesData[Index]; }

  /// Gets the size of the argument at the given index.
  size_t getSize(size_t Index) const { return SizesData[Index]; }

  /// Gets the type of the argument at the given index.
  KernelArgumentType getType(size_t Index) const { return TypesData[Index]; }

  /// Gets a pointer to the address array.
  const void *const *getAddresses() const { return AddressesData; }

  /// Gets a pointer to the sizes array.
  const size_t *getSizes() const { return SizesData; }

  /// Gets a pointer to the types array.
  const KernelArgumentType *getTypes() const { return TypesData; }

  /// Gets the number of shared device memory arguments.
  size_t getSharedCount() const { return SharedCount; }

protected:
  PackedKernelArgumentArrayBase(size_t ArgumentCount)
      : ArgumentCount(ArgumentCount), SharedCount(0u) {}

  size_t ArgumentCount;
  size_t SharedCount;
  const void *const *AddressesData;
  size_t *SizesData;
  KernelArgumentType *TypesData;
};

/// An array of packed kernel arguments with compile-time type information.
///
/// This is used by the platform-independent StreamExecutor code to pack
/// arguments in a compile-time type-safe way. In order to actually launch a
/// kernel on a specific platform, however, a reference to this class will have
/// to be passed to a virtual, platform-specific kernel launch function. Such a
/// reference will be passed as a reference to the base class rather than a
/// reference to this subclass itself because a virtual function cannot be
/// templated in such a way to maintain the template parameter types of the
/// subclass.
template <typename... ParameterTs>
class PackedKernelArgumentArray : public PackedKernelArgumentArrayBase {
public:
  /// Constructs an instance by packing the specified arguments.
  ///
  /// Rather than using this constructor directly, consider using the
  /// make_kernel_argument_pack function instead, to get the compiler to infer
  /// the parameter types for you.
  PackedKernelArgumentArray(const ParameterTs &... Arguments)
      : PackedKernelArgumentArrayBase(sizeof...(ParameterTs)) {
    AddressesData = Addresses.data();
    SizesData = Sizes.data();
    TypesData = Types.data();
    PackArguments(0, Arguments...);
  }

  ~PackedKernelArgumentArray() override = default;

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
