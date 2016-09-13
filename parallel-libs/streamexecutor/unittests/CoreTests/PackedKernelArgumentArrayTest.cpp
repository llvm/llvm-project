//===-- PackedKernelArgumentArrayTest.cpp - tests for kernel arg packing --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for kernel argument packing.
///
//===----------------------------------------------------------------------===//

#include "streamexecutor/Device.h"
#include "streamexecutor/DeviceMemory.h"
#include "streamexecutor/PackedKernelArgumentArray.h"
#include "streamexecutor/PlatformDevice.h"
#include "streamexecutor/unittests/CoreTests/SimpleHostPlatformDevice.h"

#include "llvm/ADT/Twine.h"

#include "gtest/gtest.h"

namespace {

namespace se = ::streamexecutor;

using Type = se::KernelArgumentType;

// Test fixture class for testing argument packing.
//
// Basically defines a bunch of types to be packed so they don't have to be
// defined separately in each test.
class DeviceMemoryPackingTest : public ::testing::Test {
public:
  DeviceMemoryPackingTest()
      : Device(&PDevice), Value(42), Handle(&Value), ByteCount(15),
        ElementCount(5),
        TypedGlobal(getOrDie(Device.allocateDeviceMemory<int>(ElementCount))),
        TypedShared(
            se::SharedDeviceMemory<int>::makeFromElementCount(ElementCount)) {}

  se::test::SimpleHostPlatformDevice PDevice;
  se::Device Device;
  int Value;
  void *Handle;
  size_t ByteCount;
  size_t ElementCount;
  se::GlobalDeviceMemory<int> TypedGlobal;
  se::SharedDeviceMemory<int> TypedShared;
};

// Utility method to check the expected address, size, and type for a packed
// argument at the given index of a PackedKernelArgumentArray.
template <typename... ParameterTs>
static void
ExpectEqual(const void *ExpectedAddress, size_t ExpectedSize, Type ExpectedType,
            const se::PackedKernelArgumentArray<ParameterTs...> &Observed,
            size_t Index) {
  SCOPED_TRACE(("Index = " + llvm::Twine(Index)).str());
  EXPECT_EQ(ExpectedAddress, Observed.getAddress(Index));
  EXPECT_EQ(ExpectedAddress, Observed.getAddresses()[Index]);
  EXPECT_EQ(ExpectedSize, Observed.getSize(Index));
  EXPECT_EQ(ExpectedSize, Observed.getSizes()[Index]);
  EXPECT_EQ(ExpectedType, Observed.getType(Index));
  EXPECT_EQ(ExpectedType, Observed.getTypes()[Index]);
}

TEST_F(DeviceMemoryPackingTest, SingleValue) {
  auto Array = se::make_kernel_argument_pack(Value);
  ExpectEqual(&Value, sizeof(Value), Type::VALUE, Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(0u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, SingleTypedGlobal) {
  auto Array = se::make_kernel_argument_pack(TypedGlobal);
  ExpectEqual(TypedGlobal.getHandle(), sizeof(void *),
              Type::GLOBAL_DEVICE_MEMORY, Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(0u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, SingleTypedGlobalPointer) {
  auto Array = se::make_kernel_argument_pack(&TypedGlobal);
  ExpectEqual(TypedGlobal.getHandle(), sizeof(void *),
              Type::GLOBAL_DEVICE_MEMORY, Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(0u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, SingleConstTypedGlobalPointer) {
  const se::GlobalDeviceMemory<int> *ArgumentPointer = &TypedGlobal;
  auto Array = se::make_kernel_argument_pack(ArgumentPointer);
  ExpectEqual(TypedGlobal.getHandle(), sizeof(void *),
              Type::GLOBAL_DEVICE_MEMORY, Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(0u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, SingleTypedShared) {
  auto Array = se::make_kernel_argument_pack(TypedShared);
  ExpectEqual(nullptr, TypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(1u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, SingleTypedSharedPointer) {
  auto Array = se::make_kernel_argument_pack(&TypedShared);
  ExpectEqual(nullptr, TypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(1u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, SingleConstTypedSharedPointer) {
  const se::SharedDeviceMemory<int> *ArgumentPointer = &TypedShared;
  auto Array = se::make_kernel_argument_pack(ArgumentPointer);
  ExpectEqual(nullptr, TypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(1u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, PackSeveralArguments) {
  const se::GlobalDeviceMemory<int> *TypedGlobalPointer = &TypedGlobal;
  const se::SharedDeviceMemory<int> *TypedSharedPointer = &TypedShared;
  auto Array = se::make_kernel_argument_pack(Value, TypedGlobal, &TypedGlobal,
                                             TypedGlobalPointer, TypedShared,
                                             &TypedShared, TypedSharedPointer);
  ExpectEqual(&Value, sizeof(Value), Type::VALUE, Array, 0);
  ExpectEqual(TypedGlobal.getHandle(), sizeof(void *),
              Type::GLOBAL_DEVICE_MEMORY, Array, 1);
  ExpectEqual(TypedGlobal.getHandle(), sizeof(void *),
              Type::GLOBAL_DEVICE_MEMORY, Array, 2);
  ExpectEqual(TypedGlobal.getHandle(), sizeof(void *),
              Type::GLOBAL_DEVICE_MEMORY, Array, 3);
  ExpectEqual(nullptr, TypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 4);
  ExpectEqual(nullptr, TypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 5);
  ExpectEqual(nullptr, TypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 6);
  EXPECT_EQ(7u, Array.getArgumentCount());
  EXPECT_EQ(3u, Array.getSharedCount());
}

} // namespace
