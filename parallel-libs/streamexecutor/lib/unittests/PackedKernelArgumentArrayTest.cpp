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

#include "streamexecutor/DeviceMemory.h"
#include "streamexecutor/PackedKernelArgumentArray.h"

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
      : Value(42), Handle(&Value), ByteCount(15), ElementCount(5),
        UntypedGlobal(Handle, ByteCount),
        TypedGlobal(se::GlobalDeviceMemory<int>::makeFromElementCount(
            Handle, ElementCount)),
        UntypedShared(ByteCount),
        TypedShared(
            se::SharedDeviceMemory<int>::makeFromElementCount(ElementCount)) {}

  int Value;
  void *Handle;
  size_t ByteCount;
  size_t ElementCount;
  se::GlobalDeviceMemoryBase UntypedGlobal;
  se::GlobalDeviceMemory<int> TypedGlobal;
  se::SharedDeviceMemoryBase UntypedShared;
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

TEST_F(DeviceMemoryPackingTest, SingleUntypedGlobal) {
  auto Array = se::make_kernel_argument_pack(UntypedGlobal);
  ExpectEqual(Handle, sizeof(void *), Type::GLOBAL_DEVICE_MEMORY, Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(0u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, SingleUntypedGlobalPointer) {
  auto Array = se::make_kernel_argument_pack(&UntypedGlobal);
  ExpectEqual(Handle, sizeof(void *), Type::GLOBAL_DEVICE_MEMORY, Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(0u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, SingleConstUntypedGlobalPointer) {
  const se::GlobalDeviceMemoryBase *ConstPointer = &UntypedGlobal;
  auto Array = se::make_kernel_argument_pack(ConstPointer);
  ExpectEqual(Handle, sizeof(void *), Type::GLOBAL_DEVICE_MEMORY, Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(0u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, SingleTypedGlobal) {
  auto Array = se::make_kernel_argument_pack(TypedGlobal);
  ExpectEqual(Handle, sizeof(void *), Type::GLOBAL_DEVICE_MEMORY, Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(0u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, SingleTypedGlobalPointer) {
  auto Array = se::make_kernel_argument_pack(&TypedGlobal);
  ExpectEqual(Handle, sizeof(void *), Type::GLOBAL_DEVICE_MEMORY, Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(0u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, SingleConstTypedGlobalPointer) {
  const se::GlobalDeviceMemory<int> *ArgumentPointer = &TypedGlobal;
  auto Array = se::make_kernel_argument_pack(ArgumentPointer);
  ExpectEqual(Handle, sizeof(void *), Type::GLOBAL_DEVICE_MEMORY, Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(0u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, SingleUntypedShared) {
  auto Array = se::make_kernel_argument_pack(UntypedShared);
  ExpectEqual(nullptr, UntypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(1u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, SingleUntypedSharedPointer) {
  auto Array = se::make_kernel_argument_pack(&UntypedShared);
  ExpectEqual(nullptr, UntypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(1u, Array.getSharedCount());
}

TEST_F(DeviceMemoryPackingTest, SingleConstUntypedSharedPointer) {
  const se::SharedDeviceMemoryBase *ArgumentPointer = &UntypedShared;
  auto Array = se::make_kernel_argument_pack(ArgumentPointer);
  ExpectEqual(nullptr, UntypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 0);
  EXPECT_EQ(1u, Array.getArgumentCount());
  EXPECT_EQ(1u, Array.getSharedCount());
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
  const se::GlobalDeviceMemoryBase *UntypedGlobalPointer = &UntypedGlobal;
  const se::GlobalDeviceMemory<int> *TypedGlobalPointer = &TypedGlobal;
  const se::SharedDeviceMemoryBase *UntypedSharedPointer = &UntypedShared;
  const se::SharedDeviceMemory<int> *TypedSharedPointer = &TypedShared;
  auto Array = se::make_kernel_argument_pack(
      Value, UntypedGlobal, &UntypedGlobal, UntypedGlobalPointer, TypedGlobal,
      &TypedGlobal, TypedGlobalPointer, UntypedShared, &UntypedShared,
      UntypedSharedPointer, TypedShared, &TypedShared, TypedSharedPointer);
  ExpectEqual(&Value, sizeof(Value), Type::VALUE, Array, 0);
  ExpectEqual(Handle, sizeof(void *), Type::GLOBAL_DEVICE_MEMORY, Array, 1);
  ExpectEqual(Handle, sizeof(void *), Type::GLOBAL_DEVICE_MEMORY, Array, 2);
  ExpectEqual(Handle, sizeof(void *), Type::GLOBAL_DEVICE_MEMORY, Array, 3);
  ExpectEqual(Handle, sizeof(void *), Type::GLOBAL_DEVICE_MEMORY, Array, 4);
  ExpectEqual(Handle, sizeof(void *), Type::GLOBAL_DEVICE_MEMORY, Array, 5);
  ExpectEqual(Handle, sizeof(void *), Type::GLOBAL_DEVICE_MEMORY, Array, 6);
  ExpectEqual(nullptr, UntypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 7);
  ExpectEqual(nullptr, UntypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 8);
  ExpectEqual(nullptr, UntypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 9);
  ExpectEqual(nullptr, TypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 10);
  ExpectEqual(nullptr, TypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 11);
  ExpectEqual(nullptr, TypedShared.getByteCount(), Type::SHARED_DEVICE_MEMORY,
              Array, 12);
  EXPECT_EQ(13u, Array.getArgumentCount());
  EXPECT_EQ(6u, Array.getSharedCount());
}

} // namespace
