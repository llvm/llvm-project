//===- EPCGenericMemoryAccessTest.cpp -- Tests for EPCGenericMemoryAccess -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"

#include "llvm/ExecutionEngine/Orc/EPCGenericMemoryAccess.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/Testing/Support/Error.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::shared;

namespace {

template <typename WriteT, typename SPSWriteT>
CWrapperFunctionResult testWriteUInts(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<void(SPSSequence<SPSWriteT>)>::handle(
             ArgData, ArgSize,
             [](std::vector<WriteT> Ws) {
               for (auto &W : Ws)
                 *W.Addr.template toPtr<decltype(W.Value) *>() = W.Value;
             })
      .release();
}

CWrapperFunctionResult testWritePointers(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<void(SPSSequence<SPSMemoryAccessPointerWrite>)>::
      handle(ArgData, ArgSize,
             [](std::vector<tpctypes::PointerWrite> Ws) {
               for (auto &W : Ws)
                 *W.Addr.template toPtr<void **>() = W.Value.toPtr<void *>();
             })
          .release();
}

CWrapperFunctionResult testWriteBuffers(const char *ArgData, size_t ArgSize) {
  return WrapperFunction<void(SPSSequence<SPSMemoryAccessBufferWrite>)>::handle(
             ArgData, ArgSize,
             [](std::vector<tpctypes::BufferWrite> Ws) {
               for (auto &W : Ws)
                 memcpy(W.Addr.template toPtr<char *>(), W.Buffer.data(),
                        W.Buffer.size());
             })
      .release();
}

template <typename ReadT>
CWrapperFunctionResult testReadUInts(const char *ArgData, size_t ArgSize) {
  using SPSSig = SPSSequence<ReadT>(SPSSequence<SPSExecutorAddr>);
  return WrapperFunction<SPSSig>::handle(ArgData, ArgSize,
                                         [](std::vector<ExecutorAddr> Rs) {
                                           std::vector<ReadT> Result;
                                           Result.reserve(Rs.size());
                                           for (auto &R : Rs)
                                             Result.push_back(
                                                 *R.template toPtr<ReadT *>());
                                           return Result;
                                         })
      .release();
}

CWrapperFunctionResult testReadPointers(const char *ArgData, size_t ArgSize) {
  using SPSSig = SPSSequence<SPSExecutorAddr>(SPSSequence<SPSExecutorAddr>);
  return WrapperFunction<SPSSig>::handle(
             ArgData, ArgSize,
             [](std::vector<ExecutorAddr> Rs) {
               std::vector<ExecutorAddr> Result;
               Result.reserve(Rs.size());
               for (auto &R : Rs)
                 Result.push_back(
                     ExecutorAddr::fromPtr(*R.template toPtr<void **>()));
               return Result;
             })
      .release();
}

CWrapperFunctionResult testReadBuffers(const char *ArgData, size_t ArgSize) {
  using SPSSig =
      SPSSequence<SPSSequence<uint8_t>>(SPSSequence<SPSExecutorAddrRange>);
  return WrapperFunction<SPSSig>::handle(
             ArgData, ArgSize,
             [](std::vector<ExecutorAddrRange> Rs) {
               std::vector<std::vector<uint8_t>> Result;
               Result.reserve(Rs.size());
               for (auto &R : Rs) {
                 Result.push_back({});
                 Result.back().resize(R.size());
                 memcpy(reinterpret_cast<char *>(Result.back().data()),
                        R.Start.toPtr<char *>(), R.size());
               }
               return Result;
             })
      .release();
}

CWrapperFunctionResult testReadStrings(const char *ArgData, size_t ArgSize) {
  using SPSSig = SPSSequence<SPSString>(SPSSequence<SPSExecutorAddr>);
  return WrapperFunction<SPSSig>::handle(
             ArgData, ArgSize,
             [](std::vector<ExecutorAddr> Rs) {
               std::vector<std::string> Result;
               Result.reserve(Rs.size());
               for (auto &R : Rs)
                 Result.push_back(std::string(R.toPtr<char *>()));
               return Result;
             })
      .release();
}

class EPCGenericMemoryAccessTest : public testing::Test {
public:
  EPCGenericMemoryAccessTest() {
    EPC = cantFail(SelfExecutorProcessControl::Create());

    EPCGenericMemoryAccess::FuncAddrs FAs;
    FAs.WriteUInt8s = ExecutorAddr::fromPtr(
        &testWriteUInts<tpctypes::UInt8Write, SPSMemoryAccessUInt8Write>);
    FAs.WriteUInt16s = ExecutorAddr::fromPtr(
        &testWriteUInts<tpctypes::UInt16Write, SPSMemoryAccessUInt16Write>);
    FAs.WriteUInt32s = ExecutorAddr::fromPtr(
        &testWriteUInts<tpctypes::UInt32Write, SPSMemoryAccessUInt32Write>);
    FAs.WriteUInt64s = ExecutorAddr::fromPtr(
        &testWriteUInts<tpctypes::UInt64Write, SPSMemoryAccessUInt64Write>);
    FAs.WritePointers = ExecutorAddr::fromPtr(&testWritePointers);
    FAs.WriteBuffers = ExecutorAddr::fromPtr(&testWriteBuffers);
    FAs.ReadUInt8s = ExecutorAddr::fromPtr(&testReadUInts<uint8_t>);
    FAs.ReadUInt16s = ExecutorAddr::fromPtr(&testReadUInts<uint16_t>);
    FAs.ReadUInt32s = ExecutorAddr::fromPtr(&testReadUInts<uint32_t>);
    FAs.ReadUInt64s = ExecutorAddr::fromPtr(&testReadUInts<uint64_t>);
    FAs.ReadPointers = ExecutorAddr::fromPtr(&testReadPointers);
    FAs.ReadBuffers = ExecutorAddr::fromPtr(&testReadBuffers);
    FAs.ReadStrings = ExecutorAddr::fromPtr(&testReadStrings);

    MemAccess = std::make_unique<EPCGenericMemoryAccess>(*EPC, FAs);
  }

  ~EPCGenericMemoryAccessTest() { cantFail(EPC->disconnect()); }

protected:
  std::shared_ptr<SelfExecutorProcessControl> EPC;
  std::unique_ptr<MemoryAccess> MemAccess;

  static const uint8_t UInt8_1_TestValue;
  static const uint8_t UInt8_2_TestValue;
  static const uint16_t UInt16_TestValue;
  static const uint32_t UInt32_TestValue;
  static const uint64_t UInt64_TestValue;
  static const void *Pointer_TestValue;

  static const char Buffer_TestValue[21];
  static const char *String_TestValue;
};

const uint8_t EPCGenericMemoryAccessTest::UInt8_1_TestValue = 1;
const uint8_t EPCGenericMemoryAccessTest::UInt8_2_TestValue = 2;
const uint16_t EPCGenericMemoryAccessTest::UInt16_TestValue = 3;
const uint32_t EPCGenericMemoryAccessTest::UInt32_TestValue = 4;
const uint64_t EPCGenericMemoryAccessTest::UInt64_TestValue = 5;

const void *EPCGenericMemoryAccessTest::Pointer_TestValue =
    reinterpret_cast<void *>(uintptr_t(0x6));

const char EPCGenericMemoryAccessTest::Buffer_TestValue[21] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A,
    0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14};
const char *EPCGenericMemoryAccessTest::String_TestValue = "hello, world!";

TEST_F(EPCGenericMemoryAccessTest, WriteUInt8s) {
  uint8_t UInt8_1_Storage = 0;
  uint8_t UInt8_2_Storage = 0;

  auto Err = MemAccess->writeUInt8s(
      {{ExecutorAddr::fromPtr(&UInt8_1_Storage), UInt8_1_TestValue},
       {ExecutorAddr::fromPtr(&UInt8_2_Storage), UInt8_2_TestValue}});
  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_EQ(UInt8_1_Storage, UInt8_1_TestValue);
  EXPECT_EQ(UInt8_2_Storage, UInt8_2_TestValue);
}

TEST_F(EPCGenericMemoryAccessTest, ReadUInt8s) {
  uint8_t UInt8_1_Storage = UInt8_1_TestValue;
  uint8_t UInt8_2_Storage = UInt8_2_TestValue;

  auto Vals =
      MemAccess->readUInt8s({{ExecutorAddr::fromPtr(&UInt8_1_Storage),
                              ExecutorAddr::fromPtr(&UInt8_2_Storage)}});
  static_assert(
      std::is_same_v<decltype(Vals)::value_type::value_type, uint8_t>);
  if (!Vals)
    return ADD_FAILURE() << toString(Vals.takeError());

  EXPECT_EQ(Vals->size(), 2U);
  if (Vals->size() >= 1)
    EXPECT_EQ((*Vals)[0], UInt8_1_TestValue);
  if (Vals->size() >= 2)
    EXPECT_EQ((*Vals)[1], UInt8_2_TestValue);
}

TEST_F(EPCGenericMemoryAccessTest, WriteUInt16s) {
  uint16_t UInt16_Storage = 0;

  auto Err = MemAccess->writeUInt16s(
      {{ExecutorAddr::fromPtr(&UInt16_Storage), UInt16_TestValue}});
  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_EQ(UInt16_Storage, UInt16_TestValue);
}

TEST_F(EPCGenericMemoryAccessTest, ReadUInt16s) {
  uint16_t UInt16_Storage = UInt16_TestValue;

  auto Vals =
      MemAccess->readUInt16s({{ExecutorAddr::fromPtr(&UInt16_Storage)}});
  static_assert(
      std::is_same_v<decltype(Vals)::value_type::value_type, uint16_t>);
  if (Vals) {
    EXPECT_EQ(Vals->size(), 1U);
    if (Vals->size() == 1)
      EXPECT_EQ((*Vals)[0], UInt16_TestValue);
  } else
    EXPECT_THAT_ERROR(Vals.takeError(), Succeeded());
}

TEST_F(EPCGenericMemoryAccessTest, WriteUInt32s) {
  uint32_t UInt32_Storage = 0;

  auto Err = MemAccess->writeUInt32s(
      {{ExecutorAddr::fromPtr(&UInt32_Storage), UInt32_TestValue}});
  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_EQ(UInt32_Storage, UInt32_TestValue);
}

TEST_F(EPCGenericMemoryAccessTest, ReadUInt32s) {
  uint32_t UInt32_Storage = UInt32_TestValue;

  auto Vals =
      MemAccess->readUInt32s({{ExecutorAddr::fromPtr(&UInt32_Storage)}});
  static_assert(
      std::is_same_v<decltype(Vals)::value_type::value_type, uint32_t>);
  if (Vals) {
    EXPECT_EQ(Vals->size(), 1U);
    if (Vals->size() == 1)
      EXPECT_EQ((*Vals)[0], UInt32_TestValue);
  } else
    EXPECT_THAT_ERROR(Vals.takeError(), Succeeded());
}

TEST_F(EPCGenericMemoryAccessTest, WriteUInt64s) {
  uint64_t UInt64_Storage = 0;

  auto Err = MemAccess->writeUInt64s(
      {{ExecutorAddr::fromPtr(&UInt64_Storage), UInt64_TestValue}});
  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_EQ(UInt64_Storage, UInt64_TestValue);
}

TEST_F(EPCGenericMemoryAccessTest, ReadUInt64s) {
  uint64_t UInt64_Storage = UInt64_TestValue;

  auto Vals =
      MemAccess->readUInt64s({{ExecutorAddr::fromPtr(&UInt64_Storage)}});
  static_assert(
      std::is_same_v<decltype(Vals)::value_type::value_type, uint64_t>);
  if (Vals) {
    EXPECT_EQ(Vals->size(), 1U);
    if (Vals->size() == 1)
      EXPECT_EQ((*Vals)[0], UInt64_TestValue);
  } else
    EXPECT_THAT_ERROR(Vals.takeError(), Succeeded());
}

TEST_F(EPCGenericMemoryAccessTest, WritePointers) {
  void *Pointer_Storage = nullptr;

  auto Err =
      MemAccess->writePointers({{ExecutorAddr::fromPtr(&Pointer_Storage),
                                 ExecutorAddr::fromPtr(Pointer_TestValue)}});
  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_EQ(Pointer_Storage, Pointer_TestValue);
}

TEST_F(EPCGenericMemoryAccessTest, ReadPointers) {
  auto Vals =
      MemAccess->readPointers({{ExecutorAddr::fromPtr(&Pointer_TestValue)}});
  static_assert(
      std::is_same_v<decltype(Vals)::value_type::value_type, ExecutorAddr>);
  if (Vals) {
    EXPECT_EQ(Vals->size(), 1U);
    if (Vals->size() == 1)
      EXPECT_EQ((*Vals)[0], ExecutorAddr::fromPtr(Pointer_TestValue));
  } else
    EXPECT_THAT_ERROR(Vals.takeError(), Succeeded());
}

TEST_F(EPCGenericMemoryAccessTest, WriteBuffers) {
  char Buffer_Storage[sizeof(Buffer_TestValue)];
  memset(Buffer_Storage, 0, sizeof(Buffer_TestValue));

  auto Err = MemAccess->writeBuffers(
      {{ExecutorAddr::fromPtr(&Buffer_Storage),
        ArrayRef(Buffer_TestValue, sizeof(Buffer_TestValue))}});
  EXPECT_THAT_ERROR(std::move(Err), Succeeded());
  EXPECT_EQ(ArrayRef(Buffer_Storage, sizeof(Buffer_TestValue)),
            ArrayRef(Buffer_TestValue, sizeof(Buffer_TestValue)));
}

TEST_F(EPCGenericMemoryAccessTest, ReadBuffers) {
  char Buffer_Storage[sizeof(Buffer_TestValue)];
  memcpy(Buffer_Storage, Buffer_TestValue, sizeof(Buffer_TestValue));

  auto Vals = MemAccess->readBuffers({{ExecutorAddrRange(
      ExecutorAddr::fromPtr(&Buffer_Storage), sizeof(Buffer_Storage))}});
  static_assert(std::is_same_v<decltype(Vals)::value_type::value_type,
                               std::vector<uint8_t>>);
  if (Vals) {
    EXPECT_EQ(Vals->size(), 1U);
    if (Vals->size() == 1) {
      EXPECT_EQ((*Vals)[0].size(), sizeof(Buffer_Storage));
      EXPECT_EQ(
          memcmp((*Vals)[0].data(), Buffer_TestValue, sizeof(Buffer_Storage)),
          0);
    }
  } else
    EXPECT_THAT_ERROR(Vals.takeError(), Succeeded());
}

TEST_F(EPCGenericMemoryAccessTest, ReadStrings) {
  auto Vals =
      MemAccess->readStrings({{ExecutorAddr::fromPtr(String_TestValue)}});
  static_assert(
      std::is_same_v<decltype(Vals)::value_type, std::vector<std::string>>);
  if (Vals) {
    EXPECT_EQ(Vals->size(), 1U);
    if (Vals->size() == 1)
      EXPECT_EQ((*Vals)[0], std::string(String_TestValue));
  } else
    EXPECT_THAT_ERROR(Vals.takeError(), Succeeded());
}

} // namespace
