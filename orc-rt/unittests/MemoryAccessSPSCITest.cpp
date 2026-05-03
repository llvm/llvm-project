//===- MemoryAccessSPSCITest.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for MemoryAccess's SPS Controller Interface.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/sps-ci/MemoryAccessSPSCI.h"
#include "orc-rt/SPSWrapperFunction.h"
#include "orc-rt/SimpleSymbolTable.h"

#include "DirectCaller.h"
#include "gtest/gtest.h"

using namespace orc_rt;

class MemoryAccessSPSCITest : public ::testing::Test {
protected:
  void SetUp() override { cantFail(sps_ci::addMemoryAccess(CI)); }

  DirectCaller caller(const char *Name) {
    return DirectCaller(nullptr, reinterpret_cast<orc_rt_WrapperFunction>(
                                     const_cast<void *>(CI.at(Name))));
  }

  SimpleSymbolTable CI;
};

TEST_F(MemoryAccessSPSCITest, Registration) {
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_mem_write_uint8s_sps_wrapper"));
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_mem_write_uint16s_sps_wrapper"));
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_mem_write_uint32s_sps_wrapper"));
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_mem_write_uint64s_sps_wrapper"));
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_mem_write_pointers_sps_wrapper"));
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_mem_write_buffers_sps_wrapper"));
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_mem_read_uint8s_sps_wrapper"));
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_mem_read_uint16s_sps_wrapper"));
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_mem_read_uint32s_sps_wrapper"));
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_mem_read_uint64s_sps_wrapper"));
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_mem_read_pointers_sps_wrapper"));
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_mem_read_buffers_sps_wrapper"));
  EXPECT_TRUE(CI.count("orc_rt_sps_ci_mem_read_strings_sps_wrapper"));
}

TEST_F(MemoryAccessSPSCITest, WriteUInt8s) {
  uint8_t X = 0, Y = 0;
  using SPSSig = void(SPSSequence<SPSTuple<SPSExecutorAddr, uint8_t>>);
  std::vector<std::pair<ExecutorAddr, uint8_t>> Writes = {
      {ExecutorAddr::fromPtr(&X), 42}, {ExecutorAddr::fromPtr(&Y), 255}};
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_sps_ci_mem_write_uint8s_sps_wrapper"),
      [](Error Err) { cantFail(std::move(Err)); }, std::move(Writes));
  EXPECT_EQ(X, 42U);
  EXPECT_EQ(Y, 255U);
}

TEST_F(MemoryAccessSPSCITest, WriteUInt16s) {
  uint16_t X = 0, Y = 0;
  using SPSSig = void(SPSSequence<SPSTuple<SPSExecutorAddr, uint16_t>>);
  std::vector<std::pair<ExecutorAddr, uint16_t>> Writes = {
      {ExecutorAddr::fromPtr(&X), 1000}, {ExecutorAddr::fromPtr(&Y), 65535}};
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_sps_ci_mem_write_uint16s_sps_wrapper"),
      [](Error Err) { cantFail(std::move(Err)); }, std::move(Writes));
  EXPECT_EQ(X, 1000U);
  EXPECT_EQ(Y, 65535U);
}

TEST_F(MemoryAccessSPSCITest, WriteUInt32s) {
  uint32_t X = 0, Y = 0;
  using SPSSig = void(SPSSequence<SPSTuple<SPSExecutorAddr, uint32_t>>);
  std::vector<std::pair<ExecutorAddr, uint32_t>> Writes = {
      {ExecutorAddr::fromPtr(&X), 100000},
      {ExecutorAddr::fromPtr(&Y), 0xdeadbeef}};
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_sps_ci_mem_write_uint32s_sps_wrapper"),
      [](Error Err) { cantFail(std::move(Err)); }, std::move(Writes));
  EXPECT_EQ(X, 100000U);
  EXPECT_EQ(Y, 0xdeadbeefU);
}

TEST_F(MemoryAccessSPSCITest, WriteUInt64s) {
  uint64_t X = 0, Y = 0;
  using SPSSig = void(SPSSequence<SPSTuple<SPSExecutorAddr, uint64_t>>);
  std::vector<std::pair<ExecutorAddr, uint64_t>> Writes = {
      {ExecutorAddr::fromPtr(&X), 0x0102030405060708ULL},
      {ExecutorAddr::fromPtr(&Y), 0xdeadbeefcafef00dULL}};
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_sps_ci_mem_write_uint64s_sps_wrapper"),
      [](Error Err) { cantFail(std::move(Err)); }, std::move(Writes));
  EXPECT_EQ(X, 0x0102030405060708ULL);
  EXPECT_EQ(Y, 0xdeadbeefcafef00dULL);
}

TEST_F(MemoryAccessSPSCITest, WritePointers) {
  void *X = nullptr, *Y = nullptr;
  int A = 1, B = 2;
  using SPSSig = void(SPSSequence<SPSTuple<SPSExecutorAddr, SPSExecutorAddr>>);
  std::vector<std::pair<ExecutorAddr, ExecutorAddr>> Writes = {
      {ExecutorAddr::fromPtr(&X), ExecutorAddr::fromPtr(&A)},
      {ExecutorAddr::fromPtr(&Y), ExecutorAddr::fromPtr(&B)}};
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_sps_ci_mem_write_pointers_sps_wrapper"),
      [](Error Err) { cantFail(std::move(Err)); }, std::move(Writes));
  EXPECT_EQ(X, static_cast<void *>(&A));
  EXPECT_EQ(Y, static_cast<void *>(&B));
}

TEST_F(MemoryAccessSPSCITest, WriteBuffers) {
  char Buf[8] = {};
  char Content[] = "hello";
  using SPSSig =
      void(SPSSequence<SPSTuple<SPSExecutorAddr, SPSSequence<char>>>);
  std::vector<std::pair<ExecutorAddr, span<char>>> Writes = {
      {ExecutorAddr::fromPtr(Buf), span<char>(Content, sizeof(Content))}};
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_sps_ci_mem_write_buffers_sps_wrapper"),
      [](Error Err) { cantFail(std::move(Err)); }, std::move(Writes));
  EXPECT_EQ(Buf[0], 'h');
  EXPECT_EQ(Buf[1], 'e');
  EXPECT_EQ(Buf[2], 'l');
  EXPECT_EQ(Buf[3], 'l');
  EXPECT_EQ(Buf[4], 'o');
  EXPECT_EQ(Buf[5], '\0');
}

TEST_F(MemoryAccessSPSCITest, ReadUInt8s) {
  uint8_t X = 42, Y = 255;
  using SPSSig = SPSSequence<uint8_t>(SPSSequence<SPSExecutorAddr>);
  std::vector<ExecutorAddr> Addrs = {ExecutorAddr::fromPtr(&X),
                                     ExecutorAddr::fromPtr(&Y)};
  std::vector<uint8_t> Result;
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_sps_ci_mem_read_uint8s_sps_wrapper"),
      [&](Expected<std::vector<uint8_t>> R) {
        Result = cantFail(std::move(R));
      },
      std::move(Addrs));
  ASSERT_EQ(Result.size(), 2U);
  EXPECT_EQ(Result[0], 42U);
  EXPECT_EQ(Result[1], 255U);
}

TEST_F(MemoryAccessSPSCITest, ReadUInt16s) {
  uint16_t X = 1000, Y = 65535;
  using SPSSig = SPSSequence<uint16_t>(SPSSequence<SPSExecutorAddr>);
  std::vector<ExecutorAddr> Addrs = {ExecutorAddr::fromPtr(&X),
                                     ExecutorAddr::fromPtr(&Y)};
  std::vector<uint16_t> Result;
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_sps_ci_mem_read_uint16s_sps_wrapper"),
      [&](Expected<std::vector<uint16_t>> R) {
        Result = cantFail(std::move(R));
      },
      std::move(Addrs));
  ASSERT_EQ(Result.size(), 2U);
  EXPECT_EQ(Result[0], 1000U);
  EXPECT_EQ(Result[1], 65535U);
}

TEST_F(MemoryAccessSPSCITest, ReadUInt32s) {
  uint32_t X = 100000, Y = 0xdeadbeef;
  using SPSSig = SPSSequence<uint32_t>(SPSSequence<SPSExecutorAddr>);
  std::vector<ExecutorAddr> Addrs = {ExecutorAddr::fromPtr(&X),
                                     ExecutorAddr::fromPtr(&Y)};
  std::vector<uint32_t> Result;
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_sps_ci_mem_read_uint32s_sps_wrapper"),
      [&](Expected<std::vector<uint32_t>> R) {
        Result = cantFail(std::move(R));
      },
      std::move(Addrs));
  ASSERT_EQ(Result.size(), 2U);
  EXPECT_EQ(Result[0], 100000U);
  EXPECT_EQ(Result[1], 0xdeadbeefU);
}

TEST_F(MemoryAccessSPSCITest, ReadUInt64s) {
  uint64_t X = 0x0102030405060708ULL, Y = 0xdeadbeefcafebabeULL;
  using SPSSig = SPSSequence<uint64_t>(SPSSequence<SPSExecutorAddr>);
  std::vector<ExecutorAddr> Addrs = {ExecutorAddr::fromPtr(&X),
                                     ExecutorAddr::fromPtr(&Y)};
  std::vector<uint64_t> Result;
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_sps_ci_mem_read_uint64s_sps_wrapper"),
      [&](Expected<std::vector<uint64_t>> R) {
        Result = cantFail(std::move(R));
      },
      std::move(Addrs));
  ASSERT_EQ(Result.size(), 2U);
  EXPECT_EQ(Result[0], 0x0102030405060708ULL);
  EXPECT_EQ(Result[1], 0xdeadbeefcafebabeULL);
}

TEST_F(MemoryAccessSPSCITest, ReadPointers) {
  int A = 1, B = 2;
  void *X = &A, *Y = &B;
  using SPSSig = SPSSequence<SPSExecutorAddr>(SPSSequence<SPSExecutorAddr>);
  std::vector<ExecutorAddr> Addrs = {ExecutorAddr::fromPtr(&X),
                                     ExecutorAddr::fromPtr(&Y)};
  std::vector<void *> Result;
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_sps_ci_mem_read_pointers_sps_wrapper"),
      [&](Expected<std::vector<void *>> R) { Result = cantFail(std::move(R)); },
      std::move(Addrs));
  ASSERT_EQ(Result.size(), 2U);
  EXPECT_EQ(Result[0], static_cast<void *>(&A));
  EXPECT_EQ(Result[1], static_cast<void *>(&B));
}

TEST_F(MemoryAccessSPSCITest, ReadBuffers) {
  const char Src[] = "hello world";
  using SPSSig = SPSSequence<SPSSequence<char>>(
      SPSSequence<SPSTuple<SPSExecutorAddr, uint64_t>>);
  std::vector<std::pair<ExecutorAddr, uint64_t>> Reads = {
      {ExecutorAddr::fromPtr(Src), 5},      // "hello"
      {ExecutorAddr::fromPtr(Src + 6), 5}}; // "world"
  std::vector<std::vector<char>> Result;
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_sps_ci_mem_read_buffers_sps_wrapper"),
      [&](Expected<std::vector<std::vector<char>>> R) {
        Result = cantFail(std::move(R));
      },
      std::move(Reads));
  ASSERT_EQ(Result.size(), 2U);
  EXPECT_EQ(Result[0], (std::vector<char>{'h', 'e', 'l', 'l', 'o'}));
  EXPECT_EQ(Result[1], (std::vector<char>{'w', 'o', 'r', 'l', 'd'}));
}

TEST_F(MemoryAccessSPSCITest, ReadStrings) {
  const char *Str1 = "hello";
  const char *Str2 = "world";
  using SPSSig = SPSSequence<SPSString>(SPSSequence<SPSExecutorAddr>);
  std::vector<ExecutorAddr> Addrs = {ExecutorAddr::fromPtr(Str1),
                                     ExecutorAddr::fromPtr(Str2)};
  std::vector<std::string> Result;
  SPSWrapperFunction<SPSSig>::call(
      caller("orc_rt_sps_ci_mem_read_strings_sps_wrapper"),
      [&](Expected<std::vector<std::string>> R) {
        Result = cantFail(std::move(R));
      },
      std::move(Addrs));
  ASSERT_EQ(Result.size(), 2U);
  EXPECT_EQ(Result[0], "hello");
  EXPECT_EQ(Result[1], "world");
}
