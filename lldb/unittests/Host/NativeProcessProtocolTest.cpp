//===-- NativeProcessProtocolTest.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport/Host/NativeProcessTestUtils.h"

#include "lldb/Host/common/NativeProcessProtocol.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"

using namespace lldb_private;
using namespace lldb;
using namespace testing;

TEST(NativeProcessProtocolTest, SetBreakpoint) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcess<NativeProcessProtocol> Process(DummyDelegate,
                                             ArchSpec("x86_64-pc-linux"));
  auto Trap = cantFail(Process.GetSoftwareBreakpointTrapOpcode(1));
  InSequence S;
  EXPECT_CALL(Process, ReadMemory(0x47, 1))
      .WillOnce(Return(ByMove(std::vector<uint8_t>{0xbb})));
  EXPECT_CALL(Process, WriteMemory(0x47, Trap)).WillOnce(Return(ByMove(1)));
  EXPECT_CALL(Process, ReadMemory(0x47, 1)).WillOnce(Return(ByMove(Trap)));
  EXPECT_THAT_ERROR(Process.SetBreakpoint(0x47, 0, false).ToError(),
                    llvm::Succeeded());
}

TEST(NativeProcessProtocolTest, SetBreakpointFailRead) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcess<NativeProcessProtocol> Process(DummyDelegate,
                                             ArchSpec("x86_64-pc-linux"));
  EXPECT_CALL(Process, ReadMemory(0x47, 1))
      .WillOnce(Return(ByMove(
          llvm::createStringError(llvm::inconvertibleErrorCode(), "Foo"))));
  EXPECT_THAT_ERROR(Process.SetBreakpoint(0x47, 0, false).ToError(),
                    llvm::Failed());
}

TEST(NativeProcessProtocolTest, SetBreakpointFailWrite) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcess<NativeProcessProtocol> Process(DummyDelegate,
                                             ArchSpec("x86_64-pc-linux"));
  auto Trap = cantFail(Process.GetSoftwareBreakpointTrapOpcode(1));
  InSequence S;
  EXPECT_CALL(Process, ReadMemory(0x47, 1))
      .WillOnce(Return(ByMove(std::vector<uint8_t>{0xbb})));
  EXPECT_CALL(Process, WriteMemory(0x47, Trap))
      .WillOnce(Return(ByMove(
          llvm::createStringError(llvm::inconvertibleErrorCode(), "Foo"))));
  EXPECT_THAT_ERROR(Process.SetBreakpoint(0x47, 0, false).ToError(),
                    llvm::Failed());
}

TEST(NativeProcessProtocolTest, SetBreakpointFailVerify) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcess<NativeProcessProtocol> Process(DummyDelegate,
                                             ArchSpec("x86_64-pc-linux"));
  auto Trap = cantFail(Process.GetSoftwareBreakpointTrapOpcode(1));
  InSequence S;
  EXPECT_CALL(Process, ReadMemory(0x47, 1))
      .WillOnce(Return(ByMove(std::vector<uint8_t>{0xbb})));
  EXPECT_CALL(Process, WriteMemory(0x47, Trap)).WillOnce(Return(ByMove(1)));
  EXPECT_CALL(Process, ReadMemory(0x47, 1))
      .WillOnce(Return(ByMove(
          llvm::createStringError(llvm::inconvertibleErrorCode(), "Foo"))));
  EXPECT_THAT_ERROR(Process.SetBreakpoint(0x47, 0, false).ToError(),
                    llvm::Failed());
}

TEST(NativeProcessProtocolTest, ReadMemoryWithoutTrap) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcess<NativeProcessProtocol> Process(DummyDelegate,
                                             ArchSpec("aarch64-pc-linux"));
  FakeMemory M{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}};
  EXPECT_CALL(Process, ReadMemory(_, _))
      .WillRepeatedly(Invoke(&M, &FakeMemory::Read));
  EXPECT_CALL(Process, WriteMemory(_, _))
      .WillRepeatedly(Invoke(&M, &FakeMemory::Write));

  EXPECT_THAT_ERROR(Process.SetBreakpoint(0x4, 0, false).ToError(),
                    llvm::Succeeded());
  EXPECT_THAT_EXPECTED(
      Process.ReadMemoryWithoutTrap(0, 10),
      llvm::HasValue(std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));
  EXPECT_THAT_EXPECTED(Process.ReadMemoryWithoutTrap(0, 6),
                       llvm::HasValue(std::vector<uint8_t>{0, 1, 2, 3, 4, 5}));
  EXPECT_THAT_EXPECTED(Process.ReadMemoryWithoutTrap(6, 4),
                       llvm::HasValue(std::vector<uint8_t>{6, 7, 8, 9}));
  EXPECT_THAT_EXPECTED(Process.ReadMemoryWithoutTrap(6, 2),
                       llvm::HasValue(std::vector<uint8_t>{6, 7}));
  EXPECT_THAT_EXPECTED(Process.ReadMemoryWithoutTrap(4, 2),
                       llvm::HasValue(std::vector<uint8_t>{4, 5}));
}
