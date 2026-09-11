//===-- NativeProcessProtocolTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestingSupport/Host/NativeProcessTestUtils.h"

#include "lldb/Host/common/NativeProcessProtocol.h"
#include "llvm/Support/Process.h"
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

TEST(NativeProcessProtocolTest, RemoveSoftwareBreakpoint) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcess<NativeProcessProtocol> Process(DummyDelegate,
                                             ArchSpec("x86_64-pc-linux"));
  auto Trap = cantFail(Process.GetSoftwareBreakpointTrapOpcode(1));
  auto Original = std::vector<uint8_t>{0xbb};

  // Set up a breakpoint.
  {
    InSequence S;
    EXPECT_CALL(Process, ReadMemory(0x47, 1))
        .WillOnce(Return(ByMove(Original)));
    EXPECT_CALL(Process, WriteMemory(0x47, Trap)).WillOnce(Return(ByMove(1)));
    EXPECT_CALL(Process, ReadMemory(0x47, 1)).WillOnce(Return(ByMove(Trap)));
    EXPECT_THAT_ERROR(Process.SetBreakpoint(0x47, 0, false).ToError(),
                      llvm::Succeeded());
  }

  // Remove the breakpoint for the first time. This should remove the breakpoint
  // from m_software_breakpoints.
  //
  // Should succeed.
  {
    InSequence S;
    EXPECT_CALL(Process, ReadMemory(0x47, 1)).WillOnce(Return(ByMove(Trap)));
    EXPECT_CALL(Process, WriteMemory(0x47, llvm::ArrayRef(Original)))
        .WillOnce(Return(ByMove(1)));
    EXPECT_CALL(Process, ReadMemory(0x47, 1))
        .WillOnce(Return(ByMove(Original)));
    EXPECT_THAT_ERROR(Process.RemoveBreakpoint(0x47, false).ToError(),
                      llvm::Succeeded());
  }

  // Remove the breakpoint for the second time.
  //
  // Should fail. None of the ReadMemory() or WriteMemory() should be called,
  // because the function should early return when seeing that the breakpoint
  // isn't in m_software_breakpoints.
  {
    EXPECT_CALL(Process, ReadMemory(_, _)).Times(0);
    EXPECT_CALL(Process, WriteMemory(_, _)).Times(0);
    EXPECT_THAT_ERROR(Process.RemoveBreakpoint(0x47, false).ToError(),
                      llvm::Failed());
  }
}

TEST(NativeProcessProtocolTest, RemoveSoftwareBreakpointMemoryError) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcess<NativeProcessProtocol> Process(DummyDelegate,
                                             ArchSpec("x86_64-pc-linux"));
  auto Trap = cantFail(Process.GetSoftwareBreakpointTrapOpcode(1));
  auto Original = std::vector<uint8_t>{0xbb};
  auto SomethingElse = std::vector<uint8_t>{0xaa};

  // Set up a breakpoint.
  {
    InSequence S;
    EXPECT_CALL(Process, ReadMemory(0x47, 1))
        .WillOnce(Return(ByMove(Original)));
    EXPECT_CALL(Process, WriteMemory(0x47, Trap)).WillOnce(Return(ByMove(1)));
    EXPECT_CALL(Process, ReadMemory(0x47, 1)).WillOnce(Return(ByMove(Trap)));
    EXPECT_THAT_ERROR(Process.SetBreakpoint(0x47, 0, false).ToError(),
                      llvm::Succeeded());
  }

  // Remove the breakpoint for the first time, with an unexpected value read by
  // the first ReadMemory(). This should cause an early return, with the
  // breakpoint removed from m_software_breakpoints.
  //
  // Should fail.
  {
    InSequence S;
    EXPECT_CALL(Process, ReadMemory(0x47, 1))
        .WillOnce(Return(ByMove(SomethingElse)));
    EXPECT_THAT_ERROR(Process.RemoveBreakpoint(0x47, false).ToError(),
                      llvm::Failed());
  }

  // Remove the breakpoint for the second time.
  //
  // Should fail. None of the ReadMemory() or WriteMemory() should be called,
  // because the function should early return when seeing that the breakpoint
  // isn't in m_software_breakpoints.
  {
    EXPECT_CALL(Process, ReadMemory(_, _)).Times(0);
    EXPECT_CALL(Process, WriteMemory(_, _)).Times(0);
    EXPECT_THAT_ERROR(Process.RemoveBreakpoint(0x47, false).ToError(),
                      llvm::Failed());
  }
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

TEST(NativeProcessProtocolTest, ReadCStringFromMemory) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcess<NativeProcessProtocol> Process(DummyDelegate,
                                             ArchSpec("aarch64-pc-linux"));
  FakeMemory M({'h', 'e', 'l', 'l', 'o', 0, 'w', 'o'});
  EXPECT_CALL(Process, ReadMemory(_, _))
      .WillRepeatedly(Invoke(&M, &FakeMemory::Read));

  char string[1024];
  size_t bytes_read;
  EXPECT_THAT_EXPECTED(Process.ReadCStringFromMemory(
                           0x0, &string[0], sizeof(string), bytes_read),
                       llvm::HasValue(llvm::StringRef("hello")));
  EXPECT_EQ(bytes_read, 6UL);
}

TEST(NativeProcessProtocolTest, ReadCStringFromMemory_MaxSize) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcess<NativeProcessProtocol> Process(DummyDelegate,
                                             ArchSpec("aarch64-pc-linux"));
  FakeMemory M({'h', 'e', 'l', 'l', 'o', 0, 'w', 'o'});
  EXPECT_CALL(Process, ReadMemory(_, _))
      .WillRepeatedly(Invoke(&M, &FakeMemory::Read));

  char string[4];
  size_t bytes_read;
  EXPECT_THAT_EXPECTED(Process.ReadCStringFromMemory(
                           0x0, &string[0], sizeof(string), bytes_read),
                       llvm::HasValue(llvm::StringRef("hel")));
  EXPECT_EQ(bytes_read, 3UL);
}

TEST(NativeProcessProtocolTest, ReadCStringFromMemory_CrossPageBoundary) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcess<NativeProcessProtocol> Process(DummyDelegate,
                                             ArchSpec("aarch64-pc-linux"));
  unsigned string_start = llvm::sys::Process::getPageSizeEstimate() - 3;
  FakeMemory M({'h', 'e', 'l', 'l', 'o', 0, 'w', 'o'}, string_start);
  EXPECT_CALL(Process, ReadMemory(_, _))
      .WillRepeatedly(Invoke(&M, &FakeMemory::Read));

  char string[1024];
  size_t bytes_read;
  EXPECT_THAT_EXPECTED(Process.ReadCStringFromMemory(string_start, &string[0],
                                                     sizeof(string),
                                                     bytes_read),
                       llvm::HasValue(llvm::StringRef("hello")));
  EXPECT_EQ(bytes_read, 6UL);
}

void DoTestWriteMemoryPreservingTrap(
    const std::vector<lldb::addr_t> &bp_addrs,
    std::optional<lldb::addr_t> write_addr,
    const std::vector<uint8_t> &write_data, uint32_t expected_number_of_writes,
    const std::vector<uint8_t> expected_after_write_read_memory,
    const std::vector<uint8_t> expected_after_write_read_memory_without_trap) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcess<NativeProcessProtocol> Process(DummyDelegate,
                                             ArchSpec("aarch64-pc-linux"));
  const std::vector<uint8_t> fake_memory{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  FakeMemory M{fake_memory};
  ON_CALL(Process, ReadMemory(_, _))
      .WillByDefault(Invoke(&M, &FakeMemory::Read));
  ON_CALL(Process, WriteMemory(_, _))
      .WillByDefault(Invoke(&M, &FakeMemory::Write));

  for (auto bp_addr : bp_addrs)
    EXPECT_THAT_ERROR(
        Process.SetBreakpoint(bp_addr, 0, /*hardware=*/false).ToError(),
        llvm::Succeeded());

  if (expected_number_of_writes)
    EXPECT_CALL(Process, WriteMemory(_, _))
        .Times(expected_number_of_writes)
        .WillRepeatedly(DoDefault());
  else
    EXPECT_CALL(Process, WriteMemory(_, _)).Times(0);

  if (write_addr) {
    size_t bytes_written = 0;
    Status err = Process.WriteMemory(*write_addr, write_data.data(),
                                     write_data.size(), bytes_written);
    EXPECT_THAT_ERROR(err.ToError(), llvm::Succeeded());
    EXPECT_EQ(bytes_written, write_data.size());
  }

  Mock::VerifyAndClearExpectations(&Process);

  // Anything written over a breakpoint should go into the saved bytes instead
  // of into memory.

  // ReadMemory should show that the breakpoint instructions are unchanged.
  auto memory_or_err = Process.ReadMemory(0, fake_memory.size());
  EXPECT_THAT_EXPECTED(memory_or_err, llvm::Succeeded());
  EXPECT_EQ(*memory_or_err, expected_after_write_read_memory);

  // ReadMemoryWithoutTrap should show that writes to the breakpoints have
  // updated the saved data in the breakpoint.
  memory_or_err = Process.ReadMemoryWithoutTrap(0, fake_memory.size());
  EXPECT_THAT_EXPECTED(memory_or_err, llvm::Succeeded());
  EXPECT_EQ(*memory_or_err, expected_after_write_read_memory_without_trap);

  // When the breakpoint is removed, the saved bytes are actually written
  // to memory.
  for (auto bp_addr : bp_addrs)
    EXPECT_THAT_ERROR(Process.RemoveBreakpoint(bp_addr, false).ToError(),
                      llvm::Succeeded());

  // The memory should contain the written data now.
  memory_or_err = Process.ReadMemory(0, fake_memory.size());
  EXPECT_THAT_EXPECTED(memory_or_err, llvm::Succeeded());
  EXPECT_EQ(*memory_or_err, expected_after_write_read_memory_without_trap);

  // As there are no breakpoints, the result of ReadMemoryWithoutTrap should
  // be the same.
  memory_or_err = Process.ReadMemoryWithoutTrap(0, fake_memory.size());
  EXPECT_THAT_EXPECTED(memory_or_err, llvm::Succeeded());
  EXPECT_EQ(*memory_or_err, expected_after_write_read_memory_without_trap);
}

void TestWriteMemoryPreservingTrap(
    const std::vector<lldb::addr_t> &bp_addrs,
    std::optional<lldb::addr_t> write_addr,
    const std::vector<uint8_t> &write_data, uint32_t expected_number_of_writes,
    const std::vector<uint8_t> expected_after_write_read_memory,
    const std::vector<uint8_t> expected_after_write_read_memory_without_trap) {
  auto bp_addrs_in_order = bp_addrs;
  DoTestWriteMemoryPreservingTrap(
      bp_addrs_in_order, write_addr, write_data, expected_number_of_writes,
      expected_after_write_read_memory,
      expected_after_write_read_memory_without_trap);

  // WriteMemoryPreservingTrap should not care in what order the breakpoints
  // were inserted.
  if (bp_addrs_in_order.size()) {
    std::reverse(bp_addrs_in_order.begin(), bp_addrs_in_order.end());
    DoTestWriteMemoryPreservingTrap(
        bp_addrs_in_order, write_addr, write_data, expected_number_of_writes,
        expected_after_write_read_memory,
        expected_after_write_read_memory_without_trap);
  }
}

TEST(NativeProcessProtocolTest, WriteMemoryPreservingTrap) {
// Software breakpoint instruction encoding for AArch64.
#define S__W__B__P 0x0, 0x0, 0x20, 0xd4

  // In these tests, numbers are used for the initial memory contents and
  // letters for the data being written. These variables are used for letters
  // so that the inputs can be vertically aligned.
  uint8_t a = 'a';
  uint8_t b = 'b';
  uint8_t c = 'c';
  uint8_t d = 'd';
  uint8_t e = 'e';
  uint8_t f = 'f';
  uint8_t g = 'g';
  uint8_t h = 'h';
  uint8_t i = 'i';
  uint8_t j = 'j';
  uint8_t k = 'k';

  // Write nothing, set no breakpoints, nothing changes.
  TestWriteMemoryPreservingTrap({}, std::nullopt, {}, 0,
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  // Write nothing, set a breakpoint. Breakpoint encoding should be visible in
  // memory.
  TestWriteMemoryPreservingTrap({0}, std::nullopt, {}, 0,
                                {S__W__B__P, 4, 5, 6, 7, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  // 0 size write with no breakpoints set.
  TestWriteMemoryPreservingTrap({}, 0, {}, 0,
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  // 0 size write not within a breakpoint.
  TestWriteMemoryPreservingTrap({4}, 0, {}, 0,
                                {0, 1, 2, 3, S__W__B__P, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  // 0 size write at start of a breakpoint.
  TestWriteMemoryPreservingTrap({4}, 4, {}, 0,
                                {0, 1, 2, 3, S__W__B__P, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  // 0 size write at end of a breakpoint.
  TestWriteMemoryPreservingTrap({4}, 7, {}, 0,
                                {0, 1, 2, 3, S__W__B__P, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  // 0 size write one beyond the end of a breakpoint.
  TestWriteMemoryPreservingTrap({4}, 8, {}, 0,
                                {0, 1, 2, 3, S__W__B__P, 8, 9, 10},
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  // Write something but set no breakpoints. Data is written directly to memory.
  TestWriteMemoryPreservingTrap({}, 2, {a, b}, 1,
                                {0, 1, a, b, 4, 5, 6, 7, 8, 9, 10},
                                {0, 1, a, b, 4, 5, 6, 7, 8, 9, 10});

  // Write before a breakpoint and do not overlap it.
  TestWriteMemoryPreservingTrap({4}, 1, {a, b, c}, 1,
                                {0, a, b, c, S__W__B__P, 8, 9, 10},
                                {0, a, b, c, 4, 5, 6, 7, 8, 9, 10});

  // Write within a breakpoint, without writing outside of it. Data kept in
  // saved bytes until removal.
  TestWriteMemoryPreservingTrap({4}, 4, {a, b, c, d}, 0,
                                {0, 1, 2, 3, S__W__B__P, 8, 9, 10},
                                {0, 1, 2, 3, a, b, c, d, 8, 9, 10});

  // Write immediately after a breakpoint. Data is written directly to memory.
  TestWriteMemoryPreservingTrap({2}, 6, {a, b, c}, 1,
                                {0, 1, S__W__B__P, a, b, c, 9, 10},
                                {0, 1, 2, 3, 4, 5, a, b, c, 9, 10});

  // Write a range overlapping the beginning of a breakpoint. First part will
  // go direct to memory, the other to the saved bytes.
  TestWriteMemoryPreservingTrap({2}, 0, {a, b, c, d}, 1,
                                {a, b, S__W__B__P, 6, 7, 8, 9, 10},
                                {a, b, c, d, 4, 5, 6, 7, 8, 9, 10});

  // Write overlapping end of breakpoint. First part goes to saved bytes,
  // second part direct to memory.
  TestWriteMemoryPreservingTrap({3}, 5, {a, b, c, d}, 1,
                                {0, 1, 2, S__W__B__P, c, d, 9, 10},
                                {0, 1, 2, 3, 4, a, b, c, d, 9, 10});

  // Write from before to after break. Ends go to memory, middle to saved bytes.
  TestWriteMemoryPreservingTrap({2}, 0, {a, b, c, d, e, f, g, h}, 2,
                                {a, b, S__W__B__P, g, h, 8, 9, 10},
                                {a, b, c, d, e, f, g, h, 8, 9, 10});

  // Overlap a breakpoint at the very start of memory.
  TestWriteMemoryPreservingTrap({0}, 0, {a, b, c, d, e, f}, 1,
                                {S__W__B__P, e, f, 6, 7, 8, 9, 10},
                                {a, b, c, d, e, f, 6, 7, 8, 9, 10});

  // Overlap a breakpoint at the very end of memory.
  TestWriteMemoryPreservingTrap({7}, 5, {a, b, c, d, e, f}, 1,
                                {0, 1, 2, 3, 4, a, b, S__W__B__P},
                                {0, 1, 2, 3, 4, a, b, c, d, e, f});

  // Write up to a breakpoint.
  TestWriteMemoryPreservingTrap({5}, 1, {a, b, c, d}, 1,
                                {0, a, b, c, d, S__W__B__P, 9, 10},
                                {0, a, b, c, d, 5, 6, 7, 8, 9, 10});

  // Write starting immediately after a breakpoint.
  TestWriteMemoryPreservingTrap({2}, 6, {a, b, c, d}, 1,
                                {0, 1, S__W__B__P, a, b, c, d, 10},
                                {0, 1, 2, 3, 4, 5, a, b, c, d, 10});

  // Overlap 2 breakpoints, write extends before and after them.
  TestWriteMemoryPreservingTrap({1, 6}, 0, {a, b, c, d, e, f, g, h, i, j, k}, 3,
                                {a, S__W__B__P, f, S__W__B__P, k},
                                {a, b, c, d, e, f, g, h, i, j, k});

  // Write starts within the first one, and ends after the second one.
  TestWriteMemoryPreservingTrap({1, 6}, 2, {a, b, c, d, e, f, g, h, i}, 2,
                                {0, S__W__B__P, d, S__W__B__P, i},
                                {0, 1, a, b, c, d, e, f, g, h, i});

  // Write range from before first breakpoint to within second breakpoint.
  TestWriteMemoryPreservingTrap({1, 6}, 0, {a, b, c, d, e, f, g, h, i}, 2,
                                {a, S__W__B__P, f, S__W__B__P, 10},
                                {a, b, c, d, e, f, g, h, i, 9, 10});

  // Write range from within first to beyond second.
  TestWriteMemoryPreservingTrap({1, 6}, 2, {a, b, c, d, e, f, g, h, i}, 2,
                                {0, S__W__B__P, d, S__W__B__P, i},
                                {0, 1, a, b, c, d, e, f, g, h, i});

  // Write range from within first to within second.
  TestWriteMemoryPreservingTrap({1, 6}, 2, {a, b, c, d, e, f, g}, 1,
                                {0, S__W__B__P, d, S__W__B__P, 10},
                                {0, 1, a, b, c, d, e, f, g, 9, 10});

  // Write in range between 2 breakpoints.
  TestWriteMemoryPreservingTrap({0, 7}, 4, {a, b, c}, 1,
                                {S__W__B__P, a, b, c, S__W__B__P},
                                {0, 1, 2, 3, a, b, c, 7, 8, 9, 10});

  // 2 breakpoints with no gap between them, write overlaps the first one.
  TestWriteMemoryPreservingTrap({2, 6}, 0, {a, b, c, d}, 1,
                                {a, b, S__W__B__P, S__W__B__P, 10},
                                {a, b, c, d, 4, 5, 6, 7, 8, 9, 10});

  // 2 breakpoints with no gap between them, write is within the first one.
  TestWriteMemoryPreservingTrap({2, 6}, 3, {a, b}, 0,
                                {0, 1, S__W__B__P, S__W__B__P, 10},
                                {0, 1, 2, a, b, 5, 6, 7, 8, 9, 10});

  // 2 breakpoints with no gap between them, write is across both.
  TestWriteMemoryPreservingTrap({2, 6}, 4, {a, b, c, d}, 0,
                                {0, 1, S__W__B__P, S__W__B__P, 10},
                                {0, 1, 2, 3, a, b, c, d, 8, 9, 10});

  // 2 breakpoints with no gap between them, write is within second one.
  TestWriteMemoryPreservingTrap({2, 6}, 7, {a, b}, 0,
                                {0, 1, S__W__B__P, S__W__B__P, 10},
                                {0, 1, 2, 3, 4, 5, 6, a, b, 9, 10});

  // 2 breakpoints with no gap between them, write overlaps second one.
  TestWriteMemoryPreservingTrap({2, 6}, 8, {a, b, c}, 1,
                                {0, 1, S__W__B__P, S__W__B__P, c},
                                {0, 1, 2, 3, 4, 5, 6, 7, a, b, c});

#undef S__W__B__P
}
