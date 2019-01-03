//===-- NativeProcessProtocolTest.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/common/NativeProcessProtocol.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"

using namespace lldb_private;
using namespace lldb;
using namespace testing;

namespace {
class MockDelegate : public NativeProcessProtocol::NativeDelegate {
public:
  MOCK_METHOD1(InitializeDelegate, void(NativeProcessProtocol *Process));
  MOCK_METHOD2(ProcessStateChanged,
               void(NativeProcessProtocol *Process, StateType State));
  MOCK_METHOD1(DidExec, void(NativeProcessProtocol *Process));
};

// NB: This class doesn't use the override keyword to avoid
// -Winconsistent-missing-override warnings from the compiler. The
// inconsistency comes from the overriding definitions in the MOCK_*** macros.
class MockProcess : public NativeProcessProtocol {
public:
  MockProcess(NativeDelegate &Delegate, const ArchSpec &Arch,
              lldb::pid_t Pid = 1)
      : NativeProcessProtocol(Pid, -1, Delegate), Arch(Arch) {}

  MOCK_METHOD1(Resume, Status(const ResumeActionList &ResumeActions));
  MOCK_METHOD0(Halt, Status());
  MOCK_METHOD0(Detach, Status());
  MOCK_METHOD1(Signal, Status(int Signo));
  MOCK_METHOD0(Kill, Status());
  MOCK_METHOD3(AllocateMemory,
               Status(size_t Size, uint32_t Permissions, addr_t &Addr));
  MOCK_METHOD1(DeallocateMemory, Status(addr_t Addr));
  MOCK_METHOD0(GetSharedLibraryInfoAddress, addr_t());
  MOCK_METHOD0(UpdateThreads, size_t());
  MOCK_CONST_METHOD0(GetAuxvData,
                     llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>());
  MOCK_METHOD2(GetLoadedModuleFileSpec,
               Status(const char *ModulePath, FileSpec &Spec));
  MOCK_METHOD2(GetFileLoadAddress,
               Status(const llvm::StringRef &FileName, addr_t &Addr));

  const ArchSpec &GetArchitecture() const /*override*/ { return Arch; }
  Status SetBreakpoint(lldb::addr_t Addr, uint32_t Size,
                       bool Hardware) /*override*/ {
    if (Hardware)
      return SetHardwareBreakpoint(Addr, Size);
    else
      return SetSoftwareBreakpoint(Addr, Size);
  }

  // Redirect base class Read/Write Memory methods to functions whose signatures
  // are more mock-friendly.
  Status ReadMemory(addr_t Addr, void *Buf, size_t Size,
                    size_t &BytesRead) /*override*/;
  Status WriteMemory(addr_t Addr, const void *Buf, size_t Size,
                     size_t &BytesWritten) /*override*/;

  MOCK_METHOD2(ReadMemory,
               llvm::Expected<std::vector<uint8_t>>(addr_t Addr, size_t Size));
  MOCK_METHOD2(WriteMemory,
               llvm::Expected<size_t>(addr_t Addr,
                                      llvm::ArrayRef<uint8_t> Data));

  using NativeProcessProtocol::GetSoftwareBreakpointTrapOpcode;
  llvm::Expected<std::vector<uint8_t>> ReadMemoryWithoutTrap(addr_t Addr,
                                                             size_t Size);

private:
  ArchSpec Arch;
};

class FakeMemory {
public:
  FakeMemory(llvm::ArrayRef<uint8_t> Data) : Data(Data) {}
  llvm::Expected<std::vector<uint8_t>> Read(addr_t Addr, size_t Size);
  llvm::Expected<size_t> Write(addr_t Addr, llvm::ArrayRef<uint8_t> Chunk);

private:
  std::vector<uint8_t> Data;
};
} // namespace

Status MockProcess::ReadMemory(addr_t Addr, void *Buf, size_t Size,
                               size_t &BytesRead) {
  auto ExpectedMemory = ReadMemory(Addr, Size);
  if (!ExpectedMemory) {
    BytesRead = 0;
    return Status(ExpectedMemory.takeError());
  }
  BytesRead = ExpectedMemory->size();
  assert(BytesRead <= Size);
  std::memcpy(Buf, ExpectedMemory->data(), BytesRead);
  return Status();
}

Status MockProcess::WriteMemory(addr_t Addr, const void *Buf, size_t Size,
                                size_t &BytesWritten) {
  auto ExpectedBytes = WriteMemory(
      Addr, llvm::makeArrayRef(static_cast<const uint8_t *>(Buf), Size));
  if (!ExpectedBytes) {
    BytesWritten = 0;
    return Status(ExpectedBytes.takeError());
  }
  BytesWritten = *ExpectedBytes;
  return Status();
}

llvm::Expected<std::vector<uint8_t>>
MockProcess::ReadMemoryWithoutTrap(addr_t Addr, size_t Size) {
  std::vector<uint8_t> Data(Size, 0);
  size_t BytesRead;
  Status ST = NativeProcessProtocol::ReadMemoryWithoutTrap(
      Addr, Data.data(), Data.size(), BytesRead);
  if (ST.Fail())
    return ST.ToError();
  Data.resize(BytesRead);
  return std::move(Data);
}

llvm::Expected<std::vector<uint8_t>> FakeMemory::Read(addr_t Addr,
                                                      size_t Size) {
  if (Addr >= Data.size())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Address out of range.");
  Size = std::min(Size, Data.size() - (size_t)Addr);
  auto Begin = std::next(Data.begin(), Addr);
  return std::vector<uint8_t>(Begin, std::next(Begin, Size));
}

llvm::Expected<size_t> FakeMemory::Write(addr_t Addr,
                                         llvm::ArrayRef<uint8_t> Chunk) {
  if (Addr >= Data.size())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Address out of range.");
  size_t Size = std::min(Chunk.size(), Data.size() - (size_t)Addr);
  std::copy_n(Chunk.begin(), Size, &Data[Addr]);
  return Size;
}

TEST(NativeProcessProtocolTest, SetBreakpoint) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcess Process(DummyDelegate, ArchSpec("x86_64-pc-linux"));
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
  MockProcess Process(DummyDelegate, ArchSpec("x86_64-pc-linux"));
  EXPECT_CALL(Process, ReadMemory(0x47, 1))
      .WillOnce(Return(ByMove(
          llvm::createStringError(llvm::inconvertibleErrorCode(), "Foo"))));
  EXPECT_THAT_ERROR(Process.SetBreakpoint(0x47, 0, false).ToError(),
                    llvm::Failed());
}

TEST(NativeProcessProtocolTest, SetBreakpointFailWrite) {
  NiceMock<MockDelegate> DummyDelegate;
  MockProcess Process(DummyDelegate, ArchSpec("x86_64-pc-linux"));
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
  MockProcess Process(DummyDelegate, ArchSpec("x86_64-pc-linux"));
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
  MockProcess Process(DummyDelegate, ArchSpec("aarch64-pc-linux"));
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
