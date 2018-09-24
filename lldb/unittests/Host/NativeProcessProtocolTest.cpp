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

  const ArchSpec &GetArchitecture() const override { return Arch; }
  Status SetBreakpoint(lldb::addr_t Addr, uint32_t Size,
                       bool Hardware) override {
    if (Hardware)
      return SetHardwareBreakpoint(Addr, Size);
    else
      return SetSoftwareBreakpoint(Addr, Size);
  }

  // Redirect base class Read/Write Memory methods to functions whose signatures
  // are more mock-friendly.
  Status ReadMemory(addr_t Addr, void *Buf, size_t Size,
                    size_t &BytesRead) override;
  Status WriteMemory(addr_t Addr, const void *Buf, size_t Size,
                     size_t &BytesWritten) override;

  MOCK_METHOD2(ReadMemory,
               llvm::Expected<std::vector<uint8_t>>(addr_t Addr, size_t Size));
  MOCK_METHOD2(WriteMemory,
               llvm::Expected<size_t>(addr_t Addr,
                                      llvm::ArrayRef<uint8_t> Data));

  using NativeProcessProtocol::GetSoftwareBreakpointTrapOpcode;

private:
  ArchSpec Arch;
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
