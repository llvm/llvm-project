//===-- NativeProcessTestUtils.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UNITTESTS_TESTINGSUPPORT_HOST_NATIVEPROCESSTESTUTILS_H
#define LLDB_UNITTESTS_TESTINGSUPPORT_HOST_NATIVEPROCESSTESTUTILS_H

#include "lldb/Host/common/NativeProcessProtocol.h"
#include "llvm/Testing/Support/Error.h"
#include "gmock/gmock.h"

using namespace lldb_private;
using namespace lldb;
using namespace testing;

namespace lldb_private {

class MockDelegate : public NativeProcessProtocol::NativeDelegate {
public:
  MOCK_METHOD1(InitializeDelegate, void(NativeProcessProtocol *Process));
  MOCK_METHOD2(ProcessStateChanged,
               void(NativeProcessProtocol *Process, StateType State));
  MOCK_METHOD1(DidExec, void(NativeProcessProtocol *Process));
  MOCK_METHOD2(NewSubprocessImpl,
               void(NativeProcessProtocol *parent_process,
                    std::unique_ptr<NativeProcessProtocol> &child_process));
  // This is a hack to avoid MOCK_METHOD2 incompatibility with std::unique_ptr
  // passed as value.
  void NewSubprocess(NativeProcessProtocol *parent_process,
                     std::unique_ptr<NativeProcessProtocol> child_process) {
    NewSubprocessImpl(parent_process, child_process);
  }
};

// NB: This class doesn't use the override keyword to avoid
// -Winconsistent-missing-override warnings from the compiler. The
// inconsistency comes from the overriding definitions in the MOCK_*** macros.
template <typename T> class MockProcess : public T {
public:
  MockProcess(NativeProcessProtocol::NativeDelegate &Delegate,
              const ArchSpec &Arch, lldb::pid_t Pid = 1)
      : T(Pid, -1, Delegate), Arch(Arch) {}

  MOCK_METHOD1(Resume, Status(const ResumeActionList &ResumeActions));
  MOCK_METHOD0(Halt, Status());
  MOCK_METHOD0(Detach, Status());
  MOCK_METHOD1(Signal, Status(int Signo));
  MOCK_METHOD0(Kill, Status());
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
      return this->SetHardwareBreakpoint(Addr, Size);
    else
      return this->SetSoftwareBreakpoint(Addr, Size);
  }

  // Redirect base class Read/Write Memory methods to functions whose signatures
  // are more mock-friendly.
  Status ReadMemory(addr_t Addr, void *Buf, size_t Size,
                    size_t &BytesRead) /*override*/ {
    auto ExpectedMemory = this->ReadMemory(Addr, Size);
    if (!ExpectedMemory) {
      BytesRead = 0;
      return Status(ExpectedMemory.takeError());
    }
    BytesRead = ExpectedMemory->size();
    assert(BytesRead <= Size);
    std::memcpy(Buf, ExpectedMemory->data(), BytesRead);
    return Status();
  }

  Status WriteMemory(addr_t Addr, const void *Buf, size_t Size,
                     size_t &BytesWritten) /*override*/ {
    auto ExpectedBytes = this->WriteMemory(
        Addr, llvm::ArrayRef(static_cast<const uint8_t *>(Buf), Size));
    if (!ExpectedBytes) {
      BytesWritten = 0;
      return Status(ExpectedBytes.takeError());
    }
    BytesWritten = *ExpectedBytes;
    return Status();
  }

  MOCK_METHOD2(ReadMemory,
               llvm::Expected<std::vector<uint8_t>>(addr_t Addr, size_t Size));
  MOCK_METHOD2(WriteMemory,
               llvm::Expected<size_t>(addr_t Addr,
                                      llvm::ArrayRef<uint8_t> Data));

  using T::GetSoftwareBreakpointTrapOpcode;
  llvm::Expected<std::vector<uint8_t>> ReadMemoryWithoutTrap(addr_t Addr,
                                                             size_t Size) {
    std::vector<uint8_t> Data(Size, 0);
    size_t BytesRead;
    Status ST =
        T::ReadMemoryWithoutTrap(Addr, Data.data(), Data.size(), BytesRead);
    if (ST.Fail())
      return ST.ToError();
    Data.resize(BytesRead);
    return std::move(Data);
  }

private:
  ArchSpec Arch;
};

class FakeMemory {
public:
  FakeMemory(llvm::ArrayRef<uint8_t> Data, addr_t start_addr = 0)
      : Data(Data), m_start_addr(start_addr) {}

  FakeMemory(const void *Data, size_t data_size, addr_t start_addr = 0)
      : Data((const uint8_t *)Data, ((const uint8_t *)Data) + data_size),
        m_start_addr(start_addr) {}

  llvm::Expected<std::vector<uint8_t>> Read(addr_t Addr, size_t Size) {
    Addr -= m_start_addr;
    if (Addr >= Data.size())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Address out of range.");
    Size = std::min(Size, Data.size() - (size_t)Addr);
    auto Begin = std::next(Data.begin(), Addr);
    return std::vector<uint8_t>(Begin, std::next(Begin, Size));
  }

  llvm::Expected<size_t> Write(addr_t Addr, llvm::ArrayRef<uint8_t> Chunk) {
    Addr -= m_start_addr;
    if (Addr >= Data.size())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Address out of range.");
    size_t Size = std::min(Chunk.size(), Data.size() - (size_t)Addr);
    std::copy_n(Chunk.begin(), Size, &Data[Addr]);
    return Size;
  }

private:
  std::vector<uint8_t> Data;
  addr_t m_start_addr;
};
} // namespace lldb_private

#endif
