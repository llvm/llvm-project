//===--- TargetProcessControl.h - Target process control APIs ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for interacting with target processes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_TARGETPROCESSCONTROL_H
#define LLVM_EXECUTIONENGINE_ORC_TARGETPROCESSCONTROL_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <future>
#include <vector>

namespace llvm {
namespace orc {

/// TargetProcessControl supports interaction with a JIT target process.
class TargetProcessControl {
public:
  /// APIs for manipulating memory in the target process.
  class MemoryAccess {
  public:
    template <typename T> struct UIntWrite {
      UIntWrite() = default;
      UIntWrite(JITTargetAddress Address, T Value)
          : Address(Address), Value(Value) {}

      JITTargetAddress Address = 0;
      T Value = 0;
    };

    using UInt8Write = UIntWrite<uint8_t>;
    using UInt16Write = UIntWrite<uint16_t>;
    using UInt32Write = UIntWrite<uint32_t>;
    using UInt64Write = UIntWrite<uint64_t>;

    struct BufferWrite {
      BufferWrite(JITTargetAddress Address, StringRef Buffer)
          : Address(Address), Buffer(Buffer) {}

      JITTargetAddress Address = 0;
      StringRef Buffer;
    };

    using WriteResultFn = unique_function<void(Error)>;

    virtual ~MemoryAccess();

    virtual void writeUInt8s(ArrayRef<UInt8Write> Ws,
                             WriteResultFn OnWriteComplete) = 0;

    virtual void writeUInt16s(ArrayRef<UInt16Write> Ws,
                              WriteResultFn OnWriteComplete) = 0;

    virtual void writeUInt32s(ArrayRef<UInt32Write> Ws,
                              WriteResultFn OnWriteComplete) = 0;

    virtual void writeUInt64s(ArrayRef<UInt64Write> Ws,
                              WriteResultFn OnWriteComplete) = 0;

    virtual void writeBuffers(ArrayRef<BufferWrite> Ws,
                              WriteResultFn OnWriteComplete) = 0;

    Error writeUInt8s(ArrayRef<UInt8Write> Ws) {
      std::promise<MSVCPError> ResultP;
      auto ResultF = ResultP.get_future();
      writeUInt8s(Ws, [&](Error Err) { ResultP.set_value(std::move(Err)); });
      return ResultF.get();
    }

    Error writeUInt16s(ArrayRef<UInt16Write> Ws) {
      std::promise<MSVCPError> ResultP;
      auto ResultF = ResultP.get_future();
      writeUInt16s(Ws, [&](Error Err) { ResultP.set_value(std::move(Err)); });
      return ResultF.get();
    }

    Error writeUInt32s(ArrayRef<UInt32Write> Ws) {
      std::promise<MSVCPError> ResultP;
      auto ResultF = ResultP.get_future();
      writeUInt32s(Ws, [&](Error Err) { ResultP.set_value(std::move(Err)); });
      return ResultF.get();
    }

    Error writeUInt64s(ArrayRef<UInt64Write> Ws) {
      std::promise<MSVCPError> ResultP;
      auto ResultF = ResultP.get_future();
      writeUInt64s(Ws, [&](Error Err) { ResultP.set_value(std::move(Err)); });
      return ResultF.get();
    }

    Error writeBuffers(ArrayRef<BufferWrite> Ws) {
      std::promise<MSVCPError> ResultP;
      auto ResultF = ResultP.get_future();
      writeBuffers(Ws, [&](Error Err) { ResultP.set_value(std::move(Err)); });
      return ResultF.get();
    }
  };

  using DynamicLibraryHandle = JITTargetAddress;

  /// Request lookup within a single library.
  /// If Library is None then the whole target process should be searched.
  struct LookupRequestElement {
    LookupRequestElement(DynamicLibraryHandle Handle,
                         const SymbolLookupSet &Symbols)
        : Handle(Handle), Symbols(Symbols) {}
    DynamicLibraryHandle Handle;
    const SymbolLookupSet &Symbols;
  };

  using LookupRequest = ArrayRef<LookupRequestElement>;

  using LookupResult = std::vector<std::vector<JITTargetAddress>>;

  virtual ~TargetProcessControl();

  /// Return the Triple for the target process.
  const Triple &getTargetTriple() const { return TT; }

  /// Get the page size for the target process.
  unsigned getPageSize() const { return PageSize; }

  /// Return a JITLinkMemoryManager for the target process.
  jitlink::JITLinkMemoryManager &getMemMgr() const { return *MemMgr; }

  /// Return a MemoryAccess object for the target process.
  MemoryAccess &getMemoryAccess() const { return *MemAccess; }

  /// Load the library at the given path. Returns a handle to the loaded
  /// library. If LibraryPath is null this function will return the global
  /// handle for the target process.
  virtual Expected<DynamicLibraryHandle>
  loadLibrary(const char *LibraryPath) = 0;

  /// Search for symbols in the target process.
  /// The result of the lookup is a 2-dimentional array of target addresses
  /// that correspond to the lookup order.
  virtual Expected<LookupResult> lookupSymbols(LookupRequest Request) = 0;

protected:
  TargetProcessControl(Triple TT, unsigned PageSize);

  Triple TT;
  unsigned PageSize = 0;
  jitlink::JITLinkMemoryManager *MemMgr = nullptr;
  MemoryAccess *MemAccess = nullptr;
};

/// A TargetProcessControl
class SelfTargetProcessControl : public TargetProcessControl,
                                 private TargetProcessControl::MemoryAccess {
public:
  SelfTargetProcessControl(Triple TT, unsigned PageSize);

  static Expected<std::unique_ptr<SelfTargetProcessControl>> Create();

  Expected<DynamicLibraryHandle> loadLibrary(const char *LibraryPath) override;

  Expected<LookupResult> lookupSymbols(LookupRequest Request) override;

private:
  void writeUInt8s(ArrayRef<UInt8Write> Ws,
                   WriteResultFn OnWriteComplete) override;

  void writeUInt16s(ArrayRef<UInt16Write> Ws,
                    WriteResultFn OnWriteComplete) override;

  void writeUInt32s(ArrayRef<UInt32Write> Ws,
                    WriteResultFn OnWriteComplete) override;

  void writeUInt64s(ArrayRef<UInt64Write> Ws,
                    WriteResultFn OnWriteComplete) override;

  void writeBuffers(ArrayRef<BufferWrite> Ws,
                    WriteResultFn OnWriteComplete) override;

  std::unique_ptr<jitlink::InProcessMemoryManager> IPMM =
      std::make_unique<jitlink::InProcessMemoryManager>();

  char GlobalManglingPrefix = 0;
  std::vector<std::unique_ptr<sys::DynamicLibrary>> DynamicLibraries;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_TARGETPROCESSCONTROL_H
