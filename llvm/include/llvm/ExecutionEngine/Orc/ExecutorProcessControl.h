//===- ExecutorProcessControl.h - Executor process control APIs -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for interacting with the executor processes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EXECUTORPROCESSCONTROL_H
#define LLVM_EXECUTIONENGINE_ORC_EXECUTORPROCESSCONTROL_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <future>
#include <mutex>
#include <vector>

namespace llvm {
namespace orc {

class ExecutionSession;
class SymbolLookupSet;

/// ExecutorProcessControl supports interaction with a JIT target process.
class ExecutorProcessControl {
  friend class ExecutionSession;

public:
  /// Sender to return the result of a WrapperFunction executed in the JIT.
  using SendResultFunction =
      unique_function<void(shared::WrapperFunctionResult)>;

  /// APIs for manipulating memory in the target process.
  class MemoryAccess {
  public:
    /// Callback function for asynchronous writes.
    using WriteResultFn = unique_function<void(Error)>;

    virtual ~MemoryAccess();

    virtual void writeUInt8s(ArrayRef<tpctypes::UInt8Write> Ws,
                             WriteResultFn OnWriteComplete) = 0;

    virtual void writeUInt16s(ArrayRef<tpctypes::UInt16Write> Ws,
                              WriteResultFn OnWriteComplete) = 0;

    virtual void writeUInt32s(ArrayRef<tpctypes::UInt32Write> Ws,
                              WriteResultFn OnWriteComplete) = 0;

    virtual void writeUInt64s(ArrayRef<tpctypes::UInt64Write> Ws,
                              WriteResultFn OnWriteComplete) = 0;

    virtual void writeBuffers(ArrayRef<tpctypes::BufferWrite> Ws,
                              WriteResultFn OnWriteComplete) = 0;

    Error writeUInt8s(ArrayRef<tpctypes::UInt8Write> Ws) {
      std::promise<MSVCPError> ResultP;
      auto ResultF = ResultP.get_future();
      writeUInt8s(Ws, [&](Error Err) { ResultP.set_value(std::move(Err)); });
      return ResultF.get();
    }

    Error writeUInt16s(ArrayRef<tpctypes::UInt16Write> Ws) {
      std::promise<MSVCPError> ResultP;
      auto ResultF = ResultP.get_future();
      writeUInt16s(Ws, [&](Error Err) { ResultP.set_value(std::move(Err)); });
      return ResultF.get();
    }

    Error writeUInt32s(ArrayRef<tpctypes::UInt32Write> Ws) {
      std::promise<MSVCPError> ResultP;
      auto ResultF = ResultP.get_future();
      writeUInt32s(Ws, [&](Error Err) { ResultP.set_value(std::move(Err)); });
      return ResultF.get();
    }

    Error writeUInt64s(ArrayRef<tpctypes::UInt64Write> Ws) {
      std::promise<MSVCPError> ResultP;
      auto ResultF = ResultP.get_future();
      writeUInt64s(Ws, [&](Error Err) { ResultP.set_value(std::move(Err)); });
      return ResultF.get();
    }

    Error writeBuffers(ArrayRef<tpctypes::BufferWrite> Ws) {
      std::promise<MSVCPError> ResultP;
      auto ResultF = ResultP.get_future();
      writeBuffers(Ws, [&](Error Err) { ResultP.set_value(std::move(Err)); });
      return ResultF.get();
    }
  };

  /// A pair of a dylib and a set of symbols to be looked up.
  struct LookupRequest {
    LookupRequest(tpctypes::DylibHandle Handle, const SymbolLookupSet &Symbols)
        : Handle(Handle), Symbols(Symbols) {}
    tpctypes::DylibHandle Handle;
    const SymbolLookupSet &Symbols;
  };

  /// Contains the address of the dispatch function and context that the ORC
  /// runtime can use to call functions in the JIT.
  struct JITDispatchInfo {
    ExecutorAddress JITDispatchFunctionAddress;
    ExecutorAddress JITDispatchContextAddress;
  };

  virtual ~ExecutorProcessControl();

  /// Return the ExecutionSession associated with this instance.
  /// Not callable until the ExecutionSession has been associated.
  ExecutionSession &getExecutionSession() {
    assert(ES && "No ExecutionSession associated yet");
    return *ES;
  }

  /// Intern a symbol name in the SymbolStringPool.
  SymbolStringPtr intern(StringRef SymName) { return SSP->intern(SymName); }

  /// Return a shared pointer to the SymbolStringPool for this instance.
  std::shared_ptr<SymbolStringPool> getSymbolStringPool() const { return SSP; }

  /// Return the Triple for the target process.
  const Triple &getTargetTriple() const { return TargetTriple; }

  /// Get the page size for the target process.
  unsigned getPageSize() const { return PageSize; }

  /// Get the JIT dispatch function and context address for the executor.
  const JITDispatchInfo &getJITDispatchInfo() const { return JDI; }

  /// Return a MemoryAccess object for the target process.
  MemoryAccess &getMemoryAccess() const {
    assert(MemAccess && "No MemAccess object set.");
    return *MemAccess;
  }

  /// Return a JITLinkMemoryManager for the target process.
  jitlink::JITLinkMemoryManager &getMemMgr() const {
    assert(MemMgr && "No MemMgr object set");
    return *MemMgr;
  }

  /// Load the dynamic library at the given path and return a handle to it.
  /// If LibraryPath is null this function will return the global handle for
  /// the target process.
  virtual Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) = 0;

  /// Search for symbols in the target process.
  ///
  /// The result of the lookup is a 2-dimentional array of target addresses
  /// that correspond to the lookup order. If a required symbol is not
  /// found then this method will return an error. If a weakly referenced
  /// symbol is not found then it be assigned a '0' value in the result.
  /// that correspond to the lookup order.
  virtual Expected<std::vector<tpctypes::LookupResult>>
  lookupSymbols(ArrayRef<LookupRequest> Request) = 0;

  /// Run function with a main-like signature.
  virtual Expected<int32_t> runAsMain(JITTargetAddress MainFnAddr,
                                      ArrayRef<std::string> Args) = 0;

  /// Run a wrapper function in the executor.
  ///
  /// The wrapper function should be callable as:
  ///
  /// \code{.cpp}
  ///   CWrapperFunctionResult fn(uint8_t *Data, uint64_t Size);
  /// \endcode{.cpp}
  ///
  /// The given OnComplete function will be called to return the result.
  virtual void callWrapperAsync(SendResultFunction OnComplete,
                                JITTargetAddress WrapperFnAddr,
                                ArrayRef<char> ArgBuffer) = 0;

  /// Disconnect from the target process.
  ///
  /// This should be called after the JIT session is shut down.
  virtual Error disconnect() = 0;

protected:
  ExecutorProcessControl(std::shared_ptr<SymbolStringPool> SSP)
      : SSP(std::move(SSP)) {}

  std::shared_ptr<SymbolStringPool> SSP;
  ExecutionSession *ES = nullptr;
  Triple TargetTriple;
  unsigned PageSize = 0;
  JITDispatchInfo JDI;
  MemoryAccess *MemAccess = nullptr;
  jitlink::JITLinkMemoryManager *MemMgr = nullptr;
};

/// A ExecutorProcessControl instance that asserts if any of its methods are
/// used. Suitable for use is unit tests, and by ORC clients who haven't moved
/// to ExecutorProcessControl-based APIs yet.
class UnsupportedExecutorProcessControl : public ExecutorProcessControl {
public:
  UnsupportedExecutorProcessControl(
      std::shared_ptr<SymbolStringPool> SSP = nullptr,
      const std::string &TT = "", unsigned PageSize = 0)
      : ExecutorProcessControl(SSP ? std::move(SSP)
                                   : std::make_shared<SymbolStringPool>()) {
    this->TargetTriple = Triple(TT);
    this->PageSize = PageSize;
  }

  Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) override {
    llvm_unreachable("Unsupported");
  }

  Expected<std::vector<tpctypes::LookupResult>>
  lookupSymbols(ArrayRef<LookupRequest> Request) override {
    llvm_unreachable("Unsupported");
  }

  Expected<int32_t> runAsMain(JITTargetAddress MainFnAddr,
                              ArrayRef<std::string> Args) override {
    llvm_unreachable("Unsupported");
  }

  void callWrapperAsync(SendResultFunction OnComplete,
                        JITTargetAddress WrapperFnAddr,
                        ArrayRef<char> ArgBuffer) override {
    llvm_unreachable("Unsupported");
  }

  Error disconnect() override { return Error::success(); }
};

/// A ExecutorProcessControl implementation targeting the current process.
class SelfExecutorProcessControl
    : public ExecutorProcessControl,
      private ExecutorProcessControl::MemoryAccess {
public:
  SelfExecutorProcessControl(
      std::shared_ptr<SymbolStringPool> SSP, Triple TargetTriple,
      unsigned PageSize, std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr);

  /// Create a SelfExecutorProcessControl with the given symbol string pool and
  /// memory manager.
  /// If no symbol string pool is given then one will be created.
  /// If no memory manager is given a jitlink::InProcessMemoryManager will
  /// be created and used by default.
  static Expected<std::unique_ptr<SelfExecutorProcessControl>>
  Create(std::shared_ptr<SymbolStringPool> SSP = nullptr,
         std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr = nullptr);

  Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) override;

  Expected<std::vector<tpctypes::LookupResult>>
  lookupSymbols(ArrayRef<LookupRequest> Request) override;

  Expected<int32_t> runAsMain(JITTargetAddress MainFnAddr,
                              ArrayRef<std::string> Args) override;

  void callWrapperAsync(SendResultFunction OnComplete,
                        JITTargetAddress WrapperFnAddr,
                        ArrayRef<char> ArgBuffer) override;

  Error disconnect() override;

private:
  void writeUInt8s(ArrayRef<tpctypes::UInt8Write> Ws,
                   WriteResultFn OnWriteComplete) override;

  void writeUInt16s(ArrayRef<tpctypes::UInt16Write> Ws,
                    WriteResultFn OnWriteComplete) override;

  void writeUInt32s(ArrayRef<tpctypes::UInt32Write> Ws,
                    WriteResultFn OnWriteComplete) override;

  void writeUInt64s(ArrayRef<tpctypes::UInt64Write> Ws,
                    WriteResultFn OnWriteComplete) override;

  void writeBuffers(ArrayRef<tpctypes::BufferWrite> Ws,
                    WriteResultFn OnWriteComplete) override;

  static shared::detail::CWrapperFunctionResult
  jitDispatchViaWrapperFunctionManager(void *Ctx, const void *FnTag,
                                       const char *Data, size_t Size);

  std::unique_ptr<jitlink::JITLinkMemoryManager> OwnedMemMgr;
  char GlobalManglingPrefix = 0;
  std::vector<std::unique_ptr<sys::DynamicLibrary>> DynamicLibraries;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EXECUTORPROCESSCONTROL_H
