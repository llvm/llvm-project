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
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/TargetProcessControlTypes.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/MSVCErrorWorkarounds.h"

#include <future>
#include <mutex>
#include <vector>

namespace llvm {
namespace orc {

/// ExecutorProcessControl supports interaction with a JIT target process.
class ExecutorProcessControl {
public:
  /// Sender to return the result of a WrapperFunction executed in the JIT.
  using SendResultFunction =
      unique_function<void(shared::WrapperFunctionResult)>;

  /// An asynchronous wrapper-function.
  using AsyncWrapperFunction = unique_function<void(
      SendResultFunction SendResult, const char *ArgData, size_t ArgSize)>;

  /// A map associating tag names with asynchronous wrapper function
  /// implementations in the JIT.
  using WrapperFunctionAssociationMap =
      DenseMap<SymbolStringPtr, AsyncWrapperFunction>;

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

  virtual ~ExecutorProcessControl();

  /// Intern a symbol name in the SymbolStringPool.
  SymbolStringPtr intern(StringRef SymName) { return SSP->intern(SymName); }

  /// Return a shared pointer to the SymbolStringPool for this instance.
  std::shared_ptr<SymbolStringPool> getSymbolStringPool() const { return SSP; }

  /// Return the Triple for the target process.
  const Triple &getTargetTriple() const { return TargetTriple; }

  /// Get the page size for the target process.
  unsigned getPageSize() const { return PageSize; }

  /// Return a MemoryAccess object for the target process.
  MemoryAccess &getMemoryAccess() const { return *MemAccess; }

  /// Return a JITLinkMemoryManager for the target process.
  jitlink::JITLinkMemoryManager &getMemMgr() const { return *MemMgr; }

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

  /// Run a wrapper function in the executor (async version).
  ///
  /// The wrapper function should be callable as:
  ///
  /// \code{.cpp}
  ///   CWrapperFunctionResult fn(uint8_t *Data, uint64_t Size);
  /// \endcode{.cpp}
  ///
  /// The given OnComplete function will be called to return the result.
  virtual void runWrapperAsync(SendResultFunction OnComplete,
                               JITTargetAddress WrapperFnAddr,
                               ArrayRef<char> ArgBuffer) = 0;

  /// Run a wrapper function in the executor. The wrapper function should be
  /// callable as:
  ///
  /// \code{.cpp}
  ///   CWrapperFunctionResult fn(uint8_t *Data, uint64_t Size);
  /// \endcode{.cpp}
  shared::WrapperFunctionResult runWrapper(JITTargetAddress WrapperFnAddr,
                                           ArrayRef<char> ArgBuffer) {
    std::promise<shared::WrapperFunctionResult> RP;
    auto RF = RP.get_future();
    runWrapperAsync(
        [&](shared::WrapperFunctionResult R) { RP.set_value(std::move(R)); },
        WrapperFnAddr, ArgBuffer);
    return RF.get();
  }

  /// Run a wrapper function using SPS to serialize the arguments and
  /// deserialize the results.
  template <typename SPSSignature, typename SendResultT, typename... ArgTs>
  void runSPSWrapperAsync(SendResultT &&SendResult,
                          JITTargetAddress WrapperFnAddr,
                          const ArgTs &...Args) {
    shared::WrapperFunction<SPSSignature>::callAsync(
        [this, WrapperFnAddr](SendResultFunction SendResult,
                              const char *ArgData, size_t ArgSize) {
          runWrapperAsync(std::move(SendResult), WrapperFnAddr,
                          ArrayRef<char>(ArgData, ArgSize));
        },
        std::move(SendResult), Args...);
  }

  /// Run a wrapper function using SPS to serialize the arguments and
  /// deserialize the results.
  template <typename SPSSignature, typename RetT, typename... ArgTs>
  Error runSPSWrapper(JITTargetAddress WrapperFnAddr, RetT &RetVal,
                      const ArgTs &...Args) {
    return shared::WrapperFunction<SPSSignature>::call(
        [this, WrapperFnAddr](const char *ArgData, size_t ArgSize) {
          return runWrapper(WrapperFnAddr, ArrayRef<char>(ArgData, ArgSize));
        },
        RetVal, Args...);
  }

  /// Wrap a handler that takes concrete argument types (and a sender for a
  /// concrete return type) to produce an AsyncWrapperFunction. Uses SPS to
  /// unpack the arguments and pack the result.
  ///
  /// This function is usually used when building association maps.
  template <typename SPSSignature, typename HandlerT>
  static AsyncWrapperFunction wrapAsyncWithSPS(HandlerT &&H) {
    return [H = std::forward<HandlerT>(H)](SendResultFunction SendResult,
                                           const char *ArgData,
                                           size_t ArgSize) mutable {
      shared::WrapperFunction<SPSSignature>::handleAsync(ArgData, ArgSize, H,
                                                         std::move(SendResult));
    };
  }

  /// For each symbol name, associate the AsyncWrapperFunction implementation
  /// value with the address of that symbol.
  ///
  /// Symbols will be looked up using LookupKind::Static,
  /// JITDylibLookupFlags::MatchAllSymbols (hidden tags will be found), and
  /// LookupFlags::WeaklyReferencedSymbol (missing tags will not cause an
  /// error, the implementations will simply be dropped).
  Error associateJITSideWrapperFunctions(JITDylib &JD,
                                         WrapperFunctionAssociationMap WFs);

  /// Run a registered jit-side wrapper function.
  void runJITSideWrapperFunction(SendResultFunction SendResult,
                                 JITTargetAddress TagAddr,
                                 ArrayRef<char> ArgBuffer);

  /// Disconnect from the target process.
  ///
  /// This should be called after the JIT session is shut down.
  virtual Error disconnect() = 0;

protected:
  ExecutorProcessControl(std::shared_ptr<SymbolStringPool> SSP)
      : SSP(std::move(SSP)) {}

  std::shared_ptr<SymbolStringPool> SSP;
  Triple TargetTriple;
  unsigned PageSize = 0;
  MemoryAccess *MemAccess = nullptr;
  jitlink::JITLinkMemoryManager *MemMgr = nullptr;

  std::mutex TagToFuncMapMutex;
  DenseMap<JITTargetAddress, std::shared_ptr<AsyncWrapperFunction>> TagToFunc;
};

/// Call a wrapper function via ExecutorProcessControl::runWrapper.
class EPCCaller {
public:
  EPCCaller(ExecutorProcessControl &EPC, JITTargetAddress WrapperFnAddr)
      : EPC(EPC), WrapperFnAddr(WrapperFnAddr) {}
  shared::WrapperFunctionResult operator()(const char *ArgData,
                                           size_t ArgSize) const {
    return EPC.runWrapper(WrapperFnAddr, ArrayRef<char>(ArgData, ArgSize));
  }

private:
  ExecutorProcessControl &EPC;
  JITTargetAddress WrapperFnAddr;
};

/// A ExecutorProcessControl implementation targeting the current process.
class SelfExecutorProcessControl
    : public ExecutorProcessControl,
      private ExecutorProcessControl::MemoryAccess {
public:
  SelfExecutorProcessControl(
      std::shared_ptr<SymbolStringPool> SSP, Triple TargetTriple,
      unsigned PageSize, std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr);

  /// Create a SelfExecutorProcessControl with the given memory manager.
  /// If no memory manager is given a jitlink::InProcessMemoryManager will
  /// be used by default.
  static Expected<std::unique_ptr<SelfExecutorProcessControl>>
  Create(std::shared_ptr<SymbolStringPool> SSP,
         std::unique_ptr<jitlink::JITLinkMemoryManager> MemMgr = nullptr);

  Expected<tpctypes::DylibHandle> loadDylib(const char *DylibPath) override;

  Expected<std::vector<tpctypes::LookupResult>>
  lookupSymbols(ArrayRef<LookupRequest> Request) override;

  Expected<int32_t> runAsMain(JITTargetAddress MainFnAddr,
                              ArrayRef<std::string> Args) override;

  void runWrapperAsync(SendResultFunction OnComplete,
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

  std::unique_ptr<jitlink::JITLinkMemoryManager> OwnedMemMgr;
  char GlobalManglingPrefix = 0;
  std::vector<std::unique_ptr<sys::DynamicLibrary>> DynamicLibraries;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EXECUTORPROCESSCONTROL_H
