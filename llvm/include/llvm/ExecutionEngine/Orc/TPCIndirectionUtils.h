//===--- TPCIndirectionUtils.h - TPC based indirection utils ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Indirection utilities (stubs, trampolines, lazy call-throughs) that use the
// TargetProcessControl API to interact with the target process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_TPCINDIRECTIONUTILS_H
#define LLVM_EXECUTIONENGINE_ORC_TPCINDIRECTIONUTILS_H

#include "llvm/ExecutionEngine/JITLink/JITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"
#include "llvm/ExecutionEngine/Orc/LazyReexports.h"

#include <mutex>

namespace llvm {
namespace orc {

class TargetProcessControl;

/// Provides TargetProcessControl based indirect stubs, trampoline pool and
/// lazy call through manager.
class TPCIndirectionUtils {
  friend class TPCIndirectionUtilsAccess;

public:
  /// ABI support base class. Used to write resolver, stub, and trampoline
  /// blocks.
  class ABISupport {
  protected:
    ABISupport(unsigned PointerSize, unsigned TrampolineSize, unsigned StubSize,
               unsigned StubToPointerMaxDisplacement, unsigned ResolverCodeSize)
        : PointerSize(PointerSize), TrampolineSize(TrampolineSize),
          StubSize(StubSize),
          StubToPointerMaxDisplacement(StubToPointerMaxDisplacement),
          ResolverCodeSize(ResolverCodeSize) {}

  public:
    virtual ~ABISupport();

    unsigned getPointerSize() const { return PointerSize; }
    unsigned getTrampolineSize() const { return TrampolineSize; }
    unsigned getStubSize() const { return StubSize; }
    unsigned getStubToPointerMaxDisplacement() const {
      return StubToPointerMaxDisplacement;
    }
    unsigned getResolverCodeSize() const { return ResolverCodeSize; }

    virtual void writeResolverCode(char *ResolverWorkingMem,
                                   JITTargetAddress ResolverTargetAddr,
                                   JITTargetAddress ReentryFnAddr,
                                   JITTargetAddress ReentryCtxAddr) const = 0;

    virtual void writeTrampolines(char *TrampolineBlockWorkingMem,
                                  JITTargetAddress TrampolineBlockTragetAddr,
                                  JITTargetAddress ResolverAddr,
                                  unsigned NumTrampolines) const = 0;

    virtual void
    writeIndirectStubsBlock(char *StubsBlockWorkingMem,
                            JITTargetAddress StubsBlockTargetAddress,
                            JITTargetAddress PointersBlockTargetAddress,
                            unsigned NumStubs) const = 0;

  private:
    unsigned PointerSize = 0;
    unsigned TrampolineSize = 0;
    unsigned StubSize = 0;
    unsigned StubToPointerMaxDisplacement = 0;
    unsigned ResolverCodeSize = 0;
  };

  /// Create using the given ABI class.
  template <typename ORCABI>
  static std::unique_ptr<TPCIndirectionUtils>
  CreateWithABI(TargetProcessControl &TPC);

  /// Create based on the TargetProcessControl triple.
  static Expected<std::unique_ptr<TPCIndirectionUtils>>
  Create(TargetProcessControl &TPC);

  /// Return a reference to the TargetProcessControl object.
  TargetProcessControl &getTargetProcessControl() const { return TPC; }

  /// Return a reference to the ABISupport object for this instance.
  ABISupport &getABISupport() const { return *ABI; }

  /// Release memory for resources held by this instance. This *must* be called
  /// prior to destruction of the class.
  Error cleanup();

  /// Write resolver code to the target process and return its address.
  /// This must be called before any call to createTrampolinePool or
  /// createLazyCallThroughManager.
  Expected<JITTargetAddress>
  writeResolverBlock(JITTargetAddress ReentryFnAddr,
                     JITTargetAddress ReentryCtxAddr);

  /// Returns the address of the Resolver block. Returns zero if the
  /// writeResolverBlock method has not previously been called.
  JITTargetAddress getResolverBlockAddress() const { return ResolverBlockAddr; }

  /// Create an IndirectStubsManager for the target process.
  std::unique_ptr<IndirectStubsManager> createIndirectStubsManager();

  /// Create a TrampolinePool for the target process.
  TrampolinePool &getTrampolinePool();

  /// Create a LazyCallThroughManager.
  /// This function should only be called once.
  LazyCallThroughManager &
  createLazyCallThroughManager(ExecutionSession &ES,
                               JITTargetAddress ErrorHandlerAddr);

  /// Create a LazyCallThroughManager for the target process.
  LazyCallThroughManager &getLazyCallThroughManager() {
    assert(LCTM && "createLazyCallThroughManager must be called first");
    return *LCTM;
  }

private:
  using Allocation = jitlink::JITLinkMemoryManager::Allocation;

  struct IndirectStubInfo {
    IndirectStubInfo() = default;
    IndirectStubInfo(JITTargetAddress StubAddress,
                     JITTargetAddress PointerAddress)
        : StubAddress(StubAddress), PointerAddress(PointerAddress) {}
    JITTargetAddress StubAddress = 0;
    JITTargetAddress PointerAddress = 0;
  };

  using IndirectStubInfoVector = std::vector<IndirectStubInfo>;

  /// Create a TPCIndirectionUtils instance.
  TPCIndirectionUtils(TargetProcessControl &TPC,
                      std::unique_ptr<ABISupport> ABI);

  Expected<IndirectStubInfoVector> getIndirectStubs(unsigned NumStubs);

  std::mutex TPCUIMutex;
  TargetProcessControl &TPC;
  std::unique_ptr<ABISupport> ABI;
  JITTargetAddress ResolverBlockAddr;
  std::unique_ptr<jitlink::JITLinkMemoryManager::Allocation> ResolverBlock;
  std::unique_ptr<TrampolinePool> TP;
  std::unique_ptr<LazyCallThroughManager> LCTM;

  std::vector<IndirectStubInfo> AvailableIndirectStubs;
  std::vector<std::unique_ptr<Allocation>> IndirectStubAllocs;
};

namespace detail {

template <typename ORCABI>
class ABISupportImpl : public TPCIndirectionUtils::ABISupport {
public:
  ABISupportImpl()
      : ABISupport(ORCABI::PointerSize, ORCABI::TrampolineSize,
                   ORCABI::StubSize, ORCABI::StubToPointerMaxDisplacement,
                   ORCABI::ResolverCodeSize) {}

  void writeResolverCode(char *ResolverWorkingMem,
                         JITTargetAddress ResolverTargetAddr,
                         JITTargetAddress ReentryFnAddr,
                         JITTargetAddress ReentryCtxAddr) const override {
    ORCABI::writeResolverCode(ResolverWorkingMem, ResolverTargetAddr,
                              ReentryFnAddr, ReentryCtxAddr);
  }

  void writeTrampolines(char *TrampolineBlockWorkingMem,
                        JITTargetAddress TrampolineBlockTargetAddr,
                        JITTargetAddress ResolverAddr,
                        unsigned NumTrampolines) const override {
    ORCABI::writeTrampolines(TrampolineBlockWorkingMem,
                             TrampolineBlockTargetAddr, ResolverAddr,
                             NumTrampolines);
  }

  void writeIndirectStubsBlock(char *StubsBlockWorkingMem,
                               JITTargetAddress StubsBlockTargetAddress,
                               JITTargetAddress PointersBlockTargetAddress,
                               unsigned NumStubs) const override {
    ORCABI::writeIndirectStubsBlock(StubsBlockWorkingMem,
                                    StubsBlockTargetAddress,
                                    PointersBlockTargetAddress, NumStubs);
  }
};

} // end namespace detail

template <typename ORCABI>
std::unique_ptr<TPCIndirectionUtils>
TPCIndirectionUtils::CreateWithABI(TargetProcessControl &TPC) {
  return std::unique_ptr<TPCIndirectionUtils>(new TPCIndirectionUtils(
      TPC, std::make_unique<detail::ABISupportImpl<ORCABI>>()));
}

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_TPCINDIRECTIONUTILS_H
