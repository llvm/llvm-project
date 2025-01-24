//===--- JITLinkTestUtils.h - Utilities for JITLink unit tests --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for JITLink unit tests.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_EXECUTIONENGINE_JITLINK_JITLINKTESTUTILS_H
#define LLVM_UNITTESTS_EXECUTIONENGINE_JITLINK_JITLINKTESTUTILS_H

#include "llvm/ExecutionEngine/JITLink/JITLink.h"

class MockJITLinkMemoryManager : public llvm::jitlink::JITLinkMemoryManager {
public:
  class Alloc {
  public:
    virtual ~Alloc() {}
  };

  class SimpleAlloc : public Alloc {
  public:
    SimpleAlloc(const llvm::jitlink::JITLinkDylib *JD,
                llvm::jitlink::LinkGraph &G) {
      for (auto *B : G.blocks())
        (void)B->getMutableContent(G);
    }
  };

  class MockInFlightAlloc : public InFlightAlloc {
  public:
    MockInFlightAlloc(MockJITLinkMemoryManager &MJMM, std::unique_ptr<Alloc> A)
        : MJMM(MJMM), A(std::move(A)) {}

    void abandon(OnAbandonedFunction OnAbandoned) override {
      OnAbandoned(MJMM.Abandon(std::move(A)));
    }

    void finalize(OnFinalizedFunction OnFinalized) override {
      OnFinalized(MJMM.Finalize(std::move(A)));
    }

  private:
    MockJITLinkMemoryManager &MJMM;
    std::unique_ptr<Alloc> A;
  };

  MockJITLinkMemoryManager() {
    Allocate = [this](const llvm::jitlink::JITLinkDylib *JD,
                      llvm::jitlink::LinkGraph &G) {
      return defaultAllocate(JD, G);
    };

    Deallocate = [this](std::vector<FinalizedAlloc> Allocs) {
      return defaultDeallocate(std::move(Allocs));
    };

    Abandon = [this](std::unique_ptr<Alloc> A) {
      return defaultAbandon(std::move(A));
    };

    Finalize = [this](std::unique_ptr<Alloc> A) {
      return defaultFinalize(std::move(A));
    };
  }

  void allocate(const llvm::jitlink::JITLinkDylib *JD,
                llvm::jitlink::LinkGraph &G,
                OnAllocatedFunction OnAllocated) override {
    auto A = Allocate(JD, G);
    if (!A)
      OnAllocated(A.takeError());
    else
      OnAllocated(std::make_unique<MockInFlightAlloc>(*this, std::move(*A)));
  }

  void deallocate(std::vector<FinalizedAlloc> Allocs,
                  OnDeallocatedFunction OnDeallocated) override {
    OnDeallocated(Deallocate(std::move(Allocs)));
  }

  using JITLinkMemoryManager::allocate;
  using JITLinkMemoryManager::deallocate;

  llvm::Expected<std::unique_ptr<Alloc>>
  defaultAllocate(const llvm::jitlink::JITLinkDylib *JD,
                  llvm::jitlink::LinkGraph &G) {
    return std::make_unique<SimpleAlloc>(JD, G);
  }

  llvm::Error defaultDeallocate(std::vector<FinalizedAlloc> Allocs) {
    for (auto &A : Allocs)
      delete A.release().toPtr<Alloc *>();
    return llvm::Error::success();
  }

  llvm::Error defaultAbandon(std::unique_ptr<Alloc> A) {
    return llvm::Error::success();
  }

  llvm::Expected<FinalizedAlloc> defaultFinalize(std::unique_ptr<Alloc> A) {
    return FinalizedAlloc(llvm::orc::ExecutorAddr::fromPtr(A.release()));
  }

  llvm::unique_function<llvm::Expected<std::unique_ptr<Alloc>>(
      const llvm::jitlink::JITLinkDylib *, llvm::jitlink::LinkGraph &)>
      Allocate;
  llvm::unique_function<llvm::Error(std::vector<FinalizedAlloc>)> Deallocate;
  llvm::unique_function<llvm::Error(std::unique_ptr<Alloc>)> Abandon;
  llvm::unique_function<llvm::Expected<FinalizedAlloc>(std::unique_ptr<Alloc>)>
      Finalize;
};

void lookupResolveEverythingToNull(
    const llvm::jitlink::JITLinkContext::LookupMap &Symbols,
    std::unique_ptr<llvm::jitlink::JITLinkAsyncLookupContinuation> LC);

void lookupErrorOut(
    const llvm::jitlink::JITLinkContext::LookupMap &Symbols,
    std::unique_ptr<llvm::jitlink::JITLinkAsyncLookupContinuation> LC);

class MockJITLinkContext : public llvm::jitlink::JITLinkContext {
public:
  using HandleFailedFn = llvm::unique_function<void(llvm::Error)>;

  MockJITLinkContext(std::unique_ptr<MockJITLinkMemoryManager> MJMM,
                     HandleFailedFn HandleFailed)
      : JITLinkContext(&JD), MJMM(std::move(MJMM)),
        HandleFailed(std::move(HandleFailed)) {}

  ~MockJITLinkContext() {
    if (auto Err = MJMM->deallocate(std::move(FinalizedAllocs)))
      notifyFailed(std::move(Err));
  }

  llvm::jitlink::JITLinkMemoryManager &getMemoryManager() override {
    return *MJMM;
  }

  void notifyFailed(llvm::Error Err) override { HandleFailed(std::move(Err)); }

  void lookup(const LookupMap &Symbols,
              std::unique_ptr<llvm::jitlink::JITLinkAsyncLookupContinuation> LC)
      override {
    Lookup(Symbols, std::move(LC));
  }

  llvm::Error notifyResolved(llvm::jitlink::LinkGraph &G) override {
    return NotifyResolved ? NotifyResolved(G) : llvm::Error::success();
  }

  void notifyFinalized(
      llvm::jitlink::JITLinkMemoryManager::FinalizedAlloc Alloc) override {
    if (NotifyFinalized)
      NotifyFinalized(std::move(Alloc));
    else
      FinalizedAllocs.push_back(std::move(Alloc));
  }

  bool shouldAddDefaultTargetPasses(const llvm::Triple &TT) const override {
    return true;
  }

  llvm::jitlink::LinkGraphPassFunction
  getMarkLivePass(const llvm::Triple &TT) const override {
    return MarkLivePass ? llvm::jitlink::LinkGraphPassFunction(
                              [this](llvm::jitlink::LinkGraph &G) {
                                return MarkLivePass(G);
                              })
                        : llvm::jitlink::LinkGraphPassFunction(
                              [](llvm::jitlink::LinkGraph &G) {
                                return markAllSymbolsLive(G);
                              });
  }

  llvm::Error
  modifyPassConfig(llvm::jitlink::LinkGraph &G,
                   llvm::jitlink::PassConfiguration &Config) override {
    if (ModifyPassConfig)
      return ModifyPassConfig(G, Config);
    return llvm::Error::success();
  }

  llvm::jitlink::JITLinkDylib JD{"JD"};
  std::unique_ptr<MockJITLinkMemoryManager> MJMM;
  HandleFailedFn HandleFailed;
  llvm::unique_function<void(
      const LookupMap &,
      std::unique_ptr<llvm::jitlink::JITLinkAsyncLookupContinuation>)>
      Lookup;
  llvm::unique_function<llvm::Error(llvm::jitlink::LinkGraph &)> NotifyResolved;
  llvm::unique_function<void(
      llvm::jitlink::JITLinkMemoryManager::FinalizedAlloc)>
      NotifyFinalized;
  mutable llvm::unique_function<llvm::Error(llvm::jitlink::LinkGraph &)>
      MarkLivePass;
  llvm::unique_function<llvm::Error(llvm::jitlink::LinkGraph &,
                                    llvm::jitlink::PassConfiguration &)>
      ModifyPassConfig;

  std::vector<llvm::jitlink::JITLinkMemoryManager::FinalizedAlloc>
      FinalizedAllocs;
};

std::unique_ptr<MockJITLinkContext> makeMockContext(
    llvm::unique_function<void(llvm::Error)> HandleFailed,
    llvm::unique_function<void(MockJITLinkMemoryManager &)> SetupMemMgr,
    llvm::unique_function<void(MockJITLinkContext &)> SetupContext);

void defaultMemMgrSetup(MockJITLinkMemoryManager &);
void defaultCtxSetup(MockJITLinkContext &);

class JoinErrorsInto {
public:
  JoinErrorsInto(llvm::Error &Err) : Err(Err) {}
  void operator()(llvm::Error E2) {
    Err = llvm::joinErrors(std::move(Err), std::move(E2));
  }

private:
  llvm::Error &Err;
};

extern llvm::ArrayRef<char> BlockContent;

#endif // LLVM_UNITTESTS_EXECUTIONENGINE_JITLINK_JITLINKTESTUTILS_H
