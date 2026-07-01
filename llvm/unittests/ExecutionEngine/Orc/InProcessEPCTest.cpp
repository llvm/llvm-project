//===- InProcessEPCTest.cpp -- Tests for InProcessEPC ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/InProcessEPC.h"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Shared/WrapperFunctionUtils.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"

#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::orc;

namespace {

// A minimal stand-in for orc_rt::InProcessControllerAccess. Owns a refcounted
// ConnectionImpl that mirrors the lifecycle semantics of the real one (refcount
// + Connected + InFlightCalls), exposes hooks for tests to react to incoming
// controller-side calls, and exposes helpers to drive cross-calls in the
// controller -> JIT direction.
class MockIPCA {
public:
  using Connection = InProcessEPC::Connection;
  using OnCallWrapperFn = unique_function<void(
      uint64_t CallId, void *Fn, shared::WrapperFunctionBuffer ArgBytes)>;
  using OnReturnJITDispatchResultFn = unique_function<void(
      uint64_t CallId, shared::WrapperFunctionBuffer ResultBytes)>;

  MockIPCA() : C(new ConnectionImpl(*this)) {}

  MockIPCA(const MockIPCA &) = delete;
  MockIPCA &operator=(const MockIPCA &) = delete;

  ~MockIPCA() { C->Release(C); }

  Connection *getConnection() { return C; }

  bool isConnected() const { return C->isConnected(); }

  void setOnCallWrapper(OnCallWrapperFn F) { OnCallWrapper = std::move(F); }
  void setOnReturnJITDispatchResult(OnReturnJITDispatchResultFn F) {
    OnReturnJITDispatchResult = std::move(F);
  }

  // Initiate a JIT-dispatch from the controller side. Returns the chosen
  // CallId, or std::nullopt if the message scope is closed.
  std::optional<uint64_t>
  callJITDispatch(void *HandlerTag, shared::WrapperFunctionBuffer ArgBytes) {
    if (C->EnterMessageScope(C)) {
      uint64_t CallId = NextCallId++;
      C->CallJITDispatch(C->IPEPC, CallId, HandlerTag, ArgBytes.release());
      C->LeaveMessageScope(C);
      return CallId;
    }
    return std::nullopt;
  }

  // Send a wrapper result back for a prior CallWrapper invocation, or with
  // an arbitrary CallId for the "unknown id" path.
  void returnWrapperResult(uint64_t CallId,
                           shared::WrapperFunctionBuffer ResultBytes) {
    if (C->EnterMessageScope(C)) {
      C->ReturnWrapperResult(C->IPEPC, CallId, ResultBytes.release());
      C->LeaveMessageScope(C);
    }
  }

private:
  struct ConnectionImpl : public Connection {
    ConnectionImpl(MockIPCA &Owner) {
      Retain = &retainEntry;
      Release = &releaseEntry;
      Disconnect = &disconnectEntry;
      EnterMessageScope = &enterMessageScopeEntry;
      LeaveMessageScope = &leaveMessageScopeEntry;
      IPCA = &Owner;
      CallWrapper = &callWrapperEntry;
      ReturnJITDispatchResult = &returnJITDispatchResultEntry;
    }

    bool isConnected() const {
      std::scoped_lock<std::mutex> Lock(M);
      return Connected;
    }

  private:
    void retain() {
      std::scoped_lock<std::mutex> Lock(M);
      ++RefCount;
    }

    static void retainEntry(Connection *C) {
      static_cast<ConnectionImpl *>(C)->retain();
    }

    bool release() {
      std::scoped_lock<std::mutex> Lock(M);
      return --RefCount == 0;
    }

    static void releaseEntry(Connection *C) {
      if (static_cast<ConnectionImpl *>(C)->release())
        delete static_cast<ConnectionImpl *>(C);
    }

    void disconnect() {
      std::unique_lock<std::mutex> Lock(M);
      if (!Connected)
        return;
      Connected = false;
      CV.wait(Lock, [this]() { return InFlightCalls == 0; });
    }

    static void disconnectEntry(Connection *C) {
      static_cast<ConnectionImpl *>(C)->disconnect();
    }

    int enterMessageScope() {
      std::scoped_lock<std::mutex> Lock(M);
      if (!Connected)
        return 0;
      ++InFlightCalls;
      return 1;
    }

    static int enterMessageScopeEntry(Connection *C) {
      return static_cast<ConnectionImpl *>(C)->enterMessageScope();
    }

    void leaveMessageScope() {
      bool Notify = false;
      {
        std::scoped_lock<std::mutex> Lock(M);
        --InFlightCalls;
        if (!Connected && InFlightCalls == 0)
          Notify = true;
      }
      if (Notify)
        CV.notify_one();
    }

    static void leaveMessageScopeEntry(Connection *C) {
      static_cast<ConnectionImpl *>(C)->leaveMessageScope();
    }

    mutable std::mutex M;
    std::condition_variable CV;
    bool Connected = true;
    size_t InFlightCalls = 0;
    size_t RefCount = 1;
  };

  static void callWrapperEntry(void *IPCA, uint64_t CallId, void *Fn,
                               shared::CWrapperFunctionBuffer ArgBytes) {
    auto *Self = static_cast<MockIPCA *>(IPCA);
    shared::WrapperFunctionBuffer Buf(ArgBytes);
    if (Self->OnCallWrapper)
      Self->OnCallWrapper(CallId, Fn, std::move(Buf));
  }

  static void
  returnJITDispatchResultEntry(void *IPCA, uint64_t CallId,
                               shared::CWrapperFunctionBuffer ResultBytes) {
    auto *Self = static_cast<MockIPCA *>(IPCA);
    shared::WrapperFunctionBuffer Buf(ResultBytes);
    if (Self->OnReturnJITDispatchResult)
      Self->OnReturnJITDispatchResult(CallId, std::move(Buf));
  }

  ConnectionImpl *C;
  uint64_t NextCallId = 0;
  OnCallWrapperFn OnCallWrapper;
  OnReturnJITDispatchResultFn OnReturnJITDispatchResult;
};

// Provides a BootstrapInfoAccess backed by test-supplied page-size, triple,
// value entries, and symbol entries, with knobs to simulate iteration errors.
class MockBootstrapInfoAccess : public InProcessEPC::BootstrapInfoAccess {
public:
  MockBootstrapInfoAccess() {
    GetPageSize = &getPageSizeEntry;
    GetTargetTriple = &getTargetTripleEntry;
    GetNextValue = &getNextValueEntry;
    GetNextSymbol = &getNextSymbolEntry;
  }

  void setPageSize(uint64_t PS) { PageSize = PS; }
  void setTargetTriple(std::string TT) { TargetTriple = std::move(TT); }
  void addValue(std::string Name, std::vector<char> Bytes) {
    Values.push_back({std::move(Name), std::move(Bytes)});
  }
  void addSymbol(std::string Name, uint64_t Addr) {
    Symbols.push_back({std::move(Name), Addr});
  }
  void setValueIterCorrupt() { ValuesCorrupt = true; }
  void setSymbolIterCorrupt() { SymbolsCorrupt = true; }

private:
  static uint64_t getPageSizeEntry(void *BIA) {
    return static_cast<MockBootstrapInfoAccess *>(BIA)->PageSize;
  }

  static const char *getTargetTripleEntry(void *BIA) {
    auto &Self = *static_cast<MockBootstrapInfoAccess *>(BIA);
    return Self.TargetTriple.empty() ? nullptr : Self.TargetTriple.c_str();
  }

  static int getNextValueEntry(void *BIA, const char **Name,
                               const char **ValueBytes, uint64_t *ValueSize) {
    auto &Self = *static_cast<MockBootstrapInfoAccess *>(BIA);
    if (Self.NextValue == Self.Values.size())
      return Self.ValuesCorrupt ? -1 : 0;
    auto &V = Self.Values[Self.NextValue++];
    *Name = V.first.c_str();
    *ValueBytes = V.second.data();
    *ValueSize = V.second.size();
    return 1;
  }

  static int getNextSymbolEntry(void *BIA, const char **Name, uint64_t *Addr) {
    auto &Self = *static_cast<MockBootstrapInfoAccess *>(BIA);
    if (Self.NextSymbol == Self.Symbols.size())
      return Self.SymbolsCorrupt ? -1 : 0;
    auto &S = Self.Symbols[Self.NextSymbol++];
    *Name = S.first.c_str();
    *Addr = S.second;
    return 1;
  }

  uint64_t PageSize = 4096;
  std::string TargetTriple = "x86_64-unknown-linux-gnu";
  std::vector<std::pair<std::string, std::vector<char>>> Values;
  std::vector<std::pair<std::string, uint64_t>> Symbols;
  size_t NextValue = 0;
  size_t NextSymbol = 0;
  bool ValuesCorrupt = false;
  bool SymbolsCorrupt = false;
};

Expected<std::unique_ptr<InProcessEPC>>
createIPEPC(MockIPCA &IPCA, MockBootstrapInfoAccess &BIA) {
  return InProcessEPC::Create(IPCA.getConnection(), &BIA);
}

} // namespace

TEST(InProcessEPCTest, CreateSuccess) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  BIA.setPageSize(8192);
  BIA.setTargetTriple("arm64-apple-darwin");
  BIA.addValue("greeting", {'h', 'i'});
  BIA.addSymbol("malloc", 0x1234);

  auto EPC = createIPEPC(IPCA, BIA);
  ASSERT_THAT_EXPECTED(EPC, Succeeded());

  EXPECT_EQ((*EPC)->getPageSize(), 8192U);
  EXPECT_EQ((*EPC)->getTargetTriple().str(), "arm64-apple-darwin");

  const auto &VMap = (*EPC)->getBootstrapMap();
  ASSERT_EQ(VMap.size(), 1U);
  auto VI = VMap.find("greeting");
  ASSERT_NE(VI, VMap.end());
  EXPECT_EQ(std::string(VI->second.begin(), VI->second.end()), "hi");

  const auto &SMap = (*EPC)->getBootstrapSymbolsMap();
  ASSERT_EQ(SMap.size(), 1U);
  auto SI = SMap.find("malloc");
  ASSERT_NE(SI, SMap.end());
  EXPECT_EQ(SI->second, ExecutorAddr(0x1234));

  auto *C = IPCA.getConnection();
  EXPECT_NE(C->IPEPC, nullptr);
  EXPECT_NE(C->CallJITDispatch, nullptr);
  EXPECT_NE(C->ReturnWrapperResult, nullptr);

  cantFail((*EPC)->disconnect());
}

TEST(InProcessEPCTest, CreateFailsOnZeroPageSize) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  BIA.setPageSize(0);

  auto EPC = createIPEPC(IPCA, BIA);
  EXPECT_THAT_EXPECTED(std::move(EPC), Failed());
  EXPECT_FALSE(IPCA.isConnected())
      << "Failed Create should leave the connection torn down";
}

TEST(InProcessEPCTest, CreateFailsOnEmptyTriple) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  BIA.setTargetTriple("");

  auto EPC = createIPEPC(IPCA, BIA);
  EXPECT_THAT_EXPECTED(std::move(EPC), Failed());
  EXPECT_FALSE(IPCA.isConnected());
}

TEST(InProcessEPCTest, CreateFailsOnDuplicateBootstrapValue) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  BIA.addValue("dup", {'a'});
  BIA.addValue("dup", {'b'});

  auto EPC = createIPEPC(IPCA, BIA);
  EXPECT_THAT_EXPECTED(std::move(EPC), Failed());
  EXPECT_FALSE(IPCA.isConnected());
}

TEST(InProcessEPCTest, CreateFailsOnDuplicateBootstrapSymbol) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  BIA.addSymbol("dup", 0x1);
  BIA.addSymbol("dup", 0x2);

  auto EPC = createIPEPC(IPCA, BIA);
  EXPECT_THAT_EXPECTED(std::move(EPC), Failed());
  EXPECT_FALSE(IPCA.isConnected());
}

TEST(InProcessEPCTest, CreateFailsOnCorruptedValueIteration) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  BIA.setValueIterCorrupt();

  auto EPC = createIPEPC(IPCA, BIA);
  EXPECT_THAT_EXPECTED(std::move(EPC), Failed());
  EXPECT_FALSE(IPCA.isConnected());
}

TEST(InProcessEPCTest, CreateFailsOnCorruptedSymbolIteration) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  BIA.setSymbolIterCorrupt();

  auto EPC = createIPEPC(IPCA, BIA);
  EXPECT_THAT_EXPECTED(std::move(EPC), Failed());
  EXPECT_FALSE(IPCA.isConnected());
}

#ifndef NDEBUG
TEST(InProcessEPCDeathTest, CreateAssertsOnNullIPCA) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  IPCA.getConnection()->IPCA = nullptr;

  EXPECT_DEATH(
      { (void)createIPEPC(IPCA, BIA); }, "C->IPCA not set by controller");
}
#endif

TEST(InProcessEPCTest, CallWrapperAsyncSuccess) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  auto EPCExp = createIPEPC(IPCA, BIA);
  ASSERT_THAT_EXPECTED(EPCExp, Succeeded());
  auto EPC = std::move(*EPCExp);

  // Echo back whatever the JIT side sent us.
  IPCA.setOnCallWrapper(
      [&](uint64_t CallId, void *, shared::WrapperFunctionBuffer ArgBytes) {
        IPCA.returnWrapperResult(CallId, std::move(ArgBytes));
      });

  std::optional<std::string> Result;
  std::string Payload = "hello";
  EPC->callWrapperAsync(ExecutorAddr(0x42),
                        ExecutorProcessControl::RunInPlace()(
                            [&](shared::WrapperFunctionBuffer R) {
                              ASSERT_FALSE(R.getOutOfBandError())
                                  << "Unexpected OOB error: "
                                  << R.getOutOfBandError();
                              Result = std::string(R.data(), R.size());
                            }),
                        ArrayRef<char>(Payload.data(), Payload.size()));

  ASSERT_TRUE(Result);
  EXPECT_EQ(*Result, Payload);

  cantFail(EPC->disconnect());
}

TEST(InProcessEPCTest, CallWrapperAsyncOutOfBandError) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  auto EPCExp = createIPEPC(IPCA, BIA);
  ASSERT_THAT_EXPECTED(EPCExp, Succeeded());
  auto EPC = std::move(*EPCExp);

  IPCA.setOnCallWrapper(
      [&](uint64_t CallId, void *, shared::WrapperFunctionBuffer) {
        IPCA.returnWrapperResult(
            CallId,
            shared::WrapperFunctionBuffer::createOutOfBandError("simulated"));
      });

  std::optional<std::string> ErrMsg;
  std::string Payload = "x";
  EPC->callWrapperAsync(ExecutorAddr(0x42),
                        ExecutorProcessControl::RunInPlace()(
                            [&](shared::WrapperFunctionBuffer R) {
                              if (const char *Msg = R.getOutOfBandError())
                                ErrMsg = Msg;
                            }),
                        ArrayRef<char>(Payload.data(), Payload.size()));

  ASSERT_TRUE(ErrMsg);
  EXPECT_EQ(*ErrMsg, "simulated");

  cantFail(EPC->disconnect());
}

TEST(InProcessEPCTest, CallWrapperAsyncFailsAfterDisconnect) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  auto EPCExp = createIPEPC(IPCA, BIA);
  ASSERT_THAT_EXPECTED(EPCExp, Succeeded());
  auto EPC = std::move(*EPCExp);

  cantFail(EPC->disconnect());

  std::optional<std::string> ErrMsg;
  std::string Payload = "x";
  EPC->callWrapperAsync(ExecutorAddr(0x42),
                        ExecutorProcessControl::RunInPlace()(
                            [&](shared::WrapperFunctionBuffer R) {
                              if (const char *Msg = R.getOutOfBandError())
                                ErrMsg = Msg;
                            }),
                        ArrayRef<char>(Payload.data(), Payload.size()));

  ASSERT_TRUE(ErrMsg);
  EXPECT_EQ(*ErrMsg, "connection closed");
}

TEST(InProcessEPCTest, DisconnectDrainsPendingCallWrapperHandlers) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  auto EPCExp = createIPEPC(IPCA, BIA);
  ASSERT_THAT_EXPECTED(EPCExp, Succeeded());
  auto EPC = std::move(*EPCExp);

  // Mock receives the call but never sends a result.
  IPCA.setOnCallWrapper([](uint64_t, void *, shared::WrapperFunctionBuffer) {});

  std::optional<std::string> ErrMsg;
  std::string Payload = "payload";
  EPC->callWrapperAsync(ExecutorAddr(0x42),
                        ExecutorProcessControl::RunInPlace()(
                            [&](shared::WrapperFunctionBuffer R) {
                              if (const char *Msg = R.getOutOfBandError())
                                ErrMsg = Msg;
                            }),
                        ArrayRef<char>(Payload.data(), Payload.size()));

  ASSERT_FALSE(ErrMsg) << "Handler fired before disconnect";

  cantFail(EPC->disconnect());

  ASSERT_TRUE(ErrMsg);
  EXPECT_EQ(*ErrMsg, "disconnected");
}

TEST(InProcessEPCTest, DestructorDrainsPendingCallWrapperHandlers) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  auto EPCExp = createIPEPC(IPCA, BIA);
  ASSERT_THAT_EXPECTED(EPCExp, Succeeded());
  auto EPC = std::move(*EPCExp);

  IPCA.setOnCallWrapper([](uint64_t, void *, shared::WrapperFunctionBuffer) {});

  std::optional<std::string> ErrMsg;
  std::string Payload = "x";
  EPC->callWrapperAsync(ExecutorAddr(0x42),
                        ExecutorProcessControl::RunInPlace()(
                            [&](shared::WrapperFunctionBuffer R) {
                              if (const char *Msg = R.getOutOfBandError())
                                ErrMsg = Msg;
                            }),
                        ArrayRef<char>(Payload.data(), Payload.size()));

  ASSERT_FALSE(ErrMsg);

  // Drop the IPEPC without an explicit disconnect call: the destructor must
  // still drive doDisconnect and drain the pending handler.
  EPC.reset();

  ASSERT_TRUE(ErrMsg);
  EXPECT_EQ(*ErrMsg, "disconnected");
}

TEST(InProcessEPCTest, DoubleDisconnectIsIdempotent) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  auto EPCExp = createIPEPC(IPCA, BIA);
  ASSERT_THAT_EXPECTED(EPCExp, Succeeded());
  auto EPC = std::move(*EPCExp);

  cantFail(EPC->disconnect());
  EXPECT_FALSE(IPCA.isConnected());

  // Second disconnect (here: via destructor) must not re-fire any handlers or
  // double-release the connection -- the test would deadlock/segfault if it
  // did.
  EPC.reset();
  EXPECT_FALSE(IPCA.isConnected());
}

namespace {

// Custom EPC-attached ES that hands the EPC ownership to the session but
// retains a raw pointer for tests that need to drive the EPC directly.
struct SessionFixture {
  SessionFixture(std::unique_ptr<InProcessEPC> EPC)
      : EPCPtr(EPC.get()), ES(std::move(EPC)),
        JD(ES.createBareJITDylib("TestJD")) {}

  ~SessionFixture() { cantFail(ES.endSession()); }

  InProcessEPC *EPCPtr;
  ExecutionSession ES;
  JITDylib &JD;
};

} // namespace

TEST(InProcessEPCTest, ReturnWrapperResultForInvalidCallIdIsReported) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  auto EPCExp = createIPEPC(IPCA, BIA);
  ASSERT_THAT_EXPECTED(EPCExp, Succeeded());

  SessionFixture Fix(std::move(*EPCExp));

  std::string CapturedErr;
  Fix.ES.setErrorReporter(
      [&](Error E) { CapturedErr = toString(std::move(E)); });

  IPCA.returnWrapperResult(
      /*CallId=*/0xdead, shared::WrapperFunctionBuffer::copyFrom("ignored", 7));

  EXPECT_NE(CapturedErr.find("invalid call id"), std::string::npos)
      << "Expected invalid-call-id report, got: " << CapturedErr;
}

TEST(InProcessEPCTest, JITDispatchSuccess) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  auto EPCExp = createIPEPC(IPCA, BIA);
  ASSERT_THAT_EXPECTED(EPCExp, Succeeded());

  SessionFixture Fix(std::move(*EPCExp));

  constexpr ExecutorAddr TagAddr(0xCAFEBABE);
  auto Tag = Fix.ES.intern("echo_tag");
  cantFail(Fix.JD.define(
      absoluteSymbols({{Tag, {TagAddr, JITSymbolFlags::Exported}}})));

  ExecutionSession::JITDispatchHandlerAssociationMap Assocs;
  Assocs[Tag] = [](ExecutionSession::SendResultFunction SendResult,
                   const char *ArgData, size_t ArgSize) {
    SendResult(shared::WrapperFunctionBuffer::copyFrom(ArgData, ArgSize));
  };
  cantFail(Fix.ES.registerJITDispatchHandlers(Fix.JD, std::move(Assocs)));

  std::optional<std::string> Result;
  std::optional<uint64_t> RxCallId;
  IPCA.setOnReturnJITDispatchResult(
      [&](uint64_t CallId, shared::WrapperFunctionBuffer ResultBytes) {
        RxCallId = CallId;
        Result = std::string(ResultBytes.data(), ResultBytes.size());
      });

  auto SentCallId =
      IPCA.callJITDispatch(TagAddr.toPtr<void *>(),
                           shared::WrapperFunctionBuffer::copyFrom("ping", 4));

  ASSERT_TRUE(SentCallId);
  ASSERT_TRUE(Result);
  EXPECT_EQ(*Result, "ping");
  ASSERT_TRUE(RxCallId);
  EXPECT_EQ(*RxCallId, *SentCallId);
}

TEST(InProcessEPCTest, JITDispatchUnknownHandler) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  auto EPCExp = createIPEPC(IPCA, BIA);
  ASSERT_THAT_EXPECTED(EPCExp, Succeeded());

  SessionFixture Fix(std::move(*EPCExp));

  std::optional<std::string> ErrMsg;
  IPCA.setOnReturnJITDispatchResult(
      [&](uint64_t, shared::WrapperFunctionBuffer R) {
        if (const char *Msg = R.getOutOfBandError())
          ErrMsg = Msg;
      });

  auto SentCallId =
      IPCA.callJITDispatch(reinterpret_cast<void *>(0xdeadbeef),
                           shared::WrapperFunctionBuffer::copyFrom("x", 1));
  ASSERT_TRUE(SentCallId);

  ASSERT_TRUE(ErrMsg)
      << "Expected ReturnJITDispatchResult to deliver an OOB error";
}

TEST(InProcessEPCTest, JITDispatchAfterDisconnectIsDropped) {
  MockIPCA IPCA;
  MockBootstrapInfoAccess BIA;
  auto EPCExp = createIPEPC(IPCA, BIA);
  ASSERT_THAT_EXPECTED(EPCExp, Succeeded());

  SessionFixture Fix(std::move(*EPCExp));

  bool ResultFired = false;
  IPCA.setOnReturnJITDispatchResult(
      [&](uint64_t, shared::WrapperFunctionBuffer) { ResultFired = true; });

  // Tear down the connection. After this the controller side's
  // EnterMessageScope returns 0 and CallJITDispatch is never invoked.
  cantFail(Fix.EPCPtr->disconnect());

  auto SentCallId =
      IPCA.callJITDispatch(reinterpret_cast<void *>(0xdeadbeef),
                           shared::WrapperFunctionBuffer::copyFrom("x", 1));

  EXPECT_FALSE(SentCallId)
      << "callJITDispatch should be dropped after disconnect";
  EXPECT_FALSE(ResultFired);
}
