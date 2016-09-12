//===----------- RPCUtilsTest.cpp - Unit tests the Orc RPC utils ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/RPCByteChannel.h"
#include "llvm/ExecutionEngine/Orc/RPCUtils.h"
#include "gtest/gtest.h"

#include <queue>

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::remote;

class Queue : public std::queue<char> {
public:
  std::mutex &getLock() { return Lock; }

private:
  std::mutex Lock;
};

class QueueChannel : public RPCByteChannel {
public:
  QueueChannel(Queue &InQueue, Queue &OutQueue)
      : InQueue(InQueue), OutQueue(OutQueue) {}

  Error readBytes(char *Dst, unsigned Size) override {
    while (Size != 0) {
      // If there's nothing to read then yield.
      while (InQueue.empty())
        std::this_thread::yield();

      // Lock the channel and read what we can.
      std::lock_guard<std::mutex> Lock(InQueue.getLock());
      while (!InQueue.empty() && Size) {
        *Dst++ = InQueue.front();
        --Size;
        InQueue.pop();
      }
    }
    return Error::success();
  }

  Error appendBytes(const char *Src, unsigned Size) override {
    std::lock_guard<std::mutex> Lock(OutQueue.getLock());
    while (Size--)
      OutQueue.push(*Src++);
    return Error::success();
  }

  Error send() override { return Error::success(); }

private:
  Queue &InQueue;
  Queue &OutQueue;
};

class DummyRPC : public testing::Test, public RPC<QueueChannel> {
public:
  enum FuncId : uint32_t {
    VoidBoolId = RPCFunctionIdTraits<FuncId>::FirstValidId,
    IntIntId,
    AllTheTypesId
  };

  typedef Function<VoidBoolId, void(bool)> VoidBool;
  typedef Function<IntIntId, int32_t(int32_t)> IntInt;
  typedef Function<AllTheTypesId,
                   void(int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                        int64_t, uint64_t, bool, std::string, std::vector<int>)>
      AllTheTypes;
};

TEST_F(DummyRPC, TestAsyncVoidBool) {
  Queue Q1, Q2;
  QueueChannel C1(Q1, Q2);
  QueueChannel C2(Q2, Q1);

  // Make an async call.
  auto ResOrErr = callNBWithSeq<VoidBool>(C1, true);
  EXPECT_TRUE(!!ResOrErr) << "Simple call over queue failed";

  {
    // Expect a call to Proc1.
    auto EC = expect<VoidBool>(C2, [&](bool &B) {
      EXPECT_EQ(B, true) << "Bool serialization broken";
      return Error::success();
    });
    EXPECT_FALSE(EC) << "Simple expect over queue failed";
  }

  {
    // Wait for the result.
    auto EC = waitForResult(C1, ResOrErr->second, handleNone);
    EXPECT_FALSE(EC) << "Could not read result.";
  }

  // Verify that the function returned ok.
  auto Err = ResOrErr->first.get();
  EXPECT_FALSE(!!Err) << "Remote void function failed to execute.";
}

TEST_F(DummyRPC, TestAsyncIntInt) {
  Queue Q1, Q2;
  QueueChannel C1(Q1, Q2);
  QueueChannel C2(Q2, Q1);

  // Make an async call.
  auto ResOrErr = callNBWithSeq<IntInt>(C1, 21);
  EXPECT_TRUE(!!ResOrErr) << "Simple call over queue failed";

  {
    // Expect a call to Proc1.
    auto EC = expect<IntInt>(C2, [&](int32_t I) -> Expected<int32_t> {
      EXPECT_EQ(I, 21) << "Bool serialization broken";
      return 2 * I;
    });
    EXPECT_FALSE(EC) << "Simple expect over queue failed";
  }

  {
    // Wait for the result.
    auto EC = waitForResult(C1, ResOrErr->second, handleNone);
    EXPECT_FALSE(EC) << "Could not read result.";
  }

  // Verify that the function returned ok.
  auto Val = ResOrErr->first.get();
  EXPECT_TRUE(!!Val) << "Remote int function failed to execute.";
  EXPECT_EQ(*Val, 42) << "Remote int function return wrong value.";
}

TEST_F(DummyRPC, TestSerialization) {
  Queue Q1, Q2;
  QueueChannel C1(Q1, Q2);
  QueueChannel C2(Q2, Q1);

  // Make a call to Proc1.
  std::vector<int> v({42, 7});
  auto ResOrErr = callNBWithSeq<AllTheTypes>(
      C1, -101, 250, -10000, 10000, -1000000000, 1000000000, -10000000000,
      10000000000, true, "foo", v);
  EXPECT_TRUE(!!ResOrErr) << "Big (serialization test) call over queue failed";

  {
    // Expect a call to Proc1.
    auto EC = expect<AllTheTypes>(
        C2, [&](int8_t &s8, uint8_t &u8, int16_t &s16, uint16_t &u16,
                int32_t &s32, uint32_t &u32, int64_t &s64, uint64_t &u64,
                bool &b, std::string &s, std::vector<int> &v) {

          EXPECT_EQ(s8, -101) << "int8_t serialization broken";
          EXPECT_EQ(u8, 250) << "uint8_t serialization broken";
          EXPECT_EQ(s16, -10000) << "int16_t serialization broken";
          EXPECT_EQ(u16, 10000) << "uint16_t serialization broken";
          EXPECT_EQ(s32, -1000000000) << "int32_t serialization broken";
          EXPECT_EQ(u32, 1000000000ULL) << "uint32_t serialization broken";
          EXPECT_EQ(s64, -10000000000) << "int64_t serialization broken";
          EXPECT_EQ(u64, 10000000000ULL) << "uint64_t serialization broken";
          EXPECT_EQ(b, true) << "bool serialization broken";
          EXPECT_EQ(s, "foo") << "std::string serialization broken";
          EXPECT_EQ(v, std::vector<int>({42, 7}))
              << "std::vector serialization broken";
          return Error::success();
        });
    EXPECT_FALSE(EC) << "Big (serialization test) call over queue failed";
  }

  {
    // Wait for the result.
    auto EC = waitForResult(C1, ResOrErr->second, handleNone);
    EXPECT_FALSE(EC) << "Could not read result.";
  }

  // Verify that the function returned ok.
  auto Err = ResOrErr->first.get();
  EXPECT_FALSE(!!Err) << "Remote void function failed to execute.";
}

// Test the synchronous call API.
// FIXME: Re-enable once deadlock encountered on S390 has been debugged / fixed,
//        see http://lab.llvm.org:8011/builders/clang-s390x-linux/builds/3459
// TEST_F(DummyRPC, TestSynchronousCall) {
//   Queue Q1, Q2;
//   QueueChannel C1(Q1, Q2);
//   QueueChannel C2(Q2, Q1);
//
//   auto ServerResult =
//     std::async(std::launch::async,
//       [&]() {
//         return expect<IntInt>(C2, [&](int32_t V) { return V; });
//       });
//
//   auto ValOrErr = callST<IntInt>(C1, 42);
//
//   EXPECT_FALSE(!!ServerResult.get())
//     << "Server returned an error.";
//   EXPECT_TRUE(!!ValOrErr)
//     << "callST returned an error.";
//   EXPECT_EQ(*ValOrErr, 42)
//     << "Incorrect callST<IntInt> result";
// }
