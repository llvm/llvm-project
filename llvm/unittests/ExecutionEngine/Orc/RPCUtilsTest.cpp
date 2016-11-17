//===----------- RPCUtilsTest.cpp - Unit tests the Orc RPC utils ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/RawByteChannel.h"
#include "llvm/ExecutionEngine/Orc/RPCUtils.h"
#include "gtest/gtest.h"

#include <queue>

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::rpc;

class Queue : public std::queue<char> {
public:
  std::mutex &getMutex() { return M; }
  std::condition_variable &getCondVar() { return CV; }
private:
  std::mutex M;
  std::condition_variable CV;
};

class QueueChannel : public RawByteChannel {
public:
  QueueChannel(Queue &InQueue, Queue &OutQueue)
      : InQueue(InQueue), OutQueue(OutQueue) {}

  Error readBytes(char *Dst, unsigned Size) override {
    std::unique_lock<std::mutex> Lock(InQueue.getMutex());
    while (Size) {
      while (InQueue.empty())
        InQueue.getCondVar().wait(Lock);
      *Dst++ = InQueue.front();
      --Size;
      InQueue.pop();
    }
    return Error::success();
  }

  Error appendBytes(const char *Src, unsigned Size) override {
    std::unique_lock<std::mutex> Lock(OutQueue.getMutex());
    while (Size--)
      OutQueue.push(*Src++);
    OutQueue.getCondVar().notify_one();
    return Error::success();
  }

  Error send() override { return Error::success(); }

private:
  Queue &InQueue;
  Queue &OutQueue;
};

class DummyRPCAPI {
public:

  class VoidBool : public Function<VoidBool, void(bool)> {
  public:
    static const char* getName() { return "VoidBool"; }
  };

  class IntInt : public Function<IntInt, int32_t(int32_t)> {
  public:
    static const char* getName() { return "IntInt"; }
  };

  class AllTheTypes
    : public Function<AllTheTypes,
                      void(int8_t, uint8_t, int16_t, uint16_t, int32_t,
                           uint32_t, int64_t, uint64_t, bool, std::string,
                           std::vector<int>)> {
  public:
    static const char* getName() { return "AllTheTypes"; }
  };
};

class DummyRPCEndpoint : public DummyRPCAPI,
                         public SingleThreadedRPC<QueueChannel> {
public:
  DummyRPCEndpoint(Queue &Q1, Queue &Q2)
      : SingleThreadedRPC(C, true), C(Q1, Q2) {}
private:
  QueueChannel C;
};


TEST(DummyRPC, TestAsyncVoidBool) {
  Queue Q1, Q2;
  DummyRPCEndpoint Client(Q1, Q2);
  DummyRPCEndpoint Server(Q2, Q1);

  std::thread ServerThread([&]() {
      Server.addHandler<DummyRPCAPI::VoidBool>(
          [](bool B) {
            EXPECT_EQ(B, true)
              << "Server void(bool) received unexpected result";
          });

      {
        // Poke the server to handle the negotiate call.
        auto Err = Server.handleOne();
        EXPECT_FALSE(!!Err) << "Server failed to handle call to negotiate";
      }

      {
        // Poke the server to handle the VoidBool call.
        auto Err = Server.handleOne();
        EXPECT_FALSE(!!Err) << "Server failed to handle call to void(bool)";
      }
  });

  {
    // Make an async call.
    auto Err = Client.callAsync<DummyRPCAPI::VoidBool>(
        [](Error Err) {
          EXPECT_FALSE(!!Err) << "Async void(bool) response handler failed";
          return Error::success();
        }, true);
    EXPECT_FALSE(!!Err) << "Client.callAsync failed for void(bool)";
  }

  {
    // Poke the client to process the result of the void(bool) call.
    auto Err = Client.handleOne();
    EXPECT_FALSE(!!Err) << "Client failed to handle response from void(bool)";
  }

  ServerThread.join();
}

TEST(DummyRPC, TestAsyncIntInt) {
  Queue Q1, Q2;
  DummyRPCEndpoint Client(Q1, Q2);
  DummyRPCEndpoint Server(Q2, Q1);

  std::thread ServerThread([&]() {
      Server.addHandler<DummyRPCAPI::IntInt>(
          [](int X) -> int {
            EXPECT_EQ(X, 21) << "Server int(int) receieved unexpected result";
            return 2 * X;
          });

      {
        // Poke the server to handle the negotiate call.
        auto Err = Server.handleOne();
        EXPECT_FALSE(!!Err) << "Server failed to handle call to negotiate";
      }

      {
        // Poke the server to handle the int(int) call.
        auto Err = Server.handleOne();
        EXPECT_FALSE(!!Err) << "Server failed to handle call to int(int)";
      }
    });

  {
    auto Err = Client.callAsync<DummyRPCAPI::IntInt>(
        [](Expected<int> Result) {
          EXPECT_TRUE(!!Result) << "Async int(int) response handler failed";
          EXPECT_EQ(*Result, 42)
            << "Async int(int) response handler received incorrect result";
          return Error::success();
        }, 21);
    EXPECT_FALSE(!!Err) << "Client.callAsync failed for int(int)";
  }

  {
    // Poke the client to process the result.
    auto Err = Client.handleOne();
    EXPECT_FALSE(!!Err) << "Client failed to handle response from void(bool)";
  }

  ServerThread.join();
}

TEST(DummyRPC, TestSerialization) {
  Queue Q1, Q2;
  DummyRPCEndpoint Client(Q1, Q2);
  DummyRPCEndpoint Server(Q2, Q1);

  std::thread ServerThread([&]() {
      Server.addHandler<DummyRPCAPI::AllTheTypes>(
          [&](int8_t S8, uint8_t U8, int16_t S16, uint16_t U16,
              int32_t S32, uint32_t U32, int64_t S64, uint64_t U64,
              bool B, std::string S, std::vector<int> V) {

            EXPECT_EQ(S8, -101) << "int8_t serialization broken";
            EXPECT_EQ(U8, 250) << "uint8_t serialization broken";
            EXPECT_EQ(S16, -10000) << "int16_t serialization broken";
            EXPECT_EQ(U16, 10000) << "uint16_t serialization broken";
            EXPECT_EQ(S32, -1000000000) << "int32_t serialization broken";
            EXPECT_EQ(U32, 1000000000ULL) << "uint32_t serialization broken";
            EXPECT_EQ(S64, -10000000000) << "int64_t serialization broken";
            EXPECT_EQ(U64, 10000000000ULL) << "uint64_t serialization broken";
            EXPECT_EQ(B, true) << "bool serialization broken";
            EXPECT_EQ(S, "foo") << "std::string serialization broken";
            EXPECT_EQ(V, std::vector<int>({42, 7}))
              << "std::vector serialization broken";
            return Error::success();
          });

      {
        // Poke the server to handle the negotiate call.
        auto Err = Server.handleOne();
        EXPECT_FALSE(!!Err) << "Server failed to handle call to negotiate";
      }

      {
        // Poke the server to handle the AllTheTypes call.
        auto Err = Server.handleOne();
        EXPECT_FALSE(!!Err) << "Server failed to handle call to void(bool)";
      }
    });


  {
    // Make an async call.
    std::vector<int> v({42, 7});
    auto Err = Client.callAsync<DummyRPCAPI::AllTheTypes>(
        [](Error Err) {
          EXPECT_FALSE(!!Err) << "Async AllTheTypes response handler failed";
          return Error::success();
        },
        static_cast<int8_t>(-101), static_cast<uint8_t>(250),
        static_cast<int16_t>(-10000), static_cast<uint16_t>(10000),
        static_cast<int32_t>(-1000000000), static_cast<uint32_t>(1000000000),
        static_cast<int64_t>(-10000000000), static_cast<uint64_t>(10000000000),
        true, std::string("foo"), v);
    EXPECT_FALSE(!!Err) << "Client.callAsync failed for AllTheTypes";
  }

  {
    // Poke the client to process the result of the AllTheTypes call.
    auto Err = Client.handleOne();
    EXPECT_FALSE(!!Err) << "Client failed to handle response from AllTheTypes";
  }

  ServerThread.join();
}

// Test the synchronous call API.
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
