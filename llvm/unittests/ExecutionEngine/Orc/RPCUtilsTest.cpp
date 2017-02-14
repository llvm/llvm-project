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

class RPCFoo {};

namespace llvm {
namespace orc {
namespace rpc {

  template <>
  class RPCTypeName<RPCFoo> {
  public:
    static const char* getName() { return "RPCFoo"; }
  };

  template <>
  class SerializationTraits<QueueChannel, RPCFoo, RPCFoo> {
  public:
    static Error serialize(QueueChannel&, const RPCFoo&) {
      return Error::success();
    }

    static Error deserialize(QueueChannel&, RPCFoo&) {
      return Error::success();
    }
  };

} // end namespace rpc
} // end namespace orc
} // end namespace llvm

class RPCBar {};

namespace llvm {
namespace orc {
namespace rpc {

  template <>
  class SerializationTraits<QueueChannel, RPCFoo, RPCBar> {
  public:
    static Error serialize(QueueChannel&, const RPCBar&) {
      return Error::success();
    }

    static Error deserialize(QueueChannel&, RPCBar&) {
      return Error::success();
    }
};

} // end namespace rpc
} // end namespace orc
} // end namespace llvm

namespace DummyRPCAPI {

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

  class CustomType : public Function<CustomType, RPCFoo(RPCFoo)> {
  public:
    static const char* getName() { return "CustomType"; }
  };

}

class DummyRPCEndpoint : public SingleThreadedRPCEndpoint<QueueChannel> {
public:
  DummyRPCEndpoint(Queue &Q1, Queue &Q2)
      : SingleThreadedRPCEndpoint(C, true), C(Q1, Q2) {}
private:
  QueueChannel C;
};


void freeVoidBool(bool B) {
}

TEST(DummyRPC, TestFreeFunctionHandler) {
  Queue Q1, Q2;
  DummyRPCEndpoint Server(Q2, Q1);
  Server.addHandler<DummyRPCAPI::VoidBool>(freeVoidBool);
}

TEST(DummyRPC, TestCallAsyncVoidBool) {
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

TEST(DummyRPC, TestCallAsyncIntInt) {
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

TEST(DummyRPC, TestAsyncIntIntHandler) {
  Queue Q1, Q2;
  DummyRPCEndpoint Client(Q1, Q2);
  DummyRPCEndpoint Server(Q2, Q1);

  std::thread ServerThread([&]() {
      Server.addAsyncHandler<DummyRPCAPI::IntInt>(
          [](std::function<Error(Expected<int32_t>)> SendResult,
             int32_t X) {
            EXPECT_EQ(X, 21) << "Server int(int) receieved unexpected result";
            return SendResult(2 * X);
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

TEST(DummyRPC, TestCustomType) {
  Queue Q1, Q2;
  DummyRPCEndpoint Client(Q1, Q2);
  DummyRPCEndpoint Server(Q2, Q1);

  std::thread ServerThread([&]() {
      Server.addHandler<DummyRPCAPI::CustomType>(
          [](RPCFoo F) {});

      {
        // Poke the server to handle the negotiate call.
        auto Err = Server.handleOne();
        EXPECT_FALSE(!!Err) << "Server failed to handle call to negotiate";
      }

      {
        // Poke the server to handle the CustomType call.
        auto Err = Server.handleOne();
        EXPECT_FALSE(!!Err) << "Server failed to handle call to RPCFoo(RPCFoo)";
      }
  });

  {
    // Make an async call.
    auto Err = Client.callAsync<DummyRPCAPI::CustomType>(
        [](Expected<RPCFoo> FOrErr) {
          EXPECT_TRUE(!!FOrErr)
            << "Async RPCFoo(RPCFoo) response handler failed";
          return Error::success();
        }, RPCFoo());
    EXPECT_FALSE(!!Err) << "Client.callAsync failed for RPCFoo(RPCFoo)";
  }

  {
    // Poke the client to process the result of the RPCFoo() call.
    auto Err = Client.handleOne();
    EXPECT_FALSE(!!Err)
      << "Client failed to handle response from RPCFoo(RPCFoo)";
  }

  ServerThread.join();
}

TEST(DummyRPC, TestWithAltCustomType) {
  Queue Q1, Q2;
  DummyRPCEndpoint Client(Q1, Q2);
  DummyRPCEndpoint Server(Q2, Q1);

  std::thread ServerThread([&]() {
      Server.addHandler<DummyRPCAPI::CustomType>(
          [](RPCBar F) {});

      {
        // Poke the server to handle the negotiate call.
        auto Err = Server.handleOne();
        EXPECT_FALSE(!!Err) << "Server failed to handle call to negotiate";
      }

      {
        // Poke the server to handle the CustomType call.
        auto Err = Server.handleOne();
        EXPECT_FALSE(!!Err) << "Server failed to handle call to RPCFoo(RPCFoo)";
      }
  });

  {
    // Make an async call.
    auto Err = Client.callAsync<DummyRPCAPI::CustomType>(
        [](Expected<RPCBar> FOrErr) {
          EXPECT_TRUE(!!FOrErr)
            << "Async RPCFoo(RPCFoo) response handler failed";
          return Error::success();
        }, RPCBar());
    EXPECT_FALSE(!!Err) << "Client.callAsync failed for RPCFoo(RPCFoo)";
  }

  {
    // Poke the client to process the result of the RPCFoo() call.
    auto Err = Client.handleOne();
    EXPECT_FALSE(!!Err)
      << "Client failed to handle response from RPCFoo(RPCFoo)";
  }

  ServerThread.join();
}

TEST(DummyRPC, TestParallelCallGroup) {
  Queue Q1, Q2;
  DummyRPCEndpoint Client(Q1, Q2);
  DummyRPCEndpoint Server(Q2, Q1);

  std::thread ServerThread([&]() {
      Server.addHandler<DummyRPCAPI::IntInt>(
          [](int X) -> int {
            return 2 * X;
          });

      // Handle the negotiate, plus three calls.
      for (unsigned I = 0; I != 4; ++I) {
        auto Err = Server.handleOne();
        EXPECT_FALSE(!!Err) << "Server failed to handle call to int(int)";
      }
    });

  {
    int A, B, C;
    ParallelCallGroup PCG;

    {
      auto Err = PCG.call(
        rpcAsyncDispatch<DummyRPCAPI::IntInt>(Client),
        [&A](Expected<int> Result) {
          EXPECT_TRUE(!!Result) << "Async int(int) response handler failed";
          A = *Result;
          return Error::success();
        }, 1);
      EXPECT_FALSE(!!Err) << "First parallel call failed for int(int)";
    }

    {
      auto Err = PCG.call(
        rpcAsyncDispatch<DummyRPCAPI::IntInt>(Client),
        [&B](Expected<int> Result) {
          EXPECT_TRUE(!!Result) << "Async int(int) response handler failed";
          B = *Result;
          return Error::success();
        }, 2);
      EXPECT_FALSE(!!Err) << "Second parallel call failed for int(int)";
    }

    {
      auto Err = PCG.call(
        rpcAsyncDispatch<DummyRPCAPI::IntInt>(Client),
        [&C](Expected<int> Result) {
          EXPECT_TRUE(!!Result) << "Async int(int) response handler failed";
          C = *Result;
          return Error::success();
        }, 3);
      EXPECT_FALSE(!!Err) << "Third parallel call failed for int(int)";
    }

    // Handle the three int(int) results.
    for (unsigned I = 0; I != 3; ++I) {
      auto Err = Client.handleOne();
      EXPECT_FALSE(!!Err) << "Client failed to handle response from void(bool)";
    }

    PCG.wait();

    EXPECT_EQ(A, 2) << "First parallel call returned bogus result";
    EXPECT_EQ(B, 4) << "Second parallel call returned bogus result";
    EXPECT_EQ(C, 6) << "Third parallel call returned bogus result";
  }

  ServerThread.join();
}

TEST(DummyRPC, TestAPICalls) {

  using DummyCalls1 = APICalls<DummyRPCAPI::VoidBool, DummyRPCAPI::IntInt>;
  using DummyCalls2 = APICalls<DummyRPCAPI::AllTheTypes>;
  using DummyCalls3 = APICalls<DummyCalls1, DummyRPCAPI::CustomType>;
  using DummyCallsAll = APICalls<DummyCalls1, DummyCalls2, DummyRPCAPI::CustomType>;

  static_assert(DummyCalls1::Contains<DummyRPCAPI::VoidBool>::value,
                "Contains<Func> template should return true here");
  static_assert(!DummyCalls1::Contains<DummyRPCAPI::CustomType>::value,
                "Contains<Func> template should return false here");

  Queue Q1, Q2;
  DummyRPCEndpoint Client(Q1, Q2);
  DummyRPCEndpoint Server(Q2, Q1);

  std::thread ServerThread(
    [&]() {
      Server.addHandler<DummyRPCAPI::VoidBool>([](bool b) { });
      Server.addHandler<DummyRPCAPI::IntInt>([](int x) { return x; });
      Server.addHandler<DummyRPCAPI::CustomType>([](RPCFoo F) {});

      for (unsigned I = 0; I < 4; ++I) {
        auto Err = Server.handleOne();
        (void)!!Err;
      }
    });

  {
    auto Err = DummyCalls1::negotiate(Client);
    EXPECT_FALSE(!!Err) << "DummyCalls1::negotiate failed";
  }

  {
    auto Err = DummyCalls3::negotiate(Client);
    EXPECT_FALSE(!!Err) << "DummyCalls3::negotiate failed";
  }

  {
    auto Err = DummyCallsAll::negotiate(Client);
    EXPECT_EQ(errorToErrorCode(std::move(Err)).value(),
              static_cast<int>(OrcErrorCode::UnknownRPCFunction))
      << "Expected 'UnknownRPCFunction' error for attempted negotiate of "
         "unsupported function";
  }

  ServerThread.join();
}

TEST(DummyRPC, TestRemoveHandler) {
  Queue Q1, Q2;
  DummyRPCEndpoint Server(Q1, Q2);

  Server.addHandler<DummyRPCAPI::VoidBool>(
    [](bool B) {
      EXPECT_EQ(B, true)
        << "Server void(bool) received unexpected result";
    });

  Server.removeHandler<DummyRPCAPI::VoidBool>();
}

TEST(DummyRPC, TestClearHandlers) {
  Queue Q1, Q2;
  DummyRPCEndpoint Server(Q1, Q2);

  Server.addHandler<DummyRPCAPI::VoidBool>(
    [](bool B) {
      EXPECT_EQ(B, true)
        << "Server void(bool) received unexpected result";
    });

  Server.clearHandlers();
}
