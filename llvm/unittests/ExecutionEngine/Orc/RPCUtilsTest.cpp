//===----------- RPCUtilsTest.cpp - Unit tests the Orc RPC utils ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/RPCUtils.h"
#include "QueueChannel.h"
#include "gtest/gtest.h"

#include <queue>

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::orc::rpc;

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

class DummyError : public ErrorInfo<DummyError> {
public:

  static char ID;

  DummyError(uint32_t Val) : Val(Val) {}

  std::error_code convertToErrorCode() const override {
    // Use a nonsense error code - we want to verify that errors
    // transmitted over the network are replaced with
    // OrcErrorCode::UnknownErrorCodeFromRemote.
    return orcError(OrcErrorCode::RemoteAllocatorDoesNotExist);
  }

  void log(raw_ostream &OS) const override {
    OS << "Dummy error " << Val;
  }

  uint32_t getValue() const { return Val; }

public:
  uint32_t Val;
};

char DummyError::ID = 0;

template <typename ChannelT>
void registerDummyErrorSerialization() {
  static bool AlreadyRegistered = false;
  if (!AlreadyRegistered) {
    SerializationTraits<ChannelT, Error>::
      template registerErrorType<DummyError>(
        "DummyError",
        [](ChannelT &C, const DummyError &DE) {
          return serializeSeq(C, DE.getValue());
        },
        [](ChannelT &C, Error &Err) -> Error {
          ErrorAsOutParameter EAO(&Err);
          uint32_t Val;
          if (auto Err = deserializeSeq(C, Val))
            return Err;
          Err = make_error<DummyError>(Val);
          return Error::success();
        });
    AlreadyRegistered = true;
  }
}

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

  class VoidString : public Function<VoidString, void(std::string)> {
  public:
    static const char* getName() { return "VoidString"; }
  };

  class AllTheTypes
      : public Function<AllTheTypes, void(int8_t, uint8_t, int16_t, uint16_t,
                                          int32_t, uint32_t, int64_t, uint64_t,
                                          bool, std::string, std::vector<int>,
                                          std::set<int>, std::map<int, bool>)> {
  public:
    static const char* getName() { return "AllTheTypes"; }
  };

  class CustomType : public Function<CustomType, RPCFoo(RPCFoo)> {
  public:
    static const char* getName() { return "CustomType"; }
  };

  class ErrorFunc : public Function<ErrorFunc, Error()> {
  public:
    static const char* getName() { return "ErrorFunc"; }
  };

  class ExpectedFunc : public Function<ExpectedFunc, Expected<uint32_t>()> {
  public:
    static const char* getName() { return "ExpectedFunc"; }
  };

}

class DummyRPCEndpoint : public SingleThreadedRPCEndpoint<QueueChannel> {
public:
  DummyRPCEndpoint(QueueChannel &C)
      : SingleThreadedRPCEndpoint(C, true) {}
};


void freeVoidBool(bool B) {
}

TEST(DummyRPC, TestFreeFunctionHandler) {
  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Server(*Channels.first);
  Server.addHandler<DummyRPCAPI::VoidBool>(freeVoidBool);
}

TEST(DummyRPC, TestCallAsyncVoidBool) {
  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

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
  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

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

TEST(DummyRPC, TestAsyncVoidBoolHandler) {
  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

  std::thread ServerThread([&]() {
      Server.addAsyncHandler<DummyRPCAPI::VoidBool>(
          [](std::function<Error(Error)> SendResult,
             bool B) {
            EXPECT_EQ(B, true) << "Server void(bool) receieved unexpected result";
            cantFail(SendResult(Error::success()));
            return Error::success();
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
    auto Err = Client.callAsync<DummyRPCAPI::VoidBool>(
        [](Error Result) {
          EXPECT_FALSE(!!Result) << "Async void(bool) response handler failed";
          return Error::success();
        }, true);
    EXPECT_FALSE(!!Err) << "Client.callAsync failed for void(bool)";
  }

  {
    // Poke the client to process the result.
    auto Err = Client.handleOne();
    EXPECT_FALSE(!!Err) << "Client failed to handle response from void(bool)";
  }

  ServerThread.join();
}

TEST(DummyRPC, TestAsyncIntIntHandler) {
  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

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

TEST(DummyRPC, TestAsyncIntIntHandlerMethod) {
  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

  class Dummy {
  public:
    Error handler(std::function<Error(Expected<int32_t>)> SendResult,
             int32_t X) {
      EXPECT_EQ(X, 21) << "Server int(int) receieved unexpected result";
      return SendResult(2 * X);
    }
  };

  std::thread ServerThread([&]() {
      Dummy D;
      Server.addAsyncHandler<DummyRPCAPI::IntInt>(D, &Dummy::handler);

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

TEST(DummyRPC, TestCallAsyncVoidString) {
  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

  std::thread ServerThread([&]() {
      Server.addHandler<DummyRPCAPI::VoidString>(
          [](const std::string &S) {
            EXPECT_EQ(S, "hello")
              << "Server void(std::string) received unexpected result";
          });

      // Poke the server to handle the negotiate call.
      for (int I = 0; I < 4; ++I) {
        auto Err = Server.handleOne();
        EXPECT_FALSE(!!Err) << "Server failed to handle call";
      }
  });

  {
    // Make an call using a std::string.
    auto Err = Client.callB<DummyRPCAPI::VoidString>(std::string("hello"));
    EXPECT_FALSE(!!Err) << "Client.callAsync failed for void(std::string)";
  }

  {
    // Make an call using a std::string.
    auto Err = Client.callB<DummyRPCAPI::VoidString>(StringRef("hello"));
    EXPECT_FALSE(!!Err) << "Client.callAsync failed for void(std::string)";
  }

  {
    // Make an call using a std::string.
    auto Err = Client.callB<DummyRPCAPI::VoidString>("hello");
    EXPECT_FALSE(!!Err) << "Client.callAsync failed for void(string)";
  }

  ServerThread.join();
}

TEST(DummyRPC, TestSerialization) {
  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

  std::thread ServerThread([&]() {
    Server.addHandler<DummyRPCAPI::AllTheTypes>([&](int8_t S8, uint8_t U8,
                                                    int16_t S16, uint16_t U16,
                                                    int32_t S32, uint32_t U32,
                                                    int64_t S64, uint64_t U64,
                                                    bool B, std::string S,
                                                    std::vector<int> V,
                                                    std::set<int> S2,
                                                    std::map<int, bool> M) {
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
      EXPECT_EQ(S2, std::set<int>({7, 42})) << "std::set serialization broken";
      EXPECT_EQ(M, (std::map<int, bool>({{7, false}, {42, true}})))
          << "std::map serialization broken";
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
    std::vector<int> V({42, 7});
    std::set<int> S({7, 42});
    std::map<int, bool> M({{7, false}, {42, true}});
    auto Err = Client.callAsync<DummyRPCAPI::AllTheTypes>(
        [](Error Err) {
          EXPECT_FALSE(!!Err) << "Async AllTheTypes response handler failed";
          return Error::success();
        },
        static_cast<int8_t>(-101), static_cast<uint8_t>(250),
        static_cast<int16_t>(-10000), static_cast<uint16_t>(10000),
        static_cast<int32_t>(-1000000000), static_cast<uint32_t>(1000000000),
        static_cast<int64_t>(-10000000000), static_cast<uint64_t>(10000000000),
        true, std::string("foo"), V, S, M);
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
  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

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
  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

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

TEST(DummyRPC, ReturnErrorSuccess) {
  registerDummyErrorSerialization<QueueChannel>();

  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

  std::thread ServerThread([&]() {
      Server.addHandler<DummyRPCAPI::ErrorFunc>(
        []() {
          return Error::success();
        });

      // Handle the negotiate plus one call.
      for (unsigned I = 0; I != 2; ++I)
        cantFail(Server.handleOne());
    });

  cantFail(Client.callAsync<DummyRPCAPI::ErrorFunc>(
             [&](Error Err) {
               EXPECT_FALSE(!!Err) << "Expected success value";
               return Error::success();
             }));

  cantFail(Client.handleOne());

  ServerThread.join();
}

TEST(DummyRPC, ReturnErrorFailure) {
  registerDummyErrorSerialization<QueueChannel>();

  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

  std::thread ServerThread([&]() {
      Server.addHandler<DummyRPCAPI::ErrorFunc>(
        []() {
          return make_error<DummyError>(42);
        });

      // Handle the negotiate plus one call.
      for (unsigned I = 0; I != 2; ++I)
        cantFail(Server.handleOne());
    });

  cantFail(Client.callAsync<DummyRPCAPI::ErrorFunc>(
             [&](Error Err) {
               EXPECT_TRUE(Err.isA<DummyError>())
                 << "Incorrect error type";
               return handleErrors(
                        std::move(Err),
                        [](const DummyError &DE) {
                          EXPECT_EQ(DE.getValue(), 42ULL)
                            << "Incorrect DummyError serialization";
                        });
             }));

  cantFail(Client.handleOne());

  ServerThread.join();
}

TEST(DummyRPC, ReturnExpectedSuccess) {
  registerDummyErrorSerialization<QueueChannel>();

  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

  std::thread ServerThread([&]() {
      Server.addHandler<DummyRPCAPI::ExpectedFunc>(
        []() -> uint32_t {
          return 42;
        });

      // Handle the negotiate plus one call.
      for (unsigned I = 0; I != 2; ++I)
        cantFail(Server.handleOne());
    });

  cantFail(Client.callAsync<DummyRPCAPI::ExpectedFunc>(
               [&](Expected<uint32_t> ValOrErr) {
                 EXPECT_TRUE(!!ValOrErr)
                   << "Expected success value";
                 EXPECT_EQ(*ValOrErr, 42ULL)
                   << "Incorrect Expected<uint32_t> deserialization";
                 return Error::success();
               }));

  cantFail(Client.handleOne());

  ServerThread.join();
}

TEST(DummyRPC, ReturnExpectedFailure) {
  registerDummyErrorSerialization<QueueChannel>();

  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

  std::thread ServerThread([&]() {
      Server.addHandler<DummyRPCAPI::ExpectedFunc>(
        []() -> Expected<uint32_t> {
          return make_error<DummyError>(7);
        });

      // Handle the negotiate plus one call.
      for (unsigned I = 0; I != 2; ++I)
        cantFail(Server.handleOne());
    });

  cantFail(Client.callAsync<DummyRPCAPI::ExpectedFunc>(
               [&](Expected<uint32_t> ValOrErr) {
                 EXPECT_FALSE(!!ValOrErr)
                   << "Expected failure value";
                 auto Err = ValOrErr.takeError();
                 EXPECT_TRUE(Err.isA<DummyError>())
                   << "Incorrect error type";
                 return handleErrors(
                          std::move(Err),
                          [](const DummyError &DE) {
                            EXPECT_EQ(DE.getValue(), 7ULL)
                              << "Incorrect DummyError serialization";
                          });
               }));

  cantFail(Client.handleOne());

  ServerThread.join();
}

TEST(DummyRPC, TestParallelCallGroup) {
  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

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

  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Client(*Channels.first);
  DummyRPCEndpoint Server(*Channels.second);

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
    EXPECT_TRUE(Err.isA<CouldNotNegotiate>())
      << "Expected CouldNotNegotiate error for attempted negotiate of "
         "unsupported function";
    consumeError(std::move(Err));
  }

  ServerThread.join();
}

TEST(DummyRPC, TestRemoveHandler) {
  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Server(*Channels.second);

  Server.addHandler<DummyRPCAPI::VoidBool>(
    [](bool B) {
      EXPECT_EQ(B, true)
        << "Server void(bool) received unexpected result";
    });

  Server.removeHandler<DummyRPCAPI::VoidBool>();
}

TEST(DummyRPC, TestClearHandlers) {
  auto Channels = createPairedQueueChannels();
  DummyRPCEndpoint Server(*Channels.second);

  Server.addHandler<DummyRPCAPI::VoidBool>(
    [](bool B) {
      EXPECT_EQ(B, true)
        << "Server void(bool) received unexpected result";
    });

  Server.clearHandlers();
}
