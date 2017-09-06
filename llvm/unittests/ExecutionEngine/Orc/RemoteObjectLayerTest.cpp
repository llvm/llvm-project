//===---------------------- RemoteObjectLayerTest.cpp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/NullResolver.h"
#include "llvm/ExecutionEngine/Orc/RemoteObjectLayer.h"
#include "OrcTestCommon.h"
#include "QueueChannel.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

class MockObjectLayer {
public:

  using ObjHandleT = uint64_t;

  using ObjectPtr =
    std::shared_ptr<object::OwningBinary<object::ObjectFile>>;

  using LookupFn = std::function<JITSymbol(StringRef, bool)>;
  using SymbolLookupTable = std::map<ObjHandleT, LookupFn>;

  using AddObjectFtor =
    std::function<Expected<ObjHandleT>(ObjectPtr, SymbolLookupTable&)>;

  class ObjectNotFound : public remote::ResourceNotFound<ObjHandleT> {
  public:
    ObjectNotFound(ObjHandleT H) : ResourceNotFound(H, "Object handle") {}
  };

  MockObjectLayer(AddObjectFtor AddObject)
    : AddObject(std::move(AddObject)) {}

  Expected<ObjHandleT> addObject(ObjectPtr Obj,
            std::shared_ptr<JITSymbolResolver> Resolver) {
    return AddObject(Obj, SymTab);
  }

  Error removeObject(ObjHandleT H) {
    if (SymTab.count(H))
      return Error::success();
    else
      return make_error<ObjectNotFound>(H);
  }

  JITSymbol findSymbol(StringRef Name, bool ExportedSymbolsOnly) {
    for (auto KV : SymTab) {
      if (auto Sym = KV.second(Name, ExportedSymbolsOnly))
        return Sym;
      else if (auto Err = Sym.takeError())
        return std::move(Err);
    }
    return JITSymbol(nullptr);
  }

  JITSymbol findSymbolIn(ObjHandleT H, StringRef Name,
                         bool ExportedSymbolsOnly) {
    auto LI = SymTab.find(H);
    if (LI != SymTab.end())
      return LI->second(Name, ExportedSymbolsOnly);
    else
      return make_error<ObjectNotFound>(H);
  }

  Error emitAndFinalize(ObjHandleT H) {
    if (SymTab.count(H))
      return Error::success();
    else
      return make_error<ObjectNotFound>(H);
  }

private:
  AddObjectFtor AddObject;
  SymbolLookupTable SymTab;
};

using RPCEndpoint = rpc::SingleThreadedRPCEndpoint<rpc::RawByteChannel>;

MockObjectLayer::ObjectPtr createTestObject() {
  OrcNativeTarget::initialize();
  auto TM = std::unique_ptr<TargetMachine>(EngineBuilder().selectTarget());

  if (!TM)
    return nullptr;

  LLVMContext Ctx;
  ModuleBuilder MB(Ctx, TM->getTargetTriple().str(), "TestModule");
  MB.getModule()->setDataLayout(TM->createDataLayout());
  auto *Main = MB.createFunctionDecl<void(int, char**)>("main");
  Main->getBasicBlockList().push_back(BasicBlock::Create(Ctx));
  IRBuilder<> B(&Main->back());
  B.CreateRet(ConstantInt::getSigned(Type::getInt32Ty(Ctx), 42));

  SimpleCompiler IRCompiler(*TM);
  return std::make_shared<object::OwningBinary<object::ObjectFile>>(
           IRCompiler(*MB.getModule()));
}

TEST(RemoteObjectLayer, AddObject) {
  llvm::orc::rpc::registerStringError<rpc::RawByteChannel>();
  auto TestObject = createTestObject();
  if (!TestObject)
    return;

  auto Channels = createPairedQueueChannels();

  auto ReportError =
    [](Error Err) {
      logAllUnhandledErrors(std::move(Err), llvm::errs(), "");
    };

  // Copy the bytes out of the test object: the copy will be used to verify
  // that the original is correctly transmitted over RPC to the mock layer.
  StringRef ObjBytes = TestObject->getBinary()->getData();
  std::vector<char> ObjContents(ObjBytes.size());
  std::copy(ObjBytes.begin(), ObjBytes.end(), ObjContents.begin());

  RPCEndpoint ClientEP(*Channels.first, true);
  RemoteObjectClientLayer<RPCEndpoint> Client(ClientEP, ReportError);

  RPCEndpoint ServerEP(*Channels.second, true);
  MockObjectLayer BaseLayer(
    [&ObjContents](MockObjectLayer::ObjectPtr Obj,
                   MockObjectLayer::SymbolLookupTable &SymTab) {

      // Check that the received object file content matches the original.
      StringRef RPCObjContents = Obj->getBinary()->getData();
      EXPECT_EQ(RPCObjContents.size(), ObjContents.size())
        << "RPC'd object file has incorrect size";
      EXPECT_TRUE(std::equal(RPCObjContents.begin(), RPCObjContents.end(),
                             ObjContents.begin()))
        << "RPC'd object file content does not match original content";

      return 1;
    });
  RemoteObjectServerLayer<MockObjectLayer, RPCEndpoint> Server(BaseLayer,
                                                               ServerEP,
                                                               ReportError);

  bool Finished = false;
  ServerEP.addHandler<remote::utils::TerminateSession>(
    [&]() { Finished = true; }
  );

  auto ServerThread =
    std::thread([&]() {
      while (!Finished)
        cantFail(ServerEP.handleOne());
    });

  cantFail(Client.addObject(std::move(TestObject),
                            std::make_shared<NullResolver>()));
  cantFail(ClientEP.callB<remote::utils::TerminateSession>());
  ServerThread.join();
}

TEST(RemoteObjectLayer, AddObjectFailure) {
  llvm::orc::rpc::registerStringError<rpc::RawByteChannel>();
  auto TestObject = createTestObject();
  if (!TestObject)
    return;

  auto Channels = createPairedQueueChannels();

  auto ReportError =
    [](Error Err) {
      auto ErrMsg = toString(std::move(Err));
      EXPECT_EQ(ErrMsg, "AddObjectFailure - Test Message")
        << "Expected error string to be \"AddObjectFailure - Test Message\"";
    };

  RPCEndpoint ClientEP(*Channels.first, true);
  RemoteObjectClientLayer<RPCEndpoint> Client(ClientEP, ReportError);

  RPCEndpoint ServerEP(*Channels.second, true);
  MockObjectLayer BaseLayer(
    [](MockObjectLayer::ObjectPtr Obj,
       MockObjectLayer::SymbolLookupTable &SymTab)
        -> Expected<MockObjectLayer::ObjHandleT> {
      return make_error<StringError>("AddObjectFailure - Test Message",
                                     inconvertibleErrorCode());
    });
  RemoteObjectServerLayer<MockObjectLayer, RPCEndpoint> Server(BaseLayer,
                                                               ServerEP,
                                                               ReportError);

  bool Finished = false;
  ServerEP.addHandler<remote::utils::TerminateSession>(
    [&]() { Finished = true; }
  );

  auto ServerThread =
    std::thread([&]() {
      while (!Finished)
        cantFail(ServerEP.handleOne());
    });

  auto HandleOrErr =
    Client.addObject(std::move(TestObject), std::make_shared<NullResolver>());

  EXPECT_FALSE(HandleOrErr) << "Expected error from addObject";

  auto ErrMsg = toString(HandleOrErr.takeError());
  EXPECT_EQ(ErrMsg, "AddObjectFailure - Test Message")
    << "Expected error string to be \"AddObjectFailure - Test Message\"";

  cantFail(ClientEP.callB<remote::utils::TerminateSession>());
  ServerThread.join();
}


TEST(RemoteObjectLayer, RemoveObject) {
  llvm::orc::rpc::registerStringError<rpc::RawByteChannel>();
  auto TestObject = createTestObject();
  if (!TestObject)
    return;

  auto Channels = createPairedQueueChannels();

  auto ReportError =
    [](Error Err) {
      logAllUnhandledErrors(std::move(Err), llvm::errs(), "");
    };

  RPCEndpoint ClientEP(*Channels.first, true);
  RemoteObjectClientLayer<RPCEndpoint> Client(ClientEP, ReportError);

  RPCEndpoint ServerEP(*Channels.second, true);

  MockObjectLayer BaseLayer(
    [](MockObjectLayer::ObjectPtr Obj,
       MockObjectLayer::SymbolLookupTable &SymTab) {
      SymTab[1] = MockObjectLayer::LookupFn();
      return 1;
    });
  RemoteObjectServerLayer<MockObjectLayer, RPCEndpoint> Server(BaseLayer,
                                                               ServerEP,
                                                               ReportError);

  bool Finished = false;
  ServerEP.addHandler<remote::utils::TerminateSession>(
    [&]() { Finished = true; }
  );

  auto ServerThread =
    std::thread([&]() {
      while (!Finished)
        cantFail(ServerEP.handleOne());
    });

  auto H  = cantFail(Client.addObject(std::move(TestObject),
                                      std::make_shared<NullResolver>()));

  cantFail(Client.removeObject(H));

  cantFail(ClientEP.callB<remote::utils::TerminateSession>());
  ServerThread.join();
}

TEST(RemoteObjectLayer, RemoveObjectFailure) {
  llvm::orc::rpc::registerStringError<rpc::RawByteChannel>();
  auto TestObject = createTestObject();
  if (!TestObject)
    return;

  auto Channels = createPairedQueueChannels();

  auto ReportError =
    [](Error Err) {
      auto ErrMsg = toString(std::move(Err));
      EXPECT_EQ(ErrMsg, "Object handle 42 not found")
        << "Expected error string to be \"Object handle 42 not found\"";
    };

  RPCEndpoint ClientEP(*Channels.first, true);
  RemoteObjectClientLayer<RPCEndpoint> Client(ClientEP, ReportError);

  RPCEndpoint ServerEP(*Channels.second, true);

  // AddObject lambda does not update symbol table, so removeObject will treat
  // this as a bad object handle.
  MockObjectLayer BaseLayer(
    [](MockObjectLayer::ObjectPtr Obj,
       MockObjectLayer::SymbolLookupTable &SymTab) {
      return 42;
    });
  RemoteObjectServerLayer<MockObjectLayer, RPCEndpoint> Server(BaseLayer,
                                                               ServerEP,
                                                               ReportError);

  bool Finished = false;
  ServerEP.addHandler<remote::utils::TerminateSession>(
    [&]() { Finished = true; }
  );

  auto ServerThread =
    std::thread([&]() {
      while (!Finished)
        cantFail(ServerEP.handleOne());
    });

  auto H  = cantFail(Client.addObject(std::move(TestObject),
                                      std::make_shared<NullResolver>()));

  auto Err = Client.removeObject(H);
  EXPECT_TRUE(!!Err) << "Expected error from removeObject";

  auto ErrMsg = toString(std::move(Err));
  EXPECT_EQ(ErrMsg, "Object handle 42 not found")
    << "Expected error string to be \"Object handle 42 not found\"";

  cantFail(ClientEP.callB<remote::utils::TerminateSession>());
  ServerThread.join();
}

TEST(RemoteObjectLayer, FindSymbol) {
  llvm::orc::rpc::registerStringError<rpc::RawByteChannel>();
  auto TestObject = createTestObject();
  if (!TestObject)
    return;

  auto Channels = createPairedQueueChannels();

  auto ReportError =
    [](Error Err) {
      auto ErrMsg = toString(std::move(Err));
      EXPECT_EQ(ErrMsg, "Could not find symbol 'badsymbol'")
        << "Expected error string to be \"Object handle 42 not found\"";
    };

  RPCEndpoint ClientEP(*Channels.first, true);
  RemoteObjectClientLayer<RPCEndpoint> Client(ClientEP, ReportError);

  RPCEndpoint ServerEP(*Channels.second, true);

  // AddObject lambda does not update symbol table, so removeObject will treat
  // this as a bad object handle.
  MockObjectLayer BaseLayer(
    [](MockObjectLayer::ObjectPtr Obj,
       MockObjectLayer::SymbolLookupTable &SymTab) {
      SymTab[42] =
        [](StringRef Name, bool ExportedSymbolsOnly) -> JITSymbol {
          if (Name == "foobar")
            return JITSymbol(0x12348765, JITSymbolFlags::Exported);
          if (Name == "badsymbol")
            return make_error<JITSymbolNotFound>(Name);
          return nullptr;
        };
      return 42;
    });
  RemoteObjectServerLayer<MockObjectLayer, RPCEndpoint> Server(BaseLayer,
                                                               ServerEP,
                                                               ReportError);

  bool Finished = false;
  ServerEP.addHandler<remote::utils::TerminateSession>(
    [&]() { Finished = true; }
  );

  auto ServerThread =
    std::thread([&]() {
      while (!Finished)
        cantFail(ServerEP.handleOne());
    });

  cantFail(Client.addObject(std::move(TestObject),
                            std::make_shared<NullResolver>()));

  // Check that we can find and materialize a valid symbol.
  auto Sym1 = Client.findSymbol("foobar", true);
  EXPECT_TRUE(!!Sym1) << "Symbol 'foobar' should be findable";
  EXPECT_EQ(cantFail(Sym1.getAddress()), 0x12348765ULL)
    << "Symbol 'foobar' does not return the correct address";

  {
    // Check that we can return a symbol containing an error.
    auto Sym2 = Client.findSymbol("badsymbol", true);
    EXPECT_FALSE(!!Sym2) << "Symbol 'badsymbol' should not be findable";
    auto Err = Sym2.takeError();
    EXPECT_TRUE(!!Err) << "Sym2 should contain an error value";
    auto ErrMsg = toString(std::move(Err));
    EXPECT_EQ(ErrMsg, "Could not find symbol 'badsymbol'")
      << "Expected symbol-not-found error for Sym2";
  }

  {
    // Check that we can return a 'null' symbol.
    auto Sym3 = Client.findSymbol("baz", true);
    EXPECT_FALSE(!!Sym3) << "Symbol 'baz' should convert to false";
    auto Err = Sym3.takeError();
    EXPECT_FALSE(!!Err) << "Symbol 'baz' should not contain an error";
  }

  cantFail(ClientEP.callB<remote::utils::TerminateSession>());
  ServerThread.join();
}

TEST(RemoteObjectLayer, FindSymbolIn) {
  llvm::orc::rpc::registerStringError<rpc::RawByteChannel>();
  auto TestObject = createTestObject();
  if (!TestObject)
    return;

  auto Channels = createPairedQueueChannels();

  auto ReportError =
    [](Error Err) {
      auto ErrMsg = toString(std::move(Err));
      EXPECT_EQ(ErrMsg, "Could not find symbol 'barbaz'")
        << "Expected error string to be \"Object handle 42 not found\"";
    };

  RPCEndpoint ClientEP(*Channels.first, true);
  RemoteObjectClientLayer<RPCEndpoint> Client(ClientEP, ReportError);

  RPCEndpoint ServerEP(*Channels.second, true);

  // AddObject lambda does not update symbol table, so removeObject will treat
  // this as a bad object handle.
  MockObjectLayer BaseLayer(
    [](MockObjectLayer::ObjectPtr Obj,
       MockObjectLayer::SymbolLookupTable &SymTab) {
      SymTab[42] =
        [](StringRef Name, bool ExportedSymbolsOnly) -> JITSymbol {
          if (Name == "foobar")
            return JITSymbol(0x12348765, JITSymbolFlags::Exported);
          return make_error<JITSymbolNotFound>(Name);
        };
      // Dummy symbol table entry - this should not be visible to
      // findSymbolIn.
      SymTab[43] =
        [](StringRef Name, bool ExportedSymbolsOnly) -> JITSymbol {
          if (Name == "barbaz")
            return JITSymbol(0xdeadbeef, JITSymbolFlags::Exported);
          return make_error<JITSymbolNotFound>(Name);
        };

      return 42;
    });
  RemoteObjectServerLayer<MockObjectLayer, RPCEndpoint> Server(BaseLayer,
                                                               ServerEP,
                                                               ReportError);

  bool Finished = false;
  ServerEP.addHandler<remote::utils::TerminateSession>(
    [&]() { Finished = true; }
  );

  auto ServerThread =
    std::thread([&]() {
      while (!Finished)
        cantFail(ServerEP.handleOne());
    });

  auto H = cantFail(Client.addObject(std::move(TestObject),
                                     std::make_shared<NullResolver>()));

  auto Sym1 = Client.findSymbolIn(H, "foobar", true);

  EXPECT_TRUE(!!Sym1) << "Symbol 'foobar' should be findable";
  EXPECT_EQ(cantFail(Sym1.getAddress()), 0x12348765ULL)
    << "Symbol 'foobar' does not return the correct address";

  auto Sym2 = Client.findSymbolIn(H, "barbaz", true);
  EXPECT_FALSE(!!Sym2) << "Symbol 'barbaz' should not be findable";
  auto Err = Sym2.takeError();
  EXPECT_TRUE(!!Err) << "Sym2 should contain an error value";
  auto ErrMsg = toString(std::move(Err));
  EXPECT_EQ(ErrMsg, "Could not find symbol 'barbaz'")
    << "Expected symbol-not-found error for Sym2";

  cantFail(ClientEP.callB<remote::utils::TerminateSession>());
  ServerThread.join();
}

TEST(RemoteObjectLayer, EmitAndFinalize) {
  llvm::orc::rpc::registerStringError<rpc::RawByteChannel>();
  auto TestObject = createTestObject();
  if (!TestObject)
    return;

  auto Channels = createPairedQueueChannels();

  auto ReportError =
    [](Error Err) {
      logAllUnhandledErrors(std::move(Err), llvm::errs(), "");
    };

  RPCEndpoint ClientEP(*Channels.first, true);
  RemoteObjectClientLayer<RPCEndpoint> Client(ClientEP, ReportError);

  RPCEndpoint ServerEP(*Channels.second, true);

  MockObjectLayer BaseLayer(
    [](MockObjectLayer::ObjectPtr Obj,
       MockObjectLayer::SymbolLookupTable &SymTab) {
      SymTab[1] = MockObjectLayer::LookupFn();
      return 1;
    });
  RemoteObjectServerLayer<MockObjectLayer, RPCEndpoint> Server(BaseLayer,
                                                               ServerEP,
                                                               ReportError);

  bool Finished = false;
  ServerEP.addHandler<remote::utils::TerminateSession>(
    [&]() { Finished = true; }
  );

  auto ServerThread =
    std::thread([&]() {
      while (!Finished)
        cantFail(ServerEP.handleOne());
    });

  auto H = cantFail(Client.addObject(std::move(TestObject),
                                     std::make_shared<NullResolver>()));

  auto Err = Client.emitAndFinalize(H);
  EXPECT_FALSE(!!Err) << "emitAndFinalize should work";

  cantFail(ClientEP.callB<remote::utils::TerminateSession>());
  ServerThread.join();
}

TEST(RemoteObjectLayer, EmitAndFinalizeFailure) {
  llvm::orc::rpc::registerStringError<rpc::RawByteChannel>();
  auto TestObject = createTestObject();
  if (!TestObject)
    return;

  auto Channels = createPairedQueueChannels();

  auto ReportError =
    [](Error Err) {
      auto ErrMsg = toString(std::move(Err));
      EXPECT_EQ(ErrMsg, "Object handle 1 not found")
        << "Expected bad handle error";
    };

  RPCEndpoint ClientEP(*Channels.first, true);
  RemoteObjectClientLayer<RPCEndpoint> Client(ClientEP, ReportError);

  RPCEndpoint ServerEP(*Channels.second, true);

  MockObjectLayer BaseLayer(
    [](MockObjectLayer::ObjectPtr Obj,
       MockObjectLayer::SymbolLookupTable &SymTab) {
      return 1;
    });
  RemoteObjectServerLayer<MockObjectLayer, RPCEndpoint> Server(BaseLayer,
                                                               ServerEP,
                                                               ReportError);

  bool Finished = false;
  ServerEP.addHandler<remote::utils::TerminateSession>(
    [&]() { Finished = true; }
  );

  auto ServerThread =
    std::thread([&]() {
      while (!Finished)
        cantFail(ServerEP.handleOne());
    });

  auto H = cantFail(Client.addObject(std::move(TestObject),
                                     std::make_shared<NullResolver>()));

  auto Err = Client.emitAndFinalize(H);
  EXPECT_TRUE(!!Err) << "emitAndFinalize should work";

  auto ErrMsg = toString(std::move(Err));
  EXPECT_EQ(ErrMsg, "Object handle 1 not found")
    << "emitAndFinalize returned incorrect error";

  cantFail(ClientEP.callB<remote::utils::TerminateSession>());
  ServerThread.join();
}

}
