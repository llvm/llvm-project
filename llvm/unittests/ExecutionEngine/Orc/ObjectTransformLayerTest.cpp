//===- ObjectTransformLayerTest.cpp - Unit tests for ObjectTransformLayer -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/ObjectTransformLayer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/NullResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/Object/ObjectFile.h"
#include "gtest/gtest.h"

using namespace llvm::orc;

namespace {

// stand-in for object::ObjectFile
typedef int MockObjectFile;

// stand-in for llvm::MemoryBuffer set
typedef int MockMemoryBuffer;

// Mock transform that operates on unique pointers to object files, and
// allocates new object files rather than mutating the given ones.
struct AllocatingTransform {
  std::shared_ptr<MockObjectFile>
  operator()(std::shared_ptr<MockObjectFile> Obj) const {
    return std::make_shared<MockObjectFile>(*Obj + 1);
  }
};

// Mock base layer for verifying behavior of transform layer.
// Each method "T foo(args)" is accompanied by two auxiliary methods:
//  - "void expectFoo(args)", to be called before calling foo on the transform
//      layer; saves values of args, which mock layer foo then verifies against.
// - "void verifyFoo(T)", to be called after foo, which verifies that the
//      transform layer called the base layer and forwarded any return value.
class MockBaseLayer {
public:
  typedef int ObjHandleT;

  MockBaseLayer() : MockSymbol(nullptr) { resetExpectations(); }

  template <typename ObjPtrT>
  llvm::Expected<ObjHandleT>
  addObject(ObjPtrT Obj,
            std::shared_ptr<llvm::JITSymbolResolver> Resolver) {
    EXPECT_EQ(MockResolver, Resolver) << "Resolver should pass through";
    EXPECT_EQ(MockObject + 1, *Obj) << "Transform should be applied";
    LastCalled = "addObject";
    MockObjHandle = 111;
    return MockObjHandle;
  }

  template <typename ObjPtrT>
  void expectAddObject(ObjPtrT Obj,
                       std::shared_ptr<llvm::JITSymbolResolver> Resolver) {
    MockResolver = Resolver;
    MockObject = *Obj;
  }


  void verifyAddObject(ObjHandleT Returned) {
    EXPECT_EQ("addObject", LastCalled);
    EXPECT_EQ(MockObjHandle, Returned) << "Return should pass through";
    resetExpectations();
  }

  llvm::Error removeObject(ObjHandleT H) {
    EXPECT_EQ(MockObjHandle, H);
    LastCalled = "removeObject";
    return llvm::Error::success();
  }

  void expectRemoveObject(ObjHandleT H) { MockObjHandle = H; }
  void verifyRemoveObject() {
    EXPECT_EQ("removeObject", LastCalled);
    resetExpectations();
  }

  llvm::JITSymbol findSymbol(const std::string &Name,
                             bool ExportedSymbolsOnly) {
    EXPECT_EQ(MockName, Name) << "Name should pass through";
    EXPECT_EQ(MockBool, ExportedSymbolsOnly) << "Flag should pass through";
    LastCalled = "findSymbol";
    MockSymbol = llvm::JITSymbol(122, llvm::JITSymbolFlags::None);
    return llvm::JITSymbol(122, llvm::JITSymbolFlags::None);
  }
  void expectFindSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    MockName = Name;
    MockBool = ExportedSymbolsOnly;
  }
  void verifyFindSymbol(llvm::JITSymbol Returned) {
    EXPECT_EQ("findSymbol", LastCalled);
    EXPECT_EQ(cantFail(MockSymbol.getAddress()),
              cantFail(Returned.getAddress()))
        << "Return should pass through";
    resetExpectations();
  }

  llvm::JITSymbol findSymbolIn(ObjHandleT H, const std::string &Name,
                               bool ExportedSymbolsOnly) {
    EXPECT_EQ(MockObjHandle, H) << "Handle should pass through";
    EXPECT_EQ(MockName, Name) << "Name should pass through";
    EXPECT_EQ(MockBool, ExportedSymbolsOnly) << "Flag should pass through";
    LastCalled = "findSymbolIn";
    MockSymbol = llvm::JITSymbol(122, llvm::JITSymbolFlags::None);
    return llvm::JITSymbol(122, llvm::JITSymbolFlags::None);
  }
  void expectFindSymbolIn(ObjHandleT H, const std::string &Name,
                          bool ExportedSymbolsOnly) {
    MockObjHandle = H;
    MockName = Name;
    MockBool = ExportedSymbolsOnly;
  }
  void verifyFindSymbolIn(llvm::JITSymbol Returned) {
    EXPECT_EQ("findSymbolIn", LastCalled);
    EXPECT_EQ(cantFail(MockSymbol.getAddress()),
              cantFail(Returned.getAddress()))
        << "Return should pass through";
    resetExpectations();
  }

  llvm::Error emitAndFinalize(ObjHandleT H) {
    EXPECT_EQ(MockObjHandle, H) << "Handle should pass through";
    LastCalled = "emitAndFinalize";
    return llvm::Error::success();
  }

  void expectEmitAndFinalize(ObjHandleT H) { MockObjHandle = H; }

  void verifyEmitAndFinalize() {
    EXPECT_EQ("emitAndFinalize", LastCalled);
    resetExpectations();
  }

  void mapSectionAddress(ObjHandleT H, const void *LocalAddress,
                         llvm::JITTargetAddress TargetAddr) {
    EXPECT_EQ(MockObjHandle, H);
    EXPECT_EQ(MockLocalAddress, LocalAddress);
    EXPECT_EQ(MockTargetAddress, TargetAddr);
    LastCalled = "mapSectionAddress";
  }
  void expectMapSectionAddress(ObjHandleT H, const void *LocalAddress,
                               llvm::JITTargetAddress TargetAddr) {
    MockObjHandle = H;
    MockLocalAddress = LocalAddress;
    MockTargetAddress = TargetAddr;
  }
  void verifyMapSectionAddress() {
    EXPECT_EQ("mapSectionAddress", LastCalled);
    resetExpectations();
  }

private:
  // Backing fields for remembering parameter/return values
  std::string LastCalled;
  std::shared_ptr<llvm::JITSymbolResolver> MockResolver;
  MockObjectFile MockObject;
  ObjHandleT MockObjHandle;
  std::string MockName;
  bool MockBool;
  llvm::JITSymbol MockSymbol;
  const void *MockLocalAddress;
  llvm::JITTargetAddress MockTargetAddress;
  MockMemoryBuffer MockBuffer;

  // Clear remembered parameters between calls
  void resetExpectations() {
    LastCalled = "nothing";
    MockResolver = nullptr;
    MockObject = 0;
    MockObjHandle = 0;
    MockName = "bogus";
    MockSymbol = llvm::JITSymbol(nullptr);
    MockLocalAddress = nullptr;
    MockTargetAddress = 0;
    MockBuffer = 0;
  }
};

// Test each operation on ObjectTransformLayer.
TEST(ObjectTransformLayerTest, Main) {
  MockBaseLayer M;

  // Create one object transform layer using a transform (as a functor)
  // that allocates new objects, and deals in unique pointers.
  ObjectTransformLayer<MockBaseLayer, AllocatingTransform> T1(M);

  // Create a second object transform layer using a transform (as a lambda)
  // that mutates objects in place, and deals in naked pointers
  ObjectTransformLayer<MockBaseLayer,
                         std::function<std::shared_ptr<MockObjectFile>(
                           std::shared_ptr<MockObjectFile>)>>
    T2(M, [](std::shared_ptr<MockObjectFile> Obj) {
    ++(*Obj);
    return Obj;
  });

  // Test addObject with T1 (allocating)
  auto Obj1 = std::make_shared<MockObjectFile>(211);
  auto SR = std::make_shared<NullResolver>();
  M.expectAddObject(Obj1, SR);
  auto H = cantFail(T1.addObject(std::move(Obj1), SR));
  M.verifyAddObject(H);

  // Test addObjectSet with T2 (mutating)
  auto Obj2 = std::make_shared<MockObjectFile>(222);
  M.expectAddObject(Obj2, SR);
  H = cantFail(T2.addObject(Obj2, SR));
  M.verifyAddObject(H);
  EXPECT_EQ(223, *Obj2) << "Expected mutation";

  // Test removeObjectSet
  M.expectRemoveObject(H);
  cantFail(T1.removeObject(H));
  M.verifyRemoveObject();

  // Test findSymbol
  std::string Name = "foo";
  bool ExportedOnly = true;
  M.expectFindSymbol(Name, ExportedOnly);
  llvm::JITSymbol Sym1 = T2.findSymbol(Name, ExportedOnly);
  M.verifyFindSymbol(std::move(Sym1));

  // Test findSymbolIn
  Name = "bar";
  ExportedOnly = false;
  M.expectFindSymbolIn(H, Name, ExportedOnly);
  llvm::JITSymbol Sym2 = T1.findSymbolIn(H, Name, ExportedOnly);
  M.verifyFindSymbolIn(std::move(Sym2));

  // Test emitAndFinalize
  M.expectEmitAndFinalize(H);
  cantFail(T2.emitAndFinalize(H));
  M.verifyEmitAndFinalize();

  // Test mapSectionAddress
  char Buffer[24];
  llvm::JITTargetAddress MockAddress = 255;
  M.expectMapSectionAddress(H, Buffer, MockAddress);
  T1.mapSectionAddress(H, Buffer, MockAddress);
  M.verifyMapSectionAddress();

  // Verify transform getter (non-const)
  auto Mutatee = std::make_shared<MockObjectFile>(277);
  auto Out = T2.getTransform()(Mutatee);
  EXPECT_EQ(*Mutatee, *Out) << "Expected in-place transform";
  EXPECT_EQ(278, *Mutatee) << "Expected incrementing transform";

  // Verify transform getter (const)
  auto OwnedObj = std::make_shared<MockObjectFile>(288);
  const auto &T1C = T1;
  OwnedObj = T1C.getTransform()(std::move(OwnedObj));
  EXPECT_EQ(289, *OwnedObj) << "Expected incrementing transform";

  volatile bool RunStaticChecks = false;
  if (!RunStaticChecks)
    return;

  // Make sure that ObjectTransformLayer implements the object layer concept
  // correctly by sandwitching one between an ObjectLinkingLayer and an
  // IRCompileLayer, verifying that it compiles if we have a call to the
  // IRComileLayer's addModule that should call the transform layer's
  // addObject, and also calling the other public transform layer methods
  // directly to make sure the methods they intend to forward to exist on
  // the ObjectLinkingLayer.

  // We'll need a concrete MemoryManager class.
  class NullManager : public llvm::RuntimeDyld::MemoryManager {
  public:
    uint8_t *allocateCodeSection(uintptr_t, unsigned, unsigned,
                                 llvm::StringRef) override {
      return nullptr;
    }
    uint8_t *allocateDataSection(uintptr_t, unsigned, unsigned, llvm::StringRef,
                                 bool) override {
      return nullptr;
    }
    void registerEHFrames(uint8_t *, uint64_t, size_t) override {}
    void deregisterEHFrames() override {}
    bool finalizeMemory(std::string *) override { return false; }
  };

  // Construct the jit layers.
  RTDyldObjectLinkingLayer BaseLayer(
    []() {
      return std::make_shared<llvm::SectionMemoryManager>();
    });

  auto IdentityTransform =
    [](std::shared_ptr<llvm::object::OwningBinary<llvm::object::ObjectFile>>
       Obj) {
      return Obj;
    };
  ObjectTransformLayer<decltype(BaseLayer), decltype(IdentityTransform)>
      TransformLayer(BaseLayer, IdentityTransform);
  auto NullCompiler = [](llvm::Module &) {
    return llvm::object::OwningBinary<llvm::object::ObjectFile>(nullptr,
                                                                nullptr);
  };
  IRCompileLayer<decltype(TransformLayer), decltype(NullCompiler)>
    CompileLayer(TransformLayer, NullCompiler);

  // Make sure that the calls from IRCompileLayer to ObjectTransformLayer
  // compile.
  auto Resolver = std::make_shared<NullResolver>();
  cantFail(CompileLayer.addModule(std::shared_ptr<llvm::Module>(), Resolver));

  // Make sure that the calls from ObjectTransformLayer to ObjectLinkingLayer
  // compile.
  decltype(TransformLayer)::ObjHandleT H2;
  cantFail(TransformLayer.emitAndFinalize(H2));
  TransformLayer.findSymbolIn(H2, Name, false);
  TransformLayer.findSymbol(Name, true);
  TransformLayer.mapSectionAddress(H2, nullptr, 0);
  cantFail(TransformLayer.removeObject(H2));
}
}
