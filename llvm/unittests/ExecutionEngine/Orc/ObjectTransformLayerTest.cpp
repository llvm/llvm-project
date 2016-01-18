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
#include "gtest/gtest.h"

using namespace llvm::orc;

namespace {

// Stand-in for RuntimeDyld::MemoryManager
typedef int MockMemoryManager;

// Stand-in for RuntimeDyld::SymbolResolver
typedef int MockSymbolResolver;

// stand-in for object::ObjectFile
typedef int MockObjectFile;

// stand-in for llvm::MemoryBuffer set
typedef int MockMemoryBufferSet;

// Mock transform that operates on unique pointers to object files, and
// allocates new object files rather than mutating the given ones.
struct AllocatingTransform {
  std::unique_ptr<MockObjectFile>
  operator()(std::unique_ptr<MockObjectFile> Obj) const {
    return llvm::make_unique<MockObjectFile>(*Obj + 1);
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
  typedef int ObjSetHandleT;

  MockBaseLayer() : MockSymbol(nullptr) { resetExpectations(); }

  template <typename ObjSetT, typename MemoryManagerPtrT,
            typename SymbolResolverPtrT>
  ObjSetHandleT addObjectSet(ObjSetT &Objects, MemoryManagerPtrT MemMgr,
                             SymbolResolverPtrT Resolver) {
    EXPECT_EQ(MockManager, *MemMgr) << "MM should pass through";
    EXPECT_EQ(MockResolver, *Resolver) << "Resolver should pass through";
    size_t I = 0;
    for (auto &ObjPtr : Objects) {
      EXPECT_EQ(MockObjects[I++] + 1, *ObjPtr) << "Transform should be applied";
    }
    EXPECT_EQ(MockObjects.size(), I) << "Number of objects should match";
    LastCalled = "addObjectSet";
    MockObjSetHandle = 111;
    return MockObjSetHandle;
  }
  template <typename ObjSetT>
  void expectAddObjectSet(ObjSetT &Objects, MockMemoryManager *MemMgr,
                          MockSymbolResolver *Resolver) {
    MockManager = *MemMgr;
    MockResolver = *Resolver;
    for (auto &ObjPtr : Objects) {
      MockObjects.push_back(*ObjPtr);
    }
  }
  void verifyAddObjectSet(ObjSetHandleT Returned) {
    EXPECT_EQ("addObjectSet", LastCalled);
    EXPECT_EQ(MockObjSetHandle, Returned) << "Return should pass through";
    resetExpectations();
  }

  void removeObjectSet(ObjSetHandleT H) {
    EXPECT_EQ(MockObjSetHandle, H);
    LastCalled = "removeObjectSet";
  }
  void expectRemoveObjectSet(ObjSetHandleT H) { MockObjSetHandle = H; }
  void verifyRemoveObjectSet() {
    EXPECT_EQ("removeObjectSet", LastCalled);
    resetExpectations();
  }

  JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    EXPECT_EQ(MockName, Name) << "Name should pass through";
    EXPECT_EQ(MockBool, ExportedSymbolsOnly) << "Flag should pass through";
    LastCalled = "findSymbol";
    MockSymbol = JITSymbol(122, llvm::JITSymbolFlags::None);
    return MockSymbol;
  }
  void expectFindSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    MockName = Name;
    MockBool = ExportedSymbolsOnly;
  }
  void verifyFindSymbol(llvm::orc::JITSymbol Returned) {
    EXPECT_EQ("findSymbol", LastCalled);
    EXPECT_EQ(MockSymbol.getAddress(), Returned.getAddress())
        << "Return should pass through";
    resetExpectations();
  }

  JITSymbol findSymbolIn(ObjSetHandleT H, const std::string &Name,
                         bool ExportedSymbolsOnly) {
    EXPECT_EQ(MockObjSetHandle, H) << "Handle should pass through";
    EXPECT_EQ(MockName, Name) << "Name should pass through";
    EXPECT_EQ(MockBool, ExportedSymbolsOnly) << "Flag should pass through";
    LastCalled = "findSymbolIn";
    MockSymbol = JITSymbol(122, llvm::JITSymbolFlags::None);
    return MockSymbol;
  }
  void expectFindSymbolIn(ObjSetHandleT H, const std::string &Name,
                          bool ExportedSymbolsOnly) {
    MockObjSetHandle = H;
    MockName = Name;
    MockBool = ExportedSymbolsOnly;
  }
  void verifyFindSymbolIn(llvm::orc::JITSymbol Returned) {
    EXPECT_EQ("findSymbolIn", LastCalled);
    EXPECT_EQ(MockSymbol.getAddress(), Returned.getAddress())
        << "Return should pass through";
    resetExpectations();
  }

  void emitAndFinalize(ObjSetHandleT H) {
    EXPECT_EQ(MockObjSetHandle, H) << "Handle should pass through";
    LastCalled = "emitAndFinalize";
  }
  void expectEmitAndFinalize(ObjSetHandleT H) { MockObjSetHandle = H; }
  void verifyEmitAndFinalize() {
    EXPECT_EQ("emitAndFinalize", LastCalled);
    resetExpectations();
  }

  void mapSectionAddress(ObjSetHandleT H, const void *LocalAddress,
                         TargetAddress TargetAddr) {
    EXPECT_EQ(MockObjSetHandle, H);
    EXPECT_EQ(MockLocalAddress, LocalAddress);
    EXPECT_EQ(MockTargetAddress, TargetAddr);
    LastCalled = "mapSectionAddress";
  }
  void expectMapSectionAddress(ObjSetHandleT H, const void *LocalAddress,
                               TargetAddress TargetAddr) {
    MockObjSetHandle = H;
    MockLocalAddress = LocalAddress;
    MockTargetAddress = TargetAddr;
  }
  void verifyMapSectionAddress() {
    EXPECT_EQ("mapSectionAddress", LastCalled);
    resetExpectations();
  }

  template <typename OwningMBSet>
  void takeOwnershipOfBuffers(ObjSetHandleT H, OwningMBSet MBs) {
    EXPECT_EQ(MockObjSetHandle, H);
    EXPECT_EQ(MockBufferSet, *MBs);
    LastCalled = "takeOwnershipOfBuffers";
  }
  void expectTakeOwnershipOfBuffers(ObjSetHandleT H, MockMemoryBufferSet *MBs) {
    MockObjSetHandle = H;
    MockBufferSet = *MBs;
  }
  void verifyTakeOwnershipOfBuffers() {
    EXPECT_EQ("takeOwnershipOfBuffers", LastCalled);
    resetExpectations();
  }

private:
  // Backing fields for remembering parameter/return values
  std::string LastCalled;
  MockMemoryManager MockManager;
  MockSymbolResolver MockResolver;
  std::vector<MockObjectFile> MockObjects;
  ObjSetHandleT MockObjSetHandle;
  std::string MockName;
  bool MockBool;
  JITSymbol MockSymbol;
  const void *MockLocalAddress;
  TargetAddress MockTargetAddress;
  MockMemoryBufferSet MockBufferSet;

  // Clear remembered parameters between calls
  void resetExpectations() {
    LastCalled = "nothing";
    MockManager = 0;
    MockResolver = 0;
    MockObjects.clear();
    MockObjSetHandle = 0;
    MockName = "bogus";
    MockSymbol = JITSymbol(nullptr);
    MockLocalAddress = nullptr;
    MockTargetAddress = 0;
    MockBufferSet = 0;
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
                       std::function<MockObjectFile *(MockObjectFile *)>>
  T2(M, [](MockObjectFile *Obj) {
    ++(*Obj);
    return Obj;
  });

  // Instantiate some mock objects to use below
  MockObjectFile MockObject1 = 211;
  MockObjectFile MockObject2 = 222;
  MockMemoryManager MockManager = 233;
  MockSymbolResolver MockResolver = 244;

  // Test addObjectSet with T1 (allocating, unique pointers)
  std::vector<std::unique_ptr<MockObjectFile>> Objs1;
  Objs1.push_back(llvm::make_unique<MockObjectFile>(MockObject1));
  Objs1.push_back(llvm::make_unique<MockObjectFile>(MockObject2));
  auto MM = llvm::make_unique<MockMemoryManager>(MockManager);
  auto SR = llvm::make_unique<MockSymbolResolver>(MockResolver);
  M.expectAddObjectSet(Objs1, MM.get(), SR.get());
  auto H = T1.addObjectSet(Objs1, std::move(MM), std::move(SR));
  M.verifyAddObjectSet(H);

  // Test addObjectSet with T2 (mutating, naked pointers)
  llvm::SmallVector<MockObjectFile *, 2> Objs2;
  Objs2.push_back(&MockObject1);
  Objs2.push_back(&MockObject2);
  M.expectAddObjectSet(Objs2, &MockManager, &MockResolver);
  H = T2.addObjectSet(Objs2, &MockManager, &MockResolver);
  M.verifyAddObjectSet(H);
  EXPECT_EQ(212, MockObject1) << "Expected mutation";
  EXPECT_EQ(223, MockObject2) << "Expected mutation";

  // Test removeObjectSet
  M.expectRemoveObjectSet(H);
  T1.removeObjectSet(H);
  M.verifyRemoveObjectSet();

  // Test findSymbol
  std::string Name = "foo";
  bool ExportedOnly = true;
  M.expectFindSymbol(Name, ExportedOnly);
  JITSymbol Symbol = T2.findSymbol(Name, ExportedOnly);
  M.verifyFindSymbol(Symbol);

  // Test findSymbolIn
  Name = "bar";
  ExportedOnly = false;
  M.expectFindSymbolIn(H, Name, ExportedOnly);
  Symbol = T1.findSymbolIn(H, Name, ExportedOnly);
  M.verifyFindSymbolIn(Symbol);

  // Test emitAndFinalize
  M.expectEmitAndFinalize(H);
  T2.emitAndFinalize(H);
  M.verifyEmitAndFinalize();

  // Test mapSectionAddress
  char Buffer[24];
  TargetAddress MockAddress = 255;
  M.expectMapSectionAddress(H, Buffer, MockAddress);
  T1.mapSectionAddress(H, Buffer, MockAddress);
  M.verifyMapSectionAddress();

  // Test takeOwnershipOfBuffers, using unique pointer to buffer set
  auto MockBufferSetPtr = llvm::make_unique<MockMemoryBufferSet>(366);
  M.expectTakeOwnershipOfBuffers(H, MockBufferSetPtr.get());
  T2.takeOwnershipOfBuffers(H, std::move(MockBufferSetPtr));
  M.verifyTakeOwnershipOfBuffers();

  // Test takeOwnershipOfBuffers, using naked pointer to buffer set
  MockMemoryBufferSet MockBufferSet = 266;
  M.expectTakeOwnershipOfBuffers(H, &MockBufferSet);
  T1.takeOwnershipOfBuffers(H, &MockBufferSet);
  M.verifyTakeOwnershipOfBuffers();

  // Verify transform getter (non-const)
  MockObjectFile Mutatee = 277;
  MockObjectFile *Out = T2.getTransform()(&Mutatee);
  EXPECT_EQ(&Mutatee, Out) << "Expected in-place transform";
  EXPECT_EQ(278, Mutatee) << "Expected incrementing transform";

  // Verify transform getter (const)
  auto OwnedObj = llvm::make_unique<MockObjectFile>(288);
  const auto &T1C = T1;
  OwnedObj = T1C.getTransform()(std::move(OwnedObj));
  EXPECT_EQ(289, *OwnedObj) << "Expected incrementing transform";
}
}
