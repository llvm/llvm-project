//===- RTDyldObjectLinkingLayerTest.cpp - RTDyld linking layer unit tests -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/Legacy.h"
#include "llvm/ExecutionEngine/Orc/NullResolver.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

class LegacyRTDyldObjectLinkingLayerExecutionTest : public testing::Test,
                                              public OrcExecutionTest {

};

class SectionMemoryManagerWrapper : public SectionMemoryManager {
public:
  int FinalizationCount = 0;
  int NeedsToReserveAllocationSpaceCount = 0;

  bool needsToReserveAllocationSpace() override {
    ++NeedsToReserveAllocationSpaceCount;
    return SectionMemoryManager::needsToReserveAllocationSpace();
  }

  bool finalizeMemory(std::string *ErrMsg = nullptr) override {
    ++FinalizationCount;
    return SectionMemoryManager::finalizeMemory(ErrMsg);
  }
};

TEST(LegacyRTDyldObjectLinkingLayerTest, TestSetProcessAllSections) {
  class MemoryManagerWrapper : public SectionMemoryManager {
  public:
    MemoryManagerWrapper(bool &DebugSeen) : DebugSeen(DebugSeen) {}
    uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                 unsigned SectionID,
                                 StringRef SectionName,
                                 bool IsReadOnly) override {
      if (SectionName == ".debug_str")
        DebugSeen = true;
      return SectionMemoryManager::allocateDataSection(Size, Alignment,
                                                         SectionID,
                                                         SectionName,
                                                         IsReadOnly);
    }
  private:
    bool &DebugSeen;
  };

  bool DebugSectionSeen = false;
  auto MM = std::make_shared<MemoryManagerWrapper>(DebugSectionSeen);

  ExecutionSession ES;

  LegacyRTDyldObjectLinkingLayer ObjLayer(ES, [&MM](VModuleKey) {
    return LegacyRTDyldObjectLinkingLayer::Resources{
        MM, std::make_shared<NullResolver>()};
  });

  LLVMContext Context;
  auto M = llvm::make_unique<Module>("", Context);
  M->setTargetTriple("x86_64-unknown-linux-gnu");
  Type *Int32Ty = IntegerType::get(Context, 32);
  GlobalVariable *GV =
    new GlobalVariable(*M, Int32Ty, false, GlobalValue::ExternalLinkage,
                         ConstantInt::get(Int32Ty, 42), "foo");

  GV->setSection(".debug_str");


  // Initialize the native target in case this is the first unit test
  // to try to build a TM.
  OrcNativeTarget::initialize();
  std::unique_ptr<TargetMachine> TM(
    EngineBuilder().selectTarget(Triple(M->getTargetTriple()), "", "",
                                 SmallVector<std::string, 1>()));
  if (!TM)
    return;

  auto Obj = SimpleCompiler(*TM)(*M);

  {
    // Test with ProcessAllSections = false (the default).
    auto K = ES.allocateVModule();
    cantFail(ObjLayer.addObject(
        K, MemoryBuffer::getMemBufferCopy(Obj->getBuffer())));
    cantFail(ObjLayer.emitAndFinalize(K));
    EXPECT_EQ(DebugSectionSeen, false)
      << "Unexpected debug info section";
    cantFail(ObjLayer.removeObject(K));
  }

  {
    // Test with ProcessAllSections = true.
    ObjLayer.setProcessAllSections(true);
    auto K = ES.allocateVModule();
    cantFail(ObjLayer.addObject(K, std::move(Obj)));
    cantFail(ObjLayer.emitAndFinalize(K));
    EXPECT_EQ(DebugSectionSeen, true)
      << "Expected debug info section not seen";
    cantFail(ObjLayer.removeObject(K));
  }
}

TEST_F(LegacyRTDyldObjectLinkingLayerExecutionTest, NoDuplicateFinalization) {
  if (!SupportsJIT)
    return;

  ExecutionSession ES;

  auto MM = std::make_shared<SectionMemoryManagerWrapper>();

  std::map<orc::VModuleKey, std::shared_ptr<orc::SymbolResolver>> Resolvers;

  LegacyRTDyldObjectLinkingLayer ObjLayer(ES, [&](VModuleKey K) {
    auto I = Resolvers.find(K);
    assert(I != Resolvers.end() && "Missing resolver");
    auto R = std::move(I->second);
    Resolvers.erase(I);
    return LegacyRTDyldObjectLinkingLayer::Resources{MM, std::move(R)};
  });
  SimpleCompiler Compile(*TM);

  // Create a pair of modules that will trigger recursive finalization:
  // Module 1:
  //   int bar() { return 42; }
  // Module 2:
  //   int bar();
  //   int foo() { return bar(); }
  //
  // Verify that the memory manager is only finalized once (for Module 2).
  // Failure suggests that finalize is being called on the inner RTDyld
  // instance (for Module 1) which is unsafe, as it will prevent relocation of
  // Module 2.

  ModuleBuilder MB1(Context, "", "dummy");
  {
    MB1.getModule()->setDataLayout(TM->createDataLayout());
    Function *BarImpl = MB1.createFunctionDecl<int32_t(void)>("bar");
    BasicBlock *BarEntry = BasicBlock::Create(Context, "entry", BarImpl);
    IRBuilder<> Builder(BarEntry);
    IntegerType *Int32Ty = IntegerType::get(Context, 32);
    Value *FourtyTwo = ConstantInt::getSigned(Int32Ty, 42);
    Builder.CreateRet(FourtyTwo);
  }

  auto Obj1 = Compile(*MB1.getModule());

  ModuleBuilder MB2(Context, "", "dummy");
  {
    MB2.getModule()->setDataLayout(TM->createDataLayout());
    Function *BarDecl = MB2.createFunctionDecl<int32_t(void)>("bar");
    Function *FooImpl = MB2.createFunctionDecl<int32_t(void)>("foo");
    BasicBlock *FooEntry = BasicBlock::Create(Context, "entry", FooImpl);
    IRBuilder<> Builder(FooEntry);
    Builder.CreateRet(Builder.CreateCall(BarDecl));
  }
  auto Obj2 = Compile(*MB2.getModule());

  auto K1 = ES.allocateVModule();
  Resolvers[K1] = std::make_shared<NullResolver>();
  cantFail(ObjLayer.addObject(K1, std::move(Obj1)));

  auto K2 = ES.allocateVModule();
  auto LegacyLookup = [&](const std::string &Name) {
    return ObjLayer.findSymbol(Name, true);
  };

  Resolvers[K2] = createSymbolResolver(
      [&](const SymbolNameSet &Symbols) {
        return cantFail(
            getResponsibilitySetWithLegacyFn(Symbols, LegacyLookup));
      },
      [&](std::shared_ptr<AsynchronousSymbolQuery> Query,
          const SymbolNameSet &Symbols) {
        return lookupWithLegacyFn(ES, *Query, Symbols, LegacyLookup);
      });

  cantFail(ObjLayer.addObject(K2, std::move(Obj2)));
  cantFail(ObjLayer.emitAndFinalize(K2));
  cantFail(ObjLayer.removeObject(K2));

  // Finalization of module 2 should trigger finalization of module 1.
  // Verify that finalize on SMMW is only called once.
  EXPECT_EQ(MM->FinalizationCount, 1)
      << "Extra call to finalize";
}

TEST_F(LegacyRTDyldObjectLinkingLayerExecutionTest, NoPrematureAllocation) {
  if (!SupportsJIT)
    return;

  ExecutionSession ES;

  auto MM = std::make_shared<SectionMemoryManagerWrapper>();

  LegacyRTDyldObjectLinkingLayer ObjLayer(ES, [&MM](VModuleKey K) {
    return LegacyRTDyldObjectLinkingLayer::Resources{
        MM, std::make_shared<NullResolver>()};
  });
  SimpleCompiler Compile(*TM);

  // Create a pair of unrelated modules:
  //
  // Module 1:
  //   int foo() { return 42; }
  // Module 2:
  //   int bar() { return 7; }
  //
  // Both modules will share a memory manager. We want to verify that the
  // second object is not loaded before the first one is finalized. To do this
  // in a portable way, we abuse the
  // RuntimeDyld::MemoryManager::needsToReserveAllocationSpace hook, which is
  // called once per object before any sections are allocated.

  ModuleBuilder MB1(Context, "", "dummy");
  {
    MB1.getModule()->setDataLayout(TM->createDataLayout());
    Function *BarImpl = MB1.createFunctionDecl<int32_t(void)>("foo");
    BasicBlock *BarEntry = BasicBlock::Create(Context, "entry", BarImpl);
    IRBuilder<> Builder(BarEntry);
    IntegerType *Int32Ty = IntegerType::get(Context, 32);
    Value *FourtyTwo = ConstantInt::getSigned(Int32Ty, 42);
    Builder.CreateRet(FourtyTwo);
  }

  auto Obj1 = Compile(*MB1.getModule());

  ModuleBuilder MB2(Context, "", "dummy");
  {
    MB2.getModule()->setDataLayout(TM->createDataLayout());
    Function *BarImpl = MB2.createFunctionDecl<int32_t(void)>("bar");
    BasicBlock *BarEntry = BasicBlock::Create(Context, "entry", BarImpl);
    IRBuilder<> Builder(BarEntry);
    IntegerType *Int32Ty = IntegerType::get(Context, 32);
    Value *Seven = ConstantInt::getSigned(Int32Ty, 7);
    Builder.CreateRet(Seven);
  }
  auto Obj2 = Compile(*MB2.getModule());

  auto K = ES.allocateVModule();
  cantFail(ObjLayer.addObject(K, std::move(Obj1)));
  cantFail(ObjLayer.addObject(ES.allocateVModule(), std::move(Obj2)));
  cantFail(ObjLayer.emitAndFinalize(K));
  cantFail(ObjLayer.removeObject(K));

  // Only one call to needsToReserveAllocationSpace should have been made.
  EXPECT_EQ(MM->NeedsToReserveAllocationSpaceCount, 1)
      << "More than one call to needsToReserveAllocationSpace "
         "(multiple unrelated objects loaded prior to finalization)";
}

TEST_F(LegacyRTDyldObjectLinkingLayerExecutionTest, TestNotifyLoadedSignature) {
  ExecutionSession ES;
  LegacyRTDyldObjectLinkingLayer ObjLayer(
      ES,
      [](VModuleKey) {
        return LegacyRTDyldObjectLinkingLayer::Resources{
            nullptr, std::make_shared<NullResolver>()};
      },
      [](VModuleKey, const object::ObjectFile &obj,
         const RuntimeDyld::LoadedObjectInfo &info) {});
}

} // end anonymous namespace
