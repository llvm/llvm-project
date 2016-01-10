//===-- ObjectLinkingLayerTest.cpp - Unit tests for object linking layer --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

class ObjectLinkingLayerExecutionTest : public testing::Test,
                                        public OrcExecutionTest {
};

TEST(ObjectLinkingLayerTest, TestSetProcessAllSections) {

  class SectionMemoryManagerWrapper : public SectionMemoryManager {
  public:
    SectionMemoryManagerWrapper(bool &DebugSeen) : DebugSeen(DebugSeen) {}
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
    bool DebugSeen;
  };

  ObjectLinkingLayer<> ObjLayer;

  auto M = llvm::make_unique<Module>("", getGlobalContext());
  M->setTargetTriple("x86_64-unknown-linux-gnu");
  Type *Int32Ty = IntegerType::get(getGlobalContext(), 32);
  GlobalVariable *GV =
    new GlobalVariable(*M, Int32Ty, false, GlobalValue::ExternalLinkage,
                         ConstantInt::get(Int32Ty, 42), "foo");

  GV->setSection(".debug_str");

  std::unique_ptr<TargetMachine> TM(
    EngineBuilder().selectTarget(Triple(M->getTargetTriple()), "", "",
                                 SmallVector<std::string, 1>()));
  if (!TM)
    return;

  auto OwningObj = SimpleCompiler(*TM)(*M);
  std::vector<object::ObjectFile*> Objs;
  Objs.push_back(OwningObj.getBinary());

  bool DebugSectionSeen = false;
  SectionMemoryManagerWrapper SMMW(DebugSectionSeen);
  auto Resolver =
    createLambdaResolver(
      [](const std::string &Name) {
        return RuntimeDyld::SymbolInfo(nullptr);
      },
      [](const std::string &Name) {
        return RuntimeDyld::SymbolInfo(nullptr);
      });

  {
    // Test with ProcessAllSections = false (the default).
    auto H = ObjLayer.addObjectSet(Objs, &SMMW, &*Resolver);
    EXPECT_EQ(DebugSectionSeen, false)
      << "Unexpected debug info section";
    ObjLayer.removeObjectSet(H);
  }

  {
    // Test with ProcessAllSections = true.
    ObjLayer.setProcessAllSections(true);
    auto H = ObjLayer.addObjectSet(Objs, &SMMW, &*Resolver);
    EXPECT_EQ(DebugSectionSeen, true)
      << "Expected debug info section not seen";
    ObjLayer.removeObjectSet(H);
  }
}


TEST_F(ObjectLinkingLayerExecutionTest, NoDuplicateFinalization) {

  if (!TM)
    return;

  class SectionMemoryManagerWrapper : public SectionMemoryManager {
  public:
    int FinalizationCount = 0;
    bool finalizeMemory(std::string *ErrMsg = 0) override {
      ++FinalizationCount;
      return SectionMemoryManager::finalizeMemory(ErrMsg);
    }
  };

  ObjectLinkingLayer<> ObjLayer;
  SimpleCompiler Compile(*TM);

  // Create a pair of modules that will trigger recursive finalization:
  // Module 1:
  //   int bar() { return 42; }
  // Module 2:
  //   int bar();
  //   int foo() { return bar(); }

  ModuleBuilder MB1(getGlobalContext(), "", "dummy");
  {
    MB1.getModule()->setDataLayout(TM->createDataLayout());
    Function *BarImpl = MB1.createFunctionDecl<int32_t(void)>("bar");
    BasicBlock *BarEntry = BasicBlock::Create(getGlobalContext(), "entry",
                                              BarImpl);
    IRBuilder<> Builder(BarEntry);
    IntegerType *Int32Ty = IntegerType::get(getGlobalContext(), 32);
    Value *FourtyTwo = ConstantInt::getSigned(Int32Ty, 42);
    Builder.CreateRet(FourtyTwo);
  }

  auto Obj1 = Compile(*MB1.getModule());
  std::vector<object::ObjectFile*> Obj1Set;
  Obj1Set.push_back(Obj1.getBinary());

  ModuleBuilder MB2(getGlobalContext(), "", "dummy");
  {
    MB2.getModule()->setDataLayout(TM->createDataLayout());
    Function *BarDecl = MB2.createFunctionDecl<int32_t(void)>("bar");
    Function *FooImpl = MB2.createFunctionDecl<int32_t(void)>("foo");
    BasicBlock *FooEntry = BasicBlock::Create(getGlobalContext(), "entry",
                                              FooImpl);
    IRBuilder<> Builder(FooEntry);
    Builder.CreateRet(Builder.CreateCall(BarDecl));
  }
  auto Obj2 = Compile(*MB2.getModule());
  std::vector<object::ObjectFile*> Obj2Set;
  Obj2Set.push_back(Obj2.getBinary());

  auto Resolver =
    createLambdaResolver(
      [&](const std::string &Name) {
        if (auto Sym = ObjLayer.findSymbol(Name, true))
          return RuntimeDyld::SymbolInfo(Sym.getAddress(), Sym.getFlags());
        return RuntimeDyld::SymbolInfo(nullptr);
      },
      [](const std::string &Name) {
        return RuntimeDyld::SymbolInfo(nullptr);
      });

  SectionMemoryManagerWrapper SMMW;
  ObjLayer.addObjectSet(std::move(Obj1Set), &SMMW, &*Resolver);
  auto H = ObjLayer.addObjectSet(std::move(Obj2Set), &SMMW, &*Resolver);
  ObjLayer.emitAndFinalize(H);

  // Finalization of module 2 should trigger finalization of module 1.
  // Verify that finalize on SMMW is only called once.
  EXPECT_EQ(SMMW.FinalizationCount, 1)
      << "Extra call to finalize";
}

}
