//===--- RTDyldObjectLinkingLayer2Test.cpp - RTDyld linking layer tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/Legacy.h"
#include "llvm/ExecutionEngine/Orc/NullResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

class RTDyldObjectLinkingLayer2ExecutionTest : public testing::Test,
                                               public OrcExecutionTest {};

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

// Adds an object with a debug section to RuntimeDyld and then returns whether
// the debug section was passed to the memory manager.
static bool testSetProcessAllSections(std::unique_ptr<MemoryBuffer> Obj,
                                      bool ProcessAllSections) {
  class MemoryManagerWrapper : public SectionMemoryManager {
  public:
    MemoryManagerWrapper(bool &DebugSeen) : DebugSeen(DebugSeen) {}
    uint8_t *allocateDataSection(uintptr_t Size, unsigned Alignment,
                                 unsigned SectionID, StringRef SectionName,
                                 bool IsReadOnly) override {
      if (SectionName == ".debug_str")
        DebugSeen = true;
      return SectionMemoryManager::allocateDataSection(
          Size, Alignment, SectionID, SectionName, IsReadOnly);
    }

  private:
    bool &DebugSeen;
  };

  bool DebugSectionSeen = false;
  auto MM = std::make_shared<MemoryManagerWrapper>(DebugSectionSeen);

  ExecutionSession ES(std::make_shared<SymbolStringPool>());
  auto &JD = ES.createJITDylib("main");
  auto Foo = ES.getSymbolStringPool().intern("foo");

  RTDyldObjectLinkingLayer2 ObjLayer(ES, [&MM](VModuleKey) { return MM; });

  auto OnResolveDoNothing = [](Expected<SymbolMap> R) {
    cantFail(std::move(R));
  };

  auto OnReadyDoNothing = [](Error Err) { cantFail(std::move(Err)); };

  ObjLayer.setProcessAllSections(ProcessAllSections);
  auto K = ES.allocateVModule();
  cantFail(ObjLayer.add(JD, K, std::move(Obj)));
  ES.lookup({&JD}, {Foo}, OnResolveDoNothing, OnReadyDoNothing,
            NoDependenciesToRegister);
  return DebugSectionSeen;
}

TEST(RTDyldObjectLinkingLayer2Test, TestSetProcessAllSections) {
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
  std::unique_ptr<TargetMachine> TM(EngineBuilder().selectTarget(
      Triple(M->getTargetTriple()), "", "", SmallVector<std::string, 1>()));
  if (!TM)
    return;

  auto Obj = SimpleCompiler(*TM)(*M);

  EXPECT_FALSE(testSetProcessAllSections(
      MemoryBuffer::getMemBufferCopy(Obj->getBuffer()), false))
      << "Debug section seen despite ProcessAllSections being false";
  EXPECT_TRUE(testSetProcessAllSections(std::move(Obj), true))
      << "Expected to see debug section when ProcessAllSections is true";
}

TEST(RTDyldObjectLinkingLayer2Test, NoDuplicateFinalization) {
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

  // Initialize the native target in case this is the first unit test
  // to try to build a TM.
  OrcNativeTarget::initialize();
  std::unique_ptr<TargetMachine> TM(
      EngineBuilder().selectTarget(Triple("x86_64-unknown-linux-gnu"), "", "",
                                   SmallVector<std::string, 1>()));

  if (!TM)
    return;

  LLVMContext Context;
  ExecutionSession ES(std::make_shared<SymbolStringPool>());
  auto &JD = ES.createJITDylib("main");

  auto Foo = ES.getSymbolStringPool().intern("foo");

  auto MM = std::make_shared<SectionMemoryManagerWrapper>();

  RTDyldObjectLinkingLayer2 ObjLayer(ES, [&](VModuleKey K) { return MM; });

  SimpleCompiler Compile(*TM);

  ModuleBuilder MB1(Context, TM->getTargetTriple().str(), "dummy");
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

  ModuleBuilder MB2(Context, TM->getTargetTriple().str(), "dummy");
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
  cantFail(ObjLayer.add(JD, K1, std::move(Obj1)));

  auto K2 = ES.allocateVModule();
  cantFail(ObjLayer.add(JD, K2, std::move(Obj2)));

  auto OnResolve = [](Expected<SymbolMap> Symbols) {
    cantFail(std::move(Symbols));
  };
  auto OnReady = [](Error Err) { cantFail(std::move(Err)); };

  ES.lookup({&JD}, {Foo}, OnResolve, OnReady, NoDependenciesToRegister);

  // Finalization of module 2 should trigger finalization of module 1.
  // Verify that finalize on SMMW is only called once.
  EXPECT_EQ(MM->FinalizationCount, 1) << "Extra call to finalize";
}

TEST(RTDyldObjectLinkingLayer2Test, NoPrematureAllocation) {
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

  // Initialize the native target in case this is the first unit test
  // to try to build a TM.
  OrcNativeTarget::initialize();
  std::unique_ptr<TargetMachine> TM(
      EngineBuilder().selectTarget(Triple("x86_64-unknown-linux-gnu"), "", "",
                                   SmallVector<std::string, 1>()));

  if (!TM)
    return;

  ExecutionSession ES(std::make_shared<SymbolStringPool>());
  auto &JD = ES.createJITDylib("main");

  auto Foo = ES.getSymbolStringPool().intern("foo");

  auto MM = std::make_shared<SectionMemoryManagerWrapper>();

  RTDyldObjectLinkingLayer2 ObjLayer(ES, [&MM](VModuleKey K) { return MM; });
  SimpleCompiler Compile(*TM);

  LLVMContext Context;
  ModuleBuilder MB1(Context, TM->getTargetTriple().str(), "dummy");
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

  ModuleBuilder MB2(Context, TM->getTargetTriple().str(), "dummy");
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

  cantFail(ObjLayer.add(JD, ES.allocateVModule(), std::move(Obj1)));
  cantFail(ObjLayer.add(JD, ES.allocateVModule(), std::move(Obj2)));

  auto OnResolve = [](Expected<SymbolMap> Result) {
    cantFail(std::move(Result));
  };

  auto OnReady = [](Error Err) { cantFail(std::move(Err)); };

  ES.lookup({&JD}, {Foo}, OnResolve, OnReady, NoDependenciesToRegister);

  // Only one call to needsToReserveAllocationSpace should have been made.
  EXPECT_EQ(MM->NeedsToReserveAllocationSpaceCount, 1)
      << "More than one call to needsToReserveAllocationSpace "
         "(multiple unrelated objects loaded prior to finalization)";
}

TEST(RTDyldObjectLinkingLayer2Test, TestNotifyLoadedSignature) {
  ExecutionSession ES(std::make_shared<SymbolStringPool>());
  RTDyldObjectLinkingLayer2 ObjLayer(
      ES,
      [](VModuleKey) -> std::shared_ptr<RuntimeDyld::MemoryManager> {
        return nullptr;
      },
      [](VModuleKey, const object::ObjectFile &obj,
         const RuntimeDyld::LoadedObjectInfo &info) {});
}

} // end anonymous namespace
