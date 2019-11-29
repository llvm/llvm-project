//===--- RTDyldObjectLinkingLayerTest.cpp - RTDyld linking layer tests ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
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

class RTDyldObjectLinkingLayerExecutionTest : public testing::Test,
                                               public OrcExecutionTest {};

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

  ExecutionSession ES;
  auto &JD = ES.createJITDylib("main");
  auto Foo = ES.intern("foo");

  RTDyldObjectLinkingLayer ObjLayer(ES, [&DebugSectionSeen]() {
    return std::make_unique<MemoryManagerWrapper>(DebugSectionSeen);
  });

  auto OnResolveDoNothing = [](Expected<SymbolMap> R) {
    cantFail(std::move(R));
  };

  ObjLayer.setProcessAllSections(ProcessAllSections);
  cantFail(ObjLayer.add(JD, std::move(Obj), ES.allocateVModule()));
  ES.lookup(LookupKind::Static, makeJITDylibSearchOrder(&JD),
            SymbolLookupSet(Foo), SymbolState::Resolved, OnResolveDoNothing,
            NoDependenciesToRegister);

  return DebugSectionSeen;
}

TEST(RTDyldObjectLinkingLayerTest, TestSetProcessAllSections) {
  LLVMContext Context;
  auto M = std::make_unique<Module>("", Context);
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

TEST(RTDyldObjectLinkingLayerTest, TestOverrideObjectFlags) {

  OrcNativeTarget::initialize();

  std::unique_ptr<TargetMachine> TM(
      EngineBuilder().selectTarget(Triple("x86_64-unknown-linux-gnu"), "", "",
                                   SmallVector<std::string, 1>()));

  if (!TM)
    return;

  // Our compiler is going to modify symbol visibility settings without telling
  // ORC. This will test our ability to override the flags later.
  class FunkySimpleCompiler : public SimpleCompiler {
  public:
    FunkySimpleCompiler(TargetMachine &TM) : SimpleCompiler(TM) {}

    CompileResult operator()(Module &M) {
      auto *Foo = M.getFunction("foo");
      assert(Foo && "Expected function Foo not found");
      Foo->setVisibility(GlobalValue::HiddenVisibility);
      return SimpleCompiler::operator()(M);
    }
  };

  // Create a module with two void() functions: foo and bar.
  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());
  ThreadSafeModule M;
  {
    ModuleBuilder MB(*TSCtx.getContext(), TM->getTargetTriple().str(), "dummy");
    MB.getModule()->setDataLayout(TM->createDataLayout());

    Function *FooImpl = MB.createFunctionDecl(
        FunctionType::get(Type::getVoidTy(*TSCtx.getContext()), {}, false),
        "foo");
    BasicBlock *FooEntry =
        BasicBlock::Create(*TSCtx.getContext(), "entry", FooImpl);
    IRBuilder<> B1(FooEntry);
    B1.CreateRetVoid();

    Function *BarImpl = MB.createFunctionDecl(
        FunctionType::get(Type::getVoidTy(*TSCtx.getContext()), {}, false),
        "bar");
    BasicBlock *BarEntry =
        BasicBlock::Create(*TSCtx.getContext(), "entry", BarImpl);
    IRBuilder<> B2(BarEntry);
    B2.CreateRetVoid();

    M = ThreadSafeModule(MB.takeModule(), std::move(TSCtx));
  }

  // Create a simple stack and set the override flags option.
  ExecutionSession ES;
  auto &JD = ES.createJITDylib("main");
  auto Foo = ES.intern("foo");
  RTDyldObjectLinkingLayer ObjLayer(
      ES, []() { return std::make_unique<SectionMemoryManager>(); });
  IRCompileLayer CompileLayer(ES, ObjLayer, FunkySimpleCompiler(*TM));

  ObjLayer.setOverrideObjectFlagsWithResponsibilityFlags(true);

  cantFail(CompileLayer.add(JD, std::move(M), ES.allocateVModule()));
  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD), SymbolLookupSet(Foo),
      SymbolState::Resolved,
      [](Expected<SymbolMap> R) { cantFail(std::move(R)); },
      NoDependenciesToRegister);
}

TEST(RTDyldObjectLinkingLayerTest, TestAutoClaimResponsibilityForSymbols) {

  OrcNativeTarget::initialize();

  std::unique_ptr<TargetMachine> TM(
      EngineBuilder().selectTarget(Triple("x86_64-unknown-linux-gnu"), "", "",
                                   SmallVector<std::string, 1>()));

  if (!TM)
    return;

  // Our compiler is going to add a new symbol without telling ORC.
  // This will test our ability to auto-claim responsibility later.
  class FunkySimpleCompiler : public SimpleCompiler {
  public:
    FunkySimpleCompiler(TargetMachine &TM) : SimpleCompiler(TM) {}

    CompileResult operator()(Module &M) {
      Function *BarImpl = Function::Create(
          FunctionType::get(Type::getVoidTy(M.getContext()), {}, false),
          GlobalValue::ExternalLinkage, "bar", &M);
      BasicBlock *BarEntry =
          BasicBlock::Create(M.getContext(), "entry", BarImpl);
      IRBuilder<> B(BarEntry);
      B.CreateRetVoid();

      return SimpleCompiler::operator()(M);
    }
  };

  // Create a module with two void() functions: foo and bar.
  ThreadSafeContext TSCtx(std::make_unique<LLVMContext>());
  ThreadSafeModule M;
  {
    ModuleBuilder MB(*TSCtx.getContext(), TM->getTargetTriple().str(), "dummy");
    MB.getModule()->setDataLayout(TM->createDataLayout());

    Function *FooImpl = MB.createFunctionDecl(
        FunctionType::get(Type::getVoidTy(*TSCtx.getContext()), {}, false),
        "foo");
    BasicBlock *FooEntry =
        BasicBlock::Create(*TSCtx.getContext(), "entry", FooImpl);
    IRBuilder<> B(FooEntry);
    B.CreateRetVoid();

    M = ThreadSafeModule(MB.takeModule(), std::move(TSCtx));
  }

  // Create a simple stack and set the override flags option.
  ExecutionSession ES;
  auto &JD = ES.createJITDylib("main");
  auto Foo = ES.intern("foo");
  RTDyldObjectLinkingLayer ObjLayer(
      ES, []() { return std::make_unique<SectionMemoryManager>(); });
  IRCompileLayer CompileLayer(ES, ObjLayer, FunkySimpleCompiler(*TM));

  ObjLayer.setAutoClaimResponsibilityForObjectSymbols(true);

  cantFail(CompileLayer.add(JD, std::move(M), ES.allocateVModule()));
  ES.lookup(
      LookupKind::Static, makeJITDylibSearchOrder(&JD), SymbolLookupSet(Foo),
      SymbolState::Resolved,
      [](Expected<SymbolMap> R) { cantFail(std::move(R)); },
      NoDependenciesToRegister);
}

} // end anonymous namespace
