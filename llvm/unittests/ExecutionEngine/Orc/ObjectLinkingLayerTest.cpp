//===-- ObjectLinkingLayerTest.cpp - Unit tests for object linking layer --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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

}
