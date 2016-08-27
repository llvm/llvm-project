//===----- CompileOnDemandLayerTest.cpp - Unit tests for the COD layer ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "OrcTestCommon.h"
#include "llvm/ExecutionEngine/Orc/LogicalDylib.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {


TEST(LogicalDylibTest, getLogicalModuleResourcesForSymbol) {

  std::map<int, std::set<std::string>> ModuleSymbols;

  ModuleSymbols[0] = std::set<std::string>({ "foo", "dummy" });
  ModuleSymbols[1] = std::set<std::string>({ "bar" });
  ModuleSymbols[2] = std::set<std::string>({ "baz", "dummy" });

  auto MockBaseLayer = createMockBaseLayer<int>(
      DoNothingAndReturn<int>(0),
      DoNothingAndReturn<void>(),
      [&](const std::string &Name, bool) {
        for (auto &S : ModuleSymbols)
          if (S.second.count(Name))
            return JITSymbol(1, JITSymbolFlags::Exported);
        return JITSymbol(nullptr);
      },
      [&](int H, const std::string &Name, bool) {
        if (ModuleSymbols[H].count(Name))
          return JITSymbol(1, JITSymbolFlags::Exported);
        return JITSymbol(nullptr);
      });

  struct LDResources { };
  struct LMResources {
  public:
    int ID;
    std::set<std::string> *Symbols;

    LMResources() : ID(0), Symbols(nullptr) {}
    LMResources(int ID, std::set<std::string> &Symbols)
        : ID(ID), Symbols(&Symbols) {}

    JITSymbol findSymbol(const std::string &Name, bool) {
      assert(Symbols);
      if (Symbols->count(Name))
        return JITSymbol(ID, JITSymbolFlags::Exported);
      return JITSymbol(nullptr);
    }
  };

  LogicalDylib<decltype(MockBaseLayer), LMResources, LDResources>
    LD(MockBaseLayer);

  // Add logical module resources for each of our dummy modules.
  for (int I = 0; I < 3; ++I) {
    auto H = LD.createLogicalModule();
    LD.addToLogicalModule(H, I);
    LD.getLogicalModuleResources(H) = LMResources(I, ModuleSymbols[I]);
  }

  {
    auto LMR = LD.getLogicalModuleResourcesForSymbol("bar", true);
    EXPECT_TRUE(LMR->ID == 1) << "getLogicalModuleResourcesForSymbol failed";
  }
}
}
