//===--- GlobalMappingLayerTest.cpp - Unit test the global mapping layer --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/GlobalMappingLayer.h"
#include "OrcTestCommon.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

TEST(GlobalMappingLayerTest, Empty) {
  MockBaseLayer<int, std::shared_ptr<Module>> TestBaseLayer;

  TestBaseLayer.addModuleImpl =
    [](std::shared_ptr<Module> M, std::shared_ptr<JITSymbolResolver> R) {
      return 42;
    };

  TestBaseLayer.findSymbolImpl =
    [](const std::string &Name, bool ExportedSymbolsOnly) -> JITSymbol {
      if (Name == "bar")
        return llvm::JITSymbol(0x4567, JITSymbolFlags::Exported);
      return nullptr;
    };

  GlobalMappingLayer<decltype(TestBaseLayer)> L(TestBaseLayer);

  // Test addModule interface.
  int H = cantFail(L.addModule(nullptr, nullptr));
  EXPECT_EQ(H, 42) << "Incorrect result from addModule";

  // Test fall-through for missing symbol.
  auto FooSym = L.findSymbol("foo", true);
  EXPECT_FALSE(FooSym) << "Found unexpected symbol.";

  // Test fall-through for symbol in base layer.
  auto BarSym = L.findSymbol("bar", true);
  EXPECT_EQ(cantFail(BarSym.getAddress()),
            static_cast<JITTargetAddress>(0x4567))
    << "Symbol lookup fall-through failed.";

  // Test setup of a global mapping.
  L.setGlobalMapping("foo", 0x0123);
  auto FooSym2 = L.findSymbol("foo", true);
  EXPECT_EQ(cantFail(FooSym2.getAddress()),
            static_cast<JITTargetAddress>(0x0123))
    << "Symbol mapping setup failed.";

  // Test removal of a global mapping.
  L.eraseGlobalMapping("foo");
  auto FooSym3 = L.findSymbol("foo", true);
  EXPECT_FALSE(FooSym3) << "Symbol mapping removal failed.";
}

}
