//===--- GlobalMappingLayerTest.cpp - Unit test the global mapping layer --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/GlobalMappingLayer.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

struct MockBaseLayer {

  typedef int ModuleHandleT;

  JITSymbol findSymbol(const std::string &Name, bool ExportedSymbolsOnly) {
    if (Name == "bar")
      return llvm::JITSymbol(0x4567, JITSymbolFlags::Exported);
    return nullptr;
  }

};

TEST(GlobalMappingLayerTest, Empty) {
  MockBaseLayer M;
  GlobalMappingLayer<MockBaseLayer> L(M);

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
