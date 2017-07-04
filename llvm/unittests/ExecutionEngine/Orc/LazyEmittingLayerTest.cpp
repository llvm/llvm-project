//===- LazyEmittingLayerTest.cpp - Unit tests for the lazy emitting layer -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LazyEmittingLayer.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "gtest/gtest.h"

namespace {

struct MockBaseLayer {
  typedef int ModuleHandleT;
  ModuleHandleT addModule(
                  std::shared_ptr<llvm::Module>,
                  std::unique_ptr<llvm::RuntimeDyld::MemoryManager> MemMgr,
                  std::unique_ptr<llvm::JITSymbolResolver> Resolver) {
    EXPECT_FALSE(MemMgr);
    return 42;
  }
};

TEST(LazyEmittingLayerTest, Empty) {
  MockBaseLayer M;
  llvm::orc::LazyEmittingLayer<MockBaseLayer> L(M);
  L.addModule(std::unique_ptr<llvm::Module>(), nullptr);
}

}
