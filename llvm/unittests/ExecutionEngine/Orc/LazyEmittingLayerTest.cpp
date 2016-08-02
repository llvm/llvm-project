//===- LazyEmittingLayerTest.cpp - Unit tests for the lazy emitting layer -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/Orc/LazyEmittingLayer.h"
#include "gtest/gtest.h"

namespace {

struct MockBaseLayer {
  typedef int ModuleSetHandleT;
  ModuleSetHandleT addModuleSet(
                  std::list<std::unique_ptr<llvm::Module>>,
                  std::unique_ptr<llvm::RuntimeDyld::MemoryManager> MemMgr,
                  std::unique_ptr<llvm::JITSymbolResolver> Resolver) {
    EXPECT_FALSE(MemMgr);
    return 42;
  }
};

TEST(LazyEmittingLayerTest, Empty) {
  MockBaseLayer M;
  llvm::orc::LazyEmittingLayer<MockBaseLayer> L(M);
  L.addModuleSet(std::list<std::unique_ptr<llvm::Module>>(), nullptr, nullptr);
}

}
