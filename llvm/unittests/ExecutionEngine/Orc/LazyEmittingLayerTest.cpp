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
  ModuleHandleT addModule(llvm::orc::VModuleKey,
                          std::shared_ptr<llvm::Module>) {
    return 42;
  }
};

TEST(LazyEmittingLayerTest, Empty) {
  MockBaseLayer M;
  llvm::orc::LazyEmittingLayer<MockBaseLayer> L(M);
  cantFail(
      L.addModule(llvm::orc::VModuleKey(), std::unique_ptr<llvm::Module>()));
}

}
