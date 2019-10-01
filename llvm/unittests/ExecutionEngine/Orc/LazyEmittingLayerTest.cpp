//===- LazyEmittingLayerTest.cpp - Unit tests for the lazy emitting layer -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
