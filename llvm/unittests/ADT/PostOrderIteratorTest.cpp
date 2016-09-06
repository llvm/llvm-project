//===- PostOrderIteratorTest.cpp - PostOrderIterator unit tests -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "gtest/gtest.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
using namespace llvm;

namespace {

// Whether we're able to compile
TEST(PostOrderIteratorTest, Compiles) {
  typedef SmallPtrSet<void *, 4> ExtSetTy;

  // Tests that template specializations are kept up to date
  void *Null = nullptr;
  po_iterator_storage<std::set<void *>, false> PIS;
  PIS.insertEdge(Optional<void *>(), Null);
  ExtSetTy Ext;
  po_iterator_storage<ExtSetTy, true> PISExt(Ext);
  PIS.insertEdge(Optional<void *>(), Null);

  // Test above, but going through po_iterator (which inherits from template
  // base)
  BasicBlock *NullBB = nullptr;
  auto PI = po_end(NullBB);
  PI.insertEdge(Optional<BasicBlock *>(), NullBB);
  auto PIExt = po_ext_end(NullBB, Ext);
  PIExt.insertEdge(Optional<BasicBlock *>(), NullBB);
}
}
