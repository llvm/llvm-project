//===- RegAllocSegmentTree.cpp - RA segtree scaffold ----------------------===//
//
// Scaffold only: register -regalloc=segmenttree that delegates to Greedy (NFC).
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegAllocSegmentTree.h"
#include "llvm/CodeGen/RegAllocRegistry.h"

using namespace llvm;

// 工廠：目前直接委派回 Greedy（零行為變更）
FunctionPass *llvm::createRegAllocSegmentTree() {
  extern FunctionPass *createGreedyRegisterAllocator();
  return createGreedyRegisterAllocator();
}

// 把選項掛進 -regalloc= 名單（名稱、描述、工廠）
static RegisterRegAlloc
    RAReg("segmenttree", "Segment Tree Register Allocator (scaffold)",
          createRegAllocSegmentTree);