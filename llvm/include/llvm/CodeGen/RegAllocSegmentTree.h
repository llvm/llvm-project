//===- RegAllocSegmentTree.h - RA segtree scaffold --------------*- C++ -*-===//
//
// This file declares createRegAllocSegmentTree() factory (scaffold only).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGALLOCSEGMENTTREE_H
#define LLVM_CODEGEN_REGALLOCSEGMENTTREE_H

namespace llvm {
class FunctionPass;

// Factory (for -regalloc=segtre). For now it delegates to Greedy (NFC).
FunctionPass *createRegAllocSegmentTree();
} // end namespace llvm

#endif