//===-- WebAssemblyExceptionInfo.h - WebAssembly Exception Info -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements WebAssemblyException information analysis.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYEXCEPTIONINFO_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYEXCEPTIONINFO_H

#include "WebAssembly.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {

class MachineDominatorTree;
class MachineDominanceFrontier;

// WebAssembly instructions for exception handling are structured as follows:
//   try
//     instructions*
//   catch             ----|
//     instructions*       | -> A WebAssemblyException consists of this region
//   end               ----|
//
// A WebAssemblyException object contains BBs that belong to a 'catch' part of
// the try-catch-end structure to be created later. 'try' and 'end' markers
// are not present at this stage and will be generated in CFGStackify pass.
// Because CFGSort requires all the BBs within a catch part to be sorted
// together as it does for loops, this pass calculates the nesting structure of
// catch part of exceptions in a function.
//
// An exception catch part is defined as a BB with catch instruction and all
// other BBs dominated by this BB.
class WebAssemblyException {
  MachineBasicBlock *EHPad = nullptr;

  WebAssemblyException *ParentException = nullptr;
  std::vector<std::unique_ptr<WebAssemblyException>> SubExceptions;
  std::vector<MachineBasicBlock *> Blocks;
  SmallPtrSet<MachineBasicBlock *, 8> BlockSet;

public:
  WebAssemblyException(MachineBasicBlock *EHPad) : EHPad(EHPad) {}
  WebAssemblyException(const WebAssemblyException &) = delete;
  const WebAssemblyException &operator=(const WebAssemblyException &) = delete;

  MachineBasicBlock *getEHPad() const { return EHPad; }
  MachineBasicBlock *getHeader() const { return EHPad; }
  WebAssemblyException *getParentException() const { return ParentException; }
  void setParentException(WebAssemblyException *WE) { ParentException = WE; }

  bool contains(const WebAssemblyException *WE) const {
    if (WE == this)
      return true;
    if (!WE)
      return false;
    return contains(WE->getParentException());
  }
  bool contains(const MachineBasicBlock *MBB) const {
    return BlockSet.count(MBB);
  }

  void addToBlocksSet(MachineBasicBlock *MBB) { BlockSet.insert(MBB); }
  void removeFromBlocksSet(MachineBasicBlock *MBB) { BlockSet.erase(MBB); }
  void addToBlocksVector(MachineBasicBlock *MBB) { Blocks.push_back(MBB); }
  void addBlock(MachineBasicBlock *MBB) {
    Blocks.push_back(MBB);
    BlockSet.insert(MBB);
  }
  ArrayRef<MachineBasicBlock *> getBlocks() const { return Blocks; }
  using block_iterator = ArrayRef<MachineBasicBlock *>::const_iterator;
  block_iterator block_begin() const { return getBlocks().begin(); }
  block_iterator block_end() const { return getBlocks().end(); }
  inline iterator_range<block_iterator> blocks() const {
    return make_range(block_begin(), block_end());
  }
  unsigned getNumBlocks() const { return Blocks.size(); }
  std::vector<MachineBasicBlock *> &getBlocksVector() { return Blocks; }
  SmallPtrSetImpl<MachineBasicBlock *> &getBlocksSet() { return BlockSet; }

  const std::vector<std::unique_ptr<WebAssemblyException>> &
  getSubExceptions() const {
    return SubExceptions;
  }
  std::vector<std::unique_ptr<WebAssemblyException>> &getSubExceptions() {
    return SubExceptions;
  }
  void addSubException(std::unique_ptr<WebAssemblyException> E) {
    SubExceptions.push_back(std::move(E));
  }
  using iterator = decltype(SubExceptions)::const_iterator;
  iterator begin() const { return SubExceptions.begin(); }
  iterator end() const { return SubExceptions.end(); }

  void reserveBlocks(unsigned Size) { Blocks.reserve(Size); }
  void reverseBlock(unsigned From = 0) {
    std::reverse(Blocks.begin() + From, Blocks.end());
  }

  // Return the nesting level. An outermost one has depth 1.
  unsigned getExceptionDepth() const {
    unsigned D = 1;
    for (const WebAssemblyException *CurException = ParentException;
         CurException; CurException = CurException->ParentException)
      ++D;
    return D;
  }

  void print(raw_ostream &OS, unsigned Depth = 0) const;
  void dump() const;
};

raw_ostream &operator<<(raw_ostream &OS, const WebAssemblyException &WE);

class WebAssemblyExceptionInfo {
  // Mapping of basic blocks to the innermost exception they occur in
  DenseMap<const MachineBasicBlock *, WebAssemblyException *> BBMap;
  std::vector<std::unique_ptr<WebAssemblyException>> TopLevelExceptions;

  void discoverAndMapException(WebAssemblyException *WE,
                               const MachineDominatorTree &MDT,
                               const MachineDominanceFrontier &MDF);
  WebAssemblyException *getOutermostException(MachineBasicBlock *MBB) const;

public:
  WebAssemblyExceptionInfo() {}
  ~WebAssemblyExceptionInfo() { releaseMemory(); }
  WebAssemblyExceptionInfo(const WebAssemblyExceptionInfo &) = delete;
  WebAssemblyExceptionInfo(WebAssemblyExceptionInfo &&) = default;
  WebAssemblyExceptionInfo &
  operator=(const WebAssemblyExceptionInfo &) = delete;

  void releaseMemory();
  void recalculate(MachineFunction &MF, MachineDominatorTree &MDT,
                   const MachineDominanceFrontier &MDF);

  bool empty() const { return TopLevelExceptions.empty(); }

  // Return the innermost exception that MBB lives in. If the block is not in an
  // exception, null is returned.
  WebAssemblyException *getExceptionFor(const MachineBasicBlock *MBB) const {
    return BBMap.lookup(MBB);
  }

  void changeExceptionFor(const MachineBasicBlock *MBB,
                          WebAssemblyException *WE) {
    if (!WE) {
      BBMap.erase(MBB);
      return;
    }
    BBMap[MBB] = WE;
  }

  void addTopLevelException(std::unique_ptr<WebAssemblyException> WE) {
    assert(!WE->getParentException() && "Not a top level exception!");
    TopLevelExceptions.push_back(std::move(WE));
  }

  void print(raw_ostream &OS, const Module *M) const;

  bool invalidate(MachineFunction &MF, const PreservedAnalyses &PA,
                  MachineFunctionAnalysisManager::Invalidator &);
};

class WebAssemblyExceptionInfoWrapperPass : public MachineFunctionPass {
  WebAssemblyExceptionInfo WasmExceptionInfo;

public:
  static char ID;
  WebAssemblyExceptionInfoWrapperPass() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  void print(raw_ostream &OS, const Module *M = nullptr) const override {
    WasmExceptionInfo.print(OS, M);
  }
  void releaseMemory() override { WasmExceptionInfo.releaseMemory(); }

  WebAssemblyExceptionInfo &getWEI() { return WasmExceptionInfo; }
  const WebAssemblyExceptionInfo &getWEI() const { return WasmExceptionInfo; }
};

class WebAssemblyExceptionAnalysis
    : public AnalysisInfoMixin<WebAssemblyExceptionAnalysis> {
  friend AnalysisInfoMixin<WebAssemblyExceptionAnalysis>;
  static AnalysisKey Key;

public:
  using Result = WebAssemblyExceptionInfo;

  LLVM_ABI Result run(MachineFunction &MF,
                      MachineFunctionAnalysisManager &MFAM);
};

} // end namespace llvm

#endif
