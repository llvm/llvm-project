//===- LiveStacks.h - Live Stack Slot Analysis ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the live stack slot analysis pass. It is analogous to
// live interval analysis except it's analyzing liveness of stack slots rather
// than registers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_LIVESTACKS_H
#define LLVM_CODEGEN_LIVESTACKS_H

#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include <cassert>
#include <map>
#include <unordered_map>

namespace llvm {

class AnalysisUsage;
class MachineFunction;
class Module;
class raw_ostream;
class TargetRegisterClass;
class TargetRegisterInfo;

class LiveStacks {
  const TargetRegisterInfo *TRI = nullptr;

  /// Special pool allocator for VNInfo's (LiveInterval val#).
  ///
  VNInfo::Allocator VNInfoAllocator;

  int StartIdx = -1;
  SmallVector<LiveInterval *> S2LI;
  SmallVector<const TargetRegisterClass *> S2RC;

public:
  using iterator = SmallVector<LiveInterval *>::iterator;
  using const_iterator = SmallVector<LiveInterval *>::const_iterator;

  const_iterator begin() const { return S2LI.begin(); }
  const_iterator end() const { return S2LI.end(); }
  iterator begin() { return S2LI.begin(); }
  iterator end() { return S2LI.end(); }

  unsigned getStartIdx() const { return StartIdx; }
  unsigned getNumIntervals() const { return (unsigned)S2LI.size(); }

  LiveInterval &getOrCreateInterval(int Slot, const TargetRegisterClass *RC);

  LiveInterval &getInterval(int Slot) {
    assert(Slot >= 0 && "Spill slot indice must be >= 0");
    return *S2LI[Slot - StartIdx];
  }

  const LiveInterval &getInterval(int Slot) const {
    assert(Slot >= 0 && "Spill slot indice must be >= 0");
    return *S2LI[Slot - StartIdx];
  }

  bool hasInterval(int Slot) const {
    if (Slot < StartIdx || StartIdx == -1)
      return false;
    return !getInterval(Slot).empty();
  }

  const TargetRegisterClass *getIntervalRegClass(int Slot) const {
    assert(Slot >= 0 && "Spill slot indice must be >= 0");
    return S2RC[Slot - StartIdx];
  }

  VNInfo::Allocator &getVNInfoAllocator() { return VNInfoAllocator; }

  void releaseMemory();
  /// init - analysis entry point
  void init(MachineFunction &MF);
  void print(raw_ostream &O, const Module *M = nullptr) const;
};

class LiveStacksWrapperLegacy : public MachineFunctionPass {
  LiveStacks Impl;

public:
  static char ID; // Pass identification, replacement for typeid

  LiveStacksWrapperLegacy() : MachineFunctionPass(ID) {}

  LiveStacks &getLS() { return Impl; }
  const LiveStacks &getLS() const { return Impl; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;

  /// runOnMachineFunction - pass entry point
  bool runOnMachineFunction(MachineFunction &) override;

  /// print - Implement the dump method.
  void print(raw_ostream &O, const Module * = nullptr) const override;
};

class LiveStacksAnalysis : public AnalysisInfoMixin<LiveStacksAnalysis> {
  static AnalysisKey Key;
  friend AnalysisInfoMixin<LiveStacksAnalysis>;

public:
  using Result = LiveStacks;

  LiveStacks run(MachineFunction &MF, MachineFunctionAnalysisManager &);
};

class LiveStacksPrinterPass : public PassInfoMixin<LiveStacksPrinterPass> {
  raw_ostream &OS;

public:
  LiveStacksPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &AM);
};
} // end namespace llvm

#endif
