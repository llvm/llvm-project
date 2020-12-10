//===- Transforms/IPO/SampleProfileProbe.h ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file provides the interface for the pseudo probe implementation for
/// AutoFDO.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_SAMPLEPROFILEPROBE_H
#define LLVM_TRANSFORMS_IPO_SAMPLEPROFILEPROBE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PseudoProbe.h"
#include "llvm/Target/TargetMachine.h"
#include <unordered_map>

namespace llvm {

class Module;

using BlockIdMap = std::unordered_map<BasicBlock *, uint32_t>;
using InstructionIdMap = std::unordered_map<Instruction *, uint32_t>;

enum class PseudoProbeReservedId { Invalid = 0, Last = Invalid };


/// Sample profile pseudo prober.
///
/// Insert pseudo probes for block sampling and value sampling.
class SampleProfileProber {
public:
  // Give an empty module id when the prober is not used for instrumentation.
  SampleProfileProber(Function &F);
  void instrumentOneFunc(Function &F, TargetMachine *TM);

private:
  Function *getFunction() const { return F; }
  uint32_t getBlockId(const BasicBlock *BB) const;
  uint32_t getCallsiteId(const Instruction *Call) const;
  void computeProbeIdForBlocks();
  void computeProbeIdForCallsites();

  Function *F;

  /// Map basic blocks to the their pseudo probe ids.
  BlockIdMap BlockProbeIds;

  /// Map indirect calls to the their pseudo probe ids.
  InstructionIdMap CallProbeIds;

  /// The ID of the last probe, Can be used to number a new probe.
  uint32_t LastProbeId;
};

class SampleProfileProbePass : public PassInfoMixin<SampleProfileProbePass> {
  TargetMachine *TM;

public:
  SampleProfileProbePass(TargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm
#endif // LLVM_TRANSFORMS_IPO_SAMPLEPROFILEPROBE_H
