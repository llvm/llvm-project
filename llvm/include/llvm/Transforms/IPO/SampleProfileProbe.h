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
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Target/TargetMachine.h"
#include <unordered_map>

namespace llvm {

class Module;

using namespace sampleprof;
using BlockIdMap = std::unordered_map<BasicBlock *, uint32_t>;
using InstructionIdMap = std::unordered_map<Instruction *, uint32_t>;

enum class PseudoProbeReservedId { Invalid = 0, Last = Invalid };

class PseudoProbeDescriptor {
  uint64_t FunctionGUID;
  uint64_t FunctionHash;

public:
  PseudoProbeDescriptor(uint64_t GUID, uint64_t Hash)
      : FunctionGUID(GUID), FunctionHash(Hash) {}
  uint64_t getFunctionGUID() const { return FunctionGUID; }
  uint64_t getFunctionHash() const { return FunctionHash; }
};

// This class serves sample counts correlation for SampleProfileLoader by
// analyzing pseudo probes and their function descriptors injected by
// SampleProfileProber.
class PseudoProbeManager {
  DenseMap<uint64_t, PseudoProbeDescriptor> GUIDToProbeDescMap;

  const PseudoProbeDescriptor *getDesc(const Function &F) const;

public:
  PseudoProbeManager(const Module &M);
  bool moduleIsProbed(const Module &M) const;
  bool profileIsValid(const Function &F, const FunctionSamples &Samples) const;
};

/// Sample profile pseudo prober.
///
/// Insert pseudo probes for block sampling and value sampling.
class SampleProfileProber {
public:
  // Give an empty module id when the prober is not used for instrumentation.
  SampleProfileProber(Function &F, const std::string &CurModuleUniqueId);
  void instrumentOneFunc(Function &F, TargetMachine *TM);

private:
  Function *getFunction() const { return F; }
  uint64_t getFunctionHash() const { return FunctionHash; }
  uint32_t getBlockId(const BasicBlock *BB) const;
  uint32_t getCallsiteId(const Instruction *Call) const;
  void computeCFGHash();
  void computeProbeIdForBlocks();
  void computeProbeIdForCallsites();

  Function *F;

  /// The current module ID that is used to name a static object as a comdat
  /// group.
  std::string CurModuleUniqueId;

  /// A CFG hash code used to identify a function code changes.
  uint64_t FunctionHash;

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
