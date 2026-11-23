//===- llvm/CodeGen/BasicBlockMatchingAndInference.h ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Infer weights for all basic blocks using matching and inference.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BASIC_BLOCK_AND_INFERENCE_H
#define LLVM_CODEGEN_BASIC_BLOCK_AND_INFERENCE_H

#include "llvm/CodeGen/BasicBlockSectionsProfileReader.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Transforms/Utils/SampleProfileInference.h"
#include <memory>

namespace llvm {

class Module;
namespace sampleprof {
class SampleProfileReader;
} // namespace sampleprof

class LLVM_ABI BasicBlockMatchingAndInference : public MachineFunctionPass {
private:
  using Edge = std::pair<const MachineBasicBlock *, const MachineBasicBlock *>;
  using BlockWeightMap = DenseMap<const MachineBasicBlock *, uint64_t>;
  using EdgeWeightMap = DenseMap<Edge, uint64_t>;
  using BlockEdgeMap = DenseMap<const MachineBasicBlock *,
                                SmallVector<const MachineBasicBlock *, 8>>;

  struct WeightInfo {
    // Weight of basic blocks.
    BlockWeightMap BlockWeights;
    // Weight of edges.
    EdgeWeightMap EdgeWeights;
  };

public:
  static char ID;
  BasicBlockMatchingAndInference();
  ~BasicBlockMatchingAndInference() override;

  StringRef getPassName() const override {
    return "Basic Block Matching and Inference";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool doInitialization(Module &M) override;
  bool runOnMachineFunction(MachineFunction &F) override;
  bool doFinalization(Module &M) override;

  std::optional<WeightInfo> getWeightInfo(StringRef FuncName) const;

private:
  StringMap<WeightInfo> ProgramWeightInfo;
  // The supplemental sample profile of the current module. Null if no
  // supplemental sample profile is given or it fails to load.
  std::unique_ptr<sampleprof::SampleProfileReader> SampleReader;

  void supplementWithSampleProfile(MachineFunction &MF,
                                   WeightInfo &MatchWeight);

  WeightInfo initWeightInfoByMatching(MachineFunction &MF);
  WeightInfo initWeightInfoByBBID(MachineFunction &MF, const CFGProfile &CFG);

  void generateWeightInfoByInference(MachineFunction &MF,
                                     WeightInfo &MatchWeight);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_BASIC_BLOCK_AND_INFERENCE_H
