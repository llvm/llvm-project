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

namespace llvm {

class BasicBlockMatchingAndInference : public MachineFunctionPass {
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

  StringRef getPassName() const override {
    return "Basic Block Matching and Inference";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &F) override;

  std::optional<WeightInfo> getWeightInfo(StringRef FuncName) const;

private:
  StringMap<WeightInfo> ProgramWeightInfo;

  WeightInfo initWeightInfoByMatching(MachineFunction &MF);

  void generateWeightInfoByInference(MachineFunction &MF,
                                     WeightInfo &MatchWeight);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_BASIC_BLOCK_AND_INFERENCE_H
