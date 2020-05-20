//===--- AMDGPUExportClusting.cpp - AMDGPU Export Clustering  -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation to cluster shader
///       exports.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUExportClustering.h"
#include "AMDGPUSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"

using namespace llvm;

namespace {

class ExportClustering : public ScheduleDAGMutation {
public:
  ExportClustering() {}
  void apply(ScheduleDAGInstrs *DAG) override;
};

static bool isExport(const SUnit &SU) {
  const MachineInstr *MI = SU.getInstr();
  return MI->getOpcode() == AMDGPU::EXP ||
         MI->getOpcode() == AMDGPU::EXP_DONE;
}

static void buildCluster(ArrayRef<SUnit *> Exports, ScheduleDAGInstrs *DAG) {
  // Cluster a series of exports. Also copy all dependencies to the first
  // export to avoid computation being inserted into the chain.
  SUnit *ChainHead = Exports[0];
  for (unsigned Idx = 0, End = Exports.size() - 1; Idx < End; ++Idx) {
    SUnit *SUa = Exports[Idx];
    SUnit *SUb = Exports[Idx + 1];
    if (DAG->addEdge(SUb, SDep(SUa, SDep::Cluster))) {
      for (const SDep &Pred : SUb->Preds) {
        SUnit *PredSU = Pred.getSUnit();
        if (Pred.isWeak() || isExport(*PredSU))
          continue;
        DAG->addEdge(ChainHead, SDep(PredSU, SDep::Artificial));
      }
    }
  }
}

void ExportClustering::apply(ScheduleDAGInstrs *DAG) {
  SmallVector<SmallVector<SUnit *, 8>, 4> ExportChains;
  DenseMap<unsigned, unsigned> ChainMap;

  // Build chains of exports
  for (SUnit &SU : DAG->SUnits) {
    if (!isExport(SU))
      continue;

    unsigned ChainID = ExportChains.size();
    for (const SDep &Pred : SU.Preds) {
      const SUnit &PredSU = *Pred.getSUnit();
      if (isExport(PredSU) && !Pred.isArtificial()) {
        ChainID = ChainMap.lookup(PredSU.NodeNum);
        break;
      }
    }
    ChainMap[SU.NodeNum] = ChainID;

    if (ChainID == ExportChains.size())
      ExportChains.push_back(SmallVector<SUnit *, 8>());

    auto &Chain = ExportChains[ChainID];
    Chain.push_back(&SU);
  }

  // Apply clustering
  for (auto &Chain : ExportChains)
    buildCluster(Chain, DAG);
}

} // end namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createAMDGPUExportClusteringDAGMutation() {
  return std::make_unique<ExportClustering>();
}

} // end namespace llvm
