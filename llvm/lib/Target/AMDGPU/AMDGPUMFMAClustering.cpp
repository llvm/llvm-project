//===--- AMDGPUMFMAClusting.cpp - AMDGPU MFMA Clustering  -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains a DAG scheduling mutation to cluster MFMA
///      instructions.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMFMAClustering.h"
#include "AMDGPUTargetMachine.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-mfma-clustering"

namespace {

static cl::opt<bool> EnableMFMACluster("amdgpu-mfma-cluster",
                                       cl::desc("Enable MFMA clustering"),
                                       cl::init(false));

static cl::opt<unsigned>
    MaxMFMAClusterSize("amdgpu-mfma-cluster-size", cl::init(5), cl::Hidden,
                       cl::desc("The maximum number of MFMA instructions to "
                                "attempt to cluster together."));

class MFMAClusterDAGMutation : public ScheduleDAGMutation {
  const SIInstrInfo *TII;
  ScheduleDAGMI *DAG;

public:
  MFMAClusterDAGMutation() = default;
  void apply(ScheduleDAGInstrs *DAGInstrs) override;
};

static void collectMFMASUnits(SmallVectorImpl<SUnit *> &MFMASUnits,
                              const SIInstrInfo *TII, ScheduleDAGInstrs *DAG) {
  for (SUnit &SU : DAG->SUnits) {
    MachineInstr &MAI = *SU.getInstr();
    if (!TII->isMAI(MAI) ||
        MAI.getOpcode() == AMDGPU::V_ACCVGPR_WRITE_B32_e64 ||
        MAI.getOpcode() == AMDGPU::V_ACCVGPR_READ_B32_e64)
      continue;

    MFMASUnits.push_back(&SU);

    LLVM_DEBUG(dbgs() << "Found MFMA: "; DAG->dumpNode(SU););
  }

  // Sorting the MFMAs in NodeNum order results in a good clustering order
  std::sort(MFMASUnits.begin(), MFMASUnits.end(),
            [](SUnit *a, SUnit *b) { return a->NodeNum < b->NodeNum; });
}

static void propagateDeps(DenseMap<unsigned, unsigned> &SUnit2ClusterInfo,
                          llvm::ArrayRef<SDep> ClusterPreds,
                          llvm::ArrayRef<SDep> ClusterSuccs,
                          unsigned ClusterNum, ScheduleDAGInstrs *DAG) {

  for (auto Node : SUnit2ClusterInfo) {
    if (Node.second != ClusterNum)
      continue; // Only add the combined succs to the current cluster

    LLVM_DEBUG(dbgs() << "Copying Deps To SU(" << Node.first << ")\n");

    for (const SDep &Succ : ClusterSuccs) {
      LLVM_DEBUG(dbgs() << "Copying Succ SU(" << Succ.getSUnit()->NodeNum
                        << ")\n");
      DAG->addEdge(Succ.getSUnit(),
                   SDep(&DAG->SUnits[Node.first], SDep::Artificial));
    }

    for (const SDep &Pred : ClusterPreds) {
      LLVM_DEBUG(dbgs() << "Copying Pred SU(" << Pred.getSUnit()->NodeNum
                        << ")\n");
      if (Pred.getSUnit()->NodeNum == ClusterNum)
        continue;
      DAG->addEdge(&DAG->SUnits[Node.first],
                   SDep(Pred.getSUnit(), SDep::Artificial));
    }
  }
}

static void clusterNeighboringMFMAs(llvm::ArrayRef<SUnit *> MFMASUnits,
                                    ScheduleDAGInstrs *DAG) {

  DenseMap<unsigned, unsigned> SUnit2ClusterInfo;

  for (unsigned Idx = 0, End = MFMASUnits.size(); Idx < (End - 1); ++Idx) {
    if (SUnit2ClusterInfo.count(MFMASUnits[Idx]->NodeNum))
      continue; // We don't want to cluster against a different cluster

    auto MFMAOpa = MFMASUnits[Idx];
    auto ClusterBase = MFMAOpa;
    unsigned ClusterNum = ClusterBase->NodeNum;
    SmallVector<SDep, 4> ClusterSuccs(MFMAOpa->Succs);
    SmallVector<SDep, 4> ClusterPreds(MFMAOpa->Preds);
    unsigned NextIdx = Idx + 1;
    unsigned ClusterSize = 1;

    // Attempt to cluster all the remaining MFMASunits in a chain
    // starting at ClusterBase/MFMAOpa.
    for (; NextIdx < End; ++NextIdx) {
      if (ClusterSize >= MaxMFMAClusterSize || NextIdx >= End)
        break;
      // Only add independent MFMAs that have not been previously clustered
      if (SUnit2ClusterInfo.count(MFMASUnits[NextIdx]->NodeNum) ||
          DAG->IsReachable(MFMASUnits[NextIdx], ClusterBase) ||
          DAG->IsReachable(ClusterBase, MFMASUnits[NextIdx]))
        continue;

      auto MFMAOpb = MFMASUnits[NextIdx];
      // Aggregate the cluster inst dependencies for dep propogation
      ClusterPreds.append(MFMAOpb->Preds);
      ClusterSuccs.append(MFMAOpb->Succs);
      if (!DAG->addEdge(MFMAOpb, SDep(MFMAOpa, SDep::Cluster)))
        continue;

      // Enforce ordering to ensure root/leaf of cluster chain gets
      // scheduled first/last
      DAG->addEdge(MFMAOpb, SDep(MFMAOpa, SDep::Artificial));

      LLVM_DEBUG(dbgs() << "Cluster MFMA SU(" << MFMAOpa->NodeNum << ") - SU("
                        << MFMAOpb->NodeNum << ")\n");

      SUnit2ClusterInfo[MFMAOpb->NodeNum] = ClusterNum;
      SUnit2ClusterInfo[MFMAOpa->NodeNum] = ClusterNum;
      ++ClusterSize;
      MFMAOpa = MFMAOpb;
    }
    propagateDeps(SUnit2ClusterInfo, ClusterPreds, ClusterSuccs, ClusterNum,
                  DAG);
  }
}

void MFMAClusterDAGMutation::apply(ScheduleDAGInstrs *DAGInstrs) {
  const GCNSubtarget &ST = DAGInstrs->MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  const SIMachineFunctionInfo *MFI =
      DAGInstrs->MF.getInfo<SIMachineFunctionInfo>();
  if (!ST.hasMAIInsts())
    return;
  DAG = static_cast<ScheduleDAGMI *>(DAGInstrs);
  const TargetSchedModel *TSchedModel = DAGInstrs->getSchedModel();
  if (!TSchedModel || DAG->SUnits.empty())
    return;

  SmallVector<SUnit *, 32> MFMASUnits;
  collectMFMASUnits(MFMASUnits, TII, DAG);

  if (MFMASUnits.size() < 2)
    return;

  clusterNeighboringMFMAs(MFMASUnits, DAG);
}

} // namespace

namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createMFMAClusterDAGMutation() {
  return EnableMFMACluster ? std::make_unique<MFMAClusterDAGMutation>()
                           : nullptr;
}

} // end namespace llvm
