//===-- AGXMemCluster.cpp - Keep runs of AGX memory accesses together -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A scheduling DAG mutation that clusters neighbouring memory accesses of one
// kind (device load, device store, threadgroup load, threadgroup store).
//
// A memory instruction issued shortly after an ALU instruction waits for the
// ALU pipeline to drain (AGX Performance Simulator, measured: ~34 cycles
// before a device store, ~20 before a device load), while back-to-back
// accesses pay it once: four device stores after their values cost 37 + 3 +
// 3 + 3 cycles, the same stores each after an ALU instruction 4 x 37. The
// generic load / store clustering needs base + offset operands; AGX addresses
// are a base and a register index, so this clusters by program order instead.
//
// Device loads and stores and threadgroup stores are clustered by default,
// in runs of up to 16. In agx-bench (agx_cost) stores alone lowered the ZLUDA
// PTX tests by 0.8% and agx_libs by 9.5%; the real-world CUDA kernels
// (kernels/cuda_*.ll) showed what device loads add: decode attention -34%,
// RGBA8 blur -28%, q4_0 / q8_0 mat-vec -10% / -5%, 3.8% in total, with only
// index_add on G13 slower (+17%). Runs of 4 or 6 loads lose (+0.2% to +0.7%):
// the attention loop's 8 loads must stay together. Threadgroup loads stay
// apart: they speed up tiled SGEMM and n-body but take SGEMM from 70 to 77
// GPRs (occupancy, which agx_cost does not model) and slow the f64 dot
// product by 33%.
//
//===----------------------------------------------------------------------===//

#include "AGX.h"
#include "AGXInstrInfo.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<unsigned> AGXMemClusterSize(
    "agx-mem-cluster", cl::Hidden, cl::init(16),
    cl::desc("Most AGX memory accesses of one kind the machine scheduler "
             "keeps together (0 or 1: off)"));

static cl::opt<unsigned> AGXMemClusterKinds(
    "agx-mem-cluster-kinds", cl::Hidden, cl::init(11),
    cl::desc("Mask of the AGX memory access kinds to cluster (1 device load, "
             "2 device store, 4 threadgroup load, 8 threadgroup store)"));

namespace {

enum MemKind { DeviceLoad, DeviceStore, LocalLoad, LocalStore, NumKinds };

int memKind(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AGX::DEVICE_LOAD_16:
  case AGX::DEVICE_LOAD_32:
  case AGX::DEVICE_LOAD_64:
  case AGX::DEVICE_LOAD_96:
  case AGX::DEVICE_LOAD_128:
  case AGX::DEVICE_LOAD_32_H16:
  case AGX::DEVICE_LOAD_64_H16:
    return DeviceLoad;
  case AGX::DEVICE_STORE_16:
  case AGX::DEVICE_STORE_32:
  case AGX::DEVICE_STORE_64:
  case AGX::DEVICE_STORE_96:
  case AGX::DEVICE_STORE_128:
  case AGX::DEVICE_STORE_32_H16:
  case AGX::DEVICE_STORE_64_H16:
    return DeviceStore;
  case AGX::LOCAL_LOAD_16:
  case AGX::LOCAL_LOAD_32:
  case AGX::LOCAL_LOAD_64:
  case AGX::LOCAL_LOAD_96:
  case AGX::LOCAL_LOAD_128:
    return LocalLoad;
  case AGX::LOCAL_STORE_16:
  case AGX::LOCAL_STORE_32:
  case AGX::LOCAL_STORE_64:
  case AGX::LOCAL_STORE_96:
  case AGX::LOCAL_STORE_128:
    return LocalStore;
  default:
    return -1;
  }
}

class AGXMemClusterMutation : public ScheduleDAGMutation {
public:
  void apply(ScheduleDAGInstrs *DAG) override;
};

void AGXMemClusterMutation::apply(ScheduleDAGInstrs *DAG) {
  if (AGXMemClusterSize < 2)
    return;
  EquivalenceClasses<SUnit *> Clusters;
  SmallVector<SUnit *, 16> Run[NumKinds];
  for (SUnit &SU : DAG->SUnits) {
    int K = memKind(*SU.getInstr());
    if (K < 0 || !(AGXMemClusterKinds & (1u << K)))
      continue;
    SmallVector<SUnit *, 16> &R = Run[K];
    bool Load = K == DeviceLoad || K == LocalLoad;
    // A data dependence on the run (an address from a load) leaves work that
    // must sit in between.
    bool Data = false;
    for (const SDep &D : SU.Preds)
      if (D.getKind() == SDep::Data)
        for (SUnit *M : R)
          if (D.getSUnit() == M || DAG->IsReachable(D.getSUnit(), M))
            Data = true;
    if (R.empty() || R.size() >= AGXMemClusterSize || Data ||
        !DAG->addEdge(&SU, SDep(R.back(), SDep::Cluster))) {
      R.assign(1, &SU);
      continue;
    }
    Clusters.unionSets(R.back(), &SU);
    // Every member, not only the previous one: a post-RA top-down scheduler
    // would otherwise issue the first stores as soon as their own values are
    // ready, between the computations of the later ones.
    for (SUnit *M : R) {
      if (Load) {
        // The users of the earlier loads wait for the later one.
        for (const SDep &D : M->Succs)
          if (!is_contained(R, D.getSUnit()) && D.getSUnit() != &SU)
            DAG->addEdge(D.getSUnit(), SDep(&SU, SDep::Artificial));
      } else {
        // What the later store needs is ready before the earlier ones.
        for (const SDep &D : SU.Preds)
          if (!is_contained(R, D.getSUnit()))
            DAG->addEdge(M, SDep(D.getSUnit(), SDep::Artificial));
      }
    }
    R.push_back(&SU);
  }

  auto &All = DAG->getClusters();
  for (const EquivalenceClasses<SUnit *>::ECValue *I : Clusters) {
    if (!I->isLeader())
      continue;
    ClusterInfo Group;
    unsigned Idx = All.size();
    for (SUnit *Member : Clusters.members(*I)) {
      Member->ParentClusterIdx = Idx;
      Group.insert(Member);
    }
    All.push_back(Group);
  }
}

} // namespace

std::unique_ptr<ScheduleDAGMutation> llvm::createAGXMemClusterMutation() {
  return std::make_unique<AGXMemClusterMutation>();
}
