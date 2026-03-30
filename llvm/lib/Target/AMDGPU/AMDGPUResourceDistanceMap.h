//===-- AMDGPUResourceDistanceMap.h - SU-to-resource-root distances --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Per-resource DAG analysis: for each processor resource, compute the
/// distance from each SUnit to the nearest "root" SUnit that consumes that
/// resource. Used by GCNPreRA/PostRACriticalResource to bias scheduling
/// toward instructions closer to a critical-resource consumer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPURESOURCEDISTANCEMAP_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPURESOURCEDISTANCEMAP_H

#include "llvm/ADT/DenseMap.h"

#include <functional>
#include <limits>
#include <queue>
#include <set>

namespace llvm {

class ScheduleDAGInstrs;

namespace AMDGPU {

class ResourceDistanceMaps {
public:
  struct SUDistRank {
    int OrderOfRoot;
    unsigned DistToRoot;
    SUDistRank(int OrderOfRoot,
               unsigned DistToRoot = std::numeric_limits<unsigned>::max())
        : OrderOfRoot(OrderOfRoot), DistToRoot(DistToRoot) {}
    bool operator<(const SUDistRank &Other) const {
      if (OrderOfRoot != Other.OrderOfRoot) {
        return OrderOfRoot < Other.OrderOfRoot;
      }
      return DistToRoot > Other.DistToRoot;
    }
    bool operator>(const SUDistRank &Other) const { return Other < *this; }
  };

private:
  using DistMapTy = DenseMap<SUnit *, DenseMap<SUnit *, unsigned>>;
  struct ResourceRootInfo {
    /// Number of root SU that is predecessor of this root.
    unsigned PredRootsLeft = 0;

    /// Sorted set of preds latency to this root.
    std::priority_queue<unsigned> NonRootPredLatency;

    /// Order relative to other roots, lower is more prioritized.
    int Order = std::numeric_limits<int>::max();

    unsigned getMaxLatency() const {
      if (NonRootPredLatency.empty()) {
        return 0;
      }
      return NonRootPredLatency.top();
    }
  };
  using RootInfosTy = DenseMap<SUnit *, ResourceRootInfo>;
  using RootInfosKV = RootInfosTy::value_type;
  struct RootInfoCompare {
    bool operator()(const RootInfosKV *LHS, const RootInfosKV *RHS) const {
      auto LHSLatency = LHS->second.getMaxLatency();
      auto RHSLatency = RHS->second.getMaxLatency();
      if (LHSLatency != RHSLatency) {
        return LHSLatency < RHSLatency;
      }
      return LHS->first->NodeNum < RHS->first->NodeNum;
    }
  };

  struct ResourceInfo {
    ScheduleDAGInstrs *DAG;
    DistMapTy DistMap;
    RootInfosTy RootInfos;
    std::set<RootInfosKV *, RootInfoCompare> CurrentRoots;
    DenseMap<SUnit *, SUDistRank> SURankCache;

    unsigned getOrderForRoot(SUnit *Root) const;
    void sortRoots();
    void schedNode(SUnit *SU, unsigned CurrCycle);
    SUDistRank getSURank(SUnit *SU);
    SUDistRank getSURankImpl(SUnit *SU);
    bool isRoot(SUnit *SU) const { return RootInfos.contains(SU); }

    ResourceInfo(ScheduleDAGInstrs *DAG) : DAG(DAG) {}
    void init(DistMapTy &&DistMap_, RootInfosTy &&RootInfos_);
  };

  mutable DenseMap<unsigned, ResourceInfo> Maps;
  friend class ResourceDistanceMapBuilder;
  static ResourceInfo build(ScheduleDAGInstrs *DAG, unsigned ResourceID);

  ResourceInfo &ensureResDistMap(unsigned ResourceID) const;

  ScheduleDAGInstrs *DAG = nullptr;

public:
  ResourceDistanceMaps() {}
  void initialize(ScheduleDAGInstrs *DAG) {
    this->DAG = DAG;
    Maps.clear();
  }
  SUDistRank getSUnitRankForRes(SUnit *SU, unsigned ResourceID) const;
  void schedNode(SUnit *SU, unsigned CurrCycle);
  bool isRoot(SUnit *SU) const {
    return llvm::any_of(Maps,
                        [SU](const auto &KV) { return KV.second.isRoot(SU); });
  }
};

} // namespace AMDGPU
} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPURESOURCEDISTANCEMAP_H
