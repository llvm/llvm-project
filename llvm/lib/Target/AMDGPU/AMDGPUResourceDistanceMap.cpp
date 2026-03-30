//===-- AMDGPUResourceDistanceMap.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUResourceDistanceMap.h"

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/ScheduleDAGInstrs.h"

#include <deque>
#include <limits>

namespace llvm {
namespace AMDGPU {
// TODO: Support BottomUp
class ResourceDistanceMapBuilder {
  DenseMap<SUnit *, unsigned> SuccsLeft;
  ResourceDistanceMaps::DistMapTy DistMap;
  ResourceDistanceMaps::RootInfosTy RootInfos;

  ScheduleDAGInstrs *DAG;
  unsigned ResourceID;

  bool isRoot(const SUnit *SU) const { return RootInfos.contains(SU); }

  static bool shouldPruneEdge(const SDep &Edge) {
    return Edge.isArtificial() || Edge.isWeak();
  }

  void getReachableNodes() {
    auto *SchedModel = DAG->getSchedModel();
    SmallVector<SUnit *> WorkList;
    for (auto *SU : nodes(static_cast<ScheduleDAG *>(DAG)))
      if (auto *SC = DAG->getSchedClass(SU))
        for (auto &ProcRes : make_range(SchedModel->getWriteProcResBegin(SC),
                                        SchedModel->getWriteProcResEnd(SC)))
          if (ProcRes.ProcResourceIdx == ResourceID) {
            RootInfos.try_emplace(SU, ResourceDistanceMaps::ResourceRootInfo{});
            WorkList.push_back(SU);
            break;
          }
    while (!WorkList.empty()) {
      auto *SU = WorkList.pop_back_val();
      for (auto &Pred : SU->Preds) {
        if (shouldPruneEdge(Pred)) {
          continue;
        }
        auto *PredSU = Pred.getSUnit();
        auto [Iter, Inserted] = SuccsLeft.try_emplace(PredSU, 0);
        if (Inserted && !isRoot(PredSU))
          WorkList.push_back(PredSU);
        Iter->second += 1;
      }
    }
  }

  void buildDistanceMap() {
    std::deque<SUnit *> WorkList;
    for (auto &[Root, _] : RootInfos) {
      WorkList.push_back(Root);
    }
    while (!WorkList.empty()) {
      auto *SU = WorkList.front();
      WorkList.pop_front();
      for (auto &Pred : SU->Preds) {
        if (shouldPruneEdge(Pred)) {
          continue;
        }
        auto *PredSU = Pred.getSUnit();
        if (--SuccsLeft[PredSU] == 0) {
          buildDistanceMapForNode(PredSU);
          WorkList.push_back(PredSU);
        }
      }
    }
  }

  void buildDistanceMapForNode(SUnit *SU) {
    auto &SUDists = DistMap[SU];
    for (auto &Succ : SU->Succs) {
      if (shouldPruneEdge(Succ)) {
        continue;
      }
      auto *SuccSU = Succ.getSUnit();
      if (isRoot(SuccSU)) {
        // Don't populate second-order roots.
        auto &CurrDist = SUDists[SuccSU];
        CurrDist = std::max(Succ.getLatency(), CurrDist);
      } else {
        // Don't trigger a re-hash.
        if (auto SuccDists = DistMap.find(SuccSU); SuccDists != DistMap.end())
          for (auto [SuccRoot, RootDist] : SuccDists->second) {
            auto &CurrDist = SUDists[SuccRoot];
            CurrDist = std::max(Succ.getLatency() + RootDist, CurrDist);
          }
      }
    }
  }

  void initializeRoots() {
    for (auto &[RootSU, _] : RootInfos) {
      auto Iter = DistMap.find(RootSU);
      if (Iter == DistMap.end())
        continue;
      for (auto [SuccRoot, _] : Iter->second)
        // Won't trigger a re-hash
        ++RootInfos[SuccRoot].PredRootsLeft;
    }
  }

public:
  ResourceDistanceMapBuilder(ScheduleDAGInstrs *DAG, unsigned ResourceID)
      : DAG(DAG), ResourceID(ResourceID) {}

  ResourceDistanceMaps::ResourceInfo build() {
    getReachableNodes();
    buildDistanceMap();
    initializeRoots();
    ResourceDistanceMaps::ResourceInfo Res(DAG);
    Res.init(std::move(DistMap), std::move(RootInfos));
    return Res;
  }
};
ResourceDistanceMaps::ResourceInfo
ResourceDistanceMaps::build(ScheduleDAGInstrs *DAG, unsigned ResourceID) {
  return ResourceDistanceMapBuilder(DAG, ResourceID).build();
}

void ResourceDistanceMaps::ResourceInfo::init(DistMapTy &&DistMap_,
                                              RootInfosTy &&RootInfos_) {
  DistMap = std::move(DistMap_);
  RootInfos = std::move(RootInfos_);
  for (const auto &KV : DistMap) {
    if (isRoot(KV.first)) {
      continue;
    }
    for (const auto &KV1 : KV.second) {
      RootInfos[KV1.first].NonRootPredLatency.push(KV1.second);
    }
  }
  for (auto &KV : RootInfos) {
    if (KV.second.PredRootsLeft == 0) {
      CurrentRoots.insert(&KV);
    }
  }
  sortRoots();
}

unsigned
ResourceDistanceMaps::ResourceInfo::getOrderForRoot(SUnit *Root) const {
  auto Iter = RootInfos.find(Root);
  assert(Iter != RootInfos.end());
  return Iter->second.Order;
}

void ResourceDistanceMaps::ResourceInfo::sortRoots() {
  int Order = 0;
  bool Changed = false;
  for (auto Root : CurrentRoots) {
    if (Root->second.Order != Order) {
      Changed = true;
    }
    Root->second.Order = Order++;
  }
  if (Changed) {
    SURankCache.clear();
  }
}

void ResourceDistanceMaps::ResourceInfo::schedNode(SUnit *SU,
                                                   unsigned CurrCycle) {
  // CurrCycle is part of the public schedNode contract for future use; no
  // current ResourceInfo bookkeeping depends on it.
  (void)CurrCycle;
  bool IsRoot = isRoot(SU);
  if (IsRoot) {
    for (auto Iter = CurrentRoots.begin(), End = CurrentRoots.end();
         Iter != End; ++Iter) {
      if ((*Iter)->first == SU) {
        (*Iter)->second.Order = std::numeric_limits<int>::max() - 1;
        CurrentRoots.erase(Iter);
        break;
      }
    }
  }
  auto Iter = DistMap.find(SU);
  if (Iter == DistMap.end()) {
    if (IsRoot)
      sortRoots();
    return;
  }
  if (IsRoot) {
    for (auto [RootSU, Dist] : Iter->second) {
      auto &RootInfo = RootInfos[RootSU];
      --RootInfo.PredRootsLeft;
      if (RootInfo.PredRootsLeft == 0) {
        CurrentRoots.insert(&*RootInfos.find(RootSU));
      }
    }
  } else {
    for (auto [RootSU, Dist] : Iter->second) {
      auto RootIter = RootInfos.find(RootSU);
      auto &RootInfo = RootIter->second;
      // Erase from set before mutating sort key, then re-insert to avoid UB.
      auto SetIter = CurrentRoots.find(&*RootIter);
      bool WasInSet = SetIter != CurrentRoots.end();
      if (WasInSet)
        CurrentRoots.erase(SetIter);
      RootInfo.NonRootPredLatency.pop();
      if (WasInSet)
        CurrentRoots.insert(&*RootIter);
    }
  }
  sortRoots();
}

ResourceDistanceMaps::SUDistRank
ResourceDistanceMaps::ResourceInfo::getSURank(SUnit *SU) {
  auto Iter = SURankCache.find(SU);
  if (Iter != SURankCache.end()) {
    return Iter->second;
  }
  auto Res = getSURankImpl(SU);
  SURankCache.try_emplace(SU, Res);
  return Res;
}

ResourceDistanceMaps::SUDistRank
ResourceDistanceMaps::ResourceInfo::getSURankImpl(SUnit *SU) {
  // Prefer roots.
  if (isRoot(SU)) {
    return {-1};
  }

  SUDistRank Res{std::numeric_limits<int>::max()};
  // Prefer nodes leading to roots.
  auto Iter = DistMap.find(SU);
  if (Iter == DistMap.end()) {
    return Res;
  }

  // Prefer nodes leading to closer roots.
  for (auto [Root, Dist] : Iter->second) {
    auto &RootInfo = RootInfos[Root];
    if (RootInfo.PredRootsLeft == 0) {
      SUDistRank NewRes{RootInfo.Order, Dist};
      if (NewRes < Res)
        Res = NewRes;
    }
  }
  return Res;
}

ResourceDistanceMaps::ResourceInfo &
ResourceDistanceMaps::ensureResDistMap(unsigned ResourceID) const {
  auto Iter = Maps.find(ResourceID);
  if (Iter != Maps.end())
    return Iter->second;
  return Maps.try_emplace(ResourceID, build(DAG, ResourceID)).first->second;
}

ResourceDistanceMaps::SUDistRank
ResourceDistanceMaps::getSUnitRankForRes(SUnit *SU, unsigned ResourceID) const {
  return ensureResDistMap(ResourceID).getSURank(SU);
}

void ResourceDistanceMaps::schedNode(SUnit *SU, unsigned CurrCycle) {
  for (auto &[_, ResInfo] : Maps)
    ResInfo.schedNode(SU, CurrCycle);
}

} // namespace AMDGPU
} // namespace llvm
