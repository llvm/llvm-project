//===- RegAllocSegmentTree.cpp - Segment Tree Register Allocator ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the SegmentTreeRegisterAllocator
// class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/RegAllocSegmentTree.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegAllocPriorityAdvisor.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Timer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include <limits>

using namespace llvm;

// [NEW] 旗標：是否開啟座標預先計算
static cl::opt<bool> SegTreePrecompute(
    "segtree-precompute",
    cl::desc("Enable precomputing of coordinate compression and segment trees "
             "for the Segment-Tree register allocator"),
    cl::init(true));

#define DEBUG_TYPE "regallocsegtre"

STATISTIC(NumGlobalSplits, "Number of split global live ranges");
STATISTIC(NumLocalSplits,  "Number of split local live ranges");
STATISTIC(NumEvicted,      "Number of interferences evicted");
STATISTIC(NumSegTreeUpdatesReal, "Number of actual successful segtree updates");

const char RegAllocSegmentTree::TimerGroupName[] = "segtre";
const char RegAllocSegmentTree::TimerGroupDescription[] = "Segment Tree Register Allocator";

INITIALIZE_PASS_BEGIN(RegAllocSegmentTree, "regallocsegtre",
                      "Segment Tree Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveStacksWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_END(RegAllocSegmentTree, "regallocsegtre",
                    "Segment Tree Register Allocator", false, false)

char RegAllocSegmentTree::ID = 0;

// 讓 llc/clang 能用 -regalloc=segtre 選到你
static llvm::RegisterRegAlloc
  RAReg("segmenttree",                           // 選項名稱
        "Segment Tree Register Allocator",  // 說明文字
        llvm::createRegAllocSegmentTree);   // 工廠函式（見下一步）

RegAllocSegmentTree::RegAllocSegmentTree()
  : MachineFunctionPass(ID) {
    // [NEW] 根據命令列旗標設定是否啟用 precompute
    UsePrecompute = SegTreePrecompute;

    if (TimePassesIsEnabled) {
      // 创建TimerGroup
      TimerGroupObj = std::make_unique<TimerGroup>("segtre", "Segment Tree Register Allocator");
      // 使用TimerGroup创建各个Timer
      PrecomputeTimer = std::make_unique<Timer>("Precompute", "Precompute Coordinates", *TimerGroupObj);
      AllocationTimer = std::make_unique<Timer>("Allocation", "Register Allocation", *TimerGroupObj);
      SpillTimer = std::make_unique<Timer>("Spill", "Spill Processing", *TimerGroupObj);
      CleanupTimer = std::make_unique<Timer>("Cleanup", "Cleanup Failed VRegs", *TimerGroupObj);
      PostOptTimer = std::make_unique<Timer>("PostOpt", "Post Optimization", *TimerGroupObj);
      SegTreeUpdateTimer = std::make_unique<Timer>("SegTreeUpdate", "Segment Tree Updates", *TimerGroupObj);
      SegTreeQueryTimer = std::make_unique<Timer>("SegTreeQuery", "Segment Tree Queries", *TimerGroupObj);
      GlobalSplitTimer = std::make_unique<Timer>("GlobalSplit", "Global Splitting", *TimerGroupObj);
      LocalSplitTimer = std::make_unique<Timer>("LocalSplit", "Local Splitting", *TimerGroupObj);
      RegionSplitTimer = std::make_unique<Timer>("RegionSplit", "Region Splitting", *TimerGroupObj);
      BlockSplitTimer = std::make_unique<Timer>("BlockSplit", "Block Splitting", *TimerGroupObj);
    }

    initializeSlotIndexesWrapperPassPass(*PassRegistry::getPassRegistry());
    initializeLiveIntervalsWrapperPassPass(*PassRegistry::getPassRegistry());
    initializeLiveStacksWrapperLegacyPass(*PassRegistry::getPassRegistry());
    initializeVirtRegMapWrapperLegacyPass(*PassRegistry::getPassRegistry());
  }

bool RegAllocSegmentTree::runOnMachineFunction(MachineFunction &MF) {
  // 在函數開始時重置計時器
  if (TimePassesIsEnabled && PrecomputeTimer) {
    PrecomputeTimer->clear();
    AllocationTimer->clear();
    SpillTimer->clear();
    CleanupTimer->clear();
    PostOptTimer->clear();
    SegTreeUpdateTimer->clear();
    SegTreeQueryTimer->clear();
  }
  // 每個函式開始時，將本函式的成功更新次數清零
  NumSegTreeUpdatesRealLocal = 0;

  // 获取必要的分析结果
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  MachineBlockFrequencyInfo &MBFI =
      getAnalysis<MachineBlockFrequencyInfoWrapperPass>().getMBFI();

  auto &LiveStks = getAnalysis<LiveStacksWrapperLegacy>().getLS();
  auto &MDT = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();

  // 计算溢出权重和提示
  VirtRegAuxInfo VRAI(
      MF, *LIS, *VRM, getAnalysis<MachineLoopInfoWrapperPass>().getLI(), MBFI);
  VRAI.calculateSpillWeightsAndHints();

  // 先把 LIS / VRM 解參考成 LiveIntervals& / VirtRegMap&
  Spiller::RequiredAnalyses RA{ *LIS, LiveStks, MDT, MBFI };

  // 再建立 Spiller
  SpillerInstance = std::unique_ptr<Spiller>(
    createInlineSpiller(RA, MF, *VRM, VRAI));

  MF.getRegInfo().freezeReservedRegs();

  LLVM_DEBUG(dbgs() << "Segment Tree Register Allocating for " << MF.getName() << "\n");

  // 初始化RegAllocBase
  auto &LRM = getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
  init(*VRM, *LIS, LRM);

  // 防呆：此時 MRI 已在 init() 裡設為成員指標
  assert(this->MF && LIS && VRM && MRI &&
       "RA pointers must be initialized before allocation");
  assert(SpillerInstance && "Spiller must be created before allocation");

  // === Greedy 風格：建立佇列並用 PriorityAdvisor 排程 ===
  {
    while (!Queue.empty()) Queue.pop();
    InQ.clear();
    MachineRegisterInfo &MRI_ = MF.getRegInfo();
    for (unsigned i = 0, e = MRI_.getNumVirtRegs(); i != e; ++i) {
      Register Reg = Register::index2VirtReg(i);
      if (MRI_.reg_nodbg_empty(Reg)) continue;
      LiveInterval *LI = &LIS->getInterval(Reg);
      if (LI->empty()) continue;
      enqueueImpl(LI);
    }
  }

  if (Queue.empty()) {
    finalizeAlloc(MF, *LIS, *VRM);
    postOptimization(*SpillerInstance, *LIS);
    return true;
  }

  if (TimePassesIsEnabled) {
    AllocationTimer->startTimer();
  }

  unsigned Iter = 0;
  const unsigned MaxIters = 10000000;
  while (const LiveInterval *LIc = dequeue()) {
    if (++Iter > MaxIters) {
      LLVM_DEBUG(dbgs() << "Warning: MaxIters reached in allocation loop\n");
      break;
    }
    LiveInterval &LI = *const_cast<LiveInterval*>(LIc);
    unsigned VirtReg = LI.reg();
    if (VRM->hasPhys(VirtReg)) continue;
    if (unsigned PhysReg = tryAllocateRegister(LI)) {
      VRM->assignVirt2Phys(VirtReg, PhysReg);
      updateSegmentTreeForInterval(LI, PhysReg);
      continue;
    }
    if (LI.isSpillable()) {
      spillVirtReg(LI); // 產生的新 vreg 會被 enqueue
    } else {
      FailedVRegs.insert(VirtReg);
      LLVM_DEBUG(dbgs() << "Virtual register " << VirtReg
                        << " cannot be spilled or allocated; defer\n");
    }
  }

  if (TimePassesIsEnabled) {
    AllocationTimer->stopTimer();
  }
#ifndef NDEBUG
  if (!verifyAllocation(MF, *LIS, *VRM)) {
    llvm_unreachable("寄存器分配驗證失敗!");
  }
#endif
  postOptimization(*SpillerInstance, *LIS);

  LLVM_DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << VRM << "\n");

  if (TimePassesIsEnabled) {
    // 累加当前函数的数据到总和中
    TotalPrecomputeTime += PrecomputeTimer->getTotalTime().getWallTime();
    TotalAllocationTime += AllocationTimer->getTotalTime().getWallTime();
    TotalSpillTime += SpillTimer->getTotalTime().getWallTime();
    TotalCleanupTime += CleanupTimer->getTotalTime().getWallTime();
    TotalPostOptTime += PostOptTimer->getTotalTime().getWallTime();
    TotalSegTreeUpdateTime += SegTreeUpdateTimer->getTotalTime().getWallTime();
    TotalSegTreeQueryTime += SegTreeQueryTimer->getTotalTime().getWallTime();
    
    TotalAllocAttempts += NumAllocAttempts;
    TotalInterferenceChecks += NumInterferenceChecks;
    TotalSegTreeUpdates += NumSegTreeUpdates;
    TotalCoordRebuilds += NumCoordRebuilds;
    TotalSpills += NumSpills;
    TotalSegTreeUpdatesReal += NumSegTreeUpdatesRealLocal;
    
    // 输出总表
    errs() << "\n=== Segment Tree Allocator Cumulative Statistics ===\n";
    errs() << "  Total Precompute Time: " << TotalPrecomputeTime << " seconds\n";
    errs() << "  Total Allocation Time: " << TotalAllocationTime << " seconds\n";
    errs() << "  Total Spill Time: " << TotalSpillTime << " seconds\n";
    errs() << "  Total Cleanup Time: " << TotalCleanupTime << " seconds\n";
    errs() << "  Total PostOpt Time: " << TotalPostOptTime << " seconds\n";
    errs() << "  Total SegTree Update Time: " << TotalSegTreeUpdateTime << " seconds\n";
    errs() << "  Total SegTree Query Time: " << TotalSegTreeQueryTime << " seconds\n";
    
    double TotalTime = TotalPrecomputeTime + TotalAllocationTime + TotalSpillTime +
                        TotalCleanupTime + TotalPostOptTime + TotalSegTreeUpdateTime +
                        TotalSegTreeQueryTime;
    errs() << "  Total Time: " << TotalTime << " seconds\n";
    
    errs() << "\n  Total Allocation Attempts: " << TotalAllocAttempts << "\n";
    errs() << "  Total Interference Checks: " << TotalInterferenceChecks << "\n";
    errs() << "  Total Segment Tree Updates: " << TotalSegTreeUpdates << "\n";
    errs() << "  Total Actual SegTree Updates: " << TotalSegTreeUpdatesReal << "\n";
    errs() << "  Total Coordinate Rebuilds: " << TotalCoordRebuilds << "\n";
    errs() << "  Total Spills: " << TotalSpills << "\n";
    errs() << "=====================================================\n";
  }

  // 输出时间信息
  errs() << "  Time breakdown:\n";
  TimerGroupObj->print(errs());  // 输出整个组的时间
  
  errs() << "==========================================\n";

  return true;
}

void RegAllocSegmentTree::updateSegmentTreeForInterval(const LiveInterval &LI, unsigned PhysReg) {
  // 需要更新 PhysReg 及其所有別名
  TRI = MF->getSubtarget().getRegisterInfo();
  std::set<unsigned> processedRegs;
  
  for (const auto &Seg : LI) {
    // 更新主寄存器及所有別名
    for (MCRegAliasIterator AI(PhysReg, TRI, /*IncludeSelf=*/true); AI.isValid(); ++AI) {
      unsigned A = *AI;
      if (processedRegs.insert(A).second) {
        updateSegmentTreeForPhysReg(A, Seg.start, Seg.end);
      }
    }
  }
}

void RegAllocSegmentTree::enqueueImpl(const LiveInterval *LI) { enqueue(Queue, LI); }

void RegAllocSegmentTree::enqueue(PQueue &CurQueue, const LiveInterval *LI) {
  // Prioritize live ranges by size, assigning larger ranges first.
  // The queue holds (size, reg) pairs.
  const Register Reg = LI->reg();
  assert(Reg.isVirtual() && "Can only enqueue virtual registers");


  // 可選：只有在 ExtraInfo 有被 emplace/reset 時才操作 stage
  if (ExtraInfo) {
    auto Stage = ExtraInfo->getOrInitStage(Reg);
    if (Stage == RS_New) {
      ExtraInfo->setStage(Reg, RS_Assign);
    }
  }

  // 取得優先度：若無 PriorityAdvisor，退化為簡單規則
  unsigned Pri = 0;
  if (PriorityAdvisor) {
    Pri = PriorityAdvisor->getPriority(*LI);
  } else {
    // 備援：用生命期大小或 spill weight
    Pri = static_cast<unsigned>(LI->getSize()); // 或 (unsigned)calcSpillWeight(*LI, *LIS, ...);
  }

  // 以 vreg 編號做 tie-breaker：~id() 讓較小的 vreg 有較高優先權
  CurQueue.push(std::make_pair(Pri, ~Reg.id()));
}

const LiveInterval *RegAllocSegmentTree::dequeue() { return dequeue(Queue); }

const LiveInterval *RegAllocSegmentTree::dequeue(PQueue &CurQueue) {
  if (CurQueue.empty())
    return nullptr;
  LiveInterval *LI = &LIS->getInterval(~CurQueue.top().second);
  CurQueue.pop();
  return LI;
}

MCRegister RegAllocSegmentTree::selectOrSplit(const LiveInterval &VirtReg,
                         SmallVectorImpl<Register> &splitLVRs) {
  LLVM_DEBUG(dbgs() << "selectOrSplit called for vreg " << VirtReg.reg() << "\n");
  
  // 首先嘗試分配物理寄存器
  unsigned PhysReg = tryAllocateRegister(const_cast<LiveInterval&>(VirtReg));
  
  if (PhysReg) {
    LLVM_DEBUG(dbgs() << "  Successfully allocated physreg " << PhysReg << " for vreg " << VirtReg.reg() << "\n");
    return PhysReg;
  }
  
  LLVM_DEBUG(dbgs() << "  Cannot allocate physreg for vreg " << VirtReg.reg() << ", trying to spill\n");
  
  // 如果無法分配，使用 spiller 進行處理
  // 這會自動處理拆分和溢出
  LiveInterval &LI = const_cast<LiveInterval&>(VirtReg);
  
  // 檢查是否可以溢出
  if (!LI.isSpillable()) {
    LLVM_DEBUG(dbgs() << "  Vreg " << VirtReg.reg() << " is not spillable\n");
    
    // 嘗試強制分配第一個可用的物理寄存器
    const TargetRegisterClass *RC = MRI->getRegClass(VirtReg.reg());
    if (RC && RC->getNumRegs() > 0) {
      for (MCRegister CandidateReg : *RC) {
        if (CandidateReg != 0 && !MRI->isReserved(CandidateReg)) {
          LLVM_DEBUG(dbgs() << "  Force-assigning physreg " << CandidateReg << " to unspillable vreg " << VirtReg.reg() << "\n");
          return CandidateReg;
        }
      }
    }
    return 0;
  }
  
  // 使用 LiveRangeEdit 和 Spiller 進行拆分/溢出
  SmallVector<Register, 4> NewVRegs;
  LiveRangeEdit LRE(&LI, NewVRegs, *MF, *LIS, VRM, this, &DeadRemats);
  
  // 執行溢出，這會自動處理拆分
  spiller().spill(LRE);
  
  // 將新產生的虛擬寄存器加入結果列表
  for (Register NewVReg : NewVRegs) {
    if (LIS->hasInterval(NewVReg)) {
      splitLVRs.push_back(NewVReg);
      LLVM_DEBUG(dbgs() << "  Created new vreg " << NewVReg << " from spill/split\n");
      
      // 將新的區間加入工作佇列
      LiveInterval &NewLI = LIS->getInterval(NewVReg);
      enqueueImpl(&NewLI);
    }
  }
  
  // 返回0表示已經進行了溢出/拆分，需要進一步處理新產生的虛擬寄存器
  return 0;
}

void RegAllocSegmentTree::init(VirtRegMap &vrm, LiveIntervals &lis,
                                        LiveRegMatrix &mat) {
  // 首先调用基类的init方法
  RegAllocBase::init(vrm, lis, mat);

  // === 這兩行是關鍵：把 optional / unique_ptr 建起來 ===
  ExtraInfo.emplace();  // 只要預設建構即可，之後才能 ExtraInfo->getOrInitStage(...)

  // 然后获取需要的其他信息
  // 通过vrm获取MachineFunction
  MF = &vrm.getMachineFunction();
  
  // 在这里初始化我们自己的数据结构
  // 1. 获取目标机器的物理寄存器信息
  TRI = MF->getSubtarget().getRegisterInfo();
  
  // 初始化线段树数据结构
  // 为所有物理寄存器分配线段树空间
  unsigned NumPhysRegs = TRI->getNumRegs();

  PhysRegIntervals.clear();
  PhysRegIntervals.assign(NumPhysRegs, {});

  // ★ 初始化座標與 lazy 線段樹容器（每個 physreg 一份）
  PRCoords.clear();
  PRCoords.assign(NumPhysRegs, {});
  PRTree.clear();
  PRTree.assign(NumPhysRegs, {});

  this->MRI = &MF->getRegInfo();

  // 确保保留寄存器信息已冻结
  MRI->freezeReservedRegs();

  // 预先收集所有可能的坐标点
  if (UsePrecompute) {
    precomputeAllCoordinates();
  } else {
    LLVM_DEBUG(dbgs() << "Skipping precompute, will build on-demand\n");
  }

  LLVM_DEBUG(dbgs() << "Initializing Segment Tree Register Allocator with "
                    << NumPhysRegs << " physical registers\n");
}

// ===== 線段樹分割策略群組 =====
MCRegister RegAllocSegmentTree::trySegmentTreeSplit(const LiveInterval &VirtReg,
                                                    SmallVectorImpl<Register> &NewVRegs) {
  // 使用線段樹分析找到最佳分割點
  SlotIndex BestSplitPoint = findOptimalSplitPoint(VirtReg);
  
  if (!BestSplitPoint.isValid()) {
    return MCRegister();
  }
  
  // 執行分割
  return performSplitAtPoint(VirtReg, BestSplitPoint, NewVRegs);
}

SlotIndex RegAllocSegmentTree::findOptimalSplitPoint(const LiveInterval &VirtReg) {
  // 基於線段樹分析找到最佳分割點的實作
  // 這裡可以分析干擾密度來決定分割點
  return findSplitPoint(VirtReg); // 暫時使用現有方法
}

MCRegister RegAllocSegmentTree::performSplitAtPoint(const LiveInterval &VirtReg,
                                                    SlotIndex SplitPoint,
                                                    SmallVectorImpl<Register> &NewVRegs) {
  // 在指定點執行分割的實作
  // 使用 LiveRangeEdit 進行分割
  LiveInterval &LI = const_cast<LiveInterval&>(VirtReg);
  LiveRangeEdit LRE(&LI, NewVRegs, *MF, *LIS, VRM, this, &DeadRemats);
  
  // TODO: 實作實際的分割邏輯
  
  return MCRegister(); // 表示已執行分割，需要處理新產生的寄存器
}

void RegAllocSegmentTree::precomputeAllCoordinates() {
  TimeRegion TR(*PrecomputeTimer);
  // 收集所有虚拟寄存器的所有区间端点
  std::set<SlotIndex> allTimePoints;
  
  for (unsigned i = 0, e = MF->getRegInfo().getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = Register::index2VirtReg(i);
    if (MF->getRegInfo().reg_nodbg_empty(Reg))
      continue;
    
    LiveInterval *LI = &LIS->getInterval(Reg);
    if (LI->empty())
      continue;
    
    for (const auto &Seg : *LI) {
      allTimePoints.insert(Seg.start);
      allTimePoints.insert(Seg.end);
    }
  }

  GlobalCoords.assign(allTimePoints.begin(), allTimePoints.end());

  // 每個物理暫存器共用同一組座標
  for (unsigned P = 0; P < TRI->getNumRegs(); ++P) {
    PRCoords[P] = GlobalCoords;
    segtreeBuild(P);
  }

// 添加所有物理寄存器的区间端点（如果有的话）
//   for (unsigned PhysReg = 0; PhysReg < TRI->getNumRegs(); ++PhysReg) {
//     for (const auto &Interval : PhysRegIntervals[PhysReg]) {
//       allTimePoints.insert(Interval.first);
//       allTimePoints.insert(Interval.second);
//     }
//   }
  
//   // 为每个物理寄存器设置相同的坐标点
//   for (unsigned PhysReg = 0; PhysReg < PRCoords.size(); ++PhysReg) {
//     if (PhysReg >= TRI->getNumRegs()) continue;
    
//     PRCoords[PhysReg].assign(allTimePoints.begin(), allTimePoints.end());
//     segtreeBuild(PhysReg);
//   }
}

void RegAllocSegmentTree::precomputeGlobalCoords() {
  std::set<SlotIndex> allPoints;
  // 收集所有虛擬寄存器的端點
  for (unsigned i = 0, e = MF->getRegInfo().getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = Register::index2VirtReg(i);
    if (!MF->getRegInfo().reg_nodbg_empty(Reg)) {
      LiveInterval *LI = &LIS->getInterval(Reg);
      for (const auto &Seg : *LI) {
        allPoints.insert(Seg.start);
        allPoints.insert(Seg.end);
      }
    }
  }
  GlobalCoords.assign(allPoints.begin(), allPoints.end());
}

unsigned RegAllocSegmentTree::tryAllocateRegister(LiveInterval &VirtReg) {
  assert(MF && LIS && VRM && MRI &&
         "RA pointers must be initialized before allocation (tryAllocateRegister)");
  
  LLVM_DEBUG(dbgs() << "Trying to allocate register for virtreg " << VirtReg.reg() << "\n");

  ++NumAllocAttempts;  // 加入這行

  // 1. 獲取此虛擬暫存器的暫存器類別
  const TargetRegisterClass *RC = MRI->getRegClass(VirtReg.reg());
  if (!RC) {
    LLVM_DEBUG(dbgs() << "Error: Cannot get register class for virtual register " << VirtReg.reg() << "\n");
    return 0;
  }

  // 获取目标寄存器信息
  TRI = MRI->getTargetRegisterInfo();
  if (TRI) {
    LLVM_DEBUG(dbgs() << "Register class: " << TRI->getRegClassName(RC) << "\n");
  } else {
    LLVM_DEBUG(dbgs() << "Cannot get target register info\n");
  }

  // 2. 獲取該類別中所有可分配的物理暫存器
  // 3. 對於每個候選物理暫存器，使用線段樹查詢它是否在VirtReg的生命期內可用
  for (unsigned PhysReg : *RC) {
    LLVM_DEBUG(dbgs() << "Trying physreg " << PhysReg << "\n");
    
    // 检查物理寄存器编号是否有效
    if (PhysReg == 0) {
      LLVM_DEBUG(dbgs() << "Skipping physreg 0\n");
      continue;
    }

    // 检查该物理寄存器是否是保留寄存器
    if (MRI->isReserved(PhysReg)) {
      LLVM_DEBUG(dbgs() << "Skipping reserved physreg " << PhysReg << "\n");
      continue;
    }
    
    // 检查该物理寄存器是否在VirtReg的整个生命期内都可用
    if (isPhysRegAvailable(PhysReg, VirtReg)) {
      LLVM_DEBUG(dbgs() << "Allocated physreg " << PhysReg << " for virtreg " << VirtReg.reg() << "\n");
      return PhysReg;
    } else {
      LLVM_DEBUG(dbgs() << "Physreg " << PhysReg << " not available\n");
    }
  }

  LLVM_DEBUG(dbgs() << "No available physreg for virtreg " << VirtReg.reg() << "\n");
  // 没有找到可用的物理寄存器
  return 0;
}

bool RegAllocSegmentTree::isPhysRegAvailable(unsigned PhysReg,
                                             const LiveInterval &LI) {
  TimeRegion TR(*SegTreeQueryTimer);
  
  if (PhysReg == 0) return false;

  if (PhysRegIntervals[PhysReg].empty())
    return true;  // 完全未使用的寄存器
  
  TRI = MF->getSubtarget().getRegisterInfo();

  for (MCRegAliasIterator AI(PhysReg, TRI, /*IncludeSelf=*/true); AI.isValid(); ++AI) {
    unsigned A = *AI;
    // 對每個待查 segment 先確保座標，避免端點不在 coords 造成誤判
    for (const auto &Seg : LI) {
      ++NumInterferenceChecks;  // 加入這行
      
      const auto &coords = PRCoords[A];
      const auto &tree   = PRTree[A];
      if (coords.size() < 2 || tree.empty()) continue;

      unsigned l = coordIndex(A, Seg.start);
      unsigned r = coordIndex(A, Seg.end);
      if (l >= r) continue;

      int mx = segtreeQueryMax(A, 1, 0, (unsigned)coords.size()-2, l, r-1);
      if (mx > 0) return false;   // 任一段已有占用 → 不可放
    }
  }
  return true;
}

int RegAllocSegmentTree::segtreeQueryMax(unsigned PhysReg, unsigned idx,
                                         unsigned L, unsigned R,
                                         unsigned ql, unsigned qr) {
  auto &Tree = PRTree[PhysReg];
  
  // 如果当前节点区间完全包含在查询区间内，直接返回最大值
  if (ql <= L && R <= qr) {
    return Tree[idx].maxCover;
  }
  
  unsigned Mid = (L + R) / 2;
  unsigned LeftChild = 2 * idx;
  unsigned RightChild = 2 * idx + 1;
  
  // 处理懒标记下推
  if (Tree[idx].lazyAdd != 0) {
    Tree[LeftChild].maxCover += Tree[idx].lazyAdd;
    Tree[LeftChild].lazyAdd += Tree[idx].lazyAdd;
    Tree[RightChild].maxCover += Tree[idx].lazyAdd;
    Tree[RightChild].lazyAdd += Tree[idx].lazyAdd;
    Tree[idx].lazyAdd = 0;
  }
  
  int leftMax = std::numeric_limits<int>::min();
  int rightMax = std::numeric_limits<int>::min();
  
  // 递归查询左子树
  if (ql <= Mid) {
    leftMax = segtreeQueryMax(PhysReg, LeftChild, L, Mid, ql, std::min(qr, Mid));
  }
  
  // 递归查询右子树
  if (qr > Mid) {
    rightMax = segtreeQueryMax(PhysReg, RightChild, Mid + 1, R, std::max(ql, Mid + 1), qr);
  }
  
  // 返回左右子树的最大值
  return std::max(leftMax, rightMax);
}

int RegAllocSegmentTree::segtreeQueryMaxIter(unsigned PhysReg, unsigned l, unsigned r) {
  int res = 0;
  unsigned n = PRCoords[PhysReg].size() - 1;
  l += n, r += n;
  while (l <= r) {
    if (l % 2 == 1) res = std::max(res, PRTree[PhysReg][l++].maxCover);
    if (r % 2 == 0) res = std::max(res, PRTree[PhysReg][r--].maxCover);
    l /= 2, r /= 2;
  }
  return res;
}

void RegAllocSegmentTree::updateSegmentTreeForPhysReg(unsigned PhysReg, SlotIndex Start, SlotIndex End) {
  TimeRegion TR(*SegTreeUpdateTimer);

  assert(Register::isPhysicalRegister(PhysReg) && "expected physreg");
  assert(PhysReg < PRCoords.size() && PhysReg < PRTree.size() &&
         "physreg index out of range");
  assert(Start.isValid() && End.isValid() && Start < End && "invalid segment");
  
  ++NumSegTreeUpdates;  // 加入這行
  
  // 获取坐标索引
  auto &Coords = PRCoords[PhysReg];
  auto &Intervals = PhysRegIntervals[PhysReg];

  // 檢查是否已經記錄過相同的區間
  for (const auto &existing : Intervals) {
    if (!(End <= existing.first || Start >= existing.second)) {
      LLVM_DEBUG(dbgs() << "Interval [" << Start << ", " << End 
                        << ") overlaps with existing interval [" 
                        << existing.first << ", " << existing.second 
                        << ") for PhysReg " << PhysReg << ", skipping\n");
      return;  // 重疊，直接返回
    }
  }

  // 在非預計算模式下，動態添加座標點
  if (!UsePrecompute) {
    bool coordsChanged = false;
    
    // 檢查並添加 Start
    auto startIt = std::lower_bound(Coords.begin(), Coords.end(), Start);
    if (startIt == Coords.end() || *startIt != Start) {
      Coords.insert(startIt, Start);
      coordsChanged = true;
    }
    
    // 檢查並添加 End
    auto endIt = std::lower_bound(Coords.begin(), Coords.end(), End);
    if (endIt == Coords.end() || *endIt != End) {
      Coords.insert(endIt, End);
      coordsChanged = true;
    }
    
    // 如果座標點有變化，需要重建線段樹
    if (coordsChanged) {
      segtreeBuild(PhysReg);
      
      // 重新應用所有已知區間到新線段樹
      for (const auto &interval : Intervals) {
        SlotIndex segStart = interval.first;
        SlotIndex segEnd = interval.second;
        
        auto segStartIt = std::lower_bound(Coords.begin(), Coords.end(), segStart);
        auto segEndIt = std::lower_bound(Coords.begin(), Coords.end(), segEnd);
        
        if (segStartIt != Coords.end() && segEndIt != Coords.end() &&
            *segStartIt == segStart && *segEndIt == segEnd) {
          unsigned l = segStartIt - Coords.begin();
          unsigned r = segEndIt - Coords.begin();
          
          if (l < r) {
            const unsigned segN = Coords.size() - 1;
            segtreeUpdate(PhysReg, 1, 0, segN - 1, l, r - 1, +1);
          }
        }
      }
    }
  }

  // auto startIt = std::lower_bound(Coords.begin(), Coords.end(), Start);
  // auto endIt   = std::lower_bound(Coords.begin(), Coords.end(), End);
  
  // // 确保坐标点存在
  // if (startIt == PRCoords[PhysReg].end() || *startIt != Start ||
  //     endIt == PRCoords[PhysReg].end() || *endIt != End) {
  //   // errs() << "[segtre][ERROR] Missing coordinate in PRCoords for PhysReg "
  //   //        << PhysReg << " interval [" << Start << ", " << End << ")\n";
  //   // dbgs() << "  Did you forget to call precomputeAllCoordinates() after spill?\n";
  //   llvm_unreachable("updateSegmentTreeForPhysReg: coordinate not found!");
  //   return;
  // }
  
  // unsigned startIdx = startIt - Coords.begin();
  // unsigned endIdx = endIt - Coords.begin();
  
  // // 更新线段树
  // if (startIdx < endIdx) {
  //   const unsigned segN = Coords.size() - 1;
  //   segtreeUpdate(PhysReg, 1, 0, segN - 1, startIdx, endIdx - 1, +1);
  // }
  
  // // 记录这个区间
  // PhysRegIntervals[PhysReg].emplace_back(Start, End);
  
  // LLVM_DEBUG(dbgs() << "Updated segment tree for physreg " << PhysReg
  //                   << " with interval [" << Start << ", " << End << ")\n");


  // 先記錄這個區間（在重建之後）
  Intervals.emplace_back(Start, End);

  // 然後在線段樹中更新這個新區間
  unsigned startIdx = coordIndex(PhysReg, Start);
  unsigned endIdx = coordIndex(PhysReg, End);
  
  if (startIdx < endIdx) {
    const unsigned segN = Coords.size() - 1;
    segtreeUpdate(PhysReg, 1, 0, segN - 1, startIdx, endIdx - 1, +1);
  }

  LLVM_DEBUG(dbgs() << "Added new interval [" << Start << ", " << End 
                  << ") for PhysReg " << PhysReg << "\n");
}

// void RegAllocSegmentTree::ensureCoordsAndTree(unsigned PhysReg, SlotIndex S, SlotIndex E) {
//   auto &Coords = PRCoords[PhysReg];
//   auto &Tree = PRTree[PhysReg];
  
//   // 检查是否需要添加新的坐标点
//   bool changed = false;
  
//   // 检查起点 S
//   auto it = std::lower_bound(Coords.begin(), Coords.end(), S);
//   if (it == Coords.end() || *it != S) {
//     Coords.insert(it, S);
//     changed = true;
//   }
  
//   // 检查终点 E
//   it = std::lower_bound(Coords.begin(), Coords.end(), E);
//   if (it == Coords.end() || *it != E) {
//     Coords.insert(it, E);
//     changed = true;
//   }
  
//   // 如果坐标发生了变化，需要重建线段树
//   if (changed) {
//     for (const auto &Interval : Intervals) {
//       SlotIndex Start = Interval.first;
//       SlotIndex End = Interval.second;
      
//       // 获取坐标索引
//       unsigned l = coordIndex(PhysReg, Start);
//       unsigned r = coordIndex(PhysReg, End) - 1; // 线段树区间是 [l, r-1]
      
//       // 更新线段树
//       segtreeUpdate(PhysReg, 1, 0, Coords.size() - 2, l, r, 1);
//     }
//   }
// }

void RegAllocSegmentTree::ensureCoordsForPhysReg(unsigned PhysReg) {
  if (!PRCoords[PhysReg].empty()) return;
  
  // 只收集與此物理寄存器相關的虛擬寄存器端點
  std::set<SlotIndex> points;
  for (const auto &Interval : PhysRegIntervals[PhysReg]) {
    points.insert(Interval.first);
    points.insert(Interval.second);
  }

  // 如果沒有區間，添加默認的座標點（函數開始和結束）
  if (points.empty()) {
    points.insert(LIS->getSlotIndexes()->getZeroIndex());
    points.insert(LIS->getSlotIndexes()->getLastIndex());
  }
  
  PRCoords[PhysReg].assign(points.begin(), points.end());
  segtreeBuild(PhysReg);

  // DEBUG：看看到底有哪些座標
//   dumpCoords(PhysReg);
}

void RegAllocSegmentTree::dumpCoords(unsigned PhysReg) const {
  const auto &Coords = PRCoords[PhysReg];
  dbgs() << "[segtre] PhysReg " << PhysReg
         << " coords size=" << Coords.size() << " :";
  for (unsigned i = 0; i < Coords.size(); ++i) {
    dbgs() << " [" << i << "]" << Coords[i];
  }
  dbgs() << "\n";
}

unsigned RegAllocSegmentTree::coordIndex(unsigned PhysReg, SlotIndex X) const {
  const auto &Coords = PRCoords[PhysReg];
  auto It = std::lower_bound(Coords.begin(), Coords.end(), X);
  // 在非預計算模式下，如果座標點不存在，我們需要處理這種情況
  if (It == Coords.end() || *It != X) {
    if (UsePrecompute) {
      llvm_unreachable("SlotIndex not found in coordinates in precompute mode!");
    }
    // 在非預計算模式下，返回最接近的索引
    // 這是一個保守的估計，可能不夠準確
    if (It == Coords.end()) {
      return Coords.size() - 1;
    } else if (It == Coords.begin()) {
      return 0;
    } else {
      // 返回前一個索引
      return It - Coords.begin() - 1;
    }
  }
  return It - Coords.begin();
}

void RegAllocSegmentTree::segtreeUpdate(unsigned PhysReg, unsigned idx,
                                        unsigned L, unsigned R,
                                        unsigned ql, unsigned qr, int add) {
  auto &Tree = PRTree[PhysReg];

  LLVM_DEBUG(dbgs() << "[segtre][before] PhysReg=" << PhysReg
                    << " node=" << idx
                    << " range=[" << L << "," << R << "]"
                    << " q=[" << ql << "," << qr << "]"
                    << " add=" << add
                    << " maxCover=" << Tree[idx].maxCover
                    << " lazy=" << Tree[idx].lazyAdd << "\n");
  
  if (ql > R || qr < L)
    return;

  if (ql <= L && R <= qr) {
    Tree[idx].maxCover += add;
    Tree[idx].lazyAdd += add;
    ++NumSegTreeUpdatesReal;
    ++NumSegTreeUpdatesRealLocal;

    LLVM_DEBUG(dbgs() << "[segtre][after-full] PhysReg=" << PhysReg
                    << " node=" << idx
                    << " new maxCover=" << Tree[idx].maxCover
                    << " lazy=" << Tree[idx].lazyAdd << "\n");
    return;
  }

  unsigned Mid = (L + R) / 2;
  unsigned LeftChild = 2 * idx;
  unsigned RightChild = 2 * idx + 1;

  // Push lazy value to children
  if (Tree[idx].lazyAdd != 0) {
    Tree[LeftChild].maxCover += Tree[idx].lazyAdd;
    Tree[LeftChild].lazyAdd += Tree[idx].lazyAdd;
    Tree[RightChild].maxCover += Tree[idx].lazyAdd;
    Tree[RightChild].lazyAdd += Tree[idx].lazyAdd;
    Tree[idx].lazyAdd = 0;
  }

  segtreeUpdate(PhysReg, LeftChild, L, Mid, ql, qr, add);
  segtreeUpdate(PhysReg, RightChild, Mid + 1, R, ql, qr, add);

  int oldMax = Tree[idx].maxCover;
  Tree[idx].maxCover = std::max(Tree[LeftChild].maxCover, Tree[RightChild].maxCover);

  LLVM_DEBUG(dbgs() << "[segtre][after-recalc] PhysReg=" << PhysReg
                    << " node=" << idx
                    << " oldMax=" << oldMax
                    << " newMax=" << Tree[idx].maxCover
                    << " lazy=" << Tree[idx].lazyAdd << "\n");
}

void RegAllocSegmentTree::segtreeBuild(unsigned PhysReg) {
  auto &Coords = PRCoords[PhysReg];
  auto &Tree = PRTree[PhysReg];
  
  // 确定线段树大小（n个坐标点对应n-1个区间）
  unsigned n = Coords.size();
  if (n < 2) {
    // 没有足够的坐标点构建线段树
    Tree.clear();
    return;
  }
  
  // 线段树需要4*(n-1)的大小（标准线段树大小）
  unsigned treeSize = 4 * (n - 1);
  Tree.resize(treeSize);
  
  // 初始化所有节点
  for (auto &node : Tree) {
    node.maxCover = 0;
    node.lazyAdd = 0;
  }

  // 在非預計算模式下，需要重新應用所有已知區間
  if (!UsePrecompute) {
    for (const auto &interval : PhysRegIntervals[PhysReg]) {
      SlotIndex Start = interval.first;
      SlotIndex End = interval.second;
      
      auto startIt = std::lower_bound(Coords.begin(), Coords.end(), Start);
      auto endIt = std::lower_bound(Coords.begin(), Coords.end(), End);
      
      if (startIt != Coords.end() && endIt != Coords.end() &&
          *startIt == Start && *endIt == End) {
        unsigned l = startIt - Coords.begin();
        unsigned r = endIt - Coords.begin();
        
        if (l < r) {
          const unsigned segN = Coords.size() - 1;
          segtreeUpdate(PhysReg, 1, 0, segN - 1, l, r - 1, +1);
        }
      }
    }
  }
}

void RegAllocSegmentTree::spillVirtReg(LiveInterval &VirtReg) {
  TimeRegion TR(*SpillTimer);
  assert(MF && LIS && VRM);
  LLVM_DEBUG(dbgs() << "Spilling vreg " << VirtReg.reg() << '\n');

  ++NumSpills;  // 加入這行

  SmallVector<Register, 4> NewVRegs;
  LiveRangeEdit LRE(&VirtReg, NewVRegs, *MF, *LIS, VRM, this, &DeadRemats);

  spiller().spill(LRE);

  for (Register NV : NewVRegs) {
    if (!LIS->hasInterval(NV)) continue;
    LiveInterval &NewLI = LIS->getInterval(NV);
    enqueueImpl(&NewLI);
  }
}

void RegAllocSegmentTree::finalizeAlloc(MachineFunction &MF,
                                                 LiveIntervals &LIS,
                                                 VirtRegMap &VRM) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  
  LLVM_DEBUG(dbgs() << "Finalizing allocation - checking all virtual registers\n");

  // 處理所有虛擬寄存器，確保都有分配
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(Reg))
      continue;
    
    // 如果已經分配，跳過
    if (VRM.hasPhys(Reg))
      continue;

    // 強制分配策略
    LiveInterval &LI = LIS.getInterval(Reg);
    const TargetRegisterClass &RC = *MRI.getRegClass(Reg);
    
    LLVM_DEBUG(dbgs() << "Force-allocating unassigned register "
                      << printReg(Reg, TRI) << "\n");

    // 尋找第一個可用的物理寄存器
    Register PReg = 0;
    ArrayRef<MCPhysReg> Order = RC.getRawAllocationOrder(MF);
    for (MCPhysReg CandidateReg : Order) {
      if (CandidateReg != 0 && !VRM.getRegInfo().isReserved(CandidateReg)) {
        PReg = CandidateReg;
        break;
      }
    }

    // 如果仍然找不到，使用第一個非零寄存器（即使是保留寄存器）
    if (PReg == 0 && !Order.empty()) {
      PReg = Order[0];
    }

    if (PReg != 0) {
      VRM.assignVirt2Phys(Reg, PReg);
      LLVM_DEBUG(dbgs() << "Force-assigned " << printReg(Reg, TRI)
                        << " to " << printReg(PReg, TRI) << "\n");
    } else {
      // 這應該永遠不會發生
      llvm_unreachable("Cannot find any physical register for virtual register");
    }
  }

  // 驗證所有虛擬寄存器都已分配
  LLVM_DEBUG({
    bool AllAllocated = true;
    for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
      Register Reg = Register::index2VirtReg(I);
      if (MRI.reg_nodbg_empty(Reg))
        continue;
      
      if (!VRM.hasPhys(Reg)) {
        dbgs() << "ERROR: Virtual register " << Reg << " still not allocated!\n";
        AllAllocated = false;
      }
    }
    
    if (AllAllocated) {
      dbgs() << "SUCCESS: All virtual registers have been allocated\n";
    }
  });
}

void RegAllocSegmentTree::resetAllocatorState() {
  // 重置分配器状态，为下一个函数做准备
  PhysRegIntervals.clear();
  PRCoords.clear();
  PRTree.clear();
  DeadRemats.clear();
  FailedVRegs.clear();

  CurQueue.clear();
  InQ.clear();
}

void RegAllocSegmentTree::postOptimization(Spiller &VRegSpiller, LiveIntervals &LIS) {
  // 调用基类的 postOptimization 方法（如果存在）
  // RegAllocBase::postOptimization();
  TimeRegion TR(*PostOptTimer);

  // 执行 Spiller 的后优化
  VRegSpiller.postOptimization();
  
  // 清理死指令（由于重新材料化而产生的死定义）
  LLVM_DEBUG(dbgs() << "Cleaning up dead rematerialized instructions\n");
  for (MachineInstr *MI : DeadRemats) {
    if (MI->getParent()) {
      LLVM_DEBUG(dbgs() << "  Removing: " << *MI);
      LIS.RemoveMachineInstrFromMaps(*MI);
      MI->eraseFromParent();
    }
  }
  DeadRemats.clear();
  
  // 清理失败的虚拟寄存器
  LLVM_DEBUG(dbgs() << "Cleaning up failed virtual registers\n");
  for (Register FailedVReg : FailedVRegs) {
    LLVM_DEBUG(dbgs() << "  Cleaning up failed vreg: " << FailedVReg << "\n");
    SmallVector<Register, 4> SplitRegs;
    cleanupFailedVReg(FailedVReg, 0, SplitRegs);
    
    // 处理拆分产生的寄存器
    for (Register SplitReg : SplitRegs) {
      if (LIS.hasInterval(SplitReg)) {
        LiveInterval &SplitLI = LIS.getInterval(SplitReg);
        enqueueImpl(&SplitLI);
      }
    }
  }
  FailedVRegs.clear();
  
  // 执行线段树特定的后优化
  performSegmentTreeSpecificOptimizations(LIS);
  
  // 验证优化后的状态
  validatePostOptimizationState(LIS);
  
  LLVM_DEBUG(dbgs() << "Post-optimization completed for Segment Tree allocator\n");
}

void RegAllocSegmentTree::cleanupFailedVReg(Register FailedVReg, unsigned Depth,
                                            SmallVectorImpl<Register> &SplitRegs) {
  TimeRegion TR(*CleanupTimer);
  
  // 防止無限遞歸
  if (Depth > 10) {
    LLVM_DEBUG(dbgs() << "  Max depth reached, giving up on vreg " << FailedVReg << "\n");
    return;
  }
  
  LLVM_DEBUG(dbgs() << "  Cleaning up failed vreg " << FailedVReg << " at depth " << Depth << "\n");
  
  // 檢查虛擬寄存器是否仍然存在
  if (!LIS->hasInterval(FailedVReg) || MRI->reg_nodbg_empty(FailedVReg)) {
    LLVM_DEBUG(dbgs() << "  Vreg " << FailedVReg << " no longer exists, skipping\n");
    return;
  }
  
  LiveInterval &LI = LIS->getInterval(FailedVReg);
  
  // 如果區間為空，直接返回
  if (LI.empty()) {
    LLVM_DEBUG(dbgs() << "  Vreg " << FailedVReg << " has empty interval, skipping\n");
    return;
  }
  
  // 如果已經分配了物理寄存器，就不需要清理
  if (VRM->hasPhys(FailedVReg)) {
    LLVM_DEBUG(dbgs() << "  Vreg " << FailedVReg << " already has physical register, skipping\n");
    return;
  }
  
  // 如果可以溢出，使用 spiller 處理
  if (LI.isSpillable()) {
    LLVM_DEBUG(dbgs() << "  Spilling failed vreg " << FailedVReg << "\n");
    
    // 使用 LiveRangeEdit 和 Spiller
    SmallVector<Register, 4> NewVRegs;
    LiveRangeEdit LRE(&LI, NewVRegs, *MF, *LIS, VRM, this, &DeadRemats);
    
    // 執行溢出
    spiller().spill(LRE);
    
    // 將新產生的虛擬寄存器加入結果列表
    for (Register NewVReg : NewVRegs) {
      if (LIS->hasInterval(NewVReg)) {
        SplitRegs.push_back(NewVReg);
        LLVM_DEBUG(dbgs() << "  Created new vreg " << NewVReg << " from cleanup spill\n");
      }
    }
  } else {
    // 不可溢出的寄存器，嘗試其他策略
    LLVM_DEBUG(dbgs() << "  Vreg " << FailedVReg << " is not spillable, trying alternative strategies\n");
    
    // 策略1：嘗試找到一個衝突較少的物理寄存器強制分配
    const TargetRegisterClass *RC = MRI->getRegClass(FailedVReg);
    if (RC) {
      MCRegister BestPhysReg = 0;
      int MinConflicts = INT_MAX;
      
      // 遍歷該寄存器類別中的所有物理寄存器
      for (MCRegister PhysReg : *RC) {
        if (PhysReg == 0 || MRI->isReserved(PhysReg)) {
          continue;
        }
        
        // 計算與此物理寄存器的衝突數量
        int Conflicts = 0;
        for (MCRegAliasIterator AI(PhysReg, TRI, /*IncludeSelf=*/true); AI.isValid(); ++AI) {
          unsigned A = *AI;
          if (A >= PRCoords.size()) continue;
          
          const auto &coords = PRCoords[A];
          if (coords.size() < 2) continue;
          
          for (const auto &Seg : LI) {
            // 確保座標存在
            auto it_start = std::lower_bound(coords.begin(), coords.end(), Seg.start);
            auto it_end = std::lower_bound(coords.begin(), coords.end(), Seg.end);
            
            if (it_start != coords.end() && it_end != coords.end() &&
                *it_start == Seg.start && *it_end == Seg.end) {
              unsigned l = it_start - coords.begin();
              unsigned r = it_end - coords.begin();
              
              if (l < r && PRTree[A].size() > 0) {
                int mx = segtreeQueryMax(A, 1, 0, (unsigned)coords.size()-2, l, r-1);
                Conflicts += mx;
              }
            }
          }
        }
        
        // 記錄衝突最少的物理寄存器
        if (Conflicts < MinConflicts) {
          MinConflicts = Conflicts;
          BestPhysReg = PhysReg;
        }
      }
      
      // 如果找到了一個相對較好的物理寄存器，強制分配
      if (BestPhysReg != 0) {
        LLVM_DEBUG(dbgs() << "  Force-assigning vreg " << FailedVReg
                          << " to physreg " << BestPhysReg
                          << " with " << MinConflicts << " conflicts\n");
        
        VRM->assignVirt2Phys(FailedVReg, BestPhysReg);
        updateSegmentTreeForInterval(LI, BestPhysReg);
        return;
      }
    }
    
    // 策略2：如果沒有找到合適的物理寄存器，嘗試分割生命期
    if (LI.getNumValNums() > 1) {
      LLVM_DEBUG(dbgs() << "  Trying to split multi-value vreg " << FailedVReg << "\n");
      
      // 對於有多個值的區間，嘗試按值分割
      // 這裡使用簡單的策略：如果有多個段，嘗試從中間分割
      if (LI.size() > 1) {
        // 找到中間的分割點
        auto MidSegIt = LI.begin();
        std::advance(MidSegIt, LI.size() / 2);
        SlotIndex SplitPoint = MidSegIt->start;
        
        if (SplitPoint.isValid() && SplitPoint > LI.beginIndex() && SplitPoint < LI.endIndex()) {
          // 使用手動分割方式
          // 創建新的虛擬寄存器
          Register NewVReg = MRI->createVirtualRegister(RC);
          LiveInterval &NewLI = LIS->getOrCreateEmptyInterval(NewVReg);
          
          // 將分割點之後的段移動到新區間
          auto SplitIt = LI.begin();
          while (SplitIt != LI.end() && SplitIt->start < SplitPoint) {
            ++SplitIt;
          }
          
          if (SplitIt != LI.end()) {
            // 將後半部分移動到新區間
            while (SplitIt != LI.end()) {
              NewLI.segments.push_back(*SplitIt);
              SplitIt = LI.segments.erase(SplitIt);
            }
            
            // 更新區間的結束點
            LI.verify();
            NewLI.verify();
            
            SplitRegs.push_back(NewVReg);
            LLVM_DEBUG(dbgs() << "  Manual split created vreg " << NewVReg << "\n");
          }
        }
      }
    }
    
    // 策略3：最後手段，選擇第一個可用的物理寄存器強制分配
    if (!VRM->hasPhys(FailedVReg) && RC && RC->getNumRegs() > 0) {
      MCRegister FirstPhysReg = *RC->begin();
      if (FirstPhysReg != 0) {
        LLVM_DEBUG(dbgs() << "  Last resort: force-assigning vreg " << FailedVReg
                          << " to first physreg " << FirstPhysReg << "\n");
        
        VRM->assignVirt2Phys(FailedVReg, FirstPhysReg);
        updateSegmentTreeForInterval(LI, FirstPhysReg);
      }
    }
  }
  
  // 遞歸清理新產生的分割寄存器（如果有的話）
  SmallVector<Register, 4> NewSplitRegs;
  for (Register SplitReg : SplitRegs) {
    if (!VRM->hasPhys(SplitReg)) {
      cleanupFailedVReg(SplitReg, Depth + 1, NewSplitRegs);
    }
  }
  
  // 將遞歸產生的新寄存器也加入結果
  SplitRegs.append(NewSplitRegs.begin(), NewSplitRegs.end());
}

void RegAllocSegmentTree::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<SlotIndexesWrapperPass>();
  AU.addPreserved<SlotIndexesWrapperPass>();
  AU.addRequired<LiveIntervalsWrapperPass>();
  AU.addPreserved<LiveIntervalsWrapperPass>();
  AU.addRequired<LiveStacksWrapperLegacy>();
  AU.addPreserved<LiveStacksWrapperLegacy>();
  AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
  AU.addPreserved<MachineBlockFrequencyInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  AU.addRequired<VirtRegMapWrapperLegacy>();
  AU.addPreserved<VirtRegMapWrapperLegacy>();
  AU.addRequired<LiveRegMatrixWrapperLegacy>();
  AU.addPreserved<LiveRegMatrixWrapperLegacy>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

SlotIndex RegAllocSegmentTree::findSplitPoint(const LiveInterval &LI) {
  if (LI.empty()) return SlotIndex();
  
  // 策略1：如果有多個段，在段之間分割
  if (LI.size() > 1) {
    auto MidIt = LI.begin();
    std::advance(MidIt, LI.size() / 2);
    LLVM_DEBUG(dbgs() << "    Split point found between segments at " << MidIt->start << "\n");
    return MidIt->start;
  }
  
  // 策略2：對於單段，嘗試在段內找到中點
  if (LI.size() == 1) {
    const LiveRange::Segment &Seg = *LI.begin();
    SlotIndex SegStart = Seg.start;
    SlotIndex SegEnd = Seg.end;
    
    if (!SegStart.isValid() || !SegEnd.isValid() || SegStart >= SegEnd) {
      LLVM_DEBUG(dbgs() << "    Invalid segment bounds\n");
      return SlotIndex();
    }
    
    // 使用迭代方式找到中點
    SlotIndex Current = SegStart;
    SlotIndex Target = SegEnd;
    
    // 先計算總步數
    unsigned TotalSteps = 0;
    SlotIndex Counter = Current;
    while (Counter.isValid() && Counter < Target && TotalSteps < 1000) {
      Counter = Counter.getNextIndex();
      TotalSteps++;
    }
    
    // 如果段太小（步數少於4），不適合分割
    if (TotalSteps < 4) {
      LLVM_DEBUG(dbgs() << "    Segment too small to split (steps: " << TotalSteps << ")\n");
      return SlotIndex();
    }
    
    // 移動到中間位置
    SlotIndex Middle = Current;
    unsigned TargetSteps = TotalSteps / 2;
    
    for (unsigned Step = 0; Step < TargetSteps; ++Step) {
      SlotIndex Next = Middle.getNextIndex();
      if (!Next.isValid() || Next >= Target) {
        LLVM_DEBUG(dbgs() << "    Cannot advance to middle position\n");
        break;
      }
      Middle = Next;
    }
    
    // 驗證分割點的有效性
    if (Middle.isValid() && Middle > SegStart && Middle < SegEnd) {
      LLVM_DEBUG(dbgs() << "    Split point found at " << Middle
                        << " (step " << TargetSteps << "/" << TotalSteps << ")\n");
      return Middle;
    } else {
      LLVM_DEBUG(dbgs() << "    Middle position not valid for splitting\n");
    }
  }
  
  // 策略3：如果無法在段內分割，但有多個定義點，嘗試基於值分割
  if (LI.getNumValNums() > 1) {
    LLVM_DEBUG(dbgs() << "    Attempting value-based split for " << LI.getNumValNums() << " values\n");
    
    // 遍歷所有段，找到不同值之間的邊界
    for (auto it = LI.begin(); it != LI.end(); ++it) {
      auto nextIt = it;
      ++nextIt;
      
      if (nextIt != LI.end() && it->valno != nextIt->valno) {
        // 在不同值之間分割
        SlotIndex SplitPoint = nextIt->start;
        LLVM_DEBUG(dbgs() << "    Value-based split point found at " << SplitPoint << "\n");
        return SplitPoint;
      }
    }
  }
  
  // 所有策略都失敗
  LLVM_DEBUG(dbgs() << "    No suitable split point found\n");
  return SlotIndex();
}

void RegAllocSegmentTree::performSegmentTreeSpecificOptimizations(LiveIntervals &LIS) {
  // 执行线段树特定的后优化
  // 这里可以添加任何与线段树相关的优化逻辑
  LLVM_DEBUG(dbgs() << "Performing segment tree specific optimizations\n");
}

void RegAllocSegmentTree::validatePostOptimizationState(LiveIntervals &LIS) {
  // 验证优化后的状态
  // 这里可以添加任何验证逻辑
  LLVM_DEBUG(dbgs() << "Validating post-optimization state\n");
}

// 創建Pass實例的函數
llvm::FunctionPass *llvm::createRegAllocSegmentTree() {
  return new RegAllocSegmentTree();
}

#ifndef NDEBUG
bool RegAllocSegmentTree::verifyAllocation(MachineFunction &MF,
                                          LiveIntervals &LIS,
                                          VirtRegMap &VRM) {
  bool Verified = true;
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  // 檢查1: 驗證每個物理寄存器上的區間不重疊
  for (unsigned PhysReg = 0; PhysReg < TRI->getNumRegs(); ++PhysReg) {
    // 跳过未使用的物理寄存器
    if (PRCoords[PhysReg].size() < 2) continue;

    auto Intervals = PhysRegIntervals[PhysReg];
    std::sort(Intervals.begin(), Intervals.end(),
              [](const auto &A, const auto &B) {
                return A.first < B.first;
              });

    for (unsigned i = 1; i < Intervals.size(); ++i) {
      if (Intervals[i-1].second > Intervals[i].first) {
        dbgs() << "錯誤: 物理寄存器 " << printReg(PhysReg, TRI)
               << " 上的區間重疊!\n"
               << "區間1: [" << Intervals[i-1].first << ", "
               << Intervals[i-1].second << ")\n"
               << "區間2: [" << Intervals[i].first << ", "
               << Intervals[i].second << ")\n";
        Verified = false;
      }
    }
  }

  // 檢查2: 驗證每個活躍的虛擬寄存器都有物理寄存器映射
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    
    if (!VRM.hasPhys(Reg)) {
      dbgs() << "錯誤: 虛擬寄存器 " << Reg
             << " 沒有分配物理寄存器!\n";
      Verified = false;
    }
  }

  // 檢查3: 驗證線段樹狀態與實際分配一致
  for (unsigned PhysReg = 0; PhysReg < TRI->getNumRegs(); ++PhysReg) {
    // 跳过没有坐标点的物理寄存器
    if (PRCoords[PhysReg].size() < 2) continue;

    for (const auto &Interval : PhysRegIntervals[PhysReg]) {
      SlotIndex Start = Interval.first;
      SlotIndex End = Interval.second;
      
      // 确保坐标点存在
      auto startIt = std::lower_bound(PRCoords[PhysReg].begin(), PRCoords[PhysReg].end(), Start);
      auto endIt = std::lower_bound(PRCoords[PhysReg].begin(), PRCoords[PhysReg].end(), End);
      
      if (startIt == PRCoords[PhysReg].end() || *startIt != Start ||
          endIt == PRCoords[PhysReg].end() || *endIt != End) {
        dbgs() << "警告: 物理寄存器 " << printReg(PhysReg, TRI)
               << " 的區間端點不在坐標點中: [" << Start << ", " << End << ")\n";
        continue;
      }
      
      // 獲取坐標索引
      unsigned startIdx = startIt - PRCoords[PhysReg].begin();
      unsigned endIdx = endIt - PRCoords[PhysReg].begin();
      
      // 只查询有效的区间
      if (startIdx >= endIdx) continue;
      
      // 查詢線段樹中該區間的最大值
      int MaxCover = segtreeQueryMax(PhysReg, 1, 0, PRCoords[PhysReg].size()-2, startIdx, endIdx-1);
      
      if (MaxCover <= 0) {
        dbgs() << "錯誤: 線段樹記錄與物理寄存器 " << printReg(PhysReg, TRI)
               << " 的分配不一致! 區間: [" << Start << ", " << End
               << "), 查詢結果: " << MaxCover << "\n";
        Verified = false;
      }
    }
  }

  return Verified;
}
#endif

RegAllocSegmentTree::~RegAllocSegmentTree() {
if (TimePassesIsEnabled) {
errs() << "=== Segment Tree Allocator Total Time (across all functions) ===\n";
errs() << " Precompute Coordinates: " << TotalPrecomputeTime << " seconds\n";
errs() << " Register Allocation: " << TotalAllocationTime << " seconds\n";
errs() << " Spill Processing: " << TotalSpillTime << " seconds\n";
errs() << " Cleanup Failed VRegs: " << TotalCleanupTime << " seconds\n";
errs() << " Post Optimization: " << TotalPostOptTime << " seconds\n";
errs() << " Segment Tree Updates: " << TotalSegTreeUpdateTime << " seconds\n";
errs() << " Segment Tree Queries: " << TotalSegTreeQueryTime << " seconds\n";
errs() << " Total Actual SegTree Updates: " << TotalSegTreeUpdatesReal << "\n";
errs() << "==========================================\n";
}
}
