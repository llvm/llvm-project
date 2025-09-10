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

#include "RegAllocSegmentTree.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "regallocsegtre"

// 註冊這個分配器，使其可以被LLVM命令行選項使用
static RegisterRegAlloc
RegisterSegmentTreeRegisterAllocator("segtre", "segment tree register allocator",
                                     createSegmentTreeRegisterAllocator);

char SegmentTreeRegisterAllocator::ID = 0;

// 初始化靜態Pass ID
INITIALIZE_PASS_BEGIN(SegmentTreeRegisterAllocator, "regallocsegtre",
                "Segment Tree Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_END(SegmentTreeRegisterAllocator, "regallocsegtre",
                "Segment Tree Register Allocator", false, false)

SegmentTreeRegisterAllocator::SegmentTreeRegisterAllocator(char &ID)
    : RegAllocBase(ID) {}

bool SegmentTreeRegisterAllocator::runOnMachineFunction(MachineFunction &mf) {
  // 調用基類的runOnMachineFunction，它會初始化必要的分析結果並調用allocatePhysRegs
  return RegAllocBase::runOnMachineFunction(mf);
}

void SegmentTreeRegisterAllocator::init(const MachineFunction &MF,
                                        const VirtRegMap &vrm,
                                        const LiveIntervals &lis,
                                        const MachineLoopInfo &mli,
                                        const MachineBlockFrequencyInfo &mbfi) {
  // 調用基類的init
  RegAllocBase::init(MF, vrm, lis, mli, mbfi);

  // 在這裡初始化我們自己的數據結構
  // 1. 獲取目標機器的物理暫存器信息
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // 2. 為每個暫存器類別初始化線段樹
  //    這是一個簡化的示例。實際實現中，您需要遍歷所有可分配的物理暫存器
  for (unsigned RCId = 0; RCId < TRI->getNumRegClasses(); ++RCId) {
    // 獲取該暫存器類別中的所有物理暫存器
    // 初始化對應的線段樹...
    // PhysRegSegmentTrees[RCId].resize(...);
  }

  LLVM_DEBUG(dbgs() << "Initializing Segment Tree Register Allocator\n");
}

void SegmentTreeRegisterAllocator::allocatePhysRegs() {
  // 主要的分配循環
  // 1. 我們需要對虛擬暫存器進行排序（例如，按生命期長度、權重等）
  //    這對Segment Tree分配器的效能至關重要
  std::vector<LiveInterval*> VirtRegs;
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = TargetRegisterInfo::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg))
      continue;
    LiveInterval *LI = &LIS->getInterval(Reg);
    assert(!LI->empty() && "Empty live interval?");
    VirtRegs.push_back(LI);
  }

  // 按某種策略排序，例如：開始時間、生命期長度、溢出成本等
  // 這是一個重要的設計決策點！
  // llvm::sort(VirtRegs, [](const LiveInterval *A, const LiveInterval *B) { ... });

  // 2. 按順序處理每個虛擬暫存器
  for (LiveInterval *LI : VirtRegs) {
    unsigned VirtReg = LI->reg;
    if (VRM->hasPhys(VirtReg)) // 可能已經通過合併分配了
      continue;

    // 嘗試分配物理暫存器
    unsigned PhysReg = tryAllocateRegister(*LI);

    if (PhysReg) {
      // 分配成功，更新線段樹和映射
      VRM->assignVirt2Phys(VirtReg, PhysReg);
      updateSegmentTreeForInterval(*LI, PhysReg);
    } else {
      // 分配失敗，需要溢出
      spillVirtReg(*LI);
    }
  }
}

unsigned SegmentTreeRegisterAllocator::tryAllocateRegister(LiveInterval &VirtReg) {
  // 1. 獲取此虛擬暫存器的暫存器類別
  const TargetRegisterClass *RC = MRI->getRegClass(VirtReg.reg);

  // 2. 獲取該類別中所有可分配的物理暫存器
  // 3. 對於每個候選物理暫存器，使用線段樹查詢它是否在VirtReg的生命期內可用
  for (unsigned PhysReg : *RC) {
    // 檢查該物理暫存器是否在VirtReg的整個生命期內都可用
    if (isPhysRegAvailable(PhysReg, VirtReg)) {
      return PhysReg;
    }
  }

  // 沒有找到可用的物理暫存器
  return 0;
}

bool SegmentTreeRegisterAllocator::isPhysRegAvailable(unsigned PhysReg,
                                                      const LiveInterval &VirtReg) {
  // 使用線段樹查詢：對於VirtReg的每個段，查詢PhysReg的線段樹是否有重疊
  // 這是一個簡化的實現思路
  for (const LiveInterval::Segment &Seg : VirtReg) {
    SlotIndex Start = Seg.start;
    SlotIndex End = Seg.end;
    // 查詢線段樹：在[Start, End)區間內，該物理暫存器是否已被占用
    if (!querySegmentTreeForRange(PhysReg, Start, End)) {
      return false; // 發現重疊，該物理暫存器不可用
    }
  }
  return true; // 所有段都沒有重疊，可用
}

void SegmentTreeRegisterAllocator::spillVirtReg(LiveInterval &VirtReg) {
  // 實現溢出邏輯
  // 1. 選擇一個溢出策略（例如，溢出成本最低的暫存器）
  // 2. 實際上是將該虛擬暫存器溢出到堆棧槽
  // 3. 更新LiveIntervals分析信息（這非常複雜，需要仔細處理）

  // 這部分需要大量借用現有分配器（如Greedy）中的邏輯
  LLVM_DEBUG(dbgs() << "Spilling virtual register: " << VirtReg.reg << '\n');
  // ... 具體實現省略 ...
}

// 線段樹的構建、查詢、更新函數的具體實現
void SegmentTreeRegisterAllocator::buildSegmentTree(SegmentTreeNode *tree, unsigned idx,
                                                    unsigned l, unsigned r,
                                                    const std::vector<SlotIndex> &ends) {
  // 標準的線段樹構建算法
  if (l == r) {
    tree[idx].MaxEnd = ends[l];
    return;
  }
  unsigned mid = (l + r) / 2;
  buildSegmentTree(tree, 2*idx, l, mid, ends);
  buildSegmentTree(tree, 2*idx+1, mid+1, r, ends);
  tree[idx].MaxEnd = std::max(tree[2*idx].MaxEnd, tree[2*idx+1].MaxEnd);
}

bool SegmentTreeRegisterAllocator::querySegmentTree(const SegmentTreeNode *tree, unsigned idx,
                                                    unsigned tree_l, unsigned tree_r,
                                                    unsigned query_l, unsigned query_r) const {
  // 標準的線段樹區間查詢
  if (query_r < tree_l || tree_r < query_l)
    return true; // 區間無交集，表示可用

  if (query_l <= tree_l && tree_r <= query_r) {
    // 當前節點區間完全包含在查詢區間內
    // 檢查該區間內的最大結束時間是否小於查詢的開始時間
    // 這是一個簡化的條件，實際邏輯需要根據您的設計調整
    return tree[idx].MaxEnd < query_l; // 這只是一個示例條件
  }

  unsigned mid = (tree_l + tree_r) / 2;
  return querySegmentTree(tree, 2*idx, tree_l, mid, query_l, query_r) &&
         querySegmentTree(tree, 2*idx+1, mid+1, tree_r, query_l, query_r);
}

// 創建Pass實例的函數
FunctionPass *llvm::createSegmentTreeRegisterAllocator() {
  return new SegmentTreeRegisterAllocator();
}