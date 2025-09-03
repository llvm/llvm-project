//===- RegAllocSegmentTree.h - Segment Tree Register Allocator --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the SegmentTreeRegisterAllocator class,
// which uses a segment tree data structure to manage register intervals.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGALLOCSEGMENTTREE_H
#define LLVM_CODEGEN_REGALLOCSEGMENTTREE_H

#include "RegAllocBase.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/LiveStacks.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>
#include <memory>

namespace llvm {

class LiveIntervals;
class MachineFunction;
class MachineRegisterInfo;
class TargetRegisterInfo;
class VirtRegMap;

/// SegmentTreeRegisterAllocator implements a register allocator by leveraging
/// a segment tree to efficiently track the availability of physical registers
/// across their live ranges.
class RegAllocSegmentTree : public MachineFunctionPass, public RegAllocBase, private LiveRangeEdit::Delegate {

  // 非常重要的：用於記錄每個物理暫存器當前的分配區間
  // Key: 物理暫存器編號 (PhysReg)
  // Value: 一個包含多個(LiveInterval, VirtReg)的列表，表示該物理暫存器上分配了哪些虛擬暫存器及其生命期
  // 但注意：一個物理暫存器在同一時間只能被一個虛擬暫存器佔用，所以這實際上是一系列不重疊的區間
  // 我們通常用一個按開始時間排序的區間列表來表示一個物理暫存器的使用情況
  // 我們將為每個物理暫存器建立一個線段樹

public:
  // 线段树节点结构体
  struct SegmentTreeNode {
    SlotIndex MaxEnd;  // 该节点覆盖区间内的最大结束时间
    
    SegmentTreeNode() : MaxEnd(SlotIndex()) {}
  };

  static char ID; // 静态成员声明

  RegAllocSegmentTree();

  /// 主要的分配接口，繼承自 MachineFunctionPass
  bool runOnMachineFunction(MachineFunction &mf) override;

  /// 獲取此分配器的名稱
  StringRef getPassName() const override { return "Segment Tree Register Allocator"; }

  /// 獲取分析依賴
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
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

  // 我們需要覆蓋這個方法來初始化我們的自定義數據結構
  void init(VirtRegMap &vrm, LiveIntervals &lis, LiveRegMatrix &mat);

  // 我們需要覆蓋這個方法來實現主要的分配邏輯
  void allocatePhysRegs();

  // 你自己的 finalize（宣告）
  void finalizeAlloc(MachineFunction &MF, LiveIntervals &LIS, VirtRegMap &VRM);

  // 你堅持要的同名接口（注意：**不要**寫 override）
  void postOptimization(Spiller &VRegSpiller, LiveIntervals &LIS);
protected:
  // 必须实现的纯虚函数
  Spiller &spiller() override;
  void enqueueImpl(const LiveInterval *LI) override;
  const LiveInterval *dequeue() override;
  MCRegister selectOrSplit(const LiveInterval &VirtReg,
                           SmallVectorImpl<Register> &splitLVRs) override;

private:
  void resetAllocatorState();  // ← 新增
  void performSegmentTreeSpecificOptimizations(LiveIntervals &LIS);
  void validatePostOptimizationState(LiveIntervals &LIS);

  MachineFunction *MF = nullptr;
  LiveIntervals       *LIS = nullptr;
  VirtRegMap          *VRM = nullptr;
  LiveRegMatrix       *LRM = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  std::unique_ptr<Spiller> VRegSpiller;  // Add this line

  // 改为动态分配或使用其他数据结构
  SmallVector<std::vector<SegmentTreeNode>> PhysRegSegmentTrees;

  // 或者，更高效的實現可能使用一個大的線段樹數組，並通過索引來訪問不同暫存器的樹

  // 线段树构建辅助函数（递归）
  void buildSegmentTree(SegmentTreeNode *tree, unsigned idx,
                        unsigned l, unsigned r,
                        const std::vector<SlotIndex> &ends);

  // 为物理寄存器构建线段树
  void buildSegmentTreeForPhysReg(unsigned PhysReg, const std::vector<SlotIndex>& intervals);
  
  // 查询物理寄存器在指定区间是否可用
  bool querySegmentTreeForRange(unsigned PhysReg, SlotIndex Start, SlotIndex End) const;

  // 线段树查询辅助函数
  bool querySegmentTree(const SegmentTreeNode *tree, unsigned idx,
                        unsigned tree_l, unsigned tree_r,
                        SlotIndex query_start, SlotIndex query_end) const;

  // 嘗試為給定的虛擬暫存器分配一個物理暫存器
  // 返回分配的物理暫存器，若失敗則返回0
  unsigned tryAllocateRegister(LiveInterval &VirtReg);

  // 檢查物理暫存器是否可用
  bool isPhysRegAvailable(unsigned PhysReg, const LiveInterval &VirtReg);

  // 溢出處理：當沒有可用的物理暫存器時，選擇一個虛擬暫存器溢出到內存
  void spillVirtReg(LiveInterval &VirtReg);

  // 更新线段树以反映区间的分配
  void updateSegmentTreeForInterval(const LiveInterval &LI, unsigned PhysReg);

  // 更新物理寄存器的线段树（分配区间后）
  void updateSegmentTreeForPhysReg(unsigned PhysReg, SlotIndex Start, SlotIndex End);
};

// 創建SegmentTreeRegisterAllocator實例的函數
FunctionPass *createRegAllocSegmentTree();

} // end namespace llvm

#endif // LLVM_CODEGEN_REGALLOCSEGMENTTREE_H