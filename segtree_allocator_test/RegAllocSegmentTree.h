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

#include "llvm/CodeGen/RegAllocBase.h"
#include "llvm/CodeGen/LiveInterval.h"
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
class SegmentTreeRegisterAllocator : public RegAllocBase {
  // 我們將使用一個內部類來表示線段樹節點
  class SegmentTreeNode {
  public:
    SlotIndex MaxEnd; // 該節點所代表區間內的最大結束時間
    // 可以添加更多字段，例如指向子節點的指針或暫存器狀態信息
    SegmentTreeNode() : MaxEnd(SlotIndex()) {}
  };

  // 非常重要的：用於記錄每個物理暫存器當前的分配區間
  // Key: 物理暫存器編號 (PhysReg)
  // Value: 一個包含多個(LiveInterval, VirtReg)的列表，表示該物理暫存器上分配了哪些虛擬暫存器及其生命期
  // 但注意：一個物理暫存器在同一時間只能被一個虛擬暫存器佔用，所以這實際上是一系列不重疊的區間
  // 我們通常用一個按開始時間排序的區間列表來表示一個物理暫存器的使用情況
  // 我們將為每個物理暫存器建立一個線段樹

public:
  SegmentTreeRegisterAllocator(char &ID);

  /// 主要的分配接口，繼承自 MachineFunctionPass
  bool runOnMachineFunction(MachineFunction &mf) override;

  /// 獲取此分配器的名稱
  StringRef getPassName() const override { return "Segment Tree Register Allocator"; }

  // 我們需要覆蓋這個方法來初始化我們的自定義數據結構
  void init(const MachineFunction &MF, const VirtRegMap &vrm,
            const LiveIntervals &lis, const MachineLoopInfo &mli,
            const MachineBlockFrequencyInfo &mbfi) override;

  // 我們需要覆蓋這個方法來實現主要的分配邏輯
  void allocatePhysRegs() override;

private:
  // 關鍵數據結構：為每個可用的物理暫存器維護一個線段樹
  // 注意：這是一個簡化的表示。實際實現中，您可能需要按暫存器類別分開管理
  std::vector<SegmentTreeNode> PhysRegSegmentTrees[MAX_NUM_REG_CLASSES];

  // 或者，更高效的實現可能使用一個大的線段樹數組，並通過索引來訪問不同暫存器的樹

  // 構建線段樹的輔助函數
  void buildSegmentTree(SegmentTreeNode *tree, unsigned idx, unsigned l, unsigned r,
                        const std::vector<SlotIndex> &ends);

  // 查詢線段樹：在區間 [l, r] 內查找可用的暫存器
  bool querySegmentTree(const SegmentTreeNode *tree, unsigned idx, unsigned tree_l,
                        unsigned tree_r, unsigned query_l, unsigned query_r) const;

  // 更新線段樹：當分配一個區間時更新樹
  void updateSegmentTree(SegmentTreeNode *tree, unsigned idx, unsigned tree_l,
                         unsigned tree_r, unsigned update_idx, SlotIndex new_end);

  // 嘗試為給定的虛擬暫存器分配一個物理暫存器
  // 返回分配的物理暫存器，若失敗則返回0
  unsigned tryAllocateRegister(LiveInterval &VirtReg);

  // 溢出處理：當沒有可用的物理暫存器時，選擇一個虛擬暫存器溢出到內存
  void spillVirtReg(LiveInterval &VirtReg);
};

// 創建SegmentTreeRegisterAllocator實例的函數
FunctionPass *createSegmentTreeRegisterAllocator();

} // end namespace llvm

#endif // LLVM_CODEGEN_REGALLOCSEGMENTTREE_H