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
#include <utility>
#include <algorithm>
#include <queue>

namespace llvm {

class LiveIntervals;
class MachineFunction;
class MachineRegisterInfo;
class TargetRegisterInfo;
class VirtRegMap;

class RegAllocSegmentTree : public MachineFunctionPass, 
                            public RegAllocBase, 
                            private LiveRangeEdit::Delegate {
public:
  static char ID; // 静态成员声明

  RegAllocSegmentTree();

  // MachineFunctionPass interface
  bool runOnMachineFunction(MachineFunction &mf) override;
  StringRef getPassName() const override { return "Segment Tree Register Allocator"; }

  /// 獲取分析依賴

  // 我們需要覆蓋這個方法來初始化我們的自定義數據結構
  void init(VirtRegMap &vrm, LiveIntervals &lis, LiveRegMatrix &mat);

  // 你自己的 finalize（宣告）
  void finalizeAlloc(MachineFunction &MF, LiveIntervals &LIS, VirtRegMap &VRM) const;

#ifndef NDEBUG
  bool verifyAllocation(MachineFunction &MF, LiveIntervals &LIS, VirtRegMap &VRM);
#endif

  // 你堅持要的同名接口（注意：**不要**寫 override）
  void postOptimization(Spiller &VRegSpiller, LiveIntervals &LIS);

private:
  // Timer group constants
  static const char TimerGroupName[];
  static const char TimerGroupDescription[];
  
  // Performance statistics
  mutable unsigned NumInterferenceChecks = 0;
  mutable unsigned NumSegTreeUpdates = 0;
  mutable unsigned NumCoordRebuilds = 0;
  mutable unsigned NumAllocAttempts = 0;
  mutable unsigned NumSpills = 0;

  // Convenient shortcuts.
  using PQueue = std::priority_queue<std::pair<unsigned, unsigned>>;
  using SmallLISet = SmallSetVector<const LiveInterval *, 4>;

  void resetAllocatorState();  // ← 新增
  void precomputeAllCoordinates();
  void performSegmentTreeSpecificOptimizations(LiveIntervals &LIS);
  void validatePostOptimizationState(LiveIntervals &LIS);

  MachineFunction *MF = nullptr;

  // Shortcuts to some useful interface.
  const TargetInstrInfo *TII = nullptr;

  LiveIntervals       *LIS = nullptr;
  VirtRegMap          *VRM = nullptr;
  LiveRegMatrix       *LRM = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const TargetRegisterInfo *TRI = nullptr;

  RegisterClassInfo RCI;

  std::unique_ptr<Spiller> SpillerInstance;  // Add this line

  // 工作佇列與去重集合
  llvm::SmallVector<const LiveInterval*, 64> CurQueue;
  llvm::SmallDenseSet<unsigned, 64> InQ;  // 記 vreg id，避免重複排入

  // Bookkeeping of allocated intervals per physical register (for rebuild).
  std::vector<std::vector<std::pair<SlotIndex, SlotIndex>>> PhysRegIntervals;

  //===------------------------------------------------------------------===//
  // Lazy segment tree (range add / range max) with coordinate compression
  //===------------------------------------------------------------------===//
  struct SegNode {
    int maxCover = 0;  // 節點覆蓋區間內的最大覆蓋次數
    int lazyAdd  = 0;  // lazy 累加（尚未下推）
  };

  // 每個 PhysReg 的座標壓縮點（遞增），點數 m → 段數 m-1
  std::vector<std::vector<SlotIndex>> PRCoords;

  // 每個 PhysReg 的 lazy 線段樹
  std::vector<std::vector<SegNode>> PRTree;

  // —— 內部工具 —— //
  // 確保 PR 的座標包含 [S,E) 的端點；必要時重建樹並回放既有區間
  void ensureCoordsAndTree(unsigned PhysReg, SlotIndex S, SlotIndex E);
  // 把 SlotIndex 映射成座標索引（必須已存在於 PRCoords[PR]）
  unsigned coordIndex(unsigned PhysReg, SlotIndex X) const;

  // 線段樹操作
  void segtreeBuild(unsigned PhysReg);
  void segtreeUpdate(unsigned PhysReg, unsigned idx, unsigned L, unsigned R,
                     unsigned ql, unsigned qr, int add);
  int  segtreeQueryMax(unsigned PhysReg, unsigned idx, unsigned L, unsigned R,
                       unsigned ql, unsigned qr);

  // 可放檢查（含別名）：任一段 maxCover>0 就不可放
  bool canPlaceOnPhysReg(unsigned PhysReg, const LiveInterval &LI) const;

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

  // 清理失败的虚拟寄存器
  void cleanupFailedVReg(Register FailedVReg, unsigned Depth,
                         SmallVectorImpl<Register> &SplitRegs);

  // 找到合適的拆分點
  SlotIndex findSplitPoint(const LiveInterval &LI);
public:
  // RegAllocBase interface  
  Spiller &spiller() override { return *SpillerInstance; };
  void enqueueImpl(const LiveInterval *LI) override;
  const LiveInterval *dequeue() override;
  MCRegister selectOrSplit(const LiveInterval &VirtReg,
                           SmallVectorImpl<Register> &splitLVRs) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

// 創建SegmentTreeRegisterAllocator實例的函數
FunctionPass *createRegAllocSegmentTree();

} // end namespace llvm

#endif // LLVM_CODEGEN_REGALLOCSEGMENTTREE_H
