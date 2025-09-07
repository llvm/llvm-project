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
#include "llvm/CodeGen/RegAllocPriorityAdvisor.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Timer.h"
#include <vector>
#include <memory>
#include <utility>
#include <algorithm>
#include <queue>

namespace llvm {

// 在任何 class 定義之前加入這些 type aliases
using SmallVirtRegSet = SmallSet<Register, 16>;
using RecoloringStack = SmallVector<std::pair<const LiveInterval*, MCRegister>, 8>;

class LiveIntervals;
class MachineFunction;
class MachineRegisterInfo;
class TargetRegisterInfo;
class VirtRegMap;

/// Segment Tree Register Allocator
///
/// This register allocator uses a segment tree data structure to efficiently
/// detect conflicts between live intervals and physical registers.
class RegAllocSegmentTree : public MachineFunctionPass, 
                            public RegAllocBase, 
                            private LiveRangeEdit::Delegate {
public:
  // Interface to eviction advisers
  /// Track allocation stage and eviction loop prevention during allocation.
  class ExtraRegInfo final {
    // RegInfo - Keep additional information about each live range.
    struct RegInfo {
      LiveRangeStage Stage = RS_New;

      // Cascade - Eviction loop prevention. See
      // canEvictInterferenceBasedOnCost().
      unsigned Cascade = 0;

      RegInfo() = default;
    };

    IndexedMap<RegInfo, VirtReg2IndexFunctor> Info;
    unsigned NextCascade = 1;

  public:
    ExtraRegInfo() {}
    ExtraRegInfo(const ExtraRegInfo &) = delete;

    LiveRangeStage getStage(Register Reg) const { return Info[Reg].Stage; }

    LiveRangeStage getStage(const LiveInterval &VirtReg) const {
      return getStage(VirtReg.reg());
    }

    void setStage(Register Reg, LiveRangeStage Stage) {
      Info.grow(Reg.id());
      Info[Reg].Stage = Stage;
    }

    void setStage(const LiveInterval &VirtReg, LiveRangeStage Stage) {
      setStage(VirtReg.reg(), Stage);
    }

    /// Return the current stage of the register, if present, otherwise
    /// initialize it and return that.
    LiveRangeStage getOrInitStage(Register Reg) {
      Info.grow(Reg.id());
      return getStage(Reg);
    }

    unsigned getCascade(Register Reg) const { return Info[Reg].Cascade; }

    void setCascade(Register Reg, unsigned Cascade) {
      Info.grow(Reg.id());
      Info[Reg].Cascade = Cascade;
    }

    unsigned getOrAssignNewCascade(Register Reg) {
      unsigned Cascade = getCascade(Reg);
      if (!Cascade) {
        Cascade = NextCascade++;
        setCascade(Reg, Cascade);
      }
      return Cascade;
    }

    unsigned getCascadeOrCurrentNext(Register Reg) const {
      unsigned Cascade = getCascade(Reg);
      if (!Cascade)
        Cascade = NextCascade;
      return Cascade;
    }

    template <typename Iterator>
    void setStage(Iterator Begin, Iterator End, LiveRangeStage NewStage) {
      for (; Begin != End; ++Begin) {
        Register Reg = *Begin;
        Info.grow(Reg.id());
        if (Info[Reg].Stage == RS_New)
          Info[Reg].Stage = NewStage;
      }
    }
    void LRE_DidCloneVirtReg(Register New, Register Old);
  };

public:

  static char ID; // 静态成员声明

  RegAllocSegmentTree();
  ~RegAllocSegmentTree() override;

  // MachineFunctionPass interface
  bool runOnMachineFunction(MachineFunction &mf) override;
  StringRef getPassName() const override { return "Segment Tree Register Allocator"; }

  /// 獲取分析依賴
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  // 我們需要覆蓋這個方法來初始化我們的自定義數據結構
  void init(VirtRegMap &vrm, LiveIntervals &lis, LiveRegMatrix &mat);

  /// Finalize allocation and verify results
  void finalizeAlloc(MachineFunction &MF, LiveIntervals &LIS, VirtRegMap &VRM) const;

  // 你堅持要的同名接口（注意：**不要**寫 override）
  void postOptimization(Spiller &VRegSpiller, LiveIntervals &LIS);

#ifndef NDEBUG
  bool verifyAllocation(MachineFunction &MF, LiveIntervals &LIS,
                        VirtRegMap &VRM);
#endif

  // RegAllocBase interface
  Spiller &spiller() override { return *SpillerInstance; };
  void enqueueImpl(const LiveInterval *LI) override;
  const LiveInterval *dequeue() override;
  MCRegister selectOrSplit(const LiveInterval &VirtReg,
                           SmallVectorImpl<Register> &splitLVRs) override;
  MCRegister selectOrSplitAdvanced(const LiveInterval &VirtReg,
                           SmallVectorImpl<Register> &splitLVRs);

private:
  // Convenient shortcuts.
  using PQueue = std::priority_queue<std::pair<unsigned, unsigned>>;
  using SmallLISet = SmallSetVector<const LiveInterval *, 4>;

  MCRegister selectOrSplitImpl(const LiveInterval &,
                               SmallVectorImpl<Register> &, SmallVirtRegSet &,
                               RecoloringStack &, unsigned = 0);

  void enqueue(PQueue &CurQueue, const LiveInterval *LI);
  const LiveInterval *dequeue(PQueue &CurQueue);

  // Timer group constants
  static const char TimerGroupName[];
  static const char TimerGroupDescription[];

  // Timer group
  std::unique_ptr<TimerGroup> TimerGroupObj;
  // Timers for different phases
  std::unique_ptr<Timer> PrecomputeTimer;
  std::unique_ptr<Timer> AllocationTimer;
  std::unique_ptr<Timer> SpillTimer;
  std::unique_ptr<Timer> CleanupTimer;
  std::unique_ptr<Timer> PostOptTimer;
  std::unique_ptr<Timer> SegTreeUpdateTimer;
  std::unique_ptr<Timer> SegTreeQueryTimer;
  std::unique_ptr<Timer> GlobalSplitTimer;
  std::unique_ptr<Timer> LocalSplitTimer;
  std::unique_ptr<Timer> RegionSplitTimer;
  std::unique_ptr<Timer> BlockSplitTimer;

  double TotalPrecomputeTime = 0;
  double TotalAllocationTime = 0;
  double TotalSpillTime = 0;
  double TotalCleanupTime = 0;
  double TotalPostOptTime = 0;
  double TotalSegTreeUpdateTime = 0;
  double TotalSegTreeQueryTime = 0;
  
  // Performance statistics
  mutable unsigned NumInterferenceChecks = 0;
  mutable unsigned NumSegTreeUpdates = 0;
  mutable unsigned NumSegTreeUpdatesReal = 0;
  mutable unsigned NumCoordRebuilds = 0;
  mutable unsigned NumAllocAttempts = 0;
  mutable unsigned NumSpills = 0;

  uint64_t NumSegTreeUpdatesRealLocal = 0; // 每個函式的成功更新次數

  unsigned TotalAllocAttempts = 0;
  unsigned TotalInterferenceChecks = 0;
  unsigned TotalSegTreeUpdates = 0;
  unsigned TotalSegTreeUpdatesReal = 0;
  unsigned TotalCoordRebuilds = 0;
  unsigned TotalSpills = 0;

  void resetAllocatorState();  // ← 新增

  // [NEW] 是否啟用 precompute（由 .cpp 的 cl::opt 注入）
  bool UsePrecompute = true;

  // [NEW] 讓 runOnMachineFunction 能用同一個入口呼叫（包計時）
  void doPrecomputeIfEnabled();

  void precomputeAllCoordinates();
  void precomputeGlobalCoords();
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
  PQueue Queue;
  std::optional<ExtraRegInfo> ExtraInfo;

  std::unique_ptr<RegAllocPriorityAdvisor> PriorityAdvisor;

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
  std::vector<SlotIndex> GlobalCoords; // 全局座標點（所有 PR 共用）

  // 每個 PhysReg 的 lazy 線段樹
  std::vector<std::vector<SegNode>> PRTree;

  // —— 內部工具 —— //
  // 確保 PR 的座標包含 [S,E) 的端點；必要時重建樹並回放既有區間
  // void ensureCoordsAndTree(unsigned PhysReg);
  void ensureCoordsAndTree(unsigned PhysReg, SlotIndex S, SlotIndex E);
  void ensureCoordsForPhysReg(unsigned PhysReg);
  void dumpCoords(unsigned PhysReg) const;
  // 把 SlotIndex 映射成座標索引（必須已存在於 PRCoords[PR]）
  unsigned coordIndex(unsigned PhysReg, SlotIndex X) const;

  // 線段樹操作
  void segtreeBuild(unsigned PhysReg);
  void segtreeUpdate(unsigned PhysReg, unsigned idx, unsigned L, unsigned R,
                     unsigned ql, unsigned qr, int add);
  int  segtreeQueryMax(unsigned PhysReg, unsigned idx, unsigned L, unsigned R,
                       unsigned ql, unsigned qr);
  int segtreeQueryMaxIter(unsigned PhysReg, unsigned ql, unsigned qr);

  // 可放檢查（含別名）：任一段 maxCover>0 就不可放
  bool canPlaceOnPhysReg(unsigned PhysReg, const LiveInterval &LI) const;

  // 嘗試為給定的虛擬暫存器分配一個物理暫存器
  // 返回分配的物理暫存器，若失敗則返回0
  unsigned tryAllocateRegister(LiveInterval &VirtReg);

    // ===== 分割相關方法群組 =====
  // 線段樹輔助分割策略
  MCRegister trySegmentTreeSplit(const LiveInterval &VirtReg,
                                 SmallVectorImpl<Register> &NewVRegs);

  // 基於線段樹分析找到最佳分割點
  SlotIndex findOptimalSplitPoint(const LiveInterval &VirtReg);
  
  // 在指定點執行分割
  MCRegister performSplitAtPoint(const LiveInterval &VirtReg, 
                                SlotIndex SplitPoint,
                                SmallVectorImpl<Register> &NewVRegs);

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
};

// 創建SegmentTreeRegisterAllocator實例的函數
FunctionPass *createRegAllocSegmentTree();

} // end namespace llvm

#endif // LLVM_CODEGEN_REGALLOCSEGMENTTREE_H
