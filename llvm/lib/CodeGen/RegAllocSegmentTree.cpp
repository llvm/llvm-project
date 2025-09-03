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
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

using namespace llvm;

#define DEBUG_TYPE "regallocsegtre"

char RegAllocSegmentTree::ID = 0;

// 讓 llc/clang 能用 -regalloc=segtre 選到你
static llvm::RegisterRegAlloc
  RAReg("segtre",                           // 選項名稱
        "Segment Tree Register Allocator",  // 說明文字
        llvm::createRegAllocSegmentTree);   // 工廠函式（見下一步）

RegAllocSegmentTree::RegAllocSegmentTree()
  : MachineFunctionPass(ID) {
    initializeSlotIndexesWrapperPassPass(*PassRegistry::getPassRegistry());
    initializeLiveIntervalsWrapperPassPass(*PassRegistry::getPassRegistry());
    initializeLiveStacksWrapperLegacyPass(*PassRegistry::getPassRegistry());
    initializeVirtRegMapWrapperLegacyPass(*PassRegistry::getPassRegistry());
  }

bool RegAllocSegmentTree::runOnMachineFunction(MachineFunction &MF) {
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
  VRegSpiller = std::unique_ptr<Spiller>(
    createInlineSpiller(RA, MF, *VRM, VRAI));

  MF.getRegInfo().freezeReservedRegs();

  LLVM_DEBUG(dbgs() << "Segment Tree Register Allocating for " << MF.getName() << "\n");

  // 初始化RegAllocBase
  auto &LRM = getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
  init(*VRM, *LIS, LRM);

  // 防呆：此時 MRI 已在 init() 裡設為成員指標
  assert(this->MF && LIS && VRM && MRI &&
       "RA pointers must be initialized before allocation");
  assert(VRegSpiller && "Spiller must be created before allocation");

  // 找到需要分配的虚拟寄存器区间
  std::vector<LiveInterval*> VRegsToAlloc;
  for (unsigned i = 0, e = MF.getRegInfo().getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = Register::index2VirtReg(i);
    if (MF.getRegInfo().reg_nodbg_empty(Reg))
      continue;
    LiveInterval *LI = &LIS->getInterval(Reg);
    if (LI->empty())
      continue;
    VRegsToAlloc.push_back(LI);
  }

  // 如果没有需要分配的虚拟寄存器，直接返回
  if (VRegsToAlloc.empty()) {
    finalizeAlloc(MF, *LIS, *VRM);
    postOptimization(*VRegSpiller, *LIS);
    return true;
  }

  // 对虚拟寄存器进行排序（例如，按生命期长度、权重等）
  // 这是一个重要的设计决策点！
  llvm::sort(VRegsToAlloc, [](const LiveInterval *A, const LiveInterval *B) {
    return A->getSize() > B->getSize();
  });

  // 主要的分配循环
  bool AllocComplete = false;
  unsigned Round = 0;
  const unsigned MaxRounds = 5; // 防止无限循环

  while (!AllocComplete && Round < MaxRounds) {
    LLVM_DEBUG(dbgs() << "  Segment Tree Regalloc round " << Round << ":\n");
    (void) Round;

    // 重置分配状态（如果需要）
    // 例如，清除之前的分配结果

    // 尝试分配所有虚拟寄存器
    bool HasSpills = false;
    for (LiveInterval *LI : VRegsToAlloc) {
      unsigned VirtReg = LI->reg();
      if (VRM->hasPhys(VirtReg)) // 可能已经通过合并分配了
        continue;

      // 尝试分配物理寄存器
      unsigned PhysReg = tryAllocateRegister(*LI);

      if (PhysReg) {
        // 分配成功，更新线段树和映射
        VRM->assignVirt2Phys(VirtReg, PhysReg);
        updateSegmentTreeForInterval(*LI, PhysReg);
      } else {
        // 分配失败，需要溢出
        HasSpills = true;
        spillVirtReg(*LI);
      }
    }

    // 如果没有溢出，分配完成
    if (!HasSpills) {
      AllocComplete = true;
    } else {
      // 处理溢出后，可能需要重新计算LiveIntervals并重新分配
      // 这里需要实现溢出后的处理逻辑
      
      // 重新收集需要分配的虚拟寄存器
      VRegsToAlloc.clear();
      for (unsigned i = 0, e = MF.getRegInfo().getNumVirtRegs(); i != e; ++i) {
        unsigned Reg = Register::index2VirtReg(i);
        if (MF.getRegInfo().reg_nodbg_empty(Reg))
          continue;
        LiveInterval *LI = &LIS->getInterval(Reg);
        if (LI->empty())
          continue;
        VRegsToAlloc.push_back(LI);
      }
      
      // 重新排序
      llvm::sort(VRegsToAlloc, [](const LiveInterval *A, const LiveInterval *B) {
        return A->getSize() > B->getSize();
      });
    }

    ++Round;
  }

  // 最终化分配，分配空范围
  finalizeAlloc(MF, *LIS, *VRM);
  postOptimization(*VRegSpiller, *LIS);

  LLVM_DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << VRM << "\n");

  return true;
}

void RegAllocSegmentTree::updateSegmentTreeForInterval(const LiveInterval &LI, unsigned PhysReg) {
  // 更新线段树以反映区间的分配
  for (const auto &Seg : LI) {
    updateSegmentTreeForPhysReg(PhysReg, Seg.start, Seg.end);
  }
}

Spiller &RegAllocSegmentTree::spiller() {
  return *VRegSpiller;  // Use your member variable
}

void RegAllocSegmentTree::enqueueImpl(const LiveInterval *LI) {
  // 简单实现：暂时不做任何操作
}

const LiveInterval *RegAllocSegmentTree::dequeue() {
  // 返回nullptr或简单实现
  return nullptr;
}

MCRegister RegAllocSegmentTree::selectOrSplit(const LiveInterval &VirtReg,
                         SmallVectorImpl<Register> &splitLVRs) {
  // 实现你的selectOrSplit逻辑
  return 0;
}

void RegAllocSegmentTree::init(VirtRegMap &vrm, LiveIntervals &lis,
                                        LiveRegMatrix &mat) {
  // 首先调用基类的init方法
  RegAllocBase::init(vrm, lis, mat);

  // 然后获取需要的其他信息
  // 通过vrm获取MachineFunction
  MF = &vrm.getMachineFunction();
  
  // 在这里初始化我们自己的数据结构
  // 1. 获取目标机器的物理寄存器信息
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
  
  // 初始化线段树数据结构
  // 为所有物理寄存器分配线段树空间
  unsigned NumPhysRegs = TRI->getNumRegs();
  PhysRegSegmentTrees.clear();
  PhysRegSegmentTrees.resize(NumPhysRegs);

  PhysRegIntervals.clear();
  PhysRegIntervals.assign(NumPhysRegs, {});

  // ★ 初始化座標與 lazy 線段樹容器（每個 physreg 一份）
  PRCoords.clear();
  PRCoords.assign(NumPhysRegs, {});
  PRTree.clear();
  PRTree.assign(NumPhysRegs, {});

  this->MRI = &MF->getRegInfo();

  // 2. 為每個暫存器類別初始化線段樹
  //    這是一個簡化的示例。實際實現中，您需要遍歷所有可分配的物理暫存器
  for (unsigned RCId = 0; RCId < TRI->getNumRegClasses(); ++RCId) {
    // 獲取該暫存器類別中的所有物理暫存器
    // 初始化對應的線段樹...
    // PhysRegSegmentTrees[RCId].resize(...);
  }

  LLVM_DEBUG(dbgs() << "Initializing Segment Tree Register Allocator with " 
                    << NumPhysRegs << " physical registers\n");
}

void RegAllocSegmentTree::allocatePhysRegs() {
  // 主要的分配循環
  // 1. 我們需要對虛擬暫存器進行排序（例如，按生命期長度、權重等）
  //    這對Segment Tree分配器的效能至關重要
  std::vector<LiveInterval*> VirtRegs;
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = Register::index2VirtReg(i);
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
    unsigned VirtReg = LI->reg();
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

unsigned RegAllocSegmentTree::tryAllocateRegister(LiveInterval &VirtReg) {
  assert(MF && LIS && VRM && MRI &&
         "RA pointers must be initialized before allocation (tryAllocateRegister)");
  
  LLVM_DEBUG(dbgs() << "Trying to allocate register for virtreg " << VirtReg.reg() << "\n");
  
  // 1. 獲取此虛擬暫存器的暫存器類別
  const TargetRegisterClass *RC = MRI->getRegClass(VirtReg.reg());
  if (!RC) {
    LLVM_DEBUG(dbgs() << "Error: Cannot get register class for virtual register " << VirtReg.reg() << "\n");
    return 0;
  }

  // 获取目标寄存器信息
  const TargetRegisterInfo *TRI = MRI->getTargetRegisterInfo();
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
  if (PhysReg == 0) return false;
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();

  for (MCRegAliasIterator AI(PhysReg, TRI, /*IncludeSelf=*/true); AI.isValid(); ++AI) {
    unsigned A = *AI;
    // 對每個待查 segment 先確保座標，避免端點不在 coords 造成誤判
    for (const auto &Seg : LI) {
      ensureCoordsAndTree(A, Seg.start, Seg.end);
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

void RegAllocSegmentTree::buildSegmentTreeForPhysReg(unsigned PhysReg, const std::vector<SlotIndex>& intervals) {
  if (intervals.empty()) return;
  
  // 计算线段树大小
  unsigned n = intervals.size();
  unsigned size = 1;
  while (size < n) size <<= 1;
  size <<= 1;
  
  // 初始化线段树
  std::vector<SegmentTreeNode> tree(size);
  
  // 构建线段树
  // 这里需要实现实际的构建逻辑，可能需要递归构建
  buildSegmentTree(tree.data(), 1, 0, n-1, intervals);
  
  PhysRegSegmentTrees[PhysReg] = std::move(tree);
}

bool RegAllocSegmentTree::querySegmentTreeForRange(unsigned PhysReg, SlotIndex Start, SlotIndex End) const {
  if (PhysReg >= PhysRegSegmentTrees.size() || PhysRegSegmentTrees[PhysReg].empty()) {
    return true; // 该物理寄存器尚未分配任何区间，可用
  }
  
  // 使用线段树查询物理寄存器在[Start, End)区间是否已被占用
  const auto &tree = PhysRegSegmentTrees[PhysReg];
  unsigned n = tree.size() / 4; // 估计原始区间数量
  return querySegmentTree(tree.data(), 1, 0, n-1, Start, End);
}

void RegAllocSegmentTree::updateSegmentTreeForPhysReg(unsigned PhysReg, SlotIndex Start, SlotIndex End) {
  
  assert(Register::isPhysicalRegister(PhysReg) && "expected physreg");
  assert(PhysReg < PRCoords.size() && PhysReg < PRTree.size() &&
         "physreg index out of range");
  assert(Start.isValid() && End.isValid() && Start < End && "invalid segment");
  
  // 确保物理寄存器的坐标包含这个区间
  ensureCoordsAndTree(PhysReg, Start, End);
  
  // 获取坐标索引
  unsigned startIdx = coordIndex(PhysReg, Start);
  unsigned endIdx = coordIndex(PhysReg, End);
  
  // 更新线段树
  const unsigned pts = PRCoords[PhysReg].size();
  if (pts >= 2 && startIdx < endIdx) {
    const unsigned segN = pts - 1;              // 段數
    segtreeUpdate(PhysReg, 1, 0, segN - 1, startIdx, endIdx - 1, +1);
  }
  
  // 记录这个区间
  PhysRegIntervals[PhysReg].emplace_back(Start, End);
  
  LLVM_DEBUG(dbgs() << "Updated segment tree for physreg " << PhysReg 
                    << " with interval [" << Start << ", " << End << ")\n");
}

void RegAllocSegmentTree::ensureCoordsAndTree(unsigned PhysReg, SlotIndex S, SlotIndex E) {
  auto &Coords = PRCoords[PhysReg];
  auto &Tree = PRTree[PhysReg];
  
  // 检查是否需要添加新的坐标点
  bool changed = false;
  
  // 检查起点 S
  auto it = std::lower_bound(Coords.begin(), Coords.end(), S);
  if (it == Coords.end() || *it != S) {
    Coords.insert(it, S);
    changed = true;
  }
  
  // 检查终点 E
  it = std::lower_bound(Coords.begin(), Coords.end(), E);
  if (it == Coords.end() || *it != E) {
    Coords.insert(it, E);
    changed = true;
  }
  
  // 如果坐标发生了变化，需要重建线段树
  if (changed) {
    // 重建线段树
    segtreeBuild(PhysReg);
    
    // 重新应用所有已记录的区间
    const auto &Intervals = PhysRegIntervals[PhysReg];
    for (const auto &Interval : Intervals) {
      SlotIndex Start = Interval.first;
      SlotIndex End = Interval.second;
      
      // 获取坐标索引
      unsigned l = coordIndex(PhysReg, Start);
      unsigned r = coordIndex(PhysReg, End) - 1; // 线段树区间是 [l, r-1]
      
      // 更新线段树
      segtreeUpdate(PhysReg, 1, 0, Coords.size() - 2, l, r, 1);
    }
  }
}

unsigned RegAllocSegmentTree::coordIndex(unsigned PhysReg, SlotIndex X) const {
  const auto &Coords = PRCoords[PhysReg];
  auto It = std::lower_bound(Coords.begin(), Coords.end(), X);
  assert(It != Coords.end() && *It == X && "SlotIndex not found in coordinates");
  return It - Coords.begin();
}

void RegAllocSegmentTree::segtreeUpdate(unsigned PhysReg, unsigned idx,
                                        unsigned L, unsigned R,
                                        unsigned ql, unsigned qr, int add) {
  auto &Tree = PRTree[PhysReg];
  if (ql > R || qr < L)
    return;

  if (ql <= L && R <= qr) {
    Tree[idx].maxCover += add;
    Tree[idx].lazyAdd += add;
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

  Tree[idx].maxCover = std::max(Tree[LeftChild].maxCover, Tree[RightChild].maxCover);
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
}

void RegAllocSegmentTree::spillVirtReg(LiveInterval &VirtReg) {
  assert(MF && LIS && VRM &&
         "RA pointers must be initialized before spilling");

  LLVM_DEBUG(dbgs() << "Spilling virtual register: " << VirtReg.reg() << '\n');

  // Create a LiveRangeEdit object for the virtual register
  SmallVector<Register, 4> NewVRegs;
  LiveRangeEdit LRE(&VirtReg, NewVRegs, *MF, *LIS, VRM, this, &DeadRemats);
  
  // Use the spiller with the LiveRangeEdit object
  spiller().spill(LRE);
  
  // 將新生成的虛擬寄存器加入隊列
  for (Register NewVReg : NewVRegs) {
    if (!LIS->hasInterval(NewVReg))
      continue;
    LiveInterval &NewLI = LIS->getInterval(NewVReg);
    enqueue(&NewLI);
  }
}

void RegAllocSegmentTree::finalizeAlloc(MachineFunction &MF, 
                                                LiveIntervals &LIS, 
                                                VirtRegMap &VRM) {
  // 设置机器寄存器信息指针
  MRI = &MF.getRegInfo();
  
  // 处理空区间（没有实际生命期的虚拟寄存器）
  for (unsigned i = 0, e = MF.getRegInfo().getNumVirtRegs(); i != e; ++i) {
    unsigned Reg = Register::index2VirtReg(i);
    if (MF.getRegInfo().reg_nodbg_empty(Reg))
      continue;
    
    LiveInterval *LI = &LIS.getInterval(Reg);
    if (LI->empty() && !VRM.hasPhys(Reg)) {
      // 为空区间分配任意可用的物理寄存器
      const TargetRegisterClass *RC = MF.getRegInfo().getRegClass(Reg);
      if (RC->getNumRegs() > 0) {
        // 选择第一个可用的物理寄存器
        MCRegister PhysReg = *RC->begin();
        VRM.assignVirt2Phys(Reg, PhysReg);
        LLVM_DEBUG(dbgs() << "Assigned empty interval " << Reg 
                          << " to physical register " << PhysReg << "\n");
      }
    }
  }

  // 清理线段树数据结构
  PhysRegSegmentTrees.clear();
  
  // 验证分配结果
  LLVM_DEBUG({
    bool HasErrors = false;
    
    // 检查所有虚拟寄存器是否都已分配
    for (unsigned i = 0, e = MF.getRegInfo().getNumVirtRegs(); i != e; ++i) {
      unsigned Reg = Register::index2VirtReg(i);
      if (MF.getRegInfo().reg_nodbg_empty(Reg))
        continue;
      
      LiveInterval *LI = &LIS.getInterval(Reg);
      if (!LI->empty() && !VRM.hasPhys(Reg)) {
        dbgs() << "Error: Virtual register " << Reg 
               << " was not allocated!\n";
        HasErrors = true;
      }
    }
    
    // 检查物理寄存器冲突
    for (unsigned i = 0, e = MF.getRegInfo().getNumVirtRegs(); i != e; ++i) {
      unsigned Reg = Register::index2VirtReg(i);
      if (!VRM.hasPhys(Reg) || MF.getRegInfo().reg_nodbg_empty(Reg))
        continue;
      
      MCRegister PhysReg = VRM.getPhys(Reg);
      LiveInterval *LI = &LIS.getInterval(Reg);
      
      // 检查是否有其他虚拟寄存器分配到同一个物理寄存器
      for (unsigned j = i + 1; j < e; ++j) {
        unsigned OtherReg = Register::index2VirtReg(j);
        if (!VRM.hasPhys(OtherReg) || MF.getRegInfo().reg_nodbg_empty(OtherReg))
          continue;
        
        if (VRM.getPhys(OtherReg) == PhysReg) {
          LiveInterval *OtherLI = &LIS.getInterval(OtherReg);
          
          // 检查区间是否重叠
          if (LI->overlaps(*OtherLI)) {
            dbgs() << "Error: Virtual registers " << Reg << " and " << OtherReg
                   << " both assigned to physical register " << PhysReg
                   << " with overlapping live ranges!\n";
            HasErrors = true;
          }
        }
      }
    }
    
    if (!HasErrors) {
      dbgs() << "Segment Tree allocation successfully completed for " 
             << MF.getName() << "\n";
    } else {
      dbgs() << "Segment Tree allocation completed with errors for " 
             << MF.getName() << "\n";
    }
  });
  
  // 重置分配器状态，为下一个函数做准备
  resetAllocatorState();
}

void RegAllocSegmentTree::resetAllocatorState() {
  // 重置分配器状态，为下一个函数做准备
  PhysRegSegmentTrees.clear();
  PhysRegIntervals.clear();
  PRCoords.clear();
  PRTree.clear();
  DeadRemats.clear();
  FailedVRegs.clear();
}

void RegAllocSegmentTree::postOptimization(Spiller &VRegSpiller, LiveIntervals &LIS) {
  // 调用基类的 postOptimization 方法（如果存在）
  // RegAllocBase::postOptimization();
  
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
        enqueue(&SplitLI);
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

// 線段樹的構建、查詢、更新函數的具體實現
void RegAllocSegmentTree::buildSegmentTree(SegmentTreeNode *tree, unsigned idx,
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

bool RegAllocSegmentTree::querySegmentTree(const SegmentTreeNode *tree, unsigned idx,
                                           unsigned tree_l, unsigned tree_r,
                                           SlotIndex query_start, SlotIndex query_end) const {
  if (tree_r < tree_l) return true;
  
  // 如果当前节点区间与查询区间无重叠，返回 true（可用）
  if (query_end < tree[tree_l].MaxEnd || query_start > tree[tree_r].MaxEnd)
    return true;
  
  // 如果当前节点区间完全在查询区间内，检查是否冲突
  if (query_start <= tree[tree_l].MaxEnd && query_end >= tree[tree_r].MaxEnd) {
    return tree[idx].MaxEnd < query_start;
  }
  
  // 递归检查左右子树
  unsigned mid = (tree_l + tree_r) / 2;
  bool left_available = querySegmentTree(tree, 2*idx, tree_l, mid, query_start, query_end);
  bool right_available = querySegmentTree(tree, 2*idx+1, mid+1, tree_r, query_start, query_end);
  
  return left_available && right_available;
}

// 創建Pass實例的函數
llvm::FunctionPass *llvm::createRegAllocSegmentTree() {
  return new RegAllocSegmentTree();
}

// 注册SegmentTreeRegisterAllocator
// static llvm::RegisterRegAlloc segTreeRegAlloc("segtre", "segment tree register allocator",
//                                        []() -> FunctionPass* { 
//                                          LLVM_DEBUG(dbgs() << "Creating SegmentTreeRegisterAllocator\n");
//                                          return createRegAllocSegmentTree(); 
//                                        });
