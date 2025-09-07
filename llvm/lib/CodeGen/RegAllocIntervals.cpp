//===- RegAllocIntervals.cpp - Interval-set Register Allocator ------------===//
//
// A simple register allocator using IntervalSet instead of SegmentTree.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regallocintervals"

#include "llvm/CodeGen/RegAllocIntervals.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRangeEdit.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/InitializePasses.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

char RegAllocIntervals::ID = 0;

namespace llvm {
  void initializeRegAllocIntervalsPass(PassRegistry &Registry);
}

INITIALIZE_PASS_BEGIN(RegAllocIntervals, "regallocintervals",
                      "Interval-set Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(SlotIndexesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveStacksWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(LiveRegMatrixWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_END(RegAllocIntervals, "regallocintervals",
                    "Interval-set Register Allocator", false, false)

static llvm::RegisterRegAlloc
  RAReg("intervals", "Interval-set Register Allocator", llvm::createRegAllocIntervals);

RegAllocIntervals::RegAllocIntervals() : MachineFunctionPass(ID), 
                                         MF(nullptr), LIS(nullptr), 
                                         VRM(nullptr), MRI(nullptr), 
                                         TRI(nullptr) {}

void RegAllocIntervals::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LiveIntervalsWrapperPass>();
  AU.addRequired<VirtRegMapWrapperLegacy>();
  AU.addRequired<LiveRegMatrixWrapperLegacy>();
  AU.setPreservesAll();   // ← 這行保證後續 pass 不會把分析重建掉

  MachineFunctionPass::getAnalysisUsage(AU);
}

bool RegAllocIntervals::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  MRI = &MF->getRegInfo();
  TRI = MF->getSubtarget().getRegisterInfo();

  auto &LRM = getAnalysis<LiveRegMatrixWrapperLegacy>().getLRM();
  init(*VRM, *LIS, LRM);

  MRI->freezeReservedRegs();
  RegUnitIntervals.clear();
  RegUnitIntervals.resize(TRI->getNumRegUnits());  // ★ 用 reg-unit 數量

  LLVM_DEBUG(dbgs() << "Running Interval-set RA on " << MF->getName() << "\n");

  // 添加实际的寄存器分配循环
  allocatePhysRegs();

  // 最終檢查：確保所有虛擬寄存器都已分配
  unsigned Unmapped = 0;
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (MRI->reg_nodbg_empty(Reg)) continue;
    if (!VRM->hasPhys(Reg)) {
      Unmapped++;
      errs() << "[intervals] ERROR: vreg " << Reg << " still not allocated\n";
      
      // 嘗試從使用中獲取更多信息
      for (const MachineOperand &MO : MRI->reg_operands(Reg)) {
        if (MO.isDef()) {
          const MachineInstr *MI = MO.getParent();
          errs() << "  Defined in: ";
          MI->print(errs());
        } else {
          const MachineInstr *MI = MO.getParent();
          errs() << "  Used in: ";
          MI->print(errs());
        }
      }
    }
  }

  // 在 runOnMachineFunction 的末尾，修改緊急分配部分
  if (Unmapped > 0) {
    errs() << "[intervals] " << Unmapped << " vregs were not allocated\n";
    
    // 使用更全面的方法處理未分配的虛擬寄存器
    for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
      Register Reg = Register::index2VirtReg(i);
      if (MRI->reg_nodbg_empty(Reg)) continue;
      if (!VRM->hasPhys(Reg)) {
        LiveInterval &LI = LIS->getInterval(Reg);
        const TargetRegisterClass *RC = MRI->getRegClass(Reg);
        
        if (!RC) {
          RC = TRI->getLargestLegalSuperClass(TRI->getRegClass(0), *MF);
          if (!RC) RC = TRI->getRegClass(0);
        }
        
        // 尋找可用的物理寄存器
        MCPhysReg PhysReg = 0;
        ArrayRef<MCPhysReg> Order = RC->getRawAllocationOrder(*MF);
        for (MCPhysReg Candidate : Order) {
          if (Candidate != 0 && !MRI->isReserved(Candidate)) {
            PhysReg = Candidate;
            break;
          }
        }
        
        if (PhysReg == 0 && !Order.empty()) {
          PhysReg = Order[0];
        }
        
        if (PhysReg != 0) {
          VRM->assignVirt2Phys(Reg, PhysReg);
          updatePhysReg(PhysReg, LI);
          errs() << "[intervals] Emergency allocation of vreg " << Reg 
                << " to preg " << PhysReg << "\n";
        } else {
          errs() << "[intervals] ERROR: Cannot find a physical register for vreg " << Reg << "\n";
        }
      }
    }
  }

#ifndef NDEBUG
  auto verifyAllMapped = [&]{
    for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
      Register V = Register::index2VirtReg(i);
      if (MRI->reg_nodbg_empty(V)) continue;
      if (!VRM->hasPhys(V)) {
        errs() << "[intervals][verify] unmapped vreg " << V << "\n";
        report_fatal_error("intervals RA: unmapped vreg before rewriter");
      }
      const TargetRegisterClass *RC = MRI->getRegClass(V);
      unsigned P = VRM->getPhys(V);
      if (RC && !RC->contains(P)) {
        errs() << "[intervals][verify] class mismatch: vreg " << V
               << " -> P" << P << " not in " << TRI->getRegClassName(RC) << "\n";
        report_fatal_error("intervals RA: class mismatch");
      }
    }
  };
  verifyAllMapped();
#endif

  return true;
}

// 实现 allocatePhysRegs 方法

void RegAllocIntervals::allocatePhysRegs() {
  // 使用更可靠的方法收集所有虛擬寄存器
  std::set<Register> AllVirtRegs;
  
  // 方法1: 從MRI獲取所有虛擬寄存器
  for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
    Register Reg = Register::index2VirtReg(i);
    if (!MRI->reg_nodbg_empty(Reg)) {
      AllVirtRegs.insert(Reg);
    }
  }
  
  // 方法2: 掃描所有指令中的虛擬寄存器操作數
  for (const MachineBasicBlock &MBB : *MF) {
    for (const MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.getReg().isVirtual()) {
          AllVirtRegs.insert(MO.getReg());
        }
      }
    }
  }
  
  errs() << "Found " << AllVirtRegs.size() << " virtual registers to allocate\n";
  
  // 處理每個虛擬寄存器
  for (Register VirtReg : AllVirtRegs) {
    if (VRM->hasPhys(VirtReg)) continue;
    
    // 獲取或創建LiveInterval
    LiveInterval *LI = nullptr;
    if (LIS->hasInterval(VirtReg)) {
      LI = &LIS->getInterval(VirtReg);
    } else {
      // 對於沒有LiveInterval的寄存器，創建一個
      LIS->createAndComputeVirtRegInterval(VirtReg);
      LI = &LIS->getInterval(VirtReg);
    }
    
    // 嘗試獲取寄存器類別
    const TargetRegisterClass *RC = MRI->getRegClass(VirtReg);
    
    // 如果無法獲取寄存器類別，嘗試從指令中推斷
    if (!RC) {
      for (const MachineOperand &MO : MRI->reg_operands(VirtReg)) {
        const MachineInstr *MI = MO.getParent();
        for (unsigned i = 0; i < MI->getNumOperands(); ++i) {
          if (&MI->getOperand(i) == &MO) {
            const MCInstrDesc &Desc = MI->getDesc();
            if (i < Desc.getNumOperands()) {
              int RCIdx = Desc.operands()[i].RegClass;
              if (RCIdx >= 0) {
                RC = TRI->getRegClass(RCIdx);
                break;
              }
            }
          }
        }
        if (RC) break;
      }
    }
    
    // 如果仍然無法確定寄存器類別，使用默認類別
    if (!RC) {
      RC = TRI->getLargestLegalSuperClass(TRI->getRegClass(0), *MF);
      if (!RC) {
        RC = TRI->getRegClass(0);
      }
    }
    
    // 嘗試分配物理寄存器
    // 用區間不重疊檢查來挑 physreg
    unsigned PhysReg = tryAllocateRegister(*LI);
    // 1) 真的找不到 → 立刻報錯（或改成呼叫 spiller，現在你是骨架先報錯）
    if (!PhysReg) {
      errs() << "[intervals] no legal physreg for vreg " << VirtReg
            << " (class " << TRI->getRegClassName(MRI->getRegClass(VirtReg)) << ")\n";
      MF->print(errs());
      report_fatal_error("intervals RA: allocation failed");
    }

    // 2) 類別一致性保險（這要在 if (!PhysReg) 外面）
    if (RC && !RC->contains(PhysReg)) {
      errs() << "[intervals] picked preg " << PhysReg << " not in class "
            << TRI->getRegClassName(RC) << " for vreg " << VirtReg << "\n";
      report_fatal_error("intervals RA: class mismatch");
    }

    // 3) 指派並更新 reg-unit 占用
    if (!VRM->hasPhys(VirtReg)) {
      VRM->assignVirt2Phys(VirtReg, PhysReg);
      updatePhysReg(PhysReg, *LI);
      LLVM_DEBUG(dbgs() << "Assigned vreg " << VirtReg << " to preg " << PhysReg << "\n");    
    }  
  }

#ifndef NDEBUG
  auto verifyAllMappedLocal = [&]{
    for (unsigned i = 0, e = MRI->getNumVirtRegs(); i != e; ++i) {
      Register V = Register::index2VirtReg(i);
      if (MRI->reg_nodbg_empty(V)) continue;
      if (!VRM->hasPhys(V)) {
        errs() << "[intervals][verify] unmapped vreg " << V << "\n";
        report_fatal_error("intervals RA: unmapped vreg before rewriter (post-allocatePhysRegs)");
      }
      const TargetRegisterClass *RC = MRI->getRegClass(V);
      unsigned P = VRM->getPhys(V);
      if (RC && !RC->contains(P)) {
        errs() << "[intervals][verify] class mismatch: vreg " << V
               << " -> P" << P << " not in " << TRI->getRegClassName(RC) << "\n";
        report_fatal_error("intervals RA: class mismatch (post-allocatePhysRegs)");
      }
    }
  };
  verifyAllMappedLocal();
#endif

// ★★ 最兇、最可靠的驗證：掃整個 MF 的 MI 操作數 ★★
#ifndef NDEBUG
  for (const MachineBasicBlock &MBB : *MF) {
    for (const MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg()) continue;
        Register V = MO.getReg();
        if (!V.isVirtual()) continue;

        if (!VRM->hasPhys(V)) {
          errs() << "[intervals][post-alloc check] unmapped vreg " << V
                << " in instruction: ";
          MI.print(errs());
          report_fatal_error("intervals RA: unmapped vreg slipped through");
        }

        const TargetRegisterClass *RC = MRI->getRegClass(V);
        unsigned P = VRM->getPhys(V);
        if (RC && !RC->contains(P)) {
          errs() << "[intervals][post-alloc check] class mismatch: vreg " << V
                << " -> P" << P << " not in " << TRI->getRegClassName(RC)
                << " in instruction: ";
          MI.print(errs());
          report_fatal_error("intervals RA: class mismatch after allocation");
        }
      }
    }
  }
#endif

#ifndef NDEBUG
for (const MachineBasicBlock &MBB : *MF) {
  for (const MachineInstr &MI : MBB) {
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg()) continue;
      Register V = MO.getReg();
      if (!V.isVirtual()) continue;

      if (!VRM->hasPhys(V)) {
        errs() << "[intervals][post-alloc check] unmapped vreg " << V
               << " in instruction: ";
        MI.print(errs());
        report_fatal_error("intervals RA: unmapped vreg slipped through");
      }
      if (const TargetRegisterClass *RC = MRI->getRegClass(V)) {
        unsigned P = VRM->getPhys(V);
        if (!RC->contains(P)) {
          errs() << "[intervals][post-alloc check] class mismatch: vreg " << V
                 << " -> P" << P << " not in " << TRI->getRegClassName(RC)
                 << " in instruction: ";
          MI.print(errs());
          report_fatal_error("intervals RA: class mismatch after allocation");
        }
      }
    }
  }
}
#endif
}

unsigned RegAllocIntervals::tryEvictOrAllocate(LiveInterval &LI) {
  const TargetRegisterClass *RC = MRI->getRegClass(LI.reg());
  if (!RC) return 0;
  
  // Find the physical register with minimum conflicts
  unsigned BestReg = 0;
  unsigned MinConflicts = UINT_MAX;
  
  for (unsigned PReg : *RC) {
    if (PReg == 0 || MRI->isReserved(PReg)) continue;
    bool conflict = false;
    // 檢查這顆實體暫存器的所有 reg units 是否衝突
    for (MCRegUnitIterator Unit(PReg, TRI); Unit.isValid(); ++Unit) {
      auto &Set = PhysRegIntervals[*Unit]; // 把索引改成「reg unit id」
      for (auto &Seg : LI) {
        if (Set.overlaps(Seg.start, Seg.end)) { conflict = true; break; }
      }
      if (conflict) break;
    }
    if (!conflict) return PReg;
  }
  
  // If we have a register with conflicts, evict the conflicting intervals
  if (BestReg != 0) {
    // Clear the intervals for this physical register
    // In a real implementation, you'd need to unassign the conflicting vregs
    PhysRegIntervals[BestReg].clear();
    return BestReg;
  }
  
  return 0;
}

void RegAllocIntervals::handleSpill(Register VirtReg) {
  // Check if already assigned
  if (VRM->hasPhys(VirtReg)) {
    LLVM_DEBUG(dbgs() << "VirtReg " << VirtReg << " already assigned\n");
    return;
  }
  
  const TargetRegisterClass *RC = MRI->getRegClass(VirtReg);
  
  if (RC) {
    // Try to find any register in the class, even if it causes conflicts
    for (unsigned PReg : *RC) {
      if (PReg != 0 && !MRI->isReserved(PReg)) {
        VRM->assignVirt2Phys(VirtReg, PReg);
        LLVM_DEBUG(dbgs() << "Spill assignment of vreg " << VirtReg 
                          << " to preg " << PReg << " (may conflict)\n");
        return;
      }
    }
  }
  
  // Absolute fallback - pick first unreserved register
  for (unsigned P = 1, N = TRI->getNumRegs(); P < N; ++P) {
    if (!MRI->isReserved(P)) {
      VRM->assignVirt2Phys(VirtReg, P);
      LLVM_DEBUG(dbgs() << "Emergency spill assignment of vreg " 
                        << VirtReg << " to preg " << P << "\n");
      return;
    }
  }
  
  // Last resort - use any register
  for (unsigned P = 1, N = TRI->getNumRegs(); P < N; ++P) {
    VRM->assignVirt2Phys(VirtReg, P);
    LLVM_DEBUG(dbgs() << "Last resort assignment of vreg " 
                      << VirtReg << " to preg " << P << "\n");
    return;
  }
}

unsigned RegAllocIntervals::tryAllocateRegister(LiveInterval &LI) {
  const TargetRegisterClass *RC = MRI->getRegClass(LI.reg());
  if (!RC) return 0;

  for (MCPhysReg PReg : *RC) {
    if (!PReg || MRI->isReserved(PReg)) continue;

    bool conflict = false;
    for (MCRegUnitIterator U(PReg, TRI); U.isValid(); ++U) {
      auto &Set = RegUnitIntervals[*U];
      for (const auto &Seg : LI) {
        if (Set.overlaps(Seg.start, Seg.end)) { conflict = true; break; }
      }
      if (conflict) break;
    }
    if (!conflict) return PReg;
  }
  return 0;
}

void RegAllocIntervals::updatePhysReg(unsigned PhysReg, const LiveInterval &LI) {
  for (MCRegUnitIterator U(PhysReg, TRI); U.isValid(); ++U) {
    auto &Set = RegUnitIntervals[*U];
    for (const auto &Seg : LI) Set.add(Seg.start, Seg.end);
  }
}

void RegAllocIntervals::enqueueImpl(const LiveInterval *LI) {
  Q.push(std::make_pair(LI->getSize(), ~LI->reg().id()));
}

const LiveInterval *RegAllocIntervals::dequeue() {
  if (Q.empty()) return nullptr;
  unsigned VReg = ~Q.top().second;
  Q.pop();
  return &LIS->getInterval(Register(VReg));
}

MCRegister RegAllocIntervals::selectOrSplit(
    const LiveInterval &VirtReg,
    SmallVectorImpl<Register> &SplitVRegs) {
  // 簡單實現：總是返回 NoRegister，讓主循環處理強制分配
  return MCRegister::NoRegister;
}

FunctionPass *llvm::createRegAllocIntervals() { 
  return new RegAllocIntervals(); 
}