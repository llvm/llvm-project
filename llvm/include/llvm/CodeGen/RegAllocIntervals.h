#ifndef LLVM_CODEGEN_REGALLOCINTERVALS_H
#define LLVM_CODEGEN_REGALLOCINTERVALS_H

#include "RegAllocBase.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/CodeGen/Spiller.h"
#include "llvm/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include <map>
#include <queue>

namespace llvm {

class Spiller;

/// 簡單的 [L,R) 區間集合（互斥、合併）
struct IntervalSet {
  std::map<SlotIndex, SlotIndex> S;

  // Clear all intervals
  void clear() {
    S.clear();  // Clear the map that stores intervals
  }

  bool overlaps(SlotIndex L, SlotIndex R) const {
    if (!L.isValid() || !R.isValid() || !(L < R)) return false;
    auto it = S.lower_bound(L);
    if (it != S.begin()) {
      auto pit = std::prev(it);
      if (pit->second > L) return true;
    }
    while (it != S.end() && it->first < R) return true;
    return false;
  }

  void add(SlotIndex L, SlotIndex R) {
    if (!L.isValid() || !R.isValid() || !(L < R)) return;
    auto it   = S.lower_bound(L);
    auto newL = L, newR = R;
    if (it != S.begin()) {
      auto pit = std::prev(it);
      if (pit->second >= L) {
        newL = std::min(newL, pit->first);
        newR = std::max(newR, pit->second);
        it = S.erase(pit);
      }
    }
    while (it != S.end() && it->first <= newR) {
      newL = std::min(newL, it->first);
      newR = std::max(newR, it->second);
      it = S.erase(it);
    }
    S.emplace(newL, newR);
  }
};

/// 極簡骨架：用 IntervalSet 思路，暫不做真正分配。
class RegAllocIntervals : public MachineFunctionPass, public RegAllocBase {
public:
  static char ID;
  RegAllocIntervals();

  StringRef getPassName() const override {
    return "Interval-set Register Allocator (skeleton)";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;

  // === RegAllocBase 純虛覆寫（最小骨架） ===
  Spiller &spiller() override { return *SpillerInstance; }
  void enqueueImpl(const LiveInterval *LI) override;
  const LiveInterval *dequeue() override;
  MCRegister selectOrSplit(const LiveInterval &VirtReg,
                           SmallVectorImpl<Register> &SplitVRegs) override;

  // Add missing method declarations
  void allocatePhysRegs();
  void handleSpill(Register VirtReg);
  unsigned tryEvictOrAllocate(LiveInterval &LI);
  unsigned tryAllocateRegister(LiveInterval &LI);
  void updatePhysReg(unsigned PhysReg, const LiveInterval &LI);

private:
  // Add missing member variables
  MachineFunction *MF;
  LiveIntervals *LIS;
  VirtRegMap *VRM;
  MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;
  std::vector<IntervalSet> RegUnitIntervals; // 以 reg-unit id
  
  // Physical register intervals
  std::vector<IntervalSet> PhysRegIntervals;

  std::unique_ptr<Spiller> SpillerInstance;

  // 最簡的 queue：按 (size, ~vreg) 排
  using PQueueTy = std::priority_queue<std::pair<unsigned, unsigned>>;
  PQueueTy Q;

  // 每個 physreg 一份占用集合（這只是骨架，尚未真正使用）
  DenseMap<unsigned, IntervalSet> PRUse;
};

/// 工廠函式給 RegisterRegAlloc 使用
FunctionPass *createRegAllocIntervals();

// Forward declare the initializer function
void initializeRegAllocIntervalsPass(PassRegistry &);

} // end namespace llvm

#endif // LLVM_CODEGEN_REGALLOCINTERVALS_H