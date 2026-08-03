//=== PISAOptimizeRedundantCopies.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Remove redundant COPY operations generated as result of previous passes.
//
// B = COPY A
// C = COPY B
// => B = COPY A
// => C = COPY A
//
// Subsequent DCE will eliminate 'B = COPY A' if B is no longer used.
//===----------------------------------------------------------------------===//

#include "PISA.h"
#include "PISAMCInstLower.h"
#include "PISASubtarget.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "pisa-optimize-redundant-copies"
#define DEBUG_NAME "PISA optimize redundant copies"

using namespace llvm;

namespace {

class PISAOptimizeRedundantCopies : public MachineFunctionPass {
public:
  static char ID;

  PISAOptimizeRedundantCopies();

  StringRef getPassName() const override { return DEBUG_NAME; }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // end anonymous namespace

char PISAOptimizeRedundantCopies::ID = 0;
INITIALIZE_PASS(PISAOptimizeRedundantCopies, DEBUG_TYPE, DEBUG_NAME, false,
                false)

// Erase RegMap entries for any register or subregister that is defined (written
// to) by MI. This ensures that if a register or any of its subregisters is
// redefined between COPYs, we do not propagate stale mappings. Example:
//   A = COPY B
//   B = INST ...         // B is redefined here
//   C = COPY A
//   => cannot replace C = COPY B
//
// Also handles cases where a subregister of B is modified:
//   A = COPY B
//   B.sub_0 = INST ...   // subregister of B is redefined here
//   C = COPY A
//   => cannot replace C = COPY B
static void eraseRegMapEntriesForDef(
    const MachineInstr &MI,
    SmallDenseMap<std::pair<Register, unsigned>, std::pair<Register, unsigned>>
        &RegMap) {
  for (const MachineOperand &Op : MI.operands()) {
    if (Op.isReg() && Op.isDef()) {
      Register WrittenReg = Op.getReg();
      RegMap.remove_if([WrittenReg](const auto &Entry) {
        return Entry.second.first == WrittenReg;
      });
    }
  }
}

// Erase RegMap entries for registers or subregisters that overlap with the
// destination of MI. This handles cases where an instruction overwrites part of
// a register, invalidating previous mappings. Example:
//   A = COPY B
//   A.sub_0 = INST C   // overwrites part of A
//   D = COPY A
//   => cannot replace D = COPY B
static void eraseRegMapEntriesForOverlap(
    const MachineInstr &MI,
    SmallDenseMap<std::pair<Register, unsigned>, std::pair<Register, unsigned>>
        &RegMap) {
  const static SmallDenseMap<unsigned, unsigned> OverLap = {
      {PISA::sub8_0, PISA::sub8_xy},   {PISA::sub8_1, PISA::sub8_xy},
      {PISA::sub8_2, PISA::sub8_zw},   {PISA::sub8_3, PISA::sub8_zw},
      {PISA::sub16_0, PISA::sub16_xy}, {PISA::sub16_1, PISA::sub16_xy},
      {PISA::sub16_2, PISA::sub16_zw}, {PISA::sub16_3, PISA::sub16_zw},
      {PISA::sub32_0, PISA::sub32_xy}, {PISA::sub32_1, PISA::sub32_xy},
      {PISA::sub32_2, PISA::sub32_zw}, {PISA::sub32_3, PISA::sub32_zw},
      {PISA::sub64_0, PISA::sub64_xy}, {PISA::sub64_1, PISA::sub64_xy},
      {PISA::sub64_2, PISA::sub64_zw}, {PISA::sub64_3, PISA::sub64_zw},
  };

  if (MI.getNumOperands() == 0 || !MI.getOperand(0).isReg())
    return;

  auto Dst = MI.getOperand(0);
  auto DstReg = Dst.getReg();
  auto DstSubReg = Dst.getSubReg();

  auto EraseOverlapReg = [&RegMap](Register DstReg, unsigned DstSubReg) {
    auto Key = std::make_pair(DstReg, DstSubReg);
    RegMap.erase(Key);
  };

  if (DstSubReg) {
    EraseOverlapReg(DstReg, DstSubReg);
    if (auto It = OverLap.find(DstSubReg); It != OverLap.end())
      EraseOverlapReg(DstReg, It->second); // overlap subreg
  }
  EraseOverlapReg(DstReg, 0); // overlap full reg
}

static void processInterveningInsts(
    const MachineInstr &MI,
    SmallDenseMap<std::pair<Register, unsigned>, std::pair<Register, unsigned>>
        &RegMap) {
  eraseRegMapEntriesForDef(MI, RegMap);
  eraseRegMapEntriesForOverlap(MI, RegMap);
}

void PISAOptimizeRedundantCopies::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

PISAOptimizeRedundantCopies::PISAOptimizeRedundantCopies()
    : MachineFunctionPass(ID) {
  initializePISAOptimizeRedundantCopiesPass(*PassRegistry::getPassRegistry());
}

bool PISAOptimizeRedundantCopies::runOnMachineFunction(MachineFunction &MF) {
  bool Changed = false;
  for (auto &MBB : MF) {
    SmallDenseMap<std::pair<Register, unsigned>, std::pair<Register, unsigned>>
        RegMap;
    for (auto &MI : make_early_inc_range(MBB)) {
      processInterveningInsts(MI, RegMap);
      if (!MI.isCopy())
        continue;

      auto Dst = MI.getOperand(0);
      auto Opnd = MI.getOperand(1);
      auto Key = std::make_pair(Opnd.getReg(), Opnd.getSubReg());
      if (auto It = RegMap.find(Key); It != RegMap.end()) {
        auto [SrcReg, SrcSubReg] = It->second;
        MachineIRBuilder B(MI);
        B.buildInstr(TargetOpcode::COPY)
            .addDef(Dst.getReg(), getRegState(Dst), Dst.getSubReg())
            .addUse(SrcReg, {}, SrcSubReg);
        MI.eraseFromParent();
        Changed = true;
        continue;
      }

      auto DKey = std::make_pair(Dst.getReg(), Dst.getSubReg());
      RegMap[DKey] = std::make_pair(Opnd.getReg(), Opnd.getSubReg());
    }
  }
  return Changed;
}

namespace llvm {
FunctionPass *createPISAOptimizeRedundantCopies() {
  return new PISAOptimizeRedundantCopies();
}
} // end namespace llvm
