//===- HexagonGatherPacketize.cpp -----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass ensures that producer and consumer of VTMP are paired in a bundle.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "gather-packetize"

#include "HexagonTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

cl::opt<bool> EnableGatherPacketize(
    "hexagon-enable-gather-packetize", cl::Hidden, cl::init(true),
    cl::desc("Generate gather packets before packetization"));

namespace llvm {
FunctionPass *createHexagonGatherPacketize();
void initializeHexagonGatherPacketizePass(PassRegistry &);
}

namespace {
class HexagonGatherPacketize : public MachineFunctionPass {
public:
  static char ID;
  HexagonGatherPacketize() : MachineFunctionPass(ID) {
    PassRegistry &Registry = *PassRegistry::getPassRegistry();
    initializeHexagonGatherPacketizePass(Registry);
  }

  StringRef getPassName() const override {
    return "Hexagon Gather Packetize Code";
  }
  bool runOnMachineFunction(MachineFunction &Fn) override;
};

char HexagonGatherPacketize::ID = 0;

static inline bool isVtmpDef(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isDef() && MO.isImplicit() &&
        (MO.getReg() == Hexagon::VTMP)) {
      return true;
    }
  return false;
}

static inline bool isVtmpUse(const MachineInstr &MI) {
  return (MI.mayStore() && (MI.getOperand(2)).isReg() &&
          ((MI.getOperand(2)).getReg() == Hexagon::VTMP));
}

bool HexagonGatherPacketize::runOnMachineFunction(MachineFunction &Fn) {
  if (!EnableGatherPacketize)
    return false;
  auto &ST = Fn.getSubtarget<HexagonSubtarget>();
  bool HasV65 = ST.hasV65TOps();
  bool UseHVX = ST.useHVXOps();
  if (!(HasV65 & UseHVX))
    return false;

  for (auto &MBB : Fn) {
    bool VtmpDef = false;
    MachineBasicBlock::iterator MII, MIE, DefMII;
    for (MII = MBB.begin(), MIE = MBB.end(); MII != MIE; ++MII) {
      MachineInstr &MI = *MII;
      if (VtmpDef) {
        if (!isVtmpUse(MI))
          continue;
        MBB.splice(std::next(DefMII), &MBB, MII);
        finalizeBundle(MBB, DefMII.getInstrIterator(),
                       std::next(MII).getInstrIterator());
        VtmpDef = false;
        continue;
      }
      if (!(isVtmpDef(MI)))
        continue;
      VtmpDef = true;
      DefMII = MII;
    }
    assert(!VtmpDef && "VTMP producer and consumer not in same block");
  }
  return true;
}
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

INITIALIZE_PASS(HexagonGatherPacketize, "hexagon-gather-packetize",
                "Hexagon gather packetize Code", false, false)

FunctionPass *llvm::createHexagonGatherPacketize() {
  return new HexagonGatherPacketize();
}
