//===-- RegAllocSimple.cpp - A simple generic register allocator ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple register allocator. *Very* simple: It immediate
// spills every value right after it is computed, and it reloads all used
// operands from the spill area to temporary registers before each instruction.
// It does not keep values in registers across instructions.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <map>
using namespace llvm;

STATISTIC(NumStores, "Number of stores added");
STATISTIC(NumLoads , "Number of loads added");

namespace {
  static RegisterRegAlloc
    simpleRegAlloc("simple", "simple register allocator",
                   createSimpleRegisterAllocator);

  class VISIBILITY_HIDDEN RegAllocSimple : public MachineFunctionPass {
  public:
    static char ID;
    RegAllocSimple() : MachineFunctionPass(&ID) {}
  private:
    MachineFunction *MF;
    const TargetMachine *TM;
    const TargetRegisterInfo *TRI;
    const TargetInstrInfo *TII;

    // StackSlotForVirtReg - Maps SSA Regs => frame index on the stack where
    // these values are spilled
    std::map<unsigned, int> StackSlotForVirtReg;

    // RegsUsed - Keep track of what registers are currently in use.  This is a
    // bitset.
    std::vector<bool> RegsUsed;

    // RegClassIdx - Maps RegClass => which index we can take a register
    // from. Since this is a simple register allocator, when we need a register
    // of a certain class, we just take the next available one.
    std::map<const TargetRegisterClass*, unsigned> RegClassIdx;

  public:
    virtual const char *getPassName() const {
      return "Simple Register Allocator";
    }

    /// runOnMachineFunction - Register allocate the whole function
    bool runOnMachineFunction(MachineFunction &Fn);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(PHIEliminationID);           // Eliminate PHI nodes
      MachineFunctionPass::getAnalysisUsage(AU);
    }
  private:
    /// AllocateBasicBlock - Register allocate the specified basic block.
    void AllocateBasicBlock(MachineBasicBlock &MBB);

    /// getStackSpaceFor - This returns the offset of the specified virtual
    /// register on the stack, allocating space if necessary.
    int getStackSpaceFor(unsigned VirtReg, const TargetRegisterClass *RC);

    /// Given a virtual register, return a compatible physical register that is
    /// currently unused.
    ///
    /// Side effect: marks that register as being used until manually cleared
    ///
    unsigned getFreeReg(unsigned virtualReg);

    /// Moves value from memory into that register
    unsigned reloadVirtReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I, unsigned VirtReg);

    /// Saves reg value on the stack (maps virtual register to stack value)
    void spillVirtReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                      unsigned VirtReg, unsigned PhysReg);
  };
  char RegAllocSimple::ID = 0;
}

/// getStackSpaceFor - This allocates space for the specified virtual
/// register to be held on the stack.
int RegAllocSimple::getStackSpaceFor(unsigned VirtReg,
                                     const TargetRegisterClass *RC) {
  // Find the location VirtReg would belong...
  std::map<unsigned, int>::iterator I = StackSlotForVirtReg.find(VirtReg);

  if (I != StackSlotForVirtReg.end())
    return I->second;          // Already has space allocated?

  // Allocate a new stack object for this spill location...
  int FrameIdx = MF->getFrameInfo()->CreateStackObject(RC->getSize(),
                                                       RC->getAlignment());

  // Assign the slot...
  StackSlotForVirtReg.insert(I, std::make_pair(VirtReg, FrameIdx));

  return FrameIdx;
}

unsigned RegAllocSimple::getFreeReg(unsigned virtualReg) {
  const TargetRegisterClass* RC = MF->getRegInfo().getRegClass(virtualReg);
  TargetRegisterClass::iterator RI = RC->allocation_order_begin(*MF);
#ifndef NDEBUG
  TargetRegisterClass::iterator RE = RC->allocation_order_end(*MF);
#endif

  while (1) {
    unsigned regIdx = RegClassIdx[RC]++;
    assert(RI+regIdx != RE && "Not enough registers!");
    unsigned PhysReg = *(RI+regIdx);

    if (!RegsUsed[PhysReg]) {
      MF->getRegInfo().setPhysRegUsed(PhysReg);
      return PhysReg;
    }
  }
}

unsigned RegAllocSimple::reloadVirtReg(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I,
                                       unsigned VirtReg) {
  const TargetRegisterClass* RC = MF->getRegInfo().getRegClass(VirtReg);
  int FrameIdx = getStackSpaceFor(VirtReg, RC);
  unsigned PhysReg = getFreeReg(VirtReg);

  // Add move instruction(s)
  ++NumLoads;
  TII->loadRegFromStackSlot(MBB, I, PhysReg, FrameIdx, RC);
  return PhysReg;
}

void RegAllocSimple::spillVirtReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I,
                                  unsigned VirtReg, unsigned PhysReg) {
  const TargetRegisterClass* RC = MF->getRegInfo().getRegClass(VirtReg);
  
  int FrameIdx = getStackSpaceFor(VirtReg, RC);

  // Add move instruction(s)
  ++NumStores;
  TII->storeRegToStackSlot(MBB, I, PhysReg, true, FrameIdx, RC);
}


void RegAllocSimple::AllocateBasicBlock(MachineBasicBlock &MBB) {
  // loop over each instruction
  for (MachineBasicBlock::iterator MI = MBB.begin(); MI != MBB.end(); ++MI) {
    // Made to combat the incorrect allocation of r2 = add r1, r1
    std::map<unsigned, unsigned> Virt2PhysRegMap;

    RegsUsed.resize(TRI->getNumRegs());

    // This is a preliminary pass that will invalidate any registers that are
    // used by the instruction (including implicit uses).
    const TargetInstrDesc &Desc = MI->getDesc();
    const unsigned *Regs;
    if (Desc.ImplicitUses) {
      for (Regs = Desc.ImplicitUses; *Regs; ++Regs)
        RegsUsed[*Regs] = true;
    }

    if (Desc.ImplicitDefs) {
      for (Regs = Desc.ImplicitDefs; *Regs; ++Regs) {
        RegsUsed[*Regs] = true;
        MF->getRegInfo().setPhysRegUsed(*Regs);
      }
    }

    // Loop over uses, move from memory into registers.
    for (int i = MI->getNumOperands() - 1; i >= 0; --i) {
      MachineOperand &MO = MI->getOperand(i);

      if (MO.isReg() && MO.getReg() &&
          TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
        unsigned virtualReg = (unsigned) MO.getReg();
        DOUT << "op: " << MO << "\n";
        DOUT << "\t inst[" << i << "]: ";
        DEBUG(MI->print(*cerr.stream(), TM));

        // make sure the same virtual register maps to the same physical
        // register in any given instruction
        unsigned physReg = Virt2PhysRegMap[virtualReg];
        if (physReg == 0) {
          if (MO.isDef()) {
            int TiedOp = Desc.findTiedToSrcOperand(i);
            if (TiedOp == -1) {
              physReg = getFreeReg(virtualReg);
            } else {
              // must be same register number as the source operand that is 
              // tied to. This maps a = b + c into b = b + c, and saves b into
              // a's spot.
              assert(MI->getOperand(TiedOp).isReg()  &&
                     MI->getOperand(TiedOp).getReg() &&
                     MI->getOperand(TiedOp).isUse() &&
                     "Two address instruction invalid!");

              physReg = MI->getOperand(TiedOp).getReg();
            }
            spillVirtReg(MBB, next(MI), virtualReg, physReg);
          } else {
            physReg = reloadVirtReg(MBB, MI, virtualReg);
            Virt2PhysRegMap[virtualReg] = physReg;
          }
        }
        MO.setReg(physReg);
        DOUT << "virt: " << virtualReg << ", phys: " << MO.getReg() << "\n";
      }
    }
    RegClassIdx.clear();
    RegsUsed.clear();
  }
}


/// runOnMachineFunction - Register allocate the whole function
///
bool RegAllocSimple::runOnMachineFunction(MachineFunction &Fn) {
  DOUT << "Machine Function\n";
  MF = &Fn;
  TM = &MF->getTarget();
  TRI = TM->getRegisterInfo();
  TII = TM->getInstrInfo();

  // Loop over all of the basic blocks, eliminating virtual register references
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
    AllocateBasicBlock(*MBB);

  StackSlotForVirtReg.clear();
  return true;
}

FunctionPass *llvm::createSimpleRegisterAllocator() {
  return new RegAllocSimple();
}
