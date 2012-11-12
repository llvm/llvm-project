//===-- RegisterClassInfo.cpp - Dynamic Register Class Info ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RegisterClassInfo class which provides dynamic
// information about target register classes. Callee saved and reserved
// registers depends on calling conventions and other dynamic information, so
// some things cannot be determined statically.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static cl::opt<unsigned>
StressRA("stress-regalloc", cl::Hidden, cl::init(0), cl::value_desc("N"),
         cl::desc("Limit all regclasses to N registers"));

RegisterClassInfo::RegisterClassInfo() : Tag(0), MF(0), TRI(0), CalleeSaved(0)
{}

void RegisterClassInfo::runOnMachineFunction(const MachineFunction &mf) {
  bool Update = false;
  MF = &mf;

  // Allocate new array the first time we see a new target.
  if (MF->getTarget().getRegisterInfo() != TRI) {
    TRI = MF->getTarget().getRegisterInfo();
    RegClass.reset(new RCInfo[TRI->getNumRegClasses()]);
    Update = true;
  }

  // Does this MF have different CSRs?
  const uint16_t *CSR = TRI->getCalleeSavedRegs(MF);
  if (Update || CSR != CalleeSaved) {
    // Build a CSRNum map. Every CSR alias gets an entry pointing to the last
    // overlapping CSR.
    CSRNum.clear();
    CSRNum.resize(TRI->getNumRegs(), 0);
    for (unsigned N = 0; unsigned Reg = CSR[N]; ++N)
      for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
        CSRNum[*AI] = N + 1; // 0 means no CSR, 1 means CalleeSaved[0], ...
    Update = true;
  }
  CalleeSaved = CSR;

  // Different reserved registers?
  const BitVector &RR = MF->getRegInfo().getReservedRegs();
  if (Reserved.size() != RR.size() || RR != Reserved) {
    Update = true;
    Reserved = RR;
  }

  // Invalidate cached information from previous function.
  if (Update)
    ++Tag;
}

/// compute - Compute the preferred allocation order for RC with reserved
/// registers filtered out. Volatile registers come first followed by CSR
/// aliases ordered according to the CSR order specified by the target.
void RegisterClassInfo::compute(const TargetRegisterClass *RC) const {
  RCInfo &RCI = RegClass[RC->getID()];

  // Raw register count, including all reserved regs.
  unsigned NumRegs = RC->getNumRegs();

  if (!RCI.Order)
    RCI.Order.reset(new unsigned[NumRegs]);

  unsigned N = 0;
  SmallVector<unsigned, 16> CSRAlias;

  // FIXME: Once targets reserve registers instead of removing them from the
  // allocation order, we can simply use begin/end here.
  ArrayRef<uint16_t> RawOrder = RC->getRawAllocationOrder(*MF);
  for (unsigned i = 0; i != RawOrder.size(); ++i) {
    unsigned PhysReg = RawOrder[i];
    // Remove reserved registers from the allocation order.
    if (Reserved.test(PhysReg))
      continue;
    if (CSRNum[PhysReg])
      // PhysReg aliases a CSR, save it for later.
      CSRAlias.push_back(PhysReg);
    else
      RCI.Order[N++] = PhysReg;
  }
  RCI.NumRegs = N + CSRAlias.size();
  assert (RCI.NumRegs <= NumRegs && "Allocation order larger than regclass");

  // CSR aliases go after the volatile registers, preserve the target's order.
  std::copy(CSRAlias.begin(), CSRAlias.end(), &RCI.Order[N]);

  // Register allocator stress test.  Clip register class to N registers.
  if (StressRA && RCI.NumRegs > StressRA)
    RCI.NumRegs = StressRA;

  // Check if RC is a proper sub-class.
  if (const TargetRegisterClass *Super = TRI->getLargestLegalSuperClass(RC))
    if (Super != RC && getNumAllocatableRegs(Super) > RCI.NumRegs)
      RCI.ProperSubClass = true;

  DEBUG({
    dbgs() << "AllocationOrder(" << RC->getName() << ") = [";
    for (unsigned I = 0; I != RCI.NumRegs; ++I)
      dbgs() << ' ' << PrintReg(RCI.Order[I], TRI);
    dbgs() << (RCI.ProperSubClass ? " ] (sub-class)\n" : " ]\n");
  });

  // RCI is now up-to-date.
  RCI.Tag = Tag;
}

