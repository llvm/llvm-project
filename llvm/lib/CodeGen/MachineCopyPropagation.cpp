//===- MachineCopyPropagation.cpp - Machine Copy Propagation Pass ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is an extremely simple MachineInstr-level copy propagation pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

#define DEBUG_TYPE "machine-cp"

STATISTIC(NumDeletes, "Number of dead copies deleted");

namespace {
  typedef SmallVector<unsigned, 4> RegList;
  typedef DenseMap<unsigned, RegList> SourceMap;
  typedef DenseMap<unsigned, MachineInstr*> Reg2MIMap;

  class MachineCopyPropagation : public MachineFunctionPass {
    const TargetRegisterInfo *TRI;
    const TargetInstrInfo *TII;
    const MachineRegisterInfo *MRI;

  public:
    static char ID; // Pass identification, replacement for typeid
    MachineCopyPropagation() : MachineFunctionPass(ID) {
      initializeMachineCopyPropagationPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

    MachineFunctionProperties getRequiredProperties() const override {
      return MachineFunctionProperties().set(
          MachineFunctionProperties::Property::NoVRegs);
    }

  private:
    void ClobberRegister(unsigned Reg);
    void ReadRegister(unsigned Reg);
    void CopyPropagateBlock(MachineBasicBlock &MBB);
    bool eraseIfRedundant(MachineInstr &Copy, unsigned Src, unsigned Def);

    /// Candidates for deletion.
    SmallSetVector<MachineInstr*, 8> MaybeDeadCopies;
    /// Def -> available copies map.
    Reg2MIMap AvailCopyMap;
    /// Def -> copies map.
    Reg2MIMap CopyMap;
    /// Src -> Def map
    SourceMap SrcMap;
    bool Changed;
  };
}
char MachineCopyPropagation::ID = 0;
char &llvm::MachineCopyPropagationID = MachineCopyPropagation::ID;

INITIALIZE_PASS(MachineCopyPropagation, DEBUG_TYPE,
                "Machine Copy Propagation Pass", false, false)

/// Remove any entry in \p Map where the register is a subregister or equal to
/// a register contained in \p Regs.
static void removeRegsFromMap(Reg2MIMap &Map, const RegList &Regs,
                              const TargetRegisterInfo &TRI) {
  for (unsigned Reg : Regs) {
    // Source of copy is no longer available for propagation.
    for (MCSubRegIterator SR(Reg, &TRI, true); SR.isValid(); ++SR)
      Map.erase(*SR);
  }
}

/// Remove any entry in \p Map that is marked clobbered in \p RegMask.
/// The map will typically have a lot fewer entries than the regmask clobbers,
/// so this is more efficient than iterating the clobbered registers and calling
/// ClobberRegister() on them.
static void removeClobberedRegsFromMap(Reg2MIMap &Map,
                                       const MachineOperand &RegMask) {
  for (Reg2MIMap::iterator I = Map.begin(), E = Map.end(), Next; I != E;
       I = Next) {
    Next = std::next(I);
    unsigned Reg = I->first;
    if (RegMask.clobbersPhysReg(Reg))
      Map.erase(I);
  }
}

void MachineCopyPropagation::ClobberRegister(unsigned Reg) {
  for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI) {
    CopyMap.erase(*AI);
    AvailCopyMap.erase(*AI);

    SourceMap::iterator SI = SrcMap.find(*AI);
    if (SI != SrcMap.end()) {
      removeRegsFromMap(AvailCopyMap, SI->second, *TRI);
      SrcMap.erase(SI);
    }
  }
}

void MachineCopyPropagation::ReadRegister(unsigned Reg) {
  // If 'Reg' is defined by a copy, the copy is no longer a candidate
  // for elimination.
  for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI) {
    Reg2MIMap::iterator CI = CopyMap.find(*AI);
    if (CI != CopyMap.end()) {
      DEBUG(dbgs() << "MCP: Copy is used - not dead: "; CI->second->dump());
      MaybeDeadCopies.remove(CI->second);
    }
  }
}

/// Return true if \p PreviousCopy did copy register \p Src to register \p Def.
/// This fact may have been obscured by sub register usage or may not be true at
/// all even though Src and Def are subregisters of the registers used in
/// PreviousCopy. e.g.
/// isNopCopy("ecx = COPY eax", AX, CX) == true
/// isNopCopy("ecx = COPY eax", AH, CL) == false
static bool isNopCopy(const MachineInstr &PreviousCopy, unsigned Src,
                      unsigned Def, const TargetRegisterInfo *TRI) {
  unsigned PreviousSrc = PreviousCopy.getOperand(1).getReg();
  unsigned PreviousDef = PreviousCopy.getOperand(0).getReg();
  if (Src == PreviousSrc) {
    assert(Def == PreviousDef);
    return true;
  }
  if (!TRI->isSubRegister(PreviousSrc, Src))
    return false;
  unsigned SubIdx = TRI->getSubRegIndex(PreviousSrc, Src);
  return SubIdx == TRI->getSubRegIndex(PreviousDef, Def);
}

/// Remove instruction \p Copy if there exists a previous copy that copies the
/// register \p Src to the register \p Def; This may happen indirectly by
/// copying the super registers.
bool MachineCopyPropagation::eraseIfRedundant(MachineInstr &Copy, unsigned Src,
                                              unsigned Def) {
  // Avoid eliminating a copy from/to a reserved registers as we cannot predict
  // the value (Example: The sparc zero register is writable but stays zero).
  if (MRI->isReserved(Src) || MRI->isReserved(Def))
    return false;

  // Search for an existing copy.
  Reg2MIMap::iterator CI = AvailCopyMap.find(Def);
  if (CI == AvailCopyMap.end())
    return false;

  // Check that the existing copy uses the correct sub registers.
  MachineInstr &PrevCopy = *CI->second;
  if (!isNopCopy(PrevCopy, Src, Def, TRI))
    return false;

  DEBUG(dbgs() << "MCP: copy is a NOP, removing: "; Copy.dump());

  // Copy was redundantly redefining either Src or Def. Remove earlier kill
  // flags between Copy and PrevCopy because the value will be reused now.
  assert(Copy.isCopy());
  unsigned CopyDef = Copy.getOperand(0).getReg();
  assert(CopyDef == Src || CopyDef == Def);
  for (MachineInstr &MI :
       make_range(PrevCopy.getIterator(), Copy.getIterator()))
    MI.clearRegisterKills(CopyDef, TRI);

  Copy.eraseFromParent();
  Changed = true;
  ++NumDeletes;
  return true;
}

void MachineCopyPropagation::CopyPropagateBlock(MachineBasicBlock &MBB) {
  DEBUG(dbgs() << "MCP: CopyPropagateBlock " << MBB.getName() << "\n");

  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ) {
    MachineInstr *MI = &*I;
    ++I;

    if (MI->isCopy()) {
      unsigned Def = MI->getOperand(0).getReg();
      unsigned Src = MI->getOperand(1).getReg();

      assert(!TargetRegisterInfo::isVirtualRegister(Def) &&
             !TargetRegisterInfo::isVirtualRegister(Src) &&
             "MachineCopyPropagation should be run after register allocation!");

      // The two copies cancel out and the source of the first copy
      // hasn't been overridden, eliminate the second one. e.g.
      //  %ECX<def> = COPY %EAX
      //  ... nothing clobbered EAX.
      //  %EAX<def> = COPY %ECX
      // =>
      //  %ECX<def> = COPY %EAX
      //
      // or
      //
      //  %ECX<def> = COPY %EAX
      //  ... nothing clobbered EAX.
      //  %ECX<def> = COPY %EAX
      // =>
      //  %ECX<def> = COPY %EAX
      if (eraseIfRedundant(*MI, Def, Src) || eraseIfRedundant(*MI, Src, Def))
        continue;

      // If Src is defined by a previous copy, the previous copy cannot be
      // eliminated.
      ReadRegister(Src);
      for (const MachineOperand &MO : MI->implicit_operands()) {
        if (!MO.isReg() || !MO.readsReg())
          continue;
        unsigned Reg = MO.getReg();
        if (!Reg)
          continue;
        ReadRegister(Reg);
      }

      DEBUG(dbgs() << "MCP: Copy is a deletion candidate: "; MI->dump());

      // Copy is now a candidate for deletion.
      if (!MRI->isReserved(Def))
        MaybeDeadCopies.insert(MI);

      // If 'Def' is previously source of another copy, then this earlier copy's
      // source is no longer available. e.g.
      // %xmm9<def> = copy %xmm2
      // ...
      // %xmm2<def> = copy %xmm0
      // ...
      // %xmm2<def> = copy %xmm9
      ClobberRegister(Def);
      for (const MachineOperand &MO : MI->implicit_operands()) {
        if (!MO.isReg() || !MO.isDef())
          continue;
        unsigned Reg = MO.getReg();
        if (!Reg)
          continue;
        ClobberRegister(Reg);
      }

      // Remember Def is defined by the copy.
      for (MCSubRegIterator SR(Def, TRI, /*IncludeSelf=*/true); SR.isValid();
           ++SR) {
        CopyMap[*SR] = MI;
        AvailCopyMap[*SR] = MI;
      }

      // Remember source that's copied to Def. Once it's clobbered, then
      // it's no longer available for copy propagation.
      RegList &DestList = SrcMap[Src];
      if (!is_contained(DestList, Def))
        DestList.push_back(Def);

      continue;
    }

    // Not a copy.
    SmallVector<unsigned, 2> Defs;
    const MachineOperand *RegMask = nullptr;
    for (const MachineOperand &MO : MI->operands()) {
      if (MO.isRegMask())
        RegMask = &MO;
      if (!MO.isReg())
        continue;
      unsigned Reg = MO.getReg();
      if (!Reg)
        continue;

      assert(!TargetRegisterInfo::isVirtualRegister(Reg) &&
             "MachineCopyPropagation should be run after register allocation!");

      if (MO.isDef()) {
        Defs.push_back(Reg);
        continue;
      } else if (MO.readsReg())
        ReadRegister(Reg);
    }

    // The instruction has a register mask operand which means that it clobbers
    // a large set of registers.  Treat clobbered registers the same way as
    // defined registers.
    if (RegMask) {
      // Erase any MaybeDeadCopies whose destination register is clobbered.
      for (SmallSetVector<MachineInstr *, 8>::iterator DI =
               MaybeDeadCopies.begin();
           DI != MaybeDeadCopies.end();) {
        MachineInstr *MaybeDead = *DI;
        unsigned Reg = MaybeDead->getOperand(0).getReg();
        assert(!MRI->isReserved(Reg));

        if (!RegMask->clobbersPhysReg(Reg)) {
          ++DI;
          continue;
        }

        DEBUG(dbgs() << "MCP: Removing copy due to regmask clobbering: ";
              MaybeDead->dump());

        // erase() will return the next valid iterator pointing to the next
        // element after the erased one.
        DI = MaybeDeadCopies.erase(DI);
        MaybeDead->eraseFromParent();
        Changed = true;
        ++NumDeletes;
      }

      removeClobberedRegsFromMap(AvailCopyMap, *RegMask);
      removeClobberedRegsFromMap(CopyMap, *RegMask);
      for (SourceMap::iterator I = SrcMap.begin(), E = SrcMap.end(), Next;
           I != E; I = Next) {
        Next = std::next(I);
        if (RegMask->clobbersPhysReg(I->first)) {
          removeRegsFromMap(AvailCopyMap, I->second, *TRI);
          SrcMap.erase(I);
        }
      }
    }

    // Any previous copy definition or reading the Defs is no longer available.
    for (unsigned Reg : Defs)
      ClobberRegister(Reg);
  }

  // If MBB doesn't have successors, delete the copies whose defs are not used.
  // If MBB does have successors, then conservative assume the defs are live-out
  // since we don't want to trust live-in lists.
  if (MBB.succ_empty()) {
    for (MachineInstr *MaybeDead : MaybeDeadCopies) {
      assert(!MRI->isReserved(MaybeDead->getOperand(0).getReg()));
      MaybeDead->eraseFromParent();
      Changed = true;
      ++NumDeletes;
    }
  }

  MaybeDeadCopies.clear();
  AvailCopyMap.clear();
  CopyMap.clear();
  SrcMap.clear();
}

bool MachineCopyPropagation::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;

  Changed = false;

  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  MRI = &MF.getRegInfo();

  for (MachineBasicBlock &MBB : MF)
    CopyPropagateBlock(MBB);

  return Changed;
}

