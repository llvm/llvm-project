//===-- RegAllocLocal.cpp - A BasicBlock generic register allocator -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This register allocator allocates registers to a basic block at a time,
// attempting to keep values in registers and reusing registers as appropriate.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "llvm/BasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumStores, "Number of stores added");
STATISTIC(NumLoads , "Number of loads added");

static RegisterRegAlloc
  localRegAlloc("local", "local register allocator",
                createLocalRegisterAllocator);

namespace {
  class VISIBILITY_HIDDEN RALocal : public MachineFunctionPass {
  public:
    static char ID;
    RALocal() : MachineFunctionPass(&ID), StackSlotForVirtReg(-1) {}
  private:
    const TargetMachine *TM;
    MachineFunction *MF;
    const TargetRegisterInfo *TRI;
    const TargetInstrInfo *TII;

    // StackSlotForVirtReg - Maps virtual regs to the frame index where these
    // values are spilled.
    IndexedMap<int, VirtReg2IndexFunctor> StackSlotForVirtReg;

    // Virt2PhysRegMap - This map contains entries for each virtual register
    // that is currently available in a physical register.
    IndexedMap<unsigned, VirtReg2IndexFunctor> Virt2PhysRegMap;

    unsigned &getVirt2PhysRegMapSlot(unsigned VirtReg) {
      return Virt2PhysRegMap[VirtReg];
    }

    // PhysRegsUsed - This array is effectively a map, containing entries for
    // each physical register that currently has a value (ie, it is in
    // Virt2PhysRegMap).  The value mapped to is the virtual register
    // corresponding to the physical register (the inverse of the
    // Virt2PhysRegMap), or 0.  The value is set to 0 if this register is pinned
    // because it is used by a future instruction, and to -2 if it is not
    // allocatable.  If the entry for a physical register is -1, then the
    // physical register is "not in the map".
    //
    std::vector<int> PhysRegsUsed;

    // PhysRegsUseOrder - This contains a list of the physical registers that
    // currently have a virtual register value in them.  This list provides an
    // ordering of registers, imposing a reallocation order.  This list is only
    // used if all registers are allocated and we have to spill one, in which
    // case we spill the least recently used register.  Entries at the front of
    // the list are the least recently used registers, entries at the back are
    // the most recently used.
    //
    std::vector<unsigned> PhysRegsUseOrder;

    // Virt2LastUseMap - This maps each virtual register to its last use
    // (MachineInstr*, operand index pair).
    IndexedMap<std::pair<MachineInstr*, unsigned>, VirtReg2IndexFunctor>
    Virt2LastUseMap;

    std::pair<MachineInstr*,unsigned>& getVirtRegLastUse(unsigned Reg) {
      assert(TargetRegisterInfo::isVirtualRegister(Reg) && "Illegal VirtReg!");
      return Virt2LastUseMap[Reg];
    }

    // VirtRegModified - This bitset contains information about which virtual
    // registers need to be spilled back to memory when their registers are
    // scavenged.  If a virtual register has simply been rematerialized, there
    // is no reason to spill it to memory when we need the register back.
    //
    BitVector VirtRegModified;
    
    // UsedInMultipleBlocks - Tracks whether a particular register is used in
    // more than one block.
    BitVector UsedInMultipleBlocks;

    void markVirtRegModified(unsigned Reg, bool Val = true) {
      assert(TargetRegisterInfo::isVirtualRegister(Reg) && "Illegal VirtReg!");
      Reg -= TargetRegisterInfo::FirstVirtualRegister;
      if (Val)
        VirtRegModified.set(Reg);
      else
        VirtRegModified.reset(Reg);
    }

    bool isVirtRegModified(unsigned Reg) const {
      assert(TargetRegisterInfo::isVirtualRegister(Reg) && "Illegal VirtReg!");
      assert(Reg - TargetRegisterInfo::FirstVirtualRegister < VirtRegModified.size()
             && "Illegal virtual register!");
      return VirtRegModified[Reg - TargetRegisterInfo::FirstVirtualRegister];
    }

    void AddToPhysRegsUseOrder(unsigned Reg) {
      std::vector<unsigned>::iterator It =
        std::find(PhysRegsUseOrder.begin(), PhysRegsUseOrder.end(), Reg);
      if (It != PhysRegsUseOrder.end())
        PhysRegsUseOrder.erase(It);
      PhysRegsUseOrder.push_back(Reg);
    }

    void MarkPhysRegRecentlyUsed(unsigned Reg) {
      if (PhysRegsUseOrder.empty() ||
          PhysRegsUseOrder.back() == Reg) return;  // Already most recently used

      for (unsigned i = PhysRegsUseOrder.size(); i != 0; --i)
        if (areRegsEqual(Reg, PhysRegsUseOrder[i-1])) {
          unsigned RegMatch = PhysRegsUseOrder[i-1];       // remove from middle
          PhysRegsUseOrder.erase(PhysRegsUseOrder.begin()+i-1);
          // Add it to the end of the list
          PhysRegsUseOrder.push_back(RegMatch);
          if (RegMatch == Reg)
            return;    // Found an exact match, exit early
        }
    }

  public:
    virtual const char *getPassName() const {
      return "Local Register Allocator";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequiredID(PHIEliminationID);
      AU.addRequiredID(TwoAddressInstructionPassID);
      MachineFunctionPass::getAnalysisUsage(AU);
    }

  private:
    /// runOnMachineFunction - Register allocate the whole function
    bool runOnMachineFunction(MachineFunction &Fn);

    /// AllocateBasicBlock - Register allocate the specified basic block.
    void AllocateBasicBlock(MachineBasicBlock &MBB);


    /// areRegsEqual - This method returns true if the specified registers are
    /// related to each other.  To do this, it checks to see if they are equal
    /// or if the first register is in the alias set of the second register.
    ///
    bool areRegsEqual(unsigned R1, unsigned R2) const {
      if (R1 == R2) return true;
      for (const unsigned *AliasSet = TRI->getAliasSet(R2);
           *AliasSet; ++AliasSet) {
        if (*AliasSet == R1) return true;
      }
      return false;
    }

    /// getStackSpaceFor - This returns the frame index of the specified virtual
    /// register on the stack, allocating space if necessary.
    int getStackSpaceFor(unsigned VirtReg, const TargetRegisterClass *RC);

    /// removePhysReg - This method marks the specified physical register as no
    /// longer being in use.
    ///
    void removePhysReg(unsigned PhysReg);

    /// spillVirtReg - This method spills the value specified by PhysReg into
    /// the virtual register slot specified by VirtReg.  It then updates the RA
    /// data structures to indicate the fact that PhysReg is now available.
    ///
    void spillVirtReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                      unsigned VirtReg, unsigned PhysReg);

    /// spillPhysReg - This method spills the specified physical register into
    /// the virtual register slot associated with it.  If OnlyVirtRegs is set to
    /// true, then the request is ignored if the physical register does not
    /// contain a virtual register.
    ///
    void spillPhysReg(MachineBasicBlock &MBB, MachineInstr *I,
                      unsigned PhysReg, bool OnlyVirtRegs = false);

    /// assignVirtToPhysReg - This method updates local state so that we know
    /// that PhysReg is the proper container for VirtReg now.  The physical
    /// register must not be used for anything else when this is called.
    ///
    void assignVirtToPhysReg(unsigned VirtReg, unsigned PhysReg);

    /// isPhysRegAvailable - Return true if the specified physical register is
    /// free and available for use.  This also includes checking to see if
    /// aliased registers are all free...
    ///
    bool isPhysRegAvailable(unsigned PhysReg) const;

    /// getFreeReg - Look to see if there is a free register available in the
    /// specified register class.  If not, return 0.
    ///
    unsigned getFreeReg(const TargetRegisterClass *RC);

    /// getReg - Find a physical register to hold the specified virtual
    /// register.  If all compatible physical registers are used, this method
    /// spills the last used virtual register to the stack, and uses that
    /// register. If NoFree is true, that means the caller knows there isn't
    /// a free register, do not call getFreeReg().
    unsigned getReg(MachineBasicBlock &MBB, MachineInstr *MI,
                    unsigned VirtReg, bool NoFree = false);

    /// reloadVirtReg - This method transforms the specified specified virtual
    /// register use to refer to a physical register.  This method may do this
    /// in one of several ways: if the register is available in a physical
    /// register already, it uses that physical register.  If the value is not
    /// in a physical register, and if there are physical registers available,
    /// it loads it into a register.  If register pressure is high, and it is
    /// possible, it tries to fold the load of the virtual register into the
    /// instruction itself.  It avoids doing this if register pressure is low to
    /// improve the chance that subsequent instructions can use the reloaded
    /// value.  This method returns the modified instruction.
    ///
    MachineInstr *reloadVirtReg(MachineBasicBlock &MBB, MachineInstr *MI,
                                unsigned OpNum, SmallSet<unsigned, 4> &RRegs);

    /// ComputeLocalLiveness - Computes liveness of registers within a basic
    /// block, setting the killed/dead flags as appropriate.
    void ComputeLocalLiveness(MachineBasicBlock& MBB);

    void reloadPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator &I,
                       unsigned PhysReg);
  };
  char RALocal::ID = 0;
}

/// getStackSpaceFor - This allocates space for the specified virtual register
/// to be held on the stack.
int RALocal::getStackSpaceFor(unsigned VirtReg, const TargetRegisterClass *RC) {
  // Find the location Reg would belong...
  int SS = StackSlotForVirtReg[VirtReg];
  if (SS != -1)
    return SS;          // Already has space allocated?

  // Allocate a new stack object for this spill location...
  int FrameIdx = MF->getFrameInfo()->CreateStackObject(RC->getSize(),
                                                       RC->getAlignment());

  // Assign the slot...
  StackSlotForVirtReg[VirtReg] = FrameIdx;
  return FrameIdx;
}


/// removePhysReg - This method marks the specified physical register as no
/// longer being in use.
///
void RALocal::removePhysReg(unsigned PhysReg) {
  PhysRegsUsed[PhysReg] = -1;      // PhyReg no longer used

  std::vector<unsigned>::iterator It =
    std::find(PhysRegsUseOrder.begin(), PhysRegsUseOrder.end(), PhysReg);
  if (It != PhysRegsUseOrder.end())
    PhysRegsUseOrder.erase(It);
}


/// spillVirtReg - This method spills the value specified by PhysReg into the
/// virtual register slot specified by VirtReg.  It then updates the RA data
/// structures to indicate the fact that PhysReg is now available.
///
void RALocal::spillVirtReg(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator I,
                           unsigned VirtReg, unsigned PhysReg) {
  assert(VirtReg && "Spilling a physical register is illegal!"
         " Must not have appropriate kill for the register or use exists beyond"
         " the intended one.");
  DOUT << "  Spilling register " << TRI->getName(PhysReg)
       << " containing %reg" << VirtReg;
  
  if (!isVirtRegModified(VirtReg)) {
    DOUT << " which has not been modified, so no store necessary!";
    std::pair<MachineInstr*, unsigned> &LastUse = getVirtRegLastUse(VirtReg);
    if (LastUse.first)
      LastUse.first->getOperand(LastUse.second).setIsKill();
  } else {
    // Otherwise, there is a virtual register corresponding to this physical
    // register.  We only need to spill it into its stack slot if it has been
    // modified.
    const TargetRegisterClass *RC = MF->getRegInfo().getRegClass(VirtReg);
    int FrameIndex = getStackSpaceFor(VirtReg, RC);
    DOUT << " to stack slot #" << FrameIndex;
    // If the instruction reads the register that's spilled, (e.g. this can
    // happen if it is a move to a physical register), then the spill
    // instruction is not a kill.
    bool isKill = !(I != MBB.end() && I->readsRegister(PhysReg));
    TII->storeRegToStackSlot(MBB, I, PhysReg, isKill, FrameIndex, RC);
    ++NumStores;   // Update statistics
  }

  getVirt2PhysRegMapSlot(VirtReg) = 0;   // VirtReg no longer available

  DOUT << "\n";
  removePhysReg(PhysReg);
}


/// spillPhysReg - This method spills the specified physical register into the
/// virtual register slot associated with it.  If OnlyVirtRegs is set to true,
/// then the request is ignored if the physical register does not contain a
/// virtual register.
///
void RALocal::spillPhysReg(MachineBasicBlock &MBB, MachineInstr *I,
                           unsigned PhysReg, bool OnlyVirtRegs) {
  if (PhysRegsUsed[PhysReg] != -1) {            // Only spill it if it's used!
    assert(PhysRegsUsed[PhysReg] != -2 && "Non allocable reg used!");
    if (PhysRegsUsed[PhysReg] || !OnlyVirtRegs)
      spillVirtReg(MBB, I, PhysRegsUsed[PhysReg], PhysReg);
  } else {
    // If the selected register aliases any other registers, we must make
    // sure that one of the aliases isn't alive.
    for (const unsigned *AliasSet = TRI->getAliasSet(PhysReg);
         *AliasSet; ++AliasSet)
      if (PhysRegsUsed[*AliasSet] != -1 &&     // Spill aliased register.
          PhysRegsUsed[*AliasSet] != -2)       // If allocatable.
          if (PhysRegsUsed[*AliasSet])
            spillVirtReg(MBB, I, PhysRegsUsed[*AliasSet], *AliasSet);
  }
}


/// assignVirtToPhysReg - This method updates local state so that we know
/// that PhysReg is the proper container for VirtReg now.  The physical
/// register must not be used for anything else when this is called.
///
void RALocal::assignVirtToPhysReg(unsigned VirtReg, unsigned PhysReg) {
  assert(PhysRegsUsed[PhysReg] == -1 && "Phys reg already assigned!");
  // Update information to note the fact that this register was just used, and
  // it holds VirtReg.
  PhysRegsUsed[PhysReg] = VirtReg;
  getVirt2PhysRegMapSlot(VirtReg) = PhysReg;
  AddToPhysRegsUseOrder(PhysReg);   // New use of PhysReg
}


/// isPhysRegAvailable - Return true if the specified physical register is free
/// and available for use.  This also includes checking to see if aliased
/// registers are all free...
///
bool RALocal::isPhysRegAvailable(unsigned PhysReg) const {
  if (PhysRegsUsed[PhysReg] != -1) return false;

  // If the selected register aliases any other allocated registers, it is
  // not free!
  for (const unsigned *AliasSet = TRI->getAliasSet(PhysReg);
       *AliasSet; ++AliasSet)
    if (PhysRegsUsed[*AliasSet] >= 0) // Aliased register in use?
      return false;                    // Can't use this reg then.
  return true;
}


/// getFreeReg - Look to see if there is a free register available in the
/// specified register class.  If not, return 0.
///
unsigned RALocal::getFreeReg(const TargetRegisterClass *RC) {
  // Get iterators defining the range of registers that are valid to allocate in
  // this class, which also specifies the preferred allocation order.
  TargetRegisterClass::iterator RI = RC->allocation_order_begin(*MF);
  TargetRegisterClass::iterator RE = RC->allocation_order_end(*MF);

  for (; RI != RE; ++RI)
    if (isPhysRegAvailable(*RI)) {       // Is reg unused?
      assert(*RI != 0 && "Cannot use register!");
      return *RI; // Found an unused register!
    }
  return 0;
}


/// getReg - Find a physical register to hold the specified virtual
/// register.  If all compatible physical registers are used, this method spills
/// the last used virtual register to the stack, and uses that register.
///
unsigned RALocal::getReg(MachineBasicBlock &MBB, MachineInstr *I,
                         unsigned VirtReg, bool NoFree) {
  const TargetRegisterClass *RC = MF->getRegInfo().getRegClass(VirtReg);

  // First check to see if we have a free register of the requested type...
  unsigned PhysReg = NoFree ? 0 : getFreeReg(RC);

  // If we didn't find an unused register, scavenge one now!
  if (PhysReg == 0) {
    assert(!PhysRegsUseOrder.empty() && "No allocated registers??");

    // Loop over all of the preallocated registers from the least recently used
    // to the most recently used.  When we find one that is capable of holding
    // our register, use it.
    for (unsigned i = 0; PhysReg == 0; ++i) {
      assert(i != PhysRegsUseOrder.size() &&
             "Couldn't find a register of the appropriate class!");

      unsigned R = PhysRegsUseOrder[i];

      // We can only use this register if it holds a virtual register (ie, it
      // can be spilled).  Do not use it if it is an explicitly allocated
      // physical register!
      assert(PhysRegsUsed[R] != -1 &&
             "PhysReg in PhysRegsUseOrder, but is not allocated?");
      if (PhysRegsUsed[R] && PhysRegsUsed[R] != -2) {
        // If the current register is compatible, use it.
        if (RC->contains(R)) {
          PhysReg = R;
          break;
        } else {
          // If one of the registers aliased to the current register is
          // compatible, use it.
          for (const unsigned *AliasIt = TRI->getAliasSet(R);
               *AliasIt; ++AliasIt) {
            if (RC->contains(*AliasIt) &&
                // If this is pinned down for some reason, don't use it.  For
                // example, if CL is pinned, and we run across CH, don't use
                // CH as justification for using scavenging ECX (which will
                // fail).
                PhysRegsUsed[*AliasIt] != 0 &&
                
                // Make sure the register is allocatable.  Don't allocate SIL on
                // x86-32.
                PhysRegsUsed[*AliasIt] != -2) {
              PhysReg = *AliasIt;    // Take an aliased register
              break;
            }
          }
        }
      }
    }

    assert(PhysReg && "Physical register not assigned!?!?");

    // At this point PhysRegsUseOrder[i] is the least recently used register of
    // compatible register class.  Spill it to memory and reap its remains.
    spillPhysReg(MBB, I, PhysReg);
  }

  // Now that we know which register we need to assign this to, do it now!
  assignVirtToPhysReg(VirtReg, PhysReg);
  return PhysReg;
}


/// reloadVirtReg - This method transforms the specified specified virtual
/// register use to refer to a physical register.  This method may do this in
/// one of several ways: if the register is available in a physical register
/// already, it uses that physical register.  If the value is not in a physical
/// register, and if there are physical registers available, it loads it into a
/// register.  If register pressure is high, and it is possible, it tries to
/// fold the load of the virtual register into the instruction itself.  It
/// avoids doing this if register pressure is low to improve the chance that
/// subsequent instructions can use the reloaded value.  This method returns the
/// modified instruction.
///
MachineInstr *RALocal::reloadVirtReg(MachineBasicBlock &MBB, MachineInstr *MI,
                                     unsigned OpNum,
                                     SmallSet<unsigned, 4> &ReloadedRegs) {
  unsigned VirtReg = MI->getOperand(OpNum).getReg();

  // If the virtual register is already available, just update the instruction
  // and return.
  if (unsigned PR = getVirt2PhysRegMapSlot(VirtReg)) {
    MarkPhysRegRecentlyUsed(PR);       // Already have this value available!
    MI->getOperand(OpNum).setReg(PR);  // Assign the input register
    getVirtRegLastUse(VirtReg) = std::make_pair(MI, OpNum);
    return MI;
  }

  // Otherwise, we need to fold it into the current instruction, or reload it.
  // If we have registers available to hold the value, use them.
  const TargetRegisterClass *RC = MF->getRegInfo().getRegClass(VirtReg);
  unsigned PhysReg = getFreeReg(RC);
  int FrameIndex = getStackSpaceFor(VirtReg, RC);

  if (PhysReg) {   // Register is available, allocate it!
    assignVirtToPhysReg(VirtReg, PhysReg);
  } else {         // No registers available.
    // Force some poor hapless value out of the register file to
    // make room for the new register, and reload it.
    PhysReg = getReg(MBB, MI, VirtReg, true);
  }

  markVirtRegModified(VirtReg, false);   // Note that this reg was just reloaded

  DOUT << "  Reloading %reg" << VirtReg << " into "
       << TRI->getName(PhysReg) << "\n";

  // Add move instruction(s)
  TII->loadRegFromStackSlot(MBB, MI, PhysReg, FrameIndex, RC);
  ++NumLoads;    // Update statistics

  MF->getRegInfo().setPhysRegUsed(PhysReg);
  MI->getOperand(OpNum).setReg(PhysReg);  // Assign the input register
  getVirtRegLastUse(VirtReg) = std::make_pair(MI, OpNum);

  if (!ReloadedRegs.insert(PhysReg)) {
    cerr << "Ran out of registers during register allocation!\n";
    if (MI->getOpcode() == TargetInstrInfo::INLINEASM) {
      cerr << "Please check your inline asm statement for invalid "
           << "constraints:\n";
      MI->print(cerr.stream(), TM);
    }
    exit(1);
  }
  for (const unsigned *SubRegs = TRI->getSubRegisters(PhysReg);
       *SubRegs; ++SubRegs) {
    if (!ReloadedRegs.insert(*SubRegs)) {
      cerr << "Ran out of registers during register allocation!\n";
      if (MI->getOpcode() == TargetInstrInfo::INLINEASM) {
        cerr << "Please check your inline asm statement for invalid "
             << "constraints:\n";
        MI->print(cerr.stream(), TM);
      }
      exit(1);
    }
  }

  return MI;
}

/// isReadModWriteImplicitKill - True if this is an implicit kill for a
/// read/mod/write register, i.e. update partial register.
static bool isReadModWriteImplicitKill(MachineInstr *MI, unsigned Reg) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand& MO = MI->getOperand(i);
    if (MO.isReg() && MO.getReg() == Reg && MO.isImplicit() &&
        MO.isDef() && !MO.isDead())
      return true;
  }
  return false;
}

/// isReadModWriteImplicitDef - True if this is an implicit def for a
/// read/mod/write register, i.e. update partial register.
static bool isReadModWriteImplicitDef(MachineInstr *MI, unsigned Reg) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand& MO = MI->getOperand(i);
    if (MO.isReg() && MO.getReg() == Reg && MO.isImplicit() &&
        !MO.isDef() && MO.isKill())
      return true;
  }
  return false;
}

// precedes - Helper function to determine with MachineInstr A
// precedes MachineInstr B within the same MBB.
static bool precedes(MachineBasicBlock::iterator A,
                     MachineBasicBlock::iterator B) {
  if (A == B)
    return false;
  
  MachineBasicBlock::iterator I = A->getParent()->begin();
  while (I != A->getParent()->end()) {
    if (I == A)
      return true;
    else if (I == B)
      return false;
    
    ++I;
  }
  
  return false;
}

/// ComputeLocalLiveness - Computes liveness of registers within a basic
/// block, setting the killed/dead flags as appropriate.
void RALocal::ComputeLocalLiveness(MachineBasicBlock& MBB) {
  MachineRegisterInfo& MRI = MBB.getParent()->getRegInfo();
  // Keep track of the most recently seen previous use or def of each reg, 
  // so that we can update them with dead/kill markers.
  DenseMap<unsigned, std::pair<MachineInstr*, unsigned> > LastUseDef;
  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
       I != E; ++I) {
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      MachineOperand& MO = I->getOperand(i);
      // Uses don't trigger any flags, but we need to save
      // them for later.  Also, we have to process these
      // _before_ processing the defs, since an instr
      // uses regs before it defs them.
      if (MO.isReg() && MO.getReg() && MO.isUse()) {
        LastUseDef[MO.getReg()] = std::make_pair(I, i);
        
        
        if (TargetRegisterInfo::isVirtualRegister(MO.getReg())) continue;
        
        const unsigned* Aliases = TRI->getAliasSet(MO.getReg());
        if (Aliases) {
          while (*Aliases) {
            DenseMap<unsigned, std::pair<MachineInstr*, unsigned> >::iterator
              alias = LastUseDef.find(*Aliases);
            
            if (alias != LastUseDef.end() && alias->second.first != I)
              LastUseDef[*Aliases] = std::make_pair(I, i);
            
            ++Aliases;
          }
        }
      }
    }
    
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      MachineOperand& MO = I->getOperand(i);
      // Defs others than 2-addr redefs _do_ trigger flag changes:
      //   - A def followed by a def is dead
      //   - A use followed by a def is a kill
      if (MO.isReg() && MO.getReg() && MO.isDef()) {
        DenseMap<unsigned, std::pair<MachineInstr*, unsigned> >::iterator
          last = LastUseDef.find(MO.getReg());
        if (last != LastUseDef.end()) {
          // Check if this is a two address instruction.  If so, then
          // the def does not kill the use.
          if (last->second.first == I &&
              I->isRegReDefinedByTwoAddr(i))
            continue;
          
          MachineOperand& lastUD =
                      last->second.first->getOperand(last->second.second);
          if (lastUD.isDef())
            lastUD.setIsDead(true);
          else
            lastUD.setIsKill(true);
        }
        
        LastUseDef[MO.getReg()] = std::make_pair(I, i);
      }
    }
  }
  
  // Live-out (of the function) registers contain return values of the function,
  // so we need to make sure they are alive at return time.
  if (!MBB.empty() && MBB.back().getDesc().isReturn()) {
    MachineInstr* Ret = &MBB.back();
    for (MachineRegisterInfo::liveout_iterator
         I = MF->getRegInfo().liveout_begin(),
         E = MF->getRegInfo().liveout_end(); I != E; ++I)
      if (!Ret->readsRegister(*I)) {
        Ret->addOperand(MachineOperand::CreateReg(*I, false, true));
        LastUseDef[*I] = std::make_pair(Ret, Ret->getNumOperands()-1);
      }
  }
  
  // Finally, loop over the final use/def of each reg 
  // in the block and determine if it is dead.
  for (DenseMap<unsigned, std::pair<MachineInstr*, unsigned> >::iterator
       I = LastUseDef.begin(), E = LastUseDef.end(); I != E; ++I) {
    MachineInstr* MI = I->second.first;
    unsigned idx = I->second.second;
    MachineOperand& MO = MI->getOperand(idx);
    
    bool isPhysReg = TargetRegisterInfo::isPhysicalRegister(MO.getReg());
    
    // A crude approximation of "live-out" calculation
    bool usedOutsideBlock = isPhysReg ? false :   
          UsedInMultipleBlocks.test(MO.getReg() -  
                                    TargetRegisterInfo::FirstVirtualRegister);
    if (!isPhysReg && !usedOutsideBlock)
      for (MachineRegisterInfo::reg_iterator UI = MRI.reg_begin(MO.getReg()),
           UE = MRI.reg_end(); UI != UE; ++UI)
        // Two cases:
        // - used in another block
        // - used in the same block before it is defined (loop)
        if (UI->getParent() != &MBB ||
            (MO.isDef() && UI.getOperand().isUse() && precedes(&*UI, MI))) {
          UsedInMultipleBlocks.set(MO.getReg() - 
                                   TargetRegisterInfo::FirstVirtualRegister);
          usedOutsideBlock = true;
          break;
        }
    
    // Physical registers and those that are not live-out of the block
    // are killed/dead at their last use/def within this block.
    if (isPhysReg || !usedOutsideBlock) {
      if (MO.isUse()) {
        // Don't mark uses that are tied to defs as kills.
        if (MI->getDesc().getOperandConstraint(idx, TOI::TIED_TO) == -1)
          MO.setIsKill(true);
      } else
        MO.setIsDead(true);
    }
  }
}

void RALocal::AllocateBasicBlock(MachineBasicBlock &MBB) {
  // loop over each instruction
  MachineBasicBlock::iterator MII = MBB.begin();
  
  DEBUG(const BasicBlock *LBB = MBB.getBasicBlock();
        if (LBB) DOUT << "\nStarting RegAlloc of BB: " << LBB->getName());

  // Add live-in registers as active.
  for (MachineBasicBlock::livein_iterator I = MBB.livein_begin(),
         E = MBB.livein_end(); I != E; ++I) {
    unsigned Reg = *I;
    MF->getRegInfo().setPhysRegUsed(Reg);
    PhysRegsUsed[Reg] = 0;            // It is free and reserved now
    AddToPhysRegsUseOrder(Reg); 
    for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
         *SubRegs; ++SubRegs) {
      if (PhysRegsUsed[*SubRegs] != -2) {
        AddToPhysRegsUseOrder(*SubRegs); 
        PhysRegsUsed[*SubRegs] = 0;  // It is free and reserved now
        MF->getRegInfo().setPhysRegUsed(*SubRegs);
      }
    }
  }
  
  ComputeLocalLiveness(MBB);
  
  // Otherwise, sequentially allocate each instruction in the MBB.
  while (MII != MBB.end()) {
    MachineInstr *MI = MII++;
    const TargetInstrDesc &TID = MI->getDesc();
    DEBUG(DOUT << "\nStarting RegAlloc of: " << *MI;
          DOUT << "  Regs have values: ";
          for (unsigned i = 0; i != TRI->getNumRegs(); ++i)
            if (PhysRegsUsed[i] != -1 && PhysRegsUsed[i] != -2)
               DOUT << "[" << TRI->getName(i)
                    << ",%reg" << PhysRegsUsed[i] << "] ";
          DOUT << "\n");

    // Loop over the implicit uses, making sure that they are at the head of the
    // use order list, so they don't get reallocated.
    if (TID.ImplicitUses) {
      for (const unsigned *ImplicitUses = TID.ImplicitUses;
           *ImplicitUses; ++ImplicitUses)
        MarkPhysRegRecentlyUsed(*ImplicitUses);
    }

    SmallVector<unsigned, 8> Kills;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand& MO = MI->getOperand(i);
      if (MO.isReg() && MO.isKill()) {
        if (!MO.isImplicit())
          Kills.push_back(MO.getReg());
        else if (!isReadModWriteImplicitKill(MI, MO.getReg()))
          // These are extra physical register kills when a sub-register
          // is defined (def of a sub-register is a read/mod/write of the
          // larger registers). Ignore.
          Kills.push_back(MO.getReg());
      }
    }

    // If any physical regs are earlyclobber, spill any value they might
    // have in them, then mark them unallocatable.
    // If any virtual regs are earlyclobber, allocate them now (before
    // freeing inputs that are killed).
    if (MI->getOpcode()==TargetInstrInfo::INLINEASM) {
      for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
        MachineOperand& MO = MI->getOperand(i);
        if (MO.isReg() && MO.isDef() && MO.isEarlyClobber() &&
            MO.getReg()) {
          if (TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
            unsigned DestVirtReg = MO.getReg();
            unsigned DestPhysReg;

            // If DestVirtReg already has a value, use it.
            if (!(DestPhysReg = getVirt2PhysRegMapSlot(DestVirtReg)))
              DestPhysReg = getReg(MBB, MI, DestVirtReg);
            MF->getRegInfo().setPhysRegUsed(DestPhysReg);
            markVirtRegModified(DestVirtReg);
            getVirtRegLastUse(DestVirtReg) =
                   std::make_pair((MachineInstr*)0, 0);
            DOUT << "  Assigning " << TRI->getName(DestPhysReg)
                 << " to %reg" << DestVirtReg << "\n";
            MO.setReg(DestPhysReg);  // Assign the earlyclobber register
          } else {
            unsigned Reg = MO.getReg();
            if (PhysRegsUsed[Reg] == -2) continue;  // Something like ESP.
            // These are extra physical register defs when a sub-register
            // is defined (def of a sub-register is a read/mod/write of the
            // larger registers). Ignore.
            if (isReadModWriteImplicitDef(MI, MO.getReg())) continue;

            MF->getRegInfo().setPhysRegUsed(Reg);
            spillPhysReg(MBB, MI, Reg, true); // Spill any existing value in reg
            PhysRegsUsed[Reg] = 0;            // It is free and reserved now
            AddToPhysRegsUseOrder(Reg); 

            for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
                 *SubRegs; ++SubRegs) {
              if (PhysRegsUsed[*SubRegs] != -2) {
                MF->getRegInfo().setPhysRegUsed(*SubRegs);
                PhysRegsUsed[*SubRegs] = 0;  // It is free and reserved now
                AddToPhysRegsUseOrder(*SubRegs); 
              }
            }
          }
        }
      }
    }

    // Get the used operands into registers.  This has the potential to spill
    // incoming values if we are out of registers.  Note that we completely
    // ignore physical register uses here.  We assume that if an explicit
    // physical register is referenced by the instruction, that it is guaranteed
    // to be live-in, or the input is badly hosed.
    //
    SmallSet<unsigned, 4> ReloadedRegs;
    for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
      MachineOperand& MO = MI->getOperand(i);
      // here we are looking for only used operands (never def&use)
      if (MO.isReg() && !MO.isDef() && MO.getReg() && !MO.isImplicit() &&
          TargetRegisterInfo::isVirtualRegister(MO.getReg()))
        MI = reloadVirtReg(MBB, MI, i, ReloadedRegs);
    }

    // If this instruction is the last user of this register, kill the
    // value, freeing the register being used, so it doesn't need to be
    // spilled to memory.
    //
    for (unsigned i = 0, e = Kills.size(); i != e; ++i) {
      unsigned VirtReg = Kills[i];
      unsigned PhysReg = VirtReg;
      if (TargetRegisterInfo::isVirtualRegister(VirtReg)) {
        // If the virtual register was never materialized into a register, it
        // might not be in the map, but it won't hurt to zero it out anyway.
        unsigned &PhysRegSlot = getVirt2PhysRegMapSlot(VirtReg);
        PhysReg = PhysRegSlot;
        PhysRegSlot = 0;
      } else if (PhysRegsUsed[PhysReg] == -2) {
        // Unallocatable register dead, ignore.
        continue;
      } else {
        assert((!PhysRegsUsed[PhysReg] || PhysRegsUsed[PhysReg] == -1) &&
               "Silently clearing a virtual register?");
      }

      if (PhysReg) {
        DOUT << "  Last use of " << TRI->getName(PhysReg)
             << "[%reg" << VirtReg <<"], removing it from live set\n";
        removePhysReg(PhysReg);
        for (const unsigned *SubRegs = TRI->getSubRegisters(PhysReg);
             *SubRegs; ++SubRegs) {
          if (PhysRegsUsed[*SubRegs] != -2) {
            DOUT  << "  Last use of "
                  << TRI->getName(*SubRegs)
                  << "[%reg" << VirtReg <<"], removing it from live set\n";
            removePhysReg(*SubRegs);
          }
        }
      }
    }

    // Loop over all of the operands of the instruction, spilling registers that
    // are defined, and marking explicit destinations in the PhysRegsUsed map.
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand& MO = MI->getOperand(i);
      if (MO.isReg() && MO.isDef() && !MO.isImplicit() && MO.getReg() &&
          !MO.isEarlyClobber() &&
          TargetRegisterInfo::isPhysicalRegister(MO.getReg())) {
        unsigned Reg = MO.getReg();
        if (PhysRegsUsed[Reg] == -2) continue;  // Something like ESP.
        // These are extra physical register defs when a sub-register
        // is defined (def of a sub-register is a read/mod/write of the
        // larger registers). Ignore.
        if (isReadModWriteImplicitDef(MI, MO.getReg())) continue;

        MF->getRegInfo().setPhysRegUsed(Reg);
        spillPhysReg(MBB, MI, Reg, true); // Spill any existing value in reg
        PhysRegsUsed[Reg] = 0;            // It is free and reserved now
        AddToPhysRegsUseOrder(Reg); 

        for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
             *SubRegs; ++SubRegs) {
          if (PhysRegsUsed[*SubRegs] != -2) {
            MF->getRegInfo().setPhysRegUsed(*SubRegs);
            PhysRegsUsed[*SubRegs] = 0;  // It is free and reserved now
            AddToPhysRegsUseOrder(*SubRegs); 
          }
        }
      }
    }

    // Loop over the implicit defs, spilling them as well.
    if (TID.ImplicitDefs) {
      for (const unsigned *ImplicitDefs = TID.ImplicitDefs;
           *ImplicitDefs; ++ImplicitDefs) {
        unsigned Reg = *ImplicitDefs;
        if (PhysRegsUsed[Reg] != -2) {
          spillPhysReg(MBB, MI, Reg, true);
          AddToPhysRegsUseOrder(Reg); 
          PhysRegsUsed[Reg] = 0;            // It is free and reserved now
        }
        MF->getRegInfo().setPhysRegUsed(Reg);
        for (const unsigned *SubRegs = TRI->getSubRegisters(Reg);
             *SubRegs; ++SubRegs) {
          if (PhysRegsUsed[*SubRegs] != -2) {
            AddToPhysRegsUseOrder(*SubRegs); 
            PhysRegsUsed[*SubRegs] = 0;  // It is free and reserved now
            MF->getRegInfo().setPhysRegUsed(*SubRegs);
          }
        }
      }
    }

    SmallVector<unsigned, 8> DeadDefs;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand& MO = MI->getOperand(i);
      if (MO.isReg() && MO.isDead())
        DeadDefs.push_back(MO.getReg());
    }

    // Okay, we have allocated all of the source operands and spilled any values
    // that would be destroyed by defs of this instruction.  Loop over the
    // explicit defs and assign them to a register, spilling incoming values if
    // we need to scavenge a register.
    //
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      MachineOperand& MO = MI->getOperand(i);
      if (MO.isReg() && MO.isDef() && MO.getReg() &&
          !MO.isEarlyClobber() &&
          TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
        unsigned DestVirtReg = MO.getReg();
        unsigned DestPhysReg;

        // If DestVirtReg already has a value, use it.
        if (!(DestPhysReg = getVirt2PhysRegMapSlot(DestVirtReg)))
          DestPhysReg = getReg(MBB, MI, DestVirtReg);
        MF->getRegInfo().setPhysRegUsed(DestPhysReg);
        markVirtRegModified(DestVirtReg);
        getVirtRegLastUse(DestVirtReg) = std::make_pair((MachineInstr*)0, 0);
        DOUT << "  Assigning " << TRI->getName(DestPhysReg)
             << " to %reg" << DestVirtReg << "\n";
        MO.setReg(DestPhysReg);  // Assign the output register
      }
    }

    // If this instruction defines any registers that are immediately dead,
    // kill them now.
    //
    for (unsigned i = 0, e = DeadDefs.size(); i != e; ++i) {
      unsigned VirtReg = DeadDefs[i];
      unsigned PhysReg = VirtReg;
      if (TargetRegisterInfo::isVirtualRegister(VirtReg)) {
        unsigned &PhysRegSlot = getVirt2PhysRegMapSlot(VirtReg);
        PhysReg = PhysRegSlot;
        assert(PhysReg != 0);
        PhysRegSlot = 0;
      } else if (PhysRegsUsed[PhysReg] == -2) {
        // Unallocatable register dead, ignore.
        continue;
      }

      if (PhysReg) {
        DOUT  << "  Register " << TRI->getName(PhysReg)
              << " [%reg" << VirtReg
              << "] is never used, removing it from live set\n";
        removePhysReg(PhysReg);
        for (const unsigned *AliasSet = TRI->getAliasSet(PhysReg);
             *AliasSet; ++AliasSet) {
          if (PhysRegsUsed[*AliasSet] != -2) {
            DOUT  << "  Register " << TRI->getName(*AliasSet)
                  << " [%reg" << *AliasSet
                  << "] is never used, removing it from live set\n";
            removePhysReg(*AliasSet);
          }
        }
      }
    }
    
    // Finally, if this is a noop copy instruction, zap it.
    unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
    if (TII->isMoveInstr(*MI, SrcReg, DstReg, SrcSubReg, DstSubReg) &&
        SrcReg == DstReg)
      MBB.erase(MI);
  }

  MachineBasicBlock::iterator MI = MBB.getFirstTerminator();

  // Spill all physical registers holding virtual registers now.
  for (unsigned i = 0, e = TRI->getNumRegs(); i != e; ++i)
    if (PhysRegsUsed[i] != -1 && PhysRegsUsed[i] != -2) {
      if (unsigned VirtReg = PhysRegsUsed[i])
        spillVirtReg(MBB, MI, VirtReg, i);
      else
        removePhysReg(i);
    }

#if 0
  // This checking code is very expensive.
  bool AllOk = true;
  for (unsigned i = TargetRegisterInfo::FirstVirtualRegister,
           e = MF->getRegInfo().getLastVirtReg(); i <= e; ++i)
    if (unsigned PR = Virt2PhysRegMap[i]) {
      cerr << "Register still mapped: " << i << " -> " << PR << "\n";
      AllOk = false;
    }
  assert(AllOk && "Virtual registers still in phys regs?");
#endif

  // Clear any physical register which appear live at the end of the basic
  // block, but which do not hold any virtual registers.  e.g., the stack
  // pointer.
  PhysRegsUseOrder.clear();
}

/// runOnMachineFunction - Register allocate the whole function
///
bool RALocal::runOnMachineFunction(MachineFunction &Fn) {
  DOUT << "Machine Function " << "\n";
  MF = &Fn;
  TM = &Fn.getTarget();
  TRI = TM->getRegisterInfo();
  TII = TM->getInstrInfo();

  PhysRegsUsed.assign(TRI->getNumRegs(), -1);
  
  // At various places we want to efficiently check to see whether a register
  // is allocatable.  To handle this, we mark all unallocatable registers as
  // being pinned down, permanently.
  {
    BitVector Allocable = TRI->getAllocatableSet(Fn);
    for (unsigned i = 0, e = Allocable.size(); i != e; ++i)
      if (!Allocable[i])
        PhysRegsUsed[i] = -2;  // Mark the reg unallocable.
  }

  // initialize the virtual->physical register map to have a 'null'
  // mapping for all virtual registers
  unsigned LastVirtReg = MF->getRegInfo().getLastVirtReg();
  StackSlotForVirtReg.grow(LastVirtReg);
  Virt2PhysRegMap.grow(LastVirtReg);
  Virt2LastUseMap.grow(LastVirtReg);
  VirtRegModified.resize(LastVirtReg+1-TargetRegisterInfo::FirstVirtualRegister);
  UsedInMultipleBlocks.resize(LastVirtReg+1-TargetRegisterInfo::FirstVirtualRegister);
 
  // Loop over all of the basic blocks, eliminating virtual register references
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
    AllocateBasicBlock(*MBB);

  StackSlotForVirtReg.clear();
  PhysRegsUsed.clear();
  VirtRegModified.clear();
  UsedInMultipleBlocks.clear();
  Virt2PhysRegMap.clear();
  Virt2LastUseMap.clear();
  return true;
}

FunctionPass *llvm::createLocalRegisterAllocator() {
  return new RALocal();
}
