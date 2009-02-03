//===-- llvm/CodeGen/MachineRegisterInfo.h ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachineRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEREGISTERINFO_H
#define LLVM_CODEGEN_MACHINEREGISTERINFO_H

#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/iterator.h"
#include <vector>

namespace llvm {
  
/// MachineRegisterInfo - Keep track of information for virtual and physical
/// registers, including vreg register classes, use/def chains for registers,
/// etc.
class MachineRegisterInfo {
  /// VRegInfo - Information we keep for each virtual register.  The entries in
  /// this vector are actually converted to vreg numbers by adding the 
  /// TargetRegisterInfo::FirstVirtualRegister delta to their index.
  ///
  /// Each element in this list contains the register class of the vreg and the
  /// start of the use/def list for the register.
  std::vector<std::pair<const TargetRegisterClass*, MachineOperand*> > VRegInfo;

  /// RegClassVRegMap - This vector acts as a map from TargetRegisterClass to
  /// virtual registers. For each target register class, it keeps a list of
  /// virtual registers belonging to the class.
  std::vector<std::vector<unsigned> > RegClass2VRegMap;
  
  /// PhysRegUseDefLists - This is an array of the head of the use/def list for
  /// physical registers.
  MachineOperand **PhysRegUseDefLists; 
  
  /// UsedPhysRegs - This is a bit vector that is computed and set by the
  /// register allocator, and must be kept up to date by passes that run after
  /// register allocation (though most don't modify this).  This is used
  /// so that the code generator knows which callee save registers to save and
  /// for other target specific uses.
  BitVector UsedPhysRegs;
  
  /// LiveIns/LiveOuts - Keep track of the physical registers that are
  /// livein/liveout of the function.  Live in values are typically arguments in
  /// registers, live out values are typically return values in registers.
  /// LiveIn values are allowed to have virtual registers associated with them,
  /// stored in the second element.
  std::vector<std::pair<unsigned, unsigned> > LiveIns;
  std::vector<unsigned> LiveOuts;
  
  MachineRegisterInfo(const MachineRegisterInfo&); // DO NOT IMPLEMENT
  void operator=(const MachineRegisterInfo&);      // DO NOT IMPLEMENT
public:
  explicit MachineRegisterInfo(const TargetRegisterInfo &TRI);
  ~MachineRegisterInfo();
  
  //===--------------------------------------------------------------------===//
  // Register Info
  //===--------------------------------------------------------------------===//

  /// reg_begin/reg_end - Provide iteration support to walk over all definitions
  /// and uses of a register within the MachineFunction that corresponds to this
  /// MachineRegisterInfo object.
  template<bool Uses, bool Defs>
  class defusechain_iterator;

  /// reg_iterator/reg_begin/reg_end - Walk all defs and uses of the specified
  /// register.
  typedef defusechain_iterator<true,true> reg_iterator;
  reg_iterator reg_begin(unsigned RegNo) const {
    return reg_iterator(getRegUseDefListHead(RegNo));
  }
  static reg_iterator reg_end() { return reg_iterator(0); }

  /// reg_empty - Return true if there are no instructions using or defining the
  /// specified register (it may be live-in).
  bool reg_empty(unsigned RegNo) const { return reg_begin(RegNo) == reg_end(); }

  /// def_iterator/def_begin/def_end - Walk all defs of the specified register.
  typedef defusechain_iterator<false,true> def_iterator;
  def_iterator def_begin(unsigned RegNo) const {
    return def_iterator(getRegUseDefListHead(RegNo));
  }
  static def_iterator def_end() { return def_iterator(0); }

  /// def_empty - Return true if there are no instructions defining the
  /// specified register (it may be live-in).
  bool def_empty(unsigned RegNo) const { return def_begin(RegNo) == def_end(); }

  /// use_iterator/use_begin/use_end - Walk all uses of the specified register.
  typedef defusechain_iterator<true,false> use_iterator;
  use_iterator use_begin(unsigned RegNo) const {
    return use_iterator(getRegUseDefListHead(RegNo));
  }
  static use_iterator use_end() { return use_iterator(0); }
  
  /// use_empty - Return true if there are no instructions using the specified
  /// register.
  bool use_empty(unsigned RegNo) const { return use_begin(RegNo) == use_end(); }

  
  /// replaceRegWith - Replace all instances of FromReg with ToReg in the
  /// machine function.  This is like llvm-level X->replaceAllUsesWith(Y),
  /// except that it also changes any definitions of the register as well.
  void replaceRegWith(unsigned FromReg, unsigned ToReg);
  
  /// getRegUseDefListHead - Return the head pointer for the register use/def
  /// list for the specified virtual or physical register.
  MachineOperand *&getRegUseDefListHead(unsigned RegNo) {
    if (RegNo < TargetRegisterInfo::FirstVirtualRegister)
      return PhysRegUseDefLists[RegNo];
    RegNo -= TargetRegisterInfo::FirstVirtualRegister;
    return VRegInfo[RegNo].second;
  }
  
  MachineOperand *getRegUseDefListHead(unsigned RegNo) const {
    if (RegNo < TargetRegisterInfo::FirstVirtualRegister)
      return PhysRegUseDefLists[RegNo];
    RegNo -= TargetRegisterInfo::FirstVirtualRegister;
    return VRegInfo[RegNo].second;
  }

  /// getVRegDef - Return the machine instr that defines the specified virtual
  /// register or null if none is found.  This assumes that the code is in SSA
  /// form, so there should only be one definition.
  MachineInstr *getVRegDef(unsigned Reg) const;
  
#ifndef NDEBUG
  void dumpUses(unsigned RegNo) const;
#endif
  
  //===--------------------------------------------------------------------===//
  // Virtual Register Info
  //===--------------------------------------------------------------------===//
  
  /// getRegClass - Return the register class of the specified virtual register.
  ///
  const TargetRegisterClass *getRegClass(unsigned Reg) const {
    Reg -= TargetRegisterInfo::FirstVirtualRegister;
    assert(Reg < VRegInfo.size() && "Invalid vreg!");
    return VRegInfo[Reg].first;
  }

  /// setRegClass - Set the register class of the specified virtual register.
  ///
  void setRegClass(unsigned Reg, const TargetRegisterClass *RC) {
    unsigned VR = Reg;
    Reg -= TargetRegisterInfo::FirstVirtualRegister;
    assert(Reg < VRegInfo.size() && "Invalid vreg!");
    const TargetRegisterClass *OldRC = VRegInfo[Reg].first;
    VRegInfo[Reg].first = RC;

    // Remove from old register class's vregs list. This may be slow but
    // fortunately this operation is rarely needed.
    std::vector<unsigned> &VRegs = RegClass2VRegMap[OldRC->getID()];
    std::vector<unsigned>::iterator I=std::find(VRegs.begin(), VRegs.end(), VR);
    VRegs.erase(I);

    // Add to new register class's vregs list.
    RegClass2VRegMap[RC->getID()].push_back(VR);
  }
  
  /// createVirtualRegister - Create and return a new virtual register in the
  /// function with the specified register class.
  ///
  unsigned createVirtualRegister(const TargetRegisterClass *RegClass);

  /// getLastVirtReg - Return the highest currently assigned virtual register.
  ///
  unsigned getLastVirtReg() const {
    return (unsigned)VRegInfo.size()+TargetRegisterInfo::FirstVirtualRegister-1;
  }

  /// getRegClassVirtRegs - Return the list of virtual registers of the given
  /// target register class.
  std::vector<unsigned> &getRegClassVirtRegs(const TargetRegisterClass *RC) {
    return RegClass2VRegMap[RC->getID()];
  }
  
  //===--------------------------------------------------------------------===//
  // Physical Register Use Info
  //===--------------------------------------------------------------------===//
  
  /// isPhysRegUsed - Return true if the specified register is used in this
  /// function.  This only works after register allocation.
  bool isPhysRegUsed(unsigned Reg) const { return UsedPhysRegs[Reg]; }
  
  /// setPhysRegUsed - Mark the specified register used in this function.
  /// This should only be called during and after register allocation.
  void setPhysRegUsed(unsigned Reg) { UsedPhysRegs[Reg] = true; }
  
  /// setPhysRegUnused - Mark the specified register unused in this function.
  /// This should only be called during and after register allocation.
  void setPhysRegUnused(unsigned Reg) { UsedPhysRegs[Reg] = false; }
  

  //===--------------------------------------------------------------------===//
  // LiveIn/LiveOut Management
  //===--------------------------------------------------------------------===//
  
  /// addLiveIn/Out - Add the specified register as a live in/out.  Note that it
  /// is an error to add the same register to the same set more than once.
  void addLiveIn(unsigned Reg, unsigned vreg = 0) {
    LiveIns.push_back(std::make_pair(Reg, vreg));
  }
  void addLiveOut(unsigned Reg) { LiveOuts.push_back(Reg); }
  
  // Iteration support for live in/out sets.  These sets are kept in sorted
  // order by their register number.
  typedef std::vector<std::pair<unsigned,unsigned> >::const_iterator
  livein_iterator;
  typedef std::vector<unsigned>::const_iterator liveout_iterator;
  livein_iterator livein_begin() const { return LiveIns.begin(); }
  livein_iterator livein_end()   const { return LiveIns.end(); }
  bool            livein_empty() const { return LiveIns.empty(); }
  liveout_iterator liveout_begin() const { return LiveOuts.begin(); }
  liveout_iterator liveout_end()   const { return LiveOuts.end(); }
  bool             liveout_empty() const { return LiveOuts.empty(); }

  bool isLiveIn(unsigned Reg) const {
    for (livein_iterator I = livein_begin(), E = livein_end(); I != E; ++I)
      if (I->first == Reg || I->second == Reg)
        return true;
    return false;
  }

private:
  void HandleVRegListReallocation();
  
public:
  /// defusechain_iterator - This class provides iterator support for machine
  /// operands in the function that use or define a specific register.  If
  /// ReturnUses is true it returns uses of registers, if ReturnDefs is true it
  /// returns defs.  If neither are true then you are silly and it always
  /// returns end().
  template<bool ReturnUses, bool ReturnDefs>
  class defusechain_iterator
    : public forward_iterator<MachineInstr, ptrdiff_t> {
    MachineOperand *Op;
    explicit defusechain_iterator(MachineOperand *op) : Op(op) {
      // If the first node isn't one we're interested in, advance to one that
      // we are interested in.
      if (op) {
        if ((!ReturnUses && op->isUse()) ||
            (!ReturnDefs && op->isDef()))
          ++*this;
      }
    }
    friend class MachineRegisterInfo;
  public:
    typedef forward_iterator<MachineInstr, ptrdiff_t>::reference reference;
    typedef forward_iterator<MachineInstr, ptrdiff_t>::pointer pointer;
    
    defusechain_iterator(const defusechain_iterator &I) : Op(I.Op) {}
    defusechain_iterator() : Op(0) {}
    
    bool operator==(const defusechain_iterator &x) const {
      return Op == x.Op;
    }
    bool operator!=(const defusechain_iterator &x) const {
      return !operator==(x);
    }
    
    /// atEnd - return true if this iterator is equal to reg_end() on the value.
    bool atEnd() const { return Op == 0; }
    
    // Iterator traversal: forward iteration only
    defusechain_iterator &operator++() {          // Preincrement
      assert(Op && "Cannot increment end iterator!");
      Op = Op->getNextOperandForReg();
      
      // If this is an operand we don't care about, skip it.
      while (Op && ((!ReturnUses && Op->isUse()) || 
                    (!ReturnDefs && Op->isDef())))
        Op = Op->getNextOperandForReg();
      
      return *this;
    }
    defusechain_iterator operator++(int) {        // Postincrement
      defusechain_iterator tmp = *this; ++*this; return tmp;
    }
    
    MachineOperand &getOperand() const {
      assert(Op && "Cannot dereference end iterator!");
      return *Op;
    }
    
    /// getOperandNo - Return the operand # of this MachineOperand in its
    /// MachineInstr.
    unsigned getOperandNo() const {
      assert(Op && "Cannot dereference end iterator!");
      return Op - &Op->getParent()->getOperand(0);
    }
    
    // Retrieve a reference to the current operand.
    MachineInstr &operator*() const {
      assert(Op && "Cannot dereference end iterator!");
      return *Op->getParent();
    }
    
    MachineInstr *operator->() const {
      assert(Op && "Cannot dereference end iterator!");
      return Op->getParent();
    }
  };
  
};

} // End llvm namespace

#endif
