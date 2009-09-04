//=== Target/TargetRegisterInfo.h - Target Register Information -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes an abstract interface used to get information about a
// target machines register file.  This information is used for a variety of
// purposed, especially register allocation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETREGISTERINFO_H
#define LLVM_TARGET_TARGETREGISTERINFO_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/ADT/DenseSet.h"
#include <cassert>
#include <functional>

namespace llvm {

class BitVector;
class MachineFunction;
class MachineMove;
class RegScavenger;

/// TargetRegisterDesc - This record contains all of the information known about
/// a particular register.  The AliasSet field (if not null) contains a pointer
/// to a Zero terminated array of registers that this register aliases.  This is
/// needed for architectures like X86 which have AL alias AX alias EAX.
/// Registers that this does not apply to simply should set this to null.
/// The SubRegs field is a zero terminated array of registers that are
/// sub-registers of the specific register, e.g. AL, AH are sub-registers of AX.
/// The SuperRegs field is a zero terminated array of registers that are
/// super-registers of the specific register, e.g. RAX, EAX, are super-registers
/// of AX.
///
struct TargetRegisterDesc {
  const char     *AsmName;      // Assembly language name for the register
  const char     *Name;         // Printable name for the reg (for debugging)
  const unsigned *AliasSet;     // Register Alias Set, described above
  const unsigned *SubRegs;      // Sub-register set, described above
  const unsigned *SuperRegs;    // Super-register set, described above
};

class TargetRegisterClass {
public:
  typedef const unsigned* iterator;
  typedef const unsigned* const_iterator;

  typedef const EVT* vt_iterator;
  typedef const TargetRegisterClass* const * sc_iterator;
private:
  unsigned ID;
  const char *Name;
  const vt_iterator VTs;
  const sc_iterator SubClasses;
  const sc_iterator SuperClasses;
  const sc_iterator SubRegClasses;
  const sc_iterator SuperRegClasses;
  const unsigned RegSize, Alignment;    // Size & Alignment of register in bytes
  const int CopyCost;
  const iterator RegsBegin, RegsEnd;
  DenseSet<unsigned> RegSet;
public:
  TargetRegisterClass(unsigned id,
                      const char *name,
                      const EVT *vts,
                      const TargetRegisterClass * const *subcs,
                      const TargetRegisterClass * const *supcs,
                      const TargetRegisterClass * const *subregcs,
                      const TargetRegisterClass * const *superregcs,
                      unsigned RS, unsigned Al, int CC,
                      iterator RB, iterator RE)
    : ID(id), Name(name), VTs(vts), SubClasses(subcs), SuperClasses(supcs),
    SubRegClasses(subregcs), SuperRegClasses(superregcs),
    RegSize(RS), Alignment(Al), CopyCost(CC), RegsBegin(RB), RegsEnd(RE) {
      for (iterator I = RegsBegin, E = RegsEnd; I != E; ++I)
        RegSet.insert(*I);
    }
  virtual ~TargetRegisterClass() {}     // Allow subclasses
  
  /// getID() - Return the register class ID number.
  ///
  unsigned getID() const { return ID; }

  /// getName() - Return the register class name for debugging.
  ///
  const char *getName() const { return Name; }

  /// begin/end - Return all of the registers in this class.
  ///
  iterator       begin() const { return RegsBegin; }
  iterator         end() const { return RegsEnd; }

  /// getNumRegs - Return the number of registers in this class.
  ///
  unsigned getNumRegs() const { return (unsigned)(RegsEnd-RegsBegin); }

  /// getRegister - Return the specified register in the class.
  ///
  unsigned getRegister(unsigned i) const {
    assert(i < getNumRegs() && "Register number out of range!");
    return RegsBegin[i];
  }

  /// contains - Return true if the specified register is included in this
  /// register class.
  bool contains(unsigned Reg) const {
    return RegSet.count(Reg);
  }

  /// hasType - return true if this TargetRegisterClass has the ValueType vt.
  ///
  bool hasType(EVT vt) const {
    for(int i = 0; VTs[i].getSimpleVT().SimpleTy != MVT::Other; ++i)
      if (VTs[i] == vt)
        return true;
    return false;
  }
  
  /// vt_begin / vt_end - Loop over all of the value types that can be
  /// represented by values in this register class.
  vt_iterator vt_begin() const {
    return VTs;
  }

  vt_iterator vt_end() const {
    vt_iterator I = VTs;
    while (I->getSimpleVT().SimpleTy != MVT::Other) ++I;
    return I;
  }

  /// subregclasses_begin / subregclasses_end - Loop over all of
  /// the subreg register classes of this register class.
  sc_iterator subregclasses_begin() const {
    return SubRegClasses;
  }

  sc_iterator subregclasses_end() const {
    sc_iterator I = SubRegClasses;
    while (*I != NULL) ++I;
    return I;
  }

  /// getSubRegisterRegClass - Return the register class of subregisters with
  /// index SubIdx, or NULL if no such class exists.
  const TargetRegisterClass* getSubRegisterRegClass(unsigned SubIdx) const {
    assert(SubIdx>0 && "Invalid subregister index");
    for (unsigned s = 0; s != SubIdx-1; ++s)
      if (!SubRegClasses[s])
        return NULL;
    return SubRegClasses[SubIdx-1];
  }

  /// superregclasses_begin / superregclasses_end - Loop over all of
  /// the superreg register classes of this register class.
  sc_iterator superregclasses_begin() const {
    return SuperRegClasses;
  }

  sc_iterator superregclasses_end() const {
    sc_iterator I = SuperRegClasses;
    while (*I != NULL) ++I;
    return I;
  }

  /// hasSubClass - return true if the the specified TargetRegisterClass
  /// is a proper subset of this TargetRegisterClass.
  bool hasSubClass(const TargetRegisterClass *cs) const {
    for (int i = 0; SubClasses[i] != NULL; ++i) 
      if (SubClasses[i] == cs)
        return true;
    return false;
  }

  /// subclasses_begin / subclasses_end - Loop over all of the classes
  /// that are proper subsets of this register class.
  sc_iterator subclasses_begin() const {
    return SubClasses;
  }
  
  sc_iterator subclasses_end() const {
    sc_iterator I = SubClasses;
    while (*I != NULL) ++I;
    return I;
  }
  
  /// hasSuperClass - return true if the specified TargetRegisterClass is a
  /// proper superset of this TargetRegisterClass.
  bool hasSuperClass(const TargetRegisterClass *cs) const {
    for (int i = 0; SuperClasses[i] != NULL; ++i) 
      if (SuperClasses[i] == cs)
        return true;
    return false;
  }

  /// superclasses_begin / superclasses_end - Loop over all of the classes
  /// that are proper supersets of this register class.
  sc_iterator superclasses_begin() const {
    return SuperClasses;
  }
  
  sc_iterator superclasses_end() const {
    sc_iterator I = SuperClasses;
    while (*I != NULL) ++I;
    return I;
  }

  /// isASubClass - return true if this TargetRegisterClass is a subset
  /// class of at least one other TargetRegisterClass.
  bool isASubClass() const {
    return SuperClasses[0] != 0;
  }
  
  /// allocation_order_begin/end - These methods define a range of registers
  /// which specify the registers in this class that are valid to register
  /// allocate, and the preferred order to allocate them in.  For example,
  /// callee saved registers should be at the end of the list, because it is
  /// cheaper to allocate caller saved registers.
  ///
  /// These methods take a MachineFunction argument, which can be used to tune
  /// the allocatable registers based on the characteristics of the function.
  /// One simple example is that the frame pointer register can be used if
  /// frame-pointer-elimination is performed.
  ///
  /// By default, these methods return all registers in the class.
  ///
  virtual iterator allocation_order_begin(const MachineFunction &MF) const {
    return begin();
  }
  virtual iterator allocation_order_end(const MachineFunction &MF)   const {
    return end();
  }

  /// getSize - Return the size of the register in bytes, which is also the size
  /// of a stack slot allocated to hold a spilled copy of this register.
  unsigned getSize() const { return RegSize; }

  /// getAlignment - Return the minimum required alignment for a register of
  /// this class.
  unsigned getAlignment() const { return Alignment; }

  /// getCopyCost - Return the cost of copying a value between two registers in
  /// this class. A negative number means the register class is very expensive
  /// to copy e.g. status flag register classes.
  int getCopyCost() const { return CopyCost; }
};


/// TargetRegisterInfo base class - We assume that the target defines a static
/// array of TargetRegisterDesc objects that represent all of the machine
/// registers that the target has.  As such, we simply have to track a pointer
/// to this array so that we can turn register number into a register
/// descriptor.
///
class TargetRegisterInfo {
protected:
  const unsigned* SubregHash;
  const unsigned SubregHashSize;
  const unsigned* SuperregHash;
  const unsigned SuperregHashSize;
  const unsigned* AliasesHash;
  const unsigned AliasesHashSize;
public:
  typedef const TargetRegisterClass * const * regclass_iterator;
private:
  const TargetRegisterDesc *Desc;             // Pointer to the descriptor array
  unsigned NumRegs;                           // Number of entries in the array

  regclass_iterator RegClassBegin, RegClassEnd;   // List of regclasses

  int CallFrameSetupOpcode, CallFrameDestroyOpcode;
protected:
  TargetRegisterInfo(const TargetRegisterDesc *D, unsigned NR,
                     regclass_iterator RegClassBegin,
                     regclass_iterator RegClassEnd,
                     int CallFrameSetupOpcode = -1,
                     int CallFrameDestroyOpcode = -1,
                     const unsigned* subregs = 0,
                     const unsigned subregsize = 0,
                     const unsigned* superregs = 0,
                     const unsigned superregsize = 0,
                     const unsigned* aliases = 0,
                     const unsigned aliasessize = 0);
  virtual ~TargetRegisterInfo();
public:

  enum {                        // Define some target independent constants
    /// NoRegister - This physical register is not a real target register.  It
    /// is useful as a sentinal.
    NoRegister = 0,

    /// FirstVirtualRegister - This is the first register number that is
    /// considered to be a 'virtual' register, which is part of the SSA
    /// namespace.  This must be the same for all targets, which means that each
    /// target is limited to 1024 registers.
    FirstVirtualRegister = 1024
  };

  /// isPhysicalRegister - Return true if the specified register number is in
  /// the physical register namespace.
  static bool isPhysicalRegister(unsigned Reg) {
    assert(Reg && "this is not a register!");
    return Reg < FirstVirtualRegister;
  }

  /// isVirtualRegister - Return true if the specified register number is in
  /// the virtual register namespace.
  static bool isVirtualRegister(unsigned Reg) {
    assert(Reg && "this is not a register!");
    return Reg >= FirstVirtualRegister;
  }

  /// getPhysicalRegisterRegClass - Returns the Register Class of a physical
  /// register of the given type. If type is EVT::Other, then just return any
  /// register class the register belongs to.
  virtual const TargetRegisterClass *
    getPhysicalRegisterRegClass(unsigned Reg, EVT VT = MVT::Other) const;

  /// getAllocatableSet - Returns a bitset indexed by register number
  /// indicating if a register is allocatable or not. If a register class is
  /// specified, returns the subset for the class.
  BitVector getAllocatableSet(MachineFunction &MF,
                              const TargetRegisterClass *RC = NULL) const;

  const TargetRegisterDesc &operator[](unsigned RegNo) const {
    assert(RegNo < NumRegs &&
           "Attempting to access record for invalid register number!");
    return Desc[RegNo];
  }

  /// Provide a get method, equivalent to [], but more useful if we have a
  /// pointer to this object.
  ///
  const TargetRegisterDesc &get(unsigned RegNo) const {
    return operator[](RegNo);
  }

  /// getAliasSet - Return the set of registers aliased by the specified
  /// register, or a null list of there are none.  The list returned is zero
  /// terminated.
  ///
  const unsigned *getAliasSet(unsigned RegNo) const {
    return get(RegNo).AliasSet;
  }

  /// getSubRegisters - Return the list of registers that are sub-registers of
  /// the specified register, or a null list of there are none. The list
  /// returned is zero terminated and sorted according to super-sub register
  /// relations. e.g. X86::RAX's sub-register list is EAX, AX, AL, AH.
  ///
  const unsigned *getSubRegisters(unsigned RegNo) const {
    return get(RegNo).SubRegs;
  }

  /// getSuperRegisters - Return the list of registers that are super-registers
  /// of the specified register, or a null list of there are none. The list
  /// returned is zero terminated and sorted according to super-sub register
  /// relations. e.g. X86::AL's super-register list is RAX, EAX, AX.
  ///
  const unsigned *getSuperRegisters(unsigned RegNo) const {
    return get(RegNo).SuperRegs;
  }

  /// getAsmName - Return the symbolic target-specific name for the
  /// specified physical register.
  const char *getAsmName(unsigned RegNo) const {
    return get(RegNo).AsmName;
  }

  /// getName - Return the human-readable symbolic target-specific name for the
  /// specified physical register.
  const char *getName(unsigned RegNo) const {
    return get(RegNo).Name;
  }

  /// getNumRegs - Return the number of registers this target has (useful for
  /// sizing arrays holding per register information)
  unsigned getNumRegs() const {
    return NumRegs;
  }

  /// regsOverlap - Returns true if the two registers are equal or alias each
  /// other. The registers may be virtual register.
  bool regsOverlap(unsigned regA, unsigned regB) const {
    if (regA == regB)
      return true;

    if (isVirtualRegister(regA) || isVirtualRegister(regB))
      return false;

    // regA and regB are distinct physical registers. Do they alias?
    size_t index = (regA + regB * 37) & (AliasesHashSize-1);
    unsigned ProbeAmt = 0;
    while (AliasesHash[index*2] != 0 &&
           AliasesHash[index*2+1] != 0) {
      if (AliasesHash[index*2] == regA && AliasesHash[index*2+1] == regB)
        return true;

      index = (index + ProbeAmt) & (AliasesHashSize-1);
      ProbeAmt += 2;
    }

    return false;
  }

  /// isSubRegister - Returns true if regB is a sub-register of regA.
  ///
  bool isSubRegister(unsigned regA, unsigned regB) const {
    // SubregHash is a simple quadratically probed hash table.
    size_t index = (regA + regB * 37) & (SubregHashSize-1);
    unsigned ProbeAmt = 2;
    while (SubregHash[index*2] != 0 &&
           SubregHash[index*2+1] != 0) {
      if (SubregHash[index*2] == regA && SubregHash[index*2+1] == regB)
        return true;
      
      index = (index + ProbeAmt) & (SubregHashSize-1);
      ProbeAmt += 2;
    }
    
    return false;
  }

  /// isSuperRegister - Returns true if regB is a super-register of regA.
  ///
  bool isSuperRegister(unsigned regA, unsigned regB) const {
    // SuperregHash is a simple quadratically probed hash table.
    size_t index = (regA + regB * 37) & (SuperregHashSize-1);
    unsigned ProbeAmt = 2;
    while (SuperregHash[index*2] != 0 &&
           SuperregHash[index*2+1] != 0) {
      if (SuperregHash[index*2] == regA && SuperregHash[index*2+1] == regB)
        return true;
      
      index = (index + ProbeAmt) & (SuperregHashSize-1);
      ProbeAmt += 2;
    }
    
    return false;
  }

  /// getCalleeSavedRegs - Return a null-terminated list of all of the
  /// callee saved registers on this target. The register should be in the
  /// order of desired callee-save stack frame offset. The first register is
  /// closed to the incoming stack pointer if stack grows down, and vice versa.
  virtual const unsigned* getCalleeSavedRegs(const MachineFunction *MF = 0)
                                                                      const = 0;

  /// getCalleeSavedRegClasses - Return a null-terminated list of the preferred
  /// register classes to spill each callee saved register with.  The order and
  /// length of this list match the getCalleeSaveRegs() list.
  virtual const TargetRegisterClass* const *getCalleeSavedRegClasses(
                                            const MachineFunction *MF) const =0;

  /// getReservedRegs - Returns a bitset indexed by physical register number
  /// indicating if a register is a special register that has particular uses
  /// and should be considered unavailable at all times, e.g. SP, RA. This is
  /// used by register scavenger to determine what registers are free.
  virtual BitVector getReservedRegs(const MachineFunction &MF) const = 0;

  /// getSubReg - Returns the physical register number of sub-register "Index"
  /// for physical register RegNo. Return zero if the sub-register does not
  /// exist.
  virtual unsigned getSubReg(unsigned RegNo, unsigned Index) const = 0;

  /// getMatchingSuperReg - Return a super-register of the specified register
  /// Reg so its sub-register of index SubIdx is Reg.
  unsigned getMatchingSuperReg(unsigned Reg, unsigned SubIdx, 
                               const TargetRegisterClass *RC) const {
    for (const unsigned *SRs = getSuperRegisters(Reg); unsigned SR = *SRs;++SRs)
      if (Reg == getSubReg(SR, SubIdx) && RC->contains(SR))
        return SR;
    return 0;
  }

  /// getMatchingSuperRegClass - Return a subclass of the specified register
  /// class A so that each register in it has a sub-register of the
  /// specified sub-register index which is in the specified register class B.
  virtual const TargetRegisterClass *
  getMatchingSuperRegClass(const TargetRegisterClass *A,
                           const TargetRegisterClass *B, unsigned Idx) const {
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Register Class Information
  //

  /// Register class iterators
  ///
  regclass_iterator regclass_begin() const { return RegClassBegin; }
  regclass_iterator regclass_end() const { return RegClassEnd; }

  unsigned getNumRegClasses() const {
    return (unsigned)(regclass_end()-regclass_begin());
  }
  
  /// getRegClass - Returns the register class associated with the enumeration
  /// value.  See class TargetOperandInfo.
  const TargetRegisterClass *getRegClass(unsigned i) const {
    assert(i <= getNumRegClasses() && "Register Class ID out of range");
    return i ? RegClassBegin[i - 1] : NULL;
  }

  /// getPointerRegClass - Returns a TargetRegisterClass used for pointer
  /// values.  If a target supports multiple different pointer register classes,
  /// kind specifies which one is indicated.
  virtual const TargetRegisterClass *getPointerRegClass(unsigned Kind=0) const {
    assert(0 && "Target didn't implement getPointerRegClass!");
    return 0; // Must return a value in order to compile with VS 2005
  }

  /// getCrossCopyRegClass - Returns a legal register class to copy a register
  /// in the specified class to or from. Returns NULL if it is possible to copy
  /// between a two registers of the specified class.
  virtual const TargetRegisterClass *
  getCrossCopyRegClass(const TargetRegisterClass *RC) const {
    return NULL;
  }

  /// getAllocationOrder - Returns the register allocation order for a specified
  /// register class in the form of a pair of TargetRegisterClass iterators.
  virtual std::pair<TargetRegisterClass::iterator,TargetRegisterClass::iterator>
  getAllocationOrder(const TargetRegisterClass *RC,
                     unsigned HintType, unsigned HintReg,
                     const MachineFunction &MF) const {
    return std::make_pair(RC->allocation_order_begin(MF),
                          RC->allocation_order_end(MF));
  }

  /// ResolveRegAllocHint - Resolves the specified register allocation hint
  /// to a physical register. Returns the physical register if it is successful.
  virtual unsigned ResolveRegAllocHint(unsigned Type, unsigned Reg,
                                       const MachineFunction &MF) const {
    if (Type == 0 && Reg && isPhysicalRegister(Reg))
      return Reg;
    return 0;
  }

  /// UpdateRegAllocHint - A callback to allow target a chance to update
  /// register allocation hints when a register is "changed" (e.g. coalesced)
  /// to another register. e.g. On ARM, some virtual registers should target
  /// register pairs, if one of pair is coalesced to another register, the
  /// allocation hint of the other half of the pair should be changed to point
  /// to the new register.
  virtual void UpdateRegAllocHint(unsigned Reg, unsigned NewReg,
                                  MachineFunction &MF) const {
    // Do nothing.
  }

  /// targetHandlesStackFrameRounding - Returns true if the target is
  /// responsible for rounding up the stack frame (probably at emitPrologue
  /// time).
  virtual bool targetHandlesStackFrameRounding() const {
    return false;
  }

  /// requiresRegisterScavenging - returns true if the target requires (and can
  /// make use of) the register scavenger.
  virtual bool requiresRegisterScavenging(const MachineFunction &MF) const {
    return false;
  }
  
  /// hasFP - Return true if the specified function should have a dedicated
  /// frame pointer register. For most targets this is true only if the function
  /// has variable sized allocas or if frame pointer elimination is disabled.
  virtual bool hasFP(const MachineFunction &MF) const = 0;

  /// hasReservedCallFrame - Under normal circumstances, when a frame pointer is
  /// not required, we reserve argument space for call sites in the function
  /// immediately on entry to the current function. This eliminates the need for
  /// add/sub sp brackets around call sites. Returns true if the call frame is
  /// included as part of the stack frame.
  virtual bool hasReservedCallFrame(MachineFunction &MF) const {
    return !hasFP(MF);
  }

  /// hasReservedSpillSlot - Return true if target has reserved a spill slot in
  /// the stack frame of the given function for the specified register. e.g. On
  /// x86, if the frame register is required, the first fixed stack object is
  /// reserved as its spill slot. This tells PEI not to create a new stack frame
  /// object for the given register. It should be called only after
  /// processFunctionBeforeCalleeSavedScan().
  virtual bool hasReservedSpillSlot(MachineFunction &MF, unsigned Reg,
                                    int &FrameIdx) const {
    return false;
  }

  /// needsStackRealignment - true if storage within the function requires the
  /// stack pointer to be aligned more than the normal calling convention calls
  /// for.
  virtual bool needsStackRealignment(const MachineFunction &MF) const {
    return false;
  }

  /// getCallFrameSetup/DestroyOpcode - These methods return the opcode of the
  /// frame setup/destroy instructions if they exist (-1 otherwise).  Some
  /// targets use pseudo instructions in order to abstract away the difference
  /// between operating with a frame pointer and operating without, through the
  /// use of these two instructions.
  ///
  int getCallFrameSetupOpcode() const { return CallFrameSetupOpcode; }
  int getCallFrameDestroyOpcode() const { return CallFrameDestroyOpcode; }

  /// eliminateCallFramePseudoInstr - This method is called during prolog/epilog
  /// code insertion to eliminate call frame setup and destroy pseudo
  /// instructions (but only if the Target is using them).  It is responsible
  /// for eliminating these instructions, replacing them with concrete
  /// instructions.  This method need only be implemented if using call frame
  /// setup/destroy pseudo instructions.
  ///
  virtual void
  eliminateCallFramePseudoInstr(MachineFunction &MF,
                                MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const {
    assert(getCallFrameSetupOpcode()== -1 && getCallFrameDestroyOpcode()== -1 &&
           "eliminateCallFramePseudoInstr must be implemented if using"
           " call frame setup/destroy pseudo instructions!");
    assert(0 && "Call Frame Pseudo Instructions do not exist on this target!");
  }

  /// processFunctionBeforeCalleeSavedScan - This method is called immediately
  /// before PrologEpilogInserter scans the physical registers used to determine
  /// what callee saved registers should be spilled. This method is optional.
  virtual void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                                RegScavenger *RS = NULL) const {

  }

  /// processFunctionBeforeFrameFinalized - This method is called immediately
  /// before the specified functions frame layout (MF.getFrameInfo()) is
  /// finalized.  Once the frame is finalized, MO_FrameIndex operands are
  /// replaced with direct constants.  This method is optional.
  ///
  virtual void processFunctionBeforeFrameFinalized(MachineFunction &MF) const {
  }

  /// eliminateFrameIndex - This method must be overriden to eliminate abstract
  /// frame indices from instructions which may use them.  The instruction
  /// referenced by the iterator contains an MO_FrameIndex operand which must be
  /// eliminated by this method.  This method may modify or replace the
  /// specified instruction, as long as it keeps the iterator pointing the the
  /// finished product. SPAdj is the SP adjustment due to call frame setup
  /// instruction.
  virtual void eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                   int SPAdj, RegScavenger *RS=NULL) const = 0;

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  virtual void emitPrologue(MachineFunction &MF) const = 0;
  virtual void emitEpilogue(MachineFunction &MF,
                            MachineBasicBlock &MBB) const = 0;
                            
  //===--------------------------------------------------------------------===//
  /// Debug information queries.
  
  /// getDwarfRegNum - Map a target register to an equivalent dwarf register
  /// number.  Returns -1 if there is no equivalent value.  The second
  /// parameter allows targets to use different numberings for EH info and
  /// debugging info.
  virtual int getDwarfRegNum(unsigned RegNum, bool isEH) const = 0;

  /// getFrameRegister - This method should return the register used as a base
  /// for values allocated in the current stack frame.
  virtual unsigned getFrameRegister(MachineFunction &MF) const = 0;

  /// getFrameIndexOffset - Returns the displacement from the frame register to
  /// the stack frame of the specified index.
  virtual int getFrameIndexOffset(MachineFunction &MF, int FI) const;
                           
  /// getRARegister - This method should return the register where the return
  /// address can be found.
  virtual unsigned getRARegister() const = 0;
  
  /// getInitialFrameState - Returns a list of machine moves that are assumed
  /// on entry to all functions.  Note that LabelID is ignored (assumed to be
  /// the beginning of the function.)
  virtual void getInitialFrameState(std::vector<MachineMove> &Moves) const;
};


// This is useful when building IndexedMaps keyed on virtual registers
struct VirtReg2IndexFunctor : std::unary_function<unsigned, unsigned> {
  unsigned operator()(unsigned Reg) const {
    return Reg - TargetRegisterInfo::FirstVirtualRegister;
  }
};

/// getCommonSubClass - find the largest common subclass of A and B. Return NULL
/// if there is no common subclass.
const TargetRegisterClass *getCommonSubClass(const TargetRegisterClass *A,
                                             const TargetRegisterClass *B);

} // End llvm namespace

#endif
