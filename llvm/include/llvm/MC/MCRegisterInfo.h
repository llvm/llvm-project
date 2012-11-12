//=== MC/MCRegisterInfo.h - Target Register Description ---------*- C++ -*-===//
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

#ifndef LLVM_MC_MCREGISTERINFO_H
#define LLVM_MC_MCREGISTERINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

namespace llvm {

/// MCRegisterClass - Base class of TargetRegisterClass.
class MCRegisterClass {
public:
  typedef const uint16_t* iterator;
  typedef const uint16_t* const_iterator;

  const char *Name;
  const iterator RegsBegin;
  const uint8_t *const RegSet;
  const uint16_t RegsSize;
  const uint16_t RegSetSize;
  const uint16_t ID;
  const uint16_t RegSize, Alignment; // Size & Alignment of register in bytes
  const int8_t CopyCost;
  const bool Allocatable;

  /// getID() - Return the register class ID number.
  ///
  unsigned getID() const { return ID; }

  /// getName() - Return the register class name for debugging.
  ///
  const char *getName() const { return Name; }

  /// begin/end - Return all of the registers in this class.
  ///
  iterator       begin() const { return RegsBegin; }
  iterator         end() const { return RegsBegin + RegsSize; }

  /// getNumRegs - Return the number of registers in this class.
  ///
  unsigned getNumRegs() const { return RegsSize; }

  /// getRegister - Return the specified register in the class.
  ///
  unsigned getRegister(unsigned i) const {
    assert(i < getNumRegs() && "Register number out of range!");
    return RegsBegin[i];
  }

  /// contains - Return true if the specified register is included in this
  /// register class.  This does not include virtual registers.
  bool contains(unsigned Reg) const {
    unsigned InByte = Reg % 8;
    unsigned Byte = Reg / 8;
    if (Byte >= RegSetSize)
      return false;
    return (RegSet[Byte] & (1 << InByte)) != 0;
  }

  /// contains - Return true if both registers are in this class.
  bool contains(unsigned Reg1, unsigned Reg2) const {
    return contains(Reg1) && contains(Reg2);
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

  /// isAllocatable - Return true if this register class may be used to create
  /// virtual registers.
  bool isAllocatable() const { return Allocatable; }
};

/// MCRegisterDesc - This record contains all of the information known about
/// a particular register.  The Overlaps field contains a pointer to a zero
/// terminated array of registers that this register aliases, starting with
/// itself. This is needed for architectures like X86 which have AL alias AX
/// alias EAX. The SubRegs field is a zero terminated array of registers that
/// are sub-registers of the specific register, e.g. AL, AH are sub-registers of
/// AX. The SuperRegs field is a zero terminated array of registers that are
/// super-registers of the specific register, e.g. RAX, EAX, are super-registers
/// of AX.
///
struct MCRegisterDesc {
  uint32_t Name;      // Printable name for the reg (for debugging)
  uint32_t Overlaps;  // Overlapping registers, described above
  uint32_t SubRegs;   // Sub-register set, described above
  uint32_t SuperRegs; // Super-register set, described above

  // Offset into MCRI::SubRegIndices of a list of sub-register indices for each
  // sub-register in SubRegs.
  uint32_t SubRegIndices;

  // RegUnits - Points to the list of register units. The low 4 bits holds the
  // Scale, the high bits hold an offset into DiffLists. See MCRegUnitIterator.
  uint32_t RegUnits;
};

/// MCRegisterInfo base class - We assume that the target defines a static
/// array of MCRegisterDesc objects that represent all of the machine
/// registers that the target has.  As such, we simply have to track a pointer
/// to this array so that we can turn register number into a register
/// descriptor.
///
/// Note this class is designed to be a base class of TargetRegisterInfo, which
/// is the interface used by codegen. However, specific targets *should never*
/// specialize this class. MCRegisterInfo should only contain getters to access
/// TableGen generated physical register data. It must not be extended with
/// virtual methods.
///
class MCRegisterInfo {
public:
  typedef const MCRegisterClass *regclass_iterator;

  /// DwarfLLVMRegPair - Emitted by tablegen so Dwarf<->LLVM reg mappings can be
  /// performed with a binary search.
  struct DwarfLLVMRegPair {
    unsigned FromReg;
    unsigned ToReg;

    bool operator<(DwarfLLVMRegPair RHS) const { return FromReg < RHS.FromReg; }
  };
private:
  const MCRegisterDesc *Desc;                 // Pointer to the descriptor array
  unsigned NumRegs;                           // Number of entries in the array
  unsigned RAReg;                             // Return address register
  const MCRegisterClass *Classes;             // Pointer to the regclass array
  unsigned NumClasses;                        // Number of entries in the array
  unsigned NumRegUnits;                       // Number of regunits.
  const uint16_t (*RegUnitRoots)[2];          // Pointer to regunit root table.
  const uint16_t *DiffLists;                  // Pointer to the difflists array
  const char *RegStrings;                     // Pointer to the string table.
  const uint16_t *SubRegIndices;              // Pointer to the subreg lookup
                                              // array.
  unsigned NumSubRegIndices;                  // Number of subreg indices.
  const uint16_t *RegEncodingTable;           // Pointer to array of register
                                              // encodings.

  unsigned L2DwarfRegsSize;
  unsigned EHL2DwarfRegsSize;
  unsigned Dwarf2LRegsSize;
  unsigned EHDwarf2LRegsSize;
  const DwarfLLVMRegPair *L2DwarfRegs;        // LLVM to Dwarf regs mapping
  const DwarfLLVMRegPair *EHL2DwarfRegs;      // LLVM to Dwarf regs mapping EH
  const DwarfLLVMRegPair *Dwarf2LRegs;        // Dwarf to LLVM regs mapping
  const DwarfLLVMRegPair *EHDwarf2LRegs;      // Dwarf to LLVM regs mapping EH
  DenseMap<unsigned, int> L2SEHRegs;          // LLVM to SEH regs mapping

public:
  /// DiffListIterator - Base iterator class that can traverse the
  /// differentially encoded register and regunit lists in DiffLists.
  /// Don't use this class directly, use one of the specialized sub-classes
  /// defined below.
  class DiffListIterator {
    uint16_t Val;
    const uint16_t *List;

  protected:
    /// Create an invalid iterator. Call init() to point to something useful.
    DiffListIterator() : Val(0), List(0) {}

    /// init - Point the iterator to InitVal, decoding subsequent values from
    /// DiffList. The iterator will initially point to InitVal, sub-classes are
    /// responsible for skipping the seed value if it is not part of the list.
    void init(uint16_t InitVal, const uint16_t *DiffList) {
      Val = InitVal;
      List = DiffList;
    }

    /// advance - Move to the next list position, return the applied
    /// differential. This function does not detect the end of the list, that
    /// is the caller's responsibility (by checking for a 0 return value).
    unsigned advance() {
      assert(isValid() && "Cannot move off the end of the list.");
      uint16_t D = *List++;
      Val += D;
      return D;
    }

  public:

    /// isValid - returns true if this iterator is not yet at the end.
    bool isValid() const { return List; }

    /// Dereference the iterator to get the value at the current position.
    unsigned operator*() const { return Val; }

    /// Pre-increment to move to the next position.
    void operator++() {
      // The end of the list is encoded as a 0 differential.
      if (!advance())
        List = 0;
    }
  };

  // These iterators are allowed to sub-class DiffListIterator and access
  // internal list pointers.
  friend class MCSubRegIterator;
  friend class MCSuperRegIterator;
  friend class MCRegAliasIterator;
  friend class MCRegUnitIterator;
  friend class MCRegUnitRootIterator;

  /// InitMCRegisterInfo - Initialize MCRegisterInfo, called by TableGen
  /// auto-generated routines. *DO NOT USE*.
  void InitMCRegisterInfo(const MCRegisterDesc *D, unsigned NR, unsigned RA,
                          const MCRegisterClass *C, unsigned NC,
                          const uint16_t (*RURoots)[2],
                          unsigned NRU,
                          const uint16_t *DL,
                          const char *Strings,
                          const uint16_t *SubIndices,
                          unsigned NumIndices,
                          const uint16_t *RET) {
    Desc = D;
    NumRegs = NR;
    RAReg = RA;
    Classes = C;
    DiffLists = DL;
    RegStrings = Strings;
    NumClasses = NC;
    RegUnitRoots = RURoots;
    NumRegUnits = NRU;
    SubRegIndices = SubIndices;
    NumSubRegIndices = NumIndices;
    RegEncodingTable = RET;
  }

  /// mapLLVMRegsToDwarfRegs - Used to initialize LLVM register to Dwarf
  /// register number mapping. Called by TableGen auto-generated routines.
  /// *DO NOT USE*.
  void mapLLVMRegsToDwarfRegs(const DwarfLLVMRegPair *Map, unsigned Size,
                              bool isEH) {
    if (isEH) {
      EHL2DwarfRegs = Map;
      EHL2DwarfRegsSize = Size;
    } else {
      L2DwarfRegs = Map;
      L2DwarfRegsSize = Size;
    }
  }

  /// mapDwarfRegsToLLVMRegs - Used to initialize Dwarf register to LLVM
  /// register number mapping. Called by TableGen auto-generated routines.
  /// *DO NOT USE*.
  void mapDwarfRegsToLLVMRegs(const DwarfLLVMRegPair *Map, unsigned Size,
                              bool isEH) {
    if (isEH) {
      EHDwarf2LRegs = Map;
      EHDwarf2LRegsSize = Size;
    } else {
      Dwarf2LRegs = Map;
      Dwarf2LRegsSize = Size;
    }
  }

  /// mapLLVMRegToSEHReg - Used to initialize LLVM register to SEH register
  /// number mapping. By default the SEH register number is just the same
  /// as the LLVM register number.
  /// FIXME: TableGen these numbers. Currently this requires target specific
  /// initialization code.
  void mapLLVMRegToSEHReg(unsigned LLVMReg, int SEHReg) {
    L2SEHRegs[LLVMReg] = SEHReg;
  }

  /// getRARegister - This method should return the register where the return
  /// address can be found.
  unsigned getRARegister() const {
    return RAReg;
  }

  const MCRegisterDesc &operator[](unsigned RegNo) const {
    assert(RegNo < NumRegs &&
           "Attempting to access record for invalid register number!");
    return Desc[RegNo];
  }

  /// Provide a get method, equivalent to [], but more useful if we have a
  /// pointer to this object.
  ///
  const MCRegisterDesc &get(unsigned RegNo) const {
    return operator[](RegNo);
  }

  /// getSubReg - Returns the physical register number of sub-register "Index"
  /// for physical register RegNo. Return zero if the sub-register does not
  /// exist.
  unsigned getSubReg(unsigned Reg, unsigned Idx) const;

  /// getMatchingSuperReg - Return a super-register of the specified register
  /// Reg so its sub-register of index SubIdx is Reg.
  unsigned getMatchingSuperReg(unsigned Reg, unsigned SubIdx,
                               const MCRegisterClass *RC) const;

  /// getSubRegIndex - For a given register pair, return the sub-register index
  /// if the second register is a sub-register of the first. Return zero
  /// otherwise.
  unsigned getSubRegIndex(unsigned RegNo, unsigned SubRegNo) const;

  /// getName - Return the human-readable symbolic target-specific name for the
  /// specified physical register.
  const char *getName(unsigned RegNo) const {
    return RegStrings + get(RegNo).Name;
  }

  /// getNumRegs - Return the number of registers this target has (useful for
  /// sizing arrays holding per register information)
  unsigned getNumRegs() const {
    return NumRegs;
  }

  /// getNumSubRegIndices - Return the number of sub-register indices
  /// understood by the target. Index 0 is reserved for the no-op sub-register,
  /// while 1 to getNumSubRegIndices() - 1 represent real sub-registers.
  unsigned getNumSubRegIndices() const {
    return NumSubRegIndices;
  }

  /// getNumRegUnits - Return the number of (native) register units in the
  /// target. Register units are numbered from 0 to getNumRegUnits() - 1. They
  /// can be accessed through MCRegUnitIterator defined below.
  unsigned getNumRegUnits() const {
    return NumRegUnits;
  }

  /// getDwarfRegNum - Map a target register to an equivalent dwarf register
  /// number.  Returns -1 if there is no equivalent value.  The second
  /// parameter allows targets to use different numberings for EH info and
  /// debugging info.
  int getDwarfRegNum(unsigned RegNum, bool isEH) const;

  /// getLLVMRegNum - Map a dwarf register back to a target register.
  ///
  int getLLVMRegNum(unsigned RegNum, bool isEH) const;

  /// getSEHRegNum - Map a target register to an equivalent SEH register
  /// number.  Returns LLVM register number if there is no equivalent value.
  int getSEHRegNum(unsigned RegNum) const;

  regclass_iterator regclass_begin() const { return Classes; }
  regclass_iterator regclass_end() const { return Classes+NumClasses; }

  unsigned getNumRegClasses() const {
    return (unsigned)(regclass_end()-regclass_begin());
  }

  /// getRegClass - Returns the register class associated with the enumeration
  /// value.  See class MCOperandInfo.
  const MCRegisterClass& getRegClass(unsigned i) const {
    assert(i < getNumRegClasses() && "Register Class ID out of range");
    return Classes[i];
  }

   /// getEncodingValue - Returns the encoding for RegNo
  uint16_t getEncodingValue(unsigned RegNo) const {
    assert(RegNo < NumRegs &&
           "Attempting to get encoding for invalid register number!");
    return RegEncodingTable[RegNo];
  }

};

//===----------------------------------------------------------------------===//
//                          Register List Iterators
//===----------------------------------------------------------------------===//

// MCRegisterInfo provides lists of super-registers, sub-registers, and
// aliasing registers. Use these iterator classes to traverse the lists.

/// MCSubRegIterator enumerates all sub-registers of Reg.
class MCSubRegIterator : public MCRegisterInfo::DiffListIterator {
public:
  MCSubRegIterator(unsigned Reg, const MCRegisterInfo *MCRI) {
    init(Reg, MCRI->DiffLists + MCRI->get(Reg).SubRegs);
    ++*this;
  }
};

/// MCSuperRegIterator enumerates all super-registers of Reg.
class MCSuperRegIterator : public MCRegisterInfo::DiffListIterator {
public:
  MCSuperRegIterator(unsigned Reg, const MCRegisterInfo *MCRI) {
    init(Reg, MCRI->DiffLists + MCRI->get(Reg).SuperRegs);
    ++*this;
  }
};

/// MCRegAliasIterator enumerates all registers aliasing Reg.
/// If IncludeSelf is set, Reg itself is included in the list.
class MCRegAliasIterator : public MCRegisterInfo::DiffListIterator {
public:
  MCRegAliasIterator(unsigned Reg, const MCRegisterInfo *MCRI,
                     bool IncludeSelf) {
    init(Reg, MCRI->DiffLists + MCRI->get(Reg).Overlaps);
    // Initially, the iterator points to Reg itself.
    if (!IncludeSelf)
      ++*this;
  }
};

//===----------------------------------------------------------------------===//
//                               Register Units
//===----------------------------------------------------------------------===//

// Register units are used to compute register aliasing. Every register has at
// least one register unit, but it can have more. Two registers overlap if and
// only if they have a common register unit.
//
// A target with a complicated sub-register structure will typically have many
// fewer register units than actual registers. MCRI::getNumRegUnits() returns
// the number of register units in the target.

// MCRegUnitIterator enumerates a list of register units for Reg. The list is
// in ascending numerical order.
class MCRegUnitIterator : public MCRegisterInfo::DiffListIterator {
public:
  /// MCRegUnitIterator - Create an iterator that traverses the register units
  /// in Reg.
  MCRegUnitIterator(unsigned Reg, const MCRegisterInfo *MCRI) {
    // Decode the RegUnits MCRegisterDesc field.
    unsigned RU = MCRI->get(Reg).RegUnits;
    unsigned Scale = RU & 15;
    unsigned Offset = RU >> 4;

    // Initialize the iterator to Reg * Scale, and the List pointer to
    // DiffLists + Offset.
    init(Reg * Scale, MCRI->DiffLists + Offset);

    // That may not be a valid unit, we need to advance by one to get the real
    // unit number. The first differential can be 0 which would normally
    // terminate the list, but since we know every register has at least one
    // unit, we can allow a 0 differential here.
    advance();
  }
};

// Each register unit has one or two root registers. The complete set of
// registers containing a register unit is the union of the roots and their
// super-registers. All registers aliasing Unit can be visited like this:
//
//   for (MCRegUnitRootIterator RI(Unit, MCRI); RI.isValid(); ++RI) {
//     unsigned Root = *RI;
//     visit(Root);
//     for (MCSuperRegIterator SI(Root, MCRI); SI.isValid(); ++SI)
//       visit(*SI);
//    }

/// MCRegUnitRootIterator enumerates the root registers of a register unit.
class MCRegUnitRootIterator {
  uint16_t Reg0;
  uint16_t Reg1;
public:
  MCRegUnitRootIterator(unsigned RegUnit, const MCRegisterInfo *MCRI) {
    assert(RegUnit < MCRI->getNumRegUnits() && "Invalid register unit");
    Reg0 = MCRI->RegUnitRoots[RegUnit][0];
    Reg1 = MCRI->RegUnitRoots[RegUnit][1];
  }

  /// Dereference to get the current root register.
  unsigned operator*() const {
    return Reg0;
  }

  /// isValid - Check if the iterator is at the end of the list.
  bool isValid() const {
    return Reg0;
  }

  /// Preincrement to move to the next root register.
  void operator++() {
    assert(isValid() && "Cannot move off the end of the list.");
    Reg0 = Reg1;
    Reg1 = 0;
  }
};

} // End llvm namespace

#endif
