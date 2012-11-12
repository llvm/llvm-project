//===-- llvm/CodeGen/MachineRelocation.h - Target Relocation ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachineRelocation class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINERELOCATION_H
#define LLVM_CODEGEN_MACHINERELOCATION_H

#include "llvm/Support/DataTypes.h"
#include <cassert>

namespace llvm {
class GlobalValue;
class MachineBasicBlock;

/// MachineRelocation - This represents a target-specific relocation value,
/// produced by the code emitter.  This relocation is resolved after the has
/// been emitted, either to an object file or to memory, when the target of the
/// relocation can be resolved.
///
/// A relocation is made up of the following logical portions:
///   1. An offset in the machine code buffer, the location to modify.
///   2. A target specific relocation type (a number from 0 to 63).
///   3. A symbol being referenced, either as a GlobalValue* or as a string.
///   4. An optional constant value to be added to the reference.
///   5. A bit, CanRewrite, which indicates to the JIT that a function stub is
///      not needed for the relocation.
///   6. An index into the GOT, if the target uses a GOT
///
class MachineRelocation {
  enum AddressType {
    isResult,         // Relocation has be transformed into its result pointer.
    isGV,             // The Target.GV field is valid.
    isIndirectSym,    // Relocation of an indirect symbol.
    isBB,             // Relocation of BB address.
    isExtSym,         // The Target.ExtSym field is valid.
    isConstPool,      // Relocation of constant pool address.
    isJumpTable,      // Relocation of jump table address.
    isGOTIndex        // The Target.GOTIndex field is valid.
  };
  
  /// Offset - This is the offset from the start of the code buffer of the
  /// relocation to perform.
  uintptr_t Offset;
  
  /// ConstantVal - A field that may be used by the target relocation type.
  intptr_t ConstantVal;

  union {
    void *Result;           // If this has been resolved to a resolved pointer
    GlobalValue *GV;        // If this is a pointer to a GV or an indirect ref.
    MachineBasicBlock *MBB; // If this is a pointer to a LLVM BB
    const char *ExtSym;     // If this is a pointer to a named symbol
    unsigned Index;         // Constant pool / jump table index
    unsigned GOTIndex;      // Index in the GOT of this symbol/global
  } Target;

  unsigned TargetReloType : 6; // The target relocation ID
  AddressType AddrType    : 4; // The field of Target to use
  bool MayNeedFarStub     : 1; // True if this relocation may require a far-stub
  bool GOTRelative        : 1; // Should this relocation be relative to the GOT?
  bool TargetResolve      : 1; // True if target should resolve the address

public:
 // Relocation types used in a generic implementation.  Currently, relocation
 // entries for all things use the generic VANILLA type until they are refined
 // into target relocation types.
  enum RelocationType {
    VANILLA
  };
  
  /// MachineRelocation::getGV - Return a relocation entry for a GlobalValue.
  ///
  static MachineRelocation getGV(uintptr_t offset, unsigned RelocationType, 
                                 GlobalValue *GV, intptr_t cst = 0,
                                 bool MayNeedFarStub = 0,
                                 bool GOTrelative = 0) {
    assert((RelocationType & ~63) == 0 && "Relocation type too large!");
    MachineRelocation Result;
    Result.Offset = offset;
    Result.ConstantVal = cst;
    Result.TargetReloType = RelocationType;
    Result.AddrType = isGV;
    Result.MayNeedFarStub = MayNeedFarStub;
    Result.GOTRelative = GOTrelative;
    Result.TargetResolve = false;
    Result.Target.GV = GV;
    return Result;
  }

  /// MachineRelocation::getIndirectSymbol - Return a relocation entry for an
  /// indirect symbol.
  static MachineRelocation getIndirectSymbol(uintptr_t offset,
                                             unsigned RelocationType, 
                                             GlobalValue *GV, intptr_t cst = 0,
                                             bool MayNeedFarStub = 0,
                                             bool GOTrelative = 0) {
    assert((RelocationType & ~63) == 0 && "Relocation type too large!");
    MachineRelocation Result;
    Result.Offset = offset;
    Result.ConstantVal = cst;
    Result.TargetReloType = RelocationType;
    Result.AddrType = isIndirectSym;
    Result.MayNeedFarStub = MayNeedFarStub;
    Result.GOTRelative = GOTrelative;
    Result.TargetResolve = false;
    Result.Target.GV = GV;
    return Result;
  }

  /// MachineRelocation::getBB - Return a relocation entry for a BB.
  ///
  static MachineRelocation getBB(uintptr_t offset,unsigned RelocationType,
                                 MachineBasicBlock *MBB, intptr_t cst = 0) {
    assert((RelocationType & ~63) == 0 && "Relocation type too large!");
    MachineRelocation Result;
    Result.Offset = offset;
    Result.ConstantVal = cst;
    Result.TargetReloType = RelocationType;
    Result.AddrType = isBB;
    Result.MayNeedFarStub = false;
    Result.GOTRelative = false;
    Result.TargetResolve = false;
    Result.Target.MBB = MBB;
    return Result;
  }

  /// MachineRelocation::getExtSym - Return a relocation entry for an external
  /// symbol, like "free".
  ///
  static MachineRelocation getExtSym(uintptr_t offset, unsigned RelocationType, 
                                     const char *ES, intptr_t cst = 0,
                                     bool GOTrelative = 0,
                                     bool NeedStub = true) {
    assert((RelocationType & ~63) == 0 && "Relocation type too large!");
    MachineRelocation Result;
    Result.Offset = offset;
    Result.ConstantVal = cst;
    Result.TargetReloType = RelocationType;
    Result.AddrType = isExtSym;
    Result.MayNeedFarStub = NeedStub;
    Result.GOTRelative = GOTrelative;
    Result.TargetResolve = false;
    Result.Target.ExtSym = ES;
    return Result;
  }

  /// MachineRelocation::getConstPool - Return a relocation entry for a constant
  /// pool entry.
  ///
  static MachineRelocation getConstPool(uintptr_t offset,unsigned RelocationType,
                                        unsigned CPI, intptr_t cst = 0,
                                        bool letTargetResolve = false) {
    assert((RelocationType & ~63) == 0 && "Relocation type too large!");
    MachineRelocation Result;
    Result.Offset = offset;
    Result.ConstantVal = cst;
    Result.TargetReloType = RelocationType;
    Result.AddrType = isConstPool;
    Result.MayNeedFarStub = false;
    Result.GOTRelative = false;
    Result.TargetResolve = letTargetResolve;
    Result.Target.Index = CPI;
    return Result;
  }

  /// MachineRelocation::getJumpTable - Return a relocation entry for a jump
  /// table entry.
  ///
  static MachineRelocation getJumpTable(uintptr_t offset,unsigned RelocationType,
                                        unsigned JTI, intptr_t cst = 0,
                                        bool letTargetResolve = false) {
    assert((RelocationType & ~63) == 0 && "Relocation type too large!");
    MachineRelocation Result;
    Result.Offset = offset;
    Result.ConstantVal = cst;
    Result.TargetReloType = RelocationType;
    Result.AddrType = isJumpTable;
    Result.MayNeedFarStub = false;
    Result.GOTRelative = false;
    Result.TargetResolve = letTargetResolve;
    Result.Target.Index = JTI;
    return Result;
  }

  /// getMachineCodeOffset - Return the offset into the code buffer that the
  /// relocation should be performed.
  intptr_t getMachineCodeOffset() const {
    return Offset;
  }

  /// getRelocationType - Return the target-specific relocation ID for this
  /// relocation.
  unsigned getRelocationType() const {
    return TargetReloType;
  }

  /// getConstantVal - Get the constant value associated with this relocation.
  /// This is often an offset from the symbol.
  ///
  intptr_t getConstantVal() const {
    return ConstantVal;
  }

  /// setConstantVal - Set the constant value associated with this relocation.
  /// This is often an offset from the symbol.
  ///
  void setConstantVal(intptr_t val) {
    ConstantVal = val;
  }

  /// isGlobalValue - Return true if this relocation is a GlobalValue, as
  /// opposed to a constant string.
  bool isGlobalValue() const {
    return AddrType == isGV;
  }

  /// isIndirectSymbol - Return true if this relocation is the address an
  /// indirect symbol
  bool isIndirectSymbol() const {
    return AddrType == isIndirectSym;
  }

  /// isBasicBlock - Return true if this relocation is a basic block reference.
  ///
  bool isBasicBlock() const {
    return AddrType == isBB;
  }

  /// isExternalSymbol - Return true if this is a constant string.
  ///
  bool isExternalSymbol() const {
    return AddrType == isExtSym;
  }

  /// isConstantPoolIndex - Return true if this is a constant pool reference.
  ///
  bool isConstantPoolIndex() const {
    return AddrType == isConstPool;
  }

  /// isJumpTableIndex - Return true if this is a jump table reference.
  ///
  bool isJumpTableIndex() const {
    return AddrType == isJumpTable;
  }

  /// isGOTRelative - Return true the target wants the index into the GOT of
  /// the symbol rather than the address of the symbol.
  bool isGOTRelative() const {
    return GOTRelative;
  }

  /// mayNeedFarStub - This function returns true if the JIT for this target may
  /// need either a stub function or an indirect global-variable load to handle
  /// the relocated GlobalValue reference.  For example, the x86-64 call
  /// instruction can only call functions within +/-2GB of the call site.
  /// Anything farther away needs a longer mov+call sequence, which can't just
  /// be written on top of the existing call.
  bool mayNeedFarStub() const {
    return MayNeedFarStub;
  }

  /// letTargetResolve - Return true if the target JITInfo is usually
  /// responsible for resolving the address of this relocation.
  bool letTargetResolve() const {
    return TargetResolve;
  }

  /// getGlobalValue - If this is a global value reference, return the
  /// referenced global.
  GlobalValue *getGlobalValue() const {
    assert((isGlobalValue() || isIndirectSymbol()) &&
           "This is not a global value reference!");
    return Target.GV;
  }

  MachineBasicBlock *getBasicBlock() const {
    assert(isBasicBlock() && "This is not a basic block reference!");
    return Target.MBB;
  }

  /// getString - If this is a string value, return the string reference.
  ///
  const char *getExternalSymbol() const {
    assert(isExternalSymbol() && "This is not an external symbol reference!");
    return Target.ExtSym;
  }

  /// getConstantPoolIndex - If this is a const pool reference, return
  /// the index into the constant pool.
  unsigned getConstantPoolIndex() const {
    assert(isConstantPoolIndex() && "This is not a constant pool reference!");
    return Target.Index;
  }

  /// getJumpTableIndex - If this is a jump table reference, return
  /// the index into the jump table.
  unsigned getJumpTableIndex() const {
    assert(isJumpTableIndex() && "This is not a jump table reference!");
    return Target.Index;
  }

  /// getResultPointer - Once this has been resolved to point to an actual
  /// address, this returns the pointer.
  void *getResultPointer() const {
    assert(AddrType == isResult && "Result pointer isn't set yet!");
    return Target.Result;
  }

  /// setResultPointer - Set the result to the specified pointer value.
  ///
  void setResultPointer(void *Ptr) {
    Target.Result = Ptr;
    AddrType = isResult;
  }

  /// setGOTIndex - Set the GOT index to a specific value.
  void setGOTIndex(unsigned idx) {
    AddrType = isGOTIndex;
    Target.GOTIndex = idx;
  }

  /// getGOTIndex - Once this has been resolved to an entry in the GOT,
  /// this returns that index.  The index is from the lowest address entry
  /// in the GOT.
  unsigned getGOTIndex() const {
    assert(AddrType == isGOTIndex);
    return Target.GOTIndex;
  }
};
}

#endif
