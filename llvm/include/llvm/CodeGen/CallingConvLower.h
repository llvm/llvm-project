//===-- llvm/CallingConvLower.h - Calling Conventions -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the CCState and CCValAssign classes, used for lowering
// and implementing calling conventions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_CALLINGCONVLOWER_H
#define LLVM_CODEGEN_CALLINGCONVLOWER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Target/TargetCallingConv.h"
#include "llvm/CallingConv.h"

namespace llvm {
  class TargetRegisterInfo;
  class TargetMachine;
  class CCState;

/// CCValAssign - Represent assignment of one arg/retval to a location.
class CCValAssign {
public:
  enum LocInfo {
    Full,   // The value fills the full location.
    SExt,   // The value is sign extended in the location.
    ZExt,   // The value is zero extended in the location.
    AExt,   // The value is extended with undefined upper bits.
    BCvt,   // The value is bit-converted in the location.
    VExt,   // The value is vector-widened in the location.
            // FIXME: Not implemented yet. Code that uses AExt to mean
            // vector-widen should be fixed to use VExt instead.
    Indirect // The location contains pointer to the value.
    // TODO: a subset of the value is in the location.
  };
private:
  /// ValNo - This is the value number begin assigned (e.g. an argument number).
  unsigned ValNo;

  /// Loc is either a stack offset or a register number.
  unsigned Loc;

  /// isMem - True if this is a memory loc, false if it is a register loc.
  bool isMem : 1;

  /// isCustom - True if this arg/retval requires special handling.
  bool isCustom : 1;

  /// Information about how the value is assigned.
  LocInfo HTP : 6;

  /// ValVT - The type of the value being assigned.
  MVT ValVT;

  /// LocVT - The type of the location being assigned to.
  MVT LocVT;
public:

  static CCValAssign getReg(unsigned ValNo, MVT ValVT,
                            unsigned RegNo, MVT LocVT,
                            LocInfo HTP) {
    CCValAssign Ret;
    Ret.ValNo = ValNo;
    Ret.Loc = RegNo;
    Ret.isMem = false;
    Ret.isCustom = false;
    Ret.HTP = HTP;
    Ret.ValVT = ValVT;
    Ret.LocVT = LocVT;
    return Ret;
  }

  static CCValAssign getCustomReg(unsigned ValNo, MVT ValVT,
                                  unsigned RegNo, MVT LocVT,
                                  LocInfo HTP) {
    CCValAssign Ret;
    Ret = getReg(ValNo, ValVT, RegNo, LocVT, HTP);
    Ret.isCustom = true;
    return Ret;
  }

  static CCValAssign getMem(unsigned ValNo, MVT ValVT,
                            unsigned Offset, MVT LocVT,
                            LocInfo HTP) {
    CCValAssign Ret;
    Ret.ValNo = ValNo;
    Ret.Loc = Offset;
    Ret.isMem = true;
    Ret.isCustom = false;
    Ret.HTP = HTP;
    Ret.ValVT = ValVT;
    Ret.LocVT = LocVT;
    return Ret;
  }

  static CCValAssign getCustomMem(unsigned ValNo, MVT ValVT,
                                  unsigned Offset, MVT LocVT,
                                  LocInfo HTP) {
    CCValAssign Ret;
    Ret = getMem(ValNo, ValVT, Offset, LocVT, HTP);
    Ret.isCustom = true;
    return Ret;
  }

  unsigned getValNo() const { return ValNo; }
  MVT getValVT() const { return ValVT; }

  bool isRegLoc() const { return !isMem; }
  bool isMemLoc() const { return isMem; }

  bool needsCustom() const { return isCustom; }

  unsigned getLocReg() const { assert(isRegLoc()); return Loc; }
  unsigned getLocMemOffset() const { assert(isMemLoc()); return Loc; }
  MVT getLocVT() const { return LocVT; }

  LocInfo getLocInfo() const { return HTP; }
  bool isExtInLoc() const {
    return (HTP == AExt || HTP == SExt || HTP == ZExt);
  }

};

/// CCAssignFn - This function assigns a location for Val, updating State to
/// reflect the change.  It returns 'true' if it failed to handle Val.
typedef bool CCAssignFn(unsigned ValNo, MVT ValVT,
                        MVT LocVT, CCValAssign::LocInfo LocInfo,
                        ISD::ArgFlagsTy ArgFlags, CCState &State);

/// CCCustomFn - This function assigns a location for Val, possibly updating
/// all args to reflect changes and indicates if it handled it. It must set
/// isCustom if it handles the arg and returns true.
typedef bool CCCustomFn(unsigned &ValNo, MVT &ValVT,
                        MVT &LocVT, CCValAssign::LocInfo &LocInfo,
                        ISD::ArgFlagsTy &ArgFlags, CCState &State);

/// ParmContext - This enum tracks whether calling convention lowering is in
/// the context of prologue or call generation. Not all backends make use of
/// this information.
typedef enum { Unknown, Prologue, Call } ParmContext;

/// CCState - This class holds information needed while lowering arguments and
/// return values.  It captures which registers are already assigned and which
/// stack slots are used.  It provides accessors to allocate these values.
class CCState {
private:
  CallingConv::ID CallingConv;
  bool IsVarArg;
  MachineFunction &MF;
  const TargetMachine &TM;
  const TargetRegisterInfo &TRI;
  SmallVector<CCValAssign, 16> &Locs;
  LLVMContext &Context;

  unsigned StackOffset;
  SmallVector<uint32_t, 16> UsedRegs;
  unsigned FirstByValReg;
  bool FirstByValRegValid;

protected:
  ParmContext CallOrPrologue;

public:
  CCState(CallingConv::ID CC, bool isVarArg, MachineFunction &MF,
          const TargetMachine &TM, SmallVector<CCValAssign, 16> &locs,
          LLVMContext &C);

  void addLoc(const CCValAssign &V) {
    Locs.push_back(V);
  }

  LLVMContext &getContext() const { return Context; }
  const TargetMachine &getTarget() const { return TM; }
  MachineFunction &getMachineFunction() const { return MF; }
  CallingConv::ID getCallingConv() const { return CallingConv; }
  bool isVarArg() const { return IsVarArg; }

  unsigned getNextStackOffset() const { return StackOffset; }

  /// isAllocated - Return true if the specified register (or an alias) is
  /// allocated.
  bool isAllocated(unsigned Reg) const {
    return UsedRegs[Reg/32] & (1 << (Reg&31));
  }

  /// AnalyzeFormalArguments - Analyze an array of argument values,
  /// incorporating info about the formals into this state.
  void AnalyzeFormalArguments(const SmallVectorImpl<ISD::InputArg> &Ins,
                              CCAssignFn Fn);

  /// AnalyzeReturn - Analyze the returned values of a return,
  /// incorporating info about the result values into this state.
  void AnalyzeReturn(const SmallVectorImpl<ISD::OutputArg> &Outs,
                     CCAssignFn Fn);

  /// CheckReturn - Analyze the return values of a function, returning
  /// true if the return can be performed without sret-demotion, and
  /// false otherwise.
  bool CheckReturn(const SmallVectorImpl<ISD::OutputArg> &ArgsFlags,
                   CCAssignFn Fn);

  /// AnalyzeCallOperands - Analyze the outgoing arguments to a call,
  /// incorporating info about the passed values into this state.
  void AnalyzeCallOperands(const SmallVectorImpl<ISD::OutputArg> &Outs,
                           CCAssignFn Fn);

  /// AnalyzeCallOperands - Same as above except it takes vectors of types
  /// and argument flags.
  void AnalyzeCallOperands(SmallVectorImpl<MVT> &ArgVTs,
                           SmallVectorImpl<ISD::ArgFlagsTy> &Flags,
                           CCAssignFn Fn);

  /// AnalyzeCallResult - Analyze the return values of a call,
  /// incorporating info about the passed values into this state.
  void AnalyzeCallResult(const SmallVectorImpl<ISD::InputArg> &Ins,
                         CCAssignFn Fn);

  /// AnalyzeCallResult - Same as above except it's specialized for calls which
  /// produce a single value.
  void AnalyzeCallResult(MVT VT, CCAssignFn Fn);

  /// getFirstUnallocated - Return the first unallocated register in the set, or
  /// NumRegs if they are all allocated.
  unsigned getFirstUnallocated(const uint16_t *Regs, unsigned NumRegs) const {
    for (unsigned i = 0; i != NumRegs; ++i)
      if (!isAllocated(Regs[i]))
        return i;
    return NumRegs;
  }

  /// AllocateReg - Attempt to allocate one register.  If it is not available,
  /// return zero.  Otherwise, return the register, marking it and any aliases
  /// as allocated.
  unsigned AllocateReg(unsigned Reg) {
    if (isAllocated(Reg)) return 0;
    MarkAllocated(Reg);
    return Reg;
  }

  /// Version of AllocateReg with extra register to be shadowed.
  unsigned AllocateReg(unsigned Reg, unsigned ShadowReg) {
    if (isAllocated(Reg)) return 0;
    MarkAllocated(Reg);
    MarkAllocated(ShadowReg);
    return Reg;
  }

  /// AllocateReg - Attempt to allocate one of the specified registers.  If none
  /// are available, return zero.  Otherwise, return the first one available,
  /// marking it and any aliases as allocated.
  unsigned AllocateReg(const uint16_t *Regs, unsigned NumRegs) {
    unsigned FirstUnalloc = getFirstUnallocated(Regs, NumRegs);
    if (FirstUnalloc == NumRegs)
      return 0;    // Didn't find the reg.

    // Mark the register and any aliases as allocated.
    unsigned Reg = Regs[FirstUnalloc];
    MarkAllocated(Reg);
    return Reg;
  }

  /// Version of AllocateReg with list of registers to be shadowed.
  unsigned AllocateReg(const uint16_t *Regs, const uint16_t *ShadowRegs,
                       unsigned NumRegs) {
    unsigned FirstUnalloc = getFirstUnallocated(Regs, NumRegs);
    if (FirstUnalloc == NumRegs)
      return 0;    // Didn't find the reg.

    // Mark the register and any aliases as allocated.
    unsigned Reg = Regs[FirstUnalloc], ShadowReg = ShadowRegs[FirstUnalloc];
    MarkAllocated(Reg);
    MarkAllocated(ShadowReg);
    return Reg;
  }

  /// AllocateStack - Allocate a chunk of stack space with the specified size
  /// and alignment.
  unsigned AllocateStack(unsigned Size, unsigned Align) {
    assert(Align && ((Align-1) & Align) == 0); // Align is power of 2.
    StackOffset = ((StackOffset + Align-1) & ~(Align-1));
    unsigned Result = StackOffset;
    StackOffset += Size;
    MF.getFrameInfo()->ensureMaxAlignment(Align);
    return Result;
  }

  /// Version of AllocateStack with extra register to be shadowed.
  unsigned AllocateStack(unsigned Size, unsigned Align, unsigned ShadowReg) {
    MarkAllocated(ShadowReg);
    return AllocateStack(Size, Align);
  }

  // HandleByVal - Allocate a stack slot large enough to pass an argument by
  // value. The size and alignment information of the argument is encoded in its
  // parameter attribute.
  void HandleByVal(unsigned ValNo, MVT ValVT,
                   MVT LocVT, CCValAssign::LocInfo LocInfo,
                   int MinSize, int MinAlign, ISD::ArgFlagsTy ArgFlags);

  // First GPR that carries part of a byval aggregate that's split
  // between registers and memory.
  unsigned getFirstByValReg() const { return FirstByValRegValid ? FirstByValReg : 0; }
  void setFirstByValReg(unsigned r) { FirstByValReg = r; FirstByValRegValid = true; }
  void clearFirstByValReg() { FirstByValReg = 0; FirstByValRegValid = false; }
  bool isFirstByValRegValid() const { return FirstByValRegValid; }

  ParmContext getCallOrPrologue() const { return CallOrPrologue; }

private:
  /// MarkAllocated - Mark a register and all of its aliases as allocated.
  void MarkAllocated(unsigned Reg);
};



} // end namespace llvm

#endif
