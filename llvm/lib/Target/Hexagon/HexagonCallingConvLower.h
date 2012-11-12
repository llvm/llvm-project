//===-- HexagonCallingConvLower.h - Calling Conventions ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Hexagon_CCState class, used for lowering
// and implementing calling conventions. Adapted from the target independent
// version but this handles calls to varargs functions
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_Hexagon_CODEGEN_CALLINGCONVLOWER_H
#define LLVM_Hexagon_CODEGEN_CALLINGCONVLOWER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/CallingConvLower.h"

//
// Need to handle varargs.
//
namespace llvm {
  class TargetRegisterInfo;
  class TargetMachine;
  class Hexagon_CCState;
  class SDNode;


/// Hexagon_CCAssignFn - This function assigns a location for Val, updating
/// State to reflect the change.
typedef bool Hexagon_CCAssignFn(unsigned ValNo, EVT ValVT,
                              EVT LocVT, CCValAssign::LocInfo LocInfo,
                              ISD::ArgFlagsTy ArgFlags, Hexagon_CCState &State,
                              int NonVarArgsParams,
                              int CurrentParam,
                              bool ForceMem);


/// CCState - This class holds information needed while lowering arguments and
/// return values.  It captures which registers are already assigned and which
/// stack slots are used.  It provides accessors to allocate these values.
class Hexagon_CCState {
  CallingConv::ID CallingConv;
  bool IsVarArg;
  const TargetMachine &TM;
  const TargetRegisterInfo &TRI;
  SmallVector<CCValAssign, 16> &Locs;
  LLVMContext &Context;

  unsigned StackOffset;
  SmallVector<uint32_t, 16> UsedRegs;
public:
  Hexagon_CCState(CallingConv::ID CC, bool isVarArg, const TargetMachine &TM,
                SmallVector<CCValAssign, 16> &locs, LLVMContext &c);

  void addLoc(const CCValAssign &V) {
    Locs.push_back(V);
  }

  LLVMContext &getContext() const { return Context; }
  const TargetMachine &getTarget() const { return TM; }
  unsigned getCallingConv() const { return CallingConv; }
  bool isVarArg() const { return IsVarArg; }

  unsigned getNextStackOffset() const { return StackOffset; }

  /// isAllocated - Return true if the specified register (or an alias) is
  /// allocated.
  bool isAllocated(unsigned Reg) const {
    return UsedRegs[Reg/32] & (1 << (Reg&31));
  }

  /// AnalyzeFormalArguments - Analyze an ISD::FORMAL_ARGUMENTS node,
  /// incorporating info about the formals into this state.
  void AnalyzeFormalArguments(const SmallVectorImpl<ISD::InputArg> &Ins,
                              Hexagon_CCAssignFn Fn, unsigned SretValueInRegs);

  /// AnalyzeReturn - Analyze the returned values of an ISD::RET node,
  /// incorporating info about the result values into this state.
  void AnalyzeReturn(const SmallVectorImpl<ISD::OutputArg> &Outs,
                     Hexagon_CCAssignFn Fn, unsigned SretValueInRegs);

  /// AnalyzeCallOperands - Analyze an ISD::CALL node, incorporating info
  /// about the passed values into this state.
  void AnalyzeCallOperands(const SmallVectorImpl<ISD::OutputArg> &Outs,
                           Hexagon_CCAssignFn Fn, int NonVarArgsParams,
                           unsigned SretValueSize);

  /// AnalyzeCallOperands - Same as above except it takes vectors of types
  /// and argument flags.
  void AnalyzeCallOperands(SmallVectorImpl<EVT> &ArgVTs,
                           SmallVectorImpl<ISD::ArgFlagsTy> &Flags,
                           Hexagon_CCAssignFn Fn);

  /// AnalyzeCallResult - Analyze the return values of an ISD::CALL node,
  /// incorporating info about the passed values into this state.
  void AnalyzeCallResult(const SmallVectorImpl<ISD::InputArg> &Ins,
                         Hexagon_CCAssignFn Fn, unsigned SretValueInRegs);

  /// AnalyzeCallResult - Same as above except it's specialized for calls which
  /// produce a single value.
  void AnalyzeCallResult(EVT VT, Hexagon_CCAssignFn Fn);

  /// getFirstUnallocated - Return the first unallocated register in the set, or
  /// NumRegs if they are all allocated.
  unsigned getFirstUnallocated(const unsigned *Regs, unsigned NumRegs) const {
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
  unsigned AllocateReg(const unsigned *Regs, unsigned NumRegs) {
    unsigned FirstUnalloc = getFirstUnallocated(Regs, NumRegs);
    if (FirstUnalloc == NumRegs)
      return 0;    // Didn't find the reg.

    // Mark the register and any aliases as allocated.
    unsigned Reg = Regs[FirstUnalloc];
    MarkAllocated(Reg);
    return Reg;
  }

  /// Version of AllocateReg with list of registers to be shadowed.
  unsigned AllocateReg(const unsigned *Regs, const unsigned *ShadowRegs,
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
    return Result;
  }

  // HandleByVal - Allocate a stack slot large enough to pass an argument by
  // value. The size and alignment information of the argument is encoded in its
  // parameter attribute.
  void HandleByVal(unsigned ValNo, EVT ValVT,
                   EVT LocVT, CCValAssign::LocInfo LocInfo,
                   int MinSize, int MinAlign, ISD::ArgFlagsTy ArgFlags);

private:
  /// MarkAllocated - Mark a register and all of its aliases as allocated.
  void MarkAllocated(unsigned Reg);
};



} // end namespace llvm

#endif
