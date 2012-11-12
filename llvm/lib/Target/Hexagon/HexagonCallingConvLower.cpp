//===-- llvm/CallingConvLower.cpp - Calling Convention lowering -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Hexagon_CCState class, used for lowering and
// implementing calling conventions. Adapted from the machine independent
// version of the class (CCState) but this handles calls to varargs functions
//
//===----------------------------------------------------------------------===//

#include "HexagonCallingConvLower.h"
#include "Hexagon.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

Hexagon_CCState::Hexagon_CCState(CallingConv::ID CC, bool isVarArg,
                                 const TargetMachine &tm,
                                 SmallVector<CCValAssign, 16> &locs,
                                 LLVMContext &c)
  : CallingConv(CC), IsVarArg(isVarArg), TM(tm),
    TRI(*TM.getRegisterInfo()), Locs(locs), Context(c) {
  // No stack is used.
  StackOffset = 0;

  UsedRegs.resize((TRI.getNumRegs()+31)/32);
}

// HandleByVal - Allocate a stack slot large enough to pass an argument by
// value. The size and alignment information of the argument is encoded in its
// parameter attribute.
void Hexagon_CCState::HandleByVal(unsigned ValNo, EVT ValVT,
                                EVT LocVT, CCValAssign::LocInfo LocInfo,
                                int MinSize, int MinAlign,
                                ISD::ArgFlagsTy ArgFlags) {
  unsigned Align = ArgFlags.getByValAlign();
  unsigned Size  = ArgFlags.getByValSize();
  if (MinSize > (int)Size)
    Size = MinSize;
  if (MinAlign > (int)Align)
    Align = MinAlign;
  unsigned Offset = AllocateStack(Size, Align);

  addLoc(CCValAssign::getMem(ValNo, ValVT.getSimpleVT(), Offset,
                             LocVT.getSimpleVT(), LocInfo));
}

/// MarkAllocated - Mark a register and all of its aliases as allocated.
void Hexagon_CCState::MarkAllocated(unsigned Reg) {
  for (MCRegAliasIterator AI(Reg, &TRI, true); AI.isValid(); ++AI)
    UsedRegs[*AI/32] |= 1 << (*AI&31);
}

/// AnalyzeFormalArguments - Analyze an ISD::FORMAL_ARGUMENTS node,
/// incorporating info about the formals into this state.
void
Hexagon_CCState::AnalyzeFormalArguments(const SmallVectorImpl<ISD::InputArg>
                                        &Ins,
                                        Hexagon_CCAssignFn Fn,
                                        unsigned SretValueInRegs) {
  unsigned NumArgs = Ins.size();
  unsigned i = 0;

  // If the function returns a small struct in registers, skip
  // over the first (dummy) argument.
  if (SretValueInRegs != 0) {
    ++i;
  }


  for (; i != NumArgs; ++i) {
    EVT ArgVT = Ins[i].VT;
    ISD::ArgFlagsTy ArgFlags = Ins[i].Flags;
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this, 0, 0, false)) {
      dbgs() << "Formal argument #" << i << " has unhandled type "
             << ArgVT.getEVTString() << "\n";
      abort();
    }
  }
}

/// AnalyzeReturn - Analyze the returned values of an ISD::RET node,
/// incorporating info about the result values into this state.
void
Hexagon_CCState::AnalyzeReturn(const SmallVectorImpl<ISD::OutputArg> &Outs,
                               Hexagon_CCAssignFn Fn,
                               unsigned SretValueInRegs) {

  // For Hexagon, Return small structures in registers.
  if (SretValueInRegs != 0) {
    if (SretValueInRegs <= 32) {
      unsigned Reg = Hexagon::R0;
      addLoc(CCValAssign::getReg(0, MVT::i32, Reg, MVT::i32,
                                 CCValAssign::Full));
      return;
    }
    if (SretValueInRegs <= 64) {
      unsigned Reg = Hexagon::D0;
      addLoc(CCValAssign::getReg(0, MVT::i64, Reg, MVT::i64,
                                 CCValAssign::Full));
      return;
    }
  }


  // Determine which register each value should be copied into.
  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    EVT VT = Outs[i].VT;
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    if (Fn(i, VT, VT, CCValAssign::Full, ArgFlags, *this, -1, -1, false)){
      dbgs() << "Return operand #" << i << " has unhandled type "
           << VT.getEVTString() << "\n";
      abort();
    }
  }
}


/// AnalyzeCallOperands - Analyze an ISD::CALL node, incorporating info
/// about the passed values into this state.
void
Hexagon_CCState::AnalyzeCallOperands(const SmallVectorImpl<ISD::OutputArg>
                                     &Outs,
                                     Hexagon_CCAssignFn Fn,
                                     int NonVarArgsParams,
                                     unsigned SretValueSize) {
  unsigned NumOps = Outs.size();

  unsigned i = 0;
  // If the called function returns a small struct in registers, skip
  // the first actual parameter. We do not want to pass a pointer to
  // the stack location.
  if (SretValueSize != 0) {
    ++i;
  }

  for (; i != NumOps; ++i) {
    EVT ArgVT = Outs[i].VT;
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this,
           NonVarArgsParams, i+1, false)) {
      dbgs() << "Call operand #" << i << " has unhandled type "
           << ArgVT.getEVTString() << "\n";
      abort();
    }
  }
}

/// AnalyzeCallOperands - Same as above except it takes vectors of types
/// and argument flags.
void
Hexagon_CCState::AnalyzeCallOperands(SmallVectorImpl<EVT> &ArgVTs,
                                     SmallVectorImpl<ISD::ArgFlagsTy> &Flags,
                                     Hexagon_CCAssignFn Fn) {
  unsigned NumOps = ArgVTs.size();
  for (unsigned i = 0; i != NumOps; ++i) {
    EVT ArgVT = ArgVTs[i];
    ISD::ArgFlagsTy ArgFlags = Flags[i];
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this, -1, -1,
           false)) {
      dbgs() << "Call operand #" << i << " has unhandled type "
           << ArgVT.getEVTString() << "\n";
      abort();
    }
  }
}

/// AnalyzeCallResult - Analyze the return values of an ISD::CALL node,
/// incorporating info about the passed values into this state.
void
Hexagon_CCState::AnalyzeCallResult(const SmallVectorImpl<ISD::InputArg> &Ins,
                                   Hexagon_CCAssignFn Fn,
                                   unsigned SretValueInRegs) {

  for (unsigned i = 0, e = Ins.size(); i != e; ++i) {
    EVT VT = Ins[i].VT;
    ISD::ArgFlagsTy Flags = ISD::ArgFlagsTy();
      if (Fn(i, VT, VT, CCValAssign::Full, Flags, *this, -1, -1, false)) {
        dbgs() << "Call result #" << i << " has unhandled type "
               << VT.getEVTString() << "\n";
      abort();
    }
  }
}

/// AnalyzeCallResult - Same as above except it's specialized for calls which
/// produce a single value.
void Hexagon_CCState::AnalyzeCallResult(EVT VT, Hexagon_CCAssignFn Fn) {
  if (Fn(0, VT, VT, CCValAssign::Full, ISD::ArgFlagsTy(), *this, -1, -1,
         false)) {
    dbgs() << "Call result has unhandled type "
         << VT.getEVTString() << "\n";
    abort();
  }
}
