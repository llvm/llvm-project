//===-- CallingConvLower.cpp - Calling Conventions ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CCState class, used for lowering and implementing
// calling conventions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetLowering.h"
using namespace llvm;

CCState::CCState(CallingConv::ID CC, bool isVarArg, MachineFunction &mf,
                 const TargetMachine &tm, SmallVector<CCValAssign, 16> &locs,
                 LLVMContext &C)
  : CallingConv(CC), IsVarArg(isVarArg), MF(mf), TM(tm),
    TRI(*TM.getRegisterInfo()), Locs(locs), Context(C),
    CallOrPrologue(Unknown) {
  // No stack is used.
  StackOffset = 0;

  clearFirstByValReg();
  UsedRegs.resize((TRI.getNumRegs()+31)/32);
}

// HandleByVal - Allocate space on the stack large enough to pass an argument
// by value. The size and alignment information of the argument is encoded in
// its parameter attribute.
void CCState::HandleByVal(unsigned ValNo, MVT ValVT,
                          MVT LocVT, CCValAssign::LocInfo LocInfo,
                          int MinSize, int MinAlign,
                          ISD::ArgFlagsTy ArgFlags) {
  unsigned Align = ArgFlags.getByValAlign();
  unsigned Size  = ArgFlags.getByValSize();
  if (MinSize > (int)Size)
    Size = MinSize;
  if (MinAlign > (int)Align)
    Align = MinAlign;
  MF.getFrameInfo()->ensureMaxAlignment(Align);
  TM.getTargetLowering()->HandleByVal(this, Size, Align);
  unsigned Offset = AllocateStack(Size, Align);
  addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
}

/// MarkAllocated - Mark a register and all of its aliases as allocated.
void CCState::MarkAllocated(unsigned Reg) {
  for (MCRegAliasIterator AI(Reg, &TRI, true); AI.isValid(); ++AI)
    UsedRegs[*AI/32] |= 1 << (*AI&31);
}

/// AnalyzeFormalArguments - Analyze an array of argument values,
/// incorporating info about the formals into this state.
void
CCState::AnalyzeFormalArguments(const SmallVectorImpl<ISD::InputArg> &Ins,
                                CCAssignFn Fn) {
  unsigned NumArgs = Ins.size();

  for (unsigned i = 0; i != NumArgs; ++i) {
    MVT ArgVT = Ins[i].VT;
    ISD::ArgFlagsTy ArgFlags = Ins[i].Flags;
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this)) {
#ifndef NDEBUG
      dbgs() << "Formal argument #" << i << " has unhandled type "
             << EVT(ArgVT).getEVTString();
#endif
      llvm_unreachable(0);
    }
  }
}

/// CheckReturn - Analyze the return values of a function, returning true if
/// the return can be performed without sret-demotion, and false otherwise.
bool CCState::CheckReturn(const SmallVectorImpl<ISD::OutputArg> &Outs,
                          CCAssignFn Fn) {
  // Determine which register each value should be copied into.
  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    MVT VT = Outs[i].VT;
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    if (Fn(i, VT, VT, CCValAssign::Full, ArgFlags, *this))
      return false;
  }
  return true;
}

/// AnalyzeReturn - Analyze the returned values of a return,
/// incorporating info about the result values into this state.
void CCState::AnalyzeReturn(const SmallVectorImpl<ISD::OutputArg> &Outs,
                            CCAssignFn Fn) {
  // Determine which register each value should be copied into.
  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    MVT VT = Outs[i].VT;
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    if (Fn(i, VT, VT, CCValAssign::Full, ArgFlags, *this)) {
#ifndef NDEBUG
      dbgs() << "Return operand #" << i << " has unhandled type "
             << EVT(VT).getEVTString();
#endif
      llvm_unreachable(0);
    }
  }
}

/// AnalyzeCallOperands - Analyze the outgoing arguments to a call,
/// incorporating info about the passed values into this state.
void CCState::AnalyzeCallOperands(const SmallVectorImpl<ISD::OutputArg> &Outs,
                                  CCAssignFn Fn) {
  unsigned NumOps = Outs.size();
  for (unsigned i = 0; i != NumOps; ++i) {
    MVT ArgVT = Outs[i].VT;
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this)) {
#ifndef NDEBUG
      dbgs() << "Call operand #" << i << " has unhandled type "
             << EVT(ArgVT).getEVTString();
#endif
      llvm_unreachable(0);
    }
  }
}

/// AnalyzeCallOperands - Same as above except it takes vectors of types
/// and argument flags.
void CCState::AnalyzeCallOperands(SmallVectorImpl<MVT> &ArgVTs,
                                  SmallVectorImpl<ISD::ArgFlagsTy> &Flags,
                                  CCAssignFn Fn) {
  unsigned NumOps = ArgVTs.size();
  for (unsigned i = 0; i != NumOps; ++i) {
    MVT ArgVT = ArgVTs[i];
    ISD::ArgFlagsTy ArgFlags = Flags[i];
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this)) {
#ifndef NDEBUG
      dbgs() << "Call operand #" << i << " has unhandled type "
             << EVT(ArgVT).getEVTString();
#endif
      llvm_unreachable(0);
    }
  }
}

/// AnalyzeCallResult - Analyze the return values of a call,
/// incorporating info about the passed values into this state.
void CCState::AnalyzeCallResult(const SmallVectorImpl<ISD::InputArg> &Ins,
                                CCAssignFn Fn) {
  for (unsigned i = 0, e = Ins.size(); i != e; ++i) {
    MVT VT = Ins[i].VT;
    ISD::ArgFlagsTy Flags = Ins[i].Flags;
    if (Fn(i, VT, VT, CCValAssign::Full, Flags, *this)) {
#ifndef NDEBUG
      dbgs() << "Call result #" << i << " has unhandled type "
             << EVT(VT).getEVTString() << "\n";
#endif
      llvm_unreachable(0);
    }
  }
}

/// AnalyzeCallResult - Same as above except it's specialized for calls which
/// produce a single value.
void CCState::AnalyzeCallResult(MVT VT, CCAssignFn Fn) {
  if (Fn(0, VT, VT, CCValAssign::Full, ISD::ArgFlagsTy(), *this)) {
#ifndef NDEBUG
    dbgs() << "Call result has unhandled type "
           << EVT(VT).getEVTString();
#endif
    llvm_unreachable(0);
  }
}
