//===-- llvm/CallingConvLower.cpp - Calling Conventions -------------------===//
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
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

CCState::CCState(unsigned CC, bool isVarArg, const TargetMachine &tm,
                 SmallVector<CCValAssign, 16> &locs)
  : CallingConv(CC), IsVarArg(isVarArg), TM(tm),
    TRI(*TM.getRegisterInfo()), Locs(locs) {
  // No stack is used.
  StackOffset = 0;
  
  UsedRegs.resize((TRI.getNumRegs()+31)/32);
}

// HandleByVal - Allocate a stack slot large enough to pass an argument by
// value. The size and alignment information of the argument is encoded in its
// parameter attribute.
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
  unsigned Offset = AllocateStack(Size, Align);

  addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
}

/// MarkAllocated - Mark a register and all of its aliases as allocated.
void CCState::MarkAllocated(unsigned Reg) {
  UsedRegs[Reg/32] |= 1 << (Reg&31);
  
  if (const unsigned *RegAliases = TRI.getAliasSet(Reg))
    for (; (Reg = *RegAliases); ++RegAliases)
      UsedRegs[Reg/32] |= 1 << (Reg&31);
}

/// AnalyzeFormalArguments - Analyze an ISD::FORMAL_ARGUMENTS node,
/// incorporating info about the formals into this state.
void CCState::AnalyzeFormalArguments(SDNode *TheArgs, CCAssignFn Fn) {
  unsigned NumArgs = TheArgs->getNumValues()-1;
  
  for (unsigned i = 0; i != NumArgs; ++i) {
    MVT ArgVT = TheArgs->getValueType(i);
    ISD::ArgFlagsTy ArgFlags =
      cast<ARG_FLAGSSDNode>(TheArgs->getOperand(3+i))->getArgFlags();
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this)) {
      cerr << "Formal argument #" << i << " has unhandled type "
           << ArgVT.getMVTString() << "\n";
      abort();
    }
  }
}

/// AnalyzeReturn - Analyze the returned values of an ISD::RET node,
/// incorporating info about the result values into this state.
void CCState::AnalyzeReturn(SDNode *TheRet, CCAssignFn Fn) {
  // Determine which register each value should be copied into.
  for (unsigned i = 0, e = TheRet->getNumOperands() / 2; i != e; ++i) {
    MVT VT = TheRet->getOperand(i*2+1).getValueType();
    ISD::ArgFlagsTy ArgFlags =
      cast<ARG_FLAGSSDNode>(TheRet->getOperand(i*2+2))->getArgFlags();
    if (Fn(i, VT, VT, CCValAssign::Full, ArgFlags, *this)){
      cerr << "Return operand #" << i << " has unhandled type "
           << VT.getMVTString() << "\n";
      abort();
    }
  }
}


/// AnalyzeCallOperands - Analyze an ISD::CALL node, incorporating info
/// about the passed values into this state.
void CCState::AnalyzeCallOperands(CallSDNode *TheCall, CCAssignFn Fn) {
  unsigned NumOps = TheCall->getNumArgs();
  for (unsigned i = 0; i != NumOps; ++i) {
    MVT ArgVT = TheCall->getArg(i).getValueType();
    ISD::ArgFlagsTy ArgFlags = TheCall->getArgFlags(i);
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this)) {
      cerr << "Call operand #" << i << " has unhandled type "
           << ArgVT.getMVTString() << "\n";
      abort();
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
      cerr << "Call operand #" << i << " has unhandled type "
           << ArgVT.getMVTString() << "\n";
      abort();
    }
  }
}

/// AnalyzeCallResult - Analyze the return values of an ISD::CALL node,
/// incorporating info about the passed values into this state.
void CCState::AnalyzeCallResult(CallSDNode *TheCall, CCAssignFn Fn) {
  for (unsigned i = 0, e = TheCall->getNumRetVals(); i != e; ++i) {
    MVT VT = TheCall->getRetValType(i);
    ISD::ArgFlagsTy Flags = ISD::ArgFlagsTy();
    if (TheCall->isInreg())
      Flags.setInReg();
    if (Fn(i, VT, VT, CCValAssign::Full, Flags, *this)) {
      cerr << "Call result #" << i << " has unhandled type "
           << VT.getMVTString() << "\n";
      abort();
    }
  }
}

/// AnalyzeCallResult - Same as above except it's specialized for calls which
/// produce a single value.
void CCState::AnalyzeCallResult(MVT VT, CCAssignFn Fn) {
  if (Fn(0, VT, VT, CCValAssign::Full, ISD::ArgFlagsTy(), *this)) {
    cerr << "Call result has unhandled type "
         << VT.getMVTString() << "\n";
    abort();
  }
}
