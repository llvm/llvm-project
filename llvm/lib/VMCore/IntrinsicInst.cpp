//===-- InstrinsicInst.cpp - Intrinsic Instruction Wrappers -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements methods that make it really easy to deal with intrinsic
// functions with the isa/dyncast family of functions.  In particular, this
// allows you to do things like:
//
//     if (DbgStopPointInst *SPI = dyn_cast<DbgStopPointInst>(Inst))
//        ... SPI->getFileName() ... SPI->getDirectory() ...
//
// All intrinsic function calls are instances of the call instruction, so these
// are all subclasses of the CallInst class.  Note that none of these classes
// has state or virtual methods, which is an important part of this gross/neat
// hack working.
// 
// In some cases, arguments to intrinsics need to be generic and are defined as
// type pointer to empty struct { }*.  To access the real item of interest the
// cast instruction needs to be stripped away. 
//
//===----------------------------------------------------------------------===//

#include "llvm/IntrinsicInst.h"
#include "llvm/Constants.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
/// DbgInfoIntrinsic - This is the common base class for debug info intrinsics
///

static Value *CastOperand(Value *C) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C))
    if (CE->isCast())
      return CE->getOperand(0);
  return NULL;
}

Value *DbgInfoIntrinsic::StripCast(Value *C) {
  if (Value *CO = CastOperand(C)) {
    C = StripCast(CO);
  } else if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C)) {
    if (GV->hasInitializer())
      if (Value *CO = CastOperand(GV->getInitializer()))
        C = StripCast(CO);
  }
  return dyn_cast<GlobalVariable>(C);
}

//===----------------------------------------------------------------------===//
/// DbgStopPointInst - This represents the llvm.dbg.stoppoint instruction.
///

Value *DbgStopPointInst::getFileName() const {
  // Once the operand indices are verified, update this assert
  assert(LLVMDebugVersion == (7 << 16) && "Verify operand indices");
  GlobalVariable *GV = cast<GlobalVariable>(getContext());
  if (!GV->hasInitializer()) return NULL;
  ConstantStruct *CS = cast<ConstantStruct>(GV->getInitializer());
  return CS->getOperand(3);
}

Value *DbgStopPointInst::getDirectory() const {
  // Once the operand indices are verified, update this assert
  assert(LLVMDebugVersion == (7 << 16) && "Verify operand indices");
  GlobalVariable *GV = cast<GlobalVariable>(getContext());
  if (!GV->hasInitializer()) return NULL;
  ConstantStruct *CS = cast<ConstantStruct>(GV->getInitializer());
  return CS->getOperand(4);
}
