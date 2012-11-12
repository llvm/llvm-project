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
// functions.
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
#include "llvm/Metadata.h"
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
/// DbgDeclareInst - This represents the llvm.dbg.declare instruction.
///

Value *DbgDeclareInst::getAddress() const {
  if (MDNode* MD = cast_or_null<MDNode>(getArgOperand(0)))
    return MD->getOperand(0);
  else
    return NULL;
}

//===----------------------------------------------------------------------===//
/// DbgValueInst - This represents the llvm.dbg.value instruction.
///

const Value *DbgValueInst::getValue() const {
  return cast<MDNode>(getArgOperand(0))->getOperand(0);
}

Value *DbgValueInst::getValue() {
  return cast<MDNode>(getArgOperand(0))->getOperand(0);
}
