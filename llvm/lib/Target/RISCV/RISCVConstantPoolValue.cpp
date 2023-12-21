//===------- RISCVConstantPoolValue.cpp - RISC-V constantpool value -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RISC-V specific constantpool value class.
//
//===----------------------------------------------------------------------===//

#include "RISCVConstantPoolValue.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

RISCVConstantPoolValue::RISCVConstantPoolValue(LLVMContext &C, RISCVCPKind Kind)
    : MachineConstantPoolValue((Type *)Type::getInt64Ty(C)), Kind(Kind) {}

RISCVConstantPoolValue::RISCVConstantPoolValue(Type *Ty, RISCVCPKind Kind)
    : MachineConstantPoolValue(Ty), Kind(Kind) {}

int RISCVConstantPoolValue::getExistingMachineCPValue(MachineConstantPool *CP,
                                                      Align Alignment) {
  llvm_unreachable("Shouldn't be calling this directly!");
}

RISCVConstantPoolConstant::RISCVConstantPoolConstant(Type *Ty,
                                                     const Constant *GV,
                                                     RISCVCPKind Kind)
    : RISCVConstantPoolValue(Ty, Kind), CVal(GV) {}

RISCVConstantPoolConstant *
RISCVConstantPoolConstant::Create(const GlobalValue *GV) {
  return new RISCVConstantPoolConstant(GV->getType(), GV,
                                       RISCVCPKind::GlobalValue);
}

RISCVConstantPoolConstant *
RISCVConstantPoolConstant::Create(const BlockAddress *BA) {
  return new RISCVConstantPoolConstant(BA->getType(), BA,
                                       RISCVCPKind::BlockAddress);
}

int RISCVConstantPoolConstant::getExistingMachineCPValue(
    MachineConstantPool *CP, Align Alignment) {
  return getExistingMachineCPValueImpl<RISCVConstantPoolConstant>(CP,
                                                                  Alignment);
}

void RISCVConstantPoolConstant::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddPointer(CVal);
}

void RISCVConstantPoolConstant::print(raw_ostream &O) const {
  O << CVal->getName();
}

const GlobalValue *RISCVConstantPoolConstant::getGlobalValue() const {
  return dyn_cast_or_null<GlobalValue>(CVal);
}

const BlockAddress *RISCVConstantPoolConstant::getBlockAddress() const {
  return dyn_cast_or_null<BlockAddress>(CVal);
}

RISCVConstantPoolSymbol::RISCVConstantPoolSymbol(LLVMContext &C, StringRef s)
    : RISCVConstantPoolValue(C, RISCVCPKind::ExtSymbol), S(s) {}

RISCVConstantPoolSymbol *RISCVConstantPoolSymbol::Create(LLVMContext &C,
                                                         StringRef s) {
  return new RISCVConstantPoolSymbol(C, s);
}

int RISCVConstantPoolSymbol::getExistingMachineCPValue(MachineConstantPool *CP,
                                                       Align Alignment) {
  return getExistingMachineCPValueImpl<RISCVConstantPoolSymbol>(CP, Alignment);
}

void RISCVConstantPoolSymbol::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddString(S);
}

void RISCVConstantPoolSymbol::print(raw_ostream &O) const { O << S; }
