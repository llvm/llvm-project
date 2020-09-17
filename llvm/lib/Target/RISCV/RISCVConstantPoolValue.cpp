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

RISCVConstantPoolValue::RISCVConstantPoolValue(
    LLVMContext &C, RISCVCP::RISCVCPKind Kind,
    RISCVCP::RISCVCPModifier Modifier)
    : MachineConstantPoolValue((Type *)Type::getInt64Ty(C)), Kind(Kind),
      Modifier(Modifier) {}

RISCVConstantPoolValue::RISCVConstantPoolValue(
    Type *Ty, RISCVCP::RISCVCPKind Kind, RISCVCP::RISCVCPModifier Modifier)
    : MachineConstantPoolValue(Ty), Kind(Kind), Modifier(Modifier) {}

int RISCVConstantPoolValue::getExistingMachineCPValue(MachineConstantPool *CP,
                                                      Align Alignment) {
  llvm_unreachable("Shouldn't be calling this directly!");
}

StringRef RISCVConstantPoolValue::getModifierText() const {
  switch (Modifier) {
  case RISCVCP::None:
    return "";
  }
  llvm_unreachable("Unknown modifier!");
}

void RISCVConstantPoolValue::print(raw_ostream &O) const {
  if (hasModifier())
    O << "@" << getModifierText();
}

RISCVConstantPoolConstant::RISCVConstantPoolConstant(Type *Ty,
                                                     const Constant *GV,
                                                     RISCVCP::RISCVCPKind Kind)
    : RISCVConstantPoolValue(Ty, Kind, RISCVCP::None), CVal(GV) {}

RISCVConstantPoolConstant *
RISCVConstantPoolConstant::Create(const GlobalValue *GV,
                                  RISCVCP::RISCVCPKind Kind) {
  return new RISCVConstantPoolConstant(GV->getType(), GV, Kind);
}

RISCVConstantPoolConstant *
RISCVConstantPoolConstant::Create(const Constant *C,
                                  RISCVCP::RISCVCPKind Kind) {
  return new RISCVConstantPoolConstant(C->getType(), C, Kind);
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
  RISCVConstantPoolValue::print(O);
}

const GlobalValue *RISCVConstantPoolConstant::getGlobalValue() const {
  return dyn_cast_or_null<GlobalValue>(CVal);
}

const BlockAddress *RISCVConstantPoolConstant::getBlockAddress() const {
  return dyn_cast_or_null<BlockAddress>(CVal);
}

RISCVConstantPoolSymbol::RISCVConstantPoolSymbol(
    LLVMContext &C, StringRef s, RISCVCP::RISCVCPModifier Modifier)
    : RISCVConstantPoolValue(C, RISCVCP::ExtSymbol, Modifier), S(s) {}

RISCVConstantPoolSymbol *
RISCVConstantPoolSymbol::Create(LLVMContext &C, StringRef s,
                                RISCVCP::RISCVCPModifier Modifier) {
  return new RISCVConstantPoolSymbol(C, s, Modifier);
}

int RISCVConstantPoolSymbol::getExistingMachineCPValue(MachineConstantPool *CP,
                                                       Align Alignment) {
  return getExistingMachineCPValueImpl<RISCVConstantPoolSymbol>(CP, Alignment);
}

void RISCVConstantPoolSymbol::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddString(S);
}

void RISCVConstantPoolSymbol::print(raw_ostream &O) const {
  O << S;
  RISCVConstantPoolValue::print(O);
}
