//===- XtensaConstantPoolValue.cpp - Xtensa constantpool value ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Xtensa specific constantpool value class.
//
//===----------------------------------------------------------------------===//

#include "XtensaConstantPoolValue.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
using namespace llvm;

XtensaConstantPoolValue::XtensaConstantPoolValue(
    Type *Ty, unsigned id, XtensaCP::XtensaCPKind kind,
    XtensaCP::XtensaCPModifier modifier)
    : MachineConstantPoolValue(Ty), LabelId(id), Kind(kind),
      Modifier(modifier) {}

XtensaConstantPoolValue::XtensaConstantPoolValue(
    LLVMContext &C, unsigned id, XtensaCP::XtensaCPKind kind,
    XtensaCP::XtensaCPModifier modifier)
    : MachineConstantPoolValue((Type *)Type::getInt32Ty(C)), LabelId(id),
      Kind(kind), Modifier(modifier) {}

XtensaConstantPoolValue::~XtensaConstantPoolValue() {}

StringRef XtensaConstantPoolValue::getModifierText() const {
  switch (Modifier) {
  case XtensaCP::no_modifier:
    return "";
  case XtensaCP::TPOFF:
    return "@TPOFF";
  }
  report_fatal_error("Unknown modifier!");
}

int XtensaConstantPoolValue::getExistingMachineCPValue(MachineConstantPool *CP,
                                                       Align Alignment) {
  report_fatal_error("Shouldn't be calling this directly!");
}

void XtensaConstantPoolValue::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddInteger(LabelId);
}

bool XtensaConstantPoolValue::hasSameValue(XtensaConstantPoolValue *ACPV) {
  if (ACPV->Kind == Kind) {
    if (ACPV->LabelId == LabelId)
      return true;
  }
  return false;
}

void XtensaConstantPoolValue::dump() const { errs() << "  " << *this; }

void XtensaConstantPoolValue::print(raw_ostream &O) const {}

//===----------------------------------------------------------------------===//
// XtensaConstantPoolConstant
//===----------------------------------------------------------------------===//

XtensaConstantPoolConstant::XtensaConstantPoolConstant(
    const Constant *C, unsigned ID, XtensaCP::XtensaCPKind Kind)
    : XtensaConstantPoolValue((Type *)C->getType(), ID, Kind),
      CVal(C) {}

XtensaConstantPoolConstant *
XtensaConstantPoolConstant::Create(const Constant *C, unsigned ID,
                                   XtensaCP::XtensaCPKind Kind) {
  return new XtensaConstantPoolConstant(C, ID, Kind);
}

const BlockAddress *XtensaConstantPoolConstant::getBlockAddress() const {
  return dyn_cast_or_null<BlockAddress>(CVal);
}

int XtensaConstantPoolConstant::getExistingMachineCPValue(
    MachineConstantPool *CP, Align Alignment) {
  return getExistingMachineCPValueImpl<XtensaConstantPoolConstant>(CP,
                                                                   Alignment);
}

bool XtensaConstantPoolConstant::hasSameValue(XtensaConstantPoolValue *ACPV) {
  const XtensaConstantPoolConstant *ACPC =
      dyn_cast<XtensaConstantPoolConstant>(ACPV);
  return ACPC && ACPC->CVal == CVal &&
         XtensaConstantPoolValue::hasSameValue(ACPV);
}

void XtensaConstantPoolConstant::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddPointer(CVal);
  XtensaConstantPoolValue::addSelectionDAGCSEId(ID);
}

void XtensaConstantPoolConstant::print(raw_ostream &O) const {
  O << CVal->getName();
  XtensaConstantPoolValue::print(O);
}

XtensaConstantPoolSymbol::XtensaConstantPoolSymbol(
    LLVMContext &C, const char *s, unsigned id, bool PrivLinkage,
    XtensaCP::XtensaCPModifier Modifier)
    : XtensaConstantPoolValue(C, id, XtensaCP::CPExtSymbol, Modifier), S(s),
      PrivateLinkage(PrivLinkage) {}

XtensaConstantPoolSymbol *
XtensaConstantPoolSymbol::Create(LLVMContext &C, const char *s, unsigned ID,
                                 bool PrivLinkage,
                                 XtensaCP::XtensaCPModifier Modifier)

{
  return new XtensaConstantPoolSymbol(C, s, ID, PrivLinkage, Modifier);
}

int XtensaConstantPoolSymbol::getExistingMachineCPValue(MachineConstantPool *CP,
                                                        Align Alignment) {
  return getExistingMachineCPValueImpl<XtensaConstantPoolSymbol>(CP, Alignment);
}

bool XtensaConstantPoolSymbol::hasSameValue(XtensaConstantPoolValue *ACPV) {
  const XtensaConstantPoolSymbol *ACPS =
      dyn_cast<XtensaConstantPoolSymbol>(ACPV);
  return ACPS && ACPS->S == S && XtensaConstantPoolValue::hasSameValue(ACPV);
}

void XtensaConstantPoolSymbol::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddString(S);
  XtensaConstantPoolValue::addSelectionDAGCSEId(ID);
}

void XtensaConstantPoolSymbol::print(raw_ostream &O) const {
  O << S;
  XtensaConstantPoolValue::print(O);
}

XtensaConstantPoolMBB::XtensaConstantPoolMBB(LLVMContext &C,
                                             const MachineBasicBlock *mbb,
                                             unsigned id)
    : XtensaConstantPoolValue(C, 0, XtensaCP::CPMachineBasicBlock),
      MBB(mbb) {}

XtensaConstantPoolMBB *
XtensaConstantPoolMBB::Create(LLVMContext &C, const MachineBasicBlock *mbb,
                              unsigned idx) {
  return new XtensaConstantPoolMBB(C, mbb, idx);
}

int XtensaConstantPoolMBB::getExistingMachineCPValue(MachineConstantPool *CP,
                                                     Align Alignment) {
  return getExistingMachineCPValueImpl<XtensaConstantPoolMBB>(CP, Alignment);
}

bool XtensaConstantPoolMBB::hasSameValue(XtensaConstantPoolValue *ACPV) {
  const XtensaConstantPoolMBB *ACPMBB = dyn_cast<XtensaConstantPoolMBB>(ACPV);
  return ACPMBB && ACPMBB->MBB == MBB &&
         XtensaConstantPoolValue::hasSameValue(ACPV);
}

void XtensaConstantPoolMBB::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddPointer(MBB);
  XtensaConstantPoolValue::addSelectionDAGCSEId(ID);
}

void XtensaConstantPoolMBB::print(raw_ostream &O) const {
  O << "BB#" << MBB->getNumber();
  XtensaConstantPoolValue::print(O);
}

XtensaConstantPoolJumpTable::XtensaConstantPoolJumpTable(LLVMContext &C,
                                                         unsigned idx)
    : XtensaConstantPoolValue(C, 0, XtensaCP::CPJumpTable), IDX(idx) {}

XtensaConstantPoolJumpTable *XtensaConstantPoolJumpTable::Create(LLVMContext &C,
                                                                 unsigned idx) {
  return new XtensaConstantPoolJumpTable(C, idx);
}

int XtensaConstantPoolJumpTable::getExistingMachineCPValue(
    MachineConstantPool *CP, Align Alignment) {
  return getExistingMachineCPValueImpl<XtensaConstantPoolJumpTable>(CP,
                                                                    Alignment);
}

bool XtensaConstantPoolJumpTable::hasSameValue(XtensaConstantPoolValue *ACPV) {
  const XtensaConstantPoolJumpTable *ACPJT =
      dyn_cast<XtensaConstantPoolJumpTable>(ACPV);
  return ACPJT && ACPJT->IDX == IDX &&
         XtensaConstantPoolValue::hasSameValue(ACPV);
}

void XtensaConstantPoolJumpTable::addSelectionDAGCSEId(FoldingSetNodeID &ID) {}

void XtensaConstantPoolJumpTable::print(raw_ostream &O) const {
  O << "JT" << IDX;
  XtensaConstantPoolValue::print(O);
}

