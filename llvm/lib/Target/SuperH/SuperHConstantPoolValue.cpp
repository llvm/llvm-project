//===- SuperHConstantPoolValue.cpp - SuperH constantpool value --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SuperH specific constantpool value class.
//
//===----------------------------------------------------------------------===//

#include "SuperHConstantPoolValue.h"

using namespace llvm;



//===----------------------------------------------------------------------===//
// SuperHConstantPoolValue
//===----------------------------------------------------------------------===//

SuperHConstantPoolValue::SuperHConstantPoolValue(Type *Ty, unsigned id,
                                           SHCP::SHCPKind kind,
                                           SHCP::SHCPModifier modifier)
  : MachineConstantPoolValue(Ty), LabelId(id), Kind(kind), Modifier(modifier) {}

SuperHConstantPoolValue::SuperHConstantPoolValue(LLVMContext &C, unsigned id,
                                           SHCP::SHCPKind kind,
                                           SHCP::SHCPModifier modifier)
  : MachineConstantPoolValue((Type*)Type::getInt32Ty(C)),
    LabelId(id), Kind(kind), Modifier(modifier) {}

SuperHConstantPoolValue::~SuperHConstantPoolValue() = default;

StringRef SuperHConstantPoolValue::getModifierText() const {
  switch (Modifier) {
    // FIXME: Are these case sensitive? It'd be nice to lower-case all the
    // strings if that's legal.
  case SHCP::no_modifier:
    return "none";
  case SHCP::GOT_PCREL:
    return "GOT_PCREL";
  case SHCP::GOT_PLTOFF:
    return "gotpltoff";
  case SHCP::DIR:
    return "";
  }
  llvm_unreachable("Unknown modifier!");
}

int SuperHConstantPoolValue::getExistingMachineCPValue(MachineConstantPool *CP,
                                                    Align Alignment) {
  llvm_unreachable("Shouldn't be calling this directly!");
}

void
SuperHConstantPoolValue::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddInteger(LabelId);
}

bool
SuperHConstantPoolValue::hasSameValue(SuperHConstantPoolValue *ACPV) {
  if (ACPV->Kind == Kind &&
      ACPV->Modifier == Modifier &&
      ACPV->LabelId == LabelId) {

    // Two PC relative constpool entries containing the same GV address or
    // external symbols. FIXME: What about blockaddress?
    if (Kind == SHCP::CPValue || Kind == SHCP::CPExtSymbol)
      return true;
  }
  return false;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void SuperHConstantPoolValue::dump() const {
  errs() << "  " << *this;
}
#endif

void SuperHConstantPoolValue::print(raw_ostream &O) const {
  if (Modifier) O << "(" << getModifierText() << ")";
}




//===----------------------------------------------------------------------===//
// SuperHConstantPoolConstant
//===----------------------------------------------------------------------===//

SuperHConstantPoolConstant::SuperHConstantPoolConstant(Type *Ty,
                                                 const Constant *C,
                                                 unsigned ID,
                                                 SHCP::SHCPKind Kind,
                                                 SHCP::SHCPModifier Modifier)
  : SuperHConstantPoolValue(Ty, ID, Kind, Modifier),
    CVal(C) {}

SuperHConstantPoolConstant::SuperHConstantPoolConstant(const Constant *C,
                                                 unsigned ID,
                                                 SHCP::SHCPKind Kind,
                                                 SHCP::SHCPModifier Modifier)
  : SuperHConstantPoolValue((Type*)C->getType(), ID, Kind, Modifier),
    CVal(C) {}

SuperHConstantPoolConstant::SuperHConstantPoolConstant(const GlobalVariable *GV,
                                                 const Constant *C)
    : SuperHConstantPoolValue((Type *)C->getType(), 0, SHCP::CPPromotedGlobal,
                           SHCP::no_modifier), CVal(C) {
  GVars.insert(GV);
}

SuperHConstantPoolConstant *
SuperHConstantPoolConstant::Create(const Constant *C, unsigned ID) {
  return new SuperHConstantPoolConstant(C, ID, SHCP::CPValue,
                                     SHCP::no_modifier);
}

SuperHConstantPoolConstant *
SuperHConstantPoolConstant::Create(const GlobalVariable *GVar,
                                const Constant *Initializer) {
  return new SuperHConstantPoolConstant(GVar, Initializer);
}

SuperHConstantPoolConstant *
SuperHConstantPoolConstant::Create(const GlobalValue *GV,
                                SHCP::SHCPModifier Modifier) {
  return new SuperHConstantPoolConstant((Type*)Type::getInt32Ty(GV->getContext()),
                                     GV, 0, SHCP::CPValue,
                                     Modifier);
}

SuperHConstantPoolConstant *
SuperHConstantPoolConstant::Create(const Constant *C, unsigned ID,
                                SHCP::SHCPKind Kind) {
  return new SuperHConstantPoolConstant(C, ID, Kind,
                                     SHCP::no_modifier);
}

SuperHConstantPoolConstant *
SuperHConstantPoolConstant::Create(const Constant *C, unsigned ID,
                                SHCP::SHCPKind Kind,
                                SHCP::SHCPModifier Modifier) {
  return new SuperHConstantPoolConstant(C, ID, Kind, Modifier);
}

const GlobalValue *SuperHConstantPoolConstant::getGV() const {
  return dyn_cast_or_null<GlobalValue>(CVal);
}

const BlockAddress *SuperHConstantPoolConstant::getBlockAddress() const {
  return dyn_cast_or_null<BlockAddress>(CVal);
}

int SuperHConstantPoolConstant::getExistingMachineCPValue(MachineConstantPool *CP,
                                                       Align Alignment) {
  int index =
    getExistingMachineCPValueImpl<SuperHConstantPoolConstant>(CP, Alignment);
  if (index != -1) {
    auto *CPV = static_cast<SuperHConstantPoolValue*>(
        CP->getConstants()[index].Val.MachineCPVal);
    auto *Constant = cast<SuperHConstantPoolConstant>(CPV);
    Constant->GVars.insert_range(GVars);
  }
  return index;
}

bool SuperHConstantPoolConstant::hasSameValue(SuperHConstantPoolValue *ACPV) {
  const SuperHConstantPoolConstant *ACPC = dyn_cast<SuperHConstantPoolConstant>(ACPV);
  return ACPC && ACPC->CVal == CVal && SuperHConstantPoolValue::hasSameValue(ACPV);
}

void SuperHConstantPoolConstant::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddPointer(CVal);
  for (const auto *GV : GVars)
    ID.AddPointer(GV);
  SuperHConstantPoolValue::addSelectionDAGCSEId(ID);
}

void SuperHConstantPoolConstant::print(raw_ostream &O) const {
  O << CVal->getName();
  SuperHConstantPoolValue::print(O);
}




//===----------------------------------------------------------------------===//
// SuperHConstantPoolSymbol
//===----------------------------------------------------------------------===//

SuperHConstantPoolSymbol::SuperHConstantPoolSymbol(LLVMContext &C, StringRef s,
                                             unsigned id,
                                             SHCP::SHCPModifier Modifier)
    : SuperHConstantPoolValue(C, id, SHCP::CPExtSymbol, Modifier),
      S(std::string(s)) {}

SuperHConstantPoolSymbol *SuperHConstantPoolSymbol::Create(LLVMContext &C,
                                                     StringRef s, unsigned ID) {
  return new SuperHConstantPoolSymbol(C, s, ID, SHCP::no_modifier);
}

int SuperHConstantPoolSymbol::getExistingMachineCPValue(MachineConstantPool *CP,
                                                     Align Alignment) {
  return getExistingMachineCPValueImpl<SuperHConstantPoolSymbol>(CP, Alignment);
}

bool SuperHConstantPoolSymbol::hasSameValue(SuperHConstantPoolValue *SCPV) {
  const SuperHConstantPoolSymbol *ACPS = dyn_cast<SuperHConstantPoolSymbol>(SCPV);
  return ACPS && ACPS->S == S && SuperHConstantPoolValue::hasSameValue(SCPV);
}

void SuperHConstantPoolSymbol::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddString(S);
  SuperHConstantPoolValue::addSelectionDAGCSEId(ID);
}

void SuperHConstantPoolSymbol::print(raw_ostream &O) const {
  O << S;
  SuperHConstantPoolValue::print(O);
}