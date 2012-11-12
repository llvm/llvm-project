//===-- ARMConstantPoolValue.cpp - ARM constantpool value -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ARM specific constantpool value class.
//
//===----------------------------------------------------------------------===//

#include "ARMConstantPoolValue.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Constant.h"
#include "llvm/Constants.h"
#include "llvm/GlobalValue.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
using namespace llvm;

//===----------------------------------------------------------------------===//
// ARMConstantPoolValue
//===----------------------------------------------------------------------===//

ARMConstantPoolValue::ARMConstantPoolValue(Type *Ty, unsigned id,
                                           ARMCP::ARMCPKind kind,
                                           unsigned char PCAdj,
                                           ARMCP::ARMCPModifier modifier,
                                           bool addCurrentAddress)
  : MachineConstantPoolValue(Ty), LabelId(id), Kind(kind),
    PCAdjust(PCAdj), Modifier(modifier),
    AddCurrentAddress(addCurrentAddress) {}

ARMConstantPoolValue::ARMConstantPoolValue(LLVMContext &C, unsigned id,
                                           ARMCP::ARMCPKind kind,
                                           unsigned char PCAdj,
                                           ARMCP::ARMCPModifier modifier,
                                           bool addCurrentAddress)
  : MachineConstantPoolValue((Type*)Type::getInt32Ty(C)),
    LabelId(id), Kind(kind), PCAdjust(PCAdj), Modifier(modifier),
    AddCurrentAddress(addCurrentAddress) {}

ARMConstantPoolValue::~ARMConstantPoolValue() {}

const char *ARMConstantPoolValue::getModifierText() const {
  switch (Modifier) {
    // FIXME: Are these case sensitive? It'd be nice to lower-case all the
    // strings if that's legal.
  case ARMCP::no_modifier: return "none";
  case ARMCP::TLSGD:       return "tlsgd";
  case ARMCP::GOT:         return "GOT";
  case ARMCP::GOTOFF:      return "GOTOFF";
  case ARMCP::GOTTPOFF:    return "gottpoff";
  case ARMCP::TPOFF:       return "tpoff";
  }
  llvm_unreachable("Unknown modifier!");
}

int ARMConstantPoolValue::getExistingMachineCPValue(MachineConstantPool *CP,
                                                    unsigned Alignment) {
  llvm_unreachable("Shouldn't be calling this directly!");
}

void
ARMConstantPoolValue::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddInteger(LabelId);
  ID.AddInteger(PCAdjust);
}

bool
ARMConstantPoolValue::hasSameValue(ARMConstantPoolValue *ACPV) {
  if (ACPV->Kind == Kind &&
      ACPV->PCAdjust == PCAdjust &&
      ACPV->Modifier == Modifier) {
    if (ACPV->LabelId == LabelId)
      return true;
    // Two PC relative constpool entries containing the same GV address or
    // external symbols. FIXME: What about blockaddress?
    if (Kind == ARMCP::CPValue || Kind == ARMCP::CPExtSymbol)
      return true;
  }
  return false;
}

void ARMConstantPoolValue::dump() const {
  errs() << "  " << *this;
}

void ARMConstantPoolValue::print(raw_ostream &O) const {
  if (Modifier) O << "(" << getModifierText() << ")";
  if (PCAdjust != 0) {
    O << "-(LPC" << LabelId << "+" << (unsigned)PCAdjust;
    if (AddCurrentAddress) O << "-.";
    O << ")";
  }
}

//===----------------------------------------------------------------------===//
// ARMConstantPoolConstant
//===----------------------------------------------------------------------===//

ARMConstantPoolConstant::ARMConstantPoolConstant(Type *Ty,
                                                 const Constant *C,
                                                 unsigned ID,
                                                 ARMCP::ARMCPKind Kind,
                                                 unsigned char PCAdj,
                                                 ARMCP::ARMCPModifier Modifier,
                                                 bool AddCurrentAddress)
  : ARMConstantPoolValue(Ty, ID, Kind, PCAdj, Modifier, AddCurrentAddress),
    CVal(C) {}

ARMConstantPoolConstant::ARMConstantPoolConstant(const Constant *C,
                                                 unsigned ID,
                                                 ARMCP::ARMCPKind Kind,
                                                 unsigned char PCAdj,
                                                 ARMCP::ARMCPModifier Modifier,
                                                 bool AddCurrentAddress)
  : ARMConstantPoolValue((Type*)C->getType(), ID, Kind, PCAdj, Modifier,
                         AddCurrentAddress),
    CVal(C) {}

ARMConstantPoolConstant *
ARMConstantPoolConstant::Create(const Constant *C, unsigned ID) {
  return new ARMConstantPoolConstant(C, ID, ARMCP::CPValue, 0,
                                     ARMCP::no_modifier, false);
}

ARMConstantPoolConstant *
ARMConstantPoolConstant::Create(const GlobalValue *GV,
                                ARMCP::ARMCPModifier Modifier) {
  return new ARMConstantPoolConstant((Type*)Type::getInt32Ty(GV->getContext()),
                                     GV, 0, ARMCP::CPValue, 0,
                                     Modifier, false);
}

ARMConstantPoolConstant *
ARMConstantPoolConstant::Create(const Constant *C, unsigned ID,
                                ARMCP::ARMCPKind Kind, unsigned char PCAdj) {
  return new ARMConstantPoolConstant(C, ID, Kind, PCAdj,
                                     ARMCP::no_modifier, false);
}

ARMConstantPoolConstant *
ARMConstantPoolConstant::Create(const Constant *C, unsigned ID,
                                ARMCP::ARMCPKind Kind, unsigned char PCAdj,
                                ARMCP::ARMCPModifier Modifier,
                                bool AddCurrentAddress) {
  return new ARMConstantPoolConstant(C, ID, Kind, PCAdj, Modifier,
                                     AddCurrentAddress);
}

const GlobalValue *ARMConstantPoolConstant::getGV() const {
  return dyn_cast_or_null<GlobalValue>(CVal);
}

const BlockAddress *ARMConstantPoolConstant::getBlockAddress() const {
  return dyn_cast_or_null<BlockAddress>(CVal);
}

int ARMConstantPoolConstant::getExistingMachineCPValue(MachineConstantPool *CP,
                                                       unsigned Alignment) {
  unsigned AlignMask = Alignment - 1;
  const std::vector<MachineConstantPoolEntry> Constants = CP->getConstants();
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    if (Constants[i].isMachineConstantPoolEntry() &&
        (Constants[i].getAlignment() & AlignMask) == 0) {
      ARMConstantPoolValue *CPV =
        (ARMConstantPoolValue *)Constants[i].Val.MachineCPVal;
      ARMConstantPoolConstant *APC = dyn_cast<ARMConstantPoolConstant>(CPV);
      if (!APC) continue;
      if (APC->CVal == CVal && equals(APC))
        return i;
    }
  }

  return -1;
}

bool ARMConstantPoolConstant::hasSameValue(ARMConstantPoolValue *ACPV) {
  const ARMConstantPoolConstant *ACPC = dyn_cast<ARMConstantPoolConstant>(ACPV);
  return ACPC && ACPC->CVal == CVal && ARMConstantPoolValue::hasSameValue(ACPV);
}

void ARMConstantPoolConstant::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddPointer(CVal);
  ARMConstantPoolValue::addSelectionDAGCSEId(ID);
}

void ARMConstantPoolConstant::print(raw_ostream &O) const {
  O << CVal->getName();
  ARMConstantPoolValue::print(O);
}

//===----------------------------------------------------------------------===//
// ARMConstantPoolSymbol
//===----------------------------------------------------------------------===//

ARMConstantPoolSymbol::ARMConstantPoolSymbol(LLVMContext &C, const char *s,
                                             unsigned id,
                                             unsigned char PCAdj,
                                             ARMCP::ARMCPModifier Modifier,
                                             bool AddCurrentAddress)
  : ARMConstantPoolValue(C, id, ARMCP::CPExtSymbol, PCAdj, Modifier,
                         AddCurrentAddress),
    S(strdup(s)) {}

ARMConstantPoolSymbol::~ARMConstantPoolSymbol() {
  free((void*)S);
}

ARMConstantPoolSymbol *
ARMConstantPoolSymbol::Create(LLVMContext &C, const char *s,
                              unsigned ID, unsigned char PCAdj) {
  return new ARMConstantPoolSymbol(C, s, ID, PCAdj, ARMCP::no_modifier, false);
}

static bool CPV_streq(const char *S1, const char *S2) {
  if (S1 == S2)
    return true;
  if (S1 && S2 && strcmp(S1, S2) == 0)
    return true;
  return false;
}

int ARMConstantPoolSymbol::getExistingMachineCPValue(MachineConstantPool *CP,
                                                     unsigned Alignment) {
  unsigned AlignMask = Alignment - 1;
  const std::vector<MachineConstantPoolEntry> Constants = CP->getConstants();
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    if (Constants[i].isMachineConstantPoolEntry() &&
        (Constants[i].getAlignment() & AlignMask) == 0) {
      ARMConstantPoolValue *CPV =
        (ARMConstantPoolValue *)Constants[i].Val.MachineCPVal;
      ARMConstantPoolSymbol *APS = dyn_cast<ARMConstantPoolSymbol>(CPV);
      if (!APS) continue;

      if (CPV_streq(APS->S, S) && equals(APS))
        return i;
    }
  }

  return -1;
}

bool ARMConstantPoolSymbol::hasSameValue(ARMConstantPoolValue *ACPV) {
  const ARMConstantPoolSymbol *ACPS = dyn_cast<ARMConstantPoolSymbol>(ACPV);
  return ACPS && CPV_streq(ACPS->S, S) &&
    ARMConstantPoolValue::hasSameValue(ACPV);
}

void ARMConstantPoolSymbol::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddPointer(S);
  ARMConstantPoolValue::addSelectionDAGCSEId(ID);
}

void ARMConstantPoolSymbol::print(raw_ostream &O) const {
  O << S;
  ARMConstantPoolValue::print(O);
}

//===----------------------------------------------------------------------===//
// ARMConstantPoolMBB
//===----------------------------------------------------------------------===//

ARMConstantPoolMBB::ARMConstantPoolMBB(LLVMContext &C,
                                       const MachineBasicBlock *mbb,
                                       unsigned id, unsigned char PCAdj,
                                       ARMCP::ARMCPModifier Modifier,
                                       bool AddCurrentAddress)
  : ARMConstantPoolValue(C, id, ARMCP::CPMachineBasicBlock, PCAdj,
                         Modifier, AddCurrentAddress),
    MBB(mbb) {}

ARMConstantPoolMBB *ARMConstantPoolMBB::Create(LLVMContext &C,
                                               const MachineBasicBlock *mbb,
                                               unsigned ID,
                                               unsigned char PCAdj) {
  return new ARMConstantPoolMBB(C, mbb, ID, PCAdj, ARMCP::no_modifier, false);
}

int ARMConstantPoolMBB::getExistingMachineCPValue(MachineConstantPool *CP,
                                                  unsigned Alignment) {
  unsigned AlignMask = Alignment - 1;
  const std::vector<MachineConstantPoolEntry> Constants = CP->getConstants();
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    if (Constants[i].isMachineConstantPoolEntry() &&
        (Constants[i].getAlignment() & AlignMask) == 0) {
      ARMConstantPoolValue *CPV =
        (ARMConstantPoolValue *)Constants[i].Val.MachineCPVal;
      ARMConstantPoolMBB *APMBB = dyn_cast<ARMConstantPoolMBB>(CPV);
      if (!APMBB) continue;

      if (APMBB->MBB == MBB && equals(APMBB))
        return i;
    }
  }

  return -1;
}

bool ARMConstantPoolMBB::hasSameValue(ARMConstantPoolValue *ACPV) {
  const ARMConstantPoolMBB *ACPMBB = dyn_cast<ARMConstantPoolMBB>(ACPV);
  return ACPMBB && ACPMBB->MBB == MBB &&
    ARMConstantPoolValue::hasSameValue(ACPV);
}

void ARMConstantPoolMBB::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddPointer(MBB);
  ARMConstantPoolValue::addSelectionDAGCSEId(ID);
}

void ARMConstantPoolMBB::print(raw_ostream &O) const {
  O << "BB#" << MBB->getNumber();
  ARMConstantPoolValue::print(O);
}
