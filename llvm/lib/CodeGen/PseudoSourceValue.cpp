//===-- llvm/CodeGen/PseudoSourceValue.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the PseudoSourceValue class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static const char *const PSVNames[] = {
    "Stack", "GOT", "JumpTable", "ConstantPool", "FixedStack",
    "GlobalValueCallEntry", "ExternalSymbolCallEntry"};

PseudoSourceValue::PseudoSourceValue(unsigned Kind, const TargetInstrInfo &TII)
    : Kind(Kind) {
  AddressSpace = TII.getAddressSpaceForPseudoSourceKind(Kind);
}

PseudoSourceValue::~PseudoSourceValue() = default;

void PseudoSourceValue::printCustom(raw_ostream &O) const {
  if (Kind < TargetCustom)
    O << PSVNames[Kind];
  else
    O << "TargetCustom" << Kind;
}

bool PseudoSourceValue::isConstant(const MachineFrameInfo *) const {
  if (isStack())
    return false;
  if (isGOT() || isConstantPool() || isJumpTable())
    return true;
  llvm_unreachable("Unknown PseudoSourceValue!");
}

bool PseudoSourceValue::isAliased(const MachineFrameInfo *) const {
  if (isStack() || isGOT() || isConstantPool() || isJumpTable())
    return false;
  llvm_unreachable("Unknown PseudoSourceValue!");
}

bool PseudoSourceValue::mayAlias(const MachineFrameInfo *) const {
  return !(isGOT() || isConstantPool() || isJumpTable());
}

bool FixedStackPseudoSourceValue::isConstant(
    const MachineFrameInfo *MFI) const {
  return MFI && MFI->isImmutableObjectIndex(FI);
}

bool FixedStackPseudoSourceValue::isAliased(const MachineFrameInfo *MFI) const {
  if (!MFI)
    return true;
  return MFI->isAliasedObjectIndex(FI);
}

bool FixedStackPseudoSourceValue::mayAlias(const MachineFrameInfo *MFI) const {
  if (!MFI)
    return true;
  // Spill slots will not alias any LLVM IR value.
  return !MFI->isSpillSlotObjectIndex(FI);
}

void FixedStackPseudoSourceValue::printCustom(raw_ostream &OS) const {
  OS << "FixedStack" << FI;
}

CallEntryPseudoSourceValue::CallEntryPseudoSourceValue(
    unsigned Kind, const TargetInstrInfo &TII)
    : PseudoSourceValue(Kind, TII) {}

bool CallEntryPseudoSourceValue::isConstant(const MachineFrameInfo *) const {
  return false;
}

bool CallEntryPseudoSourceValue::isAliased(const MachineFrameInfo *) const {
  return false;
}

bool CallEntryPseudoSourceValue::mayAlias(const MachineFrameInfo *) const {
  return false;
}

GlobalValuePseudoSourceValue::GlobalValuePseudoSourceValue(
    const GlobalValue *GV,
    const TargetInstrInfo &TII)
    : CallEntryPseudoSourceValue(GlobalValueCallEntry, TII), GV(GV) {}
ExternalSymbolPseudoSourceValue::ExternalSymbolPseudoSourceValue(
    const char *ES, const TargetInstrInfo &TII)
    : CallEntryPseudoSourceValue(ExternalSymbolCallEntry, TII), ES(ES) {}

PseudoSourceValueManager::PseudoSourceValueManager(
    const TargetInstrInfo &TIInfo)
    : TII(TIInfo),
      StackPSV(PseudoSourceValue::Stack, TII),
      GOTPSV(PseudoSourceValue::GOT, TII),
      JumpTablePSV(PseudoSourceValue::JumpTable, TII),
      ConstantPoolPSV(PseudoSourceValue::ConstantPool, TII) {}

const PseudoSourceValue *PseudoSourceValueManager::getStack() {
  return &StackPSV;
}

const PseudoSourceValue *PseudoSourceValueManager::getGOT() { return &GOTPSV; }

const PseudoSourceValue *PseudoSourceValueManager::getConstantPool() {
  return &ConstantPoolPSV;
}

const PseudoSourceValue *PseudoSourceValueManager::getJumpTable() {
  return &JumpTablePSV;
}

const PseudoSourceValue *
PseudoSourceValueManager::getFixedStack(int FI) {
  std::unique_ptr<FixedStackPseudoSourceValue> &V = FSValues[FI];
  if (!V)
    V = std::make_unique<FixedStackPseudoSourceValue>(FI, TII);
  return V.get();
}

const PseudoSourceValue *
PseudoSourceValueManager::getGlobalValueCallEntry(const GlobalValue *GV) {
  std::unique_ptr<const GlobalValuePseudoSourceValue> &E =
      GlobalCallEntries[GV];
  if (!E)
    E = std::make_unique<GlobalValuePseudoSourceValue>(GV, TII);
  return E.get();
}

const PseudoSourceValue *
PseudoSourceValueManager::getExternalSymbolCallEntry(const char *ES) {
  std::unique_ptr<const ExternalSymbolPseudoSourceValue> &E =
      ExternalCallEntries[ES];
  if (!E)
    E = std::make_unique<ExternalSymbolPseudoSourceValue>(ES, TII);
  return E.get();
}
