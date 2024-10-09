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

RISCVConstantPoolValue::RISCVConstantPoolValue(Type *Ty, const GlobalValue *GV)
    : MachineConstantPoolValue(Ty), GV(GV), Kind(RISCVCPKind::GlobalValue) {}

RISCVConstantPoolValue::RISCVConstantPoolValue(LLVMContext &C, StringRef S)
    : MachineConstantPoolValue(Type::getInt64Ty(C)), S(S),
      Kind(RISCVCPKind::ExtSymbol) {}

RISCVConstantPoolValue *RISCVConstantPoolValue::Create(const GlobalValue *GV) {
  return new RISCVConstantPoolValue(GV->getType(), GV);
}

RISCVConstantPoolValue *RISCVConstantPoolValue::Create(LLVMContext &C,
                                                       StringRef S) {
  return new RISCVConstantPoolValue(C, S);
}

int RISCVConstantPoolValue::getExistingMachineCPValue(MachineConstantPool *CP,
                                                      Align Alignment) {
  const std::vector<MachineConstantPoolEntry> &Constants = CP->getConstants();
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    if (Constants[i].isMachineConstantPoolEntry() &&
        Constants[i].getAlign() >= Alignment) {
      auto *CPV =
          static_cast<RISCVConstantPoolValue *>(Constants[i].Val.MachineCPVal);
      if (equals(CPV))
        return i;
    }
  }

  return -1;
}

void RISCVConstantPoolValue::addSelectionDAGCSEId(FoldingSetNodeID &ID) {
  if (isGlobalValue())
    ID.AddPointer(GV);
  else {
    assert(isExtSymbol() && "unrecognized constant pool type");
    ID.AddString(S);
  }
}

void RISCVConstantPoolValue::print(raw_ostream &O) const {
  if (isGlobalValue())
    O << GV->getName();
  else {
    assert(isExtSymbol() && "unrecognized constant pool type");
    O << S;
  }
}

bool RISCVConstantPoolValue::equals(const RISCVConstantPoolValue *A) const {
  if (isGlobalValue() && A->isGlobalValue())
    return GV == A->GV;
  if (isExtSymbol() && A->isExtSymbol())
    return S == A->S;

  return false;
}
