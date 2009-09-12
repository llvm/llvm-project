//===- ARMConstantPoolValue.cpp - ARM constantpool value --------*- C++ -*-===//
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
#include "llvm/GlobalValue.h"
#include "llvm/Type.h"
#include "llvm/Support/Streams.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
using namespace llvm;

ARMConstantPoolValue::ARMConstantPoolValue(GlobalValue *gv, unsigned id,
                                           unsigned char PCAdj,
                                           const char *Modif,
                                           bool AddCA)
  : MachineConstantPoolValue((const Type*)gv->getType()),
    GV(gv), S(NULL), LabelId(id), PCAdjust(PCAdj),
    Modifier(Modif), AddCurrentAddress(AddCA) {}

ARMConstantPoolValue::ARMConstantPoolValue(LLVMContext &C,
                                           const char *s, unsigned id,
                                           unsigned char PCAdj,
                                           const char *Modif,
                                           bool AddCA)
  : MachineConstantPoolValue((const Type*)Type::getInt32Ty(C)),
    GV(NULL), S(strdup(s)), LabelId(id), PCAdjust(PCAdj),
    Modifier(Modif), AddCurrentAddress(AddCA) {}

ARMConstantPoolValue::ARMConstantPoolValue(GlobalValue *gv, const char *Modif)
  : MachineConstantPoolValue((const Type*)Type::getInt32Ty(gv->getContext())),
    GV(gv), S(NULL), LabelId(0), PCAdjust(0),
    Modifier(Modif) {}

int ARMConstantPoolValue::getExistingMachineCPValue(MachineConstantPool *CP,
                                                    unsigned Alignment) {
  unsigned AlignMask = Alignment - 1;
  const std::vector<MachineConstantPoolEntry> Constants = CP->getConstants();
  for (unsigned i = 0, e = Constants.size(); i != e; ++i) {
    if (Constants[i].isMachineConstantPoolEntry() &&
        (Constants[i].getAlignment() & AlignMask) == 0) {
      ARMConstantPoolValue *CPV =
        (ARMConstantPoolValue *)Constants[i].Val.MachineCPVal;
      if (CPV->GV == GV &&
          CPV->S == S &&
          CPV->LabelId == LabelId &&
          CPV->PCAdjust == PCAdjust)
        return i;
    }
  }

  return -1;
}

ARMConstantPoolValue::~ARMConstantPoolValue() {
  free((void*)S);
}

void
ARMConstantPoolValue::AddSelectionDAGCSEId(FoldingSetNodeID &ID) {
  ID.AddPointer(GV);
  ID.AddPointer(S);
  ID.AddInteger(LabelId);
  ID.AddInteger(PCAdjust);
}

void ARMConstantPoolValue::dump() const {
  cerr << "  " << *this;
}

void ARMConstantPoolValue::print(std::ostream &O) const {
  raw_os_ostream RawOS(O);
  print(RawOS);
}

void ARMConstantPoolValue::print(raw_ostream &O) const {
  if (GV)
    O << GV->getName();
  else
    O << S;
  if (Modifier) O << "(" << Modifier << ")";
  if (PCAdjust != 0) {
    O << "-(LPC" << LabelId << "+" << (unsigned)PCAdjust;
    if (AddCurrentAddress) O << "-.";
    O << ")";
  }
}
