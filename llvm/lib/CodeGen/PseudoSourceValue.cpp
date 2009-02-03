//===-- llvm/CodeGen/PseudoSourceValue.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PseudoSourceValue class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
using namespace llvm;

static ManagedStatic<PseudoSourceValue[4]> PSVs;

const PseudoSourceValue *PseudoSourceValue::getStack()
{ return &(*PSVs)[0]; }
const PseudoSourceValue *PseudoSourceValue::getGOT()
{ return &(*PSVs)[1]; }
const PseudoSourceValue *PseudoSourceValue::getJumpTable()
{ return &(*PSVs)[2]; }
const PseudoSourceValue *PseudoSourceValue::getConstantPool()
{ return &(*PSVs)[3]; }

static const char *const PSVNames[] = {
  "Stack",
  "GOT",
  "JumpTable",
  "ConstantPool"
};

PseudoSourceValue::PseudoSourceValue() :
  Value(PointerType::getUnqual(Type::Int8Ty), PseudoSourceValueVal) {}

void PseudoSourceValue::dump() const {
  print(errs()); errs() << '\n'; errs().flush();
}

void PseudoSourceValue::print(raw_ostream &OS) const {
  OS << PSVNames[this - *PSVs];
}

namespace {
  /// FixedStackPseudoSourceValue - A specialized PseudoSourceValue
  /// for holding FixedStack values, which must include a frame
  /// index.
  class VISIBILITY_HIDDEN FixedStackPseudoSourceValue
    : public PseudoSourceValue {
    const int FI;
  public:
    explicit FixedStackPseudoSourceValue(int fi) : FI(fi) {}

    virtual bool isConstant(const MachineFrameInfo *MFI) const;

    virtual void print(raw_ostream &OS) const {
      OS << "FixedStack" << FI;
    }
  };
}

static ManagedStatic<std::map<int, const PseudoSourceValue *> > FSValues;

const PseudoSourceValue *PseudoSourceValue::getFixedStack(int FI) {
  const PseudoSourceValue *&V = (*FSValues)[FI];
  if (!V)
    V = new FixedStackPseudoSourceValue(FI);
  return V;
}

bool PseudoSourceValue::isConstant(const MachineFrameInfo *) const {
  if (this == getStack())
    return false;
  if (this == getGOT() ||
      this == getConstantPool() ||
      this == getJumpTable())
    return true;
  assert(0 && "Unknown PseudoSourceValue!");
  return false;
}

bool FixedStackPseudoSourceValue::isConstant(const MachineFrameInfo *MFI) const{
  return MFI && MFI->isImmutableObjectIndex(FI);
}
