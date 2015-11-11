//===- Attributes.cpp - Generate attributes -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <algorithm>
#include <string>
#include <vector>
using namespace llvm;

#define DEBUG_TYPE "attr-enum"

namespace {

class Attributes {
public:
  Attributes(RecordKeeper &R) : Records(R) {}
  void emit(raw_ostream &OS);

private:
  void emitTargetIndependentEnums(raw_ostream &OS);

  RecordKeeper &Records;
};

} // End anonymous namespace.

void Attributes::emitTargetIndependentEnums(raw_ostream &OS) {
  OS << "#ifdef GET_ATTR_ENUM\n";
  OS << "#undef GET_ATTR_ENUM\n";

  const std::vector<Record*> &Attrs =
      Records.getAllDerivedDefinitions("EnumAttr");

  for (auto A : Attrs)
    OS << A->getName() << ",\n";

  OS << "#endif\n";
}

void Attributes::emit(raw_ostream &OS) {
  emitTargetIndependentEnums(OS);
}

namespace llvm {

void EmitAttributes(RecordKeeper &RK, raw_ostream &OS) {
  Attributes(RK).emit(OS);
}

} // End llvm namespace.
