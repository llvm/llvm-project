//===- CodeEmitterGen.h - Code Emitter Generator ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// FIXME: document
//
//===----------------------------------------------------------------------===//

#ifndef CODEMITTERGEN_H
#define CODEMITTERGEN_H

#include "TableGenBackend.h"
#include <map>
#include <vector>
#include <string>

namespace llvm {

class RecordVal;
class BitsInit;

class CodeEmitterGen : public TableGenBackend {
  RecordKeeper &Records;
public:
  CodeEmitterGen(RecordKeeper &R) : Records(R) {}

  // run - Output the code emitter
  void run(std::ostream &o);
private:
  void emitMachineOpEmitter(std::ostream &o, const std::string &Namespace);
  void emitGetValueBit(std::ostream &o, const std::string &Namespace);
  void reverseBits(std::vector<Record*> &Insts);
  int getVariableBit(const std::string &VarName, BitsInit *BI, int bit);
};

} // End llvm namespace

#endif
