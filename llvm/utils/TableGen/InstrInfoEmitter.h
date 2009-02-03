//===- InstrInfoEmitter.h - Generate a Instruction Set Desc. ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of the target
// instruction set for the code generator.
//
//===----------------------------------------------------------------------===//

#ifndef INSTRINFO_EMITTER_H
#define INSTRINFO_EMITTER_H

#include "TableGenBackend.h"
#include "CodeGenDAGPatterns.h"
#include <vector>
#include <map>

namespace llvm {

class StringInit;
class IntInit;
class ListInit;
class CodeGenInstruction;

class InstrInfoEmitter : public TableGenBackend {
  RecordKeeper &Records;
  CodeGenDAGPatterns CDP;
  std::map<std::string, unsigned> ItinClassMap;
  
public:
  InstrInfoEmitter(RecordKeeper &R) : Records(R), CDP(R) { }

  // run - Output the instruction set description, returning true on failure.
  void run(std::ostream &OS);

private:
  typedef std::map<std::vector<std::string>, unsigned> OperandInfoMapTy;
  
  void emitRecord(const CodeGenInstruction &Inst, unsigned Num,
                  Record *InstrInfo, 
                  std::map<std::vector<Record*>, unsigned> &EL,
                  std::map<Record*, unsigned> &BM,
                  const OperandInfoMapTy &OpInfo,
                  std::ostream &OS);
  void emitShiftedValue(Record *R, StringInit *Val, IntInit *Shift,
                        std::ostream &OS);

  // Itinerary information.
  void GatherItinClasses();
  unsigned getItinClassNumber(const Record *InstRec);
  
  // Operand information.
  void EmitOperandInfo(std::ostream &OS, OperandInfoMapTy &OperandInfoIDs);
  std::vector<std::string> GetOperandInfo(const CodeGenInstruction &Inst);

  void DetectRegisterClassBarriers(std::vector<Record*> &Defs,
                                   const std::vector<CodeGenRegisterClass> &RCs,
                                   std::vector<Record*> &Barriers);
};

} // End llvm namespace

#endif
