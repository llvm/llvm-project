//===- AsmWriterEmitter.h - Generate an assembly writer ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting an assembly printer for the
// code generator.
//
//===----------------------------------------------------------------------===//

#ifndef ASMWRITER_EMITTER_H
#define ASMWRITER_EMITTER_H

#include "TableGenBackend.h"
#include <map>
#include <vector>
#include <cassert>

namespace llvm {
  class AsmWriterInst;
  class CodeGenInstruction;
  
  class AsmWriterEmitter : public TableGenBackend {
    RecordKeeper &Records;
    std::map<const CodeGenInstruction*, AsmWriterInst*> CGIAWIMap;
    std::vector<const CodeGenInstruction*> NumberedInstructions;
  public:
    AsmWriterEmitter(RecordKeeper &R) : Records(R) {}

    // run - Output the asmwriter, returning true on failure.
    void run(std::ostream &o);

private:
    AsmWriterInst *getAsmWriterInstByID(unsigned ID) const {
      assert(ID < NumberedInstructions.size());
      std::map<const CodeGenInstruction*, AsmWriterInst*>::const_iterator I =
        CGIAWIMap.find(NumberedInstructions[ID]);
      assert(I != CGIAWIMap.end() && "Didn't find inst!");
      return I->second;
    }
    void FindUniqueOperandCommands(std::vector<std::string> &UOC,
                                   std::vector<unsigned> &InstIdxs,
                                   std::vector<unsigned> &InstOpsUsed) const;
  };
}
#endif
