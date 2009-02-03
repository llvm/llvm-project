//===-- CPPTargetMachine.h - TargetMachine for the C++ backend --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the TargetMachine that is used by the C++ backend.
//
//===----------------------------------------------------------------------===//

#ifndef CPPTARGETMACHINE_H
#define CPPTARGETMACHINE_H

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

namespace llvm {

class raw_ostream;

struct CPPTargetMachine : public TargetMachine {
  const TargetData DataLayout;       // Calculates type size & alignment

  CPPTargetMachine(const Module &M, const std::string &FS)
    : DataLayout(&M) {}

  virtual bool WantsWholeFile() const { return true; }
  virtual bool addPassesToEmitWholeFile(PassManager &PM, raw_ostream &Out,
                                        CodeGenFileType FileType, bool Fast);

  // This class always works, but shouldn't be the default in most cases.
  static unsigned getModuleMatchQuality(const Module &M) { return 1; }

  virtual const TargetData *getTargetData() const { return &DataLayout; }
};

} // End llvm namespace


#endif
