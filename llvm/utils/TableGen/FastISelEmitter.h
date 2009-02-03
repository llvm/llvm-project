//===- FastISelEmitter.h - Generate an instruction selector -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits a "fast" instruction selector.
//
//===----------------------------------------------------------------------===//

#ifndef FASTISEL_EMITTER_H
#define FASTISEL_EMITTER_H

#include "TableGenBackend.h"
#include "CodeGenDAGPatterns.h"

namespace llvm {

class CodeGenTarget;

/// FastISelEmitter - The top-level class which coordinates construction
/// and emission of the instruction selector.
///
class FastISelEmitter : public TableGenBackend {
  RecordKeeper &Records;
  CodeGenDAGPatterns CGP;
public:
  explicit FastISelEmitter(RecordKeeper &R);

  // run - Output the isel, returning true on failure.
  void run(std::ostream &OS);
};

} // End llvm namespace

#endif
