//===- CallingConvEmitter.h - Generate calling conventions ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting descriptions of the calling
// conventions supported by this target.
//
//===----------------------------------------------------------------------===//

#ifndef CALLINGCONV_EMITTER_H
#define CALLINGCONV_EMITTER_H

#include "TableGenBackend.h"
#include <map>
#include <vector>
#include <cassert>

namespace llvm {
  class CallingConvEmitter : public TableGenBackend {
    RecordKeeper &Records;
  public:
    explicit CallingConvEmitter(RecordKeeper &R) : Records(R) {}

    // run - Output the asmwriter, returning true on failure.
    void run(std::ostream &o);
    
  private:
    void EmitCallingConv(Record *CC, std::ostream &O);
    void EmitAction(Record *Action, unsigned Indent, std::ostream &O);
    unsigned Counter;
  };
}
#endif
