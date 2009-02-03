//===- IntrinsicEmitter.h - Generate intrinsic information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits information about intrinsic functions.
//
//===----------------------------------------------------------------------===//

#ifndef INTRINSIC_EMITTER_H
#define INTRINSIC_EMITTER_H

#include "CodeGenIntrinsics.h"
#include "TableGenBackend.h"

namespace llvm {
  class IntrinsicEmitter : public TableGenBackend {
    RecordKeeper &Records;
    
  public:
    IntrinsicEmitter(RecordKeeper &R) : Records(R) {}

    void run(std::ostream &OS);
    
    void EmitEnumInfo(const std::vector<CodeGenIntrinsic> &Ints, 
                      std::ostream &OS);

    void EmitFnNameRecognizer(const std::vector<CodeGenIntrinsic> &Ints, 
                              std::ostream &OS);
    void EmitIntrinsicToNameTable(const std::vector<CodeGenIntrinsic> &Ints, 
                                  std::ostream &OS);
    void EmitVerifier(const std::vector<CodeGenIntrinsic> &Ints, 
                      std::ostream &OS);
    void EmitGenerator(const std::vector<CodeGenIntrinsic> &Ints, 
                       std::ostream &OS);
    void EmitAttributes(const std::vector<CodeGenIntrinsic> &Ints,
                        std::ostream &OS);
    void EmitGCCBuiltinList(const std::vector<CodeGenIntrinsic> &Ints, 
                            std::ostream &OS);
    void EmitIntrinsicToGCCBuiltinMap(const std::vector<CodeGenIntrinsic> &Ints, 
                                      std::ostream &OS);
  };

} // End llvm namespace

#endif



