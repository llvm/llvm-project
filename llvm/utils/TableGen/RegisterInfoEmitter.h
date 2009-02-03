//===- RegisterInfoEmitter.h - Generate a Register File Desc. ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting a description of a target
// register file for a code generator.  It uses instances of the Register,
// RegisterAliases, and RegisterClass classes to gather this information.
//
//===----------------------------------------------------------------------===//

#ifndef REGISTER_INFO_EMITTER_H
#define REGISTER_INFO_EMITTER_H

#include "TableGenBackend.h"

namespace llvm {

class RegisterInfoEmitter : public TableGenBackend {
  RecordKeeper &Records;
public:
  RegisterInfoEmitter(RecordKeeper &R) : Records(R) {}

  // run - Output the register file description, returning true on failure.
  void run(std::ostream &o);

  // runHeader - Emit a header fragment for the register info emitter.
  void runHeader(std::ostream &o);

  // runEnums - Print out enum values for all of the registers.
  void runEnums(std::ostream &o);
};

} // End llvm namespace

#endif
