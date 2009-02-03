//===- LLVMCConfigurationEmitter.cpp - Generate LLVMCC config ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting LLVMCC configuration code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_LLVMCCONF_EMITTER_H
#define LLVM_UTILS_TABLEGEN_LLVMCCONF_EMITTER_H

#include "TableGenBackend.h"

namespace llvm {

  /// LLVMCConfigurationEmitter - TableGen backend that generates
  /// configuration code for LLVMC.
  class LLVMCConfigurationEmitter : public TableGenBackend {
    RecordKeeper &Records;
  public:
    explicit LLVMCConfigurationEmitter(RecordKeeper &R) : Records(R) {}

    // run - Output the asmwriter, returning true on failure.
    void run(std::ostream &o);
  };
}

#endif //LLVM_UTILS_TABLEGEN_LLVMCCONF_EMITTER_H
