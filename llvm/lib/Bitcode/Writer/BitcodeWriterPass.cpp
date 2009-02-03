//===--- Bitcode/Writer/BitcodeWriterPass.cpp - Bitcode Writer ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// BitcodeWriterPass implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Pass.h"
using namespace llvm;

namespace {
  class WriteBitcodePass : public ModulePass {
    // FIXME: Kill off std::ostream
    std::ostream *Out;
    raw_ostream *RawOut; // raw_ostream to print on
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit WriteBitcodePass(std::ostream &o)
      : ModulePass(&ID), Out(&o), RawOut(0) {}
    explicit WriteBitcodePass(raw_ostream &o)
      : ModulePass(&ID), Out(0), RawOut(&o) {}
    
    const char *getPassName() const { return "Bitcode Writer"; }
    
    bool runOnModule(Module &M) {
      if (Out) {
        WriteBitcodeToFile(&M, *Out);
      } else {
        WriteBitcodeToFile(&M, *RawOut);
      }
      return false;
    }
  };
}

char WriteBitcodePass::ID = 0;

/// CreateBitcodeWriterPass - Create and return a pass that writes the module
/// to the specified ostream.
ModulePass *llvm::CreateBitcodeWriterPass(std::ostream &Str) {
  return new WriteBitcodePass(Str);
}


/// createBitcodeWriterPass - Create and return a pass that writes the module
/// to the specified ostream.
ModulePass *llvm::createBitcodeWriterPass(raw_ostream &Str) {
  return new WriteBitcodePass(Str);
}
