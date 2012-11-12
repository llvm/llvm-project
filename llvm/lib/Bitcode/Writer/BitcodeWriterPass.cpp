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
    raw_ostream &OS; // raw_ostream to print on
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit WriteBitcodePass(raw_ostream &o)
      : ModulePass(ID), OS(o) {}
    
    const char *getPassName() const { return "Bitcode Writer"; }
    
    bool runOnModule(Module &M) {
      WriteBitcodeToFile(&M, OS);
      return false;
    }
  };
}

char WriteBitcodePass::ID = 0;

/// createBitcodeWriterPass - Create and return a pass that writes the module
/// to the specified ostream.
ModulePass *llvm::createBitcodeWriterPass(raw_ostream &Str) {
  return new WriteBitcodePass(Str);
}
