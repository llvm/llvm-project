//===-- BrainF.h - BrainF compiler class ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===--------------------------------------------------------------------===//
//
// This class stores the data for the BrainF compiler so it doesn't have
// to pass all of it around.  The main method is parse.
//
//===--------------------------------------------------------------------===//

#ifndef BRAINF_H
#define BRAINF_H

#include "llvm/IRBuilder.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"

using namespace llvm;

/// This class provides a parser for the BrainF language.
/// The class itself is made to store values during
/// parsing so they don't have to be passed around
/// as much.
class BrainF {
  public:
    /// Options for how BrainF should compile
    enum CompileFlags {
      flag_off         = 0,
      flag_arraybounds = 1
    };

    /// This is the main method.  It parses BrainF from in1
    /// and returns the module with a function
    /// void brainf()
    /// containing the resulting code.
    /// On error, it calls abort.
    /// The caller must delete the returned module.
    Module *parse(std::istream *in1, int mem, CompileFlags cf,
                  LLVMContext& C);

  protected:
    /// The different symbols in the BrainF language
    enum Symbol {
      SYM_NONE,
      SYM_READ,
      SYM_WRITE,
      SYM_MOVE,
      SYM_CHANGE,
      SYM_LOOP,
      SYM_ENDLOOP,
      SYM_EOF
    };

    /// Names of the different parts of the language.
    /// Tape is used for reading and writing the tape.
    /// headreg is used for the position of the head.
    /// label is used for the labels for the BasicBlocks.
    /// testreg is used for testing the loop exit condition.
    static const char *tapereg;
    static const char *headreg;
    static const char *label;
    static const char *testreg;

    /// Put the brainf function preamble and other fixed pieces of code
    void header(LLVMContext& C);

    /// The main loop for parsing.  It calls itself recursively
    /// to handle the depth of nesting of "[]".
    void readloop(PHINode *phi, BasicBlock *oldbb,
                  BasicBlock *testbb, LLVMContext &Context);

    /// Constants during parsing
    int memtotal;
    CompileFlags comflag;
    std::istream *in;
    Module *module;
    Function *brainf_func;
    Function *getchar_func;
    Function *putchar_func;
    Value *ptr_arr;
    Value *ptr_arrmax;
    BasicBlock *endbb;
    BasicBlock *aberrorbb;

    /// Variables
    IRBuilder<> *builder;
    Value *curhead;
};

#endif
