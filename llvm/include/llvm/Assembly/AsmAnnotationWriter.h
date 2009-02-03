//===-- AsmAnnotationWriter.h - Itf for annotation .ll files - --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Clients of the assembly writer can use this interface to add their own
// special-purpose annotations to LLVM assembly language printouts.  Note that
// the assembly parser won't be able to parse these, in general, so
// implementations are advised to print stuff as LLVM comments.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ASSEMBLY_ASMANNOTATIONWRITER_H
#define LLVM_ASSEMBLY_ASMANNOTATIONWRITER_H

namespace llvm {

class Function;
class BasicBlock;
class Instruction;
class raw_ostream;

class AssemblyAnnotationWriter {
public:

  virtual ~AssemblyAnnotationWriter();

  // emitFunctionAnnot - This may be implemented to emit a string right before
  // the start of a function.
  virtual void emitFunctionAnnot(const Function *F, raw_ostream &OS) {}

  // emitBasicBlockStartAnnot - This may be implemented to emit a string right
  // after the basic block label, but before the first instruction in the block.
  virtual void emitBasicBlockStartAnnot(const BasicBlock *BB, raw_ostream &OS){
  }

  // emitBasicBlockEndAnnot - This may be implemented to emit a string right
  // after the basic block.
  virtual void emitBasicBlockEndAnnot(const BasicBlock *BB, raw_ostream &OS){
  }

  // emitInstructionAnnot - This may be implemented to emit a string right
  // before an instruction is emitted.
  virtual void emitInstructionAnnot(const Instruction *I, raw_ostream &OS) {}
};

} // End llvm namespace

#endif
