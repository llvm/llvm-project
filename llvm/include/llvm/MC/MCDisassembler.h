//===-- llvm/MC/MCDisassembler.h - Disassembler interface -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef MCDISASSEMBLER_H
#define MCDISASSEMBLER_H

#include "llvm/Support/DataTypes.h"
#include "llvm-c/Disassembler.h"

namespace llvm {

class MCInst;
class MCSubtargetInfo;
class MemoryObject;
class raw_ostream;
class MCContext;

struct EDInstInfo;

/// MCDisassembler - Superclass for all disassemblers.  Consumes a memory region
///   and provides an array of assembly instructions.
class MCDisassembler {
public:
  /// Ternary decode status. Most backends will just use Fail and
  /// Success, however some have a concept of an instruction with
  /// understandable semantics but which is architecturally
  /// incorrect. An example of this is ARM UNPREDICTABLE instructions
  /// which are disassemblable but cause undefined behaviour.
  ///
  /// Because it makes sense to disassemble these instructions, there
  /// is a "soft fail" failure mode that indicates the MCInst& is
  /// valid but architecturally incorrect.
  ///
  /// The enum numbers are deliberately chosen such that reduction
  /// from Success->SoftFail ->Fail can be done with a simple
  /// bitwise-AND:
  ///
  ///   LEFT & TOP =  | Success       Unpredictable   Fail
  ///   --------------+-----------------------------------
  ///   Success       | Success       Unpredictable   Fail
  ///   Unpredictable | Unpredictable Unpredictable   Fail
  ///   Fail          | Fail          Fail            Fail
  ///
  /// An easy way of encoding this is as 0b11, 0b01, 0b00 for
  /// Success, SoftFail, Fail respectively.
  enum DecodeStatus {
    Fail = 0,
    SoftFail = 1,
    Success = 3
  };

  /// Constructor     - Performs initial setup for the disassembler.
  MCDisassembler(const MCSubtargetInfo &STI) : GetOpInfo(0), SymbolLookUp(0),
                                               DisInfo(0), Ctx(0),
                                               STI(STI), CommentStream(0) {}

  virtual ~MCDisassembler();

  /// getInstruction  - Returns the disassembly of a single instruction.
  ///
  /// @param instr    - An MCInst to populate with the contents of the
  ///                   instruction.
  /// @param size     - A value to populate with the size of the instruction, or
  ///                   the number of bytes consumed while attempting to decode
  ///                   an invalid instruction.
  /// @param region   - The memory object to use as a source for machine code.
  /// @param address  - The address, in the memory space of region, of the first
  ///                   byte of the instruction.
  /// @param vStream  - The stream to print warnings and diagnostic messages on.
  /// @param cStream  - The stream to print comments and annotations on.
  /// @return         - MCDisassembler::Success if the instruction is valid,
  ///                   MCDisassembler::SoftFail if the instruction was
  ///                                            disassemblable but invalid,
  ///                   MCDisassembler::Fail if the instruction was invalid.
  virtual DecodeStatus  getInstruction(MCInst& instr,
                                       uint64_t& size,
                                       const MemoryObject &region,
                                       uint64_t address,
                                       raw_ostream &vStream,
                                       raw_ostream &cStream) const = 0;

  /// getEDInfo - Returns the enhanced instruction information corresponding to
  ///   the disassembler.
  ///
  /// @return         - An array of instruction information, with one entry for
  ///                   each MCInst opcode this disassembler returns.
  ///                   NULL if there is no info for this target.
  virtual const EDInstInfo   *getEDInfo() const { return (EDInstInfo*)0; }

private:
  //
  // Hooks for symbolic disassembly via the public 'C' interface.
  //
  // The function to get the symbolic information for operands.
  LLVMOpInfoCallback GetOpInfo;
  // The function to lookup a symbol name.
  LLVMSymbolLookupCallback SymbolLookUp;
  // The pointer to the block of symbolic information for above call back.
  void *DisInfo;
  // The assembly context for creating symbols and MCExprs in place of
  // immediate operands when there is symbolic information.
  MCContext *Ctx;
protected:
  // Subtarget information, for instruction decoding predicates if required.
  const MCSubtargetInfo &STI;

public:
  void setupForSymbolicDisassembly(LLVMOpInfoCallback getOpInfo,
                                   LLVMSymbolLookupCallback symbolLookUp,
                                   void *disInfo,
                                   MCContext *ctx) {
    GetOpInfo = getOpInfo;
    SymbolLookUp = symbolLookUp;
    DisInfo = disInfo;
    Ctx = ctx;
  }
  LLVMOpInfoCallback getLLVMOpInfoCallback() const { return GetOpInfo; }
  LLVMSymbolLookupCallback getLLVMSymbolLookupCallback() const {
    return SymbolLookUp;
  }
  void *getDisInfoBlock() const { return DisInfo; }
  MCContext *getMCContext() const { return Ctx; }

  // Marked mutable because we cache it inside the disassembler, rather than
  // having to pass it around as an argument through all the autogenerated code.
  mutable raw_ostream *CommentStream;
};

} // namespace llvm

#endif
