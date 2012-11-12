//===-EDOperand.h - LLVM Enhanced Disassembler ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the interface for the Enhanced Disassembly library's 
// operand class.  The operand is responsible for allowing evaluation given a
// particular register context.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EDOPERAND_H
#define LLVM_EDOPERAND_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

struct EDDisassembler;
struct EDInst;
  
typedef int (*EDRegisterReaderCallback)(uint64_t *value, unsigned regID, 
                                        void* arg);


/// EDOperand - Encapsulates a single operand, which can be evaluated by the
///   client
struct EDOperand {
  /// The parent disassembler
  const EDDisassembler &Disassembler;
  /// The parent instruction
  const EDInst &Inst;
  
  /// The index of the operand in the EDInst
  unsigned int OpIndex;
  /// The index of the first component of the operand in the MCInst
  unsigned int MCOpIndex;
  
  /// Constructor - Initializes an EDOperand
  ///
  /// @arg disassembler - The disassembler responsible for the operand
  /// @arg inst         - The instruction containing this operand
  /// @arg opIndex      - The index of the operand in inst
  /// @arg mcOpIndex    - The index of the operand in the original MCInst
  EDOperand(const EDDisassembler &disassembler,
            const EDInst &inst,
            unsigned int opIndex,
            unsigned int &mcOpIndex);
  ~EDOperand();
  
  /// evaluate - Returns the numeric value of an operand to the extent possible,
  ///   returning 0 on success or -1 if there was some problem (such as a 
  ///   register not being readable)
  ///
  /// @arg result   - A reference whose target is filled in with the value of
  ///                 the operand (the address if it is a memory operand)
  /// @arg callback - A function to call to obtain register values
  /// @arg arg      - An opaque argument to pass to callback
  int evaluate(uint64_t &result,
               EDRegisterReaderCallback callback,
               void *arg);

  /// isRegister - Returns 1 if the operand is a register or 0 otherwise
  int isRegister();
  /// regVal - Returns the register value.
  unsigned regVal();
  
  /// isImmediate - Returns 1 if the operand is an immediate or 0 otherwise
  int isImmediate();
  /// immediateVal - Returns the immediate value.
  uint64_t immediateVal();
  
  /// isMemory - Returns 1 if the operand is a memory location or 0 otherwise
  int isMemory();
  
#ifdef __BLOCKS__
  typedef int (^EDRegisterBlock_t)(uint64_t *value, unsigned regID);

  /// evaluate - Like evaluate for a callback, but uses a block instead
  int evaluate(uint64_t &result,
               EDRegisterBlock_t regBlock);
#endif
};

} // end namespace llvm

#endif
