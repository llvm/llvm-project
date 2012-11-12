//===-- EDInst.h - LLVM Enhanced Disassembler -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines the interface for the Enhanced Disassembly library's
// instruction class.  The instruction is responsible for vending the string
// representation, individual tokens and operands for a single instruction.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EDINST_H
#define LLVM_EDINST_H

#include "llvm/Support/DataTypes.h"
#include "llvm/ADT/SmallVector.h"
#include <string>
#include <vector>

namespace llvm {
  class MCInst;
  struct EDInstInfo;
  struct EDToken;
  struct EDDisassembler;
  struct EDOperand;

#ifdef __BLOCKS__
  typedef int (^EDTokenVisitor_t)(EDToken *token);
#endif

/// CachedResult - Encapsulates the result of a function along with the validity
///   of that result, so that slow functions don't need to run twice
struct CachedResult {
  /// True if the result has been obtained by executing the function
  bool Valid;
  /// The result last obtained from the function
  int Result;
  
  /// Constructor - Initializes an invalid result
  CachedResult() : Valid(false) { }
  /// valid - Returns true if the result has been obtained by executing the
  ///   function and false otherwise
  bool valid() { return Valid; }
  /// result - Returns the result of the function or an undefined value if
  ///   valid() is false
  int result() { return Result; }
  /// setResult - Sets the result of the function and declares it valid
  ///   returning the result (so that setResult() can be called from inside a
  ///   return statement)
  /// @arg result - The result of the function
  int setResult(int result) { Result = result; Valid = true; return result; }
};

/// EDInst - Encapsulates a single instruction, which can be queried for its
///   string representation, as well as its operands and tokens
struct EDInst {
  /// The parent disassembler
  EDDisassembler &Disassembler;
  /// The containing MCInst
  llvm::MCInst *Inst;
  /// The instruction information provided by TableGen for this instruction
  const llvm::EDInstInfo *ThisInstInfo;
  /// The number of bytes for the machine code representation of the instruction
  uint64_t ByteSize;
  
  /// The result of the stringify() function
  CachedResult StringifyResult;
  /// The string representation of the instruction
  std::string String;
  /// The order in which operands from the InstInfo's operand information appear
  /// in String
  const signed char* OperandOrder;
  
  /// The result of the parseOperands() function
  CachedResult ParseResult;
  typedef llvm::SmallVector<EDOperand*, 5> opvec_t;
  /// The instruction's operands
  opvec_t Operands;
  /// The operand corresponding to the target, if the instruction is a branch
  int BranchTarget;
  /// The operand corresponding to the source, if the instruction is a move
  int MoveSource;
  /// The operand corresponding to the target, if the instruction is a move
  int MoveTarget;
  
  /// The result of the tokenize() function
  CachedResult TokenizeResult;
  typedef std::vector<EDToken*> tokvec_t;
  /// The instruction's tokens
  tokvec_t Tokens;
  
  /// Constructor - initializes an instruction given the output of the LLVM
  ///   C++ disassembler
  ///
  /// @arg inst         - The MCInst, which will now be owned by this object
  /// @arg byteSize     - The size of the consumed instruction, in bytes
  /// @arg disassembler - The parent disassembler
  /// @arg instInfo     - The instruction information produced by the table
  ///                     generator for this instruction
  EDInst(llvm::MCInst *inst,
         uint64_t byteSize,
         EDDisassembler &disassembler,
         const llvm::EDInstInfo *instInfo);
  ~EDInst();
  
  /// byteSize - returns the number of bytes consumed by the machine code
  ///   representation of the instruction
  uint64_t byteSize();
  /// instID - returns the LLVM instruction ID of the instruction
  unsigned instID();
  
  /// stringify - populates the String and AsmString members of the instruction,
  ///   returning 0 on success or -1 otherwise
  int stringify();
  /// getString - retrieves a pointer to the string representation of the
  ///   instructinon, returning 0 on success or -1 otherwise
  ///
  /// @arg str - A reference to a pointer that, on success, is set to point to
  ///   the string representation of the instruction; this string is still owned
  ///   by the instruction and will be deleted when it is
  int getString(const char *&str);
  
  /// isBranch - Returns true if the instruction is a branch
  bool isBranch();
  /// isMove - Returns true if the instruction is a move
  bool isMove();
  
  /// parseOperands - populates the Operands member of the instruction,
  ///   returning 0 on success or -1 otherwise
  int parseOperands();
  /// branchTargetID - returns the ID (suitable for use with getOperand()) of 
  ///   the target operand if the instruction is a branch, or -1 otherwise
  int branchTargetID();
  /// moveSourceID - returns the ID of the source operand if the instruction
  ///   is a move, or -1 otherwise
  int moveSourceID();
  /// moveTargetID - returns the ID of the target operand if the instruction
  ///   is a move, or -1 otherwise
  int moveTargetID();
  
  /// numOperands - returns the number of operands available to retrieve, or -1
  ///   on error
  int numOperands();
  /// getOperand - retrieves an operand from the instruction's operand list by
  ///   index, returning 0 on success or -1 on error
  ///
  /// @arg operand  - A reference whose target is pointed at the operand on
  ///                 success, although the operand is still owned by the EDInst
  /// @arg index    - The index of the operand in the instruction
  int getOperand(EDOperand *&operand, unsigned int index);

  /// tokenize - populates the Tokens member of the instruction, returning 0 on
  ///   success or -1 otherwise
  int tokenize();
  /// numTokens - returns the number of tokens in the instruction, or -1 on
  ///   error
  int numTokens();
  /// getToken - retrieves a token from the instruction's token list by index,
  ///   returning 0 on success or -1 on error
  ///
  /// @arg token  - A reference whose target is pointed at the token on success,
  ///               although the token is still owned by the EDInst
  /// @arg index  - The index of the token in the instrcutino
  int getToken(EDToken *&token, unsigned int index);

#ifdef __BLOCKS__
  /// visitTokens - Visits each token in turn and applies a block to it,
  ///   returning 0 if all blocks are visited and/or the block signals
  ///   termination by returning 1; returns -1 on error
  ///
  /// @arg visitor  - The visitor block to apply to all tokens.
  int visitTokens(EDTokenVisitor_t visitor);
#endif
};

} // end namespace llvm

#endif
