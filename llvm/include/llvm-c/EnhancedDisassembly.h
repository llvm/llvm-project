/*===-- llvm-c/EnhancedDisassembly.h - Disassembler C Interface ---*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to EnhancedDisassembly.so, which      *|
|* implements a disassembler with the ability to extract operand values and   *|
|* individual tokens from assembly instructions.                              *|
|*                                                                            *|
|* The header declares additional interfaces if the host compiler supports    *|
|* the blocks API.                                                            *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_ENHANCEDDISASSEMBLY_H
#define LLVM_C_ENHANCEDDISASSEMBLY_H

#include "llvm/Support/DataTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup LLVMCEnhancedDisassembly Enhanced Disassembly
 * @ingroup LLVMC
 * @deprecated
 *
 * This module contains an interface to the Enhanced Disassembly (edis)
 * library. The edis library is deprecated and will likely disappear in
 * the near future. You should use the @ref LLVMCDisassembler interface
 * instead.
 *
 * @{
 */

/*!
 @typedef EDByteReaderCallback
 Interface to memory from which instructions may be read.
 @param byte A pointer whose target should be filled in with the data returned.
 @param address The address of the byte to be read.
 @param arg An anonymous argument for client use.
 @result 0 on success; -1 otherwise.
 */
typedef int (*EDByteReaderCallback)(uint8_t *byte, uint64_t address, void *arg);

/*!
 @typedef EDRegisterReaderCallback
 Interface to registers from which registers may be read.
 @param value A pointer whose target should be filled in with the value of the
   register.
 @param regID The LLVM register identifier for the register to read.
 @param arg An anonymous argument for client use.
 @result 0 if the register could be read; -1 otherwise.
 */
typedef int (*EDRegisterReaderCallback)(uint64_t *value, unsigned regID,
                                        void* arg);

/*!
 @typedef EDAssemblySyntax_t
 An assembly syntax for use in tokenizing instructions.
 */
enum {
/*! @constant kEDAssemblySyntaxX86Intel Intel syntax for i386 and x86_64. */
  kEDAssemblySyntaxX86Intel  = 0,
/*! @constant kEDAssemblySyntaxX86ATT AT&T syntax for i386 and x86_64. */
  kEDAssemblySyntaxX86ATT    = 1,
  kEDAssemblySyntaxARMUAL    = 2
};
typedef unsigned EDAssemblySyntax_t;

/*!
 @typedef EDDisassemblerRef
 Encapsulates a disassembler for a single CPU architecture.
 */
typedef void *EDDisassemblerRef;

/*!
 @typedef EDInstRef
 Encapsulates a single disassembled instruction in one assembly syntax.
 */
typedef void *EDInstRef;

/*!
 @typedef EDTokenRef
 Encapsulates a token from the disassembly of an instruction.
 */
typedef void *EDTokenRef;

/*!
 @typedef EDOperandRef
 Encapsulates an operand of an instruction.
 */
typedef void *EDOperandRef;

/*!
 @functiongroup Getting a disassembler
 */

/*!
 @function EDGetDisassembler
 Gets the disassembler for a given target.
 @param disassembler A pointer whose target will be filled in with the
   disassembler.
 @param triple Identifies the target.  Example: "x86_64-apple-darwin10"
 @param syntax The assembly syntax to use when decoding instructions.
 @result 0 on success; -1 otherwise.
 */
int EDGetDisassembler(EDDisassemblerRef *disassembler,
                      const char *triple,
                      EDAssemblySyntax_t syntax);

/*!
 @functiongroup Generic architectural queries
 */

/*!
 @function EDGetRegisterName
 Gets the human-readable name for a given register.
 @param regName A pointer whose target will be pointed at the name of the
   register.  The name does not need to be deallocated and will be
 @param disassembler The disassembler to query for the name.
 @param regID The register identifier, as returned by EDRegisterTokenValue.
 @result 0 on success; -1 otherwise.
 */
int EDGetRegisterName(const char** regName,
                      EDDisassemblerRef disassembler,
                      unsigned regID);

/*!
 @function EDRegisterIsStackPointer
 Determines if a register is one of the platform's stack-pointer registers.
 @param disassembler The disassembler to query.
 @param regID The register identifier, as returned by EDRegisterTokenValue.
 @result 1 if true; 0 otherwise.
 */
int EDRegisterIsStackPointer(EDDisassemblerRef disassembler,
                             unsigned regID);

/*!
 @function EDRegisterIsProgramCounter
 Determines if a register is one of the platform's stack-pointer registers.
 @param disassembler The disassembler to query.
 @param regID The register identifier, as returned by EDRegisterTokenValue.
 @result 1 if true; 0 otherwise.
 */
int EDRegisterIsProgramCounter(EDDisassemblerRef disassembler,
                               unsigned regID);

/*!
 @functiongroup Creating and querying instructions
 */

/*!
 @function EDCreateInst
 Gets a set of contiguous instructions from a disassembler.
 @param insts A pointer to an array that will be filled in with the
   instructions.  Must have at least count entries.  Entries not filled in will
   be set to NULL.
 @param count The maximum number of instructions to fill in.
 @param disassembler The disassembler to use when decoding the instructions.
 @param byteReader The function to use when reading the instruction's machine
   code.
 @param address The address of the first byte of the instruction.
 @param arg An anonymous argument to be passed to byteReader.
 @result The number of instructions read on success; 0 otherwise.
 */
unsigned int EDCreateInsts(EDInstRef *insts,
                           unsigned int count,
                           EDDisassemblerRef disassembler,
                           EDByteReaderCallback byteReader,
                           uint64_t address,
                           void *arg);

/*!
 @function EDReleaseInst
 Frees the memory for an instruction.  The instruction can no longer be accessed
 after this call.
 @param inst The instruction to be freed.
 */
void EDReleaseInst(EDInstRef inst);

/*!
 @function EDInstByteSize
 @param inst The instruction to be queried.
 @result The number of bytes in the instruction's machine-code representation.
 */
int EDInstByteSize(EDInstRef inst);

/*!
 @function EDGetInstString
 Gets the disassembled text equivalent of the instruction.
 @param buf A pointer whose target will be filled in with a pointer to the
   string.  (The string becomes invalid when the instruction is released.)
 @param inst The instruction to be queried.
 @result 0 on success; -1 otherwise.
 */
int EDGetInstString(const char **buf,
                    EDInstRef inst);

/*!
 @function EDInstID
 @param instID A pointer whose target will be filled in with the LLVM identifier
   for the instruction.
 @param inst The instruction to be queried.
 @result 0 on success; -1 otherwise.
 */
int EDInstID(unsigned *instID, EDInstRef inst);

/*!
 @function EDInstIsBranch
 @param inst The instruction to be queried.
 @result 1 if the instruction is a branch instruction; 0 if it is some other
   type of instruction; -1 if there was an error.
 */
int EDInstIsBranch(EDInstRef inst);

/*!
 @function EDInstIsMove
 @param inst The instruction to be queried.
 @result 1 if the instruction is a move instruction; 0 if it is some other
   type of instruction; -1 if there was an error.
 */
int EDInstIsMove(EDInstRef inst);

/*!
 @function EDBranchTargetID
 @param inst The instruction to be queried.
 @result The ID of the branch target operand, suitable for use with
   EDCopyOperand.  -1 if no such operand exists.
 */
int EDBranchTargetID(EDInstRef inst);

/*!
 @function EDMoveSourceID
 @param inst The instruction to be queried.
 @result The ID of the move source operand, suitable for use with
   EDCopyOperand.  -1 if no such operand exists.
 */
int EDMoveSourceID(EDInstRef inst);

/*!
 @function EDMoveTargetID
 @param inst The instruction to be queried.
 @result The ID of the move source operand, suitable for use with
   EDCopyOperand.  -1 if no such operand exists.
 */
int EDMoveTargetID(EDInstRef inst);

/*!
 @functiongroup Creating and querying tokens
 */

/*!
 @function EDNumTokens
 @param inst The instruction to be queried.
 @result The number of tokens in the instruction, or -1 on error.
 */
int EDNumTokens(EDInstRef inst);

/*!
 @function EDGetToken
 Retrieves a token from an instruction.  The token is valid until the
 instruction is released.
 @param token A pointer to be filled in with the token.
 @param inst The instruction to be queried.
 @param index The index of the token in the instruction.
 @result 0 on success; -1 otherwise.
 */
int EDGetToken(EDTokenRef *token,
               EDInstRef inst,
               int index);

/*!
 @function EDGetTokenString
 Gets the disassembled text for a token.
 @param buf A pointer whose target will be filled in with a pointer to the
   string.  (The string becomes invalid when the token is released.)
 @param token The token to be queried.
 @result 0 on success; -1 otherwise.
 */
int EDGetTokenString(const char **buf,
                     EDTokenRef token);

/*!
 @function EDOperandIndexForToken
 Returns the index of the operand to which a token belongs.
 @param token The token to be queried.
 @result The operand index on success; -1 otherwise
 */
int EDOperandIndexForToken(EDTokenRef token);

/*!
 @function EDTokenIsWhitespace
 @param token The token to be queried.
 @result 1 if the token is whitespace; 0 if not; -1 on error.
 */
int EDTokenIsWhitespace(EDTokenRef token);

/*!
 @function EDTokenIsPunctuation
 @param token The token to be queried.
 @result 1 if the token is punctuation; 0 if not; -1 on error.
 */
int EDTokenIsPunctuation(EDTokenRef token);

/*!
 @function EDTokenIsOpcode
 @param token The token to be queried.
 @result 1 if the token is opcode; 0 if not; -1 on error.
 */
int EDTokenIsOpcode(EDTokenRef token);

/*!
 @function EDTokenIsLiteral
 @param token The token to be queried.
 @result 1 if the token is a numeric literal; 0 if not; -1 on error.
 */
int EDTokenIsLiteral(EDTokenRef token);

/*!
 @function EDTokenIsRegister
 @param token The token to be queried.
 @result 1 if the token identifies a register; 0 if not; -1 on error.
 */
int EDTokenIsRegister(EDTokenRef token);

/*!
 @function EDTokenIsNegativeLiteral
 @param token The token to be queried.
 @result 1 if the token is a negative signed literal; 0 if not; -1 on error.
 */
int EDTokenIsNegativeLiteral(EDTokenRef token);

/*!
 @function EDLiteralTokenAbsoluteValue
 @param value A pointer whose target will be filled in with the absolute value
   of the literal.
 @param token The token to be queried.
 @result 0 on success; -1 otherwise.
 */
int EDLiteralTokenAbsoluteValue(uint64_t *value,
                                EDTokenRef token);

/*!
 @function EDRegisterTokenValue
 @param registerID A pointer whose target will be filled in with the LLVM
   register identifier for the token.
 @param token The token to be queried.
 @result 0 on success; -1 otherwise.
 */
int EDRegisterTokenValue(unsigned *registerID,
                         EDTokenRef token);

/*!
 @functiongroup Creating and querying operands
 */

/*!
 @function EDNumOperands
 @param inst The instruction to be queried.
 @result The number of operands in the instruction, or -1 on error.
 */
int EDNumOperands(EDInstRef inst);

/*!
 @function EDGetOperand
 Retrieves an operand from an instruction.  The operand is valid until the
 instruction is released.
 @param operand A pointer to be filled in with the operand.
 @param inst The instruction to be queried.
 @param index The index of the operand in the instruction.
 @result 0 on success; -1 otherwise.
 */
int EDGetOperand(EDOperandRef *operand,
                 EDInstRef inst,
                 int index);

/*!
 @function EDOperandIsRegister
 @param operand The operand to be queried.
 @result 1 if the operand names a register; 0 if not; -1 on error.
 */
int EDOperandIsRegister(EDOperandRef operand);

/*!
 @function EDOperandIsImmediate
 @param operand The operand to be queried.
 @result 1 if the operand specifies an immediate value; 0 if not; -1 on error.
 */
int EDOperandIsImmediate(EDOperandRef operand);

/*!
 @function EDOperandIsMemory
 @param operand The operand to be queried.
 @result 1 if the operand specifies a location in memory; 0 if not; -1 on error.
 */
int EDOperandIsMemory(EDOperandRef operand);

/*!
 @function EDRegisterOperandValue
 @param value A pointer whose target will be filled in with the LLVM register ID
   of the register named by the operand.
 @param operand The operand to be queried.
 @result 0 on success; -1 otherwise.
 */
int EDRegisterOperandValue(unsigned *value,
                           EDOperandRef operand);

/*!
 @function EDImmediateOperandValue
 @param value A pointer whose target will be filled in with the value of the
   immediate.
 @param operand The operand to be queried.
 @result 0 on success; -1 otherwise.
 */
int EDImmediateOperandValue(uint64_t *value,
                            EDOperandRef operand);

/*!
 @function EDEvaluateOperand
 Evaluates an operand using a client-supplied register state accessor.  Register
 operands are evaluated by reading the value of the register; immediate operands
 are evaluated by reporting the immediate value; memory operands are evaluated
 by computing the target address (with only those relocations applied that were
 already applied to the original bytes).
 @param result A pointer whose target is to be filled with the result of
   evaluating the operand.
 @param operand The operand to be evaluated.
 @param regReader The function to use when reading registers from the register
   state.
 @param arg An anonymous argument for client use.
 @result 0 if the operand could be evaluated; -1 otherwise.
 */
int EDEvaluateOperand(uint64_t *result,
                      EDOperandRef operand,
                      EDRegisterReaderCallback regReader,
                      void *arg);

#ifdef __BLOCKS__

/*!
 @typedef EDByteBlock_t
 Block-based interface to memory from which instructions may be read.
 @param byte A pointer whose target should be filled in with the data returned.
 @param address The address of the byte to be read.
 @result 0 on success; -1 otherwise.
 */
typedef int (^EDByteBlock_t)(uint8_t *byte, uint64_t address);

/*!
 @typedef EDRegisterBlock_t
 Block-based interface to registers from which registers may be read.
 @param value A pointer whose target should be filled in with the value of the
   register.
 @param regID The LLVM register identifier for the register to read.
 @result 0 if the register could be read; -1 otherwise.
 */
typedef int (^EDRegisterBlock_t)(uint64_t *value, unsigned regID);

/*!
 @typedef EDTokenVisitor_t
 Block-based handler for individual tokens.
 @param token The current token being read.
 @result 0 to continue; 1 to stop normally; -1 on error.
 */
typedef int (^EDTokenVisitor_t)(EDTokenRef token);

/*! @functiongroup Block-based interfaces */

/*!
 @function EDBlockCreateInsts
 Gets a set of contiguous instructions from a disassembler, using a block to
 read memory.
 @param insts A pointer to an array that will be filled in with the
   instructions.  Must have at least count entries.  Entries not filled in will
   be set to NULL.
 @param count The maximum number of instructions to fill in.
 @param disassembler The disassembler to use when decoding the instructions.
 @param byteBlock The block to use when reading the instruction's machine
   code.
 @param address The address of the first byte of the instruction.
 @result The number of instructions read on success; 0 otherwise.
 */
unsigned int EDBlockCreateInsts(EDInstRef *insts,
                                int count,
                                EDDisassemblerRef disassembler,
                                EDByteBlock_t byteBlock,
                                uint64_t address);

/*!
 @function EDBlockEvaluateOperand
 Evaluates an operand using a block to read registers.
 @param result A pointer whose target is to be filled with the result of
   evaluating the operand.
 @param operand The operand to be evaluated.
 @param regBlock The block to use when reading registers from the register
   state.
 @result 0 if the operand could be evaluated; -1 otherwise.
 */
int EDBlockEvaluateOperand(uint64_t *result,
                           EDOperandRef operand,
                           EDRegisterBlock_t regBlock);

/*!
 @function EDBlockVisitTokens
 Visits every token with a visitor.
 @param inst The instruction with the tokens to be visited.
 @param visitor The visitor.
 @result 0 if the visit ended normally; -1 if the visitor encountered an error
   or there was some other error.
 */
int EDBlockVisitTokens(EDInstRef inst,
                       EDTokenVisitor_t visitor);

/**
 * @}
 */

#endif

#ifdef __cplusplus
}
#endif

#endif
