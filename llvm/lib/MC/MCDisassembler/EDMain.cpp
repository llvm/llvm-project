//===-- EDMain.cpp - LLVM Enhanced Disassembly C API ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the enhanced disassembler's public C API.
//
//===----------------------------------------------------------------------===//

#include "EDDisassembler.h"
#include "EDInst.h"
#include "EDOperand.h"
#include "EDToken.h"
#include "llvm-c/EnhancedDisassembly.h"
using namespace llvm;

int EDGetDisassembler(EDDisassemblerRef *disassembler,
                      const char *triple,
                      EDAssemblySyntax_t syntax) {
  EDDisassembler::AssemblySyntax Syntax;
  switch (syntax) {
  default: llvm_unreachable("Unknown assembly syntax!");
  case kEDAssemblySyntaxX86Intel:
    Syntax = EDDisassembler::kEDAssemblySyntaxX86Intel;
    break;
  case kEDAssemblySyntaxX86ATT:
    Syntax = EDDisassembler::kEDAssemblySyntaxX86ATT;
    break;
  case kEDAssemblySyntaxARMUAL:
    Syntax = EDDisassembler::kEDAssemblySyntaxARMUAL;
    break;
  }

  EDDisassemblerRef ret = EDDisassembler::getDisassembler(triple, Syntax);

  if (!ret)
    return -1;
  *disassembler = ret;
  return 0;
}

int EDGetRegisterName(const char** regName,
                      EDDisassemblerRef disassembler,
                      unsigned regID) {
  const char *name = ((EDDisassembler*)disassembler)->nameWithRegisterID(regID);
  if (!name)
    return -1;
  *regName = name;
  return 0;
}

int EDRegisterIsStackPointer(EDDisassemblerRef disassembler,
                             unsigned regID) {
  return ((EDDisassembler*)disassembler)->registerIsStackPointer(regID) ? 1 : 0;
}

int EDRegisterIsProgramCounter(EDDisassemblerRef disassembler,
                               unsigned regID) {
  return ((EDDisassembler*)disassembler)->registerIsProgramCounter(regID) ? 1:0;
}

unsigned int EDCreateInsts(EDInstRef *insts,
                           unsigned int count,
                           EDDisassemblerRef disassembler,
                           ::EDByteReaderCallback byteReader,
                           uint64_t address,
                           void *arg) {
  unsigned int index;

  for (index = 0; index < count; ++index) {
    EDInst *inst = ((EDDisassembler*)disassembler)->createInst(byteReader,
                                                               address, arg);

    if (!inst)
      return index;

    insts[index] = inst;
    address += inst->byteSize();
  }

  return count;
}

void EDReleaseInst(EDInstRef inst) {
  delete ((EDInst*)inst);
}

int EDInstByteSize(EDInstRef inst) {
  return ((EDInst*)inst)->byteSize();
}

int EDGetInstString(const char **buf,
                    EDInstRef inst) {
  return ((EDInst*)inst)->getString(*buf);
}

int EDInstID(unsigned *instID, EDInstRef inst) {
  *instID = ((EDInst*)inst)->instID();
  return 0;
}

int EDInstIsBranch(EDInstRef inst) {
  return ((EDInst*)inst)->isBranch();
}

int EDInstIsMove(EDInstRef inst) {
  return ((EDInst*)inst)->isMove();
}

int EDBranchTargetID(EDInstRef inst) {
  return ((EDInst*)inst)->branchTargetID();
}

int EDMoveSourceID(EDInstRef inst) {
  return ((EDInst*)inst)->moveSourceID();
}

int EDMoveTargetID(EDInstRef inst) {
  return ((EDInst*)inst)->moveTargetID();
}

int EDNumTokens(EDInstRef inst) {
  return ((EDInst*)inst)->numTokens();
}

int EDGetToken(EDTokenRef *token,
               EDInstRef inst,
               int index) {
  return ((EDInst*)inst)->getToken(*(EDToken**)token, index);
}

int EDGetTokenString(const char **buf,
                     EDTokenRef token) {
  return ((EDToken*)token)->getString(*buf);
}

int EDOperandIndexForToken(EDTokenRef token) {
  return ((EDToken*)token)->operandID();
}

int EDTokenIsWhitespace(EDTokenRef token) {
  return ((EDToken*)token)->type() == EDToken::kTokenWhitespace;
}

int EDTokenIsPunctuation(EDTokenRef token) {
  return ((EDToken*)token)->type() == EDToken::kTokenPunctuation;
}

int EDTokenIsOpcode(EDTokenRef token) {
  return ((EDToken*)token)->type() == EDToken::kTokenOpcode;
}

int EDTokenIsLiteral(EDTokenRef token) {
  return ((EDToken*)token)->type() == EDToken::kTokenLiteral;
}

int EDTokenIsRegister(EDTokenRef token) {
  return ((EDToken*)token)->type() == EDToken::kTokenRegister;
}

int EDTokenIsNegativeLiteral(EDTokenRef token) {
  if (((EDToken*)token)->type() != EDToken::kTokenLiteral)
    return -1;

  return ((EDToken*)token)->literalSign();
}

int EDLiteralTokenAbsoluteValue(uint64_t *value, EDTokenRef token) {
  if (((EDToken*)token)->type() != EDToken::kTokenLiteral)
    return -1;

  return ((EDToken*)token)->literalAbsoluteValue(*value);
}

int EDRegisterTokenValue(unsigned *registerID,
                         EDTokenRef token) {
  if (((EDToken*)token)->type() != EDToken::kTokenRegister)
    return -1;

  return ((EDToken*)token)->registerID(*registerID);
}

int EDNumOperands(EDInstRef inst) {
  return ((EDInst*)inst)->numOperands();
}

int EDGetOperand(EDOperandRef *operand,
                 EDInstRef inst,
                 int index) {
  return ((EDInst*)inst)->getOperand(*(EDOperand**)operand, index);
}

int EDOperandIsRegister(EDOperandRef operand) {
  return ((EDOperand*)operand)->isRegister();
}

int EDOperandIsImmediate(EDOperandRef operand) {
  return ((EDOperand*)operand)->isImmediate();
}

int EDOperandIsMemory(EDOperandRef operand) {
  return ((EDOperand*)operand)->isMemory();
}

int EDRegisterOperandValue(unsigned *value, EDOperandRef operand) {
  if (!((EDOperand*)operand)->isRegister())
    return -1;
  *value = ((EDOperand*)operand)->regVal();
  return 0;
}

int EDImmediateOperandValue(uint64_t *value, EDOperandRef operand) {
  if (!((EDOperand*)operand)->isImmediate())
    return -1;
  *value = ((EDOperand*)operand)->immediateVal();
  return 0;
}

int EDEvaluateOperand(uint64_t *result, EDOperandRef operand,
                      ::EDRegisterReaderCallback regReader, void *arg) {
  return ((EDOperand*)operand)->evaluate(*result, regReader, arg);
}

#ifdef __BLOCKS__

struct ByteReaderWrapper {
  EDByteBlock_t byteBlock;
};

static int readerWrapperCallback(uint8_t *byte,
                          uint64_t address,
                          void *arg) {
  struct ByteReaderWrapper *wrapper = (struct ByteReaderWrapper *)arg;
  return wrapper->byteBlock(byte, address);
}

unsigned int EDBlockCreateInsts(EDInstRef *insts,
                                int count,
                                EDDisassemblerRef disassembler,
                                EDByteBlock_t byteBlock,
                                uint64_t address) {
  struct ByteReaderWrapper wrapper;
  wrapper.byteBlock = byteBlock;

  return EDCreateInsts(insts, count, disassembler, readerWrapperCallback,
                       address, (void*)&wrapper);
}

int EDBlockEvaluateOperand(uint64_t *result, EDOperandRef operand,
                           EDRegisterBlock_t regBlock) {
  return ((EDOperand*)operand)->evaluate(*result, regBlock);
}

int EDBlockVisitTokens(EDInstRef inst, ::EDTokenVisitor_t visitor) {
  return ((EDInst*)inst)->visitTokens((llvm::EDTokenVisitor_t)visitor);
}

#else

extern "C" unsigned int EDBlockCreateInsts() {
  return 0;
}

extern "C" int EDBlockEvaluateOperand() {
  return -1;
}

extern "C" int EDBlockVisitTokens() {
  return -1;
}

#endif
