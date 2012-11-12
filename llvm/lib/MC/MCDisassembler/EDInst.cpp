//===-EDInst.cpp - LLVM Enhanced Disassembler -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Enhanced Disassembly library's instruction class.
// The instruction is responsible for vending the string representation, 
// individual tokens, and operands for a single instruction.
//
//===----------------------------------------------------------------------===//

#include "EDInst.h"
#include "EDDisassembler.h"
#include "EDOperand.h"
#include "EDToken.h"

#include "llvm/MC/EDInstInfo.h"
#include "llvm/MC/MCInst.h"

using namespace llvm;

EDInst::EDInst(llvm::MCInst *inst,
               uint64_t byteSize, 
               EDDisassembler &disassembler,
               const llvm::EDInstInfo *info) :
  Disassembler(disassembler),
  Inst(inst),
  ThisInstInfo(info),
  ByteSize(byteSize),
  BranchTarget(-1),
  MoveSource(-1),
  MoveTarget(-1) {
  OperandOrder = ThisInstInfo->operandOrders[Disassembler.llvmSyntaxVariant()];
}

EDInst::~EDInst() {
  unsigned int index;
  unsigned int numOperands = Operands.size();
  
  for (index = 0; index < numOperands; ++index)
    delete Operands[index];
  
  unsigned int numTokens = Tokens.size();
  
  for (index = 0; index < numTokens; ++index)
    delete Tokens[index];
  
  delete Inst;
}

uint64_t EDInst::byteSize() {
  return ByteSize;
}

int EDInst::stringify() {
  if (StringifyResult.valid())
    return StringifyResult.result();
  
  if (Disassembler.printInst(String, *Inst))
    return StringifyResult.setResult(-1);

  String.push_back('\n');
  
  return StringifyResult.setResult(0);
}

int EDInst::getString(const char*& str) {
  if (stringify())
    return -1;
  
  str = String.c_str();
  
  return 0;
}

unsigned EDInst::instID() {
  return Inst->getOpcode();
}

bool EDInst::isBranch() {
  if (ThisInstInfo)
    return 
      ThisInstInfo->instructionType == kInstructionTypeBranch ||
      ThisInstInfo->instructionType == kInstructionTypeCall;
  else
    return false;
}

bool EDInst::isMove() {
  if (ThisInstInfo)
    return ThisInstInfo->instructionType == kInstructionTypeMove;
  else
    return false;
}

int EDInst::parseOperands() {
  if (ParseResult.valid())
    return ParseResult.result();
  
  if (!ThisInstInfo)
    return ParseResult.setResult(-1);
  
  unsigned int opIndex;
  unsigned int mcOpIndex = 0;
  
  for (opIndex = 0; opIndex < ThisInstInfo->numOperands; ++opIndex) {
    if (isBranch() &&
        (ThisInstInfo->operandFlags[opIndex] & kOperandFlagTarget)) {
      BranchTarget = opIndex;
    }
    else if (isMove()) {
      if (ThisInstInfo->operandFlags[opIndex] & kOperandFlagSource)
        MoveSource = opIndex;
      else if (ThisInstInfo->operandFlags[opIndex] & kOperandFlagTarget)
        MoveTarget = opIndex;
    }
    
    EDOperand *operand = new EDOperand(Disassembler, *this, opIndex, mcOpIndex);
    
    Operands.push_back(operand);
  }
  
  return ParseResult.setResult(0);
}

int EDInst::branchTargetID() {
  if (parseOperands())
    return -1;
  return BranchTarget;
}

int EDInst::moveSourceID() {
  if (parseOperands())
    return -1;
  return MoveSource;
}

int EDInst::moveTargetID() {
  if (parseOperands())
    return -1;
  return MoveTarget;
}

int EDInst::numOperands() {
  if (parseOperands())
    return -1;
  return Operands.size();
}

int EDInst::getOperand(EDOperand *&operand, unsigned int index) {
  if (parseOperands())
    return -1;
  
  if (index >= Operands.size())
    return -1;
  
  operand = Operands[index];
  return 0;
}

int EDInst::tokenize() {
  if (TokenizeResult.valid())
    return TokenizeResult.result();
    
  if (ThisInstInfo == NULL)
    return TokenizeResult.setResult(-1);
  
  if (stringify())
    return TokenizeResult.setResult(-1);
    
  return TokenizeResult.setResult(EDToken::tokenize(Tokens,
                                                    String,
                                                    OperandOrder,
                                                    Disassembler));
    
}

int EDInst::numTokens() {
  if (tokenize())
    return -1;
  return Tokens.size();
}

int EDInst::getToken(EDToken *&token, unsigned int index) {
  if (tokenize())
    return -1;
  token = Tokens[index];
  return 0;
}

#ifdef __BLOCKS__
int EDInst::visitTokens(EDTokenVisitor_t visitor) {
  if (tokenize())
    return -1;
  
  tokvec_t::iterator iter;
  
  for (iter = Tokens.begin(); iter != Tokens.end(); ++iter) {
    int ret = visitor(*iter);
    if (ret == 1)
      return 0;
    if (ret != 0)
      return -1;
  }
  
  return 0;
}
#endif
