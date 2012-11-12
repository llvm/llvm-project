//===-- EDToken.cpp - LLVM Enhanced Disassembler --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Enhanced Disassembler library's token class.  The
// token is responsible for vending information about the token, such as its
// type and logical value.
//
//===----------------------------------------------------------------------===//

#include "EDToken.h"
#include "EDDisassembler.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/ADT/SmallVector.h"
using namespace llvm;

EDToken::EDToken(StringRef str,
                 enum tokenType type,
                 uint64_t localType,
                 EDDisassembler &disassembler) :
  Disassembler(disassembler),
  Str(str),
  Type(type),
  LocalType(localType),
  OperandID(-1) {
}

EDToken::~EDToken() {
}

void EDToken::makeLiteral(bool sign, uint64_t absoluteValue) {
  Type = kTokenLiteral;
  LiteralSign = sign;
  LiteralAbsoluteValue = absoluteValue;
}

void EDToken::makeRegister(unsigned registerID) {
  Type = kTokenRegister;
  RegisterID = registerID;
}

void EDToken::setOperandID(int operandID) {
  OperandID = operandID;
}

enum EDToken::tokenType EDToken::type() const {
  return Type;
}

uint64_t EDToken::localType() const {
  return LocalType;
}

StringRef EDToken::string() const {
  return Str;
}

int EDToken::operandID() const {
  return OperandID;
}

int EDToken::literalSign() const {
  if (Type != kTokenLiteral)
    return -1;
  return (LiteralSign ? 1 : 0);
}

int EDToken::literalAbsoluteValue(uint64_t &value) const {
  if (Type != kTokenLiteral)
    return -1;
  value = LiteralAbsoluteValue;
  return 0;
}

int EDToken::registerID(unsigned &registerID) const {
  if (Type != kTokenRegister)
    return -1;
  registerID = RegisterID;
  return 0;
}

int EDToken::tokenize(std::vector<EDToken*> &tokens,
                      std::string &str,
                      const signed char *operandOrder,
                      EDDisassembler &disassembler) {
  SmallVector<MCParsedAsmOperand*, 5> parsedOperands;
  SmallVector<AsmToken, 10> asmTokens;
  
  if (disassembler.parseInst(parsedOperands, asmTokens, str))
  {
    for (unsigned i = 0, e = parsedOperands.size(); i != e; ++i)
      delete parsedOperands[i];
    return -1;
  }
      
  SmallVectorImpl<MCParsedAsmOperand*>::iterator operandIterator;
  unsigned int operandIndex;
  SmallVectorImpl<AsmToken>::iterator tokenIterator;
  
  operandIterator = parsedOperands.begin();
  operandIndex = 0;
  
  bool readOpcode = false;
  
  const char *wsPointer = asmTokens.begin()->getLoc().getPointer();
  
  for (tokenIterator = asmTokens.begin();
       tokenIterator != asmTokens.end();
       ++tokenIterator) {
    SMLoc tokenLoc = tokenIterator->getLoc();
    
    const char *tokenPointer = tokenLoc.getPointer();
    
    if (tokenPointer > wsPointer) {
      unsigned long wsLength = tokenPointer - wsPointer;
      
      EDToken *whitespaceToken = new EDToken(StringRef(wsPointer, wsLength),
                                             EDToken::kTokenWhitespace,
                                             0,
                                             disassembler);
      
      tokens.push_back(whitespaceToken);
    }
    
    wsPointer = tokenPointer + tokenIterator->getString().size();
    
    while (operandIterator != parsedOperands.end() &&
           tokenLoc.getPointer() > 
           (*operandIterator)->getEndLoc().getPointer()) {
      ++operandIterator;
      ++operandIndex;
    }
    
    EDToken *token;
    
    switch (tokenIterator->getKind()) {
    case AsmToken::Identifier:
      if (!readOpcode) {
        token = new EDToken(tokenIterator->getString(),
                            EDToken::kTokenOpcode,
                            (uint64_t)tokenIterator->getKind(),
                            disassembler);
        readOpcode = true;
        break;
      }
      // any identifier that isn't an opcode is mere punctuation; so we fall
      // through
    default:
      token = new EDToken(tokenIterator->getString(),
                          EDToken::kTokenPunctuation,
                          (uint64_t)tokenIterator->getKind(),
                          disassembler);
      break;
    case AsmToken::Integer:
    {
      token = new EDToken(tokenIterator->getString(),
                          EDToken::kTokenLiteral,
                          (uint64_t)tokenIterator->getKind(),
                          disassembler);
        
      int64_t intVal = tokenIterator->getIntVal();
      
      if (intVal < 0)  
        token->makeLiteral(true, -intVal);
      else
        token->makeLiteral(false, intVal);
      break;
    }
    case AsmToken::Register:
    {
      token = new EDToken(tokenIterator->getString(),
                          EDToken::kTokenLiteral,
                          (uint64_t)tokenIterator->getKind(),
                          disassembler);
      
      token->makeRegister((unsigned)tokenIterator->getRegVal());
      break;
    }
    }
    
    if (operandIterator != parsedOperands.end() &&
       tokenLoc.getPointer() >= 
       (*operandIterator)->getStartLoc().getPointer()) {
      /// operandIndex == 0 means the operand is the instruction (which the
      /// AsmParser treats as an operand but edis does not).  We therefore skip
      /// operandIndex == 0 and subtract 1 from all other operand indices.
      
      if (operandIndex > 0)
        token->setOperandID(operandOrder[operandIndex - 1]);
    }
    
    tokens.push_back(token);
  }
  
  // Free any parsed operands.
  for (unsigned i = 0, e = parsedOperands.size(); i != e; ++i)
    delete parsedOperands[i];

  return 0;
}

int EDToken::getString(const char*& buf) {
  if (PermStr.length() == 0) {
    PermStr = Str.str();
  }
  buf = PermStr.c_str();
  return 0;
}
