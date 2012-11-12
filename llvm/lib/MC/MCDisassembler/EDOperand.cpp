//===-- EDOperand.cpp - LLVM Enhanced Disassembler ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Enhanced Disassembly library's operand class.  The
// operand is responsible for allowing evaluation given a particular register 
// context.
//
//===----------------------------------------------------------------------===//

#include "EDOperand.h"
#include "EDDisassembler.h"
#include "EDInst.h"
#include "llvm/MC/EDInstInfo.h"
#include "llvm/MC/MCInst.h"
using namespace llvm;

EDOperand::EDOperand(const EDDisassembler &disassembler,
                     const EDInst &inst,
                     unsigned int opIndex,
                     unsigned int &mcOpIndex) :
  Disassembler(disassembler),
  Inst(inst),
  OpIndex(opIndex),
  MCOpIndex(mcOpIndex) {
  unsigned int numMCOperands = 0;
    
  Triple::ArchType arch = Disassembler.TgtTriple.getArch();
    
  if (arch == Triple::x86 ||
      arch == Triple::x86_64) {
    uint8_t operandType = inst.ThisInstInfo->operandTypes[opIndex];
    
    switch (operandType) {
    default:
      break;
    case kOperandTypeImmediate:
      numMCOperands = 1;
      break;
    case kOperandTypeRegister:
      numMCOperands = 1;
      break;
    case kOperandTypeX86Memory:
      numMCOperands = 5;
      break;
    case kOperandTypeX86EffectiveAddress:
      numMCOperands = 4;
      break;
    case kOperandTypeX86PCRelative:
      numMCOperands = 1;
      break;
    }
  }
  else if (arch == Triple::arm ||
           arch == Triple::thumb) {
    uint8_t operandType = inst.ThisInstInfo->operandTypes[opIndex];
    
    switch (operandType) {
    default:
    case kOperandTypeARMRegisterList:
    case kOperandTypeARMDPRRegisterList:
    case kOperandTypeARMSPRRegisterList:
      break;
    case kOperandTypeImmediate:
    case kOperandTypeRegister:
    case kOperandTypeARMBranchTarget:
    case kOperandTypeARMSoImm:
    case kOperandTypeARMRotImm:
    case kOperandTypeThumb2SoImm:
    case kOperandTypeARMSoImm2Part:
    case kOperandTypeARMPredicate:
    case kOperandTypeThumbITMask:
    case kOperandTypeThumb2AddrModeImm8Offset:
    case kOperandTypeARMTBAddrMode:
    case kOperandTypeThumb2AddrModeImm8s4Offset:
    case kOperandTypeARMAddrMode7:
    case kOperandTypeThumb2AddrModeReg:
      numMCOperands = 1;
      break;
    case kOperandTypeThumb2SoReg:
    case kOperandTypeAddrModeImm12:
    case kOperandTypeARMAddrMode2Offset:
    case kOperandTypeARMAddrMode3Offset:
    case kOperandTypeARMAddrMode4:
    case kOperandTypeARMAddrMode5:
    case kOperandTypeARMAddrModePC:
    case kOperandTypeThumb2AddrModeImm8:
    case kOperandTypeThumb2AddrModeImm12:
    case kOperandTypeThumb2AddrModeImm8s4:
    case kOperandTypeThumbAddrModeImmS1:
    case kOperandTypeThumbAddrModeImmS2:
    case kOperandTypeThumbAddrModeImmS4:
    case kOperandTypeThumbAddrModeRR:
    case kOperandTypeThumbAddrModeSP:
    case kOperandTypeThumbAddrModePC:
      numMCOperands = 2;
      break;
    case kOperandTypeARMSoReg:
    case kOperandTypeLdStSOReg:
    case kOperandTypeARMAddrMode2:
    case kOperandTypeARMAddrMode3:
    case kOperandTypeThumb2AddrModeSoReg:
    case kOperandTypeThumbAddrModeRegS1:
    case kOperandTypeThumbAddrModeRegS2:
    case kOperandTypeThumbAddrModeRegS4:
    case kOperandTypeARMAddrMode6Offset:
      numMCOperands = 3;
      break;
    case kOperandTypeARMAddrMode6:
      numMCOperands = 4;
      break;
    }
  }
    
  mcOpIndex += numMCOperands;
}

EDOperand::~EDOperand() {
}

int EDOperand::evaluate(uint64_t &result,
                        EDRegisterReaderCallback callback,
                        void *arg) {
  uint8_t operandType = Inst.ThisInstInfo->operandTypes[OpIndex];
  
  Triple::ArchType arch = Disassembler.TgtTriple.getArch();
  
  switch (arch) {
  default:
    return -1;  
  case Triple::x86:
  case Triple::x86_64:    
    switch (operandType) {
    default:
      return -1;
    case kOperandTypeImmediate:
      result = Inst.Inst->getOperand(MCOpIndex).getImm();
      return 0;
    case kOperandTypeRegister:
    {
      unsigned reg = Inst.Inst->getOperand(MCOpIndex).getReg();
      return callback(&result, reg, arg);
    }
    case kOperandTypeX86PCRelative:
    {
      int64_t displacement = Inst.Inst->getOperand(MCOpIndex).getImm();
        
      uint64_t ripVal;
        
      // TODO fix how we do this
        
      if (callback(&ripVal, Disassembler.registerIDWithName("RIP"), arg))
        return -1;
        
      result = ripVal + displacement;
      return 0;
    }
    case kOperandTypeX86Memory:
    case kOperandTypeX86EffectiveAddress:  
    {
      unsigned baseReg = Inst.Inst->getOperand(MCOpIndex).getReg();
      uint64_t scaleAmount = Inst.Inst->getOperand(MCOpIndex+1).getImm();
      unsigned indexReg = Inst.Inst->getOperand(MCOpIndex+2).getReg();
      int64_t displacement = Inst.Inst->getOperand(MCOpIndex+3).getImm();
    
      uint64_t addr = 0;
        
      unsigned segmentReg = Inst.Inst->getOperand(MCOpIndex+4).getReg();
        
      if (segmentReg != 0 && arch == Triple::x86_64) {
        unsigned fsID = Disassembler.registerIDWithName("FS");
        unsigned gsID = Disassembler.registerIDWithName("GS");
        
        if (segmentReg == fsID ||
            segmentReg == gsID) {
          uint64_t segmentBase;
          if (!callback(&segmentBase, segmentReg, arg))
            addr += segmentBase;        
        }
      }
        
      if (baseReg) {
        uint64_t baseVal;
        if (callback(&baseVal, baseReg, arg))
          return -1;
        addr += baseVal;
      }
        
      if (indexReg) {
        uint64_t indexVal;
        if (callback(&indexVal, indexReg, arg))
          return -1;
        addr += (scaleAmount * indexVal);
      }
       
      addr += displacement;
       
      result = addr;
      return 0;
    }
    } // switch (operandType)
  case Triple::arm:
  case Triple::thumb:
    switch (operandType) {
    default:
      return -1;
    case kOperandTypeImmediate:
      if (!Inst.Inst->getOperand(MCOpIndex).isImm())
        return -1;
            
      result = Inst.Inst->getOperand(MCOpIndex).getImm();
      return 0;
    case kOperandTypeRegister:
    {
      if (!Inst.Inst->getOperand(MCOpIndex).isReg())
        return -1;
        
      unsigned reg = Inst.Inst->getOperand(MCOpIndex).getReg();
      return callback(&result, reg, arg);
    }
    case kOperandTypeARMBranchTarget:
    {
      if (!Inst.Inst->getOperand(MCOpIndex).isImm())
        return -1;
        
      int64_t displacement = Inst.Inst->getOperand(MCOpIndex).getImm();
      
      uint64_t pcVal;
      
      if (callback(&pcVal, Disassembler.registerIDWithName("PC"), arg))
        return -1;
      
      result = pcVal + displacement;
      return 0;
    }
    }
  }
}

int EDOperand::isRegister() {
  return(Inst.ThisInstInfo->operandFlags[OpIndex] == kOperandTypeRegister);
}

unsigned EDOperand::regVal() {
  return Inst.Inst->getOperand(MCOpIndex).getReg(); 
}

int EDOperand::isImmediate() {
  return(Inst.ThisInstInfo->operandFlags[OpIndex] == kOperandTypeImmediate);
}

uint64_t EDOperand::immediateVal() {
  return Inst.Inst->getOperand(MCOpIndex).getImm();
}

int EDOperand::isMemory() {
  uint8_t operandType = Inst.ThisInstInfo->operandTypes[OpIndex];
    
  switch (operandType) {
  default:
    return 0;
  case kOperandTypeX86Memory:
  case kOperandTypeX86PCRelative:
  case kOperandTypeX86EffectiveAddress:
  case kOperandTypeARMSoReg:
  case kOperandTypeARMSoImm:
  case kOperandTypeARMAddrMode2:
  case kOperandTypeARMAddrMode2Offset:
  case kOperandTypeARMAddrMode3:
  case kOperandTypeARMAddrMode3Offset:
  case kOperandTypeARMAddrMode4:
  case kOperandTypeARMAddrMode5:
  case kOperandTypeARMAddrMode6:
  case kOperandTypeARMAddrMode7:
  case kOperandTypeARMAddrModePC:
  case kOperandTypeARMBranchTarget:
  case kOperandTypeThumbAddrModeRegS1:
  case kOperandTypeThumbAddrModeRegS2:
  case kOperandTypeThumbAddrModeRegS4:
  case kOperandTypeThumbAddrModeRR:
  case kOperandTypeThumbAddrModeSP:
  case kOperandTypeThumb2SoImm:
  case kOperandTypeThumb2AddrModeImm8:
  case kOperandTypeThumb2AddrModeImm8Offset:
  case kOperandTypeThumb2AddrModeImm12:
  case kOperandTypeThumb2AddrModeSoReg:
  case kOperandTypeThumb2AddrModeImm8s4:
  case kOperandTypeThumb2AddrModeReg:
    return 1;
  }
}

#ifdef __BLOCKS__
namespace {
  struct RegisterReaderWrapper {
    EDOperand::EDRegisterBlock_t regBlock;
  };
}

static int readerWrapperCallback(uint64_t *value, unsigned regID, void *arg) {
  RegisterReaderWrapper *wrapper = (RegisterReaderWrapper *)arg;
  return wrapper->regBlock(value, regID);
}

int EDOperand::evaluate(uint64_t &result, EDRegisterBlock_t regBlock) {
  RegisterReaderWrapper wrapper;
  wrapper.regBlock = regBlock;
  return evaluate(result, readerWrapperCallback, (void*)&wrapper);
}
#endif
