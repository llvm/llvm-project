//===--- SPIRVUtils.cpp ---- SPIR-V Utility Functions -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains miscellaneous utility functions.
//
//===----------------------------------------------------------------------===//

#include "SPIRVUtils.h"
#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "SPIRV.h"
#include "SPIRVInstrInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/IntrinsicsSPIRV.h"

using namespace llvm;

// The following functions are used to add these string literals as a series of
// 32-bit integer operands with the correct format, and unpack them if necessary
// when making string comparisons in compiler passes.
// SPIR-V requires null-terminated UTF-8 strings padded to 32-bit alignment.
static uint32_t convertCharsToWord(const StringRef &Str, unsigned i) {
  uint32_t Word = 0u; // Build up this 32-bit word from 4 8-bit chars.
  for (unsigned WordIndex = 0; WordIndex < 4; ++WordIndex) {
    unsigned StrIndex = i + WordIndex;
    uint8_t CharToAdd = 0;       // Initilize char as padding/null.
    if (StrIndex < Str.size()) { // If it's within the string, get a real char.
      CharToAdd = Str[StrIndex];
    }
    Word |= (CharToAdd << (WordIndex * 8));
  }
  return Word;
}

// Get length including padding and null terminator.
static size_t getPaddedLen(const StringRef &Str) {
  const size_t Len = Str.size() + 1;
  return (Len % 4 == 0) ? Len : Len + (4 - (Len % 4));
}

void addStringImm(const StringRef &Str, MCInst &Inst) {
  const size_t PaddedLen = getPaddedLen(Str);
  for (unsigned i = 0; i < PaddedLen; i += 4) {
    // Add an operand for the 32-bits of chars or padding.
    Inst.addOperand(MCOperand::createImm(convertCharsToWord(Str, i)));
  }
}

void addStringImm(const StringRef &Str, MachineInstrBuilder &MIB) {
  const size_t PaddedLen = getPaddedLen(Str);
  for (unsigned i = 0; i < PaddedLen; i += 4) {
    // Add an operand for the 32-bits of chars or padding.
    MIB.addImm(convertCharsToWord(Str, i));
  }
}

void addStringImm(const StringRef &Str, IRBuilder<> &B,
                  std::vector<Value *> &Args) {
  const size_t PaddedLen = getPaddedLen(Str);
  for (unsigned i = 0; i < PaddedLen; i += 4) {
    // Add a vector element for the 32-bits of chars or padding.
    Args.push_back(B.getInt32(convertCharsToWord(Str, i)));
  }
}

std::string getStringImm(const MachineInstr &MI, unsigned StartIndex) {
  return getSPIRVStringOperand(MI, StartIndex);
}

void addNumImm(const APInt &Imm, MachineInstrBuilder &MIB) {
  const auto Bitwidth = Imm.getBitWidth();
  switch (Bitwidth) {
  case 1:
    break; // Already handled.
  case 8:
  case 16:
  case 32:
    MIB.addImm(Imm.getZExtValue());
    break;
  case 64: {
    uint64_t FullImm = Imm.getZExtValue();
    uint32_t LowBits = FullImm & 0xffffffff;
    uint32_t HighBits = (FullImm >> 32) & 0xffffffff;
    MIB.addImm(LowBits).addImm(HighBits);
    break;
  }
  default:
    report_fatal_error("Unsupported constant bitwidth");
  }
}

void buildOpName(Register Target, const StringRef &Name,
                 MachineIRBuilder &MIRBuilder) {
  if (!Name.empty()) {
    auto MIB = MIRBuilder.buildInstr(SPIRV::OpName).addUse(Target);
    addStringImm(Name, MIB);
  }
}

static void finishBuildOpDecorate(MachineInstrBuilder &MIB,
                                  const std::vector<uint32_t> &DecArgs,
                                  StringRef StrImm) {
  if (!StrImm.empty())
    addStringImm(StrImm, MIB);
  for (const auto &DecArg : DecArgs)
    MIB.addImm(DecArg);
}

void buildOpDecorate(Register Reg, MachineIRBuilder &MIRBuilder,
                     llvm::SPIRV::Decoration Dec,
                     const std::vector<uint32_t> &DecArgs, StringRef StrImm) {
  auto MIB = MIRBuilder.buildInstr(SPIRV::OpDecorate)
                 .addUse(Reg)
                 .addImm(static_cast<uint32_t>(Dec));
  finishBuildOpDecorate(MIB, DecArgs, StrImm);
}

void buildOpDecorate(Register Reg, MachineInstr &I, const SPIRVInstrInfo &TII,
                     llvm::SPIRV::Decoration Dec,
                     const std::vector<uint32_t> &DecArgs, StringRef StrImm) {
  MachineBasicBlock &MBB = *I.getParent();
  auto MIB = BuildMI(MBB, I, I.getDebugLoc(), TII.get(SPIRV::OpDecorate))
                 .addUse(Reg)
                 .addImm(static_cast<uint32_t>(Dec));
  finishBuildOpDecorate(MIB, DecArgs, StrImm);
}

// TODO: maybe the following two functions should be handled in the subtarget
// to allow for different OpenCL vs Vulkan handling.
unsigned storageClassToAddressSpace(SPIRV::StorageClass SC) {
  switch (SC) {
  case SPIRV::StorageClass::Function:
    return 0;
  case SPIRV::StorageClass::CrossWorkgroup:
    return 1;
  case SPIRV::StorageClass::UniformConstant:
    return 2;
  case SPIRV::StorageClass::Workgroup:
    return 3;
  case SPIRV::StorageClass::Generic:
    return 4;
  case SPIRV::StorageClass::Input:
    return 7;
  default:
    llvm_unreachable("Unable to get address space id");
  }
}

SPIRV::StorageClass addressSpaceToStorageClass(unsigned AddrSpace) {
  switch (AddrSpace) {
  case 0:
    return SPIRV::StorageClass::Function;
  case 1:
    return SPIRV::StorageClass::CrossWorkgroup;
  case 2:
    return SPIRV::StorageClass::UniformConstant;
  case 3:
    return SPIRV::StorageClass::Workgroup;
  case 4:
    return SPIRV::StorageClass::Generic;
  case 7:
    return SPIRV::StorageClass::Input;
  default:
    llvm_unreachable("Unknown address space");
  }
}

SPIRV::MemorySemantics getMemSemanticsForStorageClass(SPIRV::StorageClass SC) {
  switch (SC) {
  case SPIRV::StorageClass::StorageBuffer:
  case SPIRV::StorageClass::Uniform:
    return SPIRV::MemorySemantics::UniformMemory;
  case SPIRV::StorageClass::Workgroup:
    return SPIRV::MemorySemantics::WorkgroupMemory;
  case SPIRV::StorageClass::CrossWorkgroup:
    return SPIRV::MemorySemantics::CrossWorkgroupMemory;
  case SPIRV::StorageClass::AtomicCounter:
    return SPIRV::MemorySemantics::AtomicCounterMemory;
  case SPIRV::StorageClass::Image:
    return SPIRV::MemorySemantics::ImageMemory;
  default:
    return SPIRV::MemorySemantics::None;
  }
}

SPIRV::MemorySemantics getMemSemantics(AtomicOrdering Ord) {
  switch (Ord) {
  case AtomicOrdering::Acquire:
    return SPIRV::MemorySemantics::Acquire;
  case AtomicOrdering::Release:
    return SPIRV::MemorySemantics::Release;
  case AtomicOrdering::AcquireRelease:
    return SPIRV::MemorySemantics::AcquireRelease;
  case AtomicOrdering::SequentiallyConsistent:
    return SPIRV::MemorySemantics::SequentiallyConsistent;
  case AtomicOrdering::Unordered:
  case AtomicOrdering::Monotonic:
  case AtomicOrdering::NotAtomic:
  default:
    return SPIRV::MemorySemantics::None;
  }
}

MachineInstr *getDefInstrMaybeConstant(Register &ConstReg,
                                       const MachineRegisterInfo *MRI) {
  MachineInstr *ConstInstr = MRI->getVRegDef(ConstReg);
  if (ConstInstr->getOpcode() == TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS &&
      ConstInstr->getIntrinsicID() == Intrinsic::spv_track_constant) {
    ConstReg = ConstInstr->getOperand(2).getReg();
    ConstInstr = MRI->getVRegDef(ConstReg);
  } else if (ConstInstr->getOpcode() == SPIRV::ASSIGN_TYPE) {
    ConstReg = ConstInstr->getOperand(1).getReg();
    ConstInstr = MRI->getVRegDef(ConstReg);
  }
  return ConstInstr;
}

uint64_t getIConstVal(Register ConstReg, const MachineRegisterInfo *MRI) {
  const MachineInstr *MI = getDefInstrMaybeConstant(ConstReg, MRI);
  assert(MI && MI->getOpcode() == TargetOpcode::G_CONSTANT);
  return MI->getOperand(1).getCImm()->getValue().getZExtValue();
}

bool isSpvIntrinsic(MachineInstr &MI, Intrinsic::ID IntrinsicID) {
  return MI.getOpcode() == TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS &&
         MI.getIntrinsicID() == IntrinsicID;
}

Type *getMDOperandAsType(const MDNode *N, unsigned I) {
  return cast<ValueAsMetadata>(N->getOperand(I))->getType();
}
