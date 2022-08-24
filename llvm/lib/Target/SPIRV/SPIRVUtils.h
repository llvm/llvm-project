//===--- SPIRVUtils.h ---- SPIR-V Utility Functions -------------*- C++ -*-===//
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

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVUTILS_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVUTILS_H

#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "llvm/IR/IRBuilder.h"
#include <string>

namespace llvm {
class MCInst;
class MachineFunction;
class MachineInstr;
class MachineInstrBuilder;
class MachineIRBuilder;
class MachineRegisterInfo;
class Register;
class StringRef;
class SPIRVInstrInfo;

// Add the given string as a series of integer operand, inserting null
// terminators and padding to make sure the operands all have 32-bit
// little-endian words.
void addStringImm(const StringRef &Str, MCInst &Inst);
void addStringImm(const StringRef &Str, MachineInstrBuilder &MIB);
void addStringImm(const StringRef &Str, IRBuilder<> &B,
                  std::vector<Value *> &Args);

// Read the series of integer operands back as a null-terminated string using
// the reverse of the logic in addStringImm.
std::string getStringImm(const MachineInstr &MI, unsigned StartIndex);

// Add the given numerical immediate to MIB.
void addNumImm(const APInt &Imm, MachineInstrBuilder &MIB);

// Add an OpName instruction for the given target register.
void buildOpName(Register Target, const StringRef &Name,
                 MachineIRBuilder &MIRBuilder);

// Add an OpDecorate instruction for the given Reg.
void buildOpDecorate(Register Reg, MachineIRBuilder &MIRBuilder,
                     SPIRV::Decoration::Decoration Dec,
                     const std::vector<uint32_t> &DecArgs,
                     StringRef StrImm = "");
void buildOpDecorate(Register Reg, MachineInstr &I, const SPIRVInstrInfo &TII,
                     SPIRV::Decoration::Decoration Dec,
                     const std::vector<uint32_t> &DecArgs,
                     StringRef StrImm = "");

// Convert a SPIR-V storage class to the corresponding LLVM IR address space.
unsigned storageClassToAddressSpace(SPIRV::StorageClass::StorageClass SC);

// Convert an LLVM IR address space to a SPIR-V storage class.
SPIRV::StorageClass::StorageClass
addressSpaceToStorageClass(unsigned AddrSpace);

SPIRV::MemorySemantics::MemorySemantics
getMemSemanticsForStorageClass(SPIRV::StorageClass::StorageClass SC);

SPIRV::MemorySemantics::MemorySemantics getMemSemantics(AtomicOrdering Ord);

// Find def instruction for the given ConstReg, walking through
// spv_track_constant and ASSIGN_TYPE instructions. Updates ConstReg by def
// of OpConstant instruction.
MachineInstr *getDefInstrMaybeConstant(Register &ConstReg,
                                       const MachineRegisterInfo *MRI);

// Get constant integer value of the given ConstReg.
uint64_t getIConstVal(Register ConstReg, const MachineRegisterInfo *MRI);

// Check if MI is a SPIR-V specific intrinsic call.
bool isSpvIntrinsic(MachineInstr &MI, Intrinsic::ID IntrinsicID);

// Get type of i-th operand of the metadata node.
Type *getMDOperandAsType(const MDNode *N, unsigned I);

// Return a demangled name with arg type info by itaniumDemangle().
// If the parser fails, return only function name.
std::string mayBeOclOrSpirvBuiltin(StringRef Name);
} // namespace llvm
#endif // LLVM_LIB_TARGET_SPIRV_SPIRVUTILS_H
