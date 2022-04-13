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
} // namespace llvm

// Add the given string as a series of integer operand, inserting null
// terminators and padding to make sure the operands all have 32-bit
// little-endian words.
void addStringImm(const llvm::StringRef &Str, llvm::MachineInstrBuilder &MIB);
void addStringImm(const llvm::StringRef &Str, llvm::IRBuilder<> &B,
                  std::vector<llvm::Value *> &Args);

// Read the series of integer operands back as a null-terminated string using
// the reverse of the logic in addStringImm.
std::string getStringImm(const llvm::MachineInstr &MI, unsigned StartIndex);

// Add the given numerical immediate to MIB.
void addNumImm(const llvm::APInt &Imm, llvm::MachineInstrBuilder &MIB);

// Add an OpName instruction for the given target register.
void buildOpName(llvm::Register Target, const llvm::StringRef &Name,
                 llvm::MachineIRBuilder &MIRBuilder);

// Add an OpDecorate instruction for the given Reg.
void buildOpDecorate(llvm::Register Reg, llvm::MachineIRBuilder &MIRBuilder,
                     llvm::SPIRV::Decoration Dec,
                     const std::vector<uint32_t> &DecArgs,
                     llvm::StringRef StrImm = "");
void buildOpDecorate(llvm::Register Reg, llvm::MachineInstr &I,
                     const llvm::SPIRVInstrInfo &TII,
                     llvm::SPIRV::Decoration Dec,
                     const std::vector<uint32_t> &DecArgs,
                     llvm::StringRef StrImm = "");

// Convert a SPIR-V storage class to the corresponding LLVM IR address space.
unsigned storageClassToAddressSpace(llvm::SPIRV::StorageClass SC);

// Convert an LLVM IR address space to a SPIR-V storage class.
llvm::SPIRV::StorageClass addressSpaceToStorageClass(unsigned AddrSpace);

llvm::SPIRV::MemorySemantics
getMemSemanticsForStorageClass(llvm::SPIRV::StorageClass SC);
#endif // LLVM_LIB_TARGET_SPIRV_SPIRVUTILS_H
