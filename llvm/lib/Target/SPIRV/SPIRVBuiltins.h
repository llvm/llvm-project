//===-- SPIRVBuiltins.h - SPIR-V Built-in Functions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowering builtin function calls and types using their demangled names.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVBUILTINS_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVBUILTINS_H

#include "SPIRVGlobalRegistry.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"

namespace llvm {
namespace SPIRV {
/// Lowers a builtin funtion call using the provided \p DemangledCall skeleton
/// and external instruction \p Set.
///
/// \return the lowering success status if the called function is a recognized
/// builtin, std::nullopt otherwise.
///
/// \p DemangledCall is the skeleton of the lowered builtin function call.
/// \p Set is the external instruction set containing the given builtin.
/// \p OrigRet is the single original virtual return register if defined,
/// Register(0) otherwise.
/// \p OrigRetTy is the type of the \p OrigRet.
/// \p Args are the arguments of the lowered builtin call.
std::optional<bool> lowerBuiltin(const StringRef DemangledCall,
                                 InstructionSet::InstructionSet Set,
                                 MachineIRBuilder &MIRBuilder,
                                 const Register OrigRet, const Type *OrigRetTy,
                                 const SmallVectorImpl<Register> &Args,
                                 SPIRVGlobalRegistry *GR);

/// Translates a string representing a SPIR-V or OpenCL builtin type to a
/// TargetExtType that can be further lowered with lowerBuiltinType().
///
/// \return A TargetExtType representing the builtin SPIR-V type.
///
/// \p TypeName is the full string representation of the SPIR-V or OpenCL
/// builtin type.
const TargetExtType *
parseBuiltinTypeNameToTargetExtType(std::string TypeName,
                                    MachineIRBuilder &MIRBuilder);

/// Handles the translation of the provided special opaque/builtin type \p Type
/// to SPIR-V type. Generates the corresponding machine instructions for the
/// target type or gets the already existing OpType<...> register from the
/// global registry \p GR.
///
/// \return A machine instruction representing the OpType<...> SPIR-V type.
///
/// \p Type is the special opaque/builtin type to be lowered.
SPIRVType *lowerBuiltinType(const Type *Type,
                            AccessQualifier::AccessQualifier AccessQual,
                            MachineIRBuilder &MIRBuilder,
                            SPIRVGlobalRegistry *GR);
} // namespace SPIRV
} // namespace llvm
#endif // LLVM_LIB_TARGET_SPIRV_SPIRVBUILTINS_H
