//===- HLSLRootSignature.h - HLSL Root Signature helper objects -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects for working with HLSL Root
/// Signatures.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H
#define LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H

#include "llvm/Support/DXILABI.h"
#include <variant>

namespace llvm {
namespace hlsl {
namespace rootsig {

// Definitions of the in-memory data layout structures

// Models the different registers: bReg | tReg | uReg | sReg
enum class RegisterType { BReg, TReg, UReg, SReg };
struct Register {
  RegisterType ViewType;
  uint32_t Number;
};

// Models the end of a descriptor table and stores its visibility
struct DescriptorTable {
  uint32_t NumClauses = 0; // The number of clauses in the table
};

// Models DTClause : CBV | SRV | UAV | Sampler, by collecting like parameters
using ClauseType = llvm::dxil::ResourceClass;
struct DescriptorTableClause {
  ClauseType Type;
  Register Register;
  uint32_t NumDescriptors = 1;
  uint32_t Space = 0;
};

// Models RootElement : DescriptorTable | DescriptorTableClause
using RootElement = std::variant<DescriptorTable, DescriptorTableClause>;

// Models a reference to all assignment parameter types that any RootElement
// may have. Things of the form: Keyword = Param
using ParamType = std::variant<uint32_t *>;

} // namespace rootsig
} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLROOTSIGNATURE_H
