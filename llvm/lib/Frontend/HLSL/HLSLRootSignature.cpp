//===- HLSLRootSignature.cpp - HLSL Root Signature helper objects ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helpers for working with HLSL Root Signatures.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/HLSLRootSignature.h"

namespace llvm {
namespace hlsl {
namespace rootsig {

void DescriptorTableClause::dump(raw_ostream &OS) const {
  OS << "Clause!";
}

} // namespace rootsig
} // namespace hlsl
} // namespace llvm
