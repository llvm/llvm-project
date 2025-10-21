//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Command line options for MIR Flow Sensitive discriminators.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MIRFSDISCRIMINATOR_OPTIONS_H
#define LLVM_CODEGEN_MIRFSDISCRIMINATOR_OPTIONS_H

#include "llvm/Support/CommandLine.h"

namespace llvm {
extern cl::opt<bool> ImprovedFSDiscriminator;
} // namespace llvm

#endif // LLVM_CODEGEN_MIRFSDISCRIMINATOR_OPTIONS_H
