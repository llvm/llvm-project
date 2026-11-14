//===-- MCTargetOptionsCommandFlags.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains machine code-specific flags that are shared between
// different command line tools.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCTARGETOPTIONSCOMMANDFLAGS_H
#define LLVM_MC_MCTARGETOPTIONSCOMMANDFLAGS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class MCTargetOptions;
class StringRef;

namespace cl {
class LibraryOptions;
class LibraryOptionsContext;
}

namespace mc {

LLVM_ABI StringRef getABIName(const cl::LibraryOptionsContext &C);

/// The options that a tool reading getMCTargetOptions passes to
/// cl::parseCommandLineOptions.
LLVM_ABI ArrayRef<const cl::LibraryOptions *> getToolOptions();

LLVM_ABI MCTargetOptions getMCTargetOptions(const cl::LibraryOptionsContext &C);

} // namespace mc

} // namespace llvm

#endif
