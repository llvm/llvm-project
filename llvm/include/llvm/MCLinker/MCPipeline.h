//===- MCPipeline.h - Passes to run with MCLinker  --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_MCLINKER_MCPIPELINE_H
#define LLVM_MCLINKER_MCPIPELINE_H

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MCLinker/MCLinker.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
namespace mclinker {
/// Build a pipeline that does machine specific codgen but stops before
/// AsmPrint.
bool addPassesToEmitMC(llvm::TargetMachine &, llvm::legacy::PassManagerBase &,
                       bool,
                       llvm::MachineModuleInfoWrapperPass *, unsigned);

/// Build a pipeline that does AsmPrint only.
bool addPassesToAsmPrint(llvm::TargetMachine &, llvm::legacy::PassManagerBase &,
                         llvm::raw_pwrite_stream &, llvm::CodeGenFileType, bool,
                         llvm::MachineModuleInfoWrapperPass *,
                         llvm::SmallVectorImpl<MCInfo *> &);
} // namespace mclinker

} // namespace llvm

#endif
