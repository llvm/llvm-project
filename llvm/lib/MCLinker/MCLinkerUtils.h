//===- MCLinkerUtils.h - MCLinker utility Functions -------------*- C++ -*-===//
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

#ifndef LLVM_MCLINKER_MCLINKERUTILS_H
#define LLVM_MCLINKER_MCLINKERUTILS_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSymbolTableEntry.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
namespace mclinker {
// A few helper functions to access LLVM private class/struct members:
// http://bloglitb.blogspot.com/2010/07/access-to-private-members-thats-easy.html

/// Wrapping accessing LLVM data structure's private filed accessor for
/// linking at MC-level where a few things need to be globalized such as:
/// - llvm::MachineFunction's numbering,
/// - all unique_ptrs of llvm::MachineFunctions in each split to be put
///   together for the final AsmPrint
/// - MCSymbol propagation for external global symbols to each split's
///   MCContext to avoid duplicates for X86's OrcJIT execution engine.

/// Get private field
/// DenseMap<const Function*, std::unique_ptr<MachineFunction>> MachineFunctions
/// from llvm::MachineModuleInfo.
llvm::DenseMap<const llvm::Function *, std::unique_ptr<llvm::MachineFunction>> &
getMachineFunctionsFromMachineModuleInfo(llvm::MachineModuleInfo &);

/// Set private field FunctionNumber in llvm::MachineFunction.
void setMachineFunctionNumber(llvm::MachineFunction &, unsigned);

/// Set private field NextFnNum in llvm::MachineModuleInfo.
void setNextFnNum(llvm::MachineModuleInfo &, unsigned);

/// Call private member function
/// MCSymbolTableEntry &getSymbolTableEntry(StringRef Name)
/// from llvm::MCContext.
llvm::MCSymbolTableEntry &getMCContextSymbolTableEntry(llvm::StringRef,
                                                       llvm::MCContext &);

/// Release MCSubTargetInfo.
void releaseTargetMachineConstants(llvm::TargetMachine &);

/// Clear SubtargetMap in SubtargetInfo.
void resetSubtargetInfo(llvm::TargetMachine &, llvm::MachineModuleInfo &);

} // namespace mclinker
} // namespace llvm

#endif
