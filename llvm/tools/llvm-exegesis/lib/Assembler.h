//===-- Assembler.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines classes to assemble functions composed of a single basic block of
/// MCInsts.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_ASSEMBLER_H
#define LLVM_TOOLS_LLVM_EXEGESIS_ASSEMBLER_H

#include <memory>

#include "BenchmarkCode.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
namespace exegesis {

class ExegesisTarget;

// Gather the set of reserved registers (depends on function's calling
// convention and target machine).
llvm::BitVector getFunctionReservedRegs(const llvm::TargetMachine &TM);

// Creates a temporary `void foo(char*)` function containing the provided
// Instructions. Runs a set of llvm Passes to provide correct prologue and
// epilogue. Once the MachineFunction is ready, it is assembled for TM to
// AsmStream, the temporary function is eventually discarded.
void assembleToStream(const ExegesisTarget &ET,
                      std::unique_ptr<llvm::LLVMTargetMachine> TM,
                      llvm::ArrayRef<unsigned> LiveIns,
                      llvm::ArrayRef<RegisterValue> RegisterInitialValues,
                      llvm::ArrayRef<llvm::MCInst> Instructions,
                      llvm::raw_pwrite_stream &AsmStream);

// Creates an ObjectFile in the format understood by the host.
// Note: the resulting object keeps a copy of Buffer so it can be discarded once
// this function returns.
llvm::object::OwningBinary<llvm::object::ObjectFile>
getObjectFromBuffer(llvm::StringRef Buffer);

// Loads the content of Filename as on ObjectFile and returns it.
llvm::object::OwningBinary<llvm::object::ObjectFile>
getObjectFromFile(llvm::StringRef Filename);

// Consumes an ObjectFile containing a `void foo(char*)` function and make it
// executable.
struct ExecutableFunction {
  explicit ExecutableFunction(
      std::unique_ptr<llvm::LLVMTargetMachine> TM,
      llvm::object::OwningBinary<llvm::object::ObjectFile> &&ObjectFileHolder);

  // Retrieves the function as an array of bytes.
  llvm::StringRef getFunctionBytes() const { return FunctionBytes; }

  // Executes the function.
  void operator()(char *Memory) const {
    ((void (*)(char *))(intptr_t)FunctionBytes.data())(Memory);
  }

  std::unique_ptr<llvm::LLVMContext> Context;
  std::unique_ptr<llvm::ExecutionEngine> ExecEngine;
  llvm::StringRef FunctionBytes;
};

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_ASSEMBLER_H
