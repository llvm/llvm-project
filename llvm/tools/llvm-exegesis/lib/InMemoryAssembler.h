//===-- InMemoryAssembler.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines classes to assemble functions composed of a single basic block of
/// MCInsts.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_INMEMORYASSEMBLER_H
#define LLVM_TOOLS_LLVM_EXEGESIS_INMEMORYASSEMBLER_H

#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/MC/MCInst.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace exegesis {

// Consumable context for JitFunction below.
// This temporary object allows for retrieving MachineFunction properties before
// assembling it.
class JitFunctionContext {
public:
  explicit JitFunctionContext(std::unique_ptr<llvm::LLVMTargetMachine> TM);
  // Movable
  JitFunctionContext(JitFunctionContext &&) = default;
  JitFunctionContext &operator=(JitFunctionContext &&) = default;
  // Non copyable
  JitFunctionContext(const JitFunctionContext &) = delete;
  JitFunctionContext &operator=(const JitFunctionContext &) = delete;

  const llvm::BitVector &getReservedRegs() const { return ReservedRegs; }

private:
  friend class JitFunction;

  std::unique_ptr<llvm::LLVMContext> Context;
  std::unique_ptr<llvm::LLVMTargetMachine> TM;
  std::unique_ptr<llvm::MachineModuleInfo> MMI;
  std::unique_ptr<llvm::Module> Module;
  llvm::MachineFunction *MF = nullptr;
  llvm::BitVector ReservedRegs;
};

// Creates a void() function from a sequence of llvm::MCInst.
class JitFunction {
public:
  // Assembles Instructions into an executable function.
  JitFunction(JitFunctionContext &&Context,
              llvm::ArrayRef<llvm::MCInst> Instructions);

  // Retrieves the function as an array of bytes.
  llvm::StringRef getFunctionBytes() const { return FunctionBytes; }

  // Retrieves the callable function.
  void operator()() const {
    char* const FnData = const_cast<char*>(FunctionBytes.data());
    ((void (*)())(intptr_t)FnData)();
  }

private:
  JitFunctionContext FunctionContext;
  std::unique_ptr<llvm::ExecutionEngine> ExecEngine;
  llvm::StringRef FunctionBytes;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_INMEMORYASSEMBLER_H
