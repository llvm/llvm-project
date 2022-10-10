//===- ReducerWorkItem.h - Wrapper for Module -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_REDUCE_REDUCERWORKITEM_H
#define LLVM_TOOLS_LLVM_REDUCE_REDUCERWORKITEM_H

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

class ReducerWorkItem {
public:
  std::shared_ptr<Module> M;
  std::unique_ptr<BitcodeLTOInfo> LTOInfo;
  std::unique_ptr<MachineModuleInfo> MMI;

  bool isMIR() const { return MMI != nullptr; }

  const Module &getModule() const { return *M; }

  void print(raw_ostream &ROS, void *p = nullptr) const;
  operator Module &() const { return *M; }

  /// Return a number to indicate whether there was any reduction progress.
  uint64_t getComplexityScore() const {
    return isMIR() ? computeMIRComplexityScore() : getIRSize();
  }

private:
  uint64_t computeMIRComplexityScore() const;
  uint64_t getIRSize() const;
};

std::unique_ptr<ReducerWorkItem>
parseReducerWorkItem(const char *ToolName, StringRef Filename,
                     LLVMContext &Ctxt, std::unique_ptr<TargetMachine> &TM,
                     bool IsMIR);

std::unique_ptr<ReducerWorkItem>
cloneReducerWorkItem(const ReducerWorkItem &MMM, const TargetMachine *TM);

bool verifyReducerWorkItem(const ReducerWorkItem &MMM, raw_fd_ostream *OS);

#endif
