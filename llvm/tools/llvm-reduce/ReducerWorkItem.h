//===- ReducerWorkItem.h - Wrapper for Module -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_REDUCE_REDUCERWORKITEM_H
#define LLVM_TOOLS_LLVM_REDUCE_REDUCERWORKITEM_H

#include "llvm/IR/Module.h"
#include <memory>

namespace llvm {
class LLVMContext;
class MachineModuleInfo;
class raw_ostream;
class TargetMachine;
class TestRunner;
struct BitcodeLTOInfo;

class ReducerWorkItem {
public:
  std::shared_ptr<Module> M;
  std::unique_ptr<BitcodeLTOInfo> LTOInfo;
  std::unique_ptr<MachineModuleInfo> MMI;

  bool isMIR() const { return MMI != nullptr; }

  LLVMContext &getContext() {
    return M->getContext();
  }

  Module &getModule() { return *M; }
  const Module &getModule() const { return *M; }

  void print(raw_ostream &ROS, void *p = nullptr) const;
  operator Module &() const { return *M; }

  /// Return a number to indicate whether there was any reduction progress.
  uint64_t getComplexityScore() const {
    return isMIR() ? computeMIRComplexityScore() : computeIRComplexityScore();
  }

  ReducerWorkItem();
  ~ReducerWorkItem();
  ReducerWorkItem(ReducerWorkItem &) = delete;
  ReducerWorkItem(ReducerWorkItem &&) = default;

private:
  uint64_t computeIRComplexityScore() const;
  uint64_t computeMIRComplexityScore() const;
};
} // namespace llvm

std::pair<std::unique_ptr<llvm::ReducerWorkItem>, bool>
parseReducerWorkItem(llvm::StringRef ToolName, llvm::StringRef Filename,
                     llvm::LLVMContext &Ctxt,
                     std::unique_ptr<llvm::TargetMachine> &TM, bool IsMIR);

std::unique_ptr<llvm::ReducerWorkItem>
cloneReducerWorkItem(const llvm::ReducerWorkItem &MMM,
                     const llvm::TargetMachine *TM);

bool verifyReducerWorkItem(const llvm::ReducerWorkItem &MMM,
                           llvm::raw_fd_ostream *OS);

#endif
