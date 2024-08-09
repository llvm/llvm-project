//===--- DirectToIndirectFCR.h - RISC-V specific pass ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "lldb/lldb-types.h"

#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"

namespace lldb_private {

class ExecutionContext;

// During the lldb expression execution lldb wraps a user expression, jittes
// fabricated code and then puts it into the stack memory. Thus, if user tried
// to make a function call there will be a jump from a stack address to a code
// sections's address. RISC-V Architecture doesn't have a large code model yet
// and can make only a +-2GiB jumps, but in 64-bit architecture a distance
// between stack addresses and code sections's addresses is longer. Therefore,
// relocations resolver obtains an invalid address. To avoid such problem, this
// pass should be used. It replaces function calls with appropriate function's
// addresses explicitly. By doing so it removes relocations related to function
// calls. This pass should be cosidered as temprorary solution until a large
// code model will be approved.
class DirectToIndirectFCR : public llvm::FunctionPass {

  static bool canBeReplaced(const llvm::CallInst *ci);

  static std::vector<llvm::Value *>
  getFunctionArgsAsValues(const llvm::CallInst *ci);

  std::optional<lldb::addr_t>
  getFunctionAddress(const llvm::CallInst *ci) const;

  llvm::CallInst *getInstReplace(llvm::CallInst *ci) const;

public:
  static char ID;

  DirectToIndirectFCR(const ExecutionContext &exe_ctx);
  ~DirectToIndirectFCR() override;

  bool runOnFunction(llvm::Function &func) override;

  llvm::StringRef getPassName() const override;

private:
  const ExecutionContext &m_exe_ctx;
};

llvm::FunctionPass *createDirectToIndirectFCR(const ExecutionContext &exe_ctx);
} // namespace lldb_private
