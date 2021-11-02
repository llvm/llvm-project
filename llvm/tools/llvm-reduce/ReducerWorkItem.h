//===- ReducerWorkItem.h - Wrapper for Module and MachineFunction ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_REDUCE_REDUCERWORKITEM_H
#define LLVM_TOOLS_LLVM_REDUCE_REDUCERWORKITEM_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/Module.h"

using namespace llvm;

class ReducerWorkItem {
public:
  std::shared_ptr<Module> M;
  std::unique_ptr<MachineFunction> MF;
  void print(raw_ostream &ROS, void *p = nullptr) const;
  bool isMIR() { return MF != nullptr; }
  operator Module &() const { return *M; }
  operator MachineFunction &() const { return *MF; }
};

std::unique_ptr<ReducerWorkItem> parseReducerWorkItem(StringRef Filename,
                                                      LLVMContext &Ctxt,
                                                      MachineModuleInfo *MMI);

std::unique_ptr<ReducerWorkItem>
cloneReducerWorkItem(const ReducerWorkItem &MMM);

bool verifyReducerWorkItem(const ReducerWorkItem &MMM, raw_fd_ostream *OS);

#endif
