//===- Module.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Module.h"
#include "llvm/SandboxIR/Constant.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Value.h"

using namespace llvm::sandboxir;

Function *Module::getFunction(StringRef Name) const {
  llvm::Function *LLVMF = LLVMM.getFunction(Name);
  return cast_or_null<Function>(Ctx.getValue(LLVMF));
}

GlobalVariable *Module::getGlobalVariable(StringRef Name,
                                          bool AllowInternal) const {
  return cast_or_null<GlobalVariable>(
      Ctx.getValue(LLVMM.getGlobalVariable(Name, AllowInternal)));
}

GlobalAlias *Module::getNamedAlias(StringRef Name) const {
  return cast_or_null<GlobalAlias>(Ctx.getValue(LLVMM.getNamedAlias(Name)));
}

GlobalIFunc *Module::getNamedIFunc(StringRef Name) const {
  return cast_or_null<GlobalIFunc>(Ctx.getValue(LLVMM.getNamedIFunc(Name)));
}

#ifndef NDEBUG
void Module::dumpOS(raw_ostream &OS) const { OS << LLVMM; }

void Module::dump() const {
  dumpOS(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG
