#ifndef LLVM_LIB_TARGET_X86_X86MATCHJUMPTABLEPASS_H
#define LLVM_LIB_TARGET_X86_X86MATCHJUMPTABLEPASS_H


#pragma once
#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

  FunctionPass *createX86MatchJumptablePass();

} // end namespace llvm

#endif // LLVM_LIB_TARGET_X86_X86MATCHJUMPTABLEPASS_H