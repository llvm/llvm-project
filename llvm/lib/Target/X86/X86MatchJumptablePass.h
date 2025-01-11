#ifndef LLVM_LIB_TARGET_X86_X86MATCHJUMPTABLEPASS_H
#define LLVM_LIB_TARGET_X86_X86MATCHJUMPTABLEPASS_H

#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

/// \brief Creates the X86MatchJumptablePass.
/// This pass analyzes and processes jump tables in X86 backend code generation.
FunctionPass *createX86MatchJumptablePass();

} // namespace llvm

#endif // LLVM_LIB_TARGET_X86_X86MATCHJUMPTABLEPASS_H
