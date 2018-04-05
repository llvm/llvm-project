//===-- X86.cpp --------------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "X86.h"

namespace exegesis {

static llvm::Error makeError(llvm::Twine Msg) {
  return llvm::make_error<llvm::StringError>(Msg,
                                             llvm::inconvertibleErrorCode());
}

X86Filter::~X86Filter() = default;

// Test whether we can generate a snippet for this instruction.
llvm::Error X86Filter::shouldRun(const LLVMState &State,
                                 const unsigned Opcode) const {
  const auto &InstrInfo = State.getInstrInfo();
  const llvm::MCInstrDesc &InstrDesc = InstrInfo.get(Opcode);
  if (InstrDesc.isBranch() || InstrDesc.isIndirectBranch())
    return makeError("Unsupported opcode: isBranch/isIndirectBranch");
  if (InstrDesc.isCall() || InstrDesc.isReturn())
    return makeError("Unsupported opcode: isCall/isReturn");
  const auto OpcodeName = InstrInfo.getName(Opcode);
  if (OpcodeName.startswith("POPF") || OpcodeName.startswith("PUSHF") ||
      OpcodeName.startswith("ADJCALLSTACK")) {
    return makeError("Unsupported opcode: Push/Pop/AdjCallStack");
  }
  return llvm::ErrorSuccess();
}

} // namespace exegesis
