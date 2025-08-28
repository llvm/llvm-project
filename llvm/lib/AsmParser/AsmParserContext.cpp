//===-- AsmParserContext.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/AsmParserContext.h"

namespace llvm {

std::optional<FileLocRange>
AsmParserContext::getFunctionLocation(const Function *F) const {
  if (!Functions.contains(F))
    return std::nullopt;
  return Functions.at(F);
}

std::optional<FileLocRange>
AsmParserContext::getBlockLocation(const BasicBlock *BB) const {
  if (!Blocks.contains(BB))
    return std::nullopt;
  return Blocks.at(BB);
}

std::optional<FileLocRange>
AsmParserContext::getInstructionLocation(const Instruction *I) const {
  if (!Instructions.contains(I))
    return std::nullopt;
  return Instructions.at(I);
}

std::optional<Function *>
AsmParserContext::getFunctionAtLocation(const FileLocRange &Query) const {
  for (auto &[F, Loc] : Functions) {
    if (Loc.contains(Query))
      return F;
  }
  return std::nullopt;
}

std::optional<Function *>
AsmParserContext::getFunctionAtLocation(const FileLoc &Query) const {
  return getFunctionAtLocation(FileLocRange(Query, Query));
}

std::optional<BasicBlock *>
AsmParserContext::getBlockAtLocation(const FileLocRange &Query) const {
  for (auto &[BB, Loc] : Blocks) {
    if (Loc.contains(Query))
      return BB;
  }
  return std::nullopt;
}

std::optional<BasicBlock *>
AsmParserContext::getBlockAtLocation(const FileLoc &Query) const {
  return getBlockAtLocation(FileLocRange(Query, Query));
}

std::optional<Instruction *>
AsmParserContext::getInstructionAtLocation(const FileLocRange &Query) const {
  for (auto &[I, Loc] : Instructions) {
    if (Loc.contains(Query))
      return I;
  }
  return std::nullopt;
}

std::optional<Instruction *>
AsmParserContext::getInstructionAtLocation(const FileLoc &Query) const {
  return getInstructionAtLocation(FileLocRange(Query, Query));
}

bool AsmParserContext::addFunctionLocation(Function *F,
                                           const FileLocRange &Loc) {
  return Functions.insert({F, Loc}).second;
}

bool AsmParserContext::addBlockLocation(BasicBlock *BB,
                                        const FileLocRange &Loc) {
  return Blocks.insert({BB, Loc}).second;
}

bool AsmParserContext::addInstructionLocation(Instruction *I,
                                              const FileLocRange &Loc) {
  return Instructions.insert({I, Loc}).second;
}

} // namespace llvm
