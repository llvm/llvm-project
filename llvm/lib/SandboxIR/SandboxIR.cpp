//===- SandboxIR.cpp - A transactional overlay IR on top of LLVM IR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Support/Debug.h"
#include <sstream>

using namespace llvm::sandboxir;

Value::Value(ClassID SubclassID, llvm::Value *Val, Context &Ctx)
    : SubclassID(SubclassID), Val(Val), Ctx(Ctx) {
#ifndef NDEBUG
  UID = 0; // FIXME: Once SBContext is available.
#endif
}

#ifndef NDEBUG
std::string Value::getName() const {
  std::stringstream SS;
  SS << "SB" << UID << ".";
  return SS.str();
}

void Value::dumpCommonHeader(raw_ostream &OS) const {
  OS << getName() << " " << getSubclassIDStr(SubclassID) << " ";
}

void Value::dumpCommonFooter(raw_ostream &OS) const {
  OS.indent(2) << "Val: ";
  if (Val)
    OS << *Val;
  else
    OS << "NULL";
  OS << "\n";
}

void Value::dumpCommonPrefix(raw_ostream &OS) const {
  if (Val)
    OS << *Val;
  else
    OS << "NULL ";
}

void Value::dumpCommonSuffix(raw_ostream &OS) const {
  OS << " ; " << getName() << " (" << getSubclassIDStr(SubclassID) << ") "
     << this;
}

void Value::printAsOperandCommon(raw_ostream &OS) const {
  if (Val)
    Val->printAsOperand(OS);
  else
    OS << "NULL ";
}

void Argument::printAsOperand(raw_ostream &OS) const {
  printAsOperandCommon(OS);
}
void Argument::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}
void Argument::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

bool User::classof(const Value *From) {
  switch (From->getSubclassID()) {
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)                                                    \
  case ClassID::ID:                                                            \
    return true;
#define DEF_INSTR(ID, OPC, CLASS)                                              \
  case ClassID::ID:                                                            \
    return true;
#include "llvm/SandboxIR/SandboxIRValues.def"
  default:
    return false;
  }
}

#ifndef NDEBUG
void User::dumpCommonHeader(raw_ostream &OS) const {
  Value::dumpCommonHeader(OS);
  // TODO: This is incomplete
}
#endif // NDEBUG

const char *Instruction::getOpcodeName(Opcode Opc) {
  switch (Opc) {
#define DEF_VALUE(ID, CLASS)
#define DEF_USER(ID, CLASS)
#define OP(OPC)                                                                \
  case Opcode::OPC:                                                            \
    return #OPC;
#define DEF_INSTR(ID, OPC, CLASS) OPC
#include "llvm/SandboxIR/SandboxIRValues.def"
  }
}

bool Instruction::classof(const sandboxir::Value *From) {
  switch (From->getSubclassID()) {
#define DEF_INSTR(ID, OPC, CLASS)                                              \
  case ClassID::ID:                                                            \
    return true;
#include "llvm/SandboxIR/SandboxIRValues.def"
  default:
    return false;
  }
}

#ifndef NDEBUG
void Instruction::dump(raw_ostream &OS) const {
  OS << "Unimplemented! Please override dump().";
}
void Instruction::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

void OpaqueInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void OpaqueInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

void Constant::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void Constant::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

void Function::dumpNameAndArgs(raw_ostream &OS) const {
  auto *F = cast<llvm::Function>(Val);
  OS << *getType() << " @" << F->getName() << "(";
  auto NumArgs = F->arg_size();
  for (auto [Idx, Arg] : enumerate(F->args())) {
    auto *SBArg = cast_or_null<Argument>(Ctx.getValue(&Arg));
    if (SBArg == nullptr)
      OS << "NULL";
    else
      SBArg->printAsOperand(OS);
    if (Idx + 1 < NumArgs)
      OS << ", ";
  }
  OS << ")";
}
void Function::dump(raw_ostream &OS) const {
  dumpNameAndArgs(OS);
  OS << " {\n";
  OS << "}\n";
}
void Function::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Value *Context::getValue(llvm::Value *V) const {
  auto It = LLVMValueToValueMap.find(V);
  if (It != LLVMValueToValueMap.end())
    return It->second.get();
  return nullptr;
}
