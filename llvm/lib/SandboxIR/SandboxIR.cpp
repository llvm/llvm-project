//===- SandboxIR.cpp - A transactional overlay IR on top of LLVM IR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/Debug.h"
#include <sstream>

using namespace llvm::sandboxir;

Value *Use::get() const { return Ctx->getValue(LLVMUse->get()); }

unsigned Use::getOperandNo() const { return Usr->getUseOperandNo(*this); }

#ifndef NDEBUG
void Use::dump(raw_ostream &OS) const {
  Value *Def = nullptr;
  if (LLVMUse == nullptr)
    OS << "<null> LLVM Use! ";
  else
    Def = Ctx->getValue(LLVMUse->get());
  OS << "Def:  ";
  if (Def == nullptr)
    OS << "NULL";
  else
    OS << *Def;
  OS << "\n";

  OS << "User: ";
  if (Usr == nullptr)
    OS << "NULL";
  else
    OS << *Usr;
  OS << "\n";

  OS << "OperandNo: ";
  if (Usr == nullptr)
    OS << "N/A";
  else
    OS << getOperandNo();
  OS << "\n";
}

void Use::dump() const { dump(dbgs()); }
#endif // NDEBUG

Use OperandUseIterator::operator*() const { return Use; }

OperandUseIterator &OperandUseIterator::operator++() {
  assert(Use.LLVMUse != nullptr && "Already at end!");
  User *User = Use.getUser();
  Use = User->getOperandUseInternal(Use.getOperandNo() + 1, /*Verify=*/false);
  return *this;
}

UserUseIterator &UserUseIterator::operator++() {
  llvm::Use *&LLVMUse = Use.LLVMUse;
  assert(LLVMUse != nullptr && "Already at end!");
  LLVMUse = LLVMUse->getNext();
  if (LLVMUse == nullptr) {
    Use.Usr = nullptr;
    return *this;
  }
  auto *Ctx = Use.Ctx;
  auto *LLVMUser = LLVMUse->getUser();
  Use.Usr = cast_or_null<sandboxir::User>(Ctx->getValue(LLVMUser));
  return *this;
}

Value::Value(ClassID SubclassID, llvm::Value *Val, Context &Ctx)
    : SubclassID(SubclassID), Val(Val), Ctx(Ctx) {
#ifndef NDEBUG
  UID = Ctx.getNumValues();
#endif
}

Value::use_iterator Value::use_begin() {
  llvm::Use *LLVMUse = nullptr;
  if (Val->use_begin() != Val->use_end())
    LLVMUse = &*Val->use_begin();
  User *User = LLVMUse != nullptr ? cast_or_null<sandboxir::User>(Ctx.getValue(
                                        Val->use_begin()->getUser()))
                                  : nullptr;
  return use_iterator(Use(LLVMUse, User, Ctx));
}

Value::user_iterator Value::user_begin() {
  auto UseBegin = Val->use_begin();
  auto UseEnd = Val->use_end();
  bool AtEnd = UseBegin == UseEnd;
  llvm::Use *LLVMUse = AtEnd ? nullptr : &*UseBegin;
  User *User =
      AtEnd ? nullptr
            : cast_or_null<sandboxir::User>(Ctx.getValue(&*LLVMUse->getUser()));
  return user_iterator(Use(LLVMUse, User, Ctx), UseToUser());
}

unsigned Value::getNumUses() const { return range_size(Val->users()); }

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
  OS << " ; " << getName() << " (" << getSubclassIDStr(SubclassID) << ")";
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

Use User::getOperandUseDefault(unsigned OpIdx, bool Verify) const {
  assert((!Verify || OpIdx < getNumOperands()) && "Out of bounds!");
  assert(isa<llvm::User>(Val) && "Non-users have no operands!");
  llvm::Use *LLVMUse;
  if (OpIdx != getNumOperands())
    LLVMUse = &cast<llvm::User>(Val)->getOperandUse(OpIdx);
  else
    LLVMUse = cast<llvm::User>(Val)->op_end();
  return Use(LLVMUse, const_cast<User *>(this), Ctx);
}

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

BBIterator &BBIterator::operator++() {
  auto ItE = BB->end();
  assert(It != ItE && "Already at end!");
  ++It;
  if (It == ItE)
    return *this;
  Instruction &NextI = *cast<sandboxir::Instruction>(Ctx->getValue(&*It));
  unsigned Num = NextI.getNumOfIRInstrs();
  assert(Num > 0 && "Bad getNumOfIRInstrs()");
  It = std::next(It, Num - 1);
  return *this;
}

BBIterator &BBIterator::operator--() {
  assert(It != BB->begin() && "Already at begin!");
  if (It == BB->end()) {
    --It;
    return *this;
  }
  Instruction &CurrI = **this;
  unsigned Num = CurrI.getNumOfIRInstrs();
  assert(Num > 0 && "Bad getNumOfIRInstrs()");
  assert(std::prev(It, Num - 1) != BB->begin() && "Already at begin!");
  It = std::prev(It, Num);
  return *this;
}

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
  llvm_unreachable("Unknown Opcode");
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
  OS << *F->getReturnType() << " @" << F->getName() << "(";
  interleave(
      F->args(),
      [this, &OS](const llvm::Argument &LLVMArg) {
        auto *SBArg = cast_or_null<Argument>(Ctx.getValue(&LLVMArg));
        if (SBArg == nullptr)
          OS << "NULL";
        else
          SBArg->printAsOperand(OS);
      },
      [&] { OS << ", "; });
  OS << ")";
}
void Function::dump(raw_ostream &OS) const {
  dumpNameAndArgs(OS);
  OS << " {\n";
  auto *LLVMF = cast<llvm::Function>(Val);
  interleave(
      *LLVMF,
      [this, &OS](const llvm::BasicBlock &LLVMBB) {
        auto *BB = cast_or_null<BasicBlock>(Ctx.getValue(&LLVMBB));
        if (BB == nullptr)
          OS << "NULL";
        else
          OS << *BB;
      },
      [&OS] { OS << "\n"; });
  OS << "}\n";
}
void Function::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

BasicBlock::iterator::pointer
BasicBlock::iterator::getInstr(llvm::BasicBlock::iterator It) const {
  return cast_or_null<Instruction>(Ctx->getValue(&*It));
}

Value *Context::registerValue(std::unique_ptr<Value> &&VPtr) {
  assert(VPtr->getSubclassID() != Value::ClassID::User &&
         "Can't register a user!");
  Value *V = VPtr.get();
  llvm::Value *Key = V->Val;
  LLVMValueToValueMap[Key] = std::move(VPtr);
  return V;
}

Value *Context::getOrCreateValueInternal(llvm::Value *LLVMV, llvm::User *U) {
  auto Pair = LLVMValueToValueMap.insert({LLVMV, nullptr});
  auto It = Pair.first;
  if (!Pair.second)
    return It->second.get();

  if (auto *C = dyn_cast<llvm::Constant>(LLVMV)) {
    for (llvm::Value *COp : C->operands())
      getOrCreateValueInternal(COp, C);
    It->second = std::unique_ptr<Constant>(new Constant(C, *this));
    return It->second.get();
  }
  if (auto *Arg = dyn_cast<llvm::Argument>(LLVMV)) {
    It->second = std::unique_ptr<Argument>(new Argument(Arg, *this));
    return It->second.get();
  }
  if (auto *BB = dyn_cast<llvm::BasicBlock>(LLVMV)) {
    assert(isa<BlockAddress>(U) &&
           "This won't create a SBBB, don't call this function directly!");
    if (auto *SBBB = getValue(BB))
      return SBBB;
    return nullptr;
  }
  assert(isa<llvm::Instruction>(LLVMV) && "Expected Instruction");
  It->second = std::unique_ptr<OpaqueInst>(
      new OpaqueInst(cast<llvm::Instruction>(LLVMV), *this));
  return It->second.get();
}

BasicBlock *Context::createBasicBlock(llvm::BasicBlock *LLVMBB) {
  assert(getValue(LLVMBB) == nullptr && "Already exists!");
  auto NewBBPtr = std::unique_ptr<BasicBlock>(new BasicBlock(LLVMBB, *this));
  auto *BB = cast<BasicBlock>(registerValue(std::move(NewBBPtr)));
  // Create SandboxIR for BB's body.
  BB->buildBasicBlockFromLLVMIR(LLVMBB);
  return BB;
}

Value *Context::getValue(llvm::Value *V) const {
  auto It = LLVMValueToValueMap.find(V);
  if (It != LLVMValueToValueMap.end())
    return It->second.get();
  return nullptr;
}

Function *Context::createFunction(llvm::Function *F) {
  assert(getValue(F) == nullptr && "Already exists!");
  auto NewFPtr = std::unique_ptr<Function>(new Function(F, *this));
  // Create arguments.
  for (auto &Arg : F->args())
    getOrCreateArgument(&Arg);
  // Create BBs.
  for (auto &BB : *F)
    createBasicBlock(&BB);
  auto *SBF = cast<Function>(registerValue(std::move(NewFPtr)));
  return SBF;
}

Function *BasicBlock::getParent() const {
  auto *BB = cast<llvm::BasicBlock>(Val);
  auto *F = BB->getParent();
  if (F == nullptr)
    // Detached
    return nullptr;
  return cast_or_null<Function>(Ctx.getValue(F));
}

void BasicBlock::buildBasicBlockFromLLVMIR(llvm::BasicBlock *LLVMBB) {
  for (llvm::Instruction &IRef : reverse(*LLVMBB)) {
    llvm::Instruction *I = &IRef;
    Ctx.getOrCreateValue(I);
    for (auto [OpIdx, Op] : enumerate(I->operands())) {
      // Skip instruction's label operands
      if (isa<llvm::BasicBlock>(Op))
        continue;
      // Skip metadata
      if (isa<llvm::MetadataAsValue>(Op))
        continue;
      // Skip asm
      if (isa<llvm::InlineAsm>(Op))
        continue;
      Ctx.getOrCreateValue(Op);
    }
  }
#if !defined(NDEBUG) && defined(SBVEC_EXPENSIVE_CHECKS)
  verify();
#endif
}

BasicBlock::iterator BasicBlock::begin() const {
  llvm::BasicBlock *BB = cast<llvm::BasicBlock>(Val);
  llvm::BasicBlock::iterator It = BB->begin();
  if (!BB->empty()) {
    auto *V = Ctx.getValue(&*BB->begin());
    assert(V != nullptr && "No SandboxIR for BB->begin()!");
    auto *I = cast<Instruction>(V);
    unsigned Num = I->getNumOfIRInstrs();
    assert(Num >= 1u && "Bad getNumOfIRInstrs()");
    It = std::next(It, Num - 1);
  }
  return iterator(BB, It, &Ctx);
}

Instruction *BasicBlock::getTerminator() const {
  auto *TerminatorV =
      Ctx.getValue(cast<llvm::BasicBlock>(Val)->getTerminator());
  return cast_or_null<Instruction>(TerminatorV);
}

Instruction &BasicBlock::front() const {
  auto *BB = cast<llvm::BasicBlock>(Val);
  assert(!BB->empty() && "Empty block!");
  auto *SBI = cast<Instruction>(getContext().getValue(&*BB->begin()));
  assert(SBI != nullptr && "Expected Instr!");
  return *SBI;
}

Instruction &BasicBlock::back() const {
  auto *BB = cast<llvm::BasicBlock>(Val);
  assert(!BB->empty() && "Empty block!");
  auto *SBI = cast<Instruction>(getContext().getValue(&*BB->rbegin()));
  assert(SBI != nullptr && "Expected Instr!");
  return *SBI;
}

#ifndef NDEBUG
void BasicBlock::dump(raw_ostream &OS) const {
  llvm::BasicBlock *BB = cast<llvm::BasicBlock>(Val);
  const auto &Name = BB->getName();
  OS << Name;
  if (!Name.empty())
    OS << ":\n";
  // If there are Instructions in the BB that are not mapped to SandboxIR, then
  // use a crash-proof dump.
  if (any_of(*BB, [this](llvm::Instruction &I) {
        return Ctx.getValue(&I) == nullptr;
      })) {
    OS << "<Crash-proof mode!>\n";
    DenseSet<Instruction *> Visited;
    for (llvm::Instruction &IRef : *BB) {
      Value *SBV = Ctx.getValue(&IRef);
      if (SBV == nullptr)
        OS << IRef << " *** No SandboxIR ***\n";
      else {
        auto *SBI = dyn_cast<Instruction>(SBV);
        if (SBI == nullptr) {
          OS << IRef << " *** Not a SBInstruction!!! ***\n";
        } else {
          if (Visited.insert(SBI).second)
            OS << *SBI << "\n";
        }
      }
    }
  } else {
    for (auto &SBI : *this) {
      SBI.dump(OS);
      OS << "\n";
    }
  }
}
void BasicBlock::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG
