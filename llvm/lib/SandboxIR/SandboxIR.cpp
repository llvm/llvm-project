//===- SandboxIR.cpp - A transactional overlay IR on top of LLVM IR -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/Debug.h"
#include <sstream>

using namespace llvm::sandboxir;

Value *Use::get() const { return Ctx->getValue(LLVMUse->get()); }

void Use::set(Value *V) {
  auto &Tracker = Ctx->getTracker();
  if (Tracker.isTracking())
    Tracker.track(std::make_unique<UseSet>(*this, Tracker));
  LLVMUse->set(V->Val);
}

unsigned Use::getOperandNo() const { return Usr->getUseOperandNo(*this); }

void Use::swap(Use &OtherUse) {
  auto &Tracker = Ctx->getTracker();
  if (Tracker.isTracking())
    Tracker.track(std::make_unique<UseSwap>(*this, OtherUse, Tracker));
  LLVMUse->swap(*OtherUse.LLVMUse);
}

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
  // Get the corresponding llvm::Use, get the next in the list, and update the
  // sandboxir::Use.
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

OperandUseIterator OperandUseIterator::operator+(unsigned Num) const {
  sandboxir::Use U = Use.getUser()->getOperandUseInternal(
      Use.getOperandNo() + Num, /*Verify=*/true);
  return OperandUseIterator(U);
}

OperandUseIterator OperandUseIterator::operator-(unsigned Num) const {
  assert(Use.getOperandNo() >= Num && "Out of bounds!");
  sandboxir::Use U = Use.getUser()->getOperandUseInternal(
      Use.getOperandNo() - Num, /*Verify=*/true);
  return OperandUseIterator(U);
}

int OperandUseIterator::operator-(const OperandUseIterator &Other) const {
  int ThisOpNo = Use.getOperandNo();
  int OtherOpNo = Other.Use.getOperandNo();
  return ThisOpNo - OtherOpNo;
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

void Value::replaceUsesWithIf(
    Value *OtherV, llvm::function_ref<bool(const Use &)> ShouldReplace) {
  assert(getType() == OtherV->getType() && "Can't replace with different type");
  llvm::Value *OtherVal = OtherV->Val;
  // We are delegating RUWIf to LLVM IR's RUWIf.
  Val->replaceUsesWithIf(
      OtherVal, [&ShouldReplace, this](llvm::Use &LLVMUse) -> bool {
        User *DstU = cast_or_null<User>(Ctx.getValue(LLVMUse.getUser()));
        if (DstU == nullptr)
          return false;
        Use UseToReplace(&LLVMUse, DstU, Ctx);
        if (!ShouldReplace(UseToReplace))
          return false;
        auto &Tracker = Ctx.getTracker();
        if (Tracker.isTracking())
          Tracker.track(std::make_unique<UseSet>(UseToReplace, Tracker));
        return true;
      });
}

void Value::replaceAllUsesWith(Value *Other) {
  assert(getType() == Other->getType() &&
         "Replacing with Value of different type!");
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking()) {
    for (auto Use : uses())
      Tracker.track(std::make_unique<UseSet>(Use, Tracker));
  }
  // We are delegating RAUW to LLVM IR's RAUW.
  Val->replaceAllUsesWith(Other->Val);
}

#ifndef NDEBUG
std::string Value::getUid() const {
  std::stringstream SS;
  SS << "SB" << UID << ".";
  return SS.str();
}

void Value::dumpCommonHeader(raw_ostream &OS) const {
  OS << getUid() << " " << getSubclassIDStr(SubclassID) << " ";
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
  OS << " ; " << getUid() << " (" << getSubclassIDStr(SubclassID) << ")";
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

#ifndef NDEBUG
void User::verifyUserOfLLVMUse(const llvm::Use &Use) const {
  assert(Ctx.getValue(Use.getUser()) == this &&
         "Use not found in this SBUser's operands!");
}
#endif

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

void User::setOperand(unsigned OperandIdx, Value *Operand) {
  assert(isa<llvm::User>(Val) && "No operands!");
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking())
    Tracker.track(std::make_unique<UseSet>(getOperandUse(OperandIdx), Tracker));
  // We are delegating to llvm::User::setOperand().
  cast<llvm::User>(Val)->setOperand(OperandIdx, Operand->Val);
}

bool User::replaceUsesOfWith(Value *FromV, Value *ToV) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking()) {
    for (auto OpIdx : seq<unsigned>(0, getNumOperands())) {
      auto Use = getOperandUse(OpIdx);
      if (Use.get() == FromV)
        Tracker.track(std::make_unique<UseSet>(Use, Tracker));
    }
  }
  // We are delegating RUOW to LLVM IR's RUOW.
  return cast<llvm::User>(Val)->replaceUsesOfWith(FromV->Val, ToV->Val);
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
#define OP(OPC)                                                                \
  case Opcode::OPC:                                                            \
    return #OPC;
#define OPCODES(...) __VA_ARGS__
#define DEF_INSTR(ID, OPC, CLASS) OPC
#include "llvm/SandboxIR/SandboxIRValues.def"
  }
  llvm_unreachable("Unknown Opcode");
}

llvm::Instruction *Instruction::getTopmostLLVMInstruction() const {
  Instruction *Prev = getPrevNode();
  if (Prev == nullptr) {
    // If at top of the BB, return the first BB instruction.
    return &*cast<llvm::BasicBlock>(getParent()->Val)->begin();
  }
  // Else get the Previous sandbox IR instruction's bottom IR instruction and
  // return its successor.
  llvm::Instruction *PrevBotI = cast<llvm::Instruction>(Prev->Val);
  return PrevBotI->getNextNode();
}

BBIterator Instruction::getIterator() const {
  auto *I = cast<llvm::Instruction>(Val);
  return BasicBlock::iterator(I->getParent(), I->getIterator(), &Ctx);
}

Instruction *Instruction::getNextNode() const {
  assert(getParent() != nullptr && "Detached!");
  assert(getIterator() != getParent()->end() && "Already at end!");
  // `Val` is the bottom-most LLVM IR instruction. Get the next in the chain,
  // and get the corresponding sandboxir Instruction that maps to it. This works
  // even for SandboxIR Instructions that map to more than one LLVM Instruction.
  auto *LLVMI = cast<llvm::Instruction>(Val);
  assert(LLVMI->getParent() != nullptr && "LLVM IR instr is detached!");
  auto *NextLLVMI = LLVMI->getNextNode();
  auto *NextI = cast_or_null<Instruction>(Ctx.getValue(NextLLVMI));
  if (NextI == nullptr)
    return nullptr;
  return NextI;
}

Instruction *Instruction::getPrevNode() const {
  assert(getParent() != nullptr && "Detached!");
  auto It = getIterator();
  if (It != getParent()->begin())
    return std::prev(getIterator()).get();
  return nullptr;
}

void Instruction::removeFromParent() {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking())
    Tracker.track(std::make_unique<RemoveFromParent>(this, Tracker));

  // Detach all the LLVM IR instructions from their parent BB.
  for (llvm::Instruction *I : getLLVMInstrs())
    I->removeFromParent();
}

void Instruction::eraseFromParent() {
  assert(users().empty() && "Still connected to users, can't erase!");
  std::unique_ptr<Value> Detached = Ctx.detach(this);
  auto LLVMInstrs = getLLVMInstrs();

  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking()) {
    Tracker.track(
        std::make_unique<EraseFromParent>(std::move(Detached), Tracker));
    // We don't actually delete the IR instruction, because then it would be
    // impossible to bring it back from the dead at the same memory location.
    // Instead we remove it from its BB and track its current location.
    for (llvm::Instruction *I : LLVMInstrs)
      I->removeFromParent();
    // TODO: Multi-instructions need special treatment because some of the
    // references are internal to the instruction.
    for (llvm::Instruction *I : LLVMInstrs)
      I->dropAllReferences();
  } else {
    // Erase in reverse to avoid erasing nstructions with attached uses.
    for (llvm::Instruction *I : reverse(LLVMInstrs))
      I->eraseFromParent();
  }
}

void Instruction::moveBefore(BasicBlock &BB, const BBIterator &WhereIt) {
  if (std::next(getIterator()) == WhereIt)
    // Destination is same as origin, nothing to do.
    return;

  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking())
    Tracker.track(std::make_unique<MoveInstr>(this, Tracker));

  auto *LLVMBB = cast<llvm::BasicBlock>(BB.Val);
  llvm::BasicBlock::iterator It;
  if (WhereIt == BB.end()) {
    It = LLVMBB->end();
  } else {
    Instruction *WhereI = &*WhereIt;
    It = WhereI->getTopmostLLVMInstruction()->getIterator();
  }
  // TODO: Move this to the verifier of sandboxir::Instruction.
  assert(is_sorted(getLLVMInstrs(),
                   [](auto *I1, auto *I2) { return I1->comesBefore(I2); }) &&
         "Expected program order!");
  // Do the actual move in LLVM IR.
  for (auto *I : getLLVMInstrs())
    I->moveBefore(*LLVMBB, It);
}

void Instruction::insertBefore(Instruction *BeforeI) {
  llvm::Instruction *BeforeTopI = BeforeI->getTopmostLLVMInstruction();
  // TODO: Move this to the verifier of sandboxir::Instruction.
  assert(is_sorted(getLLVMInstrs(),
                   [](auto *I1, auto *I2) { return I1->comesBefore(I2); }) &&
         "Expected program order!");

  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking())
    Tracker.track(std::make_unique<InsertIntoBB>(this, Tracker));

  // Insert the LLVM IR Instructions in program order.
  for (llvm::Instruction *I : getLLVMInstrs())
    I->insertBefore(BeforeTopI);
}

void Instruction::insertAfter(Instruction *AfterI) {
  insertInto(AfterI->getParent(), std::next(AfterI->getIterator()));
}

void Instruction::insertInto(BasicBlock *BB, const BBIterator &WhereIt) {
  llvm::BasicBlock *LLVMBB = cast<llvm::BasicBlock>(BB->Val);
  llvm::Instruction *LLVMBeforeI;
  llvm::BasicBlock::iterator LLVMBeforeIt;
  Instruction *BeforeI;
  if (WhereIt != BB->end()) {
    BeforeI = &*WhereIt;
    LLVMBeforeI = BeforeI->getTopmostLLVMInstruction();
    LLVMBeforeIt = LLVMBeforeI->getIterator();
  } else {
    BeforeI = nullptr;
    LLVMBeforeI = nullptr;
    LLVMBeforeIt = LLVMBB->end();
  }

  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking())
    Tracker.track(std::make_unique<InsertIntoBB>(this, Tracker));

  // Insert the LLVM IR Instructions in program order.
  for (llvm::Instruction *I : getLLVMInstrs())
    I->insertInto(LLVMBB, LLVMBeforeIt);
}

BasicBlock *Instruction::getParent() const {
  // Get the LLVM IR Instruction that this maps to, get its parent, and get the
  // corresponding sandboxir::BasicBlock by looking it up in sandboxir::Context.
  auto *BB = cast<llvm::Instruction>(Val)->getParent();
  if (BB == nullptr)
    return nullptr;
  return cast<BasicBlock>(Ctx.getValue(BB));
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
#endif // NDEBUG

Value *SelectInst::createCommon(Value *Cond, Value *True, Value *False,
                                const Twine &Name, IRBuilder<> &Builder,
                                Context &Ctx) {
  llvm::Value *NewV =
      Builder.CreateSelect(Cond->Val, True->Val, False->Val, Name);
  if (auto *NewSI = dyn_cast<llvm::SelectInst>(NewV))
    return Ctx.createSelectInst(NewSI);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *SelectInst::create(Value *Cond, Value *True, Value *False,
                          Instruction *InsertBefore, Context &Ctx,
                          const Twine &Name) {
  llvm::Instruction *BeforeIR = InsertBefore->getTopmostLLVMInstruction();
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  return createCommon(Cond, True, False, Name, Builder, Ctx);
}

Value *SelectInst::create(Value *Cond, Value *True, Value *False,
                          BasicBlock *InsertAtEnd, Context &Ctx,
                          const Twine &Name) {
  auto *IRInsertAtEnd = cast<llvm::BasicBlock>(InsertAtEnd->Val);
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(IRInsertAtEnd);
  return createCommon(Cond, True, False, Name, Builder, Ctx);
}

bool SelectInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Select;
}

#ifndef NDEBUG
void SelectInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void SelectInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

BranchInst *BranchInst::create(BasicBlock *IfTrue, Instruction *InsertBefore,
                               Context &Ctx) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  llvm::Instruction *LLVMBefore = InsertBefore->getTopmostLLVMInstruction();
  Builder.SetInsertPoint(cast<llvm::Instruction>(LLVMBefore));
  llvm::BranchInst *NewBr =
      Builder.CreateBr(cast<llvm::BasicBlock>(IfTrue->Val));
  return Ctx.createBranchInst(NewBr);
}

BranchInst *BranchInst::create(BasicBlock *IfTrue, BasicBlock *InsertAtEnd,
                               Context &Ctx) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(cast<llvm::BasicBlock>(InsertAtEnd->Val));
  llvm::BranchInst *NewBr =
      Builder.CreateBr(cast<llvm::BasicBlock>(IfTrue->Val));
  return Ctx.createBranchInst(NewBr);
}

BranchInst *BranchInst::create(BasicBlock *IfTrue, BasicBlock *IfFalse,
                               Value *Cond, Instruction *InsertBefore,
                               Context &Ctx) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  llvm::Instruction *LLVMBefore = InsertBefore->getTopmostLLVMInstruction();
  Builder.SetInsertPoint(LLVMBefore);
  llvm::BranchInst *NewBr =
      Builder.CreateCondBr(Cond->Val, cast<llvm::BasicBlock>(IfTrue->Val),
                           cast<llvm::BasicBlock>(IfFalse->Val));
  return Ctx.createBranchInst(NewBr);
}

BranchInst *BranchInst::create(BasicBlock *IfTrue, BasicBlock *IfFalse,
                               Value *Cond, BasicBlock *InsertAtEnd,
                               Context &Ctx) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(cast<llvm::BasicBlock>(InsertAtEnd->Val));
  llvm::BranchInst *NewBr =
      Builder.CreateCondBr(Cond->Val, cast<llvm::BasicBlock>(IfTrue->Val),
                           cast<llvm::BasicBlock>(IfFalse->Val));
  return Ctx.createBranchInst(NewBr);
}

bool BranchInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Br;
}

Value *BranchInst::getCondition() const {
  assert(isConditional() && "Cannot get condition of an uncond branch!");
  return Ctx.getValue(cast<llvm::BranchInst>(Val)->getCondition());
}

BasicBlock *BranchInst::getSuccessor(unsigned SuccIdx) const {
  assert(SuccIdx < getNumSuccessors() &&
         "Successor # out of range for Branch!");
  return cast_or_null<BasicBlock>(
      Ctx.getValue(cast<llvm::BranchInst>(Val)->getSuccessor(SuccIdx)));
}

void BranchInst::setSuccessor(unsigned Idx, BasicBlock *NewSucc) {
  assert((Idx == 0 || Idx == 1) && "Out of bounds!");
  setOperand(2u - Idx, NewSucc);
}

BasicBlock *BranchInst::LLVMBBToSBBB::operator()(llvm::BasicBlock *BB) const {
  return cast<BasicBlock>(Ctx.getValue(BB));
}
const BasicBlock *
BranchInst::ConstLLVMBBToSBBB::operator()(const llvm::BasicBlock *BB) const {
  return cast<BasicBlock>(Ctx.getValue(BB));
}
#ifndef NDEBUG
void BranchInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}
void BranchInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

void LoadInst::setVolatile(bool V) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking()) {
    Tracker.track(std::make_unique<
                  GenericSetter<&LoadInst::isVolatile, &LoadInst::setVolatile>>(
        this, Tracker));
  }
  cast<llvm::LoadInst>(Val)->setVolatile(V);
}

LoadInst *LoadInst::create(Type *Ty, Value *Ptr, MaybeAlign Align,
                           Instruction *InsertBefore, Context &Ctx,
                           const Twine &Name) {
  return create(Ty, Ptr, Align, InsertBefore, /*IsVolatile=*/false, Ctx, Name);
}

LoadInst *LoadInst::create(Type *Ty, Value *Ptr, MaybeAlign Align,
                           Instruction *InsertBefore, bool IsVolatile,
                           Context &Ctx, const Twine &Name) {
  llvm::Instruction *BeforeIR = InsertBefore->getTopmostLLVMInstruction();
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  auto *NewLI =
      Builder.CreateAlignedLoad(Ty, Ptr->Val, Align, IsVolatile, Name);
  auto *NewSBI = Ctx.createLoadInst(NewLI);
  return NewSBI;
}

LoadInst *LoadInst::create(Type *Ty, Value *Ptr, MaybeAlign Align,
                           BasicBlock *InsertAtEnd, Context &Ctx,
                           const Twine &Name) {
  return create(Ty, Ptr, Align, InsertAtEnd, /*IsVolatile=*/false, Ctx, Name);
}

LoadInst *LoadInst::create(Type *Ty, Value *Ptr, MaybeAlign Align,
                           BasicBlock *InsertAtEnd, bool IsVolatile,
                           Context &Ctx, const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(cast<llvm::BasicBlock>(InsertAtEnd->Val));
  auto *NewLI =
      Builder.CreateAlignedLoad(Ty, Ptr->Val, Align, IsVolatile, Name);
  auto *NewSBI = Ctx.createLoadInst(NewLI);
  return NewSBI;
}

bool LoadInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Load;
}

Value *LoadInst::getPointerOperand() const {
  return Ctx.getValue(cast<llvm::LoadInst>(Val)->getPointerOperand());
}

#ifndef NDEBUG
void LoadInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void LoadInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

void StoreInst::setVolatile(bool V) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking()) {
    Tracker.track(
        std::make_unique<
            GenericSetter<&StoreInst::isVolatile, &StoreInst::setVolatile>>(
            this, Tracker));
  }
  cast<llvm::StoreInst>(Val)->setVolatile(V);
}

StoreInst *StoreInst::create(Value *V, Value *Ptr, MaybeAlign Align,
                             Instruction *InsertBefore, Context &Ctx) {
  return create(V, Ptr, Align, InsertBefore, /*IsVolatile=*/false, Ctx);
}

StoreInst *StoreInst::create(Value *V, Value *Ptr, MaybeAlign Align,
                             Instruction *InsertBefore, bool IsVolatile,
                             Context &Ctx) {
  llvm::Instruction *BeforeIR = InsertBefore->getTopmostLLVMInstruction();
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  auto *NewSI = Builder.CreateAlignedStore(V->Val, Ptr->Val, Align, IsVolatile);
  auto *NewSBI = Ctx.createStoreInst(NewSI);
  return NewSBI;
}

StoreInst *StoreInst::create(Value *V, Value *Ptr, MaybeAlign Align,
                             BasicBlock *InsertAtEnd, Context &Ctx) {
  return create(V, Ptr, Align, InsertAtEnd, /*IsVolatile=*/false, Ctx);
}

StoreInst *StoreInst::create(Value *V, Value *Ptr, MaybeAlign Align,
                             BasicBlock *InsertAtEnd, bool IsVolatile,
                             Context &Ctx) {
  auto *InsertAtEndIR = cast<llvm::BasicBlock>(InsertAtEnd->Val);
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertAtEndIR);
  auto *NewSI = Builder.CreateAlignedStore(V->Val, Ptr->Val, Align, IsVolatile);
  auto *NewSBI = Ctx.createStoreInst(NewSI);
  return NewSBI;
}

bool StoreInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Store;
}

Value *StoreInst::getValueOperand() const {
  return Ctx.getValue(cast<llvm::StoreInst>(Val)->getValueOperand());
}

Value *StoreInst::getPointerOperand() const {
  return Ctx.getValue(cast<llvm::StoreInst>(Val)->getPointerOperand());
}

#ifndef NDEBUG
void StoreInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void StoreInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

UnreachableInst *UnreachableInst::create(Instruction *InsertBefore,
                                         Context &Ctx) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  llvm::Instruction *LLVMBefore = InsertBefore->getTopmostLLVMInstruction();
  Builder.SetInsertPoint(LLVMBefore);
  llvm::UnreachableInst *NewUI = Builder.CreateUnreachable();
  return Ctx.createUnreachableInst(NewUI);
}

UnreachableInst *UnreachableInst::create(BasicBlock *InsertAtEnd,
                                         Context &Ctx) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(cast<llvm::BasicBlock>(InsertAtEnd->Val));
  llvm::UnreachableInst *NewUI = Builder.CreateUnreachable();
  return Ctx.createUnreachableInst(NewUI);
}

bool UnreachableInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Unreachable;
}

#ifndef NDEBUG
void UnreachableInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void UnreachableInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

ReturnInst *ReturnInst::createCommon(Value *RetVal, IRBuilder<> &Builder,
                                     Context &Ctx) {
  llvm::ReturnInst *NewRI;
  if (RetVal != nullptr)
    NewRI = Builder.CreateRet(RetVal->Val);
  else
    NewRI = Builder.CreateRetVoid();
  return Ctx.createReturnInst(NewRI);
}

ReturnInst *ReturnInst::create(Value *RetVal, Instruction *InsertBefore,
                               Context &Ctx) {
  llvm::Instruction *BeforeIR = InsertBefore->getTopmostLLVMInstruction();
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(BeforeIR);
  return createCommon(RetVal, Builder, Ctx);
}

ReturnInst *ReturnInst::create(Value *RetVal, BasicBlock *InsertAtEnd,
                               Context &Ctx) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(cast<llvm::BasicBlock>(InsertAtEnd->Val));
  return createCommon(RetVal, Builder, Ctx);
}

Value *ReturnInst::getReturnValue() const {
  auto *LLVMRetVal = cast<llvm::ReturnInst>(Val)->getReturnValue();
  return LLVMRetVal != nullptr ? Ctx.getValue(LLVMRetVal) : nullptr;
}

#ifndef NDEBUG
void ReturnInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void ReturnInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Value *CallBase::getCalledOperand() const {
  return Ctx.getValue(cast<llvm::CallBase>(Val)->getCalledOperand());
}

Use CallBase::getCalledOperandUse() const {
  llvm::Use *LLVMUse = &cast<llvm::CallBase>(Val)->getCalledOperandUse();
  return Use(LLVMUse, cast<User>(Ctx.getValue(LLVMUse->getUser())), Ctx);
}

Function *CallBase::getCalledFunction() const {
  return cast_or_null<Function>(
      Ctx.getValue(cast<llvm::CallBase>(Val)->getCalledFunction()));
}
Function *CallBase::getCaller() {
  return cast<Function>(Ctx.getValue(cast<llvm::CallBase>(Val)->getCaller()));
}

void CallBase::setCalledFunction(Function *F) {
  // F's function type is private, so we rely on `setCalledFunction()` to update
  // it. But even though we are calling `setCalledFunction()` we also need to
  // track this change at the SandboxIR level, which is why we call
  // `setCalledOperand()` here.
  // Note: This may break if `setCalledFunction()` early returns if `F`
  // is already set, but we do have a unit test for it.
  setCalledOperand(F);
  cast<llvm::CallBase>(Val)->setCalledFunction(F->getFunctionType(),
                                               cast<llvm::Function>(F->Val));
}

CallInst *CallInst::create(FunctionType *FTy, Value *Func,
                           ArrayRef<Value *> Args, BasicBlock::iterator WhereIt,
                           BasicBlock *WhereBB, Context &Ctx,
                           const Twine &NameStr) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  SmallVector<llvm::Value *> LLVMArgs;
  LLVMArgs.reserve(Args.size());
  for (Value *Arg : Args)
    LLVMArgs.push_back(Arg->Val);
  llvm::CallInst *NewCI = Builder.CreateCall(FTy, Func->Val, LLVMArgs, NameStr);
  return Ctx.createCallInst(NewCI);
}

CallInst *CallInst::create(FunctionType *FTy, Value *Func,
                           ArrayRef<Value *> Args, Instruction *InsertBefore,
                           Context &Ctx, const Twine &NameStr) {
  return CallInst::create(FTy, Func, Args, InsertBefore->getIterator(),
                          InsertBefore->getParent(), Ctx, NameStr);
}

CallInst *CallInst::create(FunctionType *FTy, Value *Func,
                           ArrayRef<Value *> Args, BasicBlock *InsertAtEnd,
                           Context &Ctx, const Twine &NameStr) {
  return CallInst::create(FTy, Func, Args, InsertAtEnd->end(), InsertAtEnd, Ctx,
                          NameStr);
}

#ifndef NDEBUG
void CallInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void CallInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

InvokeInst *InvokeInst::create(FunctionType *FTy, Value *Func,
                               BasicBlock *IfNormal, BasicBlock *IfException,
                               ArrayRef<Value *> Args, BBIterator WhereIt,
                               BasicBlock *WhereBB, Context &Ctx,
                               const Twine &NameStr) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  SmallVector<llvm::Value *> LLVMArgs;
  LLVMArgs.reserve(Args.size());
  for (Value *Arg : Args)
    LLVMArgs.push_back(Arg->Val);
  llvm::InvokeInst *Invoke = Builder.CreateInvoke(
      FTy, Func->Val, cast<llvm::BasicBlock>(IfNormal->Val),
      cast<llvm::BasicBlock>(IfException->Val), LLVMArgs, NameStr);
  return Ctx.createInvokeInst(Invoke);
}

InvokeInst *InvokeInst::create(FunctionType *FTy, Value *Func,
                               BasicBlock *IfNormal, BasicBlock *IfException,
                               ArrayRef<Value *> Args,
                               Instruction *InsertBefore, Context &Ctx,
                               const Twine &NameStr) {
  return create(FTy, Func, IfNormal, IfException, Args,
                InsertBefore->getIterator(), InsertBefore->getParent(), Ctx,
                NameStr);
}

InvokeInst *InvokeInst::create(FunctionType *FTy, Value *Func,
                               BasicBlock *IfNormal, BasicBlock *IfException,
                               ArrayRef<Value *> Args, BasicBlock *InsertAtEnd,
                               Context &Ctx, const Twine &NameStr) {
  return create(FTy, Func, IfNormal, IfException, Args, InsertAtEnd->end(),
                InsertAtEnd, Ctx, NameStr);
}

BasicBlock *InvokeInst::getNormalDest() const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::InvokeInst>(Val)->getNormalDest()));
}
BasicBlock *InvokeInst::getUnwindDest() const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::InvokeInst>(Val)->getUnwindDest()));
}
void InvokeInst::setNormalDest(BasicBlock *BB) {
  setOperand(1, BB);
  assert(getNormalDest() == BB && "LLVM IR uses a different operan index!");
}
void InvokeInst::setUnwindDest(BasicBlock *BB) {
  setOperand(2, BB);
  assert(getUnwindDest() == BB && "LLVM IR uses a different operan index!");
}
Instruction *InvokeInst::getLandingPadInst() const {
  return cast<Instruction>(
      Ctx.getValue(cast<llvm::InvokeInst>(Val)->getLandingPadInst()));
  ;
}
BasicBlock *InvokeInst::getSuccessor(unsigned SuccIdx) const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::InvokeInst>(Val)->getSuccessor(SuccIdx)));
}

#ifndef NDEBUG
void InvokeInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}
void InvokeInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

CallBrInst *CallBrInst::create(FunctionType *FTy, Value *Func,
                               BasicBlock *DefaultDest,
                               ArrayRef<BasicBlock *> IndirectDests,
                               ArrayRef<Value *> Args, BBIterator WhereIt,
                               BasicBlock *WhereBB, Context &Ctx,
                               const Twine &NameStr) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));

  SmallVector<llvm::BasicBlock *> LLVMIndirectDests;
  LLVMIndirectDests.reserve(IndirectDests.size());
  for (BasicBlock *IndDest : IndirectDests)
    LLVMIndirectDests.push_back(cast<llvm::BasicBlock>(IndDest->Val));

  SmallVector<llvm::Value *> LLVMArgs;
  LLVMArgs.reserve(Args.size());
  for (Value *Arg : Args)
    LLVMArgs.push_back(Arg->Val);

  llvm::CallBrInst *CallBr = Builder.CreateCallBr(
      FTy, Func->Val, cast<llvm::BasicBlock>(DefaultDest->Val),
      LLVMIndirectDests, LLVMArgs, NameStr);
  return Ctx.createCallBrInst(CallBr);
}

CallBrInst *CallBrInst::create(FunctionType *FTy, Value *Func,
                               BasicBlock *DefaultDest,
                               ArrayRef<BasicBlock *> IndirectDests,
                               ArrayRef<Value *> Args,
                               Instruction *InsertBefore, Context &Ctx,
                               const Twine &NameStr) {
  return create(FTy, Func, DefaultDest, IndirectDests, Args,
                InsertBefore->getIterator(), InsertBefore->getParent(), Ctx,
                NameStr);
}
CallBrInst *CallBrInst::create(FunctionType *FTy, Value *Func,
                               BasicBlock *DefaultDest,
                               ArrayRef<BasicBlock *> IndirectDests,
                               ArrayRef<Value *> Args, BasicBlock *InsertAtEnd,
                               Context &Ctx, const Twine &NameStr) {
  return create(FTy, Func, DefaultDest, IndirectDests, Args, InsertAtEnd->end(),
                InsertAtEnd, Ctx, NameStr);
}

Value *CallBrInst::getIndirectDestLabel(unsigned Idx) const {
  return Ctx.getValue(cast<llvm::CallBrInst>(Val)->getIndirectDestLabel(Idx));
}
Value *CallBrInst::getIndirectDestLabelUse(unsigned Idx) const {
  return Ctx.getValue(
      cast<llvm::CallBrInst>(Val)->getIndirectDestLabelUse(Idx));
}
BasicBlock *CallBrInst::getDefaultDest() const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::CallBrInst>(Val)->getDefaultDest()));
}
BasicBlock *CallBrInst::getIndirectDest(unsigned Idx) const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::CallBrInst>(Val)->getIndirectDest(Idx)));
}
llvm::SmallVector<BasicBlock *, 16> CallBrInst::getIndirectDests() const {
  SmallVector<BasicBlock *, 16> BBs;
  for (llvm::BasicBlock *LLVMBB :
       cast<llvm::CallBrInst>(Val)->getIndirectDests())
    BBs.push_back(cast<BasicBlock>(Ctx.getValue(LLVMBB)));
  return BBs;
}
void CallBrInst::setDefaultDest(BasicBlock *BB) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking()) {
    Tracker.track(std::make_unique<GenericSetter<&CallBrInst::getDefaultDest,
                                                 &CallBrInst::setDefaultDest>>(
        this, Tracker));
  }
  cast<llvm::CallBrInst>(Val)->setDefaultDest(cast<llvm::BasicBlock>(BB->Val));
}
void CallBrInst::setIndirectDest(unsigned Idx, BasicBlock *BB) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking())
    Tracker.track(
        std::make_unique<CallBrInstSetIndirectDest>(this, Idx, Tracker));
  cast<llvm::CallBrInst>(Val)->setIndirectDest(Idx,
                                               cast<llvm::BasicBlock>(BB->Val));
}
BasicBlock *CallBrInst::getSuccessor(unsigned Idx) const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::CallBrInst>(Val)->getSuccessor(Idx)));
}

#ifndef NDEBUG
void CallBrInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}
void CallBrInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Value *GetElementPtrInst::create(Type *Ty, Value *Ptr,
                                 ArrayRef<Value *> IdxList,
                                 BasicBlock::iterator WhereIt,
                                 BasicBlock *WhereBB, Context &Ctx,
                                 const Twine &NameStr) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  SmallVector<llvm::Value *> LLVMIdxList;
  LLVMIdxList.reserve(IdxList.size());
  for (Value *Idx : IdxList)
    LLVMIdxList.push_back(Idx->Val);
  llvm::Value *NewV = Builder.CreateGEP(Ty, Ptr->Val, LLVMIdxList, NameStr);
  if (auto *NewGEP = dyn_cast<llvm::GetElementPtrInst>(NewV))
    return Ctx.createGetElementPtrInst(NewGEP);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *GetElementPtrInst::create(Type *Ty, Value *Ptr,
                                 ArrayRef<Value *> IdxList,
                                 Instruction *InsertBefore, Context &Ctx,
                                 const Twine &NameStr) {
  return GetElementPtrInst::create(Ty, Ptr, IdxList,
                                   InsertBefore->getIterator(),
                                   InsertBefore->getParent(), Ctx, NameStr);
}

Value *GetElementPtrInst::create(Type *Ty, Value *Ptr,
                                 ArrayRef<Value *> IdxList,
                                 BasicBlock *InsertAtEnd, Context &Ctx,
                                 const Twine &NameStr) {
  return GetElementPtrInst::create(Ty, Ptr, IdxList, InsertAtEnd->end(),
                                   InsertAtEnd, Ctx, NameStr);
}

Value *GetElementPtrInst::getPointerOperand() const {
  return Ctx.getValue(cast<llvm::GetElementPtrInst>(Val)->getPointerOperand());
}

#ifndef NDEBUG
void GetElementPtrInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void GetElementPtrInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

BasicBlock *PHINode::LLVMBBToBB::operator()(llvm::BasicBlock *LLVMBB) const {
  return cast<BasicBlock>(Ctx.getValue(LLVMBB));
}

PHINode *PHINode::create(Type *Ty, unsigned NumReservedValues,
                         Instruction *InsertBefore, Context &Ctx,
                         const Twine &Name) {
  llvm::PHINode *NewPHI = llvm::PHINode::Create(
      Ty, NumReservedValues, Name, InsertBefore->getTopmostLLVMInstruction());
  return Ctx.createPHINode(NewPHI);
}

bool PHINode::classof(const Value *From) {
  return From->getSubclassID() == ClassID::PHI;
}

Value *PHINode::getIncomingValue(unsigned Idx) const {
  return Ctx.getValue(cast<llvm::PHINode>(Val)->getIncomingValue(Idx));
}
void PHINode::setIncomingValue(unsigned Idx, Value *V) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking())
    Tracker.track(std::make_unique<PHISetIncoming>(
        *this, Idx, PHISetIncoming::What::Value, Tracker));

  cast<llvm::PHINode>(Val)->setIncomingValue(Idx, V->Val);
}
BasicBlock *PHINode::getIncomingBlock(unsigned Idx) const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::PHINode>(Val)->getIncomingBlock(Idx)));
}
BasicBlock *PHINode::getIncomingBlock(const Use &U) const {
  llvm::Use *LLVMUse = U.LLVMUse;
  llvm::BasicBlock *BB = cast<llvm::PHINode>(Val)->getIncomingBlock(*LLVMUse);
  return cast<BasicBlock>(Ctx.getValue(BB));
}
void PHINode::setIncomingBlock(unsigned Idx, BasicBlock *BB) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking())
    Tracker.track(std::make_unique<PHISetIncoming>(
        *this, Idx, PHISetIncoming::What::Block, Tracker));
  cast<llvm::PHINode>(Val)->setIncomingBlock(Idx,
                                             cast<llvm::BasicBlock>(BB->Val));
}
void PHINode::addIncoming(Value *V, BasicBlock *BB) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking())
    Tracker.track(std::make_unique<PHIAddIncoming>(*this, Tracker));

  cast<llvm::PHINode>(Val)->addIncoming(V->Val,
                                        cast<llvm::BasicBlock>(BB->Val));
}
Value *PHINode::removeIncomingValue(unsigned Idx) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking())
    Tracker.track(std::make_unique<PHIRemoveIncoming>(*this, Idx, Tracker));
  llvm::Value *LLVMV =
      cast<llvm::PHINode>(Val)->removeIncomingValue(Idx,
                                                    /*DeletePHIIfEmpty=*/false);
  return Ctx.getValue(LLVMV);
}
Value *PHINode::removeIncomingValue(BasicBlock *BB) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking())
    Tracker.track(std::make_unique<PHIRemoveIncoming>(
        *this, getBasicBlockIndex(BB), Tracker));

  auto *LLVMBB = cast<llvm::BasicBlock>(BB->Val);
  llvm::Value *LLVMV =
      cast<llvm::PHINode>(Val)->removeIncomingValue(LLVMBB,
                                                    /*DeletePHIIfEmpty=*/false);
  return Ctx.getValue(LLVMV);
}
int PHINode::getBasicBlockIndex(const BasicBlock *BB) const {
  auto *LLVMBB = cast<llvm::BasicBlock>(BB->Val);
  return cast<llvm::PHINode>(Val)->getBasicBlockIndex(LLVMBB);
}
Value *PHINode::getIncomingValueForBlock(const BasicBlock *BB) const {
  auto *LLVMBB = cast<llvm::BasicBlock>(BB->Val);
  llvm::Value *LLVMV =
      cast<llvm::PHINode>(Val)->getIncomingValueForBlock(LLVMBB);
  return Ctx.getValue(LLVMV);
}
Value *PHINode::hasConstantValue() const {
  llvm::Value *LLVMV = cast<llvm::PHINode>(Val)->hasConstantValue();
  return LLVMV != nullptr ? Ctx.getValue(LLVMV) : nullptr;
}
void PHINode::replaceIncomingBlockWith(const BasicBlock *Old, BasicBlock *New) {
  assert(New && Old && "Sandbox IR PHI node got a null basic block!");
  for (unsigned Idx = 0, NumOps = cast<llvm::PHINode>(Val)->getNumOperands();
       Idx != NumOps; ++Idx)
    if (getIncomingBlock(Idx) == Old)
      setIncomingBlock(Idx, New);
}
void PHINode::removeIncomingValueIf(function_ref<bool(unsigned)> Predicate) {
  // Avoid duplicate tracking by going through this->removeIncomingValue here at
  // the expense of some performance. Copy PHI::removeIncomingValueIf more
  // directly if performance becomes an issue.

  // Removing the element at index X, moves the element previously at X + 1
  // to X. Working from the end avoids complications from that.
  unsigned Idx = getNumIncomingValues();
  while (Idx > 0) {
    if (Predicate(Idx - 1))
      removeIncomingValue(Idx - 1);
    --Idx;
  }
}

static llvm::Instruction::CastOps getLLVMCastOp(Instruction::Opcode Opc) {
  switch (Opc) {
  case Instruction::Opcode::ZExt:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::ZExt);
  case Instruction::Opcode::SExt:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::SExt);
  case Instruction::Opcode::FPToUI:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::FPToUI);
  case Instruction::Opcode::FPToSI:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::FPToSI);
  case Instruction::Opcode::FPExt:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::FPExt);
  case Instruction::Opcode::PtrToInt:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::PtrToInt);
  case Instruction::Opcode::IntToPtr:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::IntToPtr);
  case Instruction::Opcode::SIToFP:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::SIToFP);
  case Instruction::Opcode::UIToFP:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::UIToFP);
  case Instruction::Opcode::Trunc:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::Trunc);
  case Instruction::Opcode::FPTrunc:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::FPTrunc);
  case Instruction::Opcode::BitCast:
    return static_cast<llvm::Instruction::CastOps>(llvm::Instruction::BitCast);
  case Instruction::Opcode::AddrSpaceCast:
    return static_cast<llvm::Instruction::CastOps>(
        llvm::Instruction::AddrSpaceCast);
  default:
    llvm_unreachable("Opcode not suitable for CastInst!");
  }
}

AllocaInst *AllocaInst::create(Type *Ty, unsigned AddrSpace, BBIterator WhereIt,
                               BasicBlock *WhereBB, Context &Ctx,
                               Value *ArraySize, const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt == WhereBB->end())
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  else
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  auto *NewAlloca = Builder.CreateAlloca(Ty, AddrSpace, ArraySize->Val, Name);
  return Ctx.createAllocaInst(NewAlloca);
}

AllocaInst *AllocaInst::create(Type *Ty, unsigned AddrSpace,
                               Instruction *InsertBefore, Context &Ctx,
                               Value *ArraySize, const Twine &Name) {
  return create(Ty, AddrSpace, InsertBefore->getIterator(),
                InsertBefore->getParent(), Ctx, ArraySize, Name);
}

AllocaInst *AllocaInst::create(Type *Ty, unsigned AddrSpace,
                               BasicBlock *InsertAtEnd, Context &Ctx,
                               Value *ArraySize, const Twine &Name) {
  return create(Ty, AddrSpace, InsertAtEnd->end(), InsertAtEnd, Ctx, ArraySize,
                Name);
}

void AllocaInst::setAllocatedType(Type *Ty) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking()) {
    Tracker.track(
        std::make_unique<GenericSetter<&AllocaInst::getAllocatedType,
                                       &AllocaInst::setAllocatedType>>(
            this, Tracker));
  }
  cast<llvm::AllocaInst>(Val)->setAllocatedType(Ty);
}

void AllocaInst::setAlignment(Align Align) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking()) {
    Tracker.track(
        std::make_unique<
            GenericSetter<&AllocaInst::getAlign, &AllocaInst::setAlignment>>(
            this, Tracker));
  }
  cast<llvm::AllocaInst>(Val)->setAlignment(Align);
}

void AllocaInst::setUsedWithInAlloca(bool V) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking()) {
    Tracker.track(
        std::make_unique<GenericSetter<&AllocaInst::isUsedWithInAlloca,
                                       &AllocaInst::setUsedWithInAlloca>>(
            this, Tracker));
  }
  cast<llvm::AllocaInst>(Val)->setUsedWithInAlloca(V);
}

Value *AllocaInst::getArraySize() {
  return Ctx.getValue(cast<llvm::AllocaInst>(Val)->getArraySize());
}

#ifndef NDEBUG
void AllocaInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void AllocaInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

Value *CastInst::create(Type *DestTy, Opcode Op, Value *Operand,
                        BBIterator WhereIt, BasicBlock *WhereBB, Context &Ctx,
                        const Twine &Name) {
  assert(getLLVMCastOp(Op) && "Opcode not suitable for CastInst!");
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt == WhereBB->end())
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  else
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  auto *NewV =
      Builder.CreateCast(getLLVMCastOp(Op), Operand->Val, DestTy, Name);
  if (auto *NewCI = dyn_cast<llvm::CastInst>(NewV))
    return Ctx.createCastInst(NewCI);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *CastInst::create(Type *DestTy, Opcode Op, Value *Operand,
                        Instruction *InsertBefore, Context &Ctx,
                        const Twine &Name) {
  return create(DestTy, Op, Operand, InsertBefore->getIterator(),
                InsertBefore->getParent(), Ctx, Name);
}

Value *CastInst::create(Type *DestTy, Opcode Op, Value *Operand,
                        BasicBlock *InsertAtEnd, Context &Ctx,
                        const Twine &Name) {
  return create(DestTy, Op, Operand, InsertAtEnd->end(), InsertAtEnd, Ctx,
                Name);
}

bool CastInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Cast;
}

#ifndef NDEBUG
void CastInst::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void CastInst::dump() const {
  dump(dbgs());
  dbgs() << "\n";
}

void PHINode::dump(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void PHINode::dump() const {
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
#endif // NDEBUG

Constant *Constant::createInt(Type *Ty, uint64_t V, Context &Ctx,
                              bool IsSigned) {
  llvm::Constant *LLVMC = llvm::ConstantInt::get(Ty, V, IsSigned);
  return Ctx.getOrCreateConstant(LLVMC);
}

#ifndef NDEBUG
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

std::unique_ptr<Value> Context::detachLLVMValue(llvm::Value *V) {
  std::unique_ptr<Value> Erased;
  auto It = LLVMValueToValueMap.find(V);
  if (It != LLVMValueToValueMap.end()) {
    auto *Val = It->second.release();
    Erased = std::unique_ptr<Value>(Val);
    LLVMValueToValueMap.erase(It);
  }
  return Erased;
}

std::unique_ptr<Value> Context::detach(Value *V) {
  assert(V->getSubclassID() != Value::ClassID::Constant &&
         "Can't detach a constant!");
  assert(V->getSubclassID() != Value::ClassID::User && "Can't detach a user!");
  return detachLLVMValue(V->Val);
}

Value *Context::registerValue(std::unique_ptr<Value> &&VPtr) {
  assert(VPtr->getSubclassID() != Value::ClassID::User &&
         "Can't register a user!");

  // Track creation of instructions.
  // Please note that we don't allow the creation of detached instructions,
  // meaning that the instructions need to be inserted into a block upon
  // creation. This is why the tracker class combines creation and insertion.
  auto &Tracker = getTracker();
  if (Tracker.isTracking())
    if (auto *I = dyn_cast<Instruction>(VPtr.get()))
      Tracker.track(std::make_unique<CreateAndInsertInst>(I, Tracker));

  Value *V = VPtr.get();
  [[maybe_unused]] auto Pair =
      LLVMValueToValueMap.insert({VPtr->Val, std::move(VPtr)});
  assert(Pair.second && "Already exists!");
  return V;
}

Value *Context::getOrCreateValueInternal(llvm::Value *LLVMV, llvm::User *U) {
  auto Pair = LLVMValueToValueMap.insert({LLVMV, nullptr});
  auto It = Pair.first;
  if (!Pair.second)
    return It->second.get();

  if (auto *C = dyn_cast<llvm::Constant>(LLVMV)) {
    if (auto *F = dyn_cast<llvm::Function>(LLVMV))
      It->second = std::unique_ptr<Function>(new Function(F, *this));
    else
      It->second = std::unique_ptr<Constant>(new Constant(C, *this));
    auto *NewC = It->second.get();
    for (llvm::Value *COp : C->operands())
      getOrCreateValueInternal(COp, C);
    return NewC;
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

  switch (cast<llvm::Instruction>(LLVMV)->getOpcode()) {
  case llvm::Instruction::Select: {
    auto *LLVMSel = cast<llvm::SelectInst>(LLVMV);
    It->second = std::unique_ptr<SelectInst>(new SelectInst(LLVMSel, *this));
    return It->second.get();
  }
  case llvm::Instruction::Br: {
    auto *LLVMBr = cast<llvm::BranchInst>(LLVMV);
    It->second = std::unique_ptr<BranchInst>(new BranchInst(LLVMBr, *this));
    return It->second.get();
  }
  case llvm::Instruction::Load: {
    auto *LLVMLd = cast<llvm::LoadInst>(LLVMV);
    It->second = std::unique_ptr<LoadInst>(new LoadInst(LLVMLd, *this));
    return It->second.get();
  }
  case llvm::Instruction::Store: {
    auto *LLVMSt = cast<llvm::StoreInst>(LLVMV);
    It->second = std::unique_ptr<StoreInst>(new StoreInst(LLVMSt, *this));
    return It->second.get();
  }
  case llvm::Instruction::Ret: {
    auto *LLVMRet = cast<llvm::ReturnInst>(LLVMV);
    It->second = std::unique_ptr<ReturnInst>(new ReturnInst(LLVMRet, *this));
    return It->second.get();
  }
  case llvm::Instruction::Call: {
    auto *LLVMCall = cast<llvm::CallInst>(LLVMV);
    It->second = std::unique_ptr<CallInst>(new CallInst(LLVMCall, *this));
    return It->second.get();
  }
  case llvm::Instruction::Invoke: {
    auto *LLVMInvoke = cast<llvm::InvokeInst>(LLVMV);
    It->second = std::unique_ptr<InvokeInst>(new InvokeInst(LLVMInvoke, *this));
    return It->second.get();
  }
  case llvm::Instruction::CallBr: {
    auto *LLVMCallBr = cast<llvm::CallBrInst>(LLVMV);
    It->second = std::unique_ptr<CallBrInst>(new CallBrInst(LLVMCallBr, *this));
    return It->second.get();
  }
  case llvm::Instruction::GetElementPtr: {
    auto *LLVMGEP = cast<llvm::GetElementPtrInst>(LLVMV);
    It->second = std::unique_ptr<GetElementPtrInst>(
        new GetElementPtrInst(LLVMGEP, *this));
    return It->second.get();
  }
  case llvm::Instruction::Alloca: {
    auto *LLVMAlloca = cast<llvm::AllocaInst>(LLVMV);
    It->second = std::unique_ptr<AllocaInst>(new AllocaInst(LLVMAlloca, *this));
    return It->second.get();
  }
  case llvm::Instruction::ZExt:
  case llvm::Instruction::SExt:
  case llvm::Instruction::FPToUI:
  case llvm::Instruction::FPToSI:
  case llvm::Instruction::FPExt:
  case llvm::Instruction::PtrToInt:
  case llvm::Instruction::IntToPtr:
  case llvm::Instruction::SIToFP:
  case llvm::Instruction::UIToFP:
  case llvm::Instruction::Trunc:
  case llvm::Instruction::FPTrunc:
  case llvm::Instruction::BitCast:
  case llvm::Instruction::AddrSpaceCast: {
    auto *LLVMCast = cast<llvm::CastInst>(LLVMV);
    It->second = std::unique_ptr<CastInst>(new CastInst(LLVMCast, *this));
    return It->second.get();
  }
  case llvm::Instruction::PHI: {
    auto *LLVMPhi = cast<llvm::PHINode>(LLVMV);
    It->second = std::unique_ptr<PHINode>(new PHINode(LLVMPhi, *this));
    return It->second.get();
  }
  case llvm::Instruction::Unreachable: {
    auto *LLVMUnreachable = cast<llvm::UnreachableInst>(LLVMV);
    It->second = std::unique_ptr<UnreachableInst>(
        new UnreachableInst(LLVMUnreachable, *this));
    return It->second.get();
  }
  default:
    break;
  }

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

SelectInst *Context::createSelectInst(llvm::SelectInst *SI) {
  auto NewPtr = std::unique_ptr<SelectInst>(new SelectInst(SI, *this));
  return cast<SelectInst>(registerValue(std::move(NewPtr)));
}

BranchInst *Context::createBranchInst(llvm::BranchInst *BI) {
  auto NewPtr = std::unique_ptr<BranchInst>(new BranchInst(BI, *this));
  return cast<BranchInst>(registerValue(std::move(NewPtr)));
}

LoadInst *Context::createLoadInst(llvm::LoadInst *LI) {
  auto NewPtr = std::unique_ptr<LoadInst>(new LoadInst(LI, *this));
  return cast<LoadInst>(registerValue(std::move(NewPtr)));
}

StoreInst *Context::createStoreInst(llvm::StoreInst *SI) {
  auto NewPtr = std::unique_ptr<StoreInst>(new StoreInst(SI, *this));
  return cast<StoreInst>(registerValue(std::move(NewPtr)));
}

ReturnInst *Context::createReturnInst(llvm::ReturnInst *I) {
  auto NewPtr = std::unique_ptr<ReturnInst>(new ReturnInst(I, *this));
  return cast<ReturnInst>(registerValue(std::move(NewPtr)));
}

CallInst *Context::createCallInst(llvm::CallInst *I) {
  auto NewPtr = std::unique_ptr<CallInst>(new CallInst(I, *this));
  return cast<CallInst>(registerValue(std::move(NewPtr)));
}

InvokeInst *Context::createInvokeInst(llvm::InvokeInst *I) {
  auto NewPtr = std::unique_ptr<InvokeInst>(new InvokeInst(I, *this));
  return cast<InvokeInst>(registerValue(std::move(NewPtr)));
}

CallBrInst *Context::createCallBrInst(llvm::CallBrInst *I) {
  auto NewPtr = std::unique_ptr<CallBrInst>(new CallBrInst(I, *this));
  return cast<CallBrInst>(registerValue(std::move(NewPtr)));
}

UnreachableInst *Context::createUnreachableInst(llvm::UnreachableInst *UI) {
  auto NewPtr =
      std::unique_ptr<UnreachableInst>(new UnreachableInst(UI, *this));
  return cast<UnreachableInst>(registerValue(std::move(NewPtr)));
}

GetElementPtrInst *
Context::createGetElementPtrInst(llvm::GetElementPtrInst *I) {
  auto NewPtr =
      std::unique_ptr<GetElementPtrInst>(new GetElementPtrInst(I, *this));
  return cast<GetElementPtrInst>(registerValue(std::move(NewPtr)));
}
AllocaInst *Context::createAllocaInst(llvm::AllocaInst *I) {
  auto NewPtr = std::unique_ptr<AllocaInst>(new AllocaInst(I, *this));
  return cast<AllocaInst>(registerValue(std::move(NewPtr)));
}
CastInst *Context::createCastInst(llvm::CastInst *I) {
  auto NewPtr = std::unique_ptr<CastInst>(new CastInst(I, *this));
  return cast<CastInst>(registerValue(std::move(NewPtr)));
}
PHINode *Context::createPHINode(llvm::PHINode *I) {
  auto NewPtr = std::unique_ptr<PHINode>(new PHINode(I, *this));
  return cast<PHINode>(registerValue(std::move(NewPtr)));
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
  auto *SBF = cast<Function>(registerValue(std::move(NewFPtr)));
  // Create arguments.
  for (auto &Arg : F->args())
    getOrCreateArgument(&Arg);
  // Create BBs.
  for (auto &BB : *F)
    createBasicBlock(&BB);
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
