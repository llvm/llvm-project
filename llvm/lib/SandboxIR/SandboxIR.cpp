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
  Ctx->getTracker().emplaceIfTracking<UseSet>(*this);
  LLVMUse->set(V->Val);
}

unsigned Use::getOperandNo() const { return Usr->getUseOperandNo(*this); }

void Use::swap(Use &OtherUse) {
  Ctx->getTracker().emplaceIfTracking<UseSwap>(*this, OtherUse);
  LLVMUse->swap(*OtherUse.LLVMUse);
}

#ifndef NDEBUG
void Use::dumpOS(raw_ostream &OS) const {
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

void Use::dump() const { dumpOS(dbgs()); }
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

Type *Value::getType() const { return Ctx.getType(Val->getType()); }

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
        Ctx.getTracker().emplaceIfTracking<UseSet>(UseToReplace);
        return true;
      });
}

void Value::replaceAllUsesWith(Value *Other) {
  assert(getType() == Other->getType() &&
         "Replacing with Value of different type!");
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking()) {
    for (auto Use : uses())
      Tracker.track(std::make_unique<UseSet>(Use));
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

void Value::dump() const {
  dumpOS(dbgs());
  dbgs() << "\n";
}

void Argument::printAsOperand(raw_ostream &OS) const {
  printAsOperandCommon(OS);
}
void Argument::dumpOS(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
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
  Ctx.getTracker().emplaceIfTracking<UseSet>(getOperandUse(OperandIdx));
  // We are delegating to llvm::User::setOperand().
  cast<llvm::User>(Val)->setOperand(OperandIdx, Operand->Val);
}

bool User::replaceUsesOfWith(Value *FromV, Value *ToV) {
  auto &Tracker = Ctx.getTracker();
  if (Tracker.isTracking()) {
    for (auto OpIdx : seq<unsigned>(0, getNumOperands())) {
      auto Use = getOperandUse(OpIdx);
      if (Use.get() == FromV)
        Tracker.emplaceIfTracking<UseSet>(Use);
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

BasicBlock *BBIterator::getNodeParent() const {
  llvm::BasicBlock *Parent = const_cast<BBIterator *>(this)->It.getNodeParent();
  return cast<BasicBlock>(Ctx->getValue(Parent));
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
  Ctx.getTracker().emplaceIfTracking<RemoveFromParent>(this);

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
    Tracker.track(std::make_unique<EraseFromParent>(std::move(Detached)));
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

  Ctx.getTracker().emplaceIfTracking<MoveInstr>(this);

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

  Ctx.getTracker().emplaceIfTracking<InsertIntoBB>(this);

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

  Ctx.getTracker().emplaceIfTracking<InsertIntoBB>(this);

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

void Instruction::setHasNoUnsignedWrap(bool B) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&Instruction::hasNoUnsignedWrap,
                                       &Instruction::setHasNoUnsignedWrap>>(
          this);
  cast<llvm::Instruction>(Val)->setHasNoUnsignedWrap(B);
}

void Instruction::setHasNoSignedWrap(bool B) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&Instruction::hasNoSignedWrap,
                                       &Instruction::setHasNoSignedWrap>>(this);
  cast<llvm::Instruction>(Val)->setHasNoSignedWrap(B);
}

void Instruction::setFast(bool B) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&Instruction::isFast, &Instruction::setFast>>(this);
  cast<llvm::Instruction>(Val)->setFast(B);
}

void Instruction::setIsExact(bool B) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&Instruction::isExact, &Instruction::setIsExact>>(this);
  cast<llvm::Instruction>(Val)->setIsExact(B);
}

void Instruction::setHasAllowReassoc(bool B) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&Instruction::hasAllowReassoc,
                                       &Instruction::setHasAllowReassoc>>(this);
  cast<llvm::Instruction>(Val)->setHasAllowReassoc(B);
}

void Instruction::setHasNoNaNs(bool B) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&Instruction::hasNoNaNs, &Instruction::setHasNoNaNs>>(
          this);
  cast<llvm::Instruction>(Val)->setHasNoNaNs(B);
}

void Instruction::setHasNoInfs(bool B) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&Instruction::hasNoInfs, &Instruction::setHasNoInfs>>(
          this);
  cast<llvm::Instruction>(Val)->setHasNoInfs(B);
}

void Instruction::setHasNoSignedZeros(bool B) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&Instruction::hasNoSignedZeros,
                                       &Instruction::setHasNoSignedZeros>>(
          this);
  cast<llvm::Instruction>(Val)->setHasNoSignedZeros(B);
}

void Instruction::setHasAllowReciprocal(bool B) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&Instruction::hasAllowReciprocal,
                                       &Instruction::setHasAllowReciprocal>>(
          this);
  cast<llvm::Instruction>(Val)->setHasAllowReciprocal(B);
}

void Instruction::setHasAllowContract(bool B) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&Instruction::hasAllowContract,
                                       &Instruction::setHasAllowContract>>(
          this);
  cast<llvm::Instruction>(Val)->setHasAllowContract(B);
}

void Instruction::setFastMathFlags(FastMathFlags FMF) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&Instruction::getFastMathFlags,
                                       &Instruction::copyFastMathFlags>>(this);
  cast<llvm::Instruction>(Val)->setFastMathFlags(FMF);
}

void Instruction::copyFastMathFlags(FastMathFlags FMF) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&Instruction::getFastMathFlags,
                                       &Instruction::copyFastMathFlags>>(this);
  cast<llvm::Instruction>(Val)->copyFastMathFlags(FMF);
}

Type *Instruction::getAccessType() const {
  return Ctx.getType(cast<llvm::Instruction>(Val)->getAccessType());
}

void Instruction::setHasApproxFunc(bool B) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&Instruction::hasApproxFunc,
                                       &Instruction::setHasApproxFunc>>(this);
  cast<llvm::Instruction>(Val)->setHasApproxFunc(B);
}

#ifndef NDEBUG
void Instruction::dumpOS(raw_ostream &OS) const {
  OS << "Unimplemented! Please override dump().";
}
#endif // NDEBUG

VAArgInst *VAArgInst::create(Value *List, Type *Ty, BBIterator WhereIt,
                             BasicBlock *WhereBB, Context &Ctx,
                             const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  auto *LLVMI =
      cast<llvm::VAArgInst>(Builder.CreateVAArg(List->Val, Ty->LLVMTy, Name));
  return Ctx.createVAArgInst(LLVMI);
}

Value *VAArgInst::getPointerOperand() {
  return Ctx.getValue(cast<llvm::VAArgInst>(Val)->getPointerOperand());
}

FreezeInst *FreezeInst::create(Value *V, BBIterator WhereIt,
                               BasicBlock *WhereBB, Context &Ctx,
                               const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  auto *LLVMI = cast<llvm::FreezeInst>(Builder.CreateFreeze(V->Val, Name));
  return Ctx.createFreezeInst(LLVMI);
}

FenceInst *FenceInst::create(AtomicOrdering Ordering, BBIterator WhereIt,
                             BasicBlock *WhereBB, Context &Ctx,
                             SyncScope::ID SSID) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  llvm::FenceInst *LLVMI = Builder.CreateFence(Ordering, SSID);
  return Ctx.createFenceInst(LLVMI);
}

void FenceInst::setOrdering(AtomicOrdering Ordering) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&FenceInst::getOrdering, &FenceInst::setOrdering>>(
          this);
  cast<llvm::FenceInst>(Val)->setOrdering(Ordering);
}

void FenceInst::setSyncScopeID(SyncScope::ID SSID) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&FenceInst::getSyncScopeID,
                                       &FenceInst::setSyncScopeID>>(this);
  cast<llvm::FenceInst>(Val)->setSyncScopeID(SSID);
}

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

void SelectInst::swapValues() {
  Ctx.getTracker().emplaceIfTracking<UseSwap>(getOperandUse(1),
                                              getOperandUse(2));
  cast<llvm::SelectInst>(Val)->swapValues();
}

bool SelectInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Select;
}

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

void LoadInst::setVolatile(bool V) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&LoadInst::isVolatile, &LoadInst::setVolatile>>(this);
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
      Builder.CreateAlignedLoad(Ty->LLVMTy, Ptr->Val, Align, IsVolatile, Name);
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
      Builder.CreateAlignedLoad(Ty->LLVMTy, Ptr->Val, Align, IsVolatile, Name);
  auto *NewSBI = Ctx.createLoadInst(NewLI);
  return NewSBI;
}

bool LoadInst::classof(const Value *From) {
  return From->getSubclassID() == ClassID::Load;
}

Value *LoadInst::getPointerOperand() const {
  return Ctx.getValue(cast<llvm::LoadInst>(Val)->getPointerOperand());
}

void StoreInst::setVolatile(bool V) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&StoreInst::isVolatile, &StoreInst::setVolatile>>(this);
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

FunctionType *CallBase::getFunctionType() const {
  return cast<FunctionType>(
      Ctx.getType(cast<llvm::CallBase>(Val)->getFunctionType()));
}

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
  cast<llvm::CallBase>(Val)->setCalledFunction(
      cast<llvm::FunctionType>(F->getFunctionType()->LLVMTy),
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
  llvm::CallInst *NewCI = Builder.CreateCall(
      cast<llvm::FunctionType>(FTy->LLVMTy), Func->Val, LLVMArgs, NameStr);
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
      cast<llvm::FunctionType>(FTy->LLVMTy), Func->Val,
      cast<llvm::BasicBlock>(IfNormal->Val),
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
LandingPadInst *InvokeInst::getLandingPadInst() const {
  return cast<LandingPadInst>(
      Ctx.getValue(cast<llvm::InvokeInst>(Val)->getLandingPadInst()));
  ;
}
BasicBlock *InvokeInst::getSuccessor(unsigned SuccIdx) const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::InvokeInst>(Val)->getSuccessor(SuccIdx)));
}

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

  llvm::CallBrInst *CallBr =
      Builder.CreateCallBr(cast<llvm::FunctionType>(FTy->LLVMTy), Func->Val,
                           cast<llvm::BasicBlock>(DefaultDest->Val),
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
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&CallBrInst::getDefaultDest,
                                       &CallBrInst::setDefaultDest>>(this);
  cast<llvm::CallBrInst>(Val)->setDefaultDest(cast<llvm::BasicBlock>(BB->Val));
}
void CallBrInst::setIndirectDest(unsigned Idx, BasicBlock *BB) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetterWithIdx<&CallBrInst::getIndirectDest,
                                              &CallBrInst::setIndirectDest>>(
          this, Idx);
  cast<llvm::CallBrInst>(Val)->setIndirectDest(Idx,
                                               cast<llvm::BasicBlock>(BB->Val));
}
BasicBlock *CallBrInst::getSuccessor(unsigned Idx) const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::CallBrInst>(Val)->getSuccessor(Idx)));
}

LandingPadInst *LandingPadInst::create(Type *RetTy, unsigned NumReservedClauses,
                                       BBIterator WhereIt, BasicBlock *WhereBB,
                                       Context &Ctx, const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  llvm::LandingPadInst *LLVMI =
      Builder.CreateLandingPad(RetTy->LLVMTy, NumReservedClauses, Name);
  return Ctx.createLandingPadInst(LLVMI);
}

void LandingPadInst::setCleanup(bool V) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&LandingPadInst::isCleanup,
                                       &LandingPadInst::setCleanup>>(this);
  cast<llvm::LandingPadInst>(Val)->setCleanup(V);
}

Constant *LandingPadInst::getClause(unsigned Idx) const {
  return cast<Constant>(
      Ctx.getValue(cast<llvm::LandingPadInst>(Val)->getClause(Idx)));
}

Value *FuncletPadInst::getParentPad() const {
  return Ctx.getValue(cast<llvm::FuncletPadInst>(Val)->getParentPad());
}

void FuncletPadInst::setParentPad(Value *ParentPad) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&FuncletPadInst::getParentPad,
                                       &FuncletPadInst::setParentPad>>(this);
  cast<llvm::FuncletPadInst>(Val)->setParentPad(ParentPad->Val);
}

Value *FuncletPadInst::getArgOperand(unsigned Idx) const {
  return Ctx.getValue(cast<llvm::FuncletPadInst>(Val)->getArgOperand(Idx));
}

void FuncletPadInst::setArgOperand(unsigned Idx, Value *V) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetterWithIdx<&FuncletPadInst::getArgOperand,
                                              &FuncletPadInst::setArgOperand>>(
          this, Idx);
  cast<llvm::FuncletPadInst>(Val)->setArgOperand(Idx, V->Val);
}

CatchSwitchInst *CatchPadInst::getCatchSwitch() const {
  return cast<CatchSwitchInst>(
      Ctx.getValue(cast<llvm::CatchPadInst>(Val)->getCatchSwitch()));
}

CatchPadInst *CatchPadInst::create(Value *ParentPad, ArrayRef<Value *> Args,
                                   BBIterator WhereIt, BasicBlock *WhereBB,
                                   Context &Ctx, const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  SmallVector<llvm::Value *> LLVMArgs;
  LLVMArgs.reserve(Args.size());
  for (auto *Arg : Args)
    LLVMArgs.push_back(Arg->Val);
  llvm::CatchPadInst *LLVMI =
      Builder.CreateCatchPad(ParentPad->Val, LLVMArgs, Name);
  return Ctx.createCatchPadInst(LLVMI);
}

CleanupPadInst *CleanupPadInst::create(Value *ParentPad, ArrayRef<Value *> Args,
                                       BBIterator WhereIt, BasicBlock *WhereBB,
                                       Context &Ctx, const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  SmallVector<llvm::Value *> LLVMArgs;
  LLVMArgs.reserve(Args.size());
  for (auto *Arg : Args)
    LLVMArgs.push_back(Arg->Val);
  llvm::CleanupPadInst *LLVMI =
      Builder.CreateCleanupPad(ParentPad->Val, LLVMArgs, Name);
  return Ctx.createCleanupPadInst(LLVMI);
}

CatchReturnInst *CatchReturnInst::create(CatchPadInst *CatchPad, BasicBlock *BB,
                                         BBIterator WhereIt,
                                         BasicBlock *WhereBB, Context &Ctx) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  llvm::CatchReturnInst *LLVMI = Builder.CreateCatchRet(
      cast<llvm::CatchPadInst>(CatchPad->Val), cast<llvm::BasicBlock>(BB->Val));
  return Ctx.createCatchReturnInst(LLVMI);
}

CatchPadInst *CatchReturnInst::getCatchPad() const {
  return cast<CatchPadInst>(
      Ctx.getValue(cast<llvm::CatchReturnInst>(Val)->getCatchPad()));
}

void CatchReturnInst::setCatchPad(CatchPadInst *CatchPad) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&CatchReturnInst::getCatchPad,
                                       &CatchReturnInst::setCatchPad>>(this);
  cast<llvm::CatchReturnInst>(Val)->setCatchPad(
      cast<llvm::CatchPadInst>(CatchPad->Val));
}

BasicBlock *CatchReturnInst::getSuccessor() const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::CatchReturnInst>(Val)->getSuccessor()));
}

void CatchReturnInst::setSuccessor(BasicBlock *NewSucc) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&CatchReturnInst::getSuccessor,
                                       &CatchReturnInst::setSuccessor>>(this);
  cast<llvm::CatchReturnInst>(Val)->setSuccessor(
      cast<llvm::BasicBlock>(NewSucc->Val));
}

Value *CatchReturnInst::getCatchSwitchParentPad() const {
  return Ctx.getValue(
      cast<llvm::CatchReturnInst>(Val)->getCatchSwitchParentPad());
}

CleanupReturnInst *CleanupReturnInst::create(CleanupPadInst *CleanupPad,
                                             BasicBlock *UnwindBB,
                                             BBIterator WhereIt,
                                             BasicBlock *WhereBB,
                                             Context &Ctx) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  auto *LLVMUnwindBB =
      UnwindBB != nullptr ? cast<llvm::BasicBlock>(UnwindBB->Val) : nullptr;
  llvm::CleanupReturnInst *LLVMI = Builder.CreateCleanupRet(
      cast<llvm::CleanupPadInst>(CleanupPad->Val), LLVMUnwindBB);
  return Ctx.createCleanupReturnInst(LLVMI);
}

CleanupPadInst *CleanupReturnInst::getCleanupPad() const {
  return cast<CleanupPadInst>(
      Ctx.getValue(cast<llvm::CleanupReturnInst>(Val)->getCleanupPad()));
}

void CleanupReturnInst::setCleanupPad(CleanupPadInst *CleanupPad) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&CleanupReturnInst::getCleanupPad,
                                       &CleanupReturnInst::setCleanupPad>>(
          this);
  cast<llvm::CleanupReturnInst>(Val)->setCleanupPad(
      cast<llvm::CleanupPadInst>(CleanupPad->Val));
}

BasicBlock *CleanupReturnInst::getUnwindDest() const {
  return cast_or_null<BasicBlock>(
      Ctx.getValue(cast<llvm::CleanupReturnInst>(Val)->getUnwindDest()));
}

void CleanupReturnInst::setUnwindDest(BasicBlock *NewDest) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&CleanupReturnInst::getUnwindDest,
                                       &CleanupReturnInst::setUnwindDest>>(
          this);
  cast<llvm::CleanupReturnInst>(Val)->setUnwindDest(
      cast<llvm::BasicBlock>(NewDest->Val));
}

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
  llvm::Value *NewV =
      Builder.CreateGEP(Ty->LLVMTy, Ptr->Val, LLVMIdxList, NameStr);
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

Type *GetElementPtrInst::getSourceElementType() const {
  return Ctx.getType(
      cast<llvm::GetElementPtrInst>(Val)->getSourceElementType());
}

Type *GetElementPtrInst::getResultElementType() const {
  return Ctx.getType(
      cast<llvm::GetElementPtrInst>(Val)->getResultElementType());
}

Value *GetElementPtrInst::getPointerOperand() const {
  return Ctx.getValue(cast<llvm::GetElementPtrInst>(Val)->getPointerOperand());
}

Type *GetElementPtrInst::getPointerOperandType() const {
  return Ctx.getType(
      cast<llvm::GetElementPtrInst>(Val)->getPointerOperandType());
}

BasicBlock *PHINode::LLVMBBToBB::operator()(llvm::BasicBlock *LLVMBB) const {
  return cast<BasicBlock>(Ctx.getValue(LLVMBB));
}

PHINode *PHINode::create(Type *Ty, unsigned NumReservedValues,
                         Instruction *InsertBefore, Context &Ctx,
                         const Twine &Name) {
  llvm::PHINode *NewPHI = llvm::PHINode::Create(
      Ty->LLVMTy, NumReservedValues, Name,
      InsertBefore->getTopmostLLVMInstruction()->getIterator());
  return Ctx.createPHINode(NewPHI);
}

bool PHINode::classof(const Value *From) {
  return From->getSubclassID() == ClassID::PHI;
}

Value *PHINode::getIncomingValue(unsigned Idx) const {
  return Ctx.getValue(cast<llvm::PHINode>(Val)->getIncomingValue(Idx));
}
void PHINode::setIncomingValue(unsigned Idx, Value *V) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetterWithIdx<&PHINode::getIncomingValue,
                                              &PHINode::setIncomingValue>>(this,
                                                                           Idx);
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
  // Helper to disambiguate PHINode::getIncomingBlock(unsigned).
  constexpr BasicBlock *(PHINode::*GetIncomingBlockFn)(unsigned) const =
      &PHINode::getIncomingBlock;
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetterWithIdx<GetIncomingBlockFn, &PHINode::setIncomingBlock>>(
          this, Idx);
  cast<llvm::PHINode>(Val)->setIncomingBlock(Idx,
                                             cast<llvm::BasicBlock>(BB->Val));
}
void PHINode::addIncoming(Value *V, BasicBlock *BB) {
  auto &Tracker = Ctx.getTracker();
  Tracker.emplaceIfTracking<PHIAddIncoming>(this);

  cast<llvm::PHINode>(Val)->addIncoming(V->Val,
                                        cast<llvm::BasicBlock>(BB->Val));
}
Value *PHINode::removeIncomingValue(unsigned Idx) {
  auto &Tracker = Ctx.getTracker();
  Tracker.emplaceIfTracking<PHIRemoveIncoming>(this, Idx);
  llvm::Value *LLVMV =
      cast<llvm::PHINode>(Val)->removeIncomingValue(Idx,
                                                    /*DeletePHIIfEmpty=*/false);
  return Ctx.getValue(LLVMV);
}
Value *PHINode::removeIncomingValue(BasicBlock *BB) {
  auto &Tracker = Ctx.getTracker();
  Tracker.emplaceIfTracking<PHIRemoveIncoming>(this, getBasicBlockIndex(BB));

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

/// \Returns the LLVM opcode that corresponds to \p Opc.
static llvm::Instruction::UnaryOps getLLVMUnaryOp(Instruction::Opcode Opc) {
  switch (Opc) {
  case Instruction::Opcode::FNeg:
    return static_cast<llvm::Instruction::UnaryOps>(llvm::Instruction::FNeg);
  default:
    llvm_unreachable("Not a unary op!");
  }
}

CatchSwitchInst *CatchSwitchInst::create(Value *ParentPad, BasicBlock *UnwindBB,
                                         unsigned NumHandlers,
                                         BBIterator WhereIt,
                                         BasicBlock *WhereBB, Context &Ctx,
                                         const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  llvm::CatchSwitchInst *LLVMCSI = Builder.CreateCatchSwitch(
      ParentPad->Val, cast<llvm::BasicBlock>(UnwindBB->Val), NumHandlers, Name);
  return Ctx.createCatchSwitchInst(LLVMCSI);
}

Value *CatchSwitchInst::getParentPad() const {
  return Ctx.getValue(cast<llvm::CatchSwitchInst>(Val)->getParentPad());
}

void CatchSwitchInst::setParentPad(Value *ParentPad) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&CatchSwitchInst::getParentPad,
                                       &CatchSwitchInst::setParentPad>>(this);
  cast<llvm::CatchSwitchInst>(Val)->setParentPad(ParentPad->Val);
}

BasicBlock *CatchSwitchInst::getUnwindDest() const {
  return cast_or_null<BasicBlock>(
      Ctx.getValue(cast<llvm::CatchSwitchInst>(Val)->getUnwindDest()));
}

void CatchSwitchInst::setUnwindDest(BasicBlock *UnwindDest) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&CatchSwitchInst::getUnwindDest,
                                       &CatchSwitchInst::setUnwindDest>>(this);
  cast<llvm::CatchSwitchInst>(Val)->setUnwindDest(
      cast<llvm::BasicBlock>(UnwindDest->Val));
}

void CatchSwitchInst::addHandler(BasicBlock *Dest) {
  Ctx.getTracker().emplaceIfTracking<CatchSwitchAddHandler>(this);
  cast<llvm::CatchSwitchInst>(Val)->addHandler(
      cast<llvm::BasicBlock>(Dest->Val));
}

ResumeInst *ResumeInst::create(Value *Exn, BBIterator WhereIt,
                               BasicBlock *WhereBB, Context &Ctx) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  auto *LLVMI = cast<llvm::ResumeInst>(Builder.CreateResume(Exn->Val));
  return Ctx.createResumeInst(LLVMI);
}

Value *ResumeInst::getValue() const {
  return Ctx.getValue(cast<llvm::ResumeInst>(Val)->getValue());
}

SwitchInst *SwitchInst::create(Value *V, BasicBlock *Dest, unsigned NumCases,
                               BasicBlock::iterator WhereIt,
                               BasicBlock *WhereBB, Context &Ctx,
                               const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  llvm::SwitchInst *LLVMSwitch =
      Builder.CreateSwitch(V->Val, cast<llvm::BasicBlock>(Dest->Val), NumCases);
  return Ctx.createSwitchInst(LLVMSwitch);
}

Value *SwitchInst::getCondition() const {
  return Ctx.getValue(cast<llvm::SwitchInst>(Val)->getCondition());
}

void SwitchInst::setCondition(Value *V) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&SwitchInst::getCondition, &SwitchInst::setCondition>>(
          this);
  cast<llvm::SwitchInst>(Val)->setCondition(V->Val);
}

BasicBlock *SwitchInst::getDefaultDest() const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::SwitchInst>(Val)->getDefaultDest()));
}

void SwitchInst::setDefaultDest(BasicBlock *DefaultCase) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&SwitchInst::getDefaultDest,
                                       &SwitchInst::setDefaultDest>>(this);
  cast<llvm::SwitchInst>(Val)->setDefaultDest(
      cast<llvm::BasicBlock>(DefaultCase->Val));
}
ConstantInt *SwitchInst::findCaseDest(BasicBlock *BB) {
  auto *LLVMC = cast<llvm::SwitchInst>(Val)->findCaseDest(
      cast<llvm::BasicBlock>(BB->Val));
  return LLVMC != nullptr ? cast<ConstantInt>(Ctx.getValue(LLVMC)) : nullptr;
}

void SwitchInst::addCase(ConstantInt *OnVal, BasicBlock *Dest) {
  Ctx.getTracker().emplaceIfTracking<SwitchAddCase>(this, OnVal);
  // TODO: Track this!
  cast<llvm::SwitchInst>(Val)->addCase(cast<llvm::ConstantInt>(OnVal->Val),
                                       cast<llvm::BasicBlock>(Dest->Val));
}

SwitchInst::CaseIt SwitchInst::removeCase(CaseIt It) {
  auto &Case = *It;
  Ctx.getTracker().emplaceIfTracking<SwitchRemoveCase>(
      this, Case.getCaseValue(), Case.getCaseSuccessor());

  auto *LLVMSwitch = cast<llvm::SwitchInst>(Val);
  unsigned CaseNum = It - case_begin();
  llvm::SwitchInst::CaseIt LLVMIt(LLVMSwitch, CaseNum);
  auto LLVMCaseIt = LLVMSwitch->removeCase(LLVMIt);
  unsigned Num = LLVMCaseIt - LLVMSwitch->case_begin();
  return CaseIt(this, Num);
}

BasicBlock *SwitchInst::getSuccessor(unsigned Idx) const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::SwitchInst>(Val)->getSuccessor(Idx)));
}

void SwitchInst::setSuccessor(unsigned Idx, BasicBlock *NewSucc) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetterWithIdx<&SwitchInst::getSuccessor,
                                              &SwitchInst::setSuccessor>>(this,
                                                                          Idx);
  cast<llvm::SwitchInst>(Val)->setSuccessor(
      Idx, cast<llvm::BasicBlock>(NewSucc->Val));
}

Value *UnaryOperator::create(Instruction::Opcode Op, Value *OpV,
                             BBIterator WhereIt, BasicBlock *WhereBB,
                             Context &Ctx, const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt == WhereBB->end())
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  else
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  auto *NewLLVMV = Builder.CreateUnOp(getLLVMUnaryOp(Op), OpV->Val, Name);
  if (auto *NewUnOpV = dyn_cast<llvm::UnaryOperator>(NewLLVMV)) {
    return Ctx.createUnaryOperator(NewUnOpV);
  }
  assert(isa<llvm::Constant>(NewLLVMV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewLLVMV));
}

Value *UnaryOperator::create(Instruction::Opcode Op, Value *OpV,
                             Instruction *InsertBefore, Context &Ctx,
                             const Twine &Name) {
  return create(Op, OpV, InsertBefore->getIterator(), InsertBefore->getParent(),
                Ctx, Name);
}

Value *UnaryOperator::create(Instruction::Opcode Op, Value *OpV,
                             BasicBlock *InsertAfter, Context &Ctx,
                             const Twine &Name) {
  return create(Op, OpV, InsertAfter->end(), InsertAfter, Ctx, Name);
}

Value *UnaryOperator::createWithCopiedFlags(Instruction::Opcode Op, Value *OpV,
                                            Value *CopyFrom, BBIterator WhereIt,
                                            BasicBlock *WhereBB, Context &Ctx,
                                            const Twine &Name) {
  auto *NewV = create(Op, OpV, WhereIt, WhereBB, Ctx, Name);
  if (auto *UnI = dyn_cast<llvm::UnaryOperator>(NewV->Val))
    UnI->copyIRFlags(CopyFrom->Val);
  return NewV;
}

Value *UnaryOperator::createWithCopiedFlags(Instruction::Opcode Op, Value *OpV,
                                            Value *CopyFrom,
                                            Instruction *InsertBefore,
                                            Context &Ctx, const Twine &Name) {
  return createWithCopiedFlags(Op, OpV, CopyFrom, InsertBefore->getIterator(),
                               InsertBefore->getParent(), Ctx, Name);
}

Value *UnaryOperator::createWithCopiedFlags(Instruction::Opcode Op, Value *OpV,
                                            Value *CopyFrom,
                                            BasicBlock *InsertAtEnd,
                                            Context &Ctx, const Twine &Name) {
  return createWithCopiedFlags(Op, OpV, CopyFrom, InsertAtEnd->end(),
                               InsertAtEnd, Ctx, Name);
}

/// \Returns the LLVM opcode that corresponds to \p Opc.
static llvm::Instruction::BinaryOps getLLVMBinaryOp(Instruction::Opcode Opc) {
  switch (Opc) {
  case Instruction::Opcode::Add:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::Add);
  case Instruction::Opcode::FAdd:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::FAdd);
  case Instruction::Opcode::Sub:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::Sub);
  case Instruction::Opcode::FSub:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::FSub);
  case Instruction::Opcode::Mul:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::Mul);
  case Instruction::Opcode::FMul:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::FMul);
  case Instruction::Opcode::UDiv:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::UDiv);
  case Instruction::Opcode::SDiv:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::SDiv);
  case Instruction::Opcode::FDiv:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::FDiv);
  case Instruction::Opcode::URem:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::URem);
  case Instruction::Opcode::SRem:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::SRem);
  case Instruction::Opcode::FRem:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::FRem);
  case Instruction::Opcode::Shl:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::Shl);
  case Instruction::Opcode::LShr:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::LShr);
  case Instruction::Opcode::AShr:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::AShr);
  case Instruction::Opcode::And:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::And);
  case Instruction::Opcode::Or:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::Or);
  case Instruction::Opcode::Xor:
    return static_cast<llvm::Instruction::BinaryOps>(llvm::Instruction::Xor);
  default:
    llvm_unreachable("Not a binary op!");
  }
}
Value *BinaryOperator::create(Instruction::Opcode Op, Value *LHS, Value *RHS,
                              BBIterator WhereIt, BasicBlock *WhereBB,
                              Context &Ctx, const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt == WhereBB->end())
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  else
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  llvm::Value *NewV =
      Builder.CreateBinOp(getLLVMBinaryOp(Op), LHS->Val, RHS->Val, Name);
  if (auto *NewBinOp = dyn_cast<llvm::BinaryOperator>(NewV))
    return Ctx.createBinaryOperator(NewBinOp);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *BinaryOperator::create(Instruction::Opcode Op, Value *LHS, Value *RHS,
                              Instruction *InsertBefore, Context &Ctx,
                              const Twine &Name) {
  return create(Op, LHS, RHS, InsertBefore->getIterator(),
                InsertBefore->getParent(), Ctx, Name);
}

Value *BinaryOperator::create(Instruction::Opcode Op, Value *LHS, Value *RHS,
                              BasicBlock *InsertAtEnd, Context &Ctx,
                              const Twine &Name) {
  return create(Op, LHS, RHS, InsertAtEnd->end(), InsertAtEnd, Ctx, Name);
}

Value *BinaryOperator::createWithCopiedFlags(Instruction::Opcode Op, Value *LHS,
                                             Value *RHS, Value *CopyFrom,
                                             BBIterator WhereIt,
                                             BasicBlock *WhereBB, Context &Ctx,
                                             const Twine &Name) {

  Value *NewV = create(Op, LHS, RHS, WhereIt, WhereBB, Ctx, Name);
  if (auto *NewBO = dyn_cast<BinaryOperator>(NewV))
    cast<llvm::BinaryOperator>(NewBO->Val)->copyIRFlags(CopyFrom->Val);
  return NewV;
}

Value *BinaryOperator::createWithCopiedFlags(Instruction::Opcode Op, Value *LHS,
                                             Value *RHS, Value *CopyFrom,
                                             Instruction *InsertBefore,
                                             Context &Ctx, const Twine &Name) {
  return createWithCopiedFlags(Op, LHS, RHS, CopyFrom,
                               InsertBefore->getIterator(),
                               InsertBefore->getParent(), Ctx, Name);
}

Value *BinaryOperator::createWithCopiedFlags(Instruction::Opcode Op, Value *LHS,
                                             Value *RHS, Value *CopyFrom,
                                             BasicBlock *InsertAtEnd,
                                             Context &Ctx, const Twine &Name) {
  return createWithCopiedFlags(Op, LHS, RHS, CopyFrom, InsertAtEnd->end(),
                               InsertAtEnd, Ctx, Name);
}

void PossiblyDisjointInst::setIsDisjoint(bool B) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&PossiblyDisjointInst::isDisjoint,
                                       &PossiblyDisjointInst::setIsDisjoint>>(
          this);
  cast<llvm::PossiblyDisjointInst>(Val)->setIsDisjoint(B);
}

void AtomicRMWInst::setAlignment(Align Align) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&AtomicRMWInst::getAlign,
                                       &AtomicRMWInst::setAlignment>>(this);
  cast<llvm::AtomicRMWInst>(Val)->setAlignment(Align);
}

void AtomicRMWInst::setVolatile(bool V) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&AtomicRMWInst::isVolatile,
                                       &AtomicRMWInst::setVolatile>>(this);
  cast<llvm::AtomicRMWInst>(Val)->setVolatile(V);
}

void AtomicRMWInst::setOrdering(AtomicOrdering Ordering) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&AtomicRMWInst::getOrdering,
                                       &AtomicRMWInst::setOrdering>>(this);
  cast<llvm::AtomicRMWInst>(Val)->setOrdering(Ordering);
}

void AtomicRMWInst::setSyncScopeID(SyncScope::ID SSID) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&AtomicRMWInst::getSyncScopeID,
                                       &AtomicRMWInst::setSyncScopeID>>(this);
  cast<llvm::AtomicRMWInst>(Val)->setSyncScopeID(SSID);
}

Value *AtomicRMWInst::getPointerOperand() {
  return Ctx.getValue(cast<llvm::AtomicRMWInst>(Val)->getPointerOperand());
}

Value *AtomicRMWInst::getValOperand() {
  return Ctx.getValue(cast<llvm::AtomicRMWInst>(Val)->getValOperand());
}

AtomicRMWInst *AtomicRMWInst::create(BinOp Op, Value *Ptr, Value *Val,
                                     MaybeAlign Align, AtomicOrdering Ordering,
                                     BBIterator WhereIt, BasicBlock *WhereBB,
                                     Context &Ctx, SyncScope::ID SSID,
                                     const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt == WhereBB->end())
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  else
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  auto *LLVMAtomicRMW =
      Builder.CreateAtomicRMW(Op, Ptr->Val, Val->Val, Align, Ordering, SSID);
  LLVMAtomicRMW->setName(Name);
  return Ctx.createAtomicRMWInst(LLVMAtomicRMW);
}

AtomicRMWInst *AtomicRMWInst::create(BinOp Op, Value *Ptr, Value *Val,
                                     MaybeAlign Align, AtomicOrdering Ordering,
                                     Instruction *InsertBefore, Context &Ctx,
                                     SyncScope::ID SSID, const Twine &Name) {
  return create(Op, Ptr, Val, Align, Ordering, InsertBefore->getIterator(),
                InsertBefore->getParent(), Ctx, SSID, Name);
}

AtomicRMWInst *AtomicRMWInst::create(BinOp Op, Value *Ptr, Value *Val,
                                     MaybeAlign Align, AtomicOrdering Ordering,
                                     BasicBlock *InsertAtEnd, Context &Ctx,
                                     SyncScope::ID SSID, const Twine &Name) {
  return create(Op, Ptr, Val, Align, Ordering, InsertAtEnd->end(), InsertAtEnd,
                Ctx, SSID, Name);
}

void AtomicCmpXchgInst::setSyncScopeID(SyncScope::ID SSID) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&AtomicCmpXchgInst::getSyncScopeID,
                                       &AtomicCmpXchgInst::setSyncScopeID>>(
          this);
  cast<llvm::AtomicCmpXchgInst>(Val)->setSyncScopeID(SSID);
}

Value *AtomicCmpXchgInst::getPointerOperand() {
  return Ctx.getValue(cast<llvm::AtomicCmpXchgInst>(Val)->getPointerOperand());
}

Value *AtomicCmpXchgInst::getCompareOperand() {
  return Ctx.getValue(cast<llvm::AtomicCmpXchgInst>(Val)->getCompareOperand());
}

Value *AtomicCmpXchgInst::getNewValOperand() {
  return Ctx.getValue(cast<llvm::AtomicCmpXchgInst>(Val)->getNewValOperand());
}

AtomicCmpXchgInst *
AtomicCmpXchgInst::create(Value *Ptr, Value *Cmp, Value *New, MaybeAlign Align,
                          AtomicOrdering SuccessOrdering,
                          AtomicOrdering FailureOrdering, BBIterator WhereIt,
                          BasicBlock *WhereBB, Context &Ctx, SyncScope::ID SSID,
                          const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt == WhereBB->end())
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  else
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  auto *LLVMAtomicCmpXchg =
      Builder.CreateAtomicCmpXchg(Ptr->Val, Cmp->Val, New->Val, Align,
                                  SuccessOrdering, FailureOrdering, SSID);
  LLVMAtomicCmpXchg->setName(Name);
  return Ctx.createAtomicCmpXchgInst(LLVMAtomicCmpXchg);
}

AtomicCmpXchgInst *AtomicCmpXchgInst::create(Value *Ptr, Value *Cmp, Value *New,
                                             MaybeAlign Align,
                                             AtomicOrdering SuccessOrdering,
                                             AtomicOrdering FailureOrdering,
                                             Instruction *InsertBefore,
                                             Context &Ctx, SyncScope::ID SSID,
                                             const Twine &Name) {
  return create(Ptr, Cmp, New, Align, SuccessOrdering, FailureOrdering,
                InsertBefore->getIterator(), InsertBefore->getParent(), Ctx,
                SSID, Name);
}

AtomicCmpXchgInst *AtomicCmpXchgInst::create(Value *Ptr, Value *Cmp, Value *New,
                                             MaybeAlign Align,
                                             AtomicOrdering SuccessOrdering,
                                             AtomicOrdering FailureOrdering,
                                             BasicBlock *InsertAtEnd,
                                             Context &Ctx, SyncScope::ID SSID,
                                             const Twine &Name) {
  return create(Ptr, Cmp, New, Align, SuccessOrdering, FailureOrdering,
                InsertAtEnd->end(), InsertAtEnd, Ctx, SSID, Name);
}

void AtomicCmpXchgInst::setAlignment(Align Align) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&AtomicCmpXchgInst::getAlign,
                                       &AtomicCmpXchgInst::setAlignment>>(this);
  cast<llvm::AtomicCmpXchgInst>(Val)->setAlignment(Align);
}

void AtomicCmpXchgInst::setVolatile(bool V) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&AtomicCmpXchgInst::isVolatile,
                                       &AtomicCmpXchgInst::setVolatile>>(this);
  cast<llvm::AtomicCmpXchgInst>(Val)->setVolatile(V);
}

void AtomicCmpXchgInst::setWeak(bool IsWeak) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&AtomicCmpXchgInst::isWeak,
                                       &AtomicCmpXchgInst::setWeak>>(this);
  cast<llvm::AtomicCmpXchgInst>(Val)->setWeak(IsWeak);
}

void AtomicCmpXchgInst::setSuccessOrdering(AtomicOrdering Ordering) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&AtomicCmpXchgInst::getSuccessOrdering,
                                       &AtomicCmpXchgInst::setSuccessOrdering>>(
          this);
  cast<llvm::AtomicCmpXchgInst>(Val)->setSuccessOrdering(Ordering);
}

void AtomicCmpXchgInst::setFailureOrdering(AtomicOrdering Ordering) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&AtomicCmpXchgInst::getFailureOrdering,
                                       &AtomicCmpXchgInst::setFailureOrdering>>(
          this);
  cast<llvm::AtomicCmpXchgInst>(Val)->setFailureOrdering(Ordering);
}

AllocaInst *AllocaInst::create(Type *Ty, unsigned AddrSpace, BBIterator WhereIt,
                               BasicBlock *WhereBB, Context &Ctx,
                               Value *ArraySize, const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt == WhereBB->end())
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  else
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  auto *NewAlloca =
      Builder.CreateAlloca(Ty->LLVMTy, AddrSpace, ArraySize->Val, Name);
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

Type *AllocaInst::getAllocatedType() const {
  return Ctx.getType(cast<llvm::AllocaInst>(Val)->getAllocatedType());
}

void AllocaInst::setAllocatedType(Type *Ty) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&AllocaInst::getAllocatedType,
                                       &AllocaInst::setAllocatedType>>(this);
  cast<llvm::AllocaInst>(Val)->setAllocatedType(Ty->LLVMTy);
}

void AllocaInst::setAlignment(Align Align) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&AllocaInst::getAlign, &AllocaInst::setAlignment>>(
          this);
  cast<llvm::AllocaInst>(Val)->setAlignment(Align);
}

void AllocaInst::setUsedWithInAlloca(bool V) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&AllocaInst::isUsedWithInAlloca,
                                       &AllocaInst::setUsedWithInAlloca>>(this);
  cast<llvm::AllocaInst>(Val)->setUsedWithInAlloca(V);
}

Value *AllocaInst::getArraySize() {
  return Ctx.getValue(cast<llvm::AllocaInst>(Val)->getArraySize());
}

PointerType *AllocaInst::getType() const {
  return cast<PointerType>(Ctx.getType(cast<llvm::AllocaInst>(Val)->getType()));
}

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
      Builder.CreateCast(getLLVMCastOp(Op), Operand->Val, DestTy->LLVMTy, Name);
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

Type *CastInst::getSrcTy() const {
  return Ctx.getType(cast<llvm::CastInst>(Val)->getSrcTy());
}

Type *CastInst::getDestTy() const {
  return Ctx.getType(cast<llvm::CastInst>(Val)->getDestTy());
}

void PossiblyNonNegInst::setNonNeg(bool B) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&PossiblyNonNegInst::hasNonNeg,
                                       &PossiblyNonNegInst::setNonNeg>>(this);
  cast<llvm::PossiblyNonNegInst>(Val)->setNonNeg(B);
}

Value *InsertElementInst::create(Value *Vec, Value *NewElt, Value *Idx,
                                 Instruction *InsertBefore, Context &Ctx,
                                 const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertBefore->getTopmostLLVMInstruction());
  llvm::Value *NewV =
      Builder.CreateInsertElement(Vec->Val, NewElt->Val, Idx->Val, Name);
  if (auto *NewInsert = dyn_cast<llvm::InsertElementInst>(NewV))
    return Ctx.createInsertElementInst(NewInsert);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *InsertElementInst::create(Value *Vec, Value *NewElt, Value *Idx,
                                 BasicBlock *InsertAtEnd, Context &Ctx,
                                 const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(cast<llvm::BasicBlock>(InsertAtEnd->Val));
  llvm::Value *NewV =
      Builder.CreateInsertElement(Vec->Val, NewElt->Val, Idx->Val, Name);
  if (auto *NewInsert = dyn_cast<llvm::InsertElementInst>(NewV))
    return Ctx.createInsertElementInst(NewInsert);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *ExtractElementInst::create(Value *Vec, Value *Idx,
                                  Instruction *InsertBefore, Context &Ctx,
                                  const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertBefore->getTopmostLLVMInstruction());
  llvm::Value *NewV = Builder.CreateExtractElement(Vec->Val, Idx->Val, Name);
  if (auto *NewExtract = dyn_cast<llvm::ExtractElementInst>(NewV))
    return Ctx.createExtractElementInst(NewExtract);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *ExtractElementInst::create(Value *Vec, Value *Idx,
                                  BasicBlock *InsertAtEnd, Context &Ctx,
                                  const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(cast<llvm::BasicBlock>(InsertAtEnd->Val));
  llvm::Value *NewV = Builder.CreateExtractElement(Vec->Val, Idx->Val, Name);
  if (auto *NewExtract = dyn_cast<llvm::ExtractElementInst>(NewV))
    return Ctx.createExtractElementInst(NewExtract);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *ShuffleVectorInst::create(Value *V1, Value *V2, Value *Mask,
                                 Instruction *InsertBefore, Context &Ctx,
                                 const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertBefore->getTopmostLLVMInstruction());
  llvm::Value *NewV =
      Builder.CreateShuffleVector(V1->Val, V2->Val, Mask->Val, Name);
  if (auto *NewShuffle = dyn_cast<llvm::ShuffleVectorInst>(NewV))
    return Ctx.createShuffleVectorInst(NewShuffle);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *ShuffleVectorInst::create(Value *V1, Value *V2, Value *Mask,
                                 BasicBlock *InsertAtEnd, Context &Ctx,
                                 const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(cast<llvm::BasicBlock>(InsertAtEnd->Val));
  llvm::Value *NewV =
      Builder.CreateShuffleVector(V1->Val, V2->Val, Mask->Val, Name);
  if (auto *NewShuffle = dyn_cast<llvm::ShuffleVectorInst>(NewV))
    return Ctx.createShuffleVectorInst(NewShuffle);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *ShuffleVectorInst::create(Value *V1, Value *V2, ArrayRef<int> Mask,
                                 Instruction *InsertBefore, Context &Ctx,
                                 const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertBefore->getTopmostLLVMInstruction());
  llvm::Value *NewV = Builder.CreateShuffleVector(V1->Val, V2->Val, Mask, Name);
  if (auto *NewShuffle = dyn_cast<llvm::ShuffleVectorInst>(NewV))
    return Ctx.createShuffleVectorInst(NewShuffle);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Value *ShuffleVectorInst::create(Value *V1, Value *V2, ArrayRef<int> Mask,
                                 BasicBlock *InsertAtEnd, Context &Ctx,
                                 const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(cast<llvm::BasicBlock>(InsertAtEnd->Val));
  llvm::Value *NewV = Builder.CreateShuffleVector(V1->Val, V2->Val, Mask, Name);
  if (auto *NewShuffle = dyn_cast<llvm::ShuffleVectorInst>(NewV))
    return Ctx.createShuffleVectorInst(NewShuffle);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

void ShuffleVectorInst::setShuffleMask(ArrayRef<int> Mask) {
  Ctx.getTracker().emplaceIfTracking<ShuffleVectorSetMask>(this);
  cast<llvm::ShuffleVectorInst>(Val)->setShuffleMask(Mask);
}

VectorType *ShuffleVectorInst::getType() const {
  return cast<VectorType>(
      Ctx.getType(cast<llvm::ShuffleVectorInst>(Val)->getType()));
}

void ShuffleVectorInst::commute() {
  Ctx.getTracker().emplaceIfTracking<ShuffleVectorSetMask>(this);
  Ctx.getTracker().emplaceIfTracking<UseSwap>(getOperandUse(0),
                                              getOperandUse(1));
  cast<llvm::ShuffleVectorInst>(Val)->commute();
}

Constant *ShuffleVectorInst::getShuffleMaskForBitcode() const {
  return Ctx.getOrCreateConstant(
      cast<llvm::ShuffleVectorInst>(Val)->getShuffleMaskForBitcode());
}

Constant *ShuffleVectorInst::convertShuffleMaskForBitcode(ArrayRef<int> Mask,
                                                          Type *ResultTy) {
  return ResultTy->getContext().getOrCreateConstant(
      llvm::ShuffleVectorInst::convertShuffleMaskForBitcode(Mask,
                                                            ResultTy->LLVMTy));
}

VectorType *ExtractElementInst::getVectorOperandType() const {
  return cast<VectorType>(Ctx.getType(getVectorOperand()->getType()->LLVMTy));
}

Value *ExtractValueInst::create(Value *Agg, ArrayRef<unsigned> Idxs,
                                BBIterator WhereIt, BasicBlock *WhereBB,
                                Context &Ctx, const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  llvm::Value *NewV = Builder.CreateExtractValue(Agg->Val, Idxs, Name);
  if (auto *NewExtractValueInst = dyn_cast<llvm::ExtractValueInst>(NewV))
    return Ctx.createExtractValueInst(NewExtractValueInst);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

Type *ExtractValueInst::getIndexedType(Type *Agg, ArrayRef<unsigned> Idxs) {
  auto *LLVMTy = llvm::ExtractValueInst::getIndexedType(Agg->LLVMTy, Idxs);
  return Agg->getContext().getType(LLVMTy);
}

Value *InsertValueInst::create(Value *Agg, Value *Val, ArrayRef<unsigned> Idxs,
                               BBIterator WhereIt, BasicBlock *WhereBB,
                               Context &Ctx, const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  if (WhereIt != WhereBB->end())
    Builder.SetInsertPoint((*WhereIt).getTopmostLLVMInstruction());
  else
    Builder.SetInsertPoint(cast<llvm::BasicBlock>(WhereBB->Val));
  llvm::Value *NewV = Builder.CreateInsertValue(Agg->Val, Val->Val, Idxs, Name);
  if (auto *NewInsertValueInst = dyn_cast<llvm::InsertValueInst>(NewV))
    return Ctx.createInsertValueInst(NewInsertValueInst);
  assert(isa<llvm::Constant>(NewV) && "Expected constant");
  return Ctx.getOrCreateConstant(cast<llvm::Constant>(NewV));
}

#ifndef NDEBUG
void Constant::dumpOS(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}
#endif // NDEBUG

ConstantInt *ConstantInt::getTrue(Context &Ctx) {
  auto *LLVMC = llvm::ConstantInt::getTrue(Ctx.LLVMCtx);
  return cast<ConstantInt>(Ctx.getOrCreateConstant(LLVMC));
}
ConstantInt *ConstantInt::getFalse(Context &Ctx) {
  auto *LLVMC = llvm::ConstantInt::getFalse(Ctx.LLVMCtx);
  return cast<ConstantInt>(Ctx.getOrCreateConstant(LLVMC));
}
ConstantInt *ConstantInt::getBool(Context &Ctx, bool V) {
  auto *LLVMC = llvm::ConstantInt::getBool(Ctx.LLVMCtx, V);
  return cast<ConstantInt>(Ctx.getOrCreateConstant(LLVMC));
}
Constant *ConstantInt::getTrue(Type *Ty) {
  auto *LLVMC = llvm::ConstantInt::getTrue(Ty->LLVMTy);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}
Constant *ConstantInt::getFalse(Type *Ty) {
  auto *LLVMC = llvm::ConstantInt::getFalse(Ty->LLVMTy);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}
Constant *ConstantInt::getBool(Type *Ty, bool V) {
  auto *LLVMC = llvm::ConstantInt::getBool(Ty->LLVMTy, V);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}
ConstantInt *ConstantInt::get(Type *Ty, uint64_t V, bool IsSigned) {
  auto *LLVMC = llvm::ConstantInt::get(Ty->LLVMTy, V, IsSigned);
  return cast<ConstantInt>(Ty->getContext().getOrCreateConstant(LLVMC));
}
ConstantInt *ConstantInt::get(IntegerType *Ty, uint64_t V, bool IsSigned) {
  auto *LLVMC = llvm::ConstantInt::get(Ty->LLVMTy, V, IsSigned);
  return cast<ConstantInt>(Ty->getContext().getOrCreateConstant(LLVMC));
}
ConstantInt *ConstantInt::getSigned(IntegerType *Ty, int64_t V) {
  auto *LLVMC =
      llvm::ConstantInt::getSigned(cast<llvm::IntegerType>(Ty->LLVMTy), V);
  return cast<ConstantInt>(Ty->getContext().getOrCreateConstant(LLVMC));
}
Constant *ConstantInt::getSigned(Type *Ty, int64_t V) {
  auto *LLVMC = llvm::ConstantInt::getSigned(Ty->LLVMTy, V);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}
ConstantInt *ConstantInt::get(Context &Ctx, const APInt &V) {
  auto *LLVMC = llvm::ConstantInt::get(Ctx.LLVMCtx, V);
  return cast<ConstantInt>(Ctx.getOrCreateConstant(LLVMC));
}
ConstantInt *ConstantInt::get(IntegerType *Ty, StringRef Str, uint8_t Radix) {
  auto *LLVMC =
      llvm::ConstantInt::get(cast<llvm::IntegerType>(Ty->LLVMTy), Str, Radix);
  return cast<ConstantInt>(Ty->getContext().getOrCreateConstant(LLVMC));
}
Constant *ConstantInt::get(Type *Ty, const APInt &V) {
  auto *LLVMC = llvm::ConstantInt::get(Ty->LLVMTy, V);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}
IntegerType *ConstantInt::getIntegerType() const {
  auto *LLVMTy = cast<llvm::ConstantInt>(Val)->getIntegerType();
  return cast<IntegerType>(Ctx.getType(LLVMTy));
}

bool ConstantInt::isValueValidForType(Type *Ty, uint64_t V) {
  return llvm::ConstantInt::isValueValidForType(Ty->LLVMTy, V);
}
bool ConstantInt::isValueValidForType(Type *Ty, int64_t V) {
  return llvm::ConstantInt::isValueValidForType(Ty->LLVMTy, V);
}

Constant *ConstantFP::get(Type *Ty, double V) {
  auto *LLVMC = llvm::ConstantFP::get(Ty->LLVMTy, V);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}

Constant *ConstantFP::get(Type *Ty, const APFloat &V) {
  auto *LLVMC = llvm::ConstantFP::get(Ty->LLVMTy, V);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}

Constant *ConstantFP::get(Type *Ty, StringRef Str) {
  auto *LLVMC = llvm::ConstantFP::get(Ty->LLVMTy, Str);
  return Ty->getContext().getOrCreateConstant(LLVMC);
}

ConstantFP *ConstantFP::get(const APFloat &V, Context &Ctx) {
  auto *LLVMC = llvm::ConstantFP::get(Ctx.LLVMCtx, V);
  return cast<ConstantFP>(Ctx.getOrCreateConstant(LLVMC));
}

Constant *ConstantFP::getNaN(Type *Ty, bool Negative, uint64_t Payload) {
  auto *LLVMC = llvm::ConstantFP::getNaN(Ty->LLVMTy, Negative, Payload);
  return cast<Constant>(Ty->getContext().getOrCreateConstant(LLVMC));
}
Constant *ConstantFP::getQNaN(Type *Ty, bool Negative, APInt *Payload) {
  auto *LLVMC = llvm::ConstantFP::getQNaN(Ty->LLVMTy, Negative, Payload);
  return cast<Constant>(Ty->getContext().getOrCreateConstant(LLVMC));
}
Constant *ConstantFP::getSNaN(Type *Ty, bool Negative, APInt *Payload) {
  auto *LLVMC = llvm::ConstantFP::getSNaN(Ty->LLVMTy, Negative, Payload);
  return cast<Constant>(Ty->getContext().getOrCreateConstant(LLVMC));
}
Constant *ConstantFP::getZero(Type *Ty, bool Negative) {
  auto *LLVMC = llvm::ConstantFP::getZero(Ty->LLVMTy, Negative);
  return cast<Constant>(Ty->getContext().getOrCreateConstant(LLVMC));
}
Constant *ConstantFP::getNegativeZero(Type *Ty) {
  auto *LLVMC = llvm::ConstantFP::getNegativeZero(Ty->LLVMTy);
  return cast<Constant>(Ty->getContext().getOrCreateConstant(LLVMC));
}
Constant *ConstantFP::getInfinity(Type *Ty, bool Negative) {
  auto *LLVMC = llvm::ConstantFP::getInfinity(Ty->LLVMTy, Negative);
  return cast<Constant>(Ty->getContext().getOrCreateConstant(LLVMC));
}
bool ConstantFP::isValueValidForType(Type *Ty, const APFloat &V) {
  return llvm::ConstantFP::isValueValidForType(Ty->LLVMTy, V);
}

Constant *ConstantArray::get(ArrayType *T, ArrayRef<Constant *> V) {
  auto &Ctx = T->getContext();
  SmallVector<llvm::Constant *> LLVMValues;
  LLVMValues.reserve(V.size());
  for (auto *Elm : V)
    LLVMValues.push_back(cast<llvm::Constant>(Elm->Val));
  auto *LLVMC =
      llvm::ConstantArray::get(cast<llvm::ArrayType>(T->LLVMTy), LLVMValues);
  return cast<ConstantArray>(Ctx.getOrCreateConstant(LLVMC));
}

ArrayType *ConstantArray::getType() const {
  return cast<ArrayType>(
      Ctx.getType(cast<llvm::ConstantArray>(Val)->getType()));
}

Constant *ConstantStruct::get(StructType *T, ArrayRef<Constant *> V) {
  auto &Ctx = T->getContext();
  SmallVector<llvm::Constant *> LLVMValues;
  LLVMValues.reserve(V.size());
  for (auto *Elm : V)
    LLVMValues.push_back(cast<llvm::Constant>(Elm->Val));
  auto *LLVMC =
      llvm::ConstantStruct::get(cast<llvm::StructType>(T->LLVMTy), LLVMValues);
  return cast<ConstantStruct>(Ctx.getOrCreateConstant(LLVMC));
}

StructType *ConstantStruct::getTypeForElements(Context &Ctx,
                                               ArrayRef<Constant *> V,
                                               bool Packed) {
  unsigned VecSize = V.size();
  SmallVector<Type *, 16> EltTypes;
  EltTypes.reserve(VecSize);
  for (Constant *Elm : V)
    EltTypes.push_back(Elm->getType());
  return StructType::get(Ctx, EltTypes, Packed);
}

ConstantAggregateZero *ConstantAggregateZero::get(Type *Ty) {
  auto *LLVMC = llvm::ConstantAggregateZero::get(Ty->LLVMTy);
  return cast<ConstantAggregateZero>(
      Ty->getContext().getOrCreateConstant(LLVMC));
}

Constant *ConstantAggregateZero::getSequentialElement() const {
  return cast<Constant>(Ctx.getValue(
      cast<llvm::ConstantAggregateZero>(Val)->getSequentialElement()));
}
Constant *ConstantAggregateZero::getStructElement(unsigned Elt) const {
  return cast<Constant>(Ctx.getValue(
      cast<llvm::ConstantAggregateZero>(Val)->getStructElement(Elt)));
}
Constant *ConstantAggregateZero::getElementValue(Constant *C) const {
  return cast<Constant>(
      Ctx.getValue(cast<llvm::ConstantAggregateZero>(Val)->getElementValue(
          cast<llvm::Constant>(C->Val))));
}
Constant *ConstantAggregateZero::getElementValue(unsigned Idx) const {
  return cast<Constant>(Ctx.getValue(
      cast<llvm::ConstantAggregateZero>(Val)->getElementValue(Idx)));
}

ConstantPointerNull *ConstantPointerNull::get(PointerType *Ty) {
  auto *LLVMC =
      llvm::ConstantPointerNull::get(cast<llvm::PointerType>(Ty->LLVMTy));
  return cast<ConstantPointerNull>(Ty->getContext().getOrCreateConstant(LLVMC));
}

PointerType *ConstantPointerNull::getType() const {
  return cast<PointerType>(
      Ctx.getType(cast<llvm::ConstantPointerNull>(Val)->getType()));
}

UndefValue *UndefValue::get(Type *T) {
  auto *LLVMC = llvm::UndefValue::get(T->LLVMTy);
  return cast<UndefValue>(T->getContext().getOrCreateConstant(LLVMC));
}

UndefValue *UndefValue::getSequentialElement() const {
  return cast<UndefValue>(Ctx.getOrCreateConstant(
      cast<llvm::UndefValue>(Val)->getSequentialElement()));
}

UndefValue *UndefValue::getStructElement(unsigned Elt) const {
  return cast<UndefValue>(Ctx.getOrCreateConstant(
      cast<llvm::UndefValue>(Val)->getStructElement(Elt)));
}

UndefValue *UndefValue::getElementValue(Constant *C) const {
  return cast<UndefValue>(
      Ctx.getOrCreateConstant(cast<llvm::UndefValue>(Val)->getElementValue(
          cast<llvm::Constant>(C->Val))));
}

UndefValue *UndefValue::getElementValue(unsigned Idx) const {
  return cast<UndefValue>(Ctx.getOrCreateConstant(
      cast<llvm::UndefValue>(Val)->getElementValue(Idx)));
}

PoisonValue *PoisonValue::get(Type *T) {
  auto *LLVMC = llvm::PoisonValue::get(T->LLVMTy);
  return cast<PoisonValue>(T->getContext().getOrCreateConstant(LLVMC));
}

PoisonValue *PoisonValue::getSequentialElement() const {
  return cast<PoisonValue>(Ctx.getOrCreateConstant(
      cast<llvm::PoisonValue>(Val)->getSequentialElement()));
}

PoisonValue *PoisonValue::getStructElement(unsigned Elt) const {
  return cast<PoisonValue>(Ctx.getOrCreateConstant(
      cast<llvm::PoisonValue>(Val)->getStructElement(Elt)));
}

PoisonValue *PoisonValue::getElementValue(Constant *C) const {
  return cast<PoisonValue>(
      Ctx.getOrCreateConstant(cast<llvm::PoisonValue>(Val)->getElementValue(
          cast<llvm::Constant>(C->Val))));
}

PoisonValue *PoisonValue::getElementValue(unsigned Idx) const {
  return cast<PoisonValue>(Ctx.getOrCreateConstant(
      cast<llvm::PoisonValue>(Val)->getElementValue(Idx)));
}

void GlobalObject::setAlignment(MaybeAlign Align) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&GlobalObject::getAlign, &GlobalObject::setAlignment>>(
          this);
  cast<llvm::GlobalObject>(Val)->setAlignment(Align);
}

void GlobalObject::setGlobalObjectSubClassData(unsigned V) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&GlobalObject::getGlobalObjectSubClassData,
                        &GlobalObject::setGlobalObjectSubClassData>>(this);
  cast<llvm::GlobalObject>(Val)->setGlobalObjectSubClassData(V);
}

void GlobalObject::setSection(StringRef S) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&GlobalObject::getSection, &GlobalObject::setSection>>(
          this);
  cast<llvm::GlobalObject>(Val)->setSection(S);
}

template <typename GlobalT, typename LLVMGlobalT, typename ParentT,
          typename LLVMParentT>
GlobalT &GlobalWithNodeAPI<GlobalT, LLVMGlobalT, ParentT, LLVMParentT>::
    LLVMGVToGV::operator()(LLVMGlobalT &LLVMGV) const {
  return cast<GlobalT>(*Ctx.getValue(&LLVMGV));
}

namespace llvm::sandboxir {
// Explicit instantiations.
template class GlobalWithNodeAPI<GlobalIFunc, llvm::GlobalIFunc, GlobalObject,
                                 llvm::GlobalObject>;
template class GlobalWithNodeAPI<Function, llvm::Function, GlobalObject,
                                 llvm::GlobalObject>;
template class GlobalWithNodeAPI<GlobalVariable, llvm::GlobalVariable,
                                 GlobalObject, llvm::GlobalObject>;
template class GlobalWithNodeAPI<GlobalAlias, llvm::GlobalAlias, GlobalValue,
                                 llvm::GlobalValue>;
} // namespace llvm::sandboxir

void GlobalIFunc::setResolver(Constant *Resolver) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&GlobalIFunc::getResolver, &GlobalIFunc::setResolver>>(
          this);
  cast<llvm::GlobalIFunc>(Val)->setResolver(
      cast<llvm::Constant>(Resolver->Val));
}

Constant *GlobalIFunc::getResolver() const {
  return Ctx.getOrCreateConstant(cast<llvm::GlobalIFunc>(Val)->getResolver());
}

Function *GlobalIFunc::getResolverFunction() {
  return cast<Function>(Ctx.getOrCreateConstant(
      cast<llvm::GlobalIFunc>(Val)->getResolverFunction()));
}

GlobalVariable &
GlobalVariable::LLVMGVToGV::operator()(llvm::GlobalVariable &LLVMGV) const {
  return cast<GlobalVariable>(*Ctx.getValue(&LLVMGV));
}

Constant *GlobalVariable::getInitializer() const {
  return Ctx.getOrCreateConstant(
      cast<llvm::GlobalVariable>(Val)->getInitializer());
}

void GlobalVariable::setInitializer(Constant *InitVal) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&GlobalVariable::getInitializer,
                                       &GlobalVariable::setInitializer>>(this);
  cast<llvm::GlobalVariable>(Val)->setInitializer(
      cast<llvm::Constant>(InitVal->Val));
}

void GlobalVariable::setConstant(bool V) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&GlobalVariable::isConstant,
                                       &GlobalVariable::setConstant>>(this);
  cast<llvm::GlobalVariable>(Val)->setConstant(V);
}

void GlobalVariable::setExternallyInitialized(bool V) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&GlobalVariable::isExternallyInitialized,
                        &GlobalVariable::setExternallyInitialized>>(this);
  cast<llvm::GlobalVariable>(Val)->setExternallyInitialized(V);
}

void GlobalAlias::setAliasee(Constant *Aliasee) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&GlobalAlias::getAliasee, &GlobalAlias::setAliasee>>(
          this);
  cast<llvm::GlobalAlias>(Val)->setAliasee(cast<llvm::Constant>(Aliasee->Val));
}

Constant *GlobalAlias::getAliasee() const {
  return cast<Constant>(
      Ctx.getOrCreateConstant(cast<llvm::GlobalAlias>(Val)->getAliasee()));
}

const GlobalObject *GlobalAlias::getAliaseeObject() const {
  return cast<GlobalObject>(Ctx.getOrCreateConstant(
      cast<llvm::GlobalAlias>(Val)->getAliaseeObject()));
}

void GlobalValue::setUnnamedAddr(UnnamedAddr V) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&GlobalValue::getUnnamedAddr,
                                       &GlobalValue::setUnnamedAddr>>(this);
  cast<llvm::GlobalValue>(Val)->setUnnamedAddr(V);
}

void GlobalValue::setVisibility(VisibilityTypes V) {
  Ctx.getTracker()
      .emplaceIfTracking<GenericSetter<&GlobalValue::getVisibility,
                                       &GlobalValue::setVisibility>>(this);
  cast<llvm::GlobalValue>(Val)->setVisibility(V);
}

NoCFIValue *NoCFIValue::get(GlobalValue *GV) {
  auto *LLVMC = llvm::NoCFIValue::get(cast<llvm::GlobalValue>(GV->Val));
  return cast<NoCFIValue>(GV->getContext().getOrCreateConstant(LLVMC));
}

GlobalValue *NoCFIValue::getGlobalValue() const {
  auto *LLVMC = cast<llvm::NoCFIValue>(Val)->getGlobalValue();
  return cast<GlobalValue>(Ctx.getOrCreateConstant(LLVMC));
}

PointerType *NoCFIValue::getType() const {
  return cast<PointerType>(Ctx.getType(cast<llvm::NoCFIValue>(Val)->getType()));
}

ConstantPtrAuth *ConstantPtrAuth::get(Constant *Ptr, ConstantInt *Key,
                                      ConstantInt *Disc, Constant *AddrDisc) {
  auto *LLVMC = llvm::ConstantPtrAuth::get(
      cast<llvm::Constant>(Ptr->Val), cast<llvm::ConstantInt>(Key->Val),
      cast<llvm::ConstantInt>(Disc->Val), cast<llvm::Constant>(AddrDisc->Val));
  return cast<ConstantPtrAuth>(Ptr->getContext().getOrCreateConstant(LLVMC));
}

Constant *ConstantPtrAuth::getPointer() const {
  return Ctx.getOrCreateConstant(
      cast<llvm::ConstantPtrAuth>(Val)->getPointer());
}

ConstantInt *ConstantPtrAuth::getKey() const {
  return cast<ConstantInt>(
      Ctx.getOrCreateConstant(cast<llvm::ConstantPtrAuth>(Val)->getKey()));
}

ConstantInt *ConstantPtrAuth::getDiscriminator() const {
  return cast<ConstantInt>(Ctx.getOrCreateConstant(
      cast<llvm::ConstantPtrAuth>(Val)->getDiscriminator()));
}

Constant *ConstantPtrAuth::getAddrDiscriminator() const {
  return Ctx.getOrCreateConstant(
      cast<llvm::ConstantPtrAuth>(Val)->getAddrDiscriminator());
}

ConstantPtrAuth *ConstantPtrAuth::getWithSameSchema(Constant *Pointer) const {
  auto *LLVMC = cast<llvm::ConstantPtrAuth>(Val)->getWithSameSchema(
      cast<llvm::Constant>(Pointer->Val));
  return cast<ConstantPtrAuth>(Ctx.getOrCreateConstant(LLVMC));
}

BlockAddress *BlockAddress::get(Function *F, BasicBlock *BB) {
  auto *LLVMC = llvm::BlockAddress::get(cast<llvm::Function>(F->Val),
                                        cast<llvm::BasicBlock>(BB->Val));
  return cast<BlockAddress>(F->getContext().getOrCreateConstant(LLVMC));
}

BlockAddress *BlockAddress::get(BasicBlock *BB) {
  auto *LLVMC = llvm::BlockAddress::get(cast<llvm::BasicBlock>(BB->Val));
  return cast<BlockAddress>(BB->getContext().getOrCreateConstant(LLVMC));
}

BlockAddress *BlockAddress::lookup(const BasicBlock *BB) {
  auto *LLVMC = llvm::BlockAddress::lookup(cast<llvm::BasicBlock>(BB->Val));
  return cast_or_null<BlockAddress>(BB->getContext().getValue(LLVMC));
}

Function *BlockAddress::getFunction() const {
  return cast<Function>(
      Ctx.getValue(cast<llvm::BlockAddress>(Val)->getFunction()));
}

BasicBlock *BlockAddress::getBasicBlock() const {
  return cast<BasicBlock>(
      Ctx.getValue(cast<llvm::BlockAddress>(Val)->getBasicBlock()));
}

DSOLocalEquivalent *DSOLocalEquivalent::get(GlobalValue *GV) {
  auto *LLVMC = llvm::DSOLocalEquivalent::get(cast<llvm::GlobalValue>(GV->Val));
  return cast<DSOLocalEquivalent>(GV->getContext().getValue(LLVMC));
}

GlobalValue *DSOLocalEquivalent::getGlobalValue() const {
  return cast<GlobalValue>(
      Ctx.getValue(cast<llvm::DSOLocalEquivalent>(Val)->getGlobalValue()));
}

ConstantTokenNone *ConstantTokenNone::get(Context &Ctx) {
  auto *LLVMC = llvm::ConstantTokenNone::get(Ctx.LLVMCtx);
  return cast<ConstantTokenNone>(Ctx.getOrCreateConstant(LLVMC));
}

FunctionType *Function::getFunctionType() const {
  return cast<FunctionType>(
      Ctx.getType(cast<llvm::Function>(Val)->getFunctionType()));
}

#ifndef NDEBUG
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
void Function::dumpOS(raw_ostream &OS) const {
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
  if (auto *I = dyn_cast<Instruction>(VPtr.get()))
    getTracker().emplaceIfTracking<CreateAndInsertInst>(I);

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
    switch (C->getValueID()) {
    case llvm::Value::ConstantIntVal:
      It->second = std::unique_ptr<ConstantInt>(
          new ConstantInt(cast<llvm::ConstantInt>(C), *this));
      return It->second.get();
    case llvm::Value::ConstantFPVal:
      It->second = std::unique_ptr<ConstantFP>(
          new ConstantFP(cast<llvm::ConstantFP>(C), *this));
      return It->second.get();
    case llvm::Value::BlockAddressVal:
      It->second = std::unique_ptr<BlockAddress>(
          new BlockAddress(cast<llvm::BlockAddress>(C), *this));
      return It->second.get();
    case llvm::Value::ConstantTokenNoneVal:
      It->second = std::unique_ptr<ConstantTokenNone>(
          new ConstantTokenNone(cast<llvm::ConstantTokenNone>(C), *this));
      return It->second.get();
    case llvm::Value::ConstantAggregateZeroVal: {
      auto *CAZ = cast<llvm::ConstantAggregateZero>(C);
      It->second = std::unique_ptr<ConstantAggregateZero>(
          new ConstantAggregateZero(CAZ, *this));
      auto *Ret = It->second.get();
      // Must create sandboxir for elements.
      auto EC = CAZ->getElementCount();
      if (EC.isFixed()) {
        for (auto ElmIdx : seq<unsigned>(0, EC.getFixedValue()))
          getOrCreateValueInternal(CAZ->getElementValue(ElmIdx), CAZ);
      }
      return Ret;
    }
    case llvm::Value::ConstantPointerNullVal:
      It->second = std::unique_ptr<ConstantPointerNull>(
          new ConstantPointerNull(cast<llvm::ConstantPointerNull>(C), *this));
      return It->second.get();
    case llvm::Value::PoisonValueVal:
      It->second = std::unique_ptr<PoisonValue>(
          new PoisonValue(cast<llvm::PoisonValue>(C), *this));
      return It->second.get();
    case llvm::Value::UndefValueVal:
      It->second = std::unique_ptr<UndefValue>(
          new UndefValue(cast<llvm::UndefValue>(C), *this));
      return It->second.get();
    case llvm::Value::DSOLocalEquivalentVal: {
      auto *DSOLE = cast<llvm::DSOLocalEquivalent>(C);
      It->second = std::unique_ptr<DSOLocalEquivalent>(
          new DSOLocalEquivalent(DSOLE, *this));
      auto *Ret = It->second.get();
      getOrCreateValueInternal(DSOLE->getGlobalValue(), DSOLE);
      return Ret;
    }
    case llvm::Value::ConstantArrayVal:
      It->second = std::unique_ptr<ConstantArray>(
          new ConstantArray(cast<llvm::ConstantArray>(C), *this));
      break;
    case llvm::Value::ConstantStructVal:
      It->second = std::unique_ptr<ConstantStruct>(
          new ConstantStruct(cast<llvm::ConstantStruct>(C), *this));
      break;
    case llvm::Value::ConstantVectorVal:
      It->second = std::unique_ptr<ConstantVector>(
          new ConstantVector(cast<llvm::ConstantVector>(C), *this));
      break;
    case llvm::Value::FunctionVal:
      It->second = std::unique_ptr<Function>(
          new Function(cast<llvm::Function>(C), *this));
      break;
    case llvm::Value::GlobalIFuncVal:
      It->second = std::unique_ptr<GlobalIFunc>(
          new GlobalIFunc(cast<llvm::GlobalIFunc>(C), *this));
      break;
    case llvm::Value::GlobalVariableVal:
      It->second = std::unique_ptr<GlobalVariable>(
          new GlobalVariable(cast<llvm::GlobalVariable>(C), *this));
      break;
    case llvm::Value::GlobalAliasVal:
      It->second = std::unique_ptr<GlobalAlias>(
          new GlobalAlias(cast<llvm::GlobalAlias>(C), *this));
      break;
    case llvm::Value::NoCFIValueVal:
      It->second = std::unique_ptr<NoCFIValue>(
          new NoCFIValue(cast<llvm::NoCFIValue>(C), *this));
      break;
    case llvm::Value::ConstantPtrAuthVal:
      It->second = std::unique_ptr<ConstantPtrAuth>(
          new ConstantPtrAuth(cast<llvm::ConstantPtrAuth>(C), *this));
      break;
    case llvm::Value::ConstantExprVal:
      It->second = std::unique_ptr<ConstantExpr>(
          new ConstantExpr(cast<llvm::ConstantExpr>(C), *this));
      break;
    default:
      It->second = std::unique_ptr<Constant>(new Constant(C, *this));
      break;
    }
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
    assert(isa<llvm::BlockAddress>(U) &&
           "This won't create a SBBB, don't call this function directly!");
    if (auto *SBBB = getValue(BB))
      return SBBB;
    return nullptr;
  }
  assert(isa<llvm::Instruction>(LLVMV) && "Expected Instruction");

  switch (cast<llvm::Instruction>(LLVMV)->getOpcode()) {
  case llvm::Instruction::VAArg: {
    auto *LLVMVAArg = cast<llvm::VAArgInst>(LLVMV);
    It->second = std::unique_ptr<VAArgInst>(new VAArgInst(LLVMVAArg, *this));
    return It->second.get();
  }
  case llvm::Instruction::Freeze: {
    auto *LLVMFreeze = cast<llvm::FreezeInst>(LLVMV);
    It->second = std::unique_ptr<FreezeInst>(new FreezeInst(LLVMFreeze, *this));
    return It->second.get();
  }
  case llvm::Instruction::Fence: {
    auto *LLVMFence = cast<llvm::FenceInst>(LLVMV);
    It->second = std::unique_ptr<FenceInst>(new FenceInst(LLVMFence, *this));
    return It->second.get();
  }
  case llvm::Instruction::Select: {
    auto *LLVMSel = cast<llvm::SelectInst>(LLVMV);
    It->second = std::unique_ptr<SelectInst>(new SelectInst(LLVMSel, *this));
    return It->second.get();
  }
  case llvm::Instruction::ExtractElement: {
    auto *LLVMIns = cast<llvm::ExtractElementInst>(LLVMV);
    It->second = std::unique_ptr<ExtractElementInst>(
        new ExtractElementInst(LLVMIns, *this));
    return It->second.get();
  }
  case llvm::Instruction::InsertElement: {
    auto *LLVMIns = cast<llvm::InsertElementInst>(LLVMV);
    It->second = std::unique_ptr<InsertElementInst>(
        new InsertElementInst(LLVMIns, *this));
    return It->second.get();
  }
  case llvm::Instruction::ShuffleVector: {
    auto *LLVMIns = cast<llvm::ShuffleVectorInst>(LLVMV);
    It->second = std::unique_ptr<ShuffleVectorInst>(
        new ShuffleVectorInst(LLVMIns, *this));
    return It->second.get();
  }
  case llvm::Instruction::ExtractValue: {
    auto *LLVMIns = cast<llvm::ExtractValueInst>(LLVMV);
    It->second =
        std::unique_ptr<ExtractValueInst>(new ExtractValueInst(LLVMIns, *this));
    return It->second.get();
  }
  case llvm::Instruction::InsertValue: {
    auto *LLVMIns = cast<llvm::InsertValueInst>(LLVMV);
    It->second =
        std::unique_ptr<InsertValueInst>(new InsertValueInst(LLVMIns, *this));
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
  case llvm::Instruction::LandingPad: {
    auto *LLVMLPad = cast<llvm::LandingPadInst>(LLVMV);
    It->second =
        std::unique_ptr<LandingPadInst>(new LandingPadInst(LLVMLPad, *this));
    return It->second.get();
  }
  case llvm::Instruction::CatchPad: {
    auto *LLVMCPI = cast<llvm::CatchPadInst>(LLVMV);
    It->second =
        std::unique_ptr<CatchPadInst>(new CatchPadInst(LLVMCPI, *this));
    return It->second.get();
  }
  case llvm::Instruction::CleanupPad: {
    auto *LLVMCPI = cast<llvm::CleanupPadInst>(LLVMV);
    It->second =
        std::unique_ptr<CleanupPadInst>(new CleanupPadInst(LLVMCPI, *this));
    return It->second.get();
  }
  case llvm::Instruction::CatchRet: {
    auto *LLVMCRI = cast<llvm::CatchReturnInst>(LLVMV);
    It->second =
        std::unique_ptr<CatchReturnInst>(new CatchReturnInst(LLVMCRI, *this));
    return It->second.get();
  }
  case llvm::Instruction::CleanupRet: {
    auto *LLVMCRI = cast<llvm::CleanupReturnInst>(LLVMV);
    It->second = std::unique_ptr<CleanupReturnInst>(
        new CleanupReturnInst(LLVMCRI, *this));
    return It->second.get();
  }
  case llvm::Instruction::GetElementPtr: {
    auto *LLVMGEP = cast<llvm::GetElementPtrInst>(LLVMV);
    It->second = std::unique_ptr<GetElementPtrInst>(
        new GetElementPtrInst(LLVMGEP, *this));
    return It->second.get();
  }
  case llvm::Instruction::CatchSwitch: {
    auto *LLVMCatchSwitchInst = cast<llvm::CatchSwitchInst>(LLVMV);
    It->second = std::unique_ptr<CatchSwitchInst>(
        new CatchSwitchInst(LLVMCatchSwitchInst, *this));
    return It->second.get();
  }
  case llvm::Instruction::Resume: {
    auto *LLVMResumeInst = cast<llvm::ResumeInst>(LLVMV);
    It->second =
        std::unique_ptr<ResumeInst>(new ResumeInst(LLVMResumeInst, *this));
    return It->second.get();
  }
  case llvm::Instruction::Switch: {
    auto *LLVMSwitchInst = cast<llvm::SwitchInst>(LLVMV);
    It->second =
        std::unique_ptr<SwitchInst>(new SwitchInst(LLVMSwitchInst, *this));
    return It->second.get();
  }
  case llvm::Instruction::FNeg: {
    auto *LLVMUnaryOperator = cast<llvm::UnaryOperator>(LLVMV);
    It->second = std::unique_ptr<UnaryOperator>(
        new UnaryOperator(LLVMUnaryOperator, *this));
    return It->second.get();
  }
  case llvm::Instruction::Add:
  case llvm::Instruction::FAdd:
  case llvm::Instruction::Sub:
  case llvm::Instruction::FSub:
  case llvm::Instruction::Mul:
  case llvm::Instruction::FMul:
  case llvm::Instruction::UDiv:
  case llvm::Instruction::SDiv:
  case llvm::Instruction::FDiv:
  case llvm::Instruction::URem:
  case llvm::Instruction::SRem:
  case llvm::Instruction::FRem:
  case llvm::Instruction::Shl:
  case llvm::Instruction::LShr:
  case llvm::Instruction::AShr:
  case llvm::Instruction::And:
  case llvm::Instruction::Or:
  case llvm::Instruction::Xor: {
    auto *LLVMBinaryOperator = cast<llvm::BinaryOperator>(LLVMV);
    It->second = std::unique_ptr<BinaryOperator>(
        new BinaryOperator(LLVMBinaryOperator, *this));
    return It->second.get();
  }
  case llvm::Instruction::AtomicRMW: {
    auto *LLVMAtomicRMW = cast<llvm::AtomicRMWInst>(LLVMV);
    It->second =
        std::unique_ptr<AtomicRMWInst>(new AtomicRMWInst(LLVMAtomicRMW, *this));
    return It->second.get();
  }
  case llvm::Instruction::AtomicCmpXchg: {
    auto *LLVMAtomicCmpXchg = cast<llvm::AtomicCmpXchgInst>(LLVMV);
    It->second = std::unique_ptr<AtomicCmpXchgInst>(
        new AtomicCmpXchgInst(LLVMAtomicCmpXchg, *this));
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
  case llvm::Instruction::ICmp: {
    auto *LLVMICmp = cast<llvm::ICmpInst>(LLVMV);
    It->second = std::unique_ptr<ICmpInst>(new ICmpInst(LLVMICmp, *this));
    return It->second.get();
  }
  case llvm::Instruction::FCmp: {
    auto *LLVMFCmp = cast<llvm::FCmpInst>(LLVMV);
    It->second = std::unique_ptr<FCmpInst>(new FCmpInst(LLVMFCmp, *this));
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

VAArgInst *Context::createVAArgInst(llvm::VAArgInst *SI) {
  auto NewPtr = std::unique_ptr<VAArgInst>(new VAArgInst(SI, *this));
  return cast<VAArgInst>(registerValue(std::move(NewPtr)));
}

FreezeInst *Context::createFreezeInst(llvm::FreezeInst *SI) {
  auto NewPtr = std::unique_ptr<FreezeInst>(new FreezeInst(SI, *this));
  return cast<FreezeInst>(registerValue(std::move(NewPtr)));
}

FenceInst *Context::createFenceInst(llvm::FenceInst *SI) {
  auto NewPtr = std::unique_ptr<FenceInst>(new FenceInst(SI, *this));
  return cast<FenceInst>(registerValue(std::move(NewPtr)));
}

SelectInst *Context::createSelectInst(llvm::SelectInst *SI) {
  auto NewPtr = std::unique_ptr<SelectInst>(new SelectInst(SI, *this));
  return cast<SelectInst>(registerValue(std::move(NewPtr)));
}

ExtractElementInst *
Context::createExtractElementInst(llvm::ExtractElementInst *EEI) {
  auto NewPtr =
      std::unique_ptr<ExtractElementInst>(new ExtractElementInst(EEI, *this));
  return cast<ExtractElementInst>(registerValue(std::move(NewPtr)));
}

InsertElementInst *
Context::createInsertElementInst(llvm::InsertElementInst *IEI) {
  auto NewPtr =
      std::unique_ptr<InsertElementInst>(new InsertElementInst(IEI, *this));
  return cast<InsertElementInst>(registerValue(std::move(NewPtr)));
}

ShuffleVectorInst *
Context::createShuffleVectorInst(llvm::ShuffleVectorInst *SVI) {
  auto NewPtr =
      std::unique_ptr<ShuffleVectorInst>(new ShuffleVectorInst(SVI, *this));
  return cast<ShuffleVectorInst>(registerValue(std::move(NewPtr)));
}

ExtractValueInst *Context::createExtractValueInst(llvm::ExtractValueInst *EVI) {
  auto NewPtr =
      std::unique_ptr<ExtractValueInst>(new ExtractValueInst(EVI, *this));
  return cast<ExtractValueInst>(registerValue(std::move(NewPtr)));
}

InsertValueInst *Context::createInsertValueInst(llvm::InsertValueInst *IVI) {
  auto NewPtr =
      std::unique_ptr<InsertValueInst>(new InsertValueInst(IVI, *this));
  return cast<InsertValueInst>(registerValue(std::move(NewPtr)));
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
LandingPadInst *Context::createLandingPadInst(llvm::LandingPadInst *I) {
  auto NewPtr = std::unique_ptr<LandingPadInst>(new LandingPadInst(I, *this));
  return cast<LandingPadInst>(registerValue(std::move(NewPtr)));
}
CatchPadInst *Context::createCatchPadInst(llvm::CatchPadInst *I) {
  auto NewPtr = std::unique_ptr<CatchPadInst>(new CatchPadInst(I, *this));
  return cast<CatchPadInst>(registerValue(std::move(NewPtr)));
}
CleanupPadInst *Context::createCleanupPadInst(llvm::CleanupPadInst *I) {
  auto NewPtr = std::unique_ptr<CleanupPadInst>(new CleanupPadInst(I, *this));
  return cast<CleanupPadInst>(registerValue(std::move(NewPtr)));
}
CatchReturnInst *Context::createCatchReturnInst(llvm::CatchReturnInst *I) {
  auto NewPtr = std::unique_ptr<CatchReturnInst>(new CatchReturnInst(I, *this));
  return cast<CatchReturnInst>(registerValue(std::move(NewPtr)));
}
CleanupReturnInst *
Context::createCleanupReturnInst(llvm::CleanupReturnInst *I) {
  auto NewPtr =
      std::unique_ptr<CleanupReturnInst>(new CleanupReturnInst(I, *this));
  return cast<CleanupReturnInst>(registerValue(std::move(NewPtr)));
}
GetElementPtrInst *
Context::createGetElementPtrInst(llvm::GetElementPtrInst *I) {
  auto NewPtr =
      std::unique_ptr<GetElementPtrInst>(new GetElementPtrInst(I, *this));
  return cast<GetElementPtrInst>(registerValue(std::move(NewPtr)));
}
CatchSwitchInst *Context::createCatchSwitchInst(llvm::CatchSwitchInst *I) {
  auto NewPtr = std::unique_ptr<CatchSwitchInst>(new CatchSwitchInst(I, *this));
  return cast<CatchSwitchInst>(registerValue(std::move(NewPtr)));
}
ResumeInst *Context::createResumeInst(llvm::ResumeInst *I) {
  auto NewPtr = std::unique_ptr<ResumeInst>(new ResumeInst(I, *this));
  return cast<ResumeInst>(registerValue(std::move(NewPtr)));
}
SwitchInst *Context::createSwitchInst(llvm::SwitchInst *I) {
  auto NewPtr = std::unique_ptr<SwitchInst>(new SwitchInst(I, *this));
  return cast<SwitchInst>(registerValue(std::move(NewPtr)));
}
UnaryOperator *Context::createUnaryOperator(llvm::UnaryOperator *I) {
  auto NewPtr = std::unique_ptr<UnaryOperator>(new UnaryOperator(I, *this));
  return cast<UnaryOperator>(registerValue(std::move(NewPtr)));
}
BinaryOperator *Context::createBinaryOperator(llvm::BinaryOperator *I) {
  auto NewPtr = std::unique_ptr<BinaryOperator>(new BinaryOperator(I, *this));
  return cast<BinaryOperator>(registerValue(std::move(NewPtr)));
}
AtomicRMWInst *Context::createAtomicRMWInst(llvm::AtomicRMWInst *I) {
  auto NewPtr = std::unique_ptr<AtomicRMWInst>(new AtomicRMWInst(I, *this));
  return cast<AtomicRMWInst>(registerValue(std::move(NewPtr)));
}
AtomicCmpXchgInst *
Context::createAtomicCmpXchgInst(llvm::AtomicCmpXchgInst *I) {
  auto NewPtr =
      std::unique_ptr<AtomicCmpXchgInst>(new AtomicCmpXchgInst(I, *this));
  return cast<AtomicCmpXchgInst>(registerValue(std::move(NewPtr)));
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
ICmpInst *Context::createICmpInst(llvm::ICmpInst *I) {
  auto NewPtr = std::unique_ptr<ICmpInst>(new ICmpInst(I, *this));
  return cast<ICmpInst>(registerValue(std::move(NewPtr)));
}
FCmpInst *Context::createFCmpInst(llvm::FCmpInst *I) {
  auto NewPtr = std::unique_ptr<FCmpInst>(new FCmpInst(I, *this));
  return cast<FCmpInst>(registerValue(std::move(NewPtr)));
}
CmpInst *CmpInst::create(Predicate P, Value *S1, Value *S2,
                         Instruction *InsertBefore, Context &Ctx,
                         const Twine &Name) {
  auto &Builder = Ctx.getLLVMIRBuilder();
  Builder.SetInsertPoint(InsertBefore->getTopmostLLVMInstruction());
  auto *LLVMI = Builder.CreateCmp(P, S1->Val, S2->Val, Name);
  if (dyn_cast<llvm::ICmpInst>(LLVMI))
    return Ctx.createICmpInst(cast<llvm::ICmpInst>(LLVMI));
  return Ctx.createFCmpInst(cast<llvm::FCmpInst>(LLVMI));
}
CmpInst *CmpInst::createWithCopiedFlags(Predicate P, Value *S1, Value *S2,
                                        const Instruction *F,
                                        Instruction *InsertBefore, Context &Ctx,
                                        const Twine &Name) {
  CmpInst *Inst = create(P, S1, S2, InsertBefore, Ctx, Name);
  cast<llvm::CmpInst>(Inst->Val)->copyIRFlags(F->Val);
  return Inst;
}

Type *CmpInst::makeCmpResultType(Type *OpndType) {
  if (auto *VT = dyn_cast<VectorType>(OpndType)) {
    // TODO: Cleanup when we have more complete support for
    // sandboxir::VectorType
    return OpndType->getContext().getType(llvm::VectorType::get(
        llvm::Type::getInt1Ty(OpndType->getContext().LLVMCtx),
        cast<llvm::VectorType>(VT->LLVMTy)->getElementCount()));
  }
  return Type::getInt1Ty(OpndType->getContext());
}

void CmpInst::setPredicate(Predicate P) {
  Ctx.getTracker()
      .emplaceIfTracking<
          GenericSetter<&CmpInst::getPredicate, &CmpInst::setPredicate>>(this);
  cast<llvm::CmpInst>(Val)->setPredicate(P);
}

void CmpInst::swapOperands() {
  if (ICmpInst *IC = dyn_cast<ICmpInst>(this))
    IC->swapOperands();
  else
    cast<FCmpInst>(this)->swapOperands();
}

void ICmpInst::swapOperands() {
  Ctx.getTracker().emplaceIfTracking<CmpSwapOperands>(this);
  cast<llvm::ICmpInst>(Val)->swapOperands();
}

void FCmpInst::swapOperands() {
  Ctx.getTracker().emplaceIfTracking<CmpSwapOperands>(this);
  cast<llvm::FCmpInst>(Val)->swapOperands();
}

#ifndef NDEBUG
void CmpInst::dumpOS(raw_ostream &OS) const {
  dumpCommonPrefix(OS);
  dumpCommonSuffix(OS);
}

void CmpInst::dump() const {
  dumpOS(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

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
#if !defined(NDEBUG)
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
void BasicBlock::dumpOS(raw_ostream &OS) const {
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
      SBI.dumpOS(OS);
      OS << "\n";
    }
  }
}

void BasicBlock::verify() const {
  assert(isa<llvm::BasicBlock>(Val) && "Expected BasicBlock!");
  for (const auto &I : *this) {
    I.verify();
  }
}

#endif // NDEBUG
