//=====-- DebugProgramInstruction.cpp - Implement DbgRecords/DPMarkers --=====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IntrinsicInst.h"

namespace llvm {

template <typename T>
DbgRecordParamRef<T>::DbgRecordParamRef(const T *Param)
    : Ref(const_cast<T *>(Param)) {}
template <typename T>
DbgRecordParamRef<T>::DbgRecordParamRef(const MDNode *Param)
    : Ref(const_cast<MDNode *>(Param)) {}

template <typename T> T *DbgRecordParamRef<T>::get() const {
  return cast<T>(Ref);
}

template class DbgRecordParamRef<DIExpression>;
template class DbgRecordParamRef<DILabel>;
template class DbgRecordParamRef<DILocalVariable>;

DPValue::DPValue(const DbgVariableIntrinsic *DVI)
    : DbgRecord(ValueKind, DVI->getDebugLoc()),
      DebugValueUser({DVI->getRawLocation(), nullptr, nullptr}),
      Variable(DVI->getVariable()), Expression(DVI->getExpression()),
      AddressExpression() {
  switch (DVI->getIntrinsicID()) {
  case Intrinsic::dbg_value:
    Type = LocationType::Value;
    break;
  case Intrinsic::dbg_declare:
    Type = LocationType::Declare;
    break;
  case Intrinsic::dbg_assign: {
    Type = LocationType::Assign;
    const DbgAssignIntrinsic *Assign =
        static_cast<const DbgAssignIntrinsic *>(DVI);
    resetDebugValue(1, Assign->getRawAddress());
    AddressExpression = Assign->getAddressExpression();
    setAssignId(Assign->getAssignID());
    break;
  }
  default:
    llvm_unreachable(
        "Trying to create a DPValue with an invalid intrinsic type!");
  }
}

DPValue::DPValue(const DPValue &DPV)
    : DbgRecord(ValueKind, DPV.getDebugLoc()), DebugValueUser(DPV.DebugValues),
      Type(DPV.getType()), Variable(DPV.getVariable()),
      Expression(DPV.getExpression()),
      AddressExpression(DPV.AddressExpression) {}

DPValue::DPValue(Metadata *Location, DILocalVariable *DV, DIExpression *Expr,
                 const DILocation *DI, LocationType Type)
    : DbgRecord(ValueKind, DI), DebugValueUser({Location, nullptr, nullptr}),
      Type(Type), Variable(DV), Expression(Expr) {}

DPValue::DPValue(Metadata *Value, DILocalVariable *Variable,
                 DIExpression *Expression, DIAssignID *AssignID,
                 Metadata *Address, DIExpression *AddressExpression,
                 const DILocation *DI)
    : DbgRecord(ValueKind, DI), DebugValueUser({Value, Address, AssignID}),
      Type(LocationType::Assign), Variable(Variable), Expression(Expression),
      AddressExpression(AddressExpression) {}

void DbgRecord::deleteRecord() {
  switch (RecordKind) {
  case ValueKind:
    delete cast<DPValue>(this);
    return;
  case LabelKind:
    delete cast<DPLabel>(this);
    return;
  }
  llvm_unreachable("unsupported DbgRecord kind");
}

void DbgRecord::print(raw_ostream &O, bool IsForDebug) const {
  switch (RecordKind) {
  case ValueKind:
    cast<DPValue>(this)->print(O, IsForDebug);
    return;
  case LabelKind:
    cast<DPLabel>(this)->print(O, IsForDebug);
    return;
  };
  llvm_unreachable("unsupported DbgRecord kind");
}

void DbgRecord::print(raw_ostream &O, ModuleSlotTracker &MST,
                      bool IsForDebug) const {
  switch (RecordKind) {
  case ValueKind:
    cast<DPValue>(this)->print(O, MST, IsForDebug);
    return;
  case LabelKind:
    cast<DPLabel>(this)->print(O, MST, IsForDebug);
    return;
  };
  llvm_unreachable("unsupported DbgRecord kind");
}

bool DbgRecord::isIdenticalToWhenDefined(const DbgRecord &R) const {
  if (RecordKind != R.RecordKind)
    return false;
  switch (RecordKind) {
  case ValueKind:
    return cast<DPValue>(this)->isIdenticalToWhenDefined(*cast<DPValue>(&R));
  case LabelKind:
    return cast<DPLabel>(this)->getLabel() == cast<DPLabel>(R).getLabel();
  };
  llvm_unreachable("unsupported DbgRecord kind");
}

bool DbgRecord::isEquivalentTo(const DbgRecord &R) const {
  return getDebugLoc() == R.getDebugLoc() && isIdenticalToWhenDefined(R);
}

DbgInfoIntrinsic *
DbgRecord::createDebugIntrinsic(Module *M, Instruction *InsertBefore) const {
  switch (RecordKind) {
  case ValueKind:
    return cast<DPValue>(this)->createDebugIntrinsic(M, InsertBefore);
  case LabelKind:
    return cast<DPLabel>(this)->createDebugIntrinsic(M, InsertBefore);
  };
  llvm_unreachable("unsupported DbgRecord kind");
}

DPLabel::DPLabel(MDNode *Label, MDNode *DL)
    : DbgRecord(LabelKind, DebugLoc(DL)), Label(Label) {
  assert(Label && "Unexpected nullptr");
  assert((isa<DILabel>(Label) || Label->isTemporary()) &&
         "Label type must be or resolve to a DILabel");
}
DPLabel::DPLabel(DILabel *Label, DebugLoc DL)
    : DbgRecord(LabelKind, DL), Label(Label) {
  assert(Label && "Unexpected nullptr");
}

DPLabel *DPLabel::createUnresolvedDPLabel(MDNode *Label, MDNode *DL) {
  return new DPLabel(Label, DL);
}

DPValue::DPValue(DPValue::LocationType Type, Metadata *Val, MDNode *Variable,
                 MDNode *Expression, MDNode *AssignID, Metadata *Address,
                 MDNode *AddressExpression, MDNode *DI)
    : DbgRecord(ValueKind, DebugLoc(DI)),
      DebugValueUser({Val, Address, AssignID}), Type(Type), Variable(Variable),
      Expression(Expression), AddressExpression(AddressExpression) {}

DPValue *DPValue::createUnresolvedDPValue(DPValue::LocationType Type,
                                          Metadata *Val, MDNode *Variable,
                                          MDNode *Expression, MDNode *AssignID,
                                          Metadata *Address,
                                          MDNode *AddressExpression,
                                          MDNode *DI) {
  return new DPValue(Type, Val, Variable, Expression, AssignID, Address,
                     AddressExpression, DI);
}

DPValue *DPValue::createDPValue(Value *Location, DILocalVariable *DV,
                                DIExpression *Expr, const DILocation *DI) {
  return new DPValue(ValueAsMetadata::get(Location), DV, Expr, DI,
                     LocationType::Value);
}

DPValue *DPValue::createDPValue(Value *Location, DILocalVariable *DV,
                                DIExpression *Expr, const DILocation *DI,
                                DPValue &InsertBefore) {
  auto *NewDPValue = createDPValue(Location, DV, Expr, DI);
  NewDPValue->insertBefore(&InsertBefore);
  return NewDPValue;
}

DPValue *DPValue::createDPVDeclare(Value *Address, DILocalVariable *DV,
                                   DIExpression *Expr, const DILocation *DI) {
  return new DPValue(ValueAsMetadata::get(Address), DV, Expr, DI,
                     LocationType::Declare);
}

DPValue *DPValue::createDPVDeclare(Value *Address, DILocalVariable *DV,
                                   DIExpression *Expr, const DILocation *DI,
                                   DPValue &InsertBefore) {
  auto *NewDPVDeclare = createDPVDeclare(Address, DV, Expr, DI);
  NewDPVDeclare->insertBefore(&InsertBefore);
  return NewDPVDeclare;
}

DPValue *DPValue::createDPVAssign(Value *Val, DILocalVariable *Variable,
                                  DIExpression *Expression,
                                  DIAssignID *AssignID, Value *Address,
                                  DIExpression *AddressExpression,
                                  const DILocation *DI) {
  return new DPValue(ValueAsMetadata::get(Val), Variable, Expression, AssignID,
                     ValueAsMetadata::get(Address), AddressExpression, DI);
}

DPValue *DPValue::createLinkedDPVAssign(Instruction *LinkedInstr, Value *Val,
                                        DILocalVariable *Variable,
                                        DIExpression *Expression,
                                        Value *Address,
                                        DIExpression *AddressExpression,
                                        const DILocation *DI) {
  auto *Link = LinkedInstr->getMetadata(LLVMContext::MD_DIAssignID);
  assert(Link && "Linked instruction must have DIAssign metadata attached");
  auto *NewDPVAssign = DPValue::createDPVAssign(Val, Variable, Expression,
                                                cast<DIAssignID>(Link), Address,
                                                AddressExpression, DI);
  LinkedInstr->getParent()->insertDbgRecordAfter(NewDPVAssign, LinkedInstr);
  return NewDPVAssign;
}

iterator_range<DPValue::location_op_iterator> DPValue::location_ops() const {
  auto *MD = getRawLocation();
  // If a Value has been deleted, the "location" for this DPValue will be
  // replaced by nullptr. Return an empty range.
  if (!MD)
    return {location_op_iterator(static_cast<ValueAsMetadata *>(nullptr)),
            location_op_iterator(static_cast<ValueAsMetadata *>(nullptr))};

  // If operand is ValueAsMetadata, return a range over just that operand.
  if (auto *VAM = dyn_cast<ValueAsMetadata>(MD))
    return {location_op_iterator(VAM), location_op_iterator(VAM + 1)};

  // If operand is DIArgList, return a range over its args.
  if (auto *AL = dyn_cast<DIArgList>(MD))
    return {location_op_iterator(AL->args_begin()),
            location_op_iterator(AL->args_end())};

  // Operand is an empty metadata tuple, so return empty iterator.
  assert(cast<MDNode>(MD)->getNumOperands() == 0);
  return {location_op_iterator(static_cast<ValueAsMetadata *>(nullptr)),
          location_op_iterator(static_cast<ValueAsMetadata *>(nullptr))};
}

unsigned DPValue::getNumVariableLocationOps() const {
  if (hasArgList())
    return cast<DIArgList>(getRawLocation())->getArgs().size();
  return 1;
}

Value *DPValue::getVariableLocationOp(unsigned OpIdx) const {
  auto *MD = getRawLocation();
  if (!MD)
    return nullptr;

  if (auto *AL = dyn_cast<DIArgList>(MD))
    return AL->getArgs()[OpIdx]->getValue();
  if (isa<MDNode>(MD))
    return nullptr;
  assert(isa<ValueAsMetadata>(MD) &&
         "Attempted to get location operand from DPValue with none.");
  auto *V = cast<ValueAsMetadata>(MD);
  assert(OpIdx == 0 && "Operand Index must be 0 for a debug intrinsic with a "
                       "single location operand.");
  return V->getValue();
}

static ValueAsMetadata *getAsMetadata(Value *V) {
  return isa<MetadataAsValue>(V) ? dyn_cast<ValueAsMetadata>(
                                       cast<MetadataAsValue>(V)->getMetadata())
                                 : ValueAsMetadata::get(V);
}

void DPValue::replaceVariableLocationOp(Value *OldValue, Value *NewValue,
                                        bool AllowEmpty) {
  assert(NewValue && "Values must be non-null");

  bool DbgAssignAddrReplaced = isDbgAssign() && OldValue == getAddress();
  if (DbgAssignAddrReplaced)
    setAddress(NewValue);

  auto Locations = location_ops();
  auto OldIt = find(Locations, OldValue);
  if (OldIt == Locations.end()) {
    if (AllowEmpty || DbgAssignAddrReplaced)
      return;
    llvm_unreachable("OldValue must be a current location");
  }

  if (!hasArgList()) {
    // Set our location to be the MAV wrapping the new Value.
    setRawLocation(isa<MetadataAsValue>(NewValue)
                       ? cast<MetadataAsValue>(NewValue)->getMetadata()
                       : ValueAsMetadata::get(NewValue));
    return;
  }

  // We must be referring to a DIArgList, produce a new operands vector with the
  // old value replaced, generate a new DIArgList and set it as our location.
  SmallVector<ValueAsMetadata *, 4> MDs;
  ValueAsMetadata *NewOperand = getAsMetadata(NewValue);
  for (auto *VMD : Locations)
    MDs.push_back(VMD == *OldIt ? NewOperand : getAsMetadata(VMD));
  setRawLocation(DIArgList::get(getVariableLocationOp(0)->getContext(), MDs));
}

void DPValue::replaceVariableLocationOp(unsigned OpIdx, Value *NewValue) {
  assert(OpIdx < getNumVariableLocationOps() && "Invalid Operand Index");

  if (!hasArgList()) {
    setRawLocation(isa<MetadataAsValue>(NewValue)
                       ? cast<MetadataAsValue>(NewValue)->getMetadata()
                       : ValueAsMetadata::get(NewValue));
    return;
  }

  SmallVector<ValueAsMetadata *, 4> MDs;
  ValueAsMetadata *NewOperand = getAsMetadata(NewValue);
  for (unsigned Idx = 0; Idx < getNumVariableLocationOps(); ++Idx)
    MDs.push_back(Idx == OpIdx ? NewOperand
                               : getAsMetadata(getVariableLocationOp(Idx)));

  setRawLocation(DIArgList::get(getVariableLocationOp(0)->getContext(), MDs));
}

void DPValue::addVariableLocationOps(ArrayRef<Value *> NewValues,
                                     DIExpression *NewExpr) {
  assert(NewExpr->hasAllLocationOps(getNumVariableLocationOps() +
                                    NewValues.size()) &&
         "NewExpr for debug variable intrinsic does not reference every "
         "location operand.");
  assert(!is_contained(NewValues, nullptr) && "New values must be non-null");
  setExpression(NewExpr);
  SmallVector<ValueAsMetadata *, 4> MDs;
  for (auto *VMD : location_ops())
    MDs.push_back(getAsMetadata(VMD));
  for (auto *VMD : NewValues)
    MDs.push_back(getAsMetadata(VMD));
  setRawLocation(DIArgList::get(getVariableLocationOp(0)->getContext(), MDs));
}

void DPValue::setKillLocation() {
  // TODO: When/if we remove duplicate values from DIArgLists, we don't need
  // this set anymore.
  SmallPtrSet<Value *, 4> RemovedValues;
  for (Value *OldValue : location_ops()) {
    if (!RemovedValues.insert(OldValue).second)
      continue;
    Value *Poison = PoisonValue::get(OldValue->getType());
    replaceVariableLocationOp(OldValue, Poison);
  }
}

bool DPValue::isKillLocation() const {
  return (getNumVariableLocationOps() == 0 &&
          !getExpression()->isComplex()) ||
         any_of(location_ops(), [](Value *V) { return isa<UndefValue>(V); });
}

std::optional<uint64_t> DPValue::getFragmentSizeInBits() const {
  if (auto Fragment = getExpression()->getFragmentInfo())
    return Fragment->SizeInBits;
  return getVariable()->getSizeInBits();
}

DbgRecord *DbgRecord::clone() const {
  switch (RecordKind) {
  case ValueKind:
    return cast<DPValue>(this)->clone();
  case LabelKind:
    return cast<DPLabel>(this)->clone();
  };
  llvm_unreachable("unsupported DbgRecord kind");
}

DPValue *DPValue::clone() const { return new DPValue(*this); }

DPLabel *DPLabel::clone() const {
  return new DPLabel(getLabel(), getDebugLoc());
}

DbgVariableIntrinsic *
DPValue::createDebugIntrinsic(Module *M, Instruction *InsertBefore) const {
  [[maybe_unused]] DICompileUnit *Unit =
      getDebugLoc().get()->getScope()->getSubprogram()->getUnit();
  assert(M && Unit &&
         "Cannot clone from BasicBlock that is not part of a Module or "
         "DICompileUnit!");
  LLVMContext &Context = getDebugLoc()->getContext();
  Function *IntrinsicFn;

  // Work out what sort of intrinsic we're going to produce.
  switch (getType()) {
  case DPValue::LocationType::Declare:
    IntrinsicFn = Intrinsic::getDeclaration(M, Intrinsic::dbg_declare);
    break;
  case DPValue::LocationType::Value:
    IntrinsicFn = Intrinsic::getDeclaration(M, Intrinsic::dbg_value);
    break;
  case DPValue::LocationType::Assign:
    IntrinsicFn = Intrinsic::getDeclaration(M, Intrinsic::dbg_assign);
    break;
  case DPValue::LocationType::End:
  case DPValue::LocationType::Any:
    llvm_unreachable("Invalid LocationType");
  }

  // Create the intrinsic from this DPValue's information, optionally insert
  // into the target location.
  DbgVariableIntrinsic *DVI;
  assert(getRawLocation() && "DPValue's RawLocation should be non-null.");
  if (isDbgAssign()) {
    Value *AssignArgs[] = {
        MetadataAsValue::get(Context, getRawLocation()),
        MetadataAsValue::get(Context, getVariable()),
        MetadataAsValue::get(Context, getExpression()),
        MetadataAsValue::get(Context, getAssignID()),
        MetadataAsValue::get(Context, getRawAddress()),
        MetadataAsValue::get(Context, getAddressExpression())};
    DVI = cast<DbgVariableIntrinsic>(CallInst::Create(
        IntrinsicFn->getFunctionType(), IntrinsicFn, AssignArgs));
  } else {
    Value *Args[] = {MetadataAsValue::get(Context, getRawLocation()),
                     MetadataAsValue::get(Context, getVariable()),
                     MetadataAsValue::get(Context, getExpression())};
    DVI = cast<DbgVariableIntrinsic>(
        CallInst::Create(IntrinsicFn->getFunctionType(), IntrinsicFn, Args));
  }
  DVI->setTailCall();
  DVI->setDebugLoc(getDebugLoc());
  if (InsertBefore)
    DVI->insertBefore(InsertBefore);

  return DVI;
}

DbgLabelInst *DPLabel::createDebugIntrinsic(Module *M,
                                            Instruction *InsertBefore) const {
  auto *LabelFn = Intrinsic::getDeclaration(M, Intrinsic::dbg_label);
  Value *Args[] = {
      MetadataAsValue::get(getDebugLoc()->getContext(), getLabel())};
  DbgLabelInst *DbgLabel = cast<DbgLabelInst>(
      CallInst::Create(LabelFn->getFunctionType(), LabelFn, Args));
  DbgLabel->setTailCall();
  DbgLabel->setDebugLoc(getDebugLoc());
  if (InsertBefore)
    DbgLabel->insertBefore(InsertBefore);
  return DbgLabel;
}

Value *DPValue::getAddress() const {
  auto *MD = getRawAddress();
  if (auto *V = dyn_cast<ValueAsMetadata>(MD))
    return V->getValue();

  // When the value goes to null, it gets replaced by an empty MDNode.
  assert(!cast<MDNode>(MD)->getNumOperands() && "Expected an empty MDNode");
  return nullptr;
}

DIAssignID *DPValue::getAssignID() const {
  return cast<DIAssignID>(DebugValues[2]);
}

void DPValue::setAssignId(DIAssignID *New) { resetDebugValue(2, New); }

void DPValue::setKillAddress() {
  resetDebugValue(
      1, ValueAsMetadata::get(UndefValue::get(getAddress()->getType())));
}

bool DPValue::isKillAddress() const {
  Value *Addr = getAddress();
  return !Addr || isa<UndefValue>(Addr);
}

const Instruction *DbgRecord::getInstruction() const {
  return Marker->MarkedInstr;
}

const BasicBlock *DbgRecord::getParent() const {
  return Marker->MarkedInstr->getParent();
}

BasicBlock *DbgRecord::getParent() { return Marker->MarkedInstr->getParent(); }

BasicBlock *DbgRecord::getBlock() { return Marker->getParent(); }

const BasicBlock *DbgRecord::getBlock() const { return Marker->getParent(); }

Function *DbgRecord::getFunction() { return getBlock()->getParent(); }

const Function *DbgRecord::getFunction() const {
  return getBlock()->getParent();
}

Module *DbgRecord::getModule() { return getFunction()->getParent(); }

const Module *DbgRecord::getModule() const {
  return getFunction()->getParent();
}

LLVMContext &DbgRecord::getContext() { return getBlock()->getContext(); }

const LLVMContext &DbgRecord::getContext() const {
  return getBlock()->getContext();
}

void DbgRecord::insertBefore(DbgRecord *InsertBefore) {
  assert(!getMarker() &&
         "Cannot insert a DbgRecord that is already has a DPMarker!");
  assert(InsertBefore->getMarker() &&
         "Cannot insert a DbgRecord before a DbgRecord that does not have a "
         "DPMarker!");
  InsertBefore->getMarker()->insertDbgRecord(this, InsertBefore);
}
void DbgRecord::insertAfter(DbgRecord *InsertAfter) {
  assert(!getMarker() &&
         "Cannot insert a DbgRecord that is already has a DPMarker!");
  assert(InsertAfter->getMarker() &&
         "Cannot insert a DbgRecord after a DbgRecord that does not have a "
         "DPMarker!");
  InsertAfter->getMarker()->insertDbgRecordAfter(this, InsertAfter);
}
void DbgRecord::moveBefore(DbgRecord *MoveBefore) {
  assert(getMarker() &&
         "Canot move a DbgRecord that does not currently have a DPMarker!");
  removeFromParent();
  insertBefore(MoveBefore);
}
void DbgRecord::moveAfter(DbgRecord *MoveAfter) {
  assert(getMarker() &&
         "Canot move a DbgRecord that does not currently have a DPMarker!");
  removeFromParent();
  insertAfter(MoveAfter);
}

///////////////////////////////////////////////////////////////////////////////

// An empty, global, DPMarker for the purpose of describing empty ranges of
// DbgRecords.
DPMarker DPMarker::EmptyDPMarker;

void DPMarker::dropDbgRecords() {
  while (!StoredDbgRecords.empty()) {
    auto It = StoredDbgRecords.begin();
    DbgRecord *DR = &*It;
    StoredDbgRecords.erase(It);
    DR->deleteRecord();
  }
}

void DPMarker::dropOneDbgRecord(DbgRecord *DR) {
  assert(DR->getMarker() == this);
  StoredDbgRecords.erase(DR->getIterator());
  DR->deleteRecord();
}

const BasicBlock *DPMarker::getParent() const {
  return MarkedInstr->getParent();
}

BasicBlock *DPMarker::getParent() { return MarkedInstr->getParent(); }

void DPMarker::removeMarker() {
  // Are there any DbgRecords in this DPMarker? If not, nothing to preserve.
  Instruction *Owner = MarkedInstr;
  if (StoredDbgRecords.empty()) {
    eraseFromParent();
    Owner->DbgMarker = nullptr;
    return;
  }

  // The attached DbgRecords need to be preserved; attach them to the next
  // instruction. If there isn't a next instruction, put them on the
  // "trailing" list.
  DPMarker *NextMarker = Owner->getParent()->getNextMarker(Owner);
  if (NextMarker) {
    NextMarker->absorbDebugValues(*this, true);
    eraseFromParent();
  } else {
    // We can avoid a deallocation -- just store this marker onto the next
    // instruction. Unless we're at the end of the block, in which case this
    // marker becomes the trailing marker of a degenerate block.
    BasicBlock::iterator NextIt = std::next(Owner->getIterator());
    if (NextIt == getParent()->end()) {
      getParent()->setTrailingDbgRecords(this);
      MarkedInstr = nullptr;
    } else {
      NextIt->DbgMarker = this;
      MarkedInstr = &*NextIt;
    }
  }
  Owner->DbgMarker = nullptr;
}

void DPMarker::removeFromParent() {
  MarkedInstr->DbgMarker = nullptr;
  MarkedInstr = nullptr;
}

void DPMarker::eraseFromParent() {
  if (MarkedInstr)
    removeFromParent();
  dropDbgRecords();
  delete this;
}

iterator_range<DbgRecord::self_iterator> DPMarker::getDbgRecordRange() {
  return make_range(StoredDbgRecords.begin(), StoredDbgRecords.end());
}
iterator_range<DbgRecord::const_self_iterator>
DPMarker::getDbgRecordRange() const {
  return make_range(StoredDbgRecords.begin(), StoredDbgRecords.end());
}

void DbgRecord::removeFromParent() {
  getMarker()->StoredDbgRecords.erase(getIterator());
  Marker = nullptr;
}

void DbgRecord::eraseFromParent() {
  removeFromParent();
  deleteRecord();
}

void DPMarker::insertDbgRecord(DbgRecord *New, bool InsertAtHead) {
  auto It = InsertAtHead ? StoredDbgRecords.begin() : StoredDbgRecords.end();
  StoredDbgRecords.insert(It, *New);
  New->setMarker(this);
}
void DPMarker::insertDbgRecord(DbgRecord *New, DbgRecord *InsertBefore) {
  assert(InsertBefore->getMarker() == this &&
         "DbgRecord 'InsertBefore' must be contained in this DPMarker!");
  StoredDbgRecords.insert(InsertBefore->getIterator(), *New);
  New->setMarker(this);
}
void DPMarker::insertDbgRecordAfter(DbgRecord *New, DbgRecord *InsertAfter) {
  assert(InsertAfter->getMarker() == this &&
         "DbgRecord 'InsertAfter' must be contained in this DPMarker!");
  StoredDbgRecords.insert(++(InsertAfter->getIterator()), *New);
  New->setMarker(this);
}

void DPMarker::absorbDebugValues(DPMarker &Src, bool InsertAtHead) {
  auto It = InsertAtHead ? StoredDbgRecords.begin() : StoredDbgRecords.end();
  for (DbgRecord &DPV : Src.StoredDbgRecords)
    DPV.setMarker(this);

  StoredDbgRecords.splice(It, Src.StoredDbgRecords);
}

void DPMarker::absorbDebugValues(iterator_range<DbgRecord::self_iterator> Range,
                                 DPMarker &Src, bool InsertAtHead) {
  for (DbgRecord &DR : Range)
    DR.setMarker(this);

  auto InsertPos =
      (InsertAtHead) ? StoredDbgRecords.begin() : StoredDbgRecords.end();

  StoredDbgRecords.splice(InsertPos, Src.StoredDbgRecords, Range.begin(),
                          Range.end());
}

iterator_range<simple_ilist<DbgRecord>::iterator> DPMarker::cloneDebugInfoFrom(
    DPMarker *From, std::optional<simple_ilist<DbgRecord>::iterator> from_here,
    bool InsertAtHead) {
  DbgRecord *First = nullptr;
  // Work out what range of DbgRecords to clone: normally all the contents of
  // the "From" marker, optionally we can start from the from_here position down
  // to end().
  auto Range =
      make_range(From->StoredDbgRecords.begin(), From->StoredDbgRecords.end());
  if (from_here.has_value())
    Range = make_range(*from_here, From->StoredDbgRecords.end());

  // Clone each DPValue and insert into StoreDPValues; optionally place them at
  // the start or the end of the list.
  auto Pos = (InsertAtHead) ? StoredDbgRecords.begin() : StoredDbgRecords.end();
  for (DbgRecord &DR : Range) {
    DbgRecord *New = DR.clone();
    New->setMarker(this);
    StoredDbgRecords.insert(Pos, *New);
    if (!First)
      First = New;
  }

  if (!First)
    return {StoredDbgRecords.end(), StoredDbgRecords.end()};

  if (InsertAtHead)
    // If InsertAtHead is set, we cloned a range onto the front of of the
    // StoredDbgRecords collection, return that range.
    return {StoredDbgRecords.begin(), Pos};
  else
    // We inserted a block at the end, return that range.
    return {First->getIterator(), StoredDbgRecords.end()};
}

} // end namespace llvm
