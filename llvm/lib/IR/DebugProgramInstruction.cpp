//======-- DebugProgramInstruction.cpp - Implement DPValues/DPMarkers --======//
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

DPValue::DPValue(const DbgVariableIntrinsic *DVI)
    : DebugValueUser({DVI->getRawLocation(), nullptr, nullptr}),
      Variable(DVI->getVariable()), Expression(DVI->getExpression()),
      DbgLoc(DVI->getDebugLoc()), AddressExpression(nullptr) {
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
    : DebugValueUser(DPV.DebugValues), Variable(DPV.getVariable()),
      Expression(DPV.getExpression()), DbgLoc(DPV.getDebugLoc()),
      AddressExpression(DPV.AddressExpression), Type(DPV.getType()) {}

DPValue::DPValue(Metadata *Location, DILocalVariable *DV, DIExpression *Expr,
                 const DILocation *DI, LocationType Type)
    : DebugValueUser({Location, nullptr, nullptr}), Variable(DV),
      Expression(Expr), DbgLoc(DI), Type(Type) {}

DPValue::DPValue(Metadata *Value, DILocalVariable *Variable,
                 DIExpression *Expression, DIAssignID *AssignID,
                 Metadata *Address, DIExpression *AddressExpression,
                 const DILocation *DI)
    : DebugValueUser({Value, Address, AssignID}), Variable(Variable),
      Expression(Expression), DbgLoc(DI), AddressExpression(AddressExpression),
      Type(LocationType::Assign) {}

void DPValue::deleteInstr() { delete this; }

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
  LinkedInstr->getParent()->insertDPValueAfter(NewDPVAssign, LinkedInstr);
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

DPValue *DPValue::clone() const { return new DPValue(*this); }

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

const Instruction *DPValue::getInstruction() const {
  return Marker->MarkedInstr;
}

const BasicBlock *DPValue::getParent() const {
  return Marker->MarkedInstr->getParent();
}

BasicBlock *DPValue::getParent() { return Marker->MarkedInstr->getParent(); }

BasicBlock *DPValue::getBlock() { return Marker->getParent(); }

const BasicBlock *DPValue::getBlock() const { return Marker->getParent(); }

Function *DPValue::getFunction() { return getBlock()->getParent(); }

const Function *DPValue::getFunction() const { return getBlock()->getParent(); }

Module *DPValue::getModule() { return getFunction()->getParent(); }

const Module *DPValue::getModule() const { return getFunction()->getParent(); }

LLVMContext &DPValue::getContext() { return getBlock()->getContext(); }

const LLVMContext &DPValue::getContext() const {
  return getBlock()->getContext();
}

void DPValue::insertBefore(DPValue *InsertBefore) {
  assert(!getMarker() &&
         "Cannot insert a DPValue that is already has a DPMarker!");
  assert(InsertBefore->getMarker() &&
         "Cannot insert a DPValue before a DPValue that does not have a "
         "DPMarker!");
  InsertBefore->getMarker()->insertDPValue(this, InsertBefore);
}
void DPValue::insertAfter(DPValue *InsertAfter) {
  assert(!getMarker() &&
         "Cannot insert a DPValue that is already has a DPMarker!");
  assert(InsertAfter->getMarker() &&
         "Cannot insert a DPValue after a DPValue that does not have a "
         "DPMarker!");
  InsertAfter->getMarker()->insertDPValueAfter(this, InsertAfter);
}
void DPValue::moveBefore(DPValue *MoveBefore) {
  assert(getMarker() &&
         "Canot move a DPValue that does not currently have a DPMarker!");
  removeFromParent();
  insertBefore(MoveBefore);
}
void DPValue::moveAfter(DPValue *MoveAfter) {
  assert(getMarker() &&
         "Canot move a DPValue that does not currently have a DPMarker!");
  removeFromParent();
  insertAfter(MoveAfter);
}

///////////////////////////////////////////////////////////////////////////////

// An empty, global, DPMarker for the purpose of describing empty ranges of
// DPValues.
DPMarker DPMarker::EmptyDPMarker;

void DPMarker::dropDPValues() {
  while (!StoredDPValues.empty()) {
    auto It = StoredDPValues.begin();
    DPValue *DPV = &*It;
    StoredDPValues.erase(It);
    DPV->deleteInstr();
  }
}

void DPMarker::dropOneDPValue(DPValue *DPV) {
  assert(DPV->getMarker() == this);
  StoredDPValues.erase(DPV->getIterator());
  DPV->deleteInstr();
}

const BasicBlock *DPMarker::getParent() const {
  return MarkedInstr->getParent();
}

BasicBlock *DPMarker::getParent() { return MarkedInstr->getParent(); }

void DPMarker::removeMarker() {
  // Are there any DPValues in this DPMarker? If not, nothing to preserve.
  Instruction *Owner = MarkedInstr;
  if (StoredDPValues.empty()) {
    eraseFromParent();
    Owner->DbgMarker = nullptr;
    return;
  }

  // The attached DPValues need to be preserved; attach them to the next
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
      getParent()->setTrailingDPValues(this);
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
  dropDPValues();
  delete this;
}

iterator_range<DPValue::self_iterator> DPMarker::getDbgValueRange() {
  return make_range(StoredDPValues.begin(), StoredDPValues.end());
}
iterator_range<DPValue::const_self_iterator>
DPMarker::getDbgValueRange() const {
  return make_range(StoredDPValues.begin(), StoredDPValues.end());
}

void DPValue::removeFromParent() {
  getMarker()->StoredDPValues.erase(getIterator());
  Marker = nullptr;
}

void DPValue::eraseFromParent() {
  removeFromParent();
  deleteInstr();
}

void DPMarker::insertDPValue(DPValue *New, bool InsertAtHead) {
  auto It = InsertAtHead ? StoredDPValues.begin() : StoredDPValues.end();
  StoredDPValues.insert(It, *New);
  New->setMarker(this);
}
void DPMarker::insertDPValue(DPValue *New, DPValue *InsertBefore) {
  assert(InsertBefore->getMarker() == this &&
         "DPValue 'InsertBefore' must be contained in this DPMarker!");
  StoredDPValues.insert(InsertBefore->getIterator(), *New);
  New->setMarker(this);
}
void DPMarker::insertDPValueAfter(DPValue *New, DPValue *InsertAfter) {
  assert(InsertAfter->getMarker() == this &&
         "DPValue 'InsertAfter' must be contained in this DPMarker!");
  StoredDPValues.insert(++(InsertAfter->getIterator()), *New);
  New->setMarker(this);
}

void DPMarker::absorbDebugValues(DPMarker &Src, bool InsertAtHead) {
  auto It = InsertAtHead ? StoredDPValues.begin() : StoredDPValues.end();
  for (DPValue &DPV : Src.StoredDPValues)
    DPV.setMarker(this);

  StoredDPValues.splice(It, Src.StoredDPValues);
}

void DPMarker::absorbDebugValues(iterator_range<DPValue::self_iterator> Range,
                                 DPMarker &Src, bool InsertAtHead) {
  for (DPValue &DPV : Range)
    DPV.setMarker(this);

  auto InsertPos =
      (InsertAtHead) ? StoredDPValues.begin() : StoredDPValues.end();

  StoredDPValues.splice(InsertPos, Src.StoredDPValues, Range.begin(),
                        Range.end());
}

iterator_range<simple_ilist<DPValue>::iterator> DPMarker::cloneDebugInfoFrom(
    DPMarker *From, std::optional<simple_ilist<DPValue>::iterator> from_here,
    bool InsertAtHead) {
  DPValue *First = nullptr;
  // Work out what range of DPValues to clone: normally all the contents of the
  // "From" marker, optionally we can start from the from_here position down to
  // end().
  auto Range =
      make_range(From->StoredDPValues.begin(), From->StoredDPValues.end());
  if (from_here.has_value())
    Range = make_range(*from_here, From->StoredDPValues.end());

  // Clone each DPValue and insert into StoreDPValues; optionally place them at
  // the start or the end of the list.
  auto Pos = (InsertAtHead) ? StoredDPValues.begin() : StoredDPValues.end();
  for (DPValue &DPV : Range) {
    DPValue *New = DPV.clone();
    New->setMarker(this);
    StoredDPValues.insert(Pos, *New);
    if (!First)
      First = New;
  }

  if (!First)
    return {StoredDPValues.end(), StoredDPValues.end()};

  if (InsertAtHead)
    // If InsertAtHead is set, we cloned a range onto the front of of the
    // StoredDPValues collection, return that range.
    return {StoredDPValues.begin(), Pos};
  else
    // We inserted a block at the end, return that range.
    return {First->getIterator(), StoredDPValues.end()};
}

} // end namespace llvm

