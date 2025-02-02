//===-- IntrinsicInst.cpp - Intrinsic Instruction Wrappers ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements methods that make it really easy to deal with intrinsic
// functions.
//
// All intrinsic function calls are instances of the call instruction, so these
// are all subclasses of the CallInst class.  Note that none of these classes
// has state or virtual methods, which is an important part of this gross/neat
// hack working.
//
// In some cases, arguments to intrinsics need to be generic and are defined as
// type pointer to empty struct { }*.  To access the real item of interest the
// cast instruction needs to be stripped away.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IntrinsicInst.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Statepoint.h"
#include <optional>

using namespace llvm;

bool IntrinsicInst::mayLowerToFunctionCall(Intrinsic::ID IID) {
  switch (IID) {
  case Intrinsic::objc_autorelease:
  case Intrinsic::objc_autoreleasePoolPop:
  case Intrinsic::objc_autoreleasePoolPush:
  case Intrinsic::objc_autoreleaseReturnValue:
  case Intrinsic::objc_copyWeak:
  case Intrinsic::objc_destroyWeak:
  case Intrinsic::objc_initWeak:
  case Intrinsic::objc_loadWeak:
  case Intrinsic::objc_loadWeakRetained:
  case Intrinsic::objc_moveWeak:
  case Intrinsic::objc_release:
  case Intrinsic::objc_retain:
  case Intrinsic::objc_retainAutorelease:
  case Intrinsic::objc_retainAutoreleaseReturnValue:
  case Intrinsic::objc_retainAutoreleasedReturnValue:
  case Intrinsic::objc_retainBlock:
  case Intrinsic::objc_storeStrong:
  case Intrinsic::objc_storeWeak:
  case Intrinsic::objc_unsafeClaimAutoreleasedReturnValue:
  case Intrinsic::objc_retainedObject:
  case Intrinsic::objc_unretainedObject:
  case Intrinsic::objc_unretainedPointer:
  case Intrinsic::objc_retain_autorelease:
  case Intrinsic::objc_sync_enter:
  case Intrinsic::objc_sync_exit:
    return true;
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
/// DbgVariableIntrinsic - This is the common base class for debug info
/// intrinsics for variables.
///

iterator_range<location_op_iterator> RawLocationWrapper::location_ops() const {
  Metadata *MD = getRawLocation();
  assert(MD && "First operand of DbgVariableIntrinsic should be non-null.");
  // If operand is ValueAsMetadata, return a range over just that operand.
  if (auto *VAM = dyn_cast<ValueAsMetadata>(MD)) {
    return {location_op_iterator(VAM), location_op_iterator(VAM + 1)};
  }
  // If operand is DIArgList, return a range over its args.
  if (auto *AL = dyn_cast<DIArgList>(MD))
    return {location_op_iterator(AL->args_begin()),
            location_op_iterator(AL->args_end())};
  // Operand must be an empty metadata tuple, so return empty iterator.
  return {location_op_iterator(static_cast<ValueAsMetadata *>(nullptr)),
          location_op_iterator(static_cast<ValueAsMetadata *>(nullptr))};
}

iterator_range<location_op_iterator>
DbgVariableIntrinsic::location_ops() const {
  return getWrappedLocation().location_ops();
}

Value *DbgVariableIntrinsic::getVariableLocationOp(unsigned OpIdx) const {
  return getWrappedLocation().getVariableLocationOp(OpIdx);
}

Value *RawLocationWrapper::getVariableLocationOp(unsigned OpIdx) const {
  Metadata *MD = getRawLocation();
  assert(MD && "First operand of DbgVariableIntrinsic should be non-null.");
  if (auto *AL = dyn_cast<DIArgList>(MD))
    return AL->getArgs()[OpIdx]->getValue();
  if (isa<MDNode>(MD))
    return nullptr;
  assert(
      isa<ValueAsMetadata>(MD) &&
      "Attempted to get location operand from DbgVariableIntrinsic with none.");
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

void DbgVariableIntrinsic::replaceVariableLocationOp(Value *OldValue,
                                                     Value *NewValue,
                                                     bool AllowEmpty) {
  // If OldValue is used as the address part of a dbg.assign intrinsic replace
  // it with NewValue and return true.
  auto ReplaceDbgAssignAddress = [this, OldValue, NewValue]() -> bool {
    auto *DAI = dyn_cast<DbgAssignIntrinsic>(this);
    if (!DAI || OldValue != DAI->getAddress())
      return false;
    DAI->setAddress(NewValue);
    return true;
  };
  bool DbgAssignAddrReplaced = ReplaceDbgAssignAddress();
  (void)DbgAssignAddrReplaced;

  assert(NewValue && "Values must be non-null");
  auto Locations = location_ops();
  auto OldIt = find(Locations, OldValue);
  if (OldIt == Locations.end()) {
    if (AllowEmpty || DbgAssignAddrReplaced)
      return;
    assert(DbgAssignAddrReplaced &&
           "OldValue must be dbg.assign addr if unused in DIArgList");
    return;
  }

  assert(OldIt != Locations.end() && "OldValue must be a current location");
  if (!hasArgList()) {
    Value *NewOperand = isa<MetadataAsValue>(NewValue)
                            ? NewValue
                            : MetadataAsValue::get(
                                  getContext(), ValueAsMetadata::get(NewValue));
    return setArgOperand(0, NewOperand);
  }
  SmallVector<ValueAsMetadata *, 4> MDs;
  ValueAsMetadata *NewOperand = getAsMetadata(NewValue);
  for (auto *VMD : Locations)
    MDs.push_back(VMD == *OldIt ? NewOperand : getAsMetadata(VMD));
  setArgOperand(
      0, MetadataAsValue::get(getContext(), DIArgList::get(getContext(), MDs)));
}
void DbgVariableIntrinsic::replaceVariableLocationOp(unsigned OpIdx,
                                                     Value *NewValue) {
  assert(OpIdx < getNumVariableLocationOps() && "Invalid Operand Index");
  if (!hasArgList()) {
    Value *NewOperand = isa<MetadataAsValue>(NewValue)
                            ? NewValue
                            : MetadataAsValue::get(
                                  getContext(), ValueAsMetadata::get(NewValue));
    return setArgOperand(0, NewOperand);
  }
  SmallVector<ValueAsMetadata *, 4> MDs;
  ValueAsMetadata *NewOperand = getAsMetadata(NewValue);
  for (unsigned Idx = 0; Idx < getNumVariableLocationOps(); ++Idx)
    MDs.push_back(Idx == OpIdx ? NewOperand
                               : getAsMetadata(getVariableLocationOp(Idx)));
  setArgOperand(
      0, MetadataAsValue::get(getContext(), DIArgList::get(getContext(), MDs)));
}

void DbgVariableIntrinsic::addVariableLocationOps(ArrayRef<Value *> NewValues,
                                                  DIExpression *NewExpr) {
  assert(NewExpr->hasAllLocationOps(getNumVariableLocationOps() +
                                    NewValues.size()) &&
         "NewExpr for debug variable intrinsic does not reference every "
         "location operand.");
  assert(!is_contained(NewValues, nullptr) && "New values must be non-null");
  setArgOperand(2, MetadataAsValue::get(getContext(), NewExpr));
  SmallVector<ValueAsMetadata *, 4> MDs;
  for (auto *VMD : location_ops())
    MDs.push_back(getAsMetadata(VMD));
  for (auto *VMD : NewValues)
    MDs.push_back(getAsMetadata(VMD));
  setArgOperand(
      0, MetadataAsValue::get(getContext(), DIArgList::get(getContext(), MDs)));
}

std::optional<uint64_t> DbgVariableIntrinsic::getFragmentSizeInBits() const {
  if (auto Fragment = getExpression()->getFragmentInfo())
    return Fragment->SizeInBits;
  return getVariable()->getSizeInBits();
}

Value *DbgAssignIntrinsic::getAddress() const {
  auto *MD = getRawAddress();
  if (auto *V = dyn_cast<ValueAsMetadata>(MD))
    return V->getValue();

  // When the value goes to null, it gets replaced by an empty MDNode.
  assert(!cast<MDNode>(MD)->getNumOperands() && "Expected an empty MDNode");
  return nullptr;
}

void DbgAssignIntrinsic::setAssignId(DIAssignID *New) {
  setOperand(OpAssignID, MetadataAsValue::get(getContext(), New));
}

void DbgAssignIntrinsic::setAddress(Value *V) {
  setOperand(OpAddress,
             MetadataAsValue::get(getContext(), ValueAsMetadata::get(V)));
}

void DbgAssignIntrinsic::setKillAddress() {
  if (isKillAddress())
    return;
  setAddress(PoisonValue::get(getAddress()->getType()));
}

bool DbgAssignIntrinsic::isKillAddress() const {
  Value *Addr = getAddress();
  return !Addr || isa<UndefValue>(Addr);
}

void DbgAssignIntrinsic::setValue(Value *V) {
  setOperand(OpValue,
             MetadataAsValue::get(getContext(), ValueAsMetadata::get(V)));
}

ConstantInt *InstrProfCntrInstBase::getNumCounters() const {
  if (InstrProfValueProfileInst::classof(this))
    llvm_unreachable("InstrProfValueProfileInst does not have counters!");
  return cast<ConstantInt>(const_cast<Value *>(getArgOperand(2)));
}

ConstantInt *InstrProfCntrInstBase::getIndex() const {
  if (InstrProfValueProfileInst::classof(this))
    llvm_unreachable("Please use InstrProfValueProfileInst::getIndex()");
  return cast<ConstantInt>(const_cast<Value *>(getArgOperand(3)));
}

void InstrProfCntrInstBase::setIndex(uint32_t Idx) {
  assert(isa<InstrProfCntrInstBase>(this));
  setArgOperand(3, ConstantInt::get(Type::getInt32Ty(getContext()), Idx));
}

Value *InstrProfIncrementInst::getStep() const {
  if (InstrProfIncrementInstStep::classof(this)) {
    return const_cast<Value *>(getArgOperand(4));
  }
  const Module *M = getModule();
  LLVMContext &Context = M->getContext();
  return ConstantInt::get(Type::getInt64Ty(Context), 1);
}

Value *InstrProfCallsite::getCallee() const {
  if (isa<InstrProfCallsite>(this))
    return getArgOperand(4);
  return nullptr;
}

void InstrProfCallsite::setCallee(Value *Callee) {
  assert(isa<InstrProfCallsite>(this));
  setArgOperand(4, Callee);
}

std::optional<RoundingMode> ConstrainedFPIntrinsic::getRoundingMode() const {
  unsigned NumOperands = arg_size();
  Metadata *MD = nullptr;
  auto *MAV = dyn_cast<MetadataAsValue>(getArgOperand(NumOperands - 2));
  if (MAV)
    MD = MAV->getMetadata();
  if (!MD || !isa<MDString>(MD))
    return std::nullopt;
  return convertStrToRoundingMode(cast<MDString>(MD)->getString());
}

std::optional<fp::ExceptionBehavior>
ConstrainedFPIntrinsic::getExceptionBehavior() const {
  unsigned NumOperands = arg_size();
  Metadata *MD = nullptr;
  auto *MAV = dyn_cast<MetadataAsValue>(getArgOperand(NumOperands - 1));
  if (MAV)
    MD = MAV->getMetadata();
  if (!MD || !isa<MDString>(MD))
    return std::nullopt;
  return convertStrToExceptionBehavior(cast<MDString>(MD)->getString());
}

bool ConstrainedFPIntrinsic::isDefaultFPEnvironment() const {
  std::optional<fp::ExceptionBehavior> Except = getExceptionBehavior();
  if (Except) {
    if (*Except != fp::ebIgnore)
      return false;
  }

  std::optional<RoundingMode> Rounding = getRoundingMode();
  if (Rounding) {
    if (*Rounding != RoundingMode::NearestTiesToEven)
      return false;
  }

  return true;
}

static FCmpInst::Predicate getFPPredicateFromMD(const Value *Op) {
  Metadata *MD = cast<MetadataAsValue>(Op)->getMetadata();
  if (!MD || !isa<MDString>(MD))
    return FCmpInst::BAD_FCMP_PREDICATE;
  return StringSwitch<FCmpInst::Predicate>(cast<MDString>(MD)->getString())
      .Case("oeq", FCmpInst::FCMP_OEQ)
      .Case("ogt", FCmpInst::FCMP_OGT)
      .Case("oge", FCmpInst::FCMP_OGE)
      .Case("olt", FCmpInst::FCMP_OLT)
      .Case("ole", FCmpInst::FCMP_OLE)
      .Case("one", FCmpInst::FCMP_ONE)
      .Case("ord", FCmpInst::FCMP_ORD)
      .Case("uno", FCmpInst::FCMP_UNO)
      .Case("ueq", FCmpInst::FCMP_UEQ)
      .Case("ugt", FCmpInst::FCMP_UGT)
      .Case("uge", FCmpInst::FCMP_UGE)
      .Case("ult", FCmpInst::FCMP_ULT)
      .Case("ule", FCmpInst::FCMP_ULE)
      .Case("une", FCmpInst::FCMP_UNE)
      .Default(FCmpInst::BAD_FCMP_PREDICATE);
}

FCmpInst::Predicate ConstrainedFPCmpIntrinsic::getPredicate() const {
  return getFPPredicateFromMD(getArgOperand(2));
}

unsigned ConstrainedFPIntrinsic::getNonMetadataArgCount() const {
  // All constrained fp intrinsics have "fpexcept" metadata.
  unsigned NumArgs = arg_size() - 1;

  // Some intrinsics have "round" metadata.
  if (Intrinsic::hasConstrainedFPRoundingModeOperand(getIntrinsicID()))
    NumArgs -= 1;

  // Compare intrinsics take their predicate as metadata.
  if (isa<ConstrainedFPCmpIntrinsic>(this))
    NumArgs -= 1;

  return NumArgs;
}

bool ConstrainedFPIntrinsic::classof(const IntrinsicInst *I) {
  return Intrinsic::isConstrainedFPIntrinsic(I->getIntrinsicID());
}

ElementCount VPIntrinsic::getStaticVectorLength() const {
  auto GetVectorLengthOfType = [](const Type *T) -> ElementCount {
    const auto *VT = cast<VectorType>(T);
    auto ElemCount = VT->getElementCount();
    return ElemCount;
  };

  Value *VPMask = getMaskParam();
  if (!VPMask) {
    assert((getIntrinsicID() == Intrinsic::vp_merge ||
            getIntrinsicID() == Intrinsic::vp_select) &&
           "Unexpected VP intrinsic without mask operand");
    return GetVectorLengthOfType(getType());
  }
  return GetVectorLengthOfType(VPMask->getType());
}

Value *VPIntrinsic::getMaskParam() const {
  if (auto MaskPos = getMaskParamPos(getIntrinsicID()))
    return getArgOperand(*MaskPos);
  return nullptr;
}

void VPIntrinsic::setMaskParam(Value *NewMask) {
  auto MaskPos = getMaskParamPos(getIntrinsicID());
  setArgOperand(*MaskPos, NewMask);
}

Value *VPIntrinsic::getVectorLengthParam() const {
  if (auto EVLPos = getVectorLengthParamPos(getIntrinsicID()))
    return getArgOperand(*EVLPos);
  return nullptr;
}

void VPIntrinsic::setVectorLengthParam(Value *NewEVL) {
  auto EVLPos = getVectorLengthParamPos(getIntrinsicID());
  setArgOperand(*EVLPos, NewEVL);
}

std::optional<unsigned>
VPIntrinsic::getMaskParamPos(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  default:
    return std::nullopt;

#define BEGIN_REGISTER_VP_INTRINSIC(VPID, MASKPOS, VLENPOS)                    \
  case Intrinsic::VPID:                                                        \
    return MASKPOS;
#include "llvm/IR/VPIntrinsics.def"
  }
}

std::optional<unsigned>
VPIntrinsic::getVectorLengthParamPos(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  default:
    return std::nullopt;

#define BEGIN_REGISTER_VP_INTRINSIC(VPID, MASKPOS, VLENPOS)                    \
  case Intrinsic::VPID:                                                        \
    return VLENPOS;
#include "llvm/IR/VPIntrinsics.def"
  }
}

/// \return the alignment of the pointer used by this load/store/gather or
/// scatter.
MaybeAlign VPIntrinsic::getPointerAlignment() const {
  std::optional<unsigned> PtrParamOpt =
      getMemoryPointerParamPos(getIntrinsicID());
  assert(PtrParamOpt && "no pointer argument!");
  return getParamAlign(*PtrParamOpt);
}

/// \return The pointer operand of this load,store, gather or scatter.
Value *VPIntrinsic::getMemoryPointerParam() const {
  if (auto PtrParamOpt = getMemoryPointerParamPos(getIntrinsicID()))
    return getArgOperand(*PtrParamOpt);
  return nullptr;
}

std::optional<unsigned>
VPIntrinsic::getMemoryPointerParamPos(Intrinsic::ID VPID) {
  switch (VPID) {
  default:
    return std::nullopt;
  case Intrinsic::vp_store:
  case Intrinsic::vp_scatter:
  case Intrinsic::experimental_vp_strided_store:
    return 1;
  case Intrinsic::vp_load:
  case Intrinsic::vp_gather:
  case Intrinsic::experimental_vp_strided_load:
    return 0;
  }
}

/// \return The data (payload) operand of this store or scatter.
Value *VPIntrinsic::getMemoryDataParam() const {
  auto DataParamOpt = getMemoryDataParamPos(getIntrinsicID());
  if (!DataParamOpt)
    return nullptr;
  return getArgOperand(*DataParamOpt);
}

std::optional<unsigned> VPIntrinsic::getMemoryDataParamPos(Intrinsic::ID VPID) {
  switch (VPID) {
  default:
    return std::nullopt;
  case Intrinsic::vp_store:
  case Intrinsic::vp_scatter:
  case Intrinsic::experimental_vp_strided_store:
    return 0;
  }
}

constexpr bool isVPIntrinsic(Intrinsic::ID ID) {
  switch (ID) {
  default:
    break;
#define BEGIN_REGISTER_VP_INTRINSIC(VPID, MASKPOS, VLENPOS)                    \
  case Intrinsic::VPID:                                                        \
    return true;
#include "llvm/IR/VPIntrinsics.def"
  }
  return false;
}

bool VPIntrinsic::isVPIntrinsic(Intrinsic::ID ID) {
  return ::isVPIntrinsic(ID);
}

// Equivalent non-predicated opcode
constexpr static std::optional<unsigned>
getFunctionalOpcodeForVP(Intrinsic::ID ID) {
  switch (ID) {
  default:
    break;
#define BEGIN_REGISTER_VP_INTRINSIC(VPID, ...) case Intrinsic::VPID:
#define VP_PROPERTY_FUNCTIONAL_OPC(OPC) return Instruction::OPC;
#define END_REGISTER_VP_INTRINSIC(VPID) break;
#include "llvm/IR/VPIntrinsics.def"
  }
  return std::nullopt;
}

std::optional<unsigned>
VPIntrinsic::getFunctionalOpcodeForVP(Intrinsic::ID ID) {
  return ::getFunctionalOpcodeForVP(ID);
}

// Equivalent non-predicated intrinsic ID
constexpr static std::optional<Intrinsic::ID>
getFunctionalIntrinsicIDForVP(Intrinsic::ID ID) {
  switch (ID) {
  default:
    break;
#define BEGIN_REGISTER_VP_INTRINSIC(VPID, ...) case Intrinsic::VPID:
#define VP_PROPERTY_FUNCTIONAL_INTRINSIC(INTRIN) return Intrinsic::INTRIN;
#define END_REGISTER_VP_INTRINSIC(VPID) break;
#include "llvm/IR/VPIntrinsics.def"
  }
  return std::nullopt;
}

std::optional<Intrinsic::ID>
VPIntrinsic::getFunctionalIntrinsicIDForVP(Intrinsic::ID ID) {
  return ::getFunctionalIntrinsicIDForVP(ID);
}

constexpr static bool doesVPHaveNoFunctionalEquivalent(Intrinsic::ID ID) {
  switch (ID) {
  default:
    break;
#define BEGIN_REGISTER_VP_INTRINSIC(VPID, ...) case Intrinsic::VPID:
#define VP_PROPERTY_NO_FUNCTIONAL return true;
#define END_REGISTER_VP_INTRINSIC(VPID) break;
#include "llvm/IR/VPIntrinsics.def"
  }
  return false;
}

// All VP intrinsics should have an equivalent non-VP opcode or intrinsic
// defined, or be marked that they don't have one.
#define BEGIN_REGISTER_VP_INTRINSIC(VPID, ...)                                 \
  static_assert(doesVPHaveNoFunctionalEquivalent(Intrinsic::VPID) ||           \
                getFunctionalOpcodeForVP(Intrinsic::VPID) ||                   \
                getFunctionalIntrinsicIDForVP(Intrinsic::VPID));
#include "llvm/IR/VPIntrinsics.def"

// Equivalent non-predicated constrained intrinsic
std::optional<Intrinsic::ID>
VPIntrinsic::getConstrainedIntrinsicIDForVP(Intrinsic::ID ID) {
  switch (ID) {
  default:
    break;
#define BEGIN_REGISTER_VP_INTRINSIC(VPID, ...) case Intrinsic::VPID:
#define VP_PROPERTY_CONSTRAINEDFP(CID) return Intrinsic::CID;
#define END_REGISTER_VP_INTRINSIC(VPID) break;
#include "llvm/IR/VPIntrinsics.def"
  }
  return std::nullopt;
}

Intrinsic::ID VPIntrinsic::getForOpcode(unsigned IROPC) {
  switch (IROPC) {
  default:
    break;

#define BEGIN_REGISTER_VP_INTRINSIC(VPID, ...) break;
#define VP_PROPERTY_FUNCTIONAL_OPC(OPC) case Instruction::OPC:
#define END_REGISTER_VP_INTRINSIC(VPID) return Intrinsic::VPID;
#include "llvm/IR/VPIntrinsics.def"
  }
  return Intrinsic::not_intrinsic;
}

constexpr static Intrinsic::ID getForIntrinsic(Intrinsic::ID Id) {
  if (::isVPIntrinsic(Id))
    return Id;

  switch (Id) {
  default:
    break;
#define BEGIN_REGISTER_VP_INTRINSIC(VPID, ...) break;
#define VP_PROPERTY_FUNCTIONAL_INTRINSIC(INTRIN) case Intrinsic::INTRIN:
#define END_REGISTER_VP_INTRINSIC(VPID) return Intrinsic::VPID;
#include "llvm/IR/VPIntrinsics.def"
  }
  return Intrinsic::not_intrinsic;
}

Intrinsic::ID VPIntrinsic::getForIntrinsic(Intrinsic::ID Id) {
  return ::getForIntrinsic(Id);
}

bool VPIntrinsic::canIgnoreVectorLengthParam() const {
  using namespace PatternMatch;

  ElementCount EC = getStaticVectorLength();

  // No vlen param - no lanes masked-off by it.
  auto *VLParam = getVectorLengthParam();
  if (!VLParam)
    return true;

  // Note that the VP intrinsic causes undefined behavior if the Explicit Vector
  // Length parameter is strictly greater-than the number of vector elements of
  // the operation. This function returns true when this is detected statically
  // in the IR.

  // Check whether "W == vscale * EC.getKnownMinValue()"
  if (EC.isScalable()) {
    // Compare vscale patterns
    uint64_t VScaleFactor;
    if (match(VLParam, m_Mul(m_VScale(), m_ConstantInt(VScaleFactor))))
      return VScaleFactor >= EC.getKnownMinValue();
    return (EC.getKnownMinValue() == 1) && match(VLParam, m_VScale());
  }

  // standard SIMD operation
  const auto *VLConst = dyn_cast<ConstantInt>(VLParam);
  if (!VLConst)
    return false;

  uint64_t VLNum = VLConst->getZExtValue();
  if (VLNum >= EC.getKnownMinValue())
    return true;

  return false;
}

Function *VPIntrinsic::getOrInsertDeclarationForParams(
    Module *M, Intrinsic::ID VPID, Type *ReturnType, ArrayRef<Value *> Params) {
  assert(isVPIntrinsic(VPID) && "not a VP intrinsic");
  Function *VPFunc;
  switch (VPID) {
  default: {
    Type *OverloadTy = Params[0]->getType();
    if (VPReductionIntrinsic::isVPReduction(VPID))
      OverloadTy =
          Params[*VPReductionIntrinsic::getVectorParamPos(VPID)]->getType();

    VPFunc = Intrinsic::getOrInsertDeclaration(M, VPID, OverloadTy);
    break;
  }
  case Intrinsic::vp_trunc:
  case Intrinsic::vp_sext:
  case Intrinsic::vp_zext:
  case Intrinsic::vp_fptoui:
  case Intrinsic::vp_fptosi:
  case Intrinsic::vp_uitofp:
  case Intrinsic::vp_sitofp:
  case Intrinsic::vp_fptrunc:
  case Intrinsic::vp_fpext:
  case Intrinsic::vp_ptrtoint:
  case Intrinsic::vp_inttoptr:
  case Intrinsic::vp_lrint:
  case Intrinsic::vp_llrint:
  case Intrinsic::vp_cttz_elts:
    VPFunc = Intrinsic::getOrInsertDeclaration(
        M, VPID, {ReturnType, Params[0]->getType()});
    break;
  case Intrinsic::vp_is_fpclass:
    VPFunc = Intrinsic::getOrInsertDeclaration(M, VPID, {Params[0]->getType()});
    break;
  case Intrinsic::vp_merge:
  case Intrinsic::vp_select:
    VPFunc = Intrinsic::getOrInsertDeclaration(M, VPID, {Params[1]->getType()});
    break;
  case Intrinsic::vp_load:
    VPFunc = Intrinsic::getOrInsertDeclaration(
        M, VPID, {ReturnType, Params[0]->getType()});
    break;
  case Intrinsic::experimental_vp_strided_load:
    VPFunc = Intrinsic::getOrInsertDeclaration(
        M, VPID, {ReturnType, Params[0]->getType(), Params[1]->getType()});
    break;
  case Intrinsic::vp_gather:
    VPFunc = Intrinsic::getOrInsertDeclaration(
        M, VPID, {ReturnType, Params[0]->getType()});
    break;
  case Intrinsic::vp_store:
    VPFunc = Intrinsic::getOrInsertDeclaration(
        M, VPID, {Params[0]->getType(), Params[1]->getType()});
    break;
  case Intrinsic::experimental_vp_strided_store:
    VPFunc = Intrinsic::getOrInsertDeclaration(
        M, VPID,
        {Params[0]->getType(), Params[1]->getType(), Params[2]->getType()});
    break;
  case Intrinsic::vp_scatter:
    VPFunc = Intrinsic::getOrInsertDeclaration(
        M, VPID, {Params[0]->getType(), Params[1]->getType()});
    break;
  case Intrinsic::experimental_vp_splat:
    VPFunc = Intrinsic::getOrInsertDeclaration(M, VPID, ReturnType);
    break;
  }
  assert(VPFunc && "Could not declare VP intrinsic");
  return VPFunc;
}

bool VPReductionIntrinsic::isVPReduction(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::vp_reduce_add:
  case Intrinsic::vp_reduce_mul:
  case Intrinsic::vp_reduce_and:
  case Intrinsic::vp_reduce_or:
  case Intrinsic::vp_reduce_xor:
  case Intrinsic::vp_reduce_smax:
  case Intrinsic::vp_reduce_smin:
  case Intrinsic::vp_reduce_umax:
  case Intrinsic::vp_reduce_umin:
  case Intrinsic::vp_reduce_fmax:
  case Intrinsic::vp_reduce_fmin:
  case Intrinsic::vp_reduce_fmaximum:
  case Intrinsic::vp_reduce_fminimum:
  case Intrinsic::vp_reduce_fadd:
  case Intrinsic::vp_reduce_fmul:
    return true;
  default:
    return false;
  }
}

bool VPCastIntrinsic::isVPCast(Intrinsic::ID ID) {
  // All of the vp.casts correspond to instructions
  if (std::optional<unsigned> Opc = getFunctionalOpcodeForVP(ID))
    return Instruction::isCast(*Opc);
  return false;
}

bool VPCmpIntrinsic::isVPCmp(Intrinsic::ID ID) {
  switch (ID) {
  default:
    return false;
  case Intrinsic::vp_fcmp:
  case Intrinsic::vp_icmp:
    return true;
  }
}

bool VPBinOpIntrinsic::isVPBinOp(Intrinsic::ID ID) {
  switch (ID) {
  default:
    break;
#define BEGIN_REGISTER_VP_INTRINSIC(VPID, ...) case Intrinsic::VPID:
#define VP_PROPERTY_BINARYOP return true;
#define END_REGISTER_VP_INTRINSIC(VPID) break;
#include "llvm/IR/VPIntrinsics.def"
  }
  return false;
}

static ICmpInst::Predicate getIntPredicateFromMD(const Value *Op) {
  Metadata *MD = cast<MetadataAsValue>(Op)->getMetadata();
  if (!MD || !isa<MDString>(MD))
    return ICmpInst::BAD_ICMP_PREDICATE;
  return StringSwitch<ICmpInst::Predicate>(cast<MDString>(MD)->getString())
      .Case("eq", ICmpInst::ICMP_EQ)
      .Case("ne", ICmpInst::ICMP_NE)
      .Case("ugt", ICmpInst::ICMP_UGT)
      .Case("uge", ICmpInst::ICMP_UGE)
      .Case("ult", ICmpInst::ICMP_ULT)
      .Case("ule", ICmpInst::ICMP_ULE)
      .Case("sgt", ICmpInst::ICMP_SGT)
      .Case("sge", ICmpInst::ICMP_SGE)
      .Case("slt", ICmpInst::ICMP_SLT)
      .Case("sle", ICmpInst::ICMP_SLE)
      .Default(ICmpInst::BAD_ICMP_PREDICATE);
}

CmpInst::Predicate VPCmpIntrinsic::getPredicate() const {
  assert(isVPCmp(getIntrinsicID()));
  return getIntrinsicID() == Intrinsic::vp_fcmp
             ? getFPPredicateFromMD(getArgOperand(2))
             : getIntPredicateFromMD(getArgOperand(2));
}

unsigned VPReductionIntrinsic::getVectorParamPos() const {
  return *VPReductionIntrinsic::getVectorParamPos(getIntrinsicID());
}

unsigned VPReductionIntrinsic::getStartParamPos() const {
  return *VPReductionIntrinsic::getStartParamPos(getIntrinsicID());
}

std::optional<unsigned>
VPReductionIntrinsic::getVectorParamPos(Intrinsic::ID ID) {
  if (isVPReduction(ID))
    return 1;
  return std::nullopt;
}

std::optional<unsigned>
VPReductionIntrinsic::getStartParamPos(Intrinsic::ID ID) {
  if (isVPReduction(ID))
    return 0;
  return std::nullopt;
}

Instruction::BinaryOps BinaryOpIntrinsic::getBinaryOp() const {
  switch (getIntrinsicID()) {
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::uadd_sat:
  case Intrinsic::sadd_sat:
    return Instruction::Add;
  case Intrinsic::usub_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::usub_sat:
  case Intrinsic::ssub_sat:
    return Instruction::Sub;
  case Intrinsic::umul_with_overflow:
  case Intrinsic::smul_with_overflow:
    return Instruction::Mul;
  default:
    llvm_unreachable("Invalid intrinsic");
  }
}

bool BinaryOpIntrinsic::isSigned() const {
  switch (getIntrinsicID()) {
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::smul_with_overflow:
  case Intrinsic::sadd_sat:
  case Intrinsic::ssub_sat:
    return true;
  default:
    return false;
  }
}

unsigned BinaryOpIntrinsic::getNoWrapKind() const {
  if (isSigned())
    return OverflowingBinaryOperator::NoSignedWrap;
  else
    return OverflowingBinaryOperator::NoUnsignedWrap;
}

const Value *GCProjectionInst::getStatepoint() const {
  const Value *Token = getArgOperand(0);
  if (isa<UndefValue>(Token))
    return Token;

  // Treat none token as if it was undef here
  if (isa<ConstantTokenNone>(Token))
    return UndefValue::get(Token->getType());

  // This takes care both of relocates for call statepoints and relocates
  // on normal path of invoke statepoint.
  if (!isa<LandingPadInst>(Token))
    return cast<GCStatepointInst>(Token);

  // This relocate is on exceptional path of an invoke statepoint
  const BasicBlock *InvokeBB =
    cast<Instruction>(Token)->getParent()->getUniquePredecessor();

  assert(InvokeBB && "safepoints should have unique landingpads");
  assert(InvokeBB->getTerminator() &&
         "safepoint block should be well formed");

  return cast<GCStatepointInst>(InvokeBB->getTerminator());
}

Value *GCRelocateInst::getBasePtr() const {
  auto Statepoint = getStatepoint();
  if (isa<UndefValue>(Statepoint))
    return UndefValue::get(Statepoint->getType());

  auto *GCInst = cast<GCStatepointInst>(Statepoint);
  if (auto Opt = GCInst->getOperandBundle(LLVMContext::OB_gc_live))
    return *(Opt->Inputs.begin() + getBasePtrIndex());
  return *(GCInst->arg_begin() + getBasePtrIndex());
}

Value *GCRelocateInst::getDerivedPtr() const {
  auto *Statepoint = getStatepoint();
  if (isa<UndefValue>(Statepoint))
    return UndefValue::get(Statepoint->getType());

  auto *GCInst = cast<GCStatepointInst>(Statepoint);
  if (auto Opt = GCInst->getOperandBundle(LLVMContext::OB_gc_live))
    return *(Opt->Inputs.begin() + getDerivedPtrIndex());
  return *(GCInst->arg_begin() + getDerivedPtrIndex());
}
