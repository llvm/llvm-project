//===-- InstrinsicInst.cpp - Intrinsic Instruction Wrappers ---------------===//
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
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"

#include "llvm/Support/raw_ostream.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
/// DbgVariableIntrinsic - This is the common base class for debug info
/// intrinsics for variables.
///

Value *DbgVariableIntrinsic::getVariableLocation(bool AllowNullOp) const {
  Value *Op = getArgOperand(0);
  if (AllowNullOp && !Op)
    return nullptr;

  auto *MD = cast<MetadataAsValue>(Op)->getMetadata();
  if (auto *V = dyn_cast<ValueAsMetadata>(MD))
    return V->getValue();

  // When the value goes to null, it gets replaced by an empty MDNode.
  assert(!cast<MDNode>(MD)->getNumOperands() && "Expected an empty MDNode");
  return nullptr;
}

Optional<uint64_t> DbgVariableIntrinsic::getFragmentSizeInBits() const {
  if (auto Fragment = getExpression()->getFragmentInfo())
    return Fragment->SizeInBits;
  return getVariable()->getSizeInBits();
}

int llvm::Intrinsic::lookupLLVMIntrinsicByName(ArrayRef<const char *> NameTable,
                                               StringRef Name) {
  assert(Name.startswith("llvm."));

  // Do successive binary searches of the dotted name components. For
  // "llvm.gc.experimental.statepoint.p1i8.p1i32", we will find the range of
  // intrinsics starting with "llvm.gc", then "llvm.gc.experimental", then
  // "llvm.gc.experimental.statepoint", and then we will stop as the range is
  // size 1. During the search, we can skip the prefix that we already know is
  // identical. By using strncmp we consider names with differing suffixes to
  // be part of the equal range.
  size_t CmpEnd = 4; // Skip the "llvm" component.
  const char *const *Low = NameTable.begin();
  const char *const *High = NameTable.end();
  const char *const *LastLow = Low;
  while (CmpEnd < Name.size() && High - Low > 0) {
    size_t CmpStart = CmpEnd;
    CmpEnd = Name.find('.', CmpStart + 1);
    CmpEnd = CmpEnd == StringRef::npos ? Name.size() : CmpEnd;
    auto Cmp = [CmpStart, CmpEnd](const char *LHS, const char *RHS) {
      return strncmp(LHS + CmpStart, RHS + CmpStart, CmpEnd - CmpStart) < 0;
    };
    LastLow = Low;
    std::tie(Low, High) = std::equal_range(Low, High, Name.data(), Cmp);
  }
  if (High - Low > 0)
    LastLow = Low;

  if (LastLow == NameTable.end())
    return -1;
  StringRef NameFound = *LastLow;
  if (Name == NameFound ||
      (Name.startswith(NameFound) && Name[NameFound.size()] == '.'))
    return LastLow - NameTable.begin();
  return -1;
}

Value *InstrProfIncrementInst::getStep() const {
  if (InstrProfIncrementInstStep::classof(this)) {
    return const_cast<Value *>(getArgOperand(4));
  }
  const Module *M = getModule();
  LLVMContext &Context = M->getContext();
  return ConstantInt::get(Type::getInt64Ty(Context), 1);
}

Optional<RoundingMode> ConstrainedFPIntrinsic::getRoundingMode() const {
  unsigned NumOperands = getNumArgOperands();
  Metadata *MD =
      cast<MetadataAsValue>(getArgOperand(NumOperands - 2))->getMetadata();
  if (!MD || !isa<MDString>(MD))
    return None;
  return StrToRoundingMode(cast<MDString>(MD)->getString());
}

Optional<fp::ExceptionBehavior>
ConstrainedFPIntrinsic::getExceptionBehavior() const {
  unsigned NumOperands = getNumArgOperands();
  Metadata *MD =
      cast<MetadataAsValue>(getArgOperand(NumOperands - 1))->getMetadata();
  if (!MD || !isa<MDString>(MD))
    return None;
  return StrToExceptionBehavior(cast<MDString>(MD)->getString());
}

FCmpInst::Predicate ConstrainedFPCmpIntrinsic::getPredicate() const {
  Metadata *MD = cast<MetadataAsValue>(getArgOperand(2))->getMetadata();
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

bool ConstrainedFPIntrinsic::isUnaryOp() const {
  switch (getIntrinsicID()) {
  default:
    return false;
#define INSTRUCTION(NAME, NARG, ROUND_MODE, INTRINSIC)                         \
  case Intrinsic::INTRINSIC:                                                   \
    return NARG == 1;
#include "llvm/IR/ConstrainedOps.def"
  }
}

bool ConstrainedFPIntrinsic::isTernaryOp() const {
  switch (getIntrinsicID()) {
  default:
    return false;
#define INSTRUCTION(NAME, NARG, ROUND_MODE, INTRINSIC)                         \
  case Intrinsic::INTRINSIC:                                                   \
    return NARG == 3;
#include "llvm/IR/ConstrainedOps.def"
  }
}

bool ConstrainedFPIntrinsic::classof(const IntrinsicInst *I) {
  switch (I->getIntrinsicID()) {
#define INSTRUCTION(NAME, NARGS, ROUND_MODE, INTRINSIC)                        \
  case Intrinsic::INTRINSIC:
#include "llvm/IR/ConstrainedOps.def"
    return true;
  default:
    return false;
  }
}

ElementCount VPIntrinsic::getStaticVectorLength() const {
  auto GetVectorLengthOfType = [](const Type *T) -> ElementCount {
    auto VT = cast<VectorType>(T);
    auto ElemCount = VT->getElementCount();
    return ElemCount;
  };

  auto VPMask = getMaskParam();
  return GetVectorLengthOfType(VPMask->getType());
}

Value *VPIntrinsic::getMaskParam() const {
  auto maskPos = GetMaskParamPos(getIntrinsicID());
  if (maskPos)
    return getArgOperand(maskPos.getValue());
  return nullptr;
}

Value *VPIntrinsic::getVectorLengthParam() const {
  auto vlenPos = GetVectorLengthParamPos(getIntrinsicID());
  if (vlenPos)
    return getArgOperand(vlenPos.getValue());
  return nullptr;
}

Optional<int> VPIntrinsic::GetMaskParamPos(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  default:
    return None;

#define REGISTER_VP_INTRINSIC(VPID, MASKPOS, VLENPOS)                          \
  case Intrinsic::VPID:                                                        \
    return MASKPOS;
#include "llvm/IR/VPIntrinsics.def"
  }
}

Optional<int> VPIntrinsic::GetVectorLengthParamPos(Intrinsic::ID IntrinsicID) {
  switch (IntrinsicID) {
  default:
    return None;

#define REGISTER_VP_INTRINSIC(VPID, MASKPOS, VLENPOS)                          \
  case Intrinsic::VPID:                                                        \
    return VLENPOS;
#include "llvm/IR/VPIntrinsics.def"
  }
}

bool VPIntrinsic::IsVPIntrinsic(Intrinsic::ID ID) {
  switch (ID) {
  default:
    return false;

#define REGISTER_VP_INTRINSIC(VPID, MASKPOS, VLENPOS)                          \
  case Intrinsic::VPID:                                                        \
    break;
#include "llvm/IR/VPIntrinsics.def"
  }
  return true;
}

// Equivalent non-predicated opcode
unsigned VPIntrinsic::GetFunctionalOpcodeForVP(Intrinsic::ID ID) {
  switch (ID) {
  default:
    return Instruction::Call;

#define HANDLE_VP_TO_OC(VPID, OC)                                              \
  case Intrinsic::VPID:                                                        \
    return Instruction::OC;
#include "llvm/IR/VPIntrinsics.def"
  }
}

Intrinsic::ID VPIntrinsic::GetForOpcode(unsigned OC) {
  switch (OC) {
  default:
    return Intrinsic::not_intrinsic;

#define HANDLE_VP_TO_OC(VPID, OC)                                              \
  case Instruction::OC:                                                        \
    return Intrinsic::VPID;
#include "llvm/IR/VPIntrinsics.def"
  }
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

  // Check whether "W == vscale * EC.Min"
  if (EC.Scalable) {
    // Undig the DL
    auto ParMod = this->getModule();
    if (!ParMod)
      return false;
    const auto &DL = ParMod->getDataLayout();

    // Compare vscale patterns
    uint64_t ParamFactor;
    if (EC.Min > 1 &&
        match(VLParam, m_c_BinOp(m_ConstantInt(ParamFactor), m_VScale(DL)))) {
      return ParamFactor >= EC.Min;
    }
    if (match(VLParam, m_VScale(DL))) {
      return ParamFactor;
    }
    return false;
  }

  // standard SIMD operation
  auto VLConst = dyn_cast<ConstantInt>(VLParam);
  if (!VLConst)
    return false;

  uint64_t VLNum = VLConst->getZExtValue();
  if (VLNum >= EC.Min)
    return true;

  return false;
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
