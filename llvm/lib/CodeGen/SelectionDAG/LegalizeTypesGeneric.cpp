//===-------- LegalizeTypesGeneric.cpp - Generic type legalization --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements generic type expansion and splitting for LegalizeTypes.
// The routines here perform legalization when the details of the type (such as
// whether it is an integer or a float) do not matter.
// Expansion is the act of changing a computation in an illegal type to be a
// computation in two identical registers of a smaller type.
// Splitting is the act of changing a computation in an illegal type to be a
// computation in two not necessarily identical registers of a smaller type.
//
//===----------------------------------------------------------------------===//

#include "LegalizeTypes.h"
#include "llvm/Target/TargetData.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// Generic Result Expansion.
//===----------------------------------------------------------------------===//

// These routines assume that the Lo/Hi part is stored first in memory on
// little/big-endian machines, followed by the Hi/Lo part.  This means that
// they cannot be used as is on vectors, for which Lo is always stored first.

void DAGTypeLegalizer::ExpandRes_BIT_CONVERT(SDNode *N, SDValue &Lo,
                                             SDValue &Hi) {
  MVT OutVT = N->getValueType(0);
  MVT NOutVT = TLI.getTypeToTransformTo(OutVT);
  SDValue InOp = N->getOperand(0);
  MVT InVT = InOp.getValueType();
  DebugLoc dl = N->getDebugLoc();

  // Handle some special cases efficiently.
  switch (getTypeAction(InVT)) {
    default:
      assert(false && "Unknown type action!");
    case Legal:
    case PromoteInteger:
      break;
    case SoftenFloat:
      // Convert the integer operand instead.
      SplitInteger(GetSoftenedFloat(InOp), Lo, Hi);
      Lo = DAG.getNode(ISD::BIT_CONVERT, dl, NOutVT, Lo);
      Hi = DAG.getNode(ISD::BIT_CONVERT, dl, NOutVT, Hi);
      return;
    case ExpandInteger:
    case ExpandFloat:
      // Convert the expanded pieces of the input.
      GetExpandedOp(InOp, Lo, Hi);
      Lo = DAG.getNode(ISD::BIT_CONVERT, dl, NOutVT, Lo);
      Hi = DAG.getNode(ISD::BIT_CONVERT, dl, NOutVT, Hi);
      return;
    case SplitVector:
      // Convert the split parts of the input if it was split in two.
      GetSplitVector(InOp, Lo, Hi);
      if (Lo.getValueType() == Hi.getValueType()) {
        if (TLI.isBigEndian())
          std::swap(Lo, Hi);
        Lo = DAG.getNode(ISD::BIT_CONVERT, dl, NOutVT, Lo);
        Hi = DAG.getNode(ISD::BIT_CONVERT, dl, NOutVT, Hi);
        return;
      }
      break;
    case ScalarizeVector:
      // Convert the element instead.
      SplitInteger(BitConvertToInteger(GetScalarizedVector(InOp)), Lo, Hi);
      Lo = DAG.getNode(ISD::BIT_CONVERT, dl, NOutVT, Lo);
      Hi = DAG.getNode(ISD::BIT_CONVERT, dl, NOutVT, Hi);
      return;
  }

  // Lower the bit-convert to a store/load from the stack.
  assert(NOutVT.isByteSized() && "Expanded type not byte sized!");

  // Create the stack frame object.  Make sure it is aligned for both
  // the source and expanded destination types.
  unsigned Alignment =
    TLI.getTargetData()->getPrefTypeAlignment(NOutVT.getTypeForMVT());
  SDValue StackPtr = DAG.CreateStackTemporary(InVT, Alignment);
  int SPFI = cast<FrameIndexSDNode>(StackPtr.getNode())->getIndex();
  const Value *SV = PseudoSourceValue::getFixedStack(SPFI);

  // Emit a store to the stack slot.
  SDValue Store = DAG.getStore(DAG.getEntryNode(), dl, InOp, StackPtr, SV, 0);

  // Load the first half from the stack slot.
  Lo = DAG.getLoad(NOutVT, dl, Store, StackPtr, SV, 0);

  // Increment the pointer to the other half.
  unsigned IncrementSize = NOutVT.getSizeInBits() / 8;
  StackPtr = DAG.getNode(ISD::ADD, dl, StackPtr.getValueType(), StackPtr,
                         DAG.getIntPtrConstant(IncrementSize));

  // Load the second half from the stack slot.
  Hi = DAG.getLoad(NOutVT, dl, Store, StackPtr, SV, IncrementSize, false,
                   MinAlign(Alignment, IncrementSize));

  // Handle endianness of the load.
  if (TLI.isBigEndian())
    std::swap(Lo, Hi);
}

void DAGTypeLegalizer::ExpandRes_BUILD_PAIR(SDNode *N, SDValue &Lo,
                                            SDValue &Hi) {
  // Return the operands.
  Lo = N->getOperand(0);
  Hi = N->getOperand(1);
}

void DAGTypeLegalizer::ExpandRes_EXTRACT_ELEMENT(SDNode *N, SDValue &Lo,
                                                 SDValue &Hi) {
  GetExpandedOp(N->getOperand(0), Lo, Hi);
  SDValue Part = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue() ?
                   Hi : Lo;
  DebugLoc dl = N->getDebugLoc();

  assert(Part.getValueType() == N->getValueType(0) &&
         "Type twice as big as expanded type not itself expanded!");
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));

  Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, NVT, Part,
                   DAG.getConstant(0, TLI.getPointerTy()));
  Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, dl, NVT, Part,
                   DAG.getConstant(1, TLI.getPointerTy()));
}

void DAGTypeLegalizer::ExpandRes_EXTRACT_VECTOR_ELT(SDNode *N, SDValue &Lo,
                                                    SDValue &Hi) {
  SDValue OldVec = N->getOperand(0);
  unsigned OldElts = OldVec.getValueType().getVectorNumElements();
  DebugLoc dl = N->getDebugLoc();

  // Convert to a vector of the expanded element type, for example
  // <3 x i64> -> <6 x i32>.
  MVT OldVT = N->getValueType(0);
  MVT NewVT = TLI.getTypeToTransformTo(OldVT);

  SDValue NewVec = DAG.getNode(ISD::BIT_CONVERT, dl,
                                 MVT::getVectorVT(NewVT, 2*OldElts),
                                 OldVec);

  // Extract the elements at 2 * Idx and 2 * Idx + 1 from the new vector.
  SDValue Idx = N->getOperand(1);

  // Make sure the type of Idx is big enough to hold the new values.
  if (Idx.getValueType().bitsLT(TLI.getPointerTy()))
    Idx = DAG.getNode(ISD::ZERO_EXTEND, dl, TLI.getPointerTy(), Idx);

  Idx = DAG.getNode(ISD::ADD, Idx.getValueType(), Idx, Idx);
  Lo = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, NewVT, NewVec, Idx);

  Idx = DAG.getNode(ISD::ADD, dl, Idx.getValueType(), Idx,
                    DAG.getConstant(1, Idx.getValueType()));
  Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, NewVT, NewVec, Idx);

  if (TLI.isBigEndian())
    std::swap(Lo, Hi);
}

void DAGTypeLegalizer::ExpandRes_NormalLoad(SDNode *N, SDValue &Lo,
                                            SDValue &Hi) {
  assert(ISD::isNormalLoad(N) && "This routine only for normal loads!");
  DebugLoc dl = N->getDebugLoc();

  LoadSDNode *LD = cast<LoadSDNode>(N);
  MVT NVT = TLI.getTypeToTransformTo(LD->getValueType(0));
  SDValue Chain = LD->getChain();
  SDValue Ptr = LD->getBasePtr();
  int SVOffset = LD->getSrcValueOffset();
  unsigned Alignment = LD->getAlignment();
  bool isVolatile = LD->isVolatile();

  assert(NVT.isByteSized() && "Expanded type not byte sized!");

  Lo = DAG.getLoad(NVT, dl, Chain, Ptr, LD->getSrcValue(), SVOffset,
                   isVolatile, Alignment);

  // Increment the pointer to the other half.
  unsigned IncrementSize = NVT.getSizeInBits() / 8;
  Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                    DAG.getIntPtrConstant(IncrementSize));
  Hi = DAG.getLoad(NVT, dl, Chain, Ptr, LD->getSrcValue(),
                   SVOffset+IncrementSize,
                   isVolatile, MinAlign(Alignment, IncrementSize));

  // Build a factor node to remember that this load is independent of the
  // other one.
  Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo.getValue(1),
                      Hi.getValue(1));

  // Handle endianness of the load.
  if (TLI.isBigEndian())
    std::swap(Lo, Hi);

  // Modified the chain - switch anything that used the old chain to use
  // the new one.
  ReplaceValueWith(SDValue(N, 1), Chain);
}

void DAGTypeLegalizer::ExpandRes_VAARG(SDNode *N, SDValue &Lo, SDValue &Hi) {
  MVT NVT = TLI.getTypeToTransformTo(N->getValueType(0));
  SDValue Chain = N->getOperand(0);
  SDValue Ptr = N->getOperand(1);

  Lo = DAG.getVAArg(NVT, Chain, Ptr, N->getOperand(2));
  Hi = DAG.getVAArg(NVT, Lo.getValue(1), Ptr, N->getOperand(2));

  // Handle endianness of the load.
  if (TLI.isBigEndian())
    std::swap(Lo, Hi);

  // Modified the chain - switch anything that used the old chain to use
  // the new one.
  ReplaceValueWith(SDValue(N, 1), Hi.getValue(1));
}


//===--------------------------------------------------------------------===//
// Generic Operand Expansion.
//===--------------------------------------------------------------------===//

SDValue DAGTypeLegalizer::ExpandOp_BIT_CONVERT(SDNode *N) {
  DebugLoc dl = N->getDebugLoc();
  if (N->getValueType(0).isVector()) {
    // An illegal expanding type is being converted to a legal vector type.
    // Make a two element vector out of the expanded parts and convert that
    // instead, but only if the new vector type is legal (otherwise there
    // is no point, and it might create expansion loops).  For example, on
    // x86 this turns v1i64 = BIT_CONVERT i64 into v1i64 = BIT_CONVERT v2i32.
    MVT OVT = N->getOperand(0).getValueType();
    MVT NVT = MVT::getVectorVT(TLI.getTypeToTransformTo(OVT), 2);

    if (isTypeLegal(NVT)) {
      SDValue Parts[2];
      GetExpandedOp(N->getOperand(0), Parts[0], Parts[1]);

      if (TLI.isBigEndian())
        std::swap(Parts[0], Parts[1]);

      SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, dl, NVT, Parts, 2);
      return DAG.getNode(ISD::BIT_CONVERT, dl, N->getValueType(0), Vec);
    }
  }

  // Otherwise, store to a temporary and load out again as the new type.
  return CreateStackStoreLoad(N->getOperand(0), N->getValueType(0));
}

SDValue DAGTypeLegalizer::ExpandOp_BUILD_VECTOR(SDNode *N) {
  // The vector type is legal but the element type needs expansion.
  MVT VecVT = N->getValueType(0);
  unsigned NumElts = VecVT.getVectorNumElements();
  MVT OldVT = N->getOperand(0).getValueType();
  MVT NewVT = TLI.getTypeToTransformTo(OldVT);
  DebugLoc dl = N->getDebugLoc();

  // Build a vector of twice the length out of the expanded elements.
  // For example <3 x i64> -> <6 x i32>.
  std::vector<SDValue> NewElts;
  NewElts.reserve(NumElts*2);

  for (unsigned i = 0; i < NumElts; ++i) {
    SDValue Lo, Hi;
    GetExpandedOp(N->getOperand(i), Lo, Hi);
    if (TLI.isBigEndian())
      std::swap(Lo, Hi);
    NewElts.push_back(Lo);
    NewElts.push_back(Hi);
  }

  SDValue NewVec = DAG.getNode(ISD::BUILD_VECTOR, dl,
                                 MVT::getVectorVT(NewVT, NewElts.size()),
                                 &NewElts[0], NewElts.size());

  // Convert the new vector to the old vector type.
  return DAG.getNode(ISD::BIT_CONVERT, dl, VecVT, NewVec);
}

SDValue DAGTypeLegalizer::ExpandOp_EXTRACT_ELEMENT(SDNode *N) {
  SDValue Lo, Hi;
  GetExpandedOp(N->getOperand(0), Lo, Hi);
  return cast<ConstantSDNode>(N->getOperand(1))->getZExtValue() ? Hi : Lo;
}

SDValue DAGTypeLegalizer::ExpandOp_INSERT_VECTOR_ELT(SDNode *N) {
  // The vector type is legal but the element type needs expansion.
  MVT VecVT = N->getValueType(0);
  unsigned NumElts = VecVT.getVectorNumElements();
  DebugLoc dl = N->getDebugLoc();

  SDValue Val = N->getOperand(1);
  MVT OldEVT = Val.getValueType();
  MVT NewEVT = TLI.getTypeToTransformTo(OldEVT);

  assert(OldEVT == VecVT.getVectorElementType() &&
         "Inserted element type doesn't match vector element type!");

  // Bitconvert to a vector of twice the length with elements of the expanded
  // type, insert the expanded vector elements, and then convert back.
  MVT NewVecVT = MVT::getVectorVT(NewEVT, NumElts*2);
  SDValue NewVec = DAG.getNode(ISD::BIT_CONVERT, dl,
                               NewVecVT, N->getOperand(0));

  SDValue Lo, Hi;
  GetExpandedOp(Val, Lo, Hi);
  if (TLI.isBigEndian())
    std::swap(Lo, Hi);

  SDValue Idx = N->getOperand(2);
  Idx = DAG.getNode(ISD::ADD, dl, Idx.getValueType(), Idx, Idx);
  NewVec = DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, NewVecVT, NewVec, Lo, Idx);
  Idx = DAG.getNode(ISD::ADD, dl,
                    Idx.getValueType(), Idx, DAG.getIntPtrConstant(1));
  NewVec =  DAG.getNode(ISD::INSERT_VECTOR_ELT, dl, NewVecVT, NewVec, Hi, Idx);

  // Convert the new vector to the old vector type.
  return DAG.getNode(ISD::BIT_CONVERT, dl, VecVT, NewVec);
}

SDValue DAGTypeLegalizer::ExpandOp_SCALAR_TO_VECTOR(SDNode *N) {
  DebugLoc dl = N->getDebugLoc();
  MVT VT = N->getValueType(0);
  unsigned NumElts = VT.getVectorNumElements();
  SmallVector<SDValue, 16> Ops(NumElts);
  Ops[0] = N->getOperand(0);
  SDValue UndefVal = DAG.getNode(ISD::UNDEF, dl, Ops[0].getValueType());
  for (unsigned i = 1; i < NumElts; ++i)
    Ops[i] = UndefVal;
  return DAG.getNode(ISD::BUILD_VECTOR, dl, VT, &Ops[0], NumElts);
}

SDValue DAGTypeLegalizer::ExpandOp_NormalStore(SDNode *N, unsigned OpNo) {
  assert(ISD::isNormalStore(N) && "This routine only for normal stores!");
  assert(OpNo == 1 && "Can only expand the stored value so far");
  DebugLoc dl = N->getDebugLoc();

  StoreSDNode *St = cast<StoreSDNode>(N);
  MVT NVT = TLI.getTypeToTransformTo(St->getValue().getValueType());
  SDValue Chain = St->getChain();
  SDValue Ptr = St->getBasePtr();
  int SVOffset = St->getSrcValueOffset();
  unsigned Alignment = St->getAlignment();
  bool isVolatile = St->isVolatile();

  assert(NVT.isByteSized() && "Expanded type not byte sized!");
  unsigned IncrementSize = NVT.getSizeInBits() / 8;

  SDValue Lo, Hi;
  GetExpandedOp(St->getValue(), Lo, Hi);

  if (TLI.isBigEndian())
    std::swap(Lo, Hi);

  Lo = DAG.getStore(Chain, dl, Lo, Ptr, St->getSrcValue(), SVOffset,
                    isVolatile, Alignment);

  Ptr = DAG.getNode(ISD::ADD, dl, Ptr.getValueType(), Ptr,
                    DAG.getIntPtrConstant(IncrementSize));
  assert(isTypeLegal(Ptr.getValueType()) && "Pointers must be legal!");
  Hi = DAG.getStore(Chain, dl, Hi, Ptr, St->getSrcValue(),
                    SVOffset + IncrementSize,
                    isVolatile, MinAlign(Alignment, IncrementSize));

  return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Lo, Hi);
}


//===--------------------------------------------------------------------===//
// Generic Result Splitting.
//===--------------------------------------------------------------------===//

// Be careful to make no assumptions about which of Lo/Hi is stored first in
// memory (for vectors it is always Lo first followed by Hi in the following
// bytes; for integers and floats it is Lo first if and only if the machine is
// little-endian).

void DAGTypeLegalizer::SplitRes_MERGE_VALUES(SDNode *N,
                                             SDValue &Lo, SDValue &Hi) {
  // A MERGE_VALUES node can produce any number of values.  We know that the
  // first illegal one needs to be expanded into Lo/Hi.
  unsigned i;

  // The string of legal results gets turned into input operands, which have
  // the same type.
  for (i = 0; isTypeLegal(N->getValueType(i)); ++i)
    ReplaceValueWith(SDValue(N, i), SDValue(N->getOperand(i)));

  // The first illegal result must be the one that needs to be expanded.
  GetSplitOp(N->getOperand(i), Lo, Hi);

  // Legalize the rest of the results into the input operands whether they are
  // legal or not.
  unsigned e = N->getNumValues();
  for (++i; i != e; ++i)
    ReplaceValueWith(SDValue(N, i), SDValue(N->getOperand(i)));
}

void DAGTypeLegalizer::SplitRes_SELECT(SDNode *N, SDValue &Lo,
                                       SDValue &Hi) {
  SDValue LL, LH, RL, RH;
  DebugLoc dl = N->getDebugLoc();
  GetSplitOp(N->getOperand(1), LL, LH);
  GetSplitOp(N->getOperand(2), RL, RH);

  SDValue Cond = N->getOperand(0);
  Lo = DAG.getNode(ISD::SELECT, dl, LL.getValueType(), Cond, LL, RL);
  Hi = DAG.getNode(ISD::SELECT, dl, LH.getValueType(), Cond, LH, RH);
}

void DAGTypeLegalizer::SplitRes_SELECT_CC(SDNode *N, SDValue &Lo,
                                          SDValue &Hi) {
  SDValue LL, LH, RL, RH;
  DebugLoc dl = N->getDebugLoc();
  GetSplitOp(N->getOperand(2), LL, LH);
  GetSplitOp(N->getOperand(3), RL, RH);

  Lo = DAG.getNode(ISD::SELECT_CC, dl, LL.getValueType(), N->getOperand(0),
                   N->getOperand(1), LL, RL, N->getOperand(4));
  Hi = DAG.getNode(ISD::SELECT_CC, dl, LH.getValueType(), N->getOperand(0),
                   N->getOperand(1), LH, RH, N->getOperand(4));
}

void DAGTypeLegalizer::SplitRes_UNDEF(SDNode *N, SDValue &Lo, SDValue &Hi) {
  MVT LoVT, HiVT;
  DebugLoc dl = N->getDebugLoc();
  GetSplitDestVTs(N->getValueType(0), LoVT, HiVT);
  Lo = DAG.getNode(ISD::UNDEF, dl, LoVT);
  Hi = DAG.getNode(ISD::UNDEF, dl, HiVT);
}
