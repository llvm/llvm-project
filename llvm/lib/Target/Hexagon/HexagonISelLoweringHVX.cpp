//===-- HexagonISelLoweringHVX.cpp --- Lowering HVX operations ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HexagonISelLowering.h"
#include "HexagonRegisterInfo.h"
#include "HexagonSubtarget.h"

using namespace llvm;

SDValue
HexagonTargetLowering::getInt(unsigned IntId, MVT ResTy, ArrayRef<SDValue> Ops,
                              const SDLoc &dl, SelectionDAG &DAG) const {
  SmallVector<SDValue,4> IntOps;
  IntOps.push_back(DAG.getConstant(IntId, dl, MVT::i32));
  for (const SDValue &Op : Ops)
    IntOps.push_back(Op);
  return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, ResTy, IntOps);
}

MVT
HexagonTargetLowering::typeJoin(const TypePair &Tys) const {
  assert(Tys.first.getVectorElementType() == Tys.second.getVectorElementType());

  MVT ElemTy = Tys.first.getVectorElementType();
  return MVT::getVectorVT(ElemTy, Tys.first.getVectorNumElements() +
                                  Tys.second.getVectorNumElements());
}

HexagonTargetLowering::TypePair
HexagonTargetLowering::typeSplit(MVT VecTy) const {
  assert(VecTy.isVector());
  unsigned NumElem = VecTy.getVectorNumElements();
  assert((NumElem % 2) == 0 && "Expecting even-sized vector type");
  MVT HalfTy = MVT::getVectorVT(VecTy.getVectorElementType(), NumElem/2);
  return { HalfTy, HalfTy };
}

MVT
HexagonTargetLowering::typeExtElem(MVT VecTy, unsigned Factor) const {
  MVT ElemTy = VecTy.getVectorElementType();
  MVT NewElemTy = MVT::getIntegerVT(ElemTy.getSizeInBits() * Factor);
  return MVT::getVectorVT(NewElemTy, VecTy.getVectorNumElements());
}

MVT
HexagonTargetLowering::typeTruncElem(MVT VecTy, unsigned Factor) const {
  MVT ElemTy = VecTy.getVectorElementType();
  MVT NewElemTy = MVT::getIntegerVT(ElemTy.getSizeInBits() / Factor);
  return MVT::getVectorVT(NewElemTy, VecTy.getVectorNumElements());
}

SDValue
HexagonTargetLowering::opCastElem(SDValue Vec, MVT ElemTy,
                                  SelectionDAG &DAG) const {
  if (ty(Vec).getVectorElementType() == ElemTy)
    return Vec;
  MVT CastTy = tyVector(Vec.getValueType().getSimpleVT(), ElemTy);
  return DAG.getBitcast(CastTy, Vec);
}

SDValue
HexagonTargetLowering::opJoin(const VectorPair &Ops, const SDLoc &dl,
                              SelectionDAG &DAG) const {
  return DAG.getNode(ISD::CONCAT_VECTORS, dl, typeJoin(ty(Ops)),
                     Ops.second, Ops.first);
}

HexagonTargetLowering::VectorPair
HexagonTargetLowering::opSplit(SDValue Vec, const SDLoc &dl,
                               SelectionDAG &DAG) const {
  TypePair Tys = typeSplit(ty(Vec));
  return DAG.SplitVector(Vec, dl, Tys.first, Tys.second);
}

SDValue
HexagonTargetLowering::convertToByteIndex(SDValue ElemIdx, MVT ElemTy,
                                          SelectionDAG &DAG) const {
  if (ElemIdx.getValueType().getSimpleVT() != MVT::i32)
    ElemIdx = DAG.getBitcast(MVT::i32, ElemIdx);

  unsigned ElemWidth = ElemTy.getSizeInBits();
  if (ElemWidth == 8)
    return ElemIdx;

  unsigned L = Log2_32(ElemWidth/8);
  const SDLoc &dl(ElemIdx);
  return DAG.getNode(ISD::SHL, dl, MVT::i32,
                     {ElemIdx, DAG.getConstant(L, dl, MVT::i32)});
}

SDValue
HexagonTargetLowering::getIndexInWord32(SDValue Idx, MVT ElemTy,
                                        SelectionDAG &DAG) const {
  unsigned ElemWidth = ElemTy.getSizeInBits();
  assert(ElemWidth >= 8 && ElemWidth <= 32);
  if (ElemWidth == 32)
    return Idx;

  if (ty(Idx) != MVT::i32)
    Idx = DAG.getBitcast(MVT::i32, Idx);
  const SDLoc &dl(Idx);
  SDValue Mask = DAG.getConstant(32/ElemWidth - 1, dl, MVT::i32);
  SDValue SubIdx = DAG.getNode(ISD::AND, dl, MVT::i32, {Idx, Mask});
  return SubIdx;
}

SDValue
HexagonTargetLowering::getByteShuffle(const SDLoc &dl, SDValue Op0,
                                      SDValue Op1, ArrayRef<int> Mask,
                                      SelectionDAG &DAG) const {
  MVT OpTy = ty(Op0);
  assert(OpTy == ty(Op1));

  MVT ElemTy = OpTy.getVectorElementType();
  if (ElemTy == MVT::i8)
    return DAG.getVectorShuffle(OpTy, dl, Op0, Op1, Mask);
  assert(ElemTy.getSizeInBits() >= 8);

  MVT ResTy = tyVector(OpTy, MVT::i8);
  unsigned ElemSize = ElemTy.getSizeInBits() / 8;

  SmallVector<int,128> ByteMask;
  for (int M : Mask) {
    if (M < 0) {
      for (unsigned I = 0; I != ElemSize; ++I)
        ByteMask.push_back(-1);
    } else {
      int NewM = M*ElemSize;
      for (unsigned I = 0; I != ElemSize; ++I)
        ByteMask.push_back(NewM+I);
    }
  }
  assert(ResTy.getVectorNumElements() == ByteMask.size());
  return DAG.getVectorShuffle(ResTy, dl, opCastElem(Op0, MVT::i8, DAG),
                              opCastElem(Op1, MVT::i8, DAG), ByteMask);
}

SDValue
HexagonTargetLowering::LowerHvxBuildVector(SDValue Op, SelectionDAG &DAG)
      const {
  const SDLoc &dl(Op);
  BuildVectorSDNode *BN = cast<BuildVectorSDNode>(Op.getNode());
  bool IsConst = BN->isConstant();
  MachineFunction &MF = DAG.getMachineFunction();
  MVT VecTy = ty(Op);

  if (IsConst) {
    SmallVector<Constant*, 128> Elems;
    for (SDValue V : BN->op_values()) {
      if (auto *C = dyn_cast<ConstantSDNode>(V.getNode()))
        Elems.push_back(const_cast<ConstantInt*>(C->getConstantIntValue()));
    }
    Constant *CV = ConstantVector::get(Elems);
    unsigned Align = VecTy.getSizeInBits() / 8;
    SDValue CP = LowerConstantPool(DAG.getConstantPool(CV, VecTy, Align), DAG);
    return DAG.getLoad(VecTy, dl, DAG.getEntryNode(), CP,
                       MachinePointerInfo::getConstantPool(MF), Align);
  }

  unsigned NumOps = Op.getNumOperands();
  unsigned HwLen = Subtarget.getVectorLength();
  unsigned ElemSize = VecTy.getVectorElementType().getSizeInBits() / 8;
  assert(ElemSize*NumOps == HwLen);

  SmallVector<SDValue,32> Words;
  SmallVector<SDValue,32> Ops;
  for (unsigned i = 0; i != NumOps; ++i)
    Ops.push_back(Op.getOperand(i));

  if (VecTy.getVectorElementType() != MVT::i32) {
    assert(ElemSize < 4 && "vNi64 should have been promoted to vNi32");
    assert((ElemSize == 1 || ElemSize == 2) && "Invalid element size");
    unsigned OpsPerWord = (ElemSize == 1) ? 4 : 2;
    MVT PartVT = MVT::getVectorVT(VecTy.getVectorElementType(), OpsPerWord);
    for (unsigned i = 0; i != NumOps; i += OpsPerWord) {
      SDValue W = buildVector32({&Ops[i], OpsPerWord}, dl, PartVT, DAG);
      Words.push_back(DAG.getBitcast(MVT::i32, W));
    }
  } else {
    Words.assign(Ops.begin(), Ops.end());
  }

  // Construct two halves in parallel, then or them together.
  assert(4*Words.size() == Subtarget.getVectorLength());
  SDValue HalfV0 = getNode(Hexagon::V6_vd0, dl, VecTy, {}, DAG);
  SDValue HalfV1 = getNode(Hexagon::V6_vd0, dl, VecTy, {}, DAG);
  SDValue S = DAG.getConstant(4, dl, MVT::i32);
  unsigned NumWords = Words.size();
  for (unsigned i = 0; i != NumWords/2; ++i) {
    SDValue N = DAG.getNode(HexagonISD::VINSERTW0, dl, VecTy,
                            {HalfV0, Words[i]});
    SDValue M = DAG.getNode(HexagonISD::VINSERTW0, dl, VecTy,
                            {HalfV1, Words[i+NumWords/2]});
    HalfV0 = DAG.getNode(HexagonISD::VROR, dl, VecTy, {N, S});
    HalfV1 = DAG.getNode(HexagonISD::VROR, dl, VecTy, {M, S});
  }

  HalfV0 = DAG.getNode(HexagonISD::VROR, dl, VecTy,
                       {HalfV0, DAG.getConstant(HwLen/2, dl, MVT::i32)});
  SDValue DstV = DAG.getNode(ISD::OR, dl, VecTy, {HalfV0, HalfV1});
  return DstV;
}

SDValue
HexagonTargetLowering::LowerHvxExtractElement(SDValue Op, SelectionDAG &DAG)
      const {
  // Change the type of the extracted element to i32.
  SDValue VecV = Op.getOperand(0);
  MVT ElemTy = ty(VecV).getVectorElementType();
  unsigned ElemWidth = ElemTy.getSizeInBits();
  assert(ElemWidth >= 8 && ElemWidth <= 32); // TODO i64
  (void)ElemWidth;

  const SDLoc &dl(Op);
  SDValue IdxV = Op.getOperand(1);
  if (ty(IdxV) != MVT::i32)
    IdxV = DAG.getBitcast(MVT::i32, IdxV);

  SDValue ByteIdx = convertToByteIndex(IdxV, ElemTy, DAG);
  SDValue ExWord = DAG.getNode(HexagonISD::VEXTRACTW, dl, MVT::i32,
                               {VecV, ByteIdx});
  if (ElemTy == MVT::i32)
    return ExWord;

  // Have an extracted word, need to extract the smaller element out of it.
  // 1. Extract the bits of (the original) IdxV that correspond to the index
  //    of the desired element in the 32-bit word.
  SDValue SubIdx = getIndexInWord32(IdxV, ElemTy, DAG);
  // 2. Extract the element from the word.
  SDValue ExVec = DAG.getBitcast(tyVector(ty(ExWord), ElemTy), ExWord);
  return extractVector(ExVec, SubIdx, dl, ElemTy, MVT::i32, DAG);
}

SDValue
HexagonTargetLowering::LowerHvxInsertElement(SDValue Op, SelectionDAG &DAG)
      const {
  const SDLoc &dl(Op);
  SDValue VecV = Op.getOperand(0);
  SDValue ValV = Op.getOperand(1);
  SDValue IdxV = Op.getOperand(2);
  MVT ElemTy = ty(VecV).getVectorElementType();
  unsigned ElemWidth = ElemTy.getSizeInBits();
  assert(ElemWidth >= 8 && ElemWidth <= 32); // TODO i64
  (void)ElemWidth;

  auto InsertWord = [&DAG,&dl,this] (SDValue VecV, SDValue ValV,
                                     SDValue ByteIdxV) {
    MVT VecTy = ty(VecV);
    unsigned HwLen = Subtarget.getVectorLength();
    SDValue MaskV = DAG.getNode(ISD::AND, dl, MVT::i32,
                                {ByteIdxV, DAG.getConstant(-4, dl, MVT::i32)});
    SDValue RotV = DAG.getNode(HexagonISD::VROR, dl, VecTy, {VecV, MaskV});
    SDValue InsV = DAG.getNode(HexagonISD::VINSERTW0, dl, VecTy, {RotV, ValV});
    SDValue SubV = DAG.getNode(ISD::SUB, dl, MVT::i32,
                               {DAG.getConstant(HwLen/4, dl, MVT::i32), MaskV});
    SDValue TorV = DAG.getNode(HexagonISD::VROR, dl, VecTy, {InsV, SubV});
    return TorV;
  };

  SDValue ByteIdx = convertToByteIndex(IdxV, ElemTy, DAG);
  if (ElemTy == MVT::i32)
    return InsertWord(VecV, ValV, ByteIdx);

  // If this is not inserting a 32-bit word, convert it into such a thing.
  // 1. Extract the existing word from the target vector.
  SDValue WordIdx = DAG.getNode(ISD::SRL, dl, MVT::i32,
                                {ByteIdx, DAG.getConstant(2, dl, MVT::i32)});
  SDValue Ex0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, MVT::i32,
                            {opCastElem(VecV, MVT::i32, DAG), WordIdx});
  SDValue Ext = LowerHvxExtractElement(Ex0, DAG);

  // 2. Treating the extracted word as a 32-bit vector, insert the given
  //    value into it.
  SDValue SubIdx = getIndexInWord32(IdxV, ElemTy, DAG);
  MVT SubVecTy = tyVector(ty(Ext), ElemTy);
  SDValue Ins = insertVector(DAG.getBitcast(SubVecTy, Ext),
                             ValV, SubIdx, dl, SubVecTy, DAG);

  // 3. Insert the 32-bit word back into the original vector.
  return InsertWord(VecV, Ins, ByteIdx);
}

SDValue
HexagonTargetLowering::LowerHvxExtractSubvector(SDValue Op, SelectionDAG &DAG)
      const {
  SDValue SrcV = Op.getOperand(0);
  MVT SrcTy = ty(SrcV);
  unsigned SrcElems = SrcTy.getVectorNumElements();
  SDValue IdxV = Op.getOperand(1);
  unsigned Idx = cast<ConstantSDNode>(IdxV.getNode())->getZExtValue();
  MVT DstTy = ty(Op);
  assert(Idx == 0 || DstTy.getVectorNumElements() % Idx == 0);
  const SDLoc &dl(Op);
  if (Idx == 0)
    return DAG.getTargetExtractSubreg(Hexagon::vsub_lo, dl, DstTy, SrcV);
  if (Idx == SrcElems/2)
    return DAG.getTargetExtractSubreg(Hexagon::vsub_hi, dl, DstTy, SrcV);
  return SDValue();
}

SDValue
HexagonTargetLowering::LowerHvxInsertSubvector(SDValue Op, SelectionDAG &DAG)
      const {
  // Idx may be variable.
  SDValue IdxV = Op.getOperand(2);
  auto *IdxN = dyn_cast<ConstantSDNode>(IdxV.getNode());
  if (!IdxN)
    return SDValue();
  unsigned Idx = IdxN->getZExtValue();

  SDValue DstV = Op.getOperand(0);
  SDValue SrcV = Op.getOperand(1);
  MVT DstTy = ty(DstV);
  MVT SrcTy = ty(SrcV);
  unsigned DstElems = DstTy.getVectorNumElements();
  unsigned SrcElems = SrcTy.getVectorNumElements();
  if (2*SrcElems != DstElems)
    return SDValue();

  const SDLoc &dl(Op);
  if (Idx == 0)
    return DAG.getTargetInsertSubreg(Hexagon::vsub_lo, dl, DstTy, DstV, SrcV);
  if (Idx == SrcElems)
    return DAG.getTargetInsertSubreg(Hexagon::vsub_hi, dl, DstTy, DstV, SrcV);
  return SDValue();
}

SDValue
HexagonTargetLowering::LowerHvxMul(SDValue Op, SelectionDAG &DAG) const {
  MVT ResTy = ty(Op);
  if (!ResTy.isVector())
    return SDValue();
  const SDLoc &dl(Op);
  SmallVector<int,256> ShuffMask;

  MVT ElemTy = ResTy.getVectorElementType();
  unsigned VecLen = ResTy.getVectorNumElements();
  SDValue Vs = Op.getOperand(0);
  SDValue Vt = Op.getOperand(1);

  switch (ElemTy.SimpleTy) {
    case MVT::i8:
    case MVT::i16: {
      // For i8 vectors Vs = (a0, a1, ...), Vt = (b0, b1, ...),
      // V6_vmpybv Vs, Vt produces a pair of i16 vectors Hi:Lo,
      // where Lo = (a0*b0, a2*b2, ...), Hi = (a1*b1, a3*b3, ...).
      // For i16, use V6_vmpyhv, which behaves in an analogous way to
      // V6_vmpybv: results Lo and Hi are products of even/odd elements
      // respectively.
      MVT ExtTy = typeExtElem(ResTy, 2);
      unsigned MpyOpc = ElemTy == MVT::i8 ? Hexagon::V6_vmpybv
                                          : Hexagon::V6_vmpyhv;
      SDValue M = getNode(MpyOpc, dl, ExtTy, {Vs, Vt}, DAG);

      // Discard high halves of the resulting values, collect the low halves.
      for (unsigned I = 0; I < VecLen; I += 2) {
        ShuffMask.push_back(I);         // Pick even element.
        ShuffMask.push_back(I+VecLen);  // Pick odd element.
      }
      VectorPair P = opSplit(opCastElem(M, ElemTy, DAG), dl, DAG);
      return getByteShuffle(dl, P.first, P.second, ShuffMask, DAG);
    }
    case MVT::i32: {
      // Use the following sequence for signed word multiply:
      // T0 = V6_vmpyiowh Vs, Vt
      // T1 = V6_vaslw T0, 16
      // T2 = V6_vmpyiewuh_acc T1, Vs, Vt
      SDValue S16 = DAG.getConstant(16, dl, MVT::i32);
      SDValue T0 = getNode(Hexagon::V6_vmpyiowh, dl, ResTy, {Vs, Vt}, DAG);
      SDValue T1 = getNode(Hexagon::V6_vaslw, dl, ResTy, {T0, S16}, DAG);
      SDValue T2 = getNode(Hexagon::V6_vmpyiewuh_acc, dl, ResTy,
                           {T1, Vs, Vt}, DAG);
      return T2;
    }
    default:
      break;
  }
  return SDValue();
}
