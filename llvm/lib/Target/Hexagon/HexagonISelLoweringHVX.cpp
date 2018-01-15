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

MVT
HexagonTargetLowering::getVecBoolVT() const {
  return MVT::getVectorVT(MVT::i1, 8*Subtarget.getVectorLength());
}

SDValue
HexagonTargetLowering::buildHvxVectorSingle(ArrayRef<SDValue> Values,
                                            const SDLoc &dl, MVT VecTy,
                                            SelectionDAG &DAG) const {
  unsigned VecLen = Values.size();
  MachineFunction &MF = DAG.getMachineFunction();
  MVT ElemTy = VecTy.getVectorElementType();
  unsigned ElemWidth = ElemTy.getSizeInBits();
  unsigned HwLen = Subtarget.getVectorLength();

  SmallVector<ConstantInt*, 128> Consts(VecLen);
  bool AllConst = getBuildVectorConstInts(Values, VecTy, DAG, Consts);
  if (AllConst) {
    if (llvm::all_of(Consts, [](ConstantInt *CI) { return CI->isZero(); }))
      return getZero(dl, VecTy, DAG);

    ArrayRef<Constant*> Tmp((Constant**)Consts.begin(),
                            (Constant**)Consts.end());
    Constant *CV = ConstantVector::get(Tmp);
    unsigned Align = HwLen;
    SDValue CP = LowerConstantPool(DAG.getConstantPool(CV, VecTy, Align), DAG);
    return DAG.getLoad(VecTy, dl, DAG.getEntryNode(), CP,
                       MachinePointerInfo::getConstantPool(MF), Align);
  }

  unsigned ElemSize = ElemWidth / 8;
  assert(ElemSize*VecLen == HwLen);
  SmallVector<SDValue,32> Words;

  if (VecTy.getVectorElementType() != MVT::i32) {
    assert((ElemSize == 1 || ElemSize == 2) && "Invalid element size");
    unsigned OpsPerWord = (ElemSize == 1) ? 4 : 2;
    MVT PartVT = MVT::getVectorVT(VecTy.getVectorElementType(), OpsPerWord);
    for (unsigned i = 0; i != VecLen; i += OpsPerWord) {
      SDValue W = buildVector32(Values.slice(i, OpsPerWord), dl, PartVT, DAG);
      Words.push_back(DAG.getBitcast(MVT::i32, W));
    }
  } else {
    Words.assign(Values.begin(), Values.end());
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
HexagonTargetLowering::buildHvxVectorPred(ArrayRef<SDValue> Values,
                                          const SDLoc &dl, MVT VecTy,
                                          SelectionDAG &DAG) const {
  // Construct a vector V of bytes, such that a comparison V >u 0 would
  // produce the required vector predicate.
  unsigned VecLen = Values.size();
  unsigned HwLen = Subtarget.getVectorLength();
  assert(VecLen <= HwLen || VecLen == 8*HwLen);
  SmallVector<SDValue,128> Bytes;

  if (VecLen <= HwLen) {
    // In the hardware, each bit of a vector predicate corresponds to a byte
    // of a vector register. Calculate how many bytes does a bit of VecTy
    // correspond to.
    assert(HwLen % VecLen == 0);
    unsigned BitBytes = HwLen / VecLen;
    for (SDValue V : Values) {
      SDValue Ext = !V.isUndef() ? DAG.getZExtOrTrunc(V, dl, MVT::i8)
                                 : DAG.getConstant(0, dl, MVT::i8);
      for (unsigned B = 0; B != BitBytes; ++B)
        Bytes.push_back(Ext);
    }
  } else {
    // There are as many i1 values, as there are bits in a vector register.
    // Divide the values into groups of 8 and check that each group consists
    // of the same value (ignoring undefs).
    for (unsigned I = 0; I != VecLen; I += 8) {
      unsigned B = 0;
      // Find the first non-undef value in this group.
      for (; B != 8; ++B) {
        if (!Values[I+B].isUndef())
          break;
      }
      SDValue F = Values[I+B];
      SDValue Ext = (B < 8) ? DAG.getZExtOrTrunc(F, dl, MVT::i8)
                            : DAG.getConstant(0, dl, MVT::i8);
      Bytes.push_back(Ext);
      // Verify that the rest of values in the group are the same as the
      // first.
      for (; B != 8; ++B)
        assert(Values[I+B].isUndef() || Values[I+B] == F);
    }
  }

  MVT ByteTy = MVT::getVectorVT(MVT::i8, HwLen);
  SDValue ByteVec = buildHvxVectorSingle(Bytes, dl, ByteTy, DAG);
  SDValue Cmp = DAG.getSetCC(dl, VecTy, ByteVec, getZero(dl, ByteTy, DAG),
                             ISD::SETUGT);
  return Cmp;
}

SDValue
HexagonTargetLowering::LowerHvxBuildVector(SDValue Op, SelectionDAG &DAG)
      const {
  const SDLoc &dl(Op);
  MVT VecTy = ty(Op);

  unsigned Size = Op.getNumOperands();
  SmallVector<SDValue,128> Ops;
  for (unsigned i = 0; i != Size; ++i)
    Ops.push_back(Op.getOperand(i));

  if (VecTy.getVectorElementType() == MVT::i1)
    return buildHvxVectorPred(Ops, dl, VecTy, DAG);

  if (VecTy.getSizeInBits() == 16*Subtarget.getVectorLength()) {
    ArrayRef<SDValue> A(Ops);
    MVT SingleTy = typeSplit(VecTy).first;
    SDValue V0 = buildHvxVectorSingle(A.take_front(Size/2), dl, SingleTy, DAG);
    SDValue V1 = buildHvxVectorSingle(A.drop_front(Size/2), dl, SingleTy, DAG);
    return DAG.getNode(ISD::CONCAT_VECTORS, dl, VecTy, V0, V1);
  }

  return buildHvxVectorSingle(Ops, dl, VecTy, DAG);
}

SDValue
HexagonTargetLowering::LowerHvxExtractElement(SDValue Op, SelectionDAG &DAG)
      const {
  // Change the type of the extracted element to i32.
  SDValue VecV = Op.getOperand(0);
  MVT ElemTy = ty(VecV).getVectorElementType();
  unsigned ElemWidth = ElemTy.getSizeInBits();
  assert(ElemWidth >= 8 && ElemWidth <= 32);
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
  assert(ElemWidth >= 8 && ElemWidth <= 32);
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
                             ValV, SubIdx, dl, ElemTy, DAG);

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
  assert(ResTy.isVector());
  const SDLoc &dl(Op);
  SmallVector<int,256> ShuffMask;

  MVT ElemTy = ResTy.getVectorElementType();
  unsigned VecLen = ResTy.getVectorNumElements();
  SDValue Vs = Op.getOperand(0);
  SDValue Vt = Op.getOperand(1);

  switch (ElemTy.SimpleTy) {
    case MVT::i8:
    case MVT::i16: { // V6_vmpyih
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
      SDValue BS = getByteShuffle(dl, P.first, P.second, ShuffMask, DAG);
      return DAG.getBitcast(ResTy, BS);
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

SDValue
HexagonTargetLowering::LowerHvxMulh(SDValue Op, SelectionDAG &DAG) const {
  MVT ResTy = ty(Op);
  assert(ResTy.isVector());
  const SDLoc &dl(Op);
  SmallVector<int,256> ShuffMask;

  MVT ElemTy = ResTy.getVectorElementType();
  unsigned VecLen = ResTy.getVectorNumElements();
  SDValue Vs = Op.getOperand(0);
  SDValue Vt = Op.getOperand(1);
  bool IsSigned = Op.getOpcode() == ISD::MULHS;

  if (ElemTy == MVT::i8 || ElemTy == MVT::i16) {
    // For i8 vectors Vs = (a0, a1, ...), Vt = (b0, b1, ...),
    // V6_vmpybv Vs, Vt produces a pair of i16 vectors Hi:Lo,
    // where Lo = (a0*b0, a2*b2, ...), Hi = (a1*b1, a3*b3, ...).
    // For i16, use V6_vmpyhv, which behaves in an analogous way to
    // V6_vmpybv: results Lo and Hi are products of even/odd elements
    // respectively.
    MVT ExtTy = typeExtElem(ResTy, 2);
    unsigned MpyOpc = ElemTy == MVT::i8
        ? (IsSigned ? Hexagon::V6_vmpybv : Hexagon::V6_vmpyubv)
        : (IsSigned ? Hexagon::V6_vmpyhv : Hexagon::V6_vmpyuhv);
    SDValue M = getNode(MpyOpc, dl, ExtTy, {Vs, Vt}, DAG);

    // Discard low halves of the resulting values, collect the high halves.
    for (unsigned I = 0; I < VecLen; I += 2) {
      ShuffMask.push_back(I+1);         // Pick even element.
      ShuffMask.push_back(I+VecLen+1);  // Pick odd element.
    }
    VectorPair P = opSplit(opCastElem(M, ElemTy, DAG), dl, DAG);
    SDValue BS = getByteShuffle(dl, P.first, P.second, ShuffMask, DAG);
    return DAG.getBitcast(ResTy, BS);
  }

  assert(ElemTy == MVT::i32);
  SDValue S16 = DAG.getConstant(16, dl, MVT::i32);

  if (IsSigned) {
    // mulhs(Vs,Vt) =
    //   = [(Hi(Vs)*2^16 + Lo(Vs)) *s (Hi(Vt)*2^16 + Lo(Vt))] >> 32
    //   = [Hi(Vs)*2^16 *s Hi(Vt)*2^16 + Hi(Vs) *su Lo(Vt)*2^16
    //      + Lo(Vs) *us (Hi(Vt)*2^16 + Lo(Vt))] >> 32
    //   = [Hi(Vs) *s Hi(Vt)*2^32 + Hi(Vs) *su Lo(Vt)*2^16
    //      + Lo(Vs) *us Vt] >> 32
    // The low half of Lo(Vs)*Lo(Vt) will be discarded (it's not added to
    // anything, so it cannot produce any carry over to higher bits),
    // so everything in [] can be shifted by 16 without loss of precision.
    //   = [Hi(Vs) *s Hi(Vt)*2^16 + Hi(Vs)*su Lo(Vt) + Lo(Vs)*Vt >> 16] >> 16
    //   = [Hi(Vs) *s Hi(Vt)*2^16 + Hi(Vs)*su Lo(Vt) + V6_vmpyewuh(Vs,Vt)] >> 16
    // Denote Hi(Vs) = Vs':
    //   = [Vs'*s Hi(Vt)*2^16 + Vs' *su Lo(Vt) + V6_vmpyewuh(Vt,Vs)] >> 16
    //   = Vs'*s Hi(Vt) + (V6_vmpyiewuh(Vs',Vt) + V6_vmpyewuh(Vt,Vs)) >> 16
    SDValue T0 = getNode(Hexagon::V6_vmpyewuh, dl, ResTy, {Vt, Vs}, DAG);
    // Get Vs':
    SDValue S0 = getNode(Hexagon::V6_vasrw, dl, ResTy, {Vs, S16}, DAG);
    SDValue T1 = getNode(Hexagon::V6_vmpyiewuh_acc, dl, ResTy,
                         {T0, S0, Vt}, DAG);
    // Shift by 16:
    SDValue S2 = getNode(Hexagon::V6_vasrw, dl, ResTy, {T1, S16}, DAG);
    // Get Vs'*Hi(Vt):
    SDValue T2 = getNode(Hexagon::V6_vmpyiowh, dl, ResTy, {S0, Vt}, DAG);
    // Add:
    SDValue T3 = DAG.getNode(ISD::ADD, dl, ResTy, {S2, T2});
    return T3;
  }

  // Unsigned mulhw. (Would expansion using signed mulhw be better?)

  auto LoVec = [&DAG,ResTy,dl] (SDValue Pair) {
    return DAG.getTargetExtractSubreg(Hexagon::vsub_lo, dl, ResTy, Pair);
  };
  auto HiVec = [&DAG,ResTy,dl] (SDValue Pair) {
    return DAG.getTargetExtractSubreg(Hexagon::vsub_hi, dl, ResTy, Pair);
  };

  MVT PairTy = typeJoin({ResTy, ResTy});
  SDValue P = getNode(Hexagon::V6_lvsplatw, dl, ResTy,
                      {DAG.getConstant(0x02020202, dl, MVT::i32)}, DAG);
  // Multiply-unsigned halfwords:
  //   LoVec = Vs.uh[2i] * Vt.uh[2i],
  //   HiVec = Vs.uh[2i+1] * Vt.uh[2i+1]
  SDValue T0 = getNode(Hexagon::V6_vmpyuhv, dl, PairTy, {Vs, Vt}, DAG);
  // The low halves in the LoVec of the pair can be discarded. They are
  // not added to anything (in the full-precision product), so they cannot
  // produce a carry into the higher bits.
  SDValue T1 = getNode(Hexagon::V6_vlsrw, dl, ResTy, {LoVec(T0), S16}, DAG);
  // Swap low and high halves in Vt, and do the halfword multiplication
  // to get products Vs.uh[2i] * Vt.uh[2i+1] and Vs.uh[2i+1] * Vt.uh[2i].
  SDValue D0 = getNode(Hexagon::V6_vdelta, dl, ResTy, {Vt, P}, DAG);
  SDValue T2 = getNode(Hexagon::V6_vmpyuhv, dl, PairTy, {Vs, D0}, DAG);
  // T2 has mixed products of halfwords: Lo(Vt)*Hi(Vs) and Hi(Vt)*Lo(Vs).
  // These products are words, but cannot be added directly because the
  // sums could overflow. Add these products, by halfwords, where each sum
  // of a pair of halfwords gives a word.
  SDValue T3 = getNode(Hexagon::V6_vadduhw, dl, PairTy,
                       {LoVec(T2), HiVec(T2)}, DAG);
  // Add the high halfwords from the products of the low halfwords.
  SDValue T4 = DAG.getNode(ISD::ADD, dl, ResTy, {T1, LoVec(T3)});
  SDValue T5 = getNode(Hexagon::V6_vlsrw, dl, ResTy, {T4, S16}, DAG);
  SDValue T6 = DAG.getNode(ISD::ADD, dl, ResTy, {HiVec(T0), HiVec(T3)});
  SDValue T7 = DAG.getNode(ISD::ADD, dl, ResTy, {T5, T6});
  return T7;
}

SDValue
HexagonTargetLowering::LowerHvxSetCC(SDValue Op, SelectionDAG &DAG) const {
  MVT VecTy = ty(Op.getOperand(0));
  assert(VecTy == ty(Op.getOperand(1)));

  SDValue Cmp = Op.getOperand(2);
  ISD::CondCode CC = cast<CondCodeSDNode>(Cmp)->get();
  bool Negate = false, Swap = false;

  // HVX has instructions for SETEQ, SETGT, SETUGT. The other comparisons
  // can be arranged as operand-swapped/negated versions of these. Since
  // the generated code will have the original CC expressed as
  //   (negate (swap-op NewCmp)),
  // the condition code for the NewCmp should be calculated from the original
  // CC by applying these operations in the reverse order.
  //
  // This could also be done through setCondCodeAction, but for negation it
  // uses a xor with a vector of -1s, which it obtains from BUILD_VECTOR.
  // That is far too expensive for what can be done with a single instruction.

  switch (CC) {
    case ISD::SETNE:    // !eq
    case ISD::SETLE:    // !gt
    case ISD::SETGE:    // !lt
    case ISD::SETULE:   // !ugt
    case ISD::SETUGE:   // !ult
      CC = ISD::getSetCCInverse(CC, true);
      Negate = true;
      break;
    default:
      break;
  }

  switch (CC) {
    case ISD::SETLT:    // swap gt
    case ISD::SETULT:   // swap ugt
      CC = ISD::getSetCCSwappedOperands(CC);
      Swap = true;
      break;
    default:
      break;
  }

  assert(CC == ISD::SETEQ || CC == ISD::SETGT || CC == ISD::SETUGT);

  MVT ElemTy = VecTy.getVectorElementType();
  unsigned ElemWidth = ElemTy.getSizeInBits();
  assert(isPowerOf2_32(ElemWidth));

  auto getIdx = [] (unsigned Code) {
    static const unsigned Idx[] = { ISD::SETEQ, ISD::SETGT, ISD::SETUGT };
    for (unsigned I = 0, E = array_lengthof(Idx); I != E; ++I)
      if (Code == Idx[I])
        return I;
    llvm_unreachable("Unhandled CondCode");
  };

  static unsigned OpcTable[3][3] = {
    //           SETEQ             SETGT,            SETUGT
    /* Byte */ { Hexagon::V6_veqb, Hexagon::V6_vgtb, Hexagon::V6_vgtub },
    /* Half */ { Hexagon::V6_veqh, Hexagon::V6_vgth, Hexagon::V6_vgtuh },
    /* Word */ { Hexagon::V6_veqw, Hexagon::V6_vgtw, Hexagon::V6_vgtuw }
  };

  unsigned CmpOpc = OpcTable[Log2_32(ElemWidth)-3][getIdx(CC)];

  MVT ResTy = ty(Op);
  const SDLoc &dl(Op);
  SDValue OpL = Swap ? Op.getOperand(1) : Op.getOperand(0);
  SDValue OpR = Swap ? Op.getOperand(0) : Op.getOperand(1);
  SDValue CmpV = getNode(CmpOpc, dl, ResTy, {OpL, OpR}, DAG);
  return Negate ? getNode(Hexagon::V6_pred_not, dl, ResTy, {CmpV}, DAG)
                : CmpV;
}

SDValue
HexagonTargetLowering::LowerHvxExtend(SDValue Op, SelectionDAG &DAG) const {
  // Sign- and zero-extends are legal.
  assert(Op.getOpcode() == ISD::ANY_EXTEND_VECTOR_INREG);
  return DAG.getZeroExtendVectorInReg(Op.getOperand(0), SDLoc(Op), ty(Op));
}
