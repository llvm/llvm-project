//===-- HexagonISelLoweringHLX.cpp --- Lowering HLX operations ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HexagonISelLowering.h"
#include "HexagonRegisterInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/IR/IntrinsicsHexagon.h"

using namespace llvm;

// 2048B HLX types
static const MVT AllTypes2048B[] = {MVT::v2048i8, MVT::v1024i16, MVT::v512i32,
                                    MVT::v4096i8, MVT::v2048i16, MVT::v1024i32};
static const MVT LegalTypes2048B[] = {MVT::v1024i16, MVT::v2048i16,
                                      MVT::v1024i32};
static const MVT LegalPredTypes2048B[] = {MVT::v1024i1, MVT::v2048i1};

void HexagonTargetLowering::initializeHLXLowering() {
  ArrayRef<MVT> AllTypes = AllTypes2048B;
  ArrayRef<MVT> LegalTypes = LegalTypes2048B;
  ArrayRef<MVT> LegalPredTypes = LegalPredTypes2048B;
  MVT HalfWordPairTy = MVT::v2048i16;

  // TODO: Remove i32 types for single HLX register
  // ISA only supports halfword, byte operations on single register
  addRegisterClass(MVT::v2048i8, &Hexagon::HlxLRRegClass);
  addRegisterClass(MVT::v1024i16, &Hexagon::HlxLRRegClass);
  addRegisterClass(MVT::v512i32, &Hexagon::HlxLRRegClass);
  addRegisterClass(MVT::v1024i1, &Hexagon::HlxLRRegClass);
  addRegisterClass(MVT::v4096i8, &Hexagon::HlxLLRRegClass);
  addRegisterClass(MVT::v2048i16, &Hexagon::HlxLLRRegClass);
  addRegisterClass(MVT::v1024i32, &Hexagon::HlxLLRRegClass);
  addRegisterClass(MVT::v2048i1, &Hexagon::HlxLLRRegClass);

  // Route element-sensitive operations unsupported by a type to
  // LowerHlxOperation.
  static const unsigned UnsupportedOnByteAndSingleWord[] = {
      ISD::ADD,       ISD::SUB,      ISD::UADDSAT,      ISD::SADDSAT,
      ISD::USUBSAT,   ISD::SSUBSAT,  ISD::MUL,          ISD::SHL,
      ISD::SRL,       ISD::SRA,      ISD::SPLAT_VECTOR, ISD::VSELECT,
      ISD::SMIN,      ISD::SMAX,     ISD::UMIN,         ISD::UMAX,
      ISD::ABS,       ISD::ABDS,     ISD::ABDU,         ISD::AVGFLOORS,
      ISD::AVGFLOORU, ISD::AVGCEILS, ISD::AVGCEILU};
  SmallVector<MVT, 8> UnsupportedTypes;
  for (MVT T : AllTypes)
    if (!llvm::is_contained(LegalTypes, T))
      UnsupportedTypes.push_back(T);
  setOperationAction(UnsupportedOnByteAndSingleWord, UnsupportedTypes, Custom);

  // Catch VSELECT before type legalization widens an unsupported predicate.
  for (MVT T : UnsupportedTypes) {
    MVT PredTy = MVT::getVectorVT(MVT::i1, T.getVectorNumElements());
    if (!llvm::is_contained(LegalPredTypes, PredTy))
      setOperationAction(ISD::VSELECT, PredTy, Custom);
  }

  // Reject unsupported halfword-pair operations in LowerHlxOperation.
  for (unsigned Opc : {ISD::MUL, ISD::ABDS, ISD::ABDU, ISD::AVGFLOORS,
                       ISD::AVGFLOORU, ISD::AVGCEILS, ISD::AVGCEILU})
    setOperationAction(Opc, HalfWordPairTy, Custom);

  for (MVT T : UnsupportedTypes) {
    setCondCodeAction(ISD::SETGT, T, Custom);
    setCondCodeAction(ISD::SETUGT, T, Custom);
    setCondCodeAction(ISD::SETLE, T, Custom);
    setCondCodeAction(ISD::SETGE, T, Custom);
    setCondCodeAction(ISD::SETLT, T, Custom);
    setCondCodeAction(ISD::SETULE, T, Custom);
    setCondCodeAction(ISD::SETUGE, T, Custom);
    setCondCodeAction(ISD::SETULT, T, Custom);
    setCondCodeAction(ISD::SETEQ, T, Custom);
    setCondCodeAction(ISD::SETNE, T, Custom);
  }

  for (MVT T : LegalTypes) {
    setOperationAction(ISD::ADD, T, Legal);
    setOperationAction(ISD::SUB, T, Legal);
    setOperationAction(ISD::UADDSAT, T, Legal);
    setOperationAction(ISD::SADDSAT, T, Legal);
    setOperationAction(ISD::USUBSAT, T, Legal);
    setOperationAction(ISD::SSUBSAT, T, Legal);
  }

  setOperationAction(ISD::MUL, MVT::v1024i16, Legal);
  setOperationAction(ISD::MUL, MVT::v1024i32, Legal);

  setOperationAction(ISD::MULHU, MVT::v1024i16, Legal);
  setOperationAction(ISD::MULHS, MVT::v1024i16, Legal);

  for (MVT T : AllTypes) {
    setOperationAction(ISD::AND, T, Legal);
    setOperationAction(ISD::OR, T, Legal);
    setOperationAction(ISD::XOR, T, Legal);
  }

  setOperationAction(ISD::SRA, MVT::v1024i16, Legal);
  setOperationAction(ISD::SRA, MVT::v1024i32, Legal);
  setOperationAction(ISD::SRL, MVT::v1024i16, Legal);
  setOperationAction(ISD::SRL, MVT::v1024i32, Legal);
  setOperationAction(ISD::SHL, MVT::v1024i16, Legal);
  setOperationAction(ISD::SHL, MVT::v1024i32, Legal);

  for (MVT T : LegalTypes) {
    // GT comparisons are legal
    setCondCodeAction(ISD::SETGT, T, Legal);
    setCondCodeAction(ISD::SETUGT, T, Legal);
    // Expand all non-GT comparisons
    setCondCodeAction(ISD::SETLE, T, Expand);
    setCondCodeAction(ISD::SETGE, T, Expand);
    setCondCodeAction(ISD::SETLT, T, Expand);
    setCondCodeAction(ISD::SETULE, T, Expand);
    setCondCodeAction(ISD::SETUGE, T, Expand);
    setCondCodeAction(ISD::SETULT, T, Expand);
    // Expand equality comparisons
    setCondCodeAction(ISD::SETEQ, T, Expand);
    setCondCodeAction(ISD::SETNE, T, Expand);
  }

  // Add SETCC operation for predicate result types
  for (MVT T : LegalPredTypes)
    setOperationAction(ISD::SETCC, T, Legal);

  for (MVT T : LegalTypes) {
    setOperationAction(ISD::SMIN, T, Legal);
    setOperationAction(ISD::SMAX, T, Legal);
    setOperationAction(ISD::UMIN, T, Legal);
    setOperationAction(ISD::UMAX, T, Legal);
    setOperationAction(ISD::ABS, T, Legal);
  }

  // absdiff and average lack halfword-pair instructions.
  MVT HalfWordSingleTy = MVT::v1024i16;
  MVT WordPairTy = MVT::v1024i32;
  for (MVT T : {HalfWordSingleTy, WordPairTy}) {
    setOperationAction(ISD::ABDS, T, Legal);
    setOperationAction(ISD::ABDU, T, Legal);
    setOperationAction(ISD::AVGFLOORS, T, Legal);
    setOperationAction(ISD::AVGFLOORU, T, Legal);
    setOperationAction(ISD::AVGCEILS, T, Legal);
    setOperationAction(ISD::AVGCEILU, T, Legal);
  }

  setOperationAction(ISD::SELECT, MVT::v1024i16, Custom);
  setOperationAction(ISD::SELECT, MVT::v1024i32, Custom);

  // Splat supports halfword singles and word pairs only.
  for (MVT T : LegalTypes)
    setOperationAction(ISD::SPLAT_VECTOR, T, Legal);

  setOperationAction(ISD::LOAD, MVT::v2048i8, Legal);
  setOperationAction(ISD::LOAD, MVT::v1024i16, Legal);
  setOperationAction(ISD::LOAD, MVT::v512i32, Legal);
  setOperationAction(ISD::STORE, MVT::v2048i8, Legal);
  setOperationAction(ISD::STORE, MVT::v1024i16, Legal);
  setOperationAction(ISD::STORE, MVT::v512i32, Legal);

  setOperationAction(ISD::MLOAD, MVT::v1024i16, Custom);
  setOperationAction(ISD::MSTORE, MVT::v1024i16, Custom);

  setOperationAction(ISD::SIGN_EXTEND, MVT::v1024i32, Legal);
  setOperationAction(ISD::ZERO_EXTEND, MVT::v1024i32, Legal);
  setOperationAction(ISD::ZERO_EXTEND, MVT::v2048i16, Legal);

  setOperationAction(ISD::TRUNCATE, MVT::v1024i16, Legal);
  setOperationAction(ISD::TRUNCATE, MVT::v2048i8, Legal);
}

bool HexagonTargetLowering::isHlxSingleTy(MVT Ty) const {
  return Subtarget.isHLXVectorType(Ty) &&
         Ty.getSizeInBits() == 8 * Subtarget.getHLXVectorLength();
}

bool HexagonTargetLowering::isHlxPairTy(MVT Ty) const {
  return Subtarget.isHLXVectorType(Ty) &&
         Ty.getSizeInBits() == 16 * Subtarget.getHLXVectorLength();
}

bool HexagonTargetLowering::isHlxOperation(SDNode *N, SelectionDAG &DAG) const {
  if (!Subtarget.useHLXOps())
    return false;
  // If the type of any result, or any operand type are HLX vector types,
  // this is an HLX operation.
  auto IsHLXTy = [this](EVT Ty) {
    return Ty.isSimple() && Subtarget.isHLXVectorType(Ty.getSimpleVT());
  };
  auto IsHLXOp = [this](SDValue Op) {
    return Op.getValueType().isSimple() && Subtarget.isHLXVectorType(ty(Op));
  };
  if (llvm::any_of(N->values(), IsHLXTy) || llvm::any_of(N->ops(), IsHLXOp))
    return true;

  // No widening for now
  return false;
}

bool HexagonTargetLowering::allowsHlxMemoryAccess(
    MVT VecTy, MachineMemOperand::Flags Flags, unsigned *Fast) const {
  // Disallow double vector stores (to prevent unnecessary
  // store widening in DAG combiner).
  if (VecTy.getSizeInBits() > 8 * Subtarget.getHLXVectorLength())
    return false;
  if (!Subtarget.isHLXVectorType(VecTy))
    return false;
  if (Fast)
    *Fast = 1;
  return true;
}

HexagonTargetLowering::TypePair
HexagonTargetLowering::typeHlxSplit(MVT VecTy) const {
  assert(VecTy.isVector());
  unsigned NumElem = VecTy.getVectorNumElements();
  assert((NumElem % 2) == 0 && "Expecting even-sized vector type");
  MVT ElemTy = VecTy.getVectorElementType();
  MVT HalfTy;
  if (ElemTy == MVT::i32)
    HalfTy = VecTy.changeVectorElementType(MVT::i16);
  else
    HalfTy = MVT::getVectorVT(VecTy.getVectorElementType(), NumElem / 2);
  return {HalfTy, HalfTy};
}

// Reload a stack-passed HLX pair argument: two contiguous sub-vector loads
// recombined with COMBINELL. Calling-convention use only; user pair loads
// through a pointer are unsupported (QTOOL-134974).
SDValue HexagonTargetLowering::splitHlxPairArgLoad(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue BaseAddr,
    MVT PairTy, MachinePointerInfo PtrInfo) const {
  unsigned HlxVLen = Subtarget.getHLXVectorLength();
  MVT SingleTy = PairTy.changeVectorElementType(MVT::i16);
  SDValue Base1 =
      DAG.getMemBasePlusOffset(BaseAddr, TypeSize::getFixed(HlxVLen), dl);
  MachineFunction &MF = DAG.getMachineFunction();
  Align HalfAlign = DAG.getEVTAlign(SingleTy);
  MachineMemOperand *MOp0 = MF.getMachineMemOperand(
      PtrInfo, MachineMemOperand::MOLoad, HlxVLen, HalfAlign);
  MachineMemOperand *MOp1 =
      MF.getMachineMemOperand(PtrInfo.getWithOffset(HlxVLen),
                              MachineMemOperand::MOLoad, HlxVLen, HalfAlign);
  SDValue Load0 = DAG.getLoad(SingleTy, dl, Chain, BaseAddr, MOp0);
  SDValue Load1 = DAG.getLoad(SingleTy, dl, Chain, Base1, MOp1);
  return DAG.getNode(HexagonISD::COMBINELL, dl, PairTy, Load1, Load0);
}

// Spill a stack-passed HLX pair argument: extract vsub_lo/vsub_hi and store
// them to contiguous slots, appending both chains to MemOpChains.
// Calling-convention use only (QTOOL-134974).
void HexagonTargetLowering::splitHlxPairArgStore(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue StoreVal,
    SDValue BaseAddr, MVT PairTy, MachinePointerInfo PtrInfo,
    SmallVectorImpl<SDValue> &MemOpChains) const {
  unsigned HlxVLen = Subtarget.getHLXVectorLength();
  MVT SingleTy = PairTy.changeVectorElementType(MVT::i16);
  SDValue Base1 =
      DAG.getMemBasePlusOffset(BaseAddr, TypeSize::getFixed(HlxVLen), dl);
  SDValue Lo =
      DAG.getTargetExtractSubreg(Hexagon::vsub_lo, dl, SingleTy, StoreVal);
  SDValue Hi =
      DAG.getTargetExtractSubreg(Hexagon::vsub_hi, dl, SingleTy, StoreVal);
  MemOpChains.push_back(DAG.getStore(Chain, dl, Lo, BaseAddr, PtrInfo));
  MemOpChains.push_back(
      DAG.getStore(Chain, dl, Hi, Base1, PtrInfo.getWithOffset(HlxVLen)));
}

// Lower masked load/store operations to HLX instructions.
//
// HLX does not have native masked load/store instructions. Instead, we
// lower them to a sequence of regular load/store + VSELECT operations:
//
// MLOAD(base, mask, passthru) => VSELECT(mask, LOAD(base), passthru)
// MSTORE(base, value, mask)   => STORE(base, VSELECT(mask, value,
// LOAD(base)))
//
// For MLOAD: Load the full vector, then use VSELECT to choose between
// loaded values (where mask is true) and passthru values (where mask is
// false).
//
// For MSTORE: Load the existing memory contents, use VSELECT to merge the
// new value (where mask is true) with old contents (where mask is false),
// then store the result. This preserves unmasked lanes.
SDValue HexagonTargetLowering::LowerHlxMaskedOp(SDValue Op,
                                                SelectionDAG &DAG) const {
  auto *MaskN = cast<MaskedLoadStoreSDNode>(Op.getNode());
  unsigned HwLen = Subtarget.getHLXVectorLength();

  // HLX has no native masked load/store. We emulate them with a full-width
  // load (and, for stores, a load-modify-store), which reads/writes the
  // inactive lanes. That is only sound for ordinary memory accesses: a
  // volatile or atomic access must not touch inactive lanes, so bail out
  // rather than silently change the program's memory semantics.
  if (MaskN->isVolatile() || !MaskN->getMemOperand()->isUnordered()) {
    DAG.getContext()->emitError(
        "Volatile or atomic masked load/store of HLX vector type is not "
        "supported");
    return SDValue();
  }

  if (MaskN->getAlign().value() % HwLen != 0) {
    DAG.getContext()->emitError(
        "Masked load/store of HLX vector type, requires alignment of " +
        Twine(HwLen) + " bytes for lowering to HLX");
    return SDValue();
  }

  const SDLoc &dl(Op);
  MachineFunction &MF = DAG.getMachineFunction();
  SDValue Mask = MaskN->getMask();
  SDValue Chain = MaskN->getChain();
  SDValue Base = MaskN->getBasePtr();
  MachineMemOperand *MemOp =
      MaskN->getMemOperand()
          ? MF.getMachineMemOperand(MaskN->getMemOperand(), 0, HwLen)
          : nullptr;

  unsigned Opc = Op->getOpcode();

  switch (Opc) {
  case ISD::MLOAD: {
    MVT ValTy = ty(Op);
    SDValue Load = DAG.getLoad(ValTy, dl, Chain, Base, MemOp);
    SDValue Thru = cast<MaskedLoadSDNode>(MaskN)->getPassThru();
    if (isUndef(Thru))
      return Load;
    SDValue VSel = DAG.getNode(ISD::VSELECT, dl, ValTy, Mask, Load, Thru);
    return DAG.getMergeValues({VSel, Load.getValue(1)}, dl);
  }
  case ISD::MSTORE: {
    SDValue Value = cast<MaskedStoreSDNode>(MaskN)->getValue();
    auto PtrInfo = (MemOp && MemOp->getValue())
                       ? MachinePointerInfo(MemOp->getValue())
                       : MachinePointerInfo();
    SDValue Load =
        DAG.getLoad(Value.getSimpleValueType(), dl, Chain, Base, PtrInfo,
                    Align(HwLen), MachineMemOperand::MOLoad);
    SDValue VSel = DAG.getNode(ISD::VSELECT, dl, Value.getSimpleValueType(),
                               Mask, Value, Load);
    SDValue Store = DAG.getStore(Load.getValue(1), dl, VSel, Base, MemOp);
    return Store;
  }
  default: {
    [[maybe_unused]] std::string Name =
        "Unexpected masked operation: " + Op->getOperationName(&DAG);
    llvm_unreachable(Name.c_str());
  }
  }
}

// Lowers ternary operations of the following types:
// HLX_Vector = (int)? HLX_Vector: HLX_Vector
// HLX_VectorPair = (int)? HLX_VectorPair: HLX_VectorPair
SDValue HexagonTargetLowering::LowerHlxSelect(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDValue CondVec = Op.getOperand(0); // i1
  SDValue True = Op.getOperand(1);
  SDValue False = Op.getOperand(2);

  const SDLoc &dl(Op);
  MVT ValTy = ty(Op);

  // Always splat condition into single-vector v1024i16 (HlxLR), not pair.
  // LX_mux_h only needs a single HlxLR condition register, and for pairs
  // the scalar i1 condition is uniform so the same register is reused for
  // both halves.
  MVT CondTy = MVT::getVectorVT(MVT::i16, ValTy.getVectorNumElements());
  if (ty(CondVec) == MVT::i1) {
    SDValue Ext = DAG.getNode(ISD::SIGN_EXTEND, dl, MVT::i16, CondVec);
    CondVec = DAG.getNode(ISD::SPLAT_VECTOR, dl, CondTy, Ext);
  }

  if (!isHlxPairTy(ValTy))
    return DAG.getNode(ISD::VSELECT, dl, ValTy, CondVec, True, False);

  TypePair HalfTys = typeHlxSplit(ValTy);
  MVT SingleTy = HalfTys.first; // v1024i16

  SDValue TrueLo =
      DAG.getTargetExtractSubreg(Hexagon::vsub_lo, dl, SingleTy, True);
  SDValue TrueHi =
      DAG.getTargetExtractSubreg(Hexagon::vsub_hi, dl, SingleTy, True);
  SDValue FalseLo =
      DAG.getTargetExtractSubreg(Hexagon::vsub_lo, dl, SingleTy, False);
  SDValue FalseHi =
      DAG.getTargetExtractSubreg(Hexagon::vsub_hi, dl, SingleTy, False);

  SDValue LoSelect =
      DAG.getNode(ISD::VSELECT, dl, SingleTy, CondVec, TrueLo, FalseLo);
  SDValue HiSelect =
      DAG.getNode(ISD::VSELECT, dl, SingleTy, CondVec, TrueHi, FalseHi);

  return DAG.getNode(HexagonISD::COMBINELL, dl, ValTy, HiSelect, LoSelect);
}

SDValue HexagonTargetLowering::LowerHlxOperation(SDValue Op,
                                                 SelectionDAG &DAG) const {
  unsigned Opc = Op.getOpcode();
  switch (Opc) {
  default:
    break;
  case ISD::INTRINSIC_W_CHAIN:
  case ISD::INTRINSIC_VOID:
  case ISD::INTRINSIC_WO_CHAIN:
    return Op;
  case ISD::MLOAD:
  case ISD::MSTORE:
    return LowerHlxMaskedOp(Op, DAG);
  case ISD::SELECT:
    return LowerHlxSelect(Op, DAG);
  case ISD::ADD:
  case ISD::SUB:
  case ISD::UADDSAT:
  case ISD::SADDSAT:
  case ISD::USUBSAT:
  case ISD::SSUBSAT:
  case ISD::MUL:
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
  case ISD::SMIN:
  case ISD::SMAX:
  case ISD::UMIN:
  case ISD::UMAX:
  case ISD::ABS:
  case ISD::ABDS:
  case ISD::ABDU:
  case ISD::AVGFLOORS:
  case ISD::AVGFLOORU:
  case ISD::AVGCEILS:
  case ISD::AVGCEILU:
  case ISD::SPLAT_VECTOR:
  case ISD::VSELECT:
  case ISD::SETCC: {
    MVT VT = Opc == ISD::SETCC ? ty(Op.getOperand(0)) : ty(Op);
    DAG.getContext()->emitError(
        "HLX does not support '" + Op->getOperationName(&DAG) +
        "' on vector type " + EVT(VT).getEVTString() +
        "; use an element type and vector length supported by the HLX ISA");
    return DAG.getUNDEF(ty(Op));
  }
  }
#ifndef NDEBUG
  Op.dumpr(&DAG);
#endif
  llvm_unreachable("Unhandled HLX operation");
}
