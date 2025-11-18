//===-- AArch64SelectionDAGInfo.cpp - AArch64 SelectionDAG Info -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AArch64SelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#include "AArch64SelectionDAGInfo.h"
#include "AArch64MachineFunctionInfo.h"

#define GET_SDNODE_DESC
#include "AArch64GenSDNodeInfo.inc"
#undef GET_SDNODE_DESC

using namespace llvm;

#define DEBUG_TYPE "aarch64-selectiondag-info"

static cl::opt<bool>
    LowerToSMERoutines("aarch64-lower-to-sme-routines", cl::Hidden,
                       cl::desc("Enable AArch64 SME memory operations "
                                "to lower to librt functions"),
                       cl::init(true));

AArch64SelectionDAGInfo::AArch64SelectionDAGInfo()
    : SelectionDAGGenTargetInfo(AArch64GenSDNodeInfo) {}

void AArch64SelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                               const SDNode *N) const {
  switch (N->getOpcode()) {
  case AArch64ISD::WrapperLarge:
    // operand #0 must have type i32, but has type i64
    return;
  }

  SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);

#ifndef NDEBUG
  // Some additional checks not yet implemented by verifyTargetNode.
  switch (N->getOpcode()) {
  case AArch64ISD::SADDWT:
  case AArch64ISD::SADDWB:
  case AArch64ISD::UADDWT:
  case AArch64ISD::UADDWB: {
    EVT VT = N->getValueType(0);
    EVT Op0VT = N->getOperand(0).getValueType();
    EVT Op1VT = N->getOperand(1).getValueType();
    assert(VT.isVector() && Op0VT.isVector() && Op1VT.isVector() &&
           VT.isInteger() && Op0VT.isInteger() && Op1VT.isInteger() &&
           "Expected integer vectors!");
    assert(VT == Op0VT &&
           "Expected result and first input to have the same type!");
    assert(Op0VT.getSizeInBits() == Op1VT.getSizeInBits() &&
           "Expected vectors of equal size!");
    assert(Op0VT.getVectorElementCount() * 2 == Op1VT.getVectorElementCount() &&
           "Expected result vector and first input vector to have half the "
           "lanes of the second input vector!");
    break;
  }
  case AArch64ISD::SUNPKLO:
  case AArch64ISD::SUNPKHI:
  case AArch64ISD::UUNPKLO:
  case AArch64ISD::UUNPKHI: {
    EVT VT = N->getValueType(0);
    EVT OpVT = N->getOperand(0).getValueType();
    assert(OpVT.isVector() && VT.isVector() && OpVT.isInteger() &&
           VT.isInteger() && "Expected integer vectors!");
    assert(OpVT.getSizeInBits() == VT.getSizeInBits() &&
           "Expected vectors of equal size!");
    assert(OpVT.getVectorElementCount() == VT.getVectorElementCount() * 2 &&
           "Expected result vector with half the lanes of its input!");
    break;
  }
  case AArch64ISD::TRN1:
  case AArch64ISD::TRN2:
  case AArch64ISD::UZP1:
  case AArch64ISD::UZP2:
  case AArch64ISD::ZIP1:
  case AArch64ISD::ZIP2: {
    EVT VT = N->getValueType(0);
    EVT Op0VT = N->getOperand(0).getValueType();
    EVT Op1VT = N->getOperand(1).getValueType();
    assert(VT.isVector() && Op0VT.isVector() && Op1VT.isVector() &&
           "Expected vectors!");
    assert(VT == Op0VT && VT == Op1VT && "Expected matching vectors!");
    break;
  }
  case AArch64ISD::RSHRNB_I: {
    EVT VT = N->getValueType(0);
    EVT Op0VT = N->getOperand(0).getValueType();
    assert(VT.isVector() && VT.isInteger() &&
           "Expected integer vector result type!");
    assert(Op0VT.isVector() && Op0VT.isInteger() &&
           "Expected first operand to be an integer vector!");
    assert(VT.getSizeInBits() == Op0VT.getSizeInBits() &&
           "Expected vectors of equal size!");
    assert(VT.getVectorElementCount() == Op0VT.getVectorElementCount() * 2 &&
           "Expected input vector with half the lanes of its result!");
    assert(isa<ConstantSDNode>(N->getOperand(1)) &&
           "Expected second operand to be a constant!");
    break;
  }
  }
#endif
}

SDValue AArch64SelectionDAGInfo::EmitMOPS(unsigned Opcode, SelectionDAG &DAG,
                                          const SDLoc &DL, SDValue Chain,
                                          SDValue Dst, SDValue SrcOrValue,
                                          SDValue Size, Align Alignment,
                                          bool isVolatile,
                                          MachinePointerInfo DstPtrInfo,
                                          MachinePointerInfo SrcPtrInfo) const {

  // Get the constant size of the copy/set.
  uint64_t ConstSize = 0;
  if (auto *C = dyn_cast<ConstantSDNode>(Size))
    ConstSize = C->getZExtValue();

  const bool IsSet = Opcode == AArch64::MOPSMemorySetPseudo ||
                     Opcode == AArch64::MOPSMemorySetTaggingPseudo;

  MachineFunction &MF = DAG.getMachineFunction();

  auto Vol =
      isVolatile ? MachineMemOperand::MOVolatile : MachineMemOperand::MONone;
  auto DstFlags = MachineMemOperand::MOStore | Vol;
  auto *DstOp =
      MF.getMachineMemOperand(DstPtrInfo, DstFlags, ConstSize, Alignment);

  if (IsSet) {
    // Extend value to i64, if required.
    if (SrcOrValue.getValueType() != MVT::i64)
      SrcOrValue = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i64, SrcOrValue);
    SDValue Ops[] = {Dst, Size, SrcOrValue, Chain};
    const EVT ResultTys[] = {MVT::i64, MVT::i64, MVT::Other};
    MachineSDNode *Node = DAG.getMachineNode(Opcode, DL, ResultTys, Ops);
    DAG.setNodeMemRefs(Node, {DstOp});
    return SDValue(Node, 2);
  } else {
    SDValue Ops[] = {Dst, SrcOrValue, Size, Chain};
    const EVT ResultTys[] = {MVT::i64, MVT::i64, MVT::i64, MVT::Other};
    MachineSDNode *Node = DAG.getMachineNode(Opcode, DL, ResultTys, Ops);

    auto SrcFlags = MachineMemOperand::MOLoad | Vol;
    auto *SrcOp =
        MF.getMachineMemOperand(SrcPtrInfo, SrcFlags, ConstSize, Alignment);
    DAG.setNodeMemRefs(Node, {DstOp, SrcOp});
    return SDValue(Node, 3);
  }
}

SDValue AArch64SelectionDAGInfo::EmitStreamingCompatibleMemLibCall(
    SelectionDAG &DAG, const SDLoc &DL, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, RTLIB::Libcall LC) const {
  const AArch64Subtarget &STI =
      DAG.getMachineFunction().getSubtarget<AArch64Subtarget>();
  const AArch64TargetLowering *TLI = STI.getTargetLowering();
  TargetLowering::ArgListTy Args;
  Args.emplace_back(Dst, PointerType::getUnqual(*DAG.getContext()));

  RTLIB::Libcall NewLC;
  switch (LC) {
  case RTLIB::MEMCPY: {
    NewLC = RTLIB::SC_MEMCPY;
    Args.emplace_back(Src, PointerType::getUnqual(*DAG.getContext()));
    break;
  }
  case RTLIB::MEMMOVE: {
    NewLC = RTLIB::SC_MEMMOVE;
    Args.emplace_back(Src, PointerType::getUnqual(*DAG.getContext()));
    break;
  }
  case RTLIB::MEMSET: {
    NewLC = RTLIB::SC_MEMSET;
    Args.emplace_back(DAG.getZExtOrTrunc(Src, DL, MVT::i32),
                      Type::getInt32Ty(*DAG.getContext()));
    break;
  }
  default:
    return SDValue();
  }

  EVT PointerVT = TLI->getPointerTy(DAG.getDataLayout());
  SDValue Symbol = DAG.getExternalSymbol(TLI->getLibcallName(NewLC), PointerVT);
  Args.emplace_back(Size, DAG.getDataLayout().getIntPtrType(*DAG.getContext()));

  TargetLowering::CallLoweringInfo CLI(DAG);
  PointerType *RetTy = PointerType::getUnqual(*DAG.getContext());
  CLI.setDebugLoc(DL).setChain(Chain).setLibCallee(
      TLI->getLibcallCallingConv(NewLC), RetTy, Symbol, std::move(Args));
  return TLI->LowerCallTo(CLI).second;
}

SDValue AArch64SelectionDAGInfo::EmitTargetCodeForMemcpy(
    SelectionDAG &DAG, const SDLoc &DL, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  const AArch64Subtarget &STI =
      DAG.getMachineFunction().getSubtarget<AArch64Subtarget>();

  if (STI.hasMOPS())
    return EmitMOPS(AArch64::MOPSMemoryCopyPseudo, DAG, DL, Chain, Dst, Src,
                    Size, Alignment, isVolatile, DstPtrInfo, SrcPtrInfo);

  auto *AFI = DAG.getMachineFunction().getInfo<AArch64FunctionInfo>();
  SMEAttrs Attrs = AFI->getSMEFnAttrs();
  if (LowerToSMERoutines && !Attrs.hasNonStreamingInterfaceAndBody())
    return EmitStreamingCompatibleMemLibCall(DAG, DL, Chain, Dst, Src, Size,
                                             RTLIB::MEMCPY);
  return SDValue();
}

SDValue AArch64SelectionDAGInfo::EmitTargetCodeForMemset(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo) const {
  const AArch64Subtarget &STI =
      DAG.getMachineFunction().getSubtarget<AArch64Subtarget>();

  if (STI.hasMOPS())
    return EmitMOPS(AArch64::MOPSMemorySetPseudo, DAG, dl, Chain, Dst, Src,
                    Size, Alignment, isVolatile, DstPtrInfo,
                    MachinePointerInfo{});

  auto *AFI = DAG.getMachineFunction().getInfo<AArch64FunctionInfo>();
  SMEAttrs Attrs = AFI->getSMEFnAttrs();
  if (LowerToSMERoutines && !Attrs.hasNonStreamingInterfaceAndBody())
    return EmitStreamingCompatibleMemLibCall(DAG, dl, Chain, Dst, Src, Size,
                                             RTLIB::MEMSET);
  return SDValue();
}

SDValue AArch64SelectionDAGInfo::EmitTargetCodeForMemmove(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  const AArch64Subtarget &STI =
      DAG.getMachineFunction().getSubtarget<AArch64Subtarget>();

  if (STI.hasMOPS())
    return EmitMOPS(AArch64::MOPSMemoryMovePseudo, DAG, dl, Chain, Dst, Src,
                    Size, Alignment, isVolatile, DstPtrInfo, SrcPtrInfo);

  auto *AFI = DAG.getMachineFunction().getInfo<AArch64FunctionInfo>();
  SMEAttrs Attrs = AFI->getSMEFnAttrs();
  if (LowerToSMERoutines && !Attrs.hasNonStreamingInterfaceAndBody())
    return EmitStreamingCompatibleMemLibCall(DAG, dl, Chain, Dst, Src, Size,
                                             RTLIB::MEMMOVE);

  // Handle small memmove cases with overlapping loads/stores for better codegen
  // For non-power-of-two sizes, use overlapping operations instead of
  // mixed-size operations (e.g., for 7 bytes: two i32 loads/stores with overlap
  // instead of i32 + i16 + i8)
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Size)) {
    uint64_t SizeVal = C->getZExtValue();
    const TargetLowering &TLI = DAG.getTargetLoweringInfo();

    auto AlignmentIsAcceptable = [&](EVT VT, Align AlignCheck) {
      if (Alignment >= AlignCheck)
        return true;
      unsigned Fast;
      return TLI.allowsMisalignedMemoryAccesses(
                 VT, DstPtrInfo.getAddrSpace(), Align(1),
                 MachineMemOperand::MONone, &Fast) &&
             Fast;
    };

    MachineMemOperand::Flags MMOFlags =
        isVolatile ? MachineMemOperand::MOVolatile : MachineMemOperand::MONone;

    // For sizes 5-7 bytes: use two overlapping i32 operations
    if (SizeVal >= 5 && SizeVal <= 7) {
      if (AlignmentIsAcceptable(MVT::i32, Align(1))) {
        uint64_t SecondOffset = SizeVal - 4;

        SDValue Load1 =
            DAG.getLoad(MVT::i32, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(0)),
                        SrcPtrInfo.getWithOffset(0), Alignment, MMOFlags);

        SDValue Load2 = DAG.getLoad(
            MVT::i32, dl, Chain,
            DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(SecondOffset)),
            SrcPtrInfo.getWithOffset(SecondOffset), Alignment, MMOFlags);

        Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Load1.getValue(1),
                            Load2.getValue(1));

        SDValue Store1 =
            DAG.getStore(Chain, dl, Load1,
                         DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(0)),
                         DstPtrInfo.getWithOffset(0), Alignment, MMOFlags);

        SDValue Store2 = DAG.getStore(
            Chain, dl, Load2,
            DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(SecondOffset)),
            DstPtrInfo.getWithOffset(SecondOffset), Alignment, MMOFlags);

        return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Store1, Store2);
      }
    }

    // For sizes 9-15 bytes: use i64 + overlapping i64
    if (SizeVal >= 9 && SizeVal <= 15) {
      if (AlignmentIsAcceptable(MVT::i64, Align(1))) {
        uint64_t SecondOffset = SizeVal - 8;

        SDValue Load1 =
            DAG.getLoad(MVT::i64, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(0)),
                        SrcPtrInfo.getWithOffset(0), Alignment, MMOFlags);

        SDValue Load2 = DAG.getLoad(
            MVT::i64, dl, Chain,
            DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(SecondOffset)),
            SrcPtrInfo.getWithOffset(SecondOffset), Alignment, MMOFlags);
        
        Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Load1.getValue(1),
                            Load2.getValue(1));

        SDValue Store1 =
            DAG.getStore(Chain, dl, Load1,
                         DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(0)),
                         DstPtrInfo.getWithOffset(0), Alignment, MMOFlags);

        SDValue Store2 = DAG.getStore(
            Chain, dl, Load2,
            DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(SecondOffset)),
            DstPtrInfo.getWithOffset(SecondOffset), Alignment, MMOFlags);

        return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Store1, Store2);
      }
    }
    
    // For sizes 17-23 bytes: use i64 + i64 + overlapping i64
    if (SizeVal >= 17 && SizeVal <= 23) {
      if (AlignmentIsAcceptable(MVT::i64, Align(1))) {
        uint64_t ThirdOffset = SizeVal - 8;

        SDValue Load1 =
            DAG.getLoad(MVT::i64, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(0)),
                        SrcPtrInfo.getWithOffset(0), Alignment, MMOFlags);

        SDValue Load2 =
            DAG.getLoad(MVT::i64, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(8)),
                        SrcPtrInfo.getWithOffset(8), Alignment, MMOFlags);

        SDValue Load3 = DAG.getLoad(
            MVT::i64, dl, Chain,
            DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(ThirdOffset)),
            SrcPtrInfo.getWithOffset(ThirdOffset), Alignment, MMOFlags);

        Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Load1.getValue(1),
                            Load2.getValue(1), Load3.getValue(1));

        SDValue Store1 =
            DAG.getStore(Chain, dl, Load1,
                         DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(0)),
                         DstPtrInfo.getWithOffset(0), Alignment, MMOFlags);

        SDValue Store2 =
            DAG.getStore(Chain, dl, Load2,
                         DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(8)),
                         DstPtrInfo.getWithOffset(8), Alignment, MMOFlags);

        SDValue Store3 = DAG.getStore(
            Chain, dl, Load3,
            DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(ThirdOffset)),
            DstPtrInfo.getWithOffset(ThirdOffset), Alignment, MMOFlags);

        return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Store1, Store2,
                           Store3);
      }
    }

    // For sizes 25-31 bytes: use v16i8 (vector) + overlapping i64
    if (SizeVal >= 25 && SizeVal <= 31) {
      if (AlignmentIsAcceptable(MVT::v16i8, Align(1)) &&
          AlignmentIsAcceptable(MVT::i64, Align(1))) {
        uint64_t SecondOffset = SizeVal - 8;

        SDValue Load1 =
            DAG.getLoad(MVT::v16i8, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(0)),
                        SrcPtrInfo.getWithOffset(0), Alignment, MMOFlags);

        SDValue Load2 = DAG.getLoad(
            MVT::i64, dl, Chain,
            DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(SecondOffset)),
            SrcPtrInfo.getWithOffset(SecondOffset), Alignment, MMOFlags);

        Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Load1.getValue(1),
                            Load2.getValue(1));

        SDValue Store1 =
            DAG.getStore(Chain, dl, Load1,
                         DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(0)),
                         DstPtrInfo.getWithOffset(0), Alignment, MMOFlags);

        SDValue Store2 = DAG.getStore(
            Chain, dl, Load2,
            DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(SecondOffset)),
            DstPtrInfo.getWithOffset(SecondOffset), Alignment, MMOFlags);

        return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Store1, Store2);
      }
    }

    // For sizes 33-47 bytes: use 2 x v16i8 (vectors) + overlapping i64
    if (SizeVal >= 33 && SizeVal <= 47) {
      if (AlignmentIsAcceptable(MVT::v16i8, Align(1)) &&
          AlignmentIsAcceptable(MVT::i64, Align(1))) {
        uint64_t ThirdOffset = SizeVal - 8;

        SDValue Load1 =
            DAG.getLoad(MVT::v16i8, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(0)),
                        SrcPtrInfo.getWithOffset(0), Alignment, MMOFlags);

        SDValue Load2 =
            DAG.getLoad(MVT::v16i8, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(16)),
                        SrcPtrInfo.getWithOffset(16), Alignment, MMOFlags);

        SDValue Load3 = DAG.getLoad(
            MVT::i64, dl, Chain,
            DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(ThirdOffset)),
            SrcPtrInfo.getWithOffset(ThirdOffset), Alignment, MMOFlags);

        Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Load1.getValue(1),
                            Load2.getValue(1), Load3.getValue(1));

        SDValue Store1 =
            DAG.getStore(Chain, dl, Load1,
                         DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(0)),
                         DstPtrInfo.getWithOffset(0), Alignment, MMOFlags);

        SDValue Store2 = DAG.getStore(
            Chain, dl, Load2,
            DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(16)),
            DstPtrInfo.getWithOffset(16), Alignment, MMOFlags);

        SDValue Store3 = DAG.getStore(
            Chain, dl, Load3,
            DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(ThirdOffset)),
            DstPtrInfo.getWithOffset(ThirdOffset), Alignment, MMOFlags);

        return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Store1, Store2,
                           Store3);
      }
    }

    // For sizes 49-63 bytes: use 3 x v16i8 (vectors) + overlapping i64
    if (SizeVal >= 49 && SizeVal <= 63) {
      if (AlignmentIsAcceptable(MVT::v16i8, Align(1)) &&
          AlignmentIsAcceptable(MVT::i64, Align(1))) {
        uint64_t FourthOffset = SizeVal - 8;

        SDValue Load1 =
            DAG.getLoad(MVT::v16i8, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(0)),
                        SrcPtrInfo.getWithOffset(0), Alignment, MMOFlags);

        SDValue Load2 =
            DAG.getLoad(MVT::v16i8, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(16)),
                        SrcPtrInfo.getWithOffset(16), Alignment, MMOFlags);

        SDValue Load3 =
            DAG.getLoad(MVT::v16i8, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(32)),
                        SrcPtrInfo.getWithOffset(32), Alignment, MMOFlags);

        SDValue Load4 = DAG.getLoad(
            MVT::i64, dl, Chain,
            DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(FourthOffset)),
            SrcPtrInfo.getWithOffset(FourthOffset), Alignment, MMOFlags);

        Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Load1.getValue(1),
                            Load2.getValue(1), Load3.getValue(1),
                            Load4.getValue(1));

        SDValue Store1 =
            DAG.getStore(Chain, dl, Load1,
                         DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(0)),
                         DstPtrInfo.getWithOffset(0), Alignment, MMOFlags);

        SDValue Store2 = DAG.getStore(
            Chain, dl, Load2,
            DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(16)),
            DstPtrInfo.getWithOffset(16), Alignment, MMOFlags);

        SDValue Store3 = DAG.getStore(
            Chain, dl, Load3,
            DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(32)),
            DstPtrInfo.getWithOffset(32), Alignment, MMOFlags);

        SDValue Store4 = DAG.getStore(
            Chain, dl, Load4,
            DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(FourthOffset)),
            DstPtrInfo.getWithOffset(FourthOffset), Alignment, MMOFlags);

        return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Store1, Store2,
                           Store3, Store4);
      }
    }

    // For size 65 bytes: use 4 x v16i8 (vectors) + overlapping i64
    if (SizeVal == 65) {
      if (AlignmentIsAcceptable(MVT::v16i8, Align(1)) &&
          AlignmentIsAcceptable(MVT::i64, Align(1))) {

        SDValue Load1 =
            DAG.getLoad(MVT::v16i8, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(0)),
                        SrcPtrInfo.getWithOffset(0), Alignment, MMOFlags);

        SDValue Load2 =
            DAG.getLoad(MVT::v16i8, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(16)),
                        SrcPtrInfo.getWithOffset(16), Alignment, MMOFlags);

        SDValue Load3 =
            DAG.getLoad(MVT::v16i8, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(32)),
                        SrcPtrInfo.getWithOffset(32), Alignment, MMOFlags);

        SDValue Load4 =
            DAG.getLoad(MVT::v16i8, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(48)),
                        SrcPtrInfo.getWithOffset(48), Alignment, MMOFlags);

        SDValue Load5 =
            DAG.getLoad(MVT::i64, dl, Chain,
                        DAG.getObjectPtrOffset(dl, Src, TypeSize::getFixed(57)),
                        SrcPtrInfo.getWithOffset(57), Alignment, MMOFlags);

        Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Load1.getValue(1),
                            Load2.getValue(1), Load3.getValue(1),
                            Load4.getValue(1), Load5.getValue(1));

        SDValue Store1 =
            DAG.getStore(Chain, dl, Load1,
                         DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(0)),
                         DstPtrInfo.getWithOffset(0), Alignment, MMOFlags);

        SDValue Store2 = DAG.getStore(
            Chain, dl, Load2,
            DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(16)),
            DstPtrInfo.getWithOffset(16), Alignment, MMOFlags);

        SDValue Store3 = DAG.getStore(
            Chain, dl, Load3,
            DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(32)),
            DstPtrInfo.getWithOffset(32), Alignment, MMOFlags);

        SDValue Store4 = DAG.getStore(
            Chain, dl, Load4,
            DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(48)),
            DstPtrInfo.getWithOffset(48), Alignment, MMOFlags);

        SDValue Store5 = DAG.getStore(
            Chain, dl, Load5,
            DAG.getObjectPtrOffset(dl, Dst, TypeSize::getFixed(57)),
            DstPtrInfo.getWithOffset(57), Alignment, MMOFlags);

        return DAG.getNode(ISD::TokenFactor, dl, MVT::Other, Store1, Store2,
                           Store3, Store4, Store5);
      }
    }
  }

  return SDValue();
}

static const int kSetTagLoopThreshold = 176;

static SDValue EmitUnrolledSetTag(SelectionDAG &DAG, const SDLoc &dl,
                                  SDValue Chain, SDValue Ptr, uint64_t ObjSize,
                                  const MachineMemOperand *BaseMemOperand,
                                  bool ZeroData) {
  MachineFunction &MF = DAG.getMachineFunction();
  unsigned ObjSizeScaled = ObjSize / 16;

  SDValue TagSrc = Ptr;
  if (Ptr.getOpcode() == ISD::FrameIndex) {
    int FI = cast<FrameIndexSDNode>(Ptr)->getIndex();
    Ptr = DAG.getTargetFrameIndex(FI, MVT::i64);
    // A frame index operand may end up as [SP + offset] => it is fine to use SP
    // register as the tag source.
    TagSrc = DAG.getRegister(AArch64::SP, MVT::i64);
  }

  const unsigned OpCode1 = ZeroData ? AArch64ISD::STZG : AArch64ISD::STG;
  const unsigned OpCode2 = ZeroData ? AArch64ISD::STZ2G : AArch64ISD::ST2G;

  SmallVector<SDValue, 8> OutChains;
  unsigned OffsetScaled = 0;
  while (OffsetScaled < ObjSizeScaled) {
    if (ObjSizeScaled - OffsetScaled >= 2) {
      SDValue AddrNode = DAG.getMemBasePlusOffset(
          Ptr, TypeSize::getFixed(OffsetScaled * 16), dl);
      SDValue St = DAG.getMemIntrinsicNode(
          OpCode2, dl, DAG.getVTList(MVT::Other),
          {Chain, TagSrc, AddrNode},
          MVT::v4i64,
          MF.getMachineMemOperand(BaseMemOperand, OffsetScaled * 16, 16 * 2));
      OffsetScaled += 2;
      OutChains.push_back(St);
      continue;
    }

    if (ObjSizeScaled - OffsetScaled > 0) {
      SDValue AddrNode = DAG.getMemBasePlusOffset(
          Ptr, TypeSize::getFixed(OffsetScaled * 16), dl);
      SDValue St = DAG.getMemIntrinsicNode(
          OpCode1, dl, DAG.getVTList(MVT::Other),
          {Chain, TagSrc, AddrNode},
          MVT::v2i64,
          MF.getMachineMemOperand(BaseMemOperand, OffsetScaled * 16, 16));
      OffsetScaled += 1;
      OutChains.push_back(St);
    }
  }

  SDValue Res = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, OutChains);
  return Res;
}

SDValue AArch64SelectionDAGInfo::EmitTargetCodeForSetTag(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Addr,
    SDValue Size, MachinePointerInfo DstPtrInfo, bool ZeroData) const {
  uint64_t ObjSize = Size->getAsZExtVal();
  assert(ObjSize % 16 == 0);

  MachineFunction &MF = DAG.getMachineFunction();
  MachineMemOperand *BaseMemOperand = MF.getMachineMemOperand(
      DstPtrInfo, MachineMemOperand::MOStore, ObjSize, Align(16));

  bool UseSetTagRangeLoop =
      kSetTagLoopThreshold >= 0 && (int)ObjSize >= kSetTagLoopThreshold;
  if (!UseSetTagRangeLoop)
    return EmitUnrolledSetTag(DAG, dl, Chain, Addr, ObjSize, BaseMemOperand,
                              ZeroData);

  const EVT ResTys[] = {MVT::i64, MVT::i64, MVT::Other};

  unsigned Opcode;
  if (Addr.getOpcode() == ISD::FrameIndex) {
    int FI = cast<FrameIndexSDNode>(Addr)->getIndex();
    Addr = DAG.getTargetFrameIndex(FI, MVT::i64);
    Opcode = ZeroData ? AArch64::STZGloop : AArch64::STGloop;
  } else {
    Opcode = ZeroData ? AArch64::STZGloop_wback : AArch64::STGloop_wback;
  }
  SDValue Ops[] = {DAG.getTargetConstant(ObjSize, dl, MVT::i64), Addr, Chain};
  SDNode *St = DAG.getMachineNode(Opcode, dl, ResTys, Ops);

  DAG.setNodeMemRefs(cast<MachineSDNode>(St), {BaseMemOperand});
  return SDValue(St, 2);
}
