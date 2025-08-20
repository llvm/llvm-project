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
  SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);

#ifndef NDEBUG
  // Some additional checks not yet implemented by verifyTargetNode.
  constexpr MVT FlagsVT = MVT::i32;
  switch (N->getOpcode()) {
  case AArch64ISD::SUBS:
    assert(N->getValueType(1) == FlagsVT);
    break;
  case AArch64ISD::ADC:
  case AArch64ISD::SBC:
    assert(N->getOperand(2).getValueType() == FlagsVT);
    break;
  case AArch64ISD::ADCS:
  case AArch64ISD::SBCS:
    assert(N->getValueType(1) == FlagsVT);
    assert(N->getOperand(2).getValueType() == FlagsVT);
    break;
  case AArch64ISD::CSEL:
  case AArch64ISD::CSINC:
  case AArch64ISD::BRCOND:
    assert(N->getOperand(3).getValueType() == FlagsVT);
    break;
  case AArch64ISD::SADDWT:
  case AArch64ISD::SADDWB:
  case AArch64ISD::UADDWT:
  case AArch64ISD::UADDWB: {
    assert(N->getNumValues() == 1 && "Expected one result!");
    assert(N->getNumOperands() == 2 && "Expected two operands!");
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
    assert(N->getNumValues() == 1 && "Expected one result!");
    assert(N->getNumOperands() == 1 && "Expected one operand!");
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
    assert(N->getNumValues() == 1 && "Expected one result!");
    assert(N->getNumOperands() == 2 && "Expected two operands!");
    EVT VT = N->getValueType(0);
    EVT Op0VT = N->getOperand(0).getValueType();
    EVT Op1VT = N->getOperand(1).getValueType();
    assert(VT.isVector() && Op0VT.isVector() && Op1VT.isVector() &&
           "Expected vectors!");
    assert(VT == Op0VT && VT == Op1VT && "Expected matching vectors!");
    break;
  }
  case AArch64ISD::RSHRNB_I: {
    assert(N->getNumValues() == 1 && "Expected one result!");
    assert(N->getNumOperands() == 2 && "Expected two operands!");
    EVT VT = N->getValueType(0);
    EVT Op0VT = N->getOperand(0).getValueType();
    EVT Op1VT = N->getOperand(1).getValueType();
    assert(VT.isVector() && VT.isInteger() &&
           "Expected integer vector result type!");
    assert(Op0VT.isVector() && Op0VT.isInteger() &&
           "Expected first operand to be an integer vector!");
    assert(VT.getSizeInBits() == Op0VT.getSizeInBits() &&
           "Expected vectors of equal size!");
    assert(VT.getVectorElementCount() == Op0VT.getVectorElementCount() * 2 &&
           "Expected input vector with half the lanes of its result!");
    assert(Op1VT == MVT::i32 && isa<ConstantSDNode>(N->getOperand(1)) &&
           "Expected second operand to be a constant i32!");
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
