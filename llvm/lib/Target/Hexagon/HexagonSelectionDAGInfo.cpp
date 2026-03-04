//===-- HexagonSelectionDAGInfo.cpp - Hexagon SelectionDAG Info -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the HexagonSelectionDAGInfo class.
//
//===----------------------------------------------------------------------===//

#include "HexagonSelectionDAGInfo.h"
#include "HexagonTargetMachine.h"
#include "llvm/CodeGen/SelectionDAG.h"

#define GET_SDNODE_DESC
#include "HexagonGenSDNodeInfo.inc"

using namespace llvm;

#define DEBUG_TYPE "hexagon-selectiondag-info"

HexagonSelectionDAGInfo::HexagonSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(HexagonGenSDNodeInfo) {}

const char *HexagonSelectionDAGInfo::getTargetNodeName(unsigned Opcode) const {
#define CASE(NAME)                                                             \
  case HexagonISD::NAME:                                                       \
    return "HexagonISD::" #NAME

  // These nodes don't have corresponding entries in *.td files yet.
  switch (static_cast<HexagonISD::NodeType>(Opcode)) {
    CASE(CALLR);
    CASE(VROR);
    CASE(D2P);
    CASE(P2D);
    CASE(V2Q);
    CASE(Q2V);
    CASE(TL_EXTEND);
    CASE(TL_TRUNCATE);
    CASE(TYPECAST);
    CASE(ISEL);
  }
#undef CASE

  return SelectionDAGGenTargetInfo::getTargetNodeName(Opcode);
}

void HexagonSelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                               const SDNode *N) const {
  switch (N->getOpcode()) {
  default:
    break;
  case HexagonISD::VALIGNADDR:
    // invalid number of operands; expected 1, got 2
  case HexagonISD::VINSERTW0:
    // operand #1 must have type i32, but has type v4i8/v2i16
    return;
  }

  SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
}

SDValue HexagonSelectionDAGInfo::EmitTargetCodeForMemcpy(
    SelectionDAG &DAG, const SDLoc &dl, SDValue Chain, SDValue Dst, SDValue Src,
    SDValue Size, Align Alignment, bool isVolatile, bool AlwaysInline,
    MachinePointerInfo DstPtrInfo, MachinePointerInfo SrcPtrInfo) const {
  ConstantSDNode *ConstantSize = dyn_cast<ConstantSDNode>(Size);
  if (AlwaysInline || Alignment < Align(4) || !ConstantSize)
    return SDValue();

  uint64_t SizeVal = ConstantSize->getZExtValue();
  if (SizeVal < 32 || (SizeVal % 8) != 0)
    return SDValue();

  // Special case aligned memcpys with size >= 32 bytes and a multiple of 8.
  //
  const TargetLowering &TLI = *DAG.getSubtarget().getTargetLowering();
  TargetLowering::ArgListTy Args;
  Type *ArgTy = DAG.getDataLayout().getIntPtrType(*DAG.getContext());
  Args.emplace_back(Dst, ArgTy);
  Args.emplace_back(Src, ArgTy);
  Args.emplace_back(Size, ArgTy);

  RTLIB::LibcallImpl SpecialMemcpyImpl = DAG.getLibcalls().getLibcallImpl(
      RTLIB::HEXAGON_MEMCPY_LIKELY_ALIGNED_MIN32BYTES_MULT8BYTES);
  if (SpecialMemcpyImpl == RTLIB::Unsupported)
    return SDValue();

  const MachineFunction &MF = DAG.getMachineFunction();
  bool LongCalls = MF.getSubtarget<HexagonSubtarget>().useLongCalls();
  unsigned Flags = LongCalls ? HexagonII::HMOTF_ConstExtended : 0;

  CallingConv::ID CC =
      DAG.getLibcalls().getLibcallImplCallingConv(SpecialMemcpyImpl);

  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(dl)
      .setChain(Chain)
      .setLibCallee(
          CC, Type::getVoidTy(*DAG.getContext()),
          DAG.getTargetExternalSymbol(
              SpecialMemcpyImpl, TLI.getPointerTy(DAG.getDataLayout()), Flags),
          std::move(Args))
      .setDiscardResult();

  std::pair<SDValue, SDValue> CallResult = TLI.LowerCallTo(CLI);
  return CallResult.second;
}
