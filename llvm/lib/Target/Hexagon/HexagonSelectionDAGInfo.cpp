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
  // These nodes don't have corresponding entries in *.td files yet.
  switch (static_cast<HexagonISD::NodeType>(Opcode)) {
  case HexagonISD::CALLR:
    return "HexagonISD::CALLR";
  case HexagonISD::VROR:
    return "HexagonISD::VROR";
  case HexagonISD::D2P:
    return "HexagonISD::D2P";
  case HexagonISD::P2D:
    return "HexagonISD::P2D";
  case HexagonISD::V2Q:
    return "HexagonISD::V2Q";
  case HexagonISD::Q2V:
    return "HexagonISD::Q2V";
  case HexagonISD::TL_EXTEND:
    return "HexagonISD::TL_EXTEND";
  case HexagonISD::TL_TRUNCATE:
    return "HexagonISD::TL_TRUNCATE";
  case HexagonISD::TYPECAST:
    return "HexagonISD::TYPECAST";
  case HexagonISD::ISEL:
    return "HexagonISD::ISEL";
  }

  return SelectionDAGGenTargetInfo::getTargetNodeName(Opcode);
}

void HexagonSelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                               const SDNode *N) const {
  switch (N->getOpcode()) {
  default:
    break;
  case HexagonISD::VALIGNADDR:
    // invalid number of operands; expected 1, got 2
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

  const char *SpecialMemcpyName = TLI.getLibcallName(
      RTLIB::HEXAGON_MEMCPY_LIKELY_ALIGNED_MIN32BYTES_MULT8BYTES);
  const MachineFunction &MF = DAG.getMachineFunction();
  bool LongCalls = MF.getSubtarget<HexagonSubtarget>().useLongCalls();
  unsigned Flags = LongCalls ? HexagonII::HMOTF_ConstExtended : 0;

  CallingConv::ID CC = TLI.getLibcallCallingConv(
      RTLIB::HEXAGON_MEMCPY_LIKELY_ALIGNED_MIN32BYTES_MULT8BYTES);

  TargetLowering::CallLoweringInfo CLI(DAG);
  CLI.setDebugLoc(dl)
      .setChain(Chain)
      .setLibCallee(
          CC, Type::getVoidTy(*DAG.getContext()),
          DAG.getTargetExternalSymbol(
              SpecialMemcpyName, TLI.getPointerTy(DAG.getDataLayout()), Flags),
          std::move(Args))
      .setDiscardResult();

  std::pair<SDValue, SDValue> CallResult = TLI.LowerCallTo(CLI);
  return CallResult.second;
}
