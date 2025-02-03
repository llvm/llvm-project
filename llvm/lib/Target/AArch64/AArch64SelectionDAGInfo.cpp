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
#include "AArch64TargetMachine.h"
#include "Utils/AArch64SMEAttributes.h"

#define GET_SDNODE_DESC
#include "AArch64GenSDNodeInfo.inc"

using namespace llvm;

#define DEBUG_TYPE "aarch64-selectiondag-info"

static cl::opt<bool>
    LowerToSMERoutines("aarch64-lower-to-sme-routines", cl::Hidden,
                       cl::desc("Enable AArch64 SME memory operations "
                                "to lower to librt functions"),
                       cl::init(true));

AArch64SelectionDAGInfo::AArch64SelectionDAGInfo()
    : SelectionDAGGenTargetInfo(AArch64GenSDNodeInfo) {}

const char *AArch64SelectionDAGInfo::getTargetNodeName(unsigned Opcode) const {
#define MAKE_CASE(V)                                                           \
  case V:                                                                      \
    return #V;

  // These nodes don't have corresponding entries in *.td files yet.
  switch (static_cast<AArch64ISD::NodeType>(Opcode)) {
    MAKE_CASE(AArch64ISD::LD2post)
    MAKE_CASE(AArch64ISD::LD3post)
    MAKE_CASE(AArch64ISD::LD4post)
    MAKE_CASE(AArch64ISD::ST2post)
    MAKE_CASE(AArch64ISD::ST3post)
    MAKE_CASE(AArch64ISD::ST4post)
    MAKE_CASE(AArch64ISD::LD1x2post)
    MAKE_CASE(AArch64ISD::LD1x3post)
    MAKE_CASE(AArch64ISD::LD1x4post)
    MAKE_CASE(AArch64ISD::ST1x2post)
    MAKE_CASE(AArch64ISD::ST1x3post)
    MAKE_CASE(AArch64ISD::ST1x4post)
    MAKE_CASE(AArch64ISD::LD1DUPpost)
    MAKE_CASE(AArch64ISD::LD2DUPpost)
    MAKE_CASE(AArch64ISD::LD3DUPpost)
    MAKE_CASE(AArch64ISD::LD4DUPpost)
    MAKE_CASE(AArch64ISD::LD1LANEpost)
    MAKE_CASE(AArch64ISD::LD2LANEpost)
    MAKE_CASE(AArch64ISD::LD3LANEpost)
    MAKE_CASE(AArch64ISD::LD4LANEpost)
    MAKE_CASE(AArch64ISD::ST2LANEpost)
    MAKE_CASE(AArch64ISD::ST3LANEpost)
    MAKE_CASE(AArch64ISD::ST4LANEpost)
    MAKE_CASE(AArch64ISD::SVE_LD2_MERGE_ZERO)
    MAKE_CASE(AArch64ISD::SVE_LD3_MERGE_ZERO)
    MAKE_CASE(AArch64ISD::SVE_LD4_MERGE_ZERO)
    MAKE_CASE(AArch64ISD::GLD1Q_INDEX_MERGE_ZERO)
    MAKE_CASE(AArch64ISD::GLDNT1_INDEX_MERGE_ZERO)
    MAKE_CASE(AArch64ISD::SST1Q_INDEX_PRED)
    MAKE_CASE(AArch64ISD::SSTNT1_INDEX_PRED)
    MAKE_CASE(AArch64ISD::INDEX_VECTOR)
    MAKE_CASE(AArch64ISD::MRRS)
    MAKE_CASE(AArch64ISD::MSRR)
  }
#undef MAKE_CASE

  return SelectionDAGGenTargetInfo::getTargetNodeName(Opcode);
}

bool AArch64SelectionDAGInfo::isTargetMemoryOpcode(unsigned Opcode) const {
  // These nodes don't have corresponding entries in *.td files yet.
  if (Opcode >= AArch64ISD::FIRST_MEMORY_OPCODE &&
      Opcode <= AArch64ISD::LAST_MEMORY_OPCODE)
    return true;

  return SelectionDAGGenTargetInfo::isTargetMemoryOpcode(Opcode);
}

void AArch64SelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                               const SDNode *N) const {
  switch (N->getOpcode()) {
  default:
    break;
  case AArch64ISD::GLDFF1S_IMM_MERGE_ZERO:
  case AArch64ISD::GLDFF1S_MERGE_ZERO:
  case AArch64ISD::GLDFF1S_SCALED_MERGE_ZERO:
  case AArch64ISD::GLDFF1S_SXTW_MERGE_ZERO:
  case AArch64ISD::GLDFF1S_SXTW_SCALED_MERGE_ZERO:
  case AArch64ISD::GLDFF1S_UXTW_MERGE_ZERO:
  case AArch64ISD::GLDFF1S_UXTW_SCALED_MERGE_ZERO:
  case AArch64ISD::GLDFF1_IMM_MERGE_ZERO:
  case AArch64ISD::GLDFF1_MERGE_ZERO:
  case AArch64ISD::GLDFF1_SCALED_MERGE_ZERO:
  case AArch64ISD::GLDFF1_SXTW_MERGE_ZERO:
  case AArch64ISD::GLDFF1_SXTW_SCALED_MERGE_ZERO:
  case AArch64ISD::GLDFF1_UXTW_MERGE_ZERO:
  case AArch64ISD::GLDFF1_UXTW_SCALED_MERGE_ZERO:
  case AArch64ISD::LDFF1S_MERGE_ZERO:
  case AArch64ISD::LDFF1_MERGE_ZERO:
  case AArch64ISD::LDNF1S_MERGE_ZERO:
  case AArch64ISD::LDNF1_MERGE_ZERO:
    // invalid number of results; expected 3, got 2
  case AArch64ISD::SMSTOP:
  case AArch64ISD::COALESCER_BARRIER:
    // invalid number of results; expected 2, got 1
  case AArch64ISD::SMSTART:
    // variadic operand #3 must be Register or RegisterMask
  case AArch64ISD::REVD_MERGE_PASSTHRU:
    // invalid number of operands; expected 3, got 4
    return;
  }

  SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
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
  SDValue Symbol;
  TargetLowering::ArgListEntry DstEntry;
  DstEntry.Ty = PointerType::getUnqual(*DAG.getContext());
  DstEntry.Node = Dst;
  TargetLowering::ArgListTy Args;
  Args.push_back(DstEntry);
  EVT PointerVT = TLI->getPointerTy(DAG.getDataLayout());

  switch (LC) {
  case RTLIB::MEMCPY: {
    TargetLowering::ArgListEntry Entry;
    Entry.Ty = PointerType::getUnqual(*DAG.getContext());
    Symbol = DAG.getExternalSymbol("__arm_sc_memcpy", PointerVT);
    Entry.Node = Src;
    Args.push_back(Entry);
    break;
  }
  case RTLIB::MEMMOVE: {
    TargetLowering::ArgListEntry Entry;
    Entry.Ty = PointerType::getUnqual(*DAG.getContext());
    Symbol = DAG.getExternalSymbol("__arm_sc_memmove", PointerVT);
    Entry.Node = Src;
    Args.push_back(Entry);
    break;
  }
  case RTLIB::MEMSET: {
    TargetLowering::ArgListEntry Entry;
    Entry.Ty = Type::getInt32Ty(*DAG.getContext());
    Symbol = DAG.getExternalSymbol("__arm_sc_memset", PointerVT);
    Src = DAG.getZExtOrTrunc(Src, DL, MVT::i32);
    Entry.Node = Src;
    Args.push_back(Entry);
    break;
  }
  default:
    return SDValue();
  }

  TargetLowering::ArgListEntry SizeEntry;
  SizeEntry.Node = Size;
  SizeEntry.Ty = DAG.getDataLayout().getIntPtrType(*DAG.getContext());
  Args.push_back(SizeEntry);
  assert(Symbol->getOpcode() == ISD::ExternalSymbol &&
         "Function name is not set");

  TargetLowering::CallLoweringInfo CLI(DAG);
  PointerType *RetTy = PointerType::getUnqual(*DAG.getContext());
  CLI.setDebugLoc(DL).setChain(Chain).setLibCallee(
      TLI->getLibcallCallingConv(LC), RetTy, Symbol, std::move(Args));
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

  SMEAttrs Attrs(DAG.getMachineFunction().getFunction());
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

  SMEAttrs Attrs(DAG.getMachineFunction().getFunction());
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

  SMEAttrs Attrs(DAG.getMachineFunction().getFunction());
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
