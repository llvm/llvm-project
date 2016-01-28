//===-- SIISelLowering.h - SI DAG Lowering Interface ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief SI DAG Lowering interface definition
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_R600_SIISELLOWERING_H
#define LLVM_LIB_TARGET_R600_SIISELLOWERING_H

#include "AMDGPUISelLowering.h"
#include "SIInstrInfo.h"

namespace llvm {

class SITargetLowering : public AMDGPUTargetLowering {
  SDValue LowerParameter(SelectionDAG &DAG, EVT VT, EVT MemVT, SDLoc DL,
                         SDValue Chain, unsigned Offset, bool Signed) const;
  SDValue LowerGlobalAddress(AMDGPUMachineFunction *MFI, SDValue Op,
                             SelectionDAG &DAG) const override;

  SDValue lowerImplicitZextParam(SelectionDAG &DAG, SDValue Op,
                                 MVT VT, unsigned Offset) const;

  SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINTRINSIC_VOID(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFrameIndex(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFastFDIV(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV32(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV64(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINT_TO_FP(SDValue Op, SelectionDAG &DAG, bool Signed) const;
  SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerTrig(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBRCOND(SDValue Op, SelectionDAG &DAG) const;

  void adjustWritemask(MachineSDNode *&N, SelectionDAG &DAG) const;

  SDValue performUCharToFloatCombine(SDNode *N,
                                     DAGCombinerInfo &DCI) const;
  SDValue performSHLPtrCombine(SDNode *N,
                               unsigned AS,
                               DAGCombinerInfo &DCI) const;
  SDValue performAndCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performOrCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performClassCombine(SDNode *N, DAGCombinerInfo &DCI) const;

  SDValue performMinMaxCombine(SDNode *N, DAGCombinerInfo &DCI) const;

  SDValue performSetCCCombine(SDNode *N, DAGCombinerInfo &DCI) const;

  bool isLegalFlatAddressingMode(const AddrMode &AM) const;
  bool isLegalMUBUFAddressingMode(const AddrMode &AM) const;
public:
  SITargetLowering(TargetMachine &tm, const AMDGPUSubtarget &STI);

  bool isShuffleMaskLegal(const SmallVectorImpl<int> &/*Mask*/,
                          EVT /*VT*/) const override;

  bool isLegalAddressingMode(const DataLayout &DL, const AddrMode &AM, Type *Ty,
                             unsigned AS) const override;

  bool allowsMisalignedMemoryAccesses(EVT VT, unsigned AS,
                                      unsigned Align,
                                      bool *IsFast) const override;

  EVT getOptimalMemOpType(uint64_t Size, unsigned DstAlign,
                          unsigned SrcAlign, bool IsMemset,
                          bool ZeroMemset,
                          bool MemcpyStrSrc,
                          MachineFunction &MF) const override;

  bool isMemOpUniform(const SDNode *N) const;
  bool isNoopAddrSpaceCast(unsigned SrcAS, unsigned DestAS) const override;

  TargetLoweringBase::LegalizeTypeAction
  getPreferredVectorAction(EVT VT) const override;

  bool shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                        Type *Ty) const override;

  bool isTypeDesirableForOp(unsigned Op, EVT VT) const override;

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool isVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               SDLoc DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerReturn(SDValue Chain,
                      CallingConv::ID CallConv,
                      bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals,
                      SDLoc DL, SelectionDAG &DAG) const override;

  unsigned getRegisterByName(const char* RegName, EVT VT,
                             SelectionDAG &DAG) const override;

  MachineBasicBlock * EmitInstrWithCustomInserter(MachineInstr * MI,
                                      MachineBasicBlock * BB) const override;
  bool enableAggressiveFMAFusion(EVT VT) const override;
  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                         EVT VT) const override;
  MVT getScalarShiftAmountTy(const DataLayout &, EVT) const override;
  bool isFMAFasterThanFMulAndFAdd(EVT VT) const override;
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;
  SDNode *PostISelFolding(MachineSDNode *N, SelectionDAG &DAG) const override;
  void AdjustInstrPostInstrSelection(MachineInstr *MI,
                                     SDNode *Node) const override;

  int32_t analyzeImmediate(const SDNode *N) const;
  SDValue CreateLiveInRegister(SelectionDAG &DAG, const TargetRegisterClass *RC,
                               unsigned Reg, EVT VT) const override;
  void legalizeTargetIndependentNode(SDNode *Node, SelectionDAG &DAG) const;

  MachineSDNode *wrapAddr64Rsrc(SelectionDAG &DAG, SDLoc DL, SDValue Ptr) const;
  MachineSDNode *buildRSRC(SelectionDAG &DAG,
                           SDLoc DL,
                           SDValue Ptr,
                           uint32_t RsrcDword1,
                           uint64_t RsrcDword2And3) const;
  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;
  ConstraintType getConstraintType(StringRef Constraint) const override;
  SDValue copyToM0(SelectionDAG &DAG, SDValue Chain, SDLoc DL, SDValue V) const;
};

} // End namespace llvm

#endif
