//===-- Next32ISelLowering.h - Next32 DAG Lowering Interface --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Next32 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Next32_Next32ISELLOWERING_H
#define LLVM_LIB_TARGET_Next32_Next32ISELLOWERING_H

#include "Next32.h"
#include "Next32InstrInfo.h"
#include "TargetInfo/Next32BaseInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/MC/MCContext.h"

struct Next32LeaAddressMode;
struct Next32SHLSimplifyPattern;

namespace llvm {
class Next32Subtarget;
class MCContext;
namespace Next32ISD {
enum NodeType : unsigned {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  ADDFLAGS,
  SUBFLAGS,
  ADCFLAGS,
  SBBFLAGS,
  ADDc,
  SUBc,
  XORc,
  ORc,
  ANDc,
  SHLc,
  SHRc,
  SHRIc,
  SHL64,
  SHR64,
  SHRI64,
  CTLZc,
  CTTZc,
  SELECT,
  SELECTc,
  CHAIN,
  CHAINc,
  CHAINP,
  CHAINPc,
  WRITER,
  NOTc,
  NEGc,
  INC,
  INCc,
  DEC,
  DECc,
  ZEXT8,
  ZEXT8c,
  ZEXT16,
  ZEXT16c,
  RET_FLAG,
  FEEDER_ARGS,
  CALL,
  CALLPTR,
  CALLPTR_WRAPPER,
  CALL_TERMINATOR_TID,
  CALL_TERMINATOR,
  BR_CC,
  DUP,
  SYM,
  SET_FRAME,
  RESET_FRAME,
  ALLOCA,
  WRAPPER,
  FRAME_OFFSET_WRAPPER,
  PSEUDO_LEA,
  PREFETCH,
  G_MEM_WRITE,
  G_MEM_READ_1,
  G_MEM_READ_2,
  G_MEM_READ_4,
  G_MEM_READ_8,
  G_MEM_READ_16,
  G_VMEM_WRITE,
  G_VMEM_READ_1,
  G_VMEM_READ_2,
  G_VMEM_READ_4,
  G_VMEM_READ_8,
  G_VMEM_READ_16,
  G_MEM_FAOP_S,
  G_MEM_FAOP_D,
  G_MEM_CAS_S,
  G_MEM_CAS_D,
  TOKEN_ID_F,
  BARRIER,
  SET_TID,
};

} // namespace Next32ISD

class Next32TargetLowering : public TargetLowering {
public:
  explicit Next32TargetLowering(const TargetMachine &TM,
                                const Next32Subtarget &STI);

  // Provide custom lowering hooks for some operations.
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;

  // This method returns the name of a target specific DAG node.
  const char *getTargetNodeName(unsigned Opcode) const override;

  // This method decides whether folding a constant offset
  // with the given GlobalAddress is legal.
  bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const override;

  // Return true for instructions that should be expanded in the legalizer
  // rather than lowered.
  bool forceExpandNode(SDNode *N) const override;

  // Return true for instructions that should be lowered even though its
  // legalization actions isn't custom.
  bool forceCustomLowering(SDNode *N) const override;

  /// isFMAFasterThanFMulAndFAdd - Return true if an FMA operation is faster
  /// than a pair of fmul and fadd instructions. fmuladd intrinsics will be
  /// expanded to FMAs when this method returns true, otherwise fmuladd is
  /// expanded to fmul + fadd.
  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF,
                                  EVT VT) const override;

  TargetLoweringBase::AtomicExpansionKind
  shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const override;

  ShiftLegalizationStrategy
  preferredShiftLegalizationStrategy(SelectionDAG &DAG, SDNode *N,
                                     unsigned ExpansionFactor) const override;

  // Return true if we support LEA instruction.
  bool shouldConvertOrToAdd() const override;

  /// Places new result values for the node in Results (their number
  /// and types must exactly match those of the original return values of
  /// the node), or leaves Results empty, which indicates that the node is not
  /// to be custom lowered after all.
  void LowerOperationWrapper(SDNode *N, SmallVectorImpl<SDValue> &Results,
                             SelectionDAG &DAG) const override;

  void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                          SelectionDAG &DAG) const override;

  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;

  MachineBasicBlock *
  EmitInstrWithCustomInserter(MachineInstr &MI,
                              MachineBasicBlock *BB) const override;

  MVT getScalarShiftAmountTy(const DataLayout &DL, EVT) const override;

  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                         EVT VT) const override;

  /// In case we can't encode data size in feeders and writers, return
  /// the number of registers which size we can encode.
  unsigned getNumRegistersForCallingConv(LLVMContext &Context,
                                         CallingConv::ID CC,
                                         EVT VT) const override;

  bool shouldExpandGetActiveLaneMask(EVT ResVT, EVT OpVT) const override;

private:
  const Next32Subtarget &Subtarget;

  SDValue LowerTargetGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerADD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSUB(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerUADDSUBO(SDValue Op, SelectionDAG &Dag) const;
  SDValue LowerUADDSUBO_CARRY(SDValue Op, SelectionDAG &DAG) const;
  bool ShouldLowerLongShift(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLongShift(SDValue Op, SelectionDAG &DAG) const;
  bool ShouldForceShiftLibCall(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerShiftLibCall(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSRL(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSHIFT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBRCOND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLOADVector(MemSDNode *Load, SDValue Mask,
                          SelectionDAG &DAG) const;
  SDValue LowerSTOREVector(MemSDNode *Store, SDValue Mask, SDValue Value,
                           SelectionDAG &DAG) const;
  SDValue LowerMLOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerMSTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerATOMIC_LOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerATOMIC_STORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerATOMIC_CMP_SWAP(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerATOMIC_LOAD_OP(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFrameIndex(SDValue Op, SelectionDAG &DAG, uint64_t Offset,
                          SDLoc DL) const;
  SDValue LowerADDRSPACECAST(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTACKSAVE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTACKRESTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVAEND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVACOPY(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVAARG(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerPREFETCH(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerDIVREM(SDValue Op, SelectionDAG &Dag) const;
  SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;

  bool MatchLeaBase(SDValue N, Next32LeaAddressMode &LAM) const;
  bool MatchAdd(SDValue &N, SelectionDAG &DAG, Next32LeaAddressMode &LAM,
                unsigned Depth) const;
  bool ShouldMatchMulToLea(SDValue N) const;
  bool MatchLeaRecursively(SDValue N, SelectionDAG &DAG,
                           Next32LeaAddressMode &LAM, unsigned Depth) const;
  bool ShouldCreateLea(Next32LeaAddressMode &LAM) const;
  SDValue PerformLeaCandidateCombine(SDValue Op, SelectionDAG &DAG) const;

  bool MatchSHLPart(SDValue Op, SelectionDAG &DAG,
                    Next32SHLSimplifyPattern &CP) const;
  SDValue PerformSHLSimplify(SDValue Op, SelectionDAG &DAG) const;

  SDValue PerformExtractVectorEltWithFMulCombine(SDNode *N,
                                                 SelectionDAG &DAG) const;
  SDValue PerformFADDFMAFMULCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue PerformGlobalTLSAddressCombine(SDNode *N, SelectionDAG &DAG) const;
  SDValue PerformBRCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue PerformADDCombine(SDNode *N, SelectionDAG &DCI) const;
  SDValue PerformADDORCombine(SDNode *N, SelectionDAG &DCI) const;
  SDValue PerformBoolCarryPropagationCombine(SDNode *N,
                                             SelectionDAG &DAG) const;
  SDValue PerformLSCombine(SDNode *N, SelectionDAG &DAG) const;
  SDValue PerformAtomicFADDorFSUBCombine(SDValue Op, SelectionDAG &DAG) const;
  SDValue FuseLEAScaleToMul(SDNode *N, SelectionDAG &DAG) const;
  SDValue FuseLEAToLEA(SDNode *N, SelectionDAG &DAG) const;
  SDValue PerformPseudoLeaCombine(SDNode *N, SelectionDAG &DAG) const;

  SDValue ReplaceOpcode(unsigned Opcode, SDValue Op, SelectionDAG &DAG,
                        ArrayRef<SDValue> Ops,
                        SDVTList VTs = {nullptr, 0}) const;
  SDValue ReplaceUnaryOpcode(unsigned Opcode, SDValue Op, SelectionDAG &DAG,
                             SDVTList VTs = {nullptr, 0}) const;

  // Lower the result values of a call, copying them out of physregs into vregs
  SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                          CallingConv::ID CallConv, bool IsVarArg,
                          const SmallVectorImpl<ISD::InputArg> &Ins,
                          const SDLoc &DL, SelectionDAG &DAG,
                          SmallVectorImpl<SDValue> &InVals,
                          const SmallVectorImpl<SDValue> &RetSiteSyms) const;

  // Lower a call into CALLSEQ_START - Next32ISD:CALL - CALLSEQ_END chain
  SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  // Lower incoming arguments, copy physregs into vregs
  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;

  Register getRegisterByName(const char *RegName, LLT Ty,
                             const MachineFunction &MF) const override;

  EVT getOptimalMemOpType(const MemOp &, const AttributeList &) const override {
    return MVT::i32;
  }

  static void
  AnalyzeVarArgsCallOperands(CCState &CCInfo,
                             const SmallVectorImpl<ISD::OutputArg> &Outs,
                             CCAssignFn FixedFn, CCAssignFn VarArgsFn);

  void expandOrPromoteNonRoundLOAD(SDValue Op, SelectionDAG &DAG,
                                   SmallVectorImpl<SDValue> &Ops) const;
  void expandOrPromoteNonRoundSTORE(SDValue Op, SelectionDAG &DAG,
                                    SmallVectorImpl<SDValue> &Ops) const;
};
} // namespace llvm

#endif
