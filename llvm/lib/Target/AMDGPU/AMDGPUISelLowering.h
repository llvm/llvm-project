//===-- AMDGPUISelLowering.h - AMDGPU Lowering Interface --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Interface definition of the TargetLowering class that is common
/// to all AMD GPUs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_R600_AMDGPUISELLOWERING_H
#define LLVM_LIB_TARGET_R600_AMDGPUISELLOWERING_H

#include "llvm/Target/TargetLowering.h"

namespace llvm {

class AMDGPUMachineFunction;
class AMDGPUSubtarget;
class MachineRegisterInfo;

class AMDGPUTargetLowering : public TargetLowering {
protected:
  const AMDGPUSubtarget *Subtarget;

private:
  SDValue LowerConstantInitializer(const Constant* Init, const GlobalValue *GV,
                                   const SDValue &InitPtr,
                                   SDValue Chain,
                                   SelectionDAG &DAG) const;
  SDValue LowerFrameIndex(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerEXTRACT_SUBVECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  /// \brief Lower vector stores by merging the vector elements into an integer
  /// of the same bitwidth.
  SDValue MergeVectorStore(const SDValue &Op, SelectionDAG &DAG) const;
  /// \brief Split a vector store into multiple scalar stores.
  /// \returns The resulting chain.

  SDValue LowerFREM(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFCEIL(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFTRUNC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFRINT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFNEARBYINT(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerFROUND32(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFROUND64(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFROUND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFFLOOR(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerCTLZ(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerINT_TO_FP64(SDValue Op, SelectionDAG &DAG, bool Signed) const;
  SDValue LowerUINT_TO_FP(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSINT_TO_FP(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerFP64_TO_INT(SDValue Op, SelectionDAG &DAG, bool Signed) const;
  SDValue LowerFP_TO_UINT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFP_TO_SINT(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerSIGN_EXTEND_INREG(SDValue Op, SelectionDAG &DAG) const;

  SDValue performStoreCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performShlCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performMulCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue performCtlzCombine(SDLoc SL, SDValue Cond, SDValue LHS, SDValue RHS,
                             DAGCombinerInfo &DCI) const;
  SDValue performSelectCombine(SDNode *N, DAGCombinerInfo &DCI) const;

protected:
  static EVT getEquivalentMemType(LLVMContext &Context, EVT VT);
  static EVT getEquivalentLoadRegType(LLVMContext &Context, EVT VT);

  virtual SDValue LowerGlobalAddress(AMDGPUMachineFunction *MFI, SDValue Op,
                                     SelectionDAG &DAG) const;

  /// \brief Split a vector load into a scalar load of each component.
  SDValue ScalarizeVectorLoad(SDValue Op, SelectionDAG &DAG) const;

  /// \brief Split a vector load into 2 loads of half the vector.
  SDValue SplitVectorLoad(SDValue Op, SelectionDAG &DAG) const;

  /// \brief Split a vector store into a scalar store of each component.
  SDValue ScalarizeVectorStore(SDValue Op, SelectionDAG &DAG) const;

  /// \brief Split a vector store into 2 stores of half the vector.
  SDValue SplitVectorStore(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSDIVREM(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerUDIVREM(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerDIVREM24(SDValue Op, SelectionDAG &DAG, bool sign) const;
  void LowerUDIVREM64(SDValue Op, SelectionDAG &DAG,
                                    SmallVectorImpl<SDValue> &Results) const;
  bool isHWTrueValue(SDValue Op) const;
  bool isHWFalseValue(SDValue Op) const;

  /// The SelectionDAGBuilder will automatically promote function arguments
  /// with illegal types.  However, this does not work for the AMDGPU targets
  /// since the function arguments are stored in memory as these illegal types.
  /// In order to handle this properly we need to get the origianl types sizes
  /// from the LLVM IR Function and fixup the ISD:InputArg values before
  /// passing them to AnalyzeFormalArguments()
  void getOriginalFunctionArgs(SelectionDAG &DAG,
                               const Function *F,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               SmallVectorImpl<ISD::InputArg> &OrigIns) const;
  void AnalyzeFormalArguments(CCState &State,
                              const SmallVectorImpl<ISD::InputArg> &Ins) const;

public:
  AMDGPUTargetLowering(TargetMachine &TM, const AMDGPUSubtarget &STI);

  bool isFAbsFree(EVT VT) const override;
  bool isFNegFree(EVT VT) const override;
  bool isTruncateFree(EVT Src, EVT Dest) const override;
  bool isTruncateFree(Type *Src, Type *Dest) const override;

  bool isZExtFree(Type *Src, Type *Dest) const override;
  bool isZExtFree(EVT Src, EVT Dest) const override;
  bool isZExtFree(SDValue Val, EVT VT2) const override;

  bool isNarrowingProfitable(EVT VT1, EVT VT2) const override;

  MVT getVectorIdxTy(const DataLayout &) const override;
  bool isSelectSupported(SelectSupportKind) const override;

  bool isFPImmLegal(const APFloat &Imm, EVT VT) const override;
  bool ShouldShrinkFPConstant(EVT VT) const override;
  bool shouldReduceLoadWidth(SDNode *Load,
                             ISD::LoadExtType ExtType,
                             EVT ExtVT) const override;

  bool isLoadBitCastBeneficial(EVT, EVT) const override;

  bool storeOfVectorConstantIsCheap(EVT MemVT,
                                    unsigned NumElem,
                                    unsigned AS) const override;
  bool aggressivelyPreferBuildVectorSources(EVT VecVT) const override;
  bool isCheapToSpeculateCttz() const override;
  bool isCheapToSpeculateCtlz() const override;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                      bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals,
                      SDLoc DL, SelectionDAG &DAG) const override;
  SDValue LowerCall(CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  SDValue LowerDYNAMIC_STACKALLOC(SDValue Op,
                                  SelectionDAG &DAG) const;

  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;
  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;
  void ReplaceNodeResults(SDNode * N,
                          SmallVectorImpl<SDValue> &Results,
                          SelectionDAG &DAG) const override;

  SDValue LowerIntrinsicIABS(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerIntrinsicLRP(SDValue Op, SelectionDAG &DAG) const;
  SDValue CombineFMinMaxLegacy(SDLoc DL,
                               EVT VT,
                               SDValue LHS,
                               SDValue RHS,
                               SDValue True,
                               SDValue False,
                               SDValue CC,
                               DAGCombinerInfo &DCI) const;

  const char* getTargetNodeName(unsigned Opcode) const override;

  SDValue getRsqrtEstimate(SDValue Operand,
                           DAGCombinerInfo &DCI,
                           unsigned &RefinementSteps,
                           bool &UseOneConstNR) const override;
  SDValue getRecipEstimate(SDValue Operand,
                           DAGCombinerInfo &DCI,
                           unsigned &RefinementSteps) const override;

  virtual SDNode *PostISelFolding(MachineSDNode *N,
                                  SelectionDAG &DAG) const {
    return N;
  }

  /// \brief Determine which of the bits specified in \p Mask are known to be
  /// either zero or one and return them in the \p KnownZero and \p KnownOne
  /// bitsets.
  void computeKnownBitsForTargetNode(const SDValue Op,
                                     APInt &KnownZero,
                                     APInt &KnownOne,
                                     const SelectionDAG &DAG,
                                     unsigned Depth = 0) const override;

  unsigned ComputeNumSignBitsForTargetNode(SDValue Op, const SelectionDAG &DAG,
                                           unsigned Depth = 0) const override;

  /// \brief Helper function that adds Reg to the LiveIn list of the DAG's
  /// MachineFunction.
  ///
  /// \returns a RegisterSDNode representing Reg.
  virtual SDValue CreateLiveInRegister(SelectionDAG &DAG,
                                       const TargetRegisterClass *RC,
                                       unsigned Reg, EVT VT) const;

  enum ImplicitParameter {
    GRID_DIM,
    GRID_OFFSET
  };

  /// \brief Helper function that returns the byte offset of the given
  /// type of implicit parameter.
  uint32_t getImplicitParameterOffset(const AMDGPUMachineFunction *MFI,
                                      const ImplicitParameter Param) const;
};

namespace AMDGPUISD {

enum NodeType : unsigned {
  // AMDIL ISD Opcodes
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  CALL,        // Function call based on a single integer
  UMUL,        // 32bit unsigned multiplication
  RET_FLAG,
  BRANCH_COND,
  // End AMDIL ISD Opcodes
  DWORDADDR,
  FRACT,
  CLAMP,

  // SIN_HW, COS_HW - f32 for SI, 1 ULP max error, valid from -100 pi to 100 pi.
  // Denormals handled on some parts.
  COS_HW,
  SIN_HW,
  FMAX_LEGACY,
  FMIN_LEGACY,
  FMAX3,
  SMAX3,
  UMAX3,
  FMIN3,
  SMIN3,
  UMIN3,
  URECIP,
  DIV_SCALE,
  DIV_FMAS,
  DIV_FIXUP,
  TRIG_PREOP, // 1 ULP max error for f64

  // RCP, RSQ - For f32, 1 ULP max error, no denormal handling.
  //            For f64, max error 2^29 ULP, handles denormals.
  RCP,
  RSQ,
  RSQ_LEGACY,
  RSQ_CLAMPED,
  LDEXP,
  FP_CLASS,
  DOT4,
  CARRY,
  BORROW,
  BFE_U32, // Extract range of bits with zero extension to 32-bits.
  BFE_I32, // Extract range of bits with sign extension to 32-bits.
  BFI, // (src0 & src1) | (~src0 & src2)
  BFM, // Insert a range of bits into a 32-bit word.
  FFBH_U32, // ctlz with -1 if input is zero.
  MUL_U24,
  MUL_I24,
  MAD_U24,
  MAD_I24,
  TEXTURE_FETCH,
  EXPORT,
  CONST_ADDRESS,
  REGISTER_LOAD,
  REGISTER_STORE,
  LOAD_INPUT,
  SAMPLE,
  SAMPLEB,
  SAMPLED,
  SAMPLEL,

  // These cvt_f32_ubyte* nodes need to remain consecutive and in order.
  CVT_F32_UBYTE0,
  CVT_F32_UBYTE1,
  CVT_F32_UBYTE2,
  CVT_F32_UBYTE3,
  /// This node is for VLIW targets and it is used to represent a vector
  /// that is stored in consecutive registers with the same channel.
  /// For example:
  ///   |X  |Y|Z|W|
  /// T0|v.x| | | |
  /// T1|v.y| | | |
  /// T2|v.z| | | |
  /// T3|v.w| | | |
  BUILD_VERTICAL_VECTOR,
  /// Pointer to the start of the shader's constant data.
  CONST_DATA_PTR,
  SENDMSG,
  INTERP_MOV,
  INTERP_P1,
  INTERP_P2,
  FIRST_MEM_OPCODE_NUMBER = ISD::FIRST_TARGET_MEMORY_OPCODE,
  STORE_MSKOR,
  LOAD_CONSTANT,
  TBUFFER_STORE_FORMAT,
  LAST_AMDGPU_ISD_NUMBER
};


} // End namespace AMDGPUISD

} // End namespace llvm

#endif
