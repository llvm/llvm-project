//===-- SPUISelLowering.h - Cell SPU DAG Lowering Interface -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Cell SPU uses to lower LLVM code into
// a selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef SPU_ISELLOWERING_H
#define SPU_ISELLOWERING_H

#include "SPU.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/CodeGen/SelectionDAG.h"

namespace llvm {
  namespace SPUISD {
    enum NodeType {
      // Start the numbering where the builting ops and target ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      // Pseudo instructions:
      RET_FLAG,                 ///< Return with flag, matched by bi instruction

      Hi,                       ///< High address component (upper 16)
      Lo,                       ///< Low address component (lower 16)
      PCRelAddr,                ///< Program counter relative address
      AFormAddr,                ///< A-form address (local store)
      IndirectAddr,             ///< D-Form "imm($r)" and X-form "$r($r)"

      LDRESULT,                 ///< Load result (value, chain)
      CALL,                     ///< CALL instruction
      SHUFB,                    ///< Vector shuffle (permute)
      SHUFFLE_MASK,             ///< Shuffle mask
      CNTB,                     ///< Count leading ones in bytes
      PREFSLOT2VEC,             ///< Promote scalar->vector
      VEC2PREFSLOT,             ///< Extract element 0
      SHL_BITS,                 ///< Shift quad left, by bits
      SHL_BYTES,                ///< Shift quad left, by bytes
      SRL_BYTES,                ///< Shift quad right, by bytes. Insert zeros.
      VEC_ROTL,                 ///< Vector rotate left
      VEC_ROTR,                 ///< Vector rotate right
      ROTBYTES_LEFT,            ///< Rotate bytes (loads -> ROTQBYI)
      ROTBYTES_LEFT_BITS,       ///< Rotate bytes left by bit shift count
      SELECT_MASK,              ///< Select Mask (FSM, FSMB, FSMH, FSMBI)
      SELB,                     ///< Select bits -> (b & mask) | (a & ~mask)
      // Markers: These aren't used to generate target-dependent nodes, but
      // are used during instruction selection.
      ADD64_MARKER,             ///< i64 addition marker
      SUB64_MARKER,             ///< i64 subtraction marker
      MUL64_MARKER,             ///< i64 multiply marker
      LAST_SPUISD               ///< Last user-defined instruction
    };
  }

  //! Utility functions specific to CellSPU:
  namespace SPU {
    SDValue get_vec_u18imm(SDNode *N, SelectionDAG &DAG,
                             EVT ValueType);
    SDValue get_vec_i16imm(SDNode *N, SelectionDAG &DAG,
                             EVT ValueType);
    SDValue get_vec_i10imm(SDNode *N, SelectionDAG &DAG,
                             EVT ValueType);
    SDValue get_vec_i8imm(SDNode *N, SelectionDAG &DAG,
                            EVT ValueType);
    SDValue get_ILHUvec_imm(SDNode *N, SelectionDAG &DAG,
                              EVT ValueType);
    SDValue get_v4i32_imm(SDNode *N, SelectionDAG &DAG);
    SDValue get_v2i64_imm(SDNode *N, SelectionDAG &DAG);

    SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG,
                              const SPUTargetMachine &TM);
    //! Simplify a EVT::v2i64 constant splat to CellSPU-ready form
    SDValue LowerV2I64Splat(EVT OpVT, SelectionDAG &DAG, uint64_t splat,
                             DebugLoc dl);
  }

  class SPUTargetMachine;            // forward dec'l.

  class SPUTargetLowering :
    public TargetLowering
  {
    SPUTargetMachine &SPUTM;

  public:
    //! The venerable constructor
    /*!
     This is where the CellSPU backend sets operation handling (i.e., legal,
     custom, expand or promote.)
     */
    SPUTargetLowering(SPUTargetMachine &TM);

    //! Get the target machine
    SPUTargetMachine &getSPUTargetMachine() {
      return SPUTM;
    }

    /// getTargetNodeName() - This method returns the name of a target specific
    /// DAG node.
    virtual const char *getTargetNodeName(unsigned Opcode) const;

    /// getSetCCResultType - Return the ValueType for ISD::SETCC
    virtual EVT getSetCCResultType(EVT VT) const;

    virtual MVT getShiftAmountTy(EVT LHSTy) const { return MVT::i32; }

    //! Custom lowering hooks
    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

    //! Custom lowering hook for nodes with illegal result types.
    virtual void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue>&Results,
                                    SelectionDAG &DAG) const;

    virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;

    virtual void computeMaskedBitsForTargetNode(const SDValue Op,
                                                APInt &KnownZero,
                                                APInt &KnownOne,
                                                const SelectionDAG &DAG,
                                                unsigned Depth = 0) const;

    virtual unsigned ComputeNumSignBitsForTargetNode(SDValue Op,
                                                   unsigned Depth = 0) const;

    ConstraintType getConstraintType(const std::string &ConstraintLetter) const;

    /// Examine constraint string and operand type and determine a weight value.
    /// The operand object must already have been set up with the operand type.
    ConstraintWeight getSingleConstraintMatchWeight(
      AsmOperandInfo &info, const char *constraint) const;

    std::pair<unsigned, const TargetRegisterClass*>
      getRegForInlineAsmConstraint(const std::string &Constraint,
                                   EVT VT) const;

    void LowerAsmOperandForConstraint(SDValue Op, std::string &Constraint,
                                      std::vector<SDValue> &Ops,
                                      SelectionDAG &DAG) const;

    /// isLegalAddressImmediate - Return true if the integer value can be used
    /// as the offset of the target addressing mode.
    virtual bool isLegalAddressImmediate(int64_t V, Type *Ty) const;
    virtual bool isLegalAddressImmediate(GlobalValue *) const;

    virtual bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;

    virtual SDValue
      LowerFormalArguments(SDValue Chain,
                           CallingConv::ID CallConv, bool isVarArg,
                           const SmallVectorImpl<ISD::InputArg> &Ins,
                           DebugLoc dl, SelectionDAG &DAG,
                           SmallVectorImpl<SDValue> &InVals) const;

    virtual SDValue
      LowerCall(TargetLowering::CallLoweringInfo &CLI,
                SmallVectorImpl<SDValue> &InVals) const;

    virtual SDValue
      LowerReturn(SDValue Chain,
                  CallingConv::ID CallConv, bool isVarArg,
                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                  const SmallVectorImpl<SDValue> &OutVals,
                  DebugLoc dl, SelectionDAG &DAG) const;

    virtual bool isLegalICmpImmediate(int64_t Imm) const;

    virtual bool isLegalAddressingMode(const AddrMode &AM,
                                       Type *Ty) const;
  };
}

#endif
