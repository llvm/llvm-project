//===-- HexagonISelLowering.h - Hexagon DAG Lowering Interface --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Hexagon uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef Hexagon_ISELLOWERING_H
#define Hexagon_ISELLOWERING_H

#include "Hexagon.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/CallingConv.h"
#include "llvm/CodeGen/CallingConvLower.h"

namespace llvm {
  namespace HexagonISD {
    enum {
      FIRST_NUMBER = ISD::BUILTIN_OP_END,

      CONST32,
      CONST32_GP,  // For marking data present in GP.
      FCONST32,
      SETCC,
      ADJDYNALLOC,
      ARGEXTEND,

      CMPICC,      // Compare two GPR operands, set icc.
      CMPFCC,      // Compare two FP operands, set fcc.
      BRICC,       // Branch to dest on icc condition
      BRFCC,       // Branch to dest on fcc condition
      SELECT_ICC,  // Select between two values using the current ICC flags.
      SELECT_FCC,  // Select between two values using the current FCC flags.

      Hi, Lo,      // Hi/Lo operations, typically on a global address.

      FTOI,        // FP to Int within a FP register.
      ITOF,        // Int to FP within a FP register.

      CALL,        // A call instruction.
      RET_FLAG,    // Return with a flag operand.
      BR_JT,       // Jump table.
      BARRIER,     // Memory barrier.
      WrapperJT,
      WrapperCP,
      TC_RETURN
    };
  }

  class HexagonTargetLowering : public TargetLowering {
    int VarArgsFrameOffset;   // Frame offset to start of varargs area.

    bool CanReturnSmallStruct(const Function* CalleeFn,
                              unsigned& RetSize) const;

  public:
    HexagonTargetMachine &TM;
    explicit HexagonTargetLowering(HexagonTargetMachine &targetmachine);

    /// IsEligibleForTailCallOptimization - Check whether the call is eligible
    /// for tail call optimization. Targets which want to do tail call
    /// optimization should implement this function.
    bool
    IsEligibleForTailCallOptimization(SDValue Callee,
                                      CallingConv::ID CalleeCC,
                                      bool isVarArg,
                                      bool isCalleeStructRet,
                                      bool isCallerStructRet,
                                      const
                                      SmallVectorImpl<ISD::OutputArg> &Outs,
                                      const SmallVectorImpl<SDValue> &OutVals,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                      SelectionDAG& DAG) const;

    virtual bool isTruncateFree(Type *Ty1, Type *Ty2) const;
    virtual bool isTruncateFree(EVT VT1, EVT VT2) const;

    virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

    virtual const char *getTargetNodeName(unsigned Opcode) const;
    SDValue  LowerBR_JT(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerINLINEASM(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerEH_LABEL(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFormalArguments(SDValue Chain,
                                 CallingConv::ID CallConv, bool isVarArg,
                                 const SmallVectorImpl<ISD::InputArg> &Ins,
                                 DebugLoc dl, SelectionDAG &DAG,
                                 SmallVectorImpl<SDValue> &InVals) const;
    SDValue LowerGLOBALADDRESS(SDValue Op, SelectionDAG &DAG) const;

    SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
                      SmallVectorImpl<SDValue> &InVals) const;

    SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                            CallingConv::ID CallConv, bool isVarArg,
                            const SmallVectorImpl<ISD::InputArg> &Ins,
                            DebugLoc dl, SelectionDAG &DAG,
                            SmallVectorImpl<SDValue> &InVals,
                            const SmallVectorImpl<SDValue> &OutVals,
                            SDValue Callee) const;

    SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;
    SDValue LowerMEMBARRIER(SDValue Op, SelectionDAG& DAG) const;
    SDValue LowerATOMIC_FENCE(SDValue Op, SelectionDAG& DAG) const;
    SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;

    SDValue LowerReturn(SDValue Chain,
                        CallingConv::ID CallConv, bool isVarArg,
                        const SmallVectorImpl<ISD::OutputArg> &Outs,
                        const SmallVectorImpl<SDValue> &OutVals,
                        DebugLoc dl, SelectionDAG &DAG) const;

    virtual MachineBasicBlock
    *EmitInstrWithCustomInserter(MachineInstr *MI,
                                 MachineBasicBlock *BB) const;

    SDValue  LowerVASTART(SDValue Op, SelectionDAG &DAG) const;
    SDValue  LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;
    virtual EVT getSetCCResultType(EVT VT) const {
      return MVT::i1;
    }

    virtual bool getPostIndexedAddressParts(SDNode *N, SDNode *Op,
                                            SDValue &Base, SDValue &Offset,
                                            ISD::MemIndexedMode &AM,
                                            SelectionDAG &DAG) const;

    std::pair<unsigned, const TargetRegisterClass*>
    getRegForInlineAsmConstraint(const std::string &Constraint,
                                 EVT VT) const;

    // Intrinsics
    virtual SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op,
                                            SelectionDAG &DAG) const;
    /// isLegalAddressingMode - Return true if the addressing mode represented
    /// by AM is legal for this target, for a load/store of the specified type.
    /// The type may be VoidTy, in which case only return true if the addressing
    /// mode is legal for a load/store of any legal type.
    /// TODO: Handle pre/postinc as well.
    virtual bool isLegalAddressingMode(const AddrMode &AM, Type *Ty) const;
    virtual bool isFPImmLegal(const APFloat &Imm, EVT VT) const;

    /// isLegalICmpImmediate - Return true if the specified immediate is legal
    /// icmp immediate, that is the target has icmp instructions which can
    /// compare a register against the immediate without having to materialize
    /// the immediate into a register.
    virtual bool isLegalICmpImmediate(int64_t Imm) const;
  };
} // end namespace llvm

#endif    // Hexagon_ISELLOWERING_H
