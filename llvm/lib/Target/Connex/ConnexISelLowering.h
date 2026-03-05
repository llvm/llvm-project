//===-- ConnexISelLowering.h - Connex DAG Lowering Interface ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the interfaces that Connex uses to lower LLVM code into a
/// selection DAG.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CONNEX_CONNEXISELLOWERING_H
#define LLVM_LIB_TARGET_CONNEX_CONNEXISELLOWERING_H

#include "Connex.h"
#include "ConnexConfig.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {
class ConnexSubtarget;

namespace ConnexISD {
/*
From http://llvm.org/docs/doxygen/html/namespacellvm_1_1ISD.html:
  <<Targets may also define target-dependent operator codes for SDNodes.
  For example, on x86, these are the enum values in the X86ISD namespace.
   Targets should aim to use target-independent operators to model their
     instruction sets as much as possible, and only use target-dependent
      operators when they have special requirements.
  Finally, during and after selection proper, SNodes may use special operator
    codes that correspond directly with MachineInstr opcodes.
   These are used to represent selected instructions.
   See the isMachineOpcode() and getMachineOpcode() member functions of
     SDNode.>>
*/
enum NodeType : unsigned {
  FIRST_NUMBER = ISD::BUILTIN_OP_END,
  RET_FLAG,
  CALL,
  SELECT_CC,
  BR_CC,

  /* Inspired from lib/Target/X86/X86ISelLowering.h
  /// A wrapper node for TargetConstantPool,
  /// TargetExternalSymbol, and TargetGlobalAddress.
  */
  Wrapper,

  // From llvm/lib/Target/Mips/MipsISelLowering.h
  // Extended vector element extraction
  VEXTRACT_SEXT_ELT,
  VEXTRACT_ZEXT_ELT,

  // ConstantPool,

  // Vector Shuffle with mask as an operand
  VSHF,  // Generic shuffle
  SHF,   // 4-element set shuffle.
  ILVEV, // Interleave even elements
  ILVOD, // Interleave odd elements
  ILVL,  // Interleave left elements
  ILVR,  // Interleave right elements
  PCKEV, // Pack even elements
  PCKOD, // Pack odd elements
};
} // end namespace ConnexISD

class ConnexTargetLowering : public TargetLowering {
public:
  explicit ConnexTargetLowering(const TargetMachine &TM,
                                const ConnexSubtarget &STI);

  SDValue LowerConstantPool(SDValue Op, SelectionDAG &DAG) const;

  // Inspired from lib/Target/AMDGPU/AMDGPUISelLowering.h
  SDValue LowerDYNAMIC_STACKALLOC(SDValue Op, SelectionDAG &DAG) const;

  // Provide custom lowering hooks for some operations.
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

  // This method returns the name of a target specific DAG node.
  const char *getTargetNodeName(unsigned Opcode) const override;

  MachineBasicBlock *
  EmitInstrWithCustomInserter(MachineInstr &MI,
                              MachineBasicBlock *BB) const override;

private:
  /*
  // From llvm/lib/Target/Mips/MipsISelLowering.h
  // Create a TargetGlobalAddress node.
  SDValue getTargetNode(GlobalAddressSDNode *N, EVT Ty, SelectionDAG &DAG,
                        unsigned Flag) const;

  // Create a TargetExternalSymbol node.
  SDValue getTargetNode(ExternalSymbolSDNode *N, EVT Ty, SelectionDAG &DAG,
                        unsigned Flag) const;

  // Create a TargetBlockAddress node.
  SDValue getTargetNode(BlockAddressSDNode *N, EVT Ty, SelectionDAG &DAG,
                        unsigned Flag) const;

  // Create a TargetJumpTable node.
  SDValue getTargetNode(JumpTableSDNode *N, EVT Ty, SelectionDAG &DAG,
                        unsigned Flag) const;
  */
  // Create a TargetConstantPool node.
  SDValue getTargetNode(ConstantPoolSDNode *N, EVT Ty, SelectionDAG &DAG,
                        unsigned Flag) const;

  // Added from lib/Target/Mips/MipsSEISelLowering.cpp (method addMSAIntType)
  void addVectorIntType(MVT::SimpleValueType Ty, const TargetRegisterClass *RC);

  // Inspired from lib/Target/Mips/MipsSEISelLowering.cpp, addMSAFloatType()
  void addVectorFloatType(MVT::SimpleValueType Ty,
                          const TargetRegisterClass *RC);

  bool allowsMisalignedMemoryAccesses(EVT VT, unsigned, unsigned,
                                      bool *Fast) const;

  void replaceAddI32UseWithADDVH(MVT &aType, SDValue &Index,
                                 SelectionDAG &DAG) const;

  SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;
  /*static */ SDValue LowerMGATHER(SDValue &Op,
                                   // const ConnexSubtarget &Subtarget,
                                   SelectionDAG &DAG) const;
  /*static */ SDValue LowerMSCATTER(SDValue &Op,
                                    // const ConnexSubtarget &Subtarget,
                                    SelectionDAG &DAG) const;

  // Lower the result values of a call, copying them out of physregs into vregs
  SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                          CallingConv::ID CallConv, bool IsVarArg,
                          const SmallVectorImpl<ISD::InputArg> &Ins,
                          const SDLoc &DL, SelectionDAG &DAG,
                          SmallVectorImpl<SDValue> &InVals) const;

  // Maximum number of arguments to a call
  static const unsigned MaxArgs;

  // Lower a call into CALLSEQ_START - ConnexISD:CALL - CALLSEQ_END chain
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

  // Inspired from lib/Target/BPF/BPFISelLowering.h (from Oct 2025)
  EVT getOptimalMemOpType(LLVMContext &Context, const MemOp &Op,
                          const AttributeList &FuncAttributes) const override {
#define DEBUG_TYPE "connex-lower"

    LLVM_DEBUG(dbgs() << "Entered getOptimalMemOpType(): Op.size() = "
                      << Op.size() << ")\n");

    // return Size >= 8 ? MVT::i64 : MVT::i32;
    // Inspired from lib/Target/BPF/BPFISelLowering.h
    return Op.size() >= 8 ? MVT::i64 : MVT::i32;

    // TODO_CHANGE_BACKEND - Seems it's NOT required:
    // return Size >= 8 ? TYPE_VECTOR_ELEMENT : MVT::i32;

#undef DEBUG_TYPE
  }

  bool shouldConvertConstantLoadToIntImm(const APInt &Imm,
                                         Type *Ty) const override {
    return true;
  }

  SDValue LowerVSELECT(SDValue &Op, SelectionDAG &DAG) const;

  // From llvm/lib/Target/Mips/MipsSEISelLowering.h
  SDValue LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerADD_I32(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerADD_F16(SDValue &Op, SelectionDAG *DAG) const;
  SDValue LowerMUL_F16(SDValue &Op, SelectionDAG *DAG) const;
  SDValue LowerREDUCE_F16(SDValue &Op, SelectionDAG *DAG) const;

  SDValue LowerBITCAST(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) const;
  //
  EVT getSetCCResultType(const DataLayout &, LLVMContext &,
                         EVT VT) const override;
}; // end class ConnexTargetLowering
} // end namespace llvm

#endif
