//==-- AArch64ISelLowering.h - AArch64 DAG Lowering Interface ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that AArch64 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_AARCH64_ISELLOWERING_H
#define LLVM_TARGET_AARCH64_ISELLOWERING_H

#include "Utils/AArch64BaseInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/IR/Intrinsics.h"

namespace llvm {
namespace AArch64ISD {
  enum NodeType {
    // Start the numbering from where ISD NodeType finishes.
    FIRST_NUMBER = ISD::BUILTIN_OP_END,

    // This is a conditional branch which also notes the flag needed
    // (eq/sgt/...). A64 puts this information on the branches rather than
    // compares as LLVM does.
    BR_CC,

    // A node to be selected to an actual call operation: either BL or BLR in
    // the absence of tail calls.
    Call,

    // Indicates a floating-point immediate which fits into the format required
    // by the FMOV instructions. First (and only) operand is the 8-bit encoded
    // value of that immediate.
    FPMOV,

    // Corresponds directly to an EXTR instruction. Operands are an LHS an RHS
    // and an LSB.
    EXTR,

    // Wraps a load from the GOT, which should always be performed with a 64-bit
    // load instruction. This prevents the DAG combiner folding a truncate to
    // form a smaller memory access.
    GOTLoad,

    // Performs a bitfield insert. Arguments are: the value being inserted into;
    // the value being inserted; least significant bit changed; width of the
    // field.
    BFI,

    // Simply a convenient node inserted during ISelLowering to represent
    // procedure return. Will almost certainly be selected to "RET".
    Ret,

    /// Extracts a field of contiguous bits from the source and sign extends
    /// them into a single register. Arguments are: source; immr; imms. Note
    /// these are pre-encoded since DAG matching can't cope with combining LSB
    /// and Width into these values itself.
    SBFX,

    /// This is an A64-ification of the standard LLVM SELECT_CC operation. The
    /// main difference is that it only has the values and an A64 condition,
    /// which will be produced by a setcc instruction.
    SELECT_CC,

    /// This serves most of the functions of the LLVM SETCC instruction, for two
    /// purposes. First, it prevents optimisations from fiddling with the
    /// compare after we've moved the CondCode information onto the SELECT_CC or
    /// BR_CC instructions. Second, it gives a legal instruction for the actual
    /// comparison.
    ///
    /// It keeps a record of the condition flags asked for because certain
    /// instructions are only valid for a subset of condition codes.
    SETCC,

    // Designates a node which is a tail call: both a call and a return
    // instruction as far as selction is concerned. It should be selected to an
    // unconditional branch. Has the usual plethora of call operands, but: 1st
    // is callee, 2nd is stack adjustment required immediately before branch.
    TC_RETURN,

    // Designates a call used to support the TLS descriptor ABI. The call itself
    // will be indirect ("BLR xN") but a relocation-specifier (".tlsdesccall
    // var") must be attached somehow during code generation. It takes two
    // operands: the callee and the symbol to be relocated against.
    TLSDESCCALL,

    // Leaf node which will be lowered to an appropriate MRS to obtain the
    // thread pointer: TPIDR_EL0.
    THREAD_POINTER,

    /// Extracts a field of contiguous bits from the source and zero extends
    /// them into a single register. Arguments are: source; immr; imms. Note
    /// these are pre-encoded since DAG matching can't cope with combining LSB
    /// and Width into these values itself.
    UBFX,

    // Wraps an address which the ISelLowering phase has decided should be
    // created using the large memory model style: i.e. a sequence of four
    // movz/movk instructions.
    WrapperLarge,

    // Wraps an address which the ISelLowering phase has decided should be
    // created using the small memory model style: i.e. adrp/add or
    // adrp/mem-op. This exists to prevent bare TargetAddresses which may never
    // get selected.
    WrapperSmall,

    // Vector bitwise select
    NEON_BSL,

    // Vector move immediate
    NEON_MOVIMM,

    // Vector Move Inverted Immediate
    NEON_MVNIMM,

    // Vector FP move immediate
    NEON_FMOVIMM,

    // Vector permute
    NEON_UZP1,
    NEON_UZP2,
    NEON_ZIP1,
    NEON_ZIP2,
    NEON_TRN1,
    NEON_TRN2,

    // Vector Element reverse
    NEON_REV64,
    NEON_REV32,
    NEON_REV16,

    // Vector compare
    NEON_CMP,

    // Vector compare zero
    NEON_CMPZ,

    // Vector compare bitwise test
    NEON_TST,

    // Vector saturating shift
    NEON_QSHLs,
    NEON_QSHLu,

    // Vector dup
    NEON_VDUP,

    // Vector dup by lane
    NEON_VDUPLANE,

    // Vector extract
    NEON_VEXTRACT,

    // NEON duplicate lane loads
    NEON_LD2DUP = ISD::FIRST_TARGET_MEMORY_OPCODE,
    NEON_LD3DUP,
    NEON_LD4DUP,

    // NEON loads with post-increment base updates:
    NEON_LD1_UPD,
    NEON_LD2_UPD,
    NEON_LD3_UPD,
    NEON_LD4_UPD,
    NEON_LD1x2_UPD,
    NEON_LD1x3_UPD,
    NEON_LD1x4_UPD,

    // NEON stores with post-increment base updates:
    NEON_ST1_UPD,
    NEON_ST2_UPD,
    NEON_ST3_UPD,
    NEON_ST4_UPD,
    NEON_ST1x2_UPD,
    NEON_ST1x3_UPD,
    NEON_ST1x4_UPD,

    // NEON duplicate lane loads with post-increment base updates:
    NEON_LD2DUP_UPD,
    NEON_LD3DUP_UPD,
    NEON_LD4DUP_UPD,

    // NEON lane loads with post-increment base updates:
    NEON_LD2LN_UPD,
    NEON_LD3LN_UPD,
    NEON_LD4LN_UPD,

    // NEON lane store with post-increment base updates:
    NEON_ST2LN_UPD,
    NEON_ST3LN_UPD,
    NEON_ST4LN_UPD
  };
}


class AArch64Subtarget;
class AArch64TargetMachine;

class AArch64TargetLowering : public TargetLowering {
public:
  explicit AArch64TargetLowering(AArch64TargetMachine &TM);

  const char *getTargetNodeName(unsigned Opcode) const;

  CCAssignFn *CCAssignFnForNode(CallingConv::ID CC) const;

  SDValue LowerFormalArguments(SDValue Chain,
                               CallingConv::ID CallConv, bool isVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               SDLoc dl, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const;

  SDValue LowerReturn(SDValue Chain,
                      CallingConv::ID CallConv, bool isVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals,
                      SDLoc dl, SelectionDAG &DAG) const;

  SDValue LowerCall(CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const;

  SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                          CallingConv::ID CallConv, bool IsVarArg,
                          const SmallVectorImpl<ISD::InputArg> &Ins,
                          SDLoc dl, SelectionDAG &DAG,
                          SmallVectorImpl<SDValue> &InVals) const;

  bool isKnownShuffleVector(SDValue Op, SelectionDAG &DAG, SDValue &Res) const;

  SDValue LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG,
                            const AArch64Subtarget *ST) const;

  SDValue LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) const;

  void SaveVarArgRegisters(CCState &CCInfo, SelectionDAG &DAG, SDLoc DL,
                           SDValue &Chain) const;

  /// IsEligibleForTailCallOptimization - Check whether the call is eligible
  /// for tail call optimization. Targets which want to do tail call
  /// optimization should implement this function.
  bool IsEligibleForTailCallOptimization(SDValue Callee,
                                    CallingConv::ID CalleeCC,
                                    bool IsVarArg,
                                    bool IsCalleeStructRet,
                                    bool IsCallerStructRet,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const SmallVectorImpl<SDValue> &OutVals,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                    SelectionDAG& DAG) const;

  /// Finds the incoming stack arguments which overlap the given fixed stack
  /// object and incorporates their load into the current chain. This prevents
  /// an upcoming store from clobbering the stack argument before it's used.
  SDValue addTokenForArgument(SDValue Chain, SelectionDAG &DAG,
                              MachineFrameInfo *MFI, int ClobberedFI) const;

  EVT getSetCCResultType(LLVMContext &Context, EVT VT) const;

  bool DoesCalleeRestoreStack(CallingConv::ID CallCC, bool TailCallOpt) const;

  bool IsTailCallConvention(CallingConv::ID CallCC) const;

  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

  bool isLegalICmpImmediate(int64_t Val) const;
  SDValue getSelectableIntSetCC(SDValue LHS, SDValue RHS, ISD::CondCode CC,
                         SDValue &A64cc, SelectionDAG &DAG, SDLoc &dl) const;

  virtual MachineBasicBlock *
  EmitInstrWithCustomInserter(MachineInstr *MI, MachineBasicBlock *MBB) const;

  MachineBasicBlock *
  emitAtomicBinary(MachineInstr *MI, MachineBasicBlock *MBB,
                   unsigned Size, unsigned Opcode) const;

  MachineBasicBlock *
  emitAtomicBinaryMinMax(MachineInstr *MI, MachineBasicBlock *BB,
                         unsigned Size, unsigned CmpOp,
                         A64CC::CondCodes Cond) const;
  MachineBasicBlock *
  emitAtomicCmpSwap(MachineInstr *MI, MachineBasicBlock *BB,
                    unsigned Size) const;

  MachineBasicBlock *
  EmitF128CSEL(MachineInstr *MI, MachineBasicBlock *MBB) const;

  SDValue LowerATOMIC_FENCE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerATOMIC_STORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBRCOND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBR_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerF128ToCall(SDValue Op, SelectionDAG &DAG,
                          RTLIB::Libcall Call) const;
  SDValue LowerFP_EXTEND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFP_ROUND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFP_TO_INT(SDValue Op, SelectionDAG &DAG, bool IsSigned) const;
  SDValue LowerRETURNADDR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFRAMEADDR(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerGlobalAddressELFSmall(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerGlobalAddressELFLarge(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerGlobalAddressELF(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerTLSDescCall(SDValue SymAddr, SDValue DescAddr, SDLoc DL,
                           SelectionDAG &DAG) const;
  SDValue LowerGlobalTLSAddress(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINT_TO_FP(SDValue Op, SelectionDAG &DAG, bool IsSigned) const;
  SDValue LowerJumpTable(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVACOPY(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;

  virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;

  /// isFMAFasterThanFMulAndFAdd - Return true if an FMA operation is faster
  /// than a pair of fmul and fadd instructions. fmuladd intrinsics will be
  /// expanded to FMAs when this method returns true, otherwise fmuladd is
  /// expanded to fmul + fadd.
  virtual bool isFMAFasterThanFMulAndFAdd(EVT VT) const;

  ConstraintType getConstraintType(const std::string &Constraint) const;

  ConstraintWeight getSingleConstraintMatchWeight(AsmOperandInfo &Info,
                                                  const char *Constraint) const;
  void LowerAsmOperandForConstraint(SDValue Op,
                                    std::string &Constraint,
                                    std::vector<SDValue> &Ops,
                                    SelectionDAG &DAG) const;

  std::pair<unsigned, const TargetRegisterClass*>
  getRegForInlineAsmConstraint(const std::string &Constraint, MVT VT) const;

  virtual bool getTgtMemIntrinsic(IntrinsicInfo &Info, const CallInst &I,
                                  unsigned Intrinsic) const LLVM_OVERRIDE;

protected:
  std::pair<const TargetRegisterClass*, uint8_t>
  findRepresentativeClass(MVT VT) const;

private:
  const InstrItineraryData *Itins;

  const AArch64Subtarget *getSubtarget() const {
    return &getTargetMachine().getSubtarget<AArch64Subtarget>();
  }
};
enum NeonModImmType {
  Neon_Mov_Imm,
  Neon_Mvn_Imm
};

extern SDValue ScanBUILD_VECTOR(SDValue Op, bool &isOnlyLowElement,
                                bool &usesOnlyOneValue, bool &hasDominantValue,
                                bool &isConstant, bool &isUNDEF);
} // namespace llvm

#endif // LLVM_TARGET_AARCH64_ISELLOWERING_H
