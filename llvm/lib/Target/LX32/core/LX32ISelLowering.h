//===-- LX32ISelLowering.h - LX32 SelectionDAG Lowering Interface --------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file declares LX32TargetLowering, the class that controls how LLVM IR
// operations are lowered to LX32 SelectionDAG nodes during instruction
// selection.
//
// It is organized into the following sections:
//
//   Section 0 — Role in the backend pipeline
//   Section 1 — Custom DAG node opcodes (LX32ISD)
//   Section 2 — Class declaration and constructor
//   Section 3 — Calling convention lowering
//   Section 4 — Custom operation lowering helpers
//
//===----------------------------------------------------------------------===//
//
// Section 0 — Role in the backend pipeline
//
// The SelectionDAG instruction selection pipeline runs in three stages:
//
//   1. IR → DAG lowering  (this class, via LowerOperation and the CC helpers)
//      Converts LLVM IR intrinsics, calling conventions, and operations that
//      have no direct LX32 instruction into legal SelectionDAG nodes.
//
//   2. DAG → DAG legalization  (controlled by setOperationAction calls in the
//      constructor)
//      Tells the legalizer what to do with each ISD node on LX32:
//        Legal   — the node maps directly to an instruction; leave it alone.
//        Expand  — decompose into simpler nodes (e.g., sdiv → __divsi3 call).
//        Custom  — call LowerOperation to produce a hand-written DAG sequence.
//
//   3. DAG → MachineInstr selection  (LX32ISelDAGToDAG, Day 10)
//      Pattern-matches the legalised DAG against the patterns declared in
//      LX32InstrInfo.td and emits concrete MachineInstrs.
//
// The constructor (Section 2) is where all setOperationAction calls live.
// This is the first thing to implement when adding support for a new
// operation: decide whether it is Legal, Expand, or Custom, then add the
// corresponding entry.
//
// Operations marked Expand that map to runtime library calls (e.g., __divsi3,
// soft-float functions) require no additional implementation — the legalizer
// generates the call automatically using the names registered with the
// RuntimeLibcallsInfo.
//
// Operations marked Custom require a corresponding case in LowerOperation and
// a private helper method (Section 4).
//
//===----------------------------------------------------------------------===//

#ifndef LX32_LX32ISELLOWERING_H
#define LX32_LX32ISELLOWERING_H

#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

class LX32Subtarget;

//===----------------------------------------------------------------------===//
// Section 1 — Custom DAG node opcodes (LX32ISD)
//
// These opcodes identify SelectionDAG nodes that have LX32-specific semantics
// and cannot be expressed using the generic ISD namespace nodes.
//
// Each opcode is used in two places:
//   1. ISelLowering creates nodes with these opcodes during LowerOperation.
//   2. ISelDAGToDAG (Day 10) pattern-matches them to concrete instructions.
//
// The mapping to assembly:
//   LX32ISD::RET    → PseudoRET → JALR x0, ra, 0
//   LX32ISD::CALL   → PseudoCALL (direct) or JALR rs, 0 (indirect)
//
//===----------------------------------------------------------------------===//

namespace LX32ISD {
enum NodeType : unsigned {
  // FIRST_NUMBER marks the start of the LX32-specific range so that the
  // opcodes do not overlap with generic ISD opcodes.
  FIRST_NUMBER = ISD::BUILTIN_OP_END,

  // RET — function return node.
  //
  // Produced by LowerReturn().  Carries an optional glue operand (from
  // CopyToReg nodes that place return values in a0/a1) and a chain.
  // Pattern in LX32InstrInfo.td: [(LX32ret)] → PseudoRET → JALR x0, ra, 0.
  RET,

  // CALL — direct function call node.
  //
  // Produced by LowerCall() for calls to named symbols.  The callee address
  // is the first operand.  Chains and glue operands follow.
  // Lowered in LX32ISelDAGToDAG to PseudoCALL, which expands to
  // AUIPC ra, hi20(sym) + JALR ra, lo12(sym)(ra).
  CALL,

  // SELECT_CC — conditional select with explicit condition code.
  //
  // Produced by LowerSELECT_CC when the legalizer cannot handle ISD::SELECT
  // directly.  Lowered in LX32ISelDAGToDAG to a branch sequence.
  SELECT_CC,
};
} // namespace LX32ISD

//===----------------------------------------------------------------------===//
// Section 2 — Class declaration and constructor
//===----------------------------------------------------------------------===//

class LX32TargetLowering : public TargetLowering {
  const LX32Subtarget &STI;

public:
  // Construct with a reference to the active TargetMachine and subtarget.
  //
  // The constructor body (in LX32ISelLowering.cpp) calls setOperationAction
  // for every LLVM IR operation to declare whether LX32 can handle it
  // directly (Legal), needs the legalizer to decompose it (Expand), or
  // requires a custom DAG sequence (Custom).
  //
  // Key legalisation decisions in LX32 v1:
  //   - No hardware divider: SDIV/UDIV/SREM/UREM → Expand (→ __divsi3 etc.)
  //   - No FPU: all floating-point ops → Expand (→ soft-float library)
  //   - No SELECT instruction: SELECT → Custom (→ branch sequence)
  //   - No flags register: SETCC variants → Custom (→ SLT/SLTU/SUB combos)
  //   - Global addresses: GlobalAddress → Custom (→ AUIPC + ADDI)
  //   - Variadic calls: VASTART → Custom (→ spill of argument registers)
  explicit LX32TargetLowering(const TargetMachine &TM,
                              const LX32Subtarget &STI);

  // getTargetNodeName — return a human-readable name for a LX32ISD opcode.
  //
  // Used by the SelectionDAG printer (llc -view-dag-combine1) and error
  // messages.  Returns nullptr for unknown opcodes.
  const char *getTargetNodeName(unsigned Opcode) const override;

  //===--------------------------------------------------------------------===//
  // Section 3 — Calling convention lowering
  //===--------------------------------------------------------------------===//

  // LowerFormalArguments — lower incoming function arguments.
  //
  // Called once at the start of each function.  Assigns incoming argument
  // values to virtual registers (for register arguments a0-a7) or loads them
  // from the caller's stack (for stack arguments beyond the 8-register limit).
  //
  // The ILP32 calling convention is defined in LX32CallingConv.td as CC_LX32.
  // This function uses CCState to query the convention and emits CopyFromReg
  // or load nodes for each argument.
  //
  // Returns the function's chain value, extended with all argument setup nodes.
  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool IsVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  // LowerCall — lower an outgoing function call.
  //
  // Emits:
  //   1. ADJCALLSTACKDOWN — reserve stack space for call arguments.
  //   2. Argument placement — CopyToReg for register arguments (a0-a7),
  //      stores for stack arguments.
  //   3. LX32ISD::CALL (direct) or JALR (indirect).
  //   4. ADJCALLSTACKUP — release call-argument stack space.
  //   5. CopyFromReg for return values from a0/a1.
  //
  // Uses CC_LX32 and RetCC_LX32 from LX32CallingConv.td.
  SDValue LowerCall(TargetLowering::CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  // LowerReturn — lower the function return sequence.
  //
  // Emits CopyToReg nodes to place return values in a0 (and a1 for i64),
  // then creates the LX32ISD::RET node.  The RetCC_LX32 convention from
  // LX32CallingConv.td governs which registers carry return values.
  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;

  // LowerOperation — dispatch Custom-legalised operations to their handlers.
  //
  // Called by the legalizer for any operation registered as Custom in the
  // constructor.  Dispatches to the appropriate private helper (Section 4).
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

  //===--------------------------------------------------------------------===//
  // Section 4 — Custom operation lowering helpers
  //
  // Each helper implements the Custom lowering for one ISD node type.
  // They are private because only LowerOperation dispatches to them.
  //===--------------------------------------------------------------------===//

private:
  // lowerGlobalAddress — lower ISD::GlobalAddress to AUIPC + ADDI.
  //
  // LX32 uses PC-relative addressing for globals:
  //   AUIPC rd, %pcrel_hi(sym)   — rd = PC + hi20(sym - PC)
  //   ADDI  rd, rd, %pcrel_lo(.) — rd = rd + lo12(sym - PC)
  // The two-instruction sequence is needed because a single 12-bit ADDI
  // cannot reach arbitrary 32-bit addresses.
  SDValue lowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const;

  // lowerBlockAddress — lower ISD::BlockAddress (address of a basic block).
  //
  // Used by computed gotos and jump tables.  Same AUIPC+ADDI sequence as
  // lowerGlobalAddress, but with a block-address relocation.
  SDValue lowerBlockAddress(SDValue Op, SelectionDAG &DAG) const;

  // lowerSELECT — lower ISD::SELECT to a branch sequence.
  //
  // LX32 has no conditional-move instruction.  A SELECT is lowered to:
  //   BEQZ cond, else_bb      — jump if condition is false
  // then_bb:
  //   < use true_val >
  //   J end_bb
  // else_bb:
  //   < use false_val >
  // end_bb:
  //   PHI result, then_bb:true_val, else_bb:false_val
  //
  // This introduces a branch misprediction cost in the common case, but it
  // is correct and simple.  A peephole optimiser could convert some SELECT
  // sequences to arithmetic idioms (e.g., select(a < b, a, b) → MIN via SLT).
  SDValue lowerSELECT(SDValue Op, SelectionDAG &DAG) const;

  // lowerSETCC — lower ISD::SETCC to SLT/SLTU/XOR/ADD combinations.
  //
  // LX32 has no condition-flag register.  Comparison results live in GPRs.
  // The available comparison instructions are SLT (signed) and SLTU (unsigned).
  // All other condition codes are synthesised:
  //
  //   SETEQ  a, b  → (a XOR b) == 0  → XORI (a XOR b), 1    using SLTIU
  //   SETNE  a, b  → (a XOR b) != 0  → SLTU x0, (a XOR b)
  //   SETLE  a, b  → NOT (b < a)     → XORI (SLT b, a), 1
  //   SETGE  a, b  → NOT (a < b)     → XORI (SLT a, b), 1
  //   SETGT  a, b  → b < a           → SLT b, a
  //   ... (unsigned variants use SLTU instead of SLT)
  SDValue lowerSETCC(SDValue Op, SelectionDAG &DAG) const;

  // lowerVASTART — lower ISD::VASTART for variadic function support.
  //
  // At the start of a variadic function, the register arguments that were not
  // consumed by named parameters are spilled to a contiguous area of the
  // frame.  lowerVASTART emits a store of the address of that area into the
  // va_list structure, so that va_arg can walk it.
  SDValue lowerVASTART(SDValue Op, SelectionDAG &DAG) const;
};

} // namespace llvm

#endif // LX32_LX32ISELLOWERING_H
