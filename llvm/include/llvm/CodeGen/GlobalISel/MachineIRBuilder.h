//===-- llvm/CodeGen/GlobalISel/MachineIRBuilder.h - MIBuilder --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares the MachineIRBuilder class.
/// This is a helper class to build MachineInstr.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_MACHINEIRBUILDER_H
#define LLVM_CODEGEN_GLOBALISEL_MACHINEIRBUILDER_H

#include "llvm/CodeGen/GlobalISel/Types.h"

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/IR/DebugLoc.h"

#include <queue>

namespace llvm {

// Forward declarations.
class MachineFunction;
class MachineInstr;
class TargetInstrInfo;

/// Helper class to build MachineInstr.
/// It keeps internally the insertion point and debug location for all
/// the new instructions we want to create.
/// This information can be modify via the related setters.
class MachineIRBuilder {
  /// MachineFunction under construction.
  MachineFunction *MF;
  /// Information used to access the description of the opcodes.
  const TargetInstrInfo *TII;
  /// Information used to verify types are consistent.
  const MachineRegisterInfo *MRI;
  /// Debug location to be set to any instruction we create.
  DebugLoc DL;

  /// Fields describing the insertion point.
  /// @{
  MachineBasicBlock *MBB;
  MachineInstr *MI;
  bool Before;
  /// @}

  std::function<void(MachineInstr *)> InsertedInstr;

  const TargetInstrInfo &getTII() {
    assert(TII && "TargetInstrInfo is not set");
    return *TII;
  }

  void validateTruncExt(unsigned Dst, unsigned Src, bool IsExtend);

public:
  /// Getter for the function we currently build.
  MachineFunction &getMF() {
    assert(MF && "MachineFunction is not set");
    return *MF;
  }

  /// Getter for the basic block we currently build.
  MachineBasicBlock &getMBB() {
    assert(MBB && "MachineBasicBlock is not set");
    return *MBB;
  }

  /// Current insertion point for new instructions.
  MachineBasicBlock::iterator getInsertPt();

  /// Setters for the insertion point.
  /// @{
  /// Set the MachineFunction where to build instructions.
  void setMF(MachineFunction &);

  /// Set the insertion point to the beginning (\p Beginning = true) or end
  /// (\p Beginning = false) of \p MBB.
  /// \pre \p MBB must be contained by getMF().
  void setMBB(MachineBasicBlock &MBB, bool Beginning = false);

  /// Set the insertion point to before (\p Before = true) or after
  /// (\p Before = false) \p MI.
  /// \pre MI must be in getMF().
  void setInstr(MachineInstr &MI, bool Before = true);
  /// @}

  /// Control where instructions we create are recorded (typically for
  /// visiting again later during legalization).
  /// @{
  void recordInsertions(std::function<void(MachineInstr *)> InsertedInstr);
  void stopRecordingInsertions();
  /// @}

  /// Set the debug location to \p DL for all the next build instructions.
  void setDebugLoc(const DebugLoc &DL) { this->DL = DL; }

  /// Build and insert <empty> = \p Opcode <empty>.
  /// The insertion point is the one set by the last call of either
  /// setBasicBlock or setMI.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildInstr(unsigned Opcode);

  /// Build and insert \p Res<def> = G_FRAME_INDEX \p Idx
  ///
  /// G_FRAME_INDEX materializes the address of an alloca value or other
  /// stack-based object.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with pointer type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildFrameIndex(unsigned Res, int Idx);

  /// Build and insert \p Res<def> = G_ADD \p Op0, \p Op1
  ///
  /// G_ADD sets \p Res to the sum of integer parameters \p Op0 and \p Op1,
  /// truncated to their width.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res, \p Op0 and \p Op1 must be generic virtual registers
  ///      with the same (scalar or vector) type).
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAdd(unsigned Res, unsigned Op0,
                               unsigned Op1);

  /// Build and insert \p Res<def> = G_SUB \p Op0, \p Op1
  ///
  /// G_SUB sets \p Res to the sum of integer parameters \p Op0 and \p Op1,
  /// truncated to their width.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res, \p Op0 and \p Op1 must be generic virtual registers
  ///      with the same (scalar or vector) type).
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildSub(unsigned Res, unsigned Op0,
                               unsigned Op1);

  /// Build and insert \p Res<def> = G_MUL \p Op0, \p Op1
  ///
  /// G_MUL sets \p Res to the sum of integer parameters \p Op0 and \p Op1,
  /// truncated to their width.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res, \p Op0 and \p Op1 must be generic virtual registers
  ///      with the same (scalar or vector) type).
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildMul(unsigned Res, unsigned Op0,
                               unsigned Op1);

  /// Build and insert \p Res<def>, \p CarryOut<def> = G_UADDE \p Op0,
  /// \p Op1, \p CarryIn
  ///
  /// G_UADDE sets \p Res to \p Op0 + \p Op1 + \p CarryIn (truncated to the bit
  /// width) and sets \p CarryOut to 1 if the result overflowed in unsigned
  /// arithmetic.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res, \p Op0 and \p Op1 must be generic virtual registers
  ///      with the same scalar type.
  /// \pre \p CarryOut and \p CarryIn must be generic virtual
  ///      registers with the same scalar type (typically s1)
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildUAdde(unsigned Res, unsigned CarryOut, unsigned Op0,
                                 unsigned Op1, unsigned CarryIn);

  /// Build and insert \p Res<def> = G_ANYEXT \p Op0
  ///
  /// G_ANYEXT produces a register of the specified width, with bits 0 to
  /// sizeof(\p Ty) * 8 set to \p Op. The remaining bits are unspecified
  /// (i.e. this is neither zero nor sign-extension). For a vector register,
  /// each element is extended individually.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be smaller than \p Res
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildAnyExt(unsigned Res, unsigned Op);

  /// Build and insert \p Res<def> = G_SEXT \p Op
  ///
  /// G_SEXT produces a register of the specified width, with bits 0 to
  /// sizeof(\p Ty) * 8 set to \p Op. The remaining bits are duplicated from the
  /// high bit of \p Op (i.e. 2s-complement sign extended).
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be smaller than \p Res
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildSExt(unsigned Res, unsigned Op);

  /// Build and insert \p Res<def> = G_ZEXT \p Op
  ///
  /// G_ZEXT produces a register of the specified width, with bits 0 to
  /// sizeof(\p Ty) * 8 set to \p Op. The remaining bits are 0. For a vector
  /// register, each element is extended individually.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be smaller than \p Res
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildZExt(unsigned Res, unsigned Op);

  /// Build and insert G_BR \p Dest
  ///
  /// G_BR is an unconditional branch to \p Dest.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildBr(MachineBasicBlock &BB);

  /// Build and insert G_BRCOND \p Tst, \p Dest
  ///
  /// G_BRCOND is a conditional branch to \p Dest.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Tst must be a generic virtual register with scalar
  ///      type. At the beginning of legalization, this will be a single
  ///      bit (s1). Targets with interesting flags registers may change
  ///      this. For a wider type, whether the branch is taken must only
  ///      depend on bit 0 (for now).
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildBrCond(unsigned Tst, MachineBasicBlock &BB);

  /// Build and insert \p Res = G_CONSTANT \p Val
  ///
  /// G_CONSTANT is an integer constant with the specified size and value.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar type.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildConstant(unsigned Res, int64_t Val);

  /// Build and insert \p Res = G_FCONSTANT \p Val
  ///
  /// G_FCONSTANT is a floating-point constant with the specified size and
  /// value.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar type.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildFConstant(unsigned Res, const ConstantFP &Val);

  /// Build and insert \p Res<def> = COPY Op
  ///
  /// Register-to-register COPY sets \p Res to \p Op.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildCopy(unsigned Res, unsigned Op);

  /// Build and insert `Res<def> = G_LOAD Addr, MMO`.
  ///
  /// Loads the value stored at \p Addr. Puts the result in \p Res.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildLoad(unsigned Res, unsigned Addr,
                                MachineMemOperand &MMO);

  /// Build and insert `G_STORE Val, Addr, MMO`.
  ///
  /// Stores the value \p Val to \p Addr.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Val must be a generic virtual register.
  /// \pre \p Addr must be a generic virtual register with pointer type.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildStore(unsigned Val, unsigned Addr,
                                 MachineMemOperand &MMO);

  /// Build and insert `Res0<def>, ... = G_EXTRACT Src, Idx0, ...`.
  ///
  /// If \p Res[i] has size N bits, G_EXTRACT sets \p Res[i] to bits `[Idxs[i],
  /// Idxs[i] + N)` of \p Src.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre Indices must be in ascending order of bit position.
  /// \pre Each member of \p Results and \p Src must be a generic
  ///      virtual register.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildExtract(ArrayRef<unsigned> Results,
                                   ArrayRef<uint64_t> Indices, unsigned Src);

  /// Build and insert \p Res<def> = G_SEQUENCE \p Op0, \p Idx0...
  ///
  /// G_SEQUENCE inserts each element of Ops into an IMPLICIT_DEF register,
  /// where each entry starts at the bit-index specified by \p Indices.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre The final element of the sequence must not extend past the end of the
  ///      destination register.
  /// \pre The bits defined by each Op (derived from index and scalar size) must
  ///      not overlap.
  /// \pre \p Indices must be in ascending order of bit position.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildSequence(unsigned Res,
                                    ArrayRef<unsigned> Ops,
                                    ArrayRef<unsigned> Indices);

  void addUsesWithIndices(MachineInstrBuilder MIB) {}

  template <typename... ArgTys>
  void addUsesWithIndices(MachineInstrBuilder MIB, unsigned Reg,
                          unsigned BitIndex, ArgTys... Args) {
    MIB.addUse(Reg).addImm(BitIndex);
    addUsesWithIndices(MIB, Args...);
  }

  template <typename... ArgTys>
  MachineInstrBuilder buildSequence(unsigned Res, unsigned Op,
                                    unsigned Index, ArgTys... Args) {
    MachineInstrBuilder MIB =
        buildInstr(TargetOpcode::G_SEQUENCE).addDef(Res);
    addUsesWithIndices(MIB, Op, Index, Args...);
    return MIB;
  }

  template <typename... ArgTys>
  MachineInstrBuilder buildInsert(unsigned Res, unsigned Src,
                                  unsigned Op, unsigned Index, ArgTys... Args) {
    MachineInstrBuilder MIB =
        buildInstr(TargetOpcode::G_INSERT).addDef(Res).addUse(Src);
    addUsesWithIndices(MIB, Op, Index, Args...);
    return MIB;
  }

  /// Build and insert either a G_INTRINSIC (if \p HasSideEffects is false) or
  /// G_INTRINSIC_W_SIDE_EFFECTS instruction. Its first operand will be the
  /// result register definition unless \p Reg is NoReg (== 0). The second
  /// operand will be the intrinsic's ID.
  ///
  /// Callers are expected to add the required definitions and uses afterwards.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildIntrinsic(Intrinsic::ID ID, unsigned Res,
                                     bool HasSideEffects);

  /// Build and insert \p Res<def> = G_FPTRUNC \p Op
  ///
  /// G_FPTRUNC converts a floating-point value into one with a smaller type.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be a generic virtual register with scalar or vector type.
  /// \pre \p Res must be smaller than \p Op
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildFPTrunc(unsigned Res, unsigned Op);

  /// Build and insert \p Res<def> = G_TRUNC \p Op
  ///
  /// G_TRUNC extracts the low bits of a type. For a vector type each element is
  /// truncated independently before being packed into the destination.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res must be a generic virtual register with scalar or vector type.
  /// \pre \p Op must be a generic virtual register with scalar or vector type.
  /// \pre \p Res must be smaller than \p Op
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildTrunc(unsigned Res, unsigned Op);

  /// Build and insert a \p Res = G_ICMP \p Pred, \p Op0, \p Op1
  ///
  /// \pre setBasicBlock or setMI must have been called.

  /// \pre \p Res must be a generic virtual register with scalar or
  ///      vector type. Typically this starts as s1 or <N x s1>.
  /// \pre \p Op0 and Op1 must be generic virtual registers with the
  ///      same number of elements as \p Res (or scalar, if \p Res is
  ///      scalar).
  /// \pre \p Pred must be an integer predicate.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildICmp(CmpInst::Predicate Pred,
                                unsigned Res, unsigned Op0, unsigned Op1);

  /// Build and insert a \p Res = G_FCMP \p Pred\p Op0, \p Op1
  ///
  /// \pre setBasicBlock or setMI must have been called.

  /// \pre \p Res must be a generic virtual register with scalar or
  ///      vector type. Typically this starts as s1 or <N x s1>.
  /// \pre \p Op0 and Op1 must be generic virtual registers with the
  ///      same number of elements as \p Res (or scalar, if \p Res is
  ///      scalar).
  /// \pre \p Pred must be a floating-point predicate.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildFCmp(CmpInst::Predicate Pred,
                                unsigned Res, unsigned Op0, unsigned Op1);

  /// Build and insert a \p Res = G_SELECT \p Tst, \p Op0, \p Op1
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre \p Res, \p Op0 and \p Op1 must be generic virtual registers
  ///      with the same type.
  /// \pre \p Tst must be a generic virtual register with scalar or
  ///      vector type. If vector then it must have the same number of
  ///      elements as the other parameters.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildSelect(unsigned Res, unsigned Tst,
                                  unsigned Op0, unsigned Op1);
};

} // End namespace llvm.
#endif // LLVM_CODEGEN_GLOBALISEL_MACHINEIRBUILDER_H
