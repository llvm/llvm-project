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
  /// Debug location to be set to any instruction we create.
  DebugLoc DL;

  /// Fields describing the insertion point.
  /// @{
  MachineBasicBlock *MBB;
  MachineInstr *MI;
  bool Before;
  /// @}

  const TargetInstrInfo &getTII() {
    assert(TII && "TargetInstrInfo is not set");
    return *TII;
  }

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

  /// Set the debug location to \p DL for all the next build instructions.
  void setDebugLoc(const DebugLoc &DL) { this->DL = DL; }

  /// Build and insert <empty> = \p Opcode [ { \p Tys } ] <empty>.
  /// \p Ty is the type of the instruction if \p Opcode describes
  /// a generic machine instruction. \p Ty must be LLT{} if \p Opcode
  /// does not describe a generic instruction.
  /// The insertion point is the one set by the last call of either
  /// setBasicBlock or setMI.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre Ty == LLT{} or isPreISelGenericOpcode(Opcode)
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildInstr(unsigned Opcode, ArrayRef<LLT> Tys);

  /// Build and insert <empty> = \p Opcode <empty>.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre not isPreISelGenericOpcode(\p Opcode)
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildInstr(unsigned Opcode) {
    return buildInstr(Opcode, ArrayRef<LLT>());
  }

  /// Build and insert \p Res<def> = G_FRAME_INDEX \p Ty \p Idx
  ///
  /// G_FRAME_INDEX materializes the address of an alloca value or other
  /// stack-based object.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildFrameIndex(LLT Ty, unsigned Res, int Idx);

  /// Build and insert \p Res<def> = G_ADD \p Ty \p Op0, \p Op1
  ///
  /// G_ADD sets \p Res to the sum of integer parameters \p Op0 and \p Op1,
  /// truncated to their width.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildAdd(LLT Ty, unsigned Res, unsigned Op0,
                                unsigned Op1);

  /// Build and insert \p Res<def>, \p CarryOut = G_UADDE \p Ty \p Op0, \p Op1,
  /// \p CarryIn
  ///
  /// G_UADDE sets \p Res to \p Op0 + \p Op1 + \p CarryIn (truncated to the bit
  /// width) and sets \p CarryOut to 1 if the result overflowed in unsigned
  /// arithmetic.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildUAdde(LLT Ty, unsigned Res, unsigned CarryOut,
                                 unsigned Op0, unsigned Op1, unsigned CarryIn);

  /// Build and insert \p Res<def> = G_ANYEXTEND \p Ty \p Op0
  ///
  /// G_ANYEXTEND produces a register of the specified width, with bits 0 to
  /// sizeof(\p Ty) * 8 set to \p Op. The remaining bits are unspecified
  /// (i.e. this is neither zero nor sign-extension). For a vector register,
  /// each element is extended individually.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildAnyExtend(LLT Ty, unsigned Res, unsigned Op);

  /// Build and insert G_BR unsized \p Dest
  ///
  /// G_BR is an unconditional branch to \p Dest.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildBr(MachineBasicBlock &BB);

  /// Build and insert G_BRCOND \p Ty \p Tst, \p Dest
  ///
  /// G_BRCOND is a conditional branch to \p Dest. At the beginning of
  /// legalization, \p Ty will be a single bit (s1). Targets with interesting
  /// flags registers may change this.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildBrCond(LLT Ty, unsigned Tst, MachineBasicBlock &BB);

  /// Build and insert \p Res = G_CONSTANT \p Ty \p Val
  ///
  /// G_CONSTANT is an integer constant with the specified size and value.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildConstant(LLT Ty, unsigned Res, int64_t Val);

  /// Build and insert \p Res = G_FCONSTANT \p Ty \p Val
  ///
  /// G_FCONSTANT is a floating-point constant with the specified size and
  /// value.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildFConstant(LLT Ty, unsigned Res,
                                     const ConstantFP &Val);

  /// Build and insert \p Res<def> = COPY Op
  ///
  /// Register-to-register COPY sets \p Res to \p Op.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildCopy(unsigned Res, unsigned Op);

  /// Build and insert `Res<def> = G_LOAD { VTy, PTy } Addr, MMO`.
  ///
  /// Loads the value of (sized) type \p VTy stored at \p Addr (in address space
  /// given by \p PTy). Puts the result in Res.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildLoad(LLT VTy, LLT PTy, unsigned Res, unsigned Addr,
                                MachineMemOperand &MMO);

  /// Build and insert `G_STORE { VTy, PTy } Val, Addr, MMO`.
  ///
  /// Stores the value \p Val of (sized) \p VTy to \p Addr (in address space
  /// given by \p PTy).
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildStore(LLT VTy, LLT PTy, unsigned Val, unsigned Addr,
                                 MachineMemOperand &MMO);

  /// Build and insert `Res0<def>, ... = G_EXTRACT { ResTys, SrcTy } Src, Idx0,
  /// ...`.
  ///
  /// If \p SrcTy has size N bits, G_EXTRACT sets \p Res[0] to bits `[Idxs[0],
  /// Idxs[0] + N)` of \p Src and similarly for subsequent bit-indexes.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildExtract(ArrayRef<LLT> ResTys,
                                   ArrayRef<unsigned> Results,
                                   ArrayRef<uint64_t> Indices, LLT SrcTy,
                                   unsigned Src);

  /// Build and insert \p Res<def> = G_SEQUENCE \p { \pResTy, \p Op0Ty, ... }
  /// \p Op0, \p Idx0...
  ///
  /// G_SEQUENCE inserts each element of Ops into an IMPLICIT_DEF register,
  /// where each entry starts at the bit-index specified by \p Indices.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  /// \pre The final element of the sequence must not extend past the end of the
  ///      destination register.
  /// \pre The bits defined by each Op (derived from index and scalar size) must
  ///      not overlap.
  /// \pre Each source operand must have a
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildSequence(LLT ResTy, unsigned Res,
                                    ArrayRef<LLT> OpTys,
                                    ArrayRef<unsigned> Ops,
                                    ArrayRef<unsigned> Indices);

  void addUsesWithIndices(MachineInstrBuilder MIB) {}

  template <typename... ArgTys>
  void addUsesWithIndices(MachineInstrBuilder MIB, LLT Ty, unsigned Reg,
                          unsigned BitIndex, ArgTys... Args) {
    MIB.addUse(Reg).addImm(BitIndex);
    MIB->setType(Ty, MIB->getNumTypes());

    addUsesWithIndices(MIB, Args...);
  }

  template <typename... ArgTys>
  MachineInstrBuilder buildSequence(LLT Ty, unsigned Res, LLT OpTy, unsigned Op,
                                    unsigned Index, ArgTys... Args) {
    MachineInstrBuilder MIB =
        buildInstr(TargetOpcode::G_SEQUENCE, Ty).addDef(Res);
    addUsesWithIndices(MIB, OpTy, Op, Index, Args...);
    return MIB;
  }

  template <typename... ArgTys>
  MachineInstrBuilder buildInsert(LLT Ty, unsigned Res, unsigned Src, LLT OpTy,
                                  unsigned Op, unsigned Index, ArgTys... Args) {
    MachineInstrBuilder MIB =
        buildInstr(TargetOpcode::G_INSERT, Ty).addDef(Res).addUse(Src);
    addUsesWithIndices(MIB, OpTy, Op, Index, Args...);
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
  MachineInstrBuilder buildIntrinsic(ArrayRef<LLT> Tys, Intrinsic::ID ID,
                                     unsigned Res, bool HasSideEffects);

  /// Build and insert \p Res<def> = G_FPTRUNC \p Ty \p Op
  ///
  /// G_FPTRUNC converts a floating-point value into one with a smaller type.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildFPTrunc(LLT Ty, unsigned Res, unsigned Op);

  /// Build and insert \p Res<def> = G_TRUNC \p Ty \p Op
  ///
  /// G_TRUNC extracts the low bits of a type. For a vector type each element is
  /// truncated independently before being packed into the destination.
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return The newly created instruction.
  MachineInstrBuilder buildTrunc(LLT Ty, unsigned Res, unsigned Op);

  /// Build and insert a G_ICMP
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildICmp(ArrayRef<LLT> Tys, CmpInst::Predicate Pred,
                                unsigned Res, unsigned Op0, unsigned Op1);

  /// Build and insert a G_FCMP
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildFCmp(ArrayRef<LLT> Tys, CmpInst::Predicate Pred,
                                unsigned Res, unsigned Op0, unsigned Op1);

  /// Build and insert a \p Res = G_SELECT { \p Ty, s1 } \p Tst, \p Op0, \p Op1
  ///
  /// \pre setBasicBlock or setMI must have been called.
  ///
  /// \return a MachineInstrBuilder for the newly created instruction.
  MachineInstrBuilder buildSelect(LLT Ty, unsigned Res, unsigned Tst,
                                  unsigned Op0, unsigned Op1);
};

} // End namespace llvm.
#endif // LLVM_CODEGEN_GLOBALISEL_MACHINEIRBUILDER_H
