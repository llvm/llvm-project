//===-- FastISel.h - Definition of the FastISel class ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the FastISel class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_FASTISEL_H
#define LLVM_CODEGEN_FASTISEL_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"

namespace llvm {

class AllocaInst;
class Constant;
class ConstantFP;
class FunctionLoweringInfo;
class Instruction;
class LoadInst;
class MachineBasicBlock;
class MachineConstantPool;
class MachineFunction;
class MachineInstr;
class MachineFrameInfo;
class MachineRegisterInfo;
class DataLayout;
class TargetInstrInfo;
class TargetLibraryInfo;
class TargetLowering;
class TargetMachine;
class TargetRegisterClass;
class TargetRegisterInfo;
class User;
class Value;

/// FastISel - This is a fast-path instruction selection class that
/// generates poor code and doesn't support illegal types or non-trivial
/// lowering, but runs quickly.
class FastISel {
protected:
  DenseMap<const Value *, unsigned> LocalValueMap;
  FunctionLoweringInfo &FuncInfo;
  MachineRegisterInfo &MRI;
  MachineFrameInfo &MFI;
  MachineConstantPool &MCP;
  DebugLoc DL;
  const TargetMachine &TM;
  const DataLayout &TD;
  const TargetInstrInfo &TII;
  const TargetLowering &TLI;
  const TargetRegisterInfo &TRI;
  const TargetLibraryInfo *LibInfo;

  /// The position of the last instruction for materializing constants
  /// for use in the current block. It resets to EmitStartPt when it
  /// makes sense (for example, it's usually profitable to avoid function
  /// calls between the definition and the use)
  MachineInstr *LastLocalValue;

  /// The top most instruction in the current block that is allowed for
  /// emitting local variables. LastLocalValue resets to EmitStartPt when
  /// it makes sense (for example, on function calls)
  MachineInstr *EmitStartPt;

public:
  /// getLastLocalValue - Return the position of the last instruction
  /// emitted for materializing constants for use in the current block.
  MachineInstr *getLastLocalValue() { return LastLocalValue; }

  /// setLastLocalValue - Update the position of the last instruction
  /// emitted for materializing constants for use in the current block.
  void setLastLocalValue(MachineInstr *I) {
    EmitStartPt = I;
    LastLocalValue = I;
  }

  /// startNewBlock - Set the current block to which generated machine
  /// instructions will be appended, and clear the local CSE map.
  ///
  void startNewBlock();

  /// getCurDebugLoc() - Return current debug location information.
  DebugLoc getCurDebugLoc() const { return DL; }

  /// SelectInstruction - Do "fast" instruction selection for the given
  /// LLVM IR instruction, and append generated machine instructions to
  /// the current block. Return true if selection was successful.
  ///
  bool SelectInstruction(const Instruction *I);

  /// SelectOperator - Do "fast" instruction selection for the given
  /// LLVM IR operator (Instruction or ConstantExpr), and append
  /// generated machine instructions to the current block. Return true
  /// if selection was successful.
  ///
  bool SelectOperator(const User *I, unsigned Opcode);

  /// getRegForValue - Create a virtual register and arrange for it to
  /// be assigned the value for the given LLVM value.
  unsigned getRegForValue(const Value *V);

  /// lookUpRegForValue - Look up the value to see if its value is already
  /// cached in a register. It may be defined by instructions across blocks or
  /// defined locally.
  unsigned lookUpRegForValue(const Value *V);

  /// getRegForGEPIndex - This is a wrapper around getRegForValue that also
  /// takes care of truncating or sign-extending the given getelementptr
  /// index value.
  std::pair<unsigned, bool> getRegForGEPIndex(const Value *V);

  /// TryToFoldLoad - The specified machine instr operand is a vreg, and that
  /// vreg is being provided by the specified load instruction.  If possible,
  /// try to fold the load as an operand to the instruction, returning true if
  /// possible.
  virtual bool TryToFoldLoad(MachineInstr * /*MI*/, unsigned /*OpNo*/,
                             const LoadInst * /*LI*/) {
    return false;
  }

  /// recomputeInsertPt - Reset InsertPt to prepare for inserting instructions
  /// into the current block.
  void recomputeInsertPt();

  struct SavePoint {
    MachineBasicBlock::iterator InsertPt;
    DebugLoc DL;
  };

  /// enterLocalValueArea - Prepare InsertPt to begin inserting instructions
  /// into the local value area and return the old insert position.
  SavePoint enterLocalValueArea();

  /// leaveLocalValueArea - Reset InsertPt to the given old insert position.
  void leaveLocalValueArea(SavePoint Old);

  virtual ~FastISel();

protected:
  explicit FastISel(FunctionLoweringInfo &funcInfo,
                    const TargetLibraryInfo *libInfo);

  /// TargetSelectInstruction - This method is called by target-independent
  /// code when the normal FastISel process fails to select an instruction.
  /// This gives targets a chance to emit code for anything that doesn't
  /// fit into FastISel's framework. It returns true if it was successful.
  ///
  virtual bool
  TargetSelectInstruction(const Instruction *I) = 0;

  /// FastEmit_r - This method is called by target-independent code
  /// to request that an instruction with the given type and opcode
  /// be emitted.
  virtual unsigned FastEmit_(MVT VT,
                             MVT RetVT,
                             unsigned Opcode);

  /// FastEmit_r - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// register operand be emitted.
  ///
  virtual unsigned FastEmit_r(MVT VT,
                              MVT RetVT,
                              unsigned Opcode,
                              unsigned Op0, bool Op0IsKill);

  /// FastEmit_rr - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// register operands be emitted.
  ///
  virtual unsigned FastEmit_rr(MVT VT,
                               MVT RetVT,
                               unsigned Opcode,
                               unsigned Op0, bool Op0IsKill,
                               unsigned Op1, bool Op1IsKill);

  /// FastEmit_ri - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// register and immediate operands be emitted.
  ///
  virtual unsigned FastEmit_ri(MVT VT,
                               MVT RetVT,
                               unsigned Opcode,
                               unsigned Op0, bool Op0IsKill,
                               uint64_t Imm);

  /// FastEmit_rf - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// register and floating-point immediate operands be emitted.
  ///
  virtual unsigned FastEmit_rf(MVT VT,
                               MVT RetVT,
                               unsigned Opcode,
                               unsigned Op0, bool Op0IsKill,
                               const ConstantFP *FPImm);

  /// FastEmit_rri - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// register and immediate operands be emitted.
  ///
  virtual unsigned FastEmit_rri(MVT VT,
                                MVT RetVT,
                                unsigned Opcode,
                                unsigned Op0, bool Op0IsKill,
                                unsigned Op1, bool Op1IsKill,
                                uint64_t Imm);

  /// FastEmit_ri_ - This method is a wrapper of FastEmit_ri. It first tries
  /// to emit an instruction with an immediate operand using FastEmit_ri.
  /// If that fails, it materializes the immediate into a register and try
  /// FastEmit_rr instead.
  unsigned FastEmit_ri_(MVT VT,
                        unsigned Opcode,
                        unsigned Op0, bool Op0IsKill,
                        uint64_t Imm, MVT ImmType);

  /// FastEmit_i - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// immediate operand be emitted.
  virtual unsigned FastEmit_i(MVT VT,
                              MVT RetVT,
                              unsigned Opcode,
                              uint64_t Imm);

  /// FastEmit_f - This method is called by target-independent code
  /// to request that an instruction with the given type, opcode, and
  /// floating-point immediate operand be emitted.
  virtual unsigned FastEmit_f(MVT VT,
                              MVT RetVT,
                              unsigned Opcode,
                              const ConstantFP *FPImm);

  /// FastEmitInst_ - Emit a MachineInstr with no operands and a
  /// result register in the given register class.
  ///
  unsigned FastEmitInst_(unsigned MachineInstOpcode,
                         const TargetRegisterClass *RC);

  /// FastEmitInst_r - Emit a MachineInstr with one register operand
  /// and a result register in the given register class.
  ///
  unsigned FastEmitInst_r(unsigned MachineInstOpcode,
                          const TargetRegisterClass *RC,
                          unsigned Op0, bool Op0IsKill);

  /// FastEmitInst_rr - Emit a MachineInstr with two register operands
  /// and a result register in the given register class.
  ///
  unsigned FastEmitInst_rr(unsigned MachineInstOpcode,
                           const TargetRegisterClass *RC,
                           unsigned Op0, bool Op0IsKill,
                           unsigned Op1, bool Op1IsKill);

  /// FastEmitInst_rrr - Emit a MachineInstr with three register operands
  /// and a result register in the given register class.
  ///
  unsigned FastEmitInst_rrr(unsigned MachineInstOpcode,
                           const TargetRegisterClass *RC,
                           unsigned Op0, bool Op0IsKill,
                           unsigned Op1, bool Op1IsKill,
                           unsigned Op2, bool Op2IsKill);

  /// FastEmitInst_ri - Emit a MachineInstr with a register operand,
  /// an immediate, and a result register in the given register class.
  ///
  unsigned FastEmitInst_ri(unsigned MachineInstOpcode,
                           const TargetRegisterClass *RC,
                           unsigned Op0, bool Op0IsKill,
                           uint64_t Imm);

  /// FastEmitInst_rii - Emit a MachineInstr with one register operand
  /// and two immediate operands.
  ///
  unsigned FastEmitInst_rii(unsigned MachineInstOpcode,
                           const TargetRegisterClass *RC,
                           unsigned Op0, bool Op0IsKill,
                           uint64_t Imm1, uint64_t Imm2);

  /// FastEmitInst_rf - Emit a MachineInstr with two register operands
  /// and a result register in the given register class.
  ///
  unsigned FastEmitInst_rf(unsigned MachineInstOpcode,
                           const TargetRegisterClass *RC,
                           unsigned Op0, bool Op0IsKill,
                           const ConstantFP *FPImm);

  /// FastEmitInst_rri - Emit a MachineInstr with two register operands,
  /// an immediate, and a result register in the given register class.
  ///
  unsigned FastEmitInst_rri(unsigned MachineInstOpcode,
                            const TargetRegisterClass *RC,
                            unsigned Op0, bool Op0IsKill,
                            unsigned Op1, bool Op1IsKill,
                            uint64_t Imm);

  /// FastEmitInst_rrii - Emit a MachineInstr with two register operands,
  /// two immediates operands, and a result register in the given register
  /// class.
  unsigned FastEmitInst_rrii(unsigned MachineInstOpcode,
                             const TargetRegisterClass *RC,
                             unsigned Op0, bool Op0IsKill,
                             unsigned Op1, bool Op1IsKill,
                             uint64_t Imm1, uint64_t Imm2);

  /// FastEmitInst_i - Emit a MachineInstr with a single immediate
  /// operand, and a result register in the given register class.
  unsigned FastEmitInst_i(unsigned MachineInstrOpcode,
                          const TargetRegisterClass *RC,
                          uint64_t Imm);

  /// FastEmitInst_ii - Emit a MachineInstr with a two immediate operands.
  unsigned FastEmitInst_ii(unsigned MachineInstrOpcode,
                          const TargetRegisterClass *RC,
                          uint64_t Imm1, uint64_t Imm2);

  /// FastEmitInst_extractsubreg - Emit a MachineInstr for an extract_subreg
  /// from a specified index of a superregister to a specified type.
  unsigned FastEmitInst_extractsubreg(MVT RetVT,
                                      unsigned Op0, bool Op0IsKill,
                                      uint32_t Idx);

  /// FastEmitZExtFromI1 - Emit MachineInstrs to compute the value of Op
  /// with all but the least significant bit set to zero.
  unsigned FastEmitZExtFromI1(MVT VT,
                              unsigned Op0, bool Op0IsKill);

  /// FastEmitBranch - Emit an unconditional branch to the given block,
  /// unless it is the immediate (fall-through) successor, and update
  /// the CFG.
  void FastEmitBranch(MachineBasicBlock *MBB, DebugLoc DL);

  void UpdateValueMap(const Value* I, unsigned Reg, unsigned NumRegs = 1);

  unsigned createResultReg(const TargetRegisterClass *RC);

  /// TargetMaterializeConstant - Emit a constant in a register using
  /// target-specific logic, such as constant pool loads.
  virtual unsigned TargetMaterializeConstant(const Constant* C) {
    return 0;
  }

  /// TargetMaterializeAlloca - Emit an alloca address in a register using
  /// target-specific logic.
  virtual unsigned TargetMaterializeAlloca(const AllocaInst* C) {
    return 0;
  }

  virtual unsigned TargetMaterializeFloatZero(const ConstantFP* CF) {
    return 0;
  }

private:
  bool SelectBinaryOp(const User *I, unsigned ISDOpcode);

  bool SelectFNeg(const User *I);

  bool SelectGetElementPtr(const User *I);

  bool SelectCall(const User *I);

  bool SelectBitCast(const User *I);

  bool SelectCast(const User *I, unsigned Opcode);

  bool SelectExtractValue(const User *I);

  bool SelectInsertValue(const User *I);

  /// HandlePHINodesInSuccessorBlocks - Handle PHI nodes in successor blocks.
  /// Emit code to ensure constants are copied into registers when needed.
  /// Remember the virtual registers that need to be added to the Machine PHI
  /// nodes as input.  We cannot just directly add them, because expansion
  /// might result in multiple MBB's for one BB.  As such, the start of the
  /// BB might correspond to a different MBB than the end.
  bool HandlePHINodesInSuccessorBlocks(const BasicBlock *LLVMBB);

  /// materializeRegForValue - Helper for getRegForVale. This function is
  /// called when the value isn't already available in a register and must
  /// be materialized with new instructions.
  unsigned materializeRegForValue(const Value *V, MVT VT);

  /// flushLocalValueMap - clears LocalValueMap and moves the area for the
  /// new local variables to the beginning of the block. It helps to avoid
  /// spilling cached variables across heavy instructions like calls.
  void flushLocalValueMap();

  /// hasTrivialKill - Test whether the given value has exactly one use.
  bool hasTrivialKill(const Value *V) const;

  /// removeDeadCode - Remove all dead instructions between the I and E.
  void removeDeadCode(MachineBasicBlock::iterator I,
                      MachineBasicBlock::iterator E);
};

}

#endif
