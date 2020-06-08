//==-- llvm/CodeGen/GlobalISel/Utils.h ---------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file declares the API of helper functions used throughout the
/// GlobalISel pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_UTILS_H
#define LLVM_CODEGEN_GLOBALISEL_UTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/LowLevelTypeImpl.h"
#include "llvm/Support/MachineValueType.h"

namespace llvm {

class AnalysisUsage;
class MachineFunction;
class MachineInstr;
class MachineOperand;
class MachineOptimizationRemarkEmitter;
class MachineOptimizationRemarkMissed;
struct MachinePointerInfo;
class MachineRegisterInfo;
class MCInstrDesc;
class RegisterBankInfo;
class TargetInstrInfo;
class TargetPassConfig;
class TargetRegisterInfo;
class TargetRegisterClass;
class Twine;
class ConstantFP;
class APFloat;

/// Try to constrain Reg to the specified register class. If this fails,
/// create a new virtual register in the correct class.
///
/// \return The virtual register constrained to the right register class.
Register constrainRegToClass(MachineRegisterInfo &MRI,
                             const TargetInstrInfo &TII,
                             const RegisterBankInfo &RBI, Register Reg,
                             const TargetRegisterClass &RegClass);

/// Constrain the Register operand OpIdx, so that it is now constrained to the
/// TargetRegisterClass passed as an argument (RegClass).
/// If this fails, create a new virtual register in the correct class and
/// insert a COPY before \p InsertPt if it is a use or after if it is a
/// definition. The debug location of \p InsertPt is used for the new copy.
///
/// \return The virtual register constrained to the right register class.
Register constrainOperandRegClass(const MachineFunction &MF,
                                  const TargetRegisterInfo &TRI,
                                  MachineRegisterInfo &MRI,
                                  const TargetInstrInfo &TII,
                                  const RegisterBankInfo &RBI,
                                  MachineInstr &InsertPt,
                                  const TargetRegisterClass &RegClass,
                                  const MachineOperand &RegMO);

/// Try to constrain Reg so that it is usable by argument OpIdx of the
/// provided MCInstrDesc \p II. If this fails, create a new virtual
/// register in the correct class and insert a COPY before \p InsertPt
/// if it is a use or after if it is a definition.
/// This is equivalent to constrainOperandRegClass(..., RegClass, ...)
/// with RegClass obtained from the MCInstrDesc. The debug location of \p
/// InsertPt is used for the new copy.
///
/// \return The virtual register constrained to the right register class.
Register constrainOperandRegClass(const MachineFunction &MF,
                                  const TargetRegisterInfo &TRI,
                                  MachineRegisterInfo &MRI,
                                  const TargetInstrInfo &TII,
                                  const RegisterBankInfo &RBI,
                                  MachineInstr &InsertPt, const MCInstrDesc &II,
                                  const MachineOperand &RegMO, unsigned OpIdx);

/// Mutate the newly-selected instruction \p I to constrain its (possibly
/// generic) virtual register operands to the instruction's register class.
/// This could involve inserting COPYs before (for uses) or after (for defs).
/// This requires the number of operands to match the instruction description.
/// \returns whether operand regclass constraining succeeded.
///
// FIXME: Not all instructions have the same number of operands. We should
// probably expose a constrain helper per operand and let the target selector
// constrain individual registers, like fast-isel.
bool constrainSelectedInstRegOperands(MachineInstr &I,
                                      const TargetInstrInfo &TII,
                                      const TargetRegisterInfo &TRI,
                                      const RegisterBankInfo &RBI);

/// Check if DstReg can be replaced with SrcReg depending on the register
/// constraints.
bool canReplaceReg(Register DstReg, Register SrcReg, MachineRegisterInfo &MRI);

/// Check whether an instruction \p MI is dead: it only defines dead virtual
/// registers, and doesn't have other side effects.
bool isTriviallyDead(const MachineInstr &MI, const MachineRegisterInfo &MRI);

/// Report an ISel error as a missed optimization remark to the LLVMContext's
/// diagnostic stream.  Set the FailedISel MachineFunction property.
void reportGISelFailure(MachineFunction &MF, const TargetPassConfig &TPC,
                        MachineOptimizationRemarkEmitter &MORE,
                        MachineOptimizationRemarkMissed &R);

void reportGISelFailure(MachineFunction &MF, const TargetPassConfig &TPC,
                        MachineOptimizationRemarkEmitter &MORE,
                        const char *PassName, StringRef Msg,
                        const MachineInstr &MI);

/// Report an ISel warning as a missed optimization remark to the LLVMContext's
/// diagnostic stream.
void reportGISelWarning(MachineFunction &MF, const TargetPassConfig &TPC,
                        MachineOptimizationRemarkEmitter &MORE,
                        MachineOptimizationRemarkMissed &R);

/// If \p VReg is defined by a G_CONSTANT fits in int64_t
/// returns it.
Optional<int64_t> getConstantVRegVal(Register VReg,
                                     const MachineRegisterInfo &MRI);
/// Simple struct used to hold a constant integer value and a virtual
/// register.
struct ValueAndVReg {
  int64_t Value;
  Register VReg;
};
/// If \p VReg is defined by a statically evaluable chain of
/// instructions rooted on a G_F/CONSTANT (\p LookThroughInstrs == true)
/// and that constant fits in int64_t, returns its value as well as the
/// virtual register defined by this G_F/CONSTANT.
/// When \p LookThroughInstrs == false this function behaves like
/// getConstantVRegVal.
/// When \p HandleFConstants == false the function bails on G_FCONSTANTs.
Optional<ValueAndVReg>
getConstantVRegValWithLookThrough(Register VReg, const MachineRegisterInfo &MRI,
                                  bool LookThroughInstrs = true,
                                  bool HandleFConstants = true);
const ConstantFP* getConstantFPVRegVal(Register VReg,
                                       const MachineRegisterInfo &MRI);

/// See if Reg is defined by an single def instruction that is
/// Opcode. Also try to do trivial folding if it's a COPY with
/// same types. Returns null otherwise.
MachineInstr *getOpcodeDef(unsigned Opcode, Register Reg,
                           const MachineRegisterInfo &MRI);

/// Find the def instruction for \p Reg, folding away any trivial copies. Note
/// it may still return a COPY, if it changes the type. May return nullptr if \p
/// Reg is not a generic virtual register.
MachineInstr *getDefIgnoringCopies(Register Reg,
                                   const MachineRegisterInfo &MRI);

/// Find the source register for \p Reg, folding away any trivial copies. It
/// will be an output register of the instruction that getDefIgnoringCopies
/// returns. May return an invalid register if \p Reg is not a generic virtual
/// register.
Register getSrcRegIgnoringCopies(Register Reg,
                                 const MachineRegisterInfo &MRI);

/// Returns an APFloat from Val converted to the appropriate size.
APFloat getAPFloatFromSize(double Val, unsigned Size);

/// Modify analysis usage so it preserves passes required for the SelectionDAG
/// fallback.
void getSelectionDAGFallbackAnalysisUsage(AnalysisUsage &AU);

Optional<APInt> ConstantFoldBinOp(unsigned Opcode, const Register Op1,
                                  const Register Op2,
                                  const MachineRegisterInfo &MRI);

Optional<APInt> ConstantFoldExtOp(unsigned Opcode, const Register Op1,
                                  uint64_t Imm, const MachineRegisterInfo &MRI);

/// Returns true if \p Val can be assumed to never be a NaN. If \p SNaN is true,
/// this returns if \p Val can be assumed to never be a signaling NaN.
bool isKnownNeverNaN(Register Val, const MachineRegisterInfo &MRI,
                     bool SNaN = false);

/// Returns true if \p Val can be assumed to never be a signaling NaN.
inline bool isKnownNeverSNaN(Register Val, const MachineRegisterInfo &MRI) {
  return isKnownNeverNaN(Val, MRI, true);
}

Align inferAlignFromPtrInfo(MachineFunction &MF, const MachinePointerInfo &MPO);

/// Return the least common multiple type of \p Ty0 and \p Ty1, by changing
/// the number of vector elements or scalar bitwidth. The intent is a
/// G_MERGE_VALUES can be constructed from \p Ty0 elements, and unmerged into
/// \p Ty1.
LLT getLCMType(LLT Ty0, LLT Ty1);

/// Return a type that is greatest common divisor of \p OrigTy and \p
/// TargetTy. This will either change the number of vector elements, or
/// bitwidth of scalars. The intent is the result type can be used as the
/// result of a G_UNMERGE_VALUES from \p OrigTy.
LLT getGCDType(LLT OrigTy, LLT TargetTy);

} // End namespace llvm.
#endif
