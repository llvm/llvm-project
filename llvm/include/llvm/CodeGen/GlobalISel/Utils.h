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
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/LowLevelTypeImpl.h"
#include <cstdint>

namespace llvm {

class AnalysisUsage;
class BlockFrequencyInfo;
class GISelKnownBits;
class MachineFunction;
class MachineInstr;
class MachineOperand;
class MachineOptimizationRemarkEmitter;
class MachineOptimizationRemarkMissed;
struct MachinePointerInfo;
class MachineRegisterInfo;
class MCInstrDesc;
class ProfileSummaryInfo;
class RegisterBankInfo;
class TargetInstrInfo;
class TargetLowering;
class TargetPassConfig;
class TargetRegisterInfo;
class TargetRegisterClass;
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
/// If this fails, create a new virtual register in the correct class and insert
/// a COPY before \p InsertPt if it is a use or after if it is a definition.
/// In both cases, the function also updates the register of RegMo. The debug
/// location of \p InsertPt is used for the new copy.
///
/// \return The virtual register constrained to the right register class.
Register constrainOperandRegClass(const MachineFunction &MF,
                                  const TargetRegisterInfo &TRI,
                                  MachineRegisterInfo &MRI,
                                  const TargetInstrInfo &TII,
                                  const RegisterBankInfo &RBI,
                                  MachineInstr &InsertPt,
                                  const TargetRegisterClass &RegClass,
                                  MachineOperand &RegMO);

/// Try to constrain Reg so that it is usable by argument OpIdx of the provided
/// MCInstrDesc \p II. If this fails, create a new virtual register in the
/// correct class and insert a COPY before \p InsertPt if it is a use or after
/// if it is a definition. In both cases, the function also updates the register
/// of RegMo.
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
                                  MachineOperand &RegMO, unsigned OpIdx);

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

/// If \p VReg is defined by a G_CONSTANT, return the corresponding value.
Optional<APInt> getConstantVRegVal(Register VReg,
                                   const MachineRegisterInfo &MRI);

/// If \p VReg is defined by a G_CONSTANT fits in int64_t
/// returns it.
Optional<int64_t> getConstantVRegSExtVal(Register VReg,
                                         const MachineRegisterInfo &MRI);

/// Simple struct used to hold a constant integer value and a virtual
/// register.
struct ValueAndVReg {
  APInt Value;
  Register VReg;
};
/// If \p VReg is defined by a statically evaluable chain of
/// instructions rooted on a G_F/CONSTANT (\p LookThroughInstrs == true)
/// and that constant fits in int64_t, returns its value as well as the
/// virtual register defined by this G_F/CONSTANT.
/// When \p LookThroughInstrs == false this function behaves like
/// getConstantVRegVal.
/// When \p HandleFConstants == false the function bails on G_FCONSTANTs.
/// When \p LookThroughAnyExt == true the function treats G_ANYEXT same as
/// G_SEXT.
Optional<ValueAndVReg>
getConstantVRegValWithLookThrough(Register VReg, const MachineRegisterInfo &MRI,
                                  bool LookThroughInstrs = true,
                                  bool HandleFConstants = true,
                                  bool LookThroughAnyExt = false);
const ConstantFP* getConstantFPVRegVal(Register VReg,
                                       const MachineRegisterInfo &MRI);

/// See if Reg is defined by an single def instruction that is
/// Opcode. Also try to do trivial folding if it's a COPY with
/// same types. Returns null otherwise.
MachineInstr *getOpcodeDef(unsigned Opcode, Register Reg,
                           const MachineRegisterInfo &MRI);

/// Simple struct used to hold a Register value and the instruction which
/// defines it.
struct DefinitionAndSourceRegister {
  MachineInstr *MI;
  Register Reg;
};

/// Find the def instruction for \p Reg, and underlying value Register folding
/// away any copies.
///
/// Also walks through hints such as G_ASSERT_ZEXT.
Optional<DefinitionAndSourceRegister>
getDefSrcRegIgnoringCopies(Register Reg, const MachineRegisterInfo &MRI);

/// Find the def instruction for \p Reg, folding away any trivial copies. May
/// return nullptr if \p Reg is not a generic virtual register.
///
/// Also walks through hints such as G_ASSERT_ZEXT.
MachineInstr *getDefIgnoringCopies(Register Reg,
                                   const MachineRegisterInfo &MRI);

/// Find the source register for \p Reg, folding away any trivial copies. It
/// will be an output register of the instruction that getDefIgnoringCopies
/// returns. May return an invalid register if \p Reg is not a generic virtual
/// register.
///
/// Also walks through hints such as G_ASSERT_ZEXT.
Register getSrcRegIgnoringCopies(Register Reg, const MachineRegisterInfo &MRI);

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

/// Test if the given value is known to have exactly one bit set. This differs
/// from computeKnownBits in that it doesn't necessarily determine which bit is
/// set.
bool isKnownToBeAPowerOfTwo(Register Val, const MachineRegisterInfo &MRI,
                            GISelKnownBits *KnownBits = nullptr);

/// Returns true if \p Val can be assumed to never be a NaN. If \p SNaN is true,
/// this returns if \p Val can be assumed to never be a signaling NaN.
bool isKnownNeverNaN(Register Val, const MachineRegisterInfo &MRI,
                     bool SNaN = false);

/// Returns true if \p Val can be assumed to never be a signaling NaN.
inline bool isKnownNeverSNaN(Register Val, const MachineRegisterInfo &MRI) {
  return isKnownNeverNaN(Val, MRI, true);
}

Align inferAlignFromPtrInfo(MachineFunction &MF, const MachinePointerInfo &MPO);

/// Return a virtual register corresponding to the incoming argument register \p
/// PhysReg. This register is expected to have class \p RC, and optional type \p
/// RegTy. This assumes all references to the register will use the same type.
///
/// If there is an existing live-in argument register, it will be returned.
/// This will also ensure there is a valid copy
Register getFunctionLiveInPhysReg(MachineFunction &MF, const TargetInstrInfo &TII,
                                  MCRegister PhysReg,
                                  const TargetRegisterClass &RC,
                                  LLT RegTy = LLT());

/// Return the least common multiple type of \p OrigTy and \p TargetTy, by changing the
/// number of vector elements or scalar bitwidth. The intent is a
/// G_MERGE_VALUES, G_BUILD_VECTOR, or G_CONCAT_VECTORS can be constructed from
/// \p OrigTy elements, and unmerged into \p TargetTy
LLVM_READNONE
LLT getLCMType(LLT OrigTy, LLT TargetTy);

/// Return a type where the total size is the greatest common divisor of \p
/// OrigTy and \p TargetTy. This will try to either change the number of vector
/// elements, or bitwidth of scalars. The intent is the result type can be used
/// as the result of a G_UNMERGE_VALUES from \p OrigTy, and then some
/// combination of G_MERGE_VALUES, G_BUILD_VECTOR and G_CONCAT_VECTORS (possibly
/// with intermediate casts) can re-form \p TargetTy.
///
/// If these are vectors with different element types, this will try to produce
/// a vector with a compatible total size, but the element type of \p OrigTy. If
/// this can't be satisfied, this will produce a scalar smaller than the
/// original vector elements.
///
/// In the worst case, this returns LLT::scalar(1)
LLVM_READNONE
LLT getGCDType(LLT OrigTy, LLT TargetTy);

/// \returns The splat index of a G_SHUFFLE_VECTOR \p MI when \p MI is a splat.
/// If \p MI is not a splat, returns None.
Optional<int> getSplatIndex(MachineInstr &MI);

/// Returns a scalar constant of a G_BUILD_VECTOR splat if it exists.
Optional<int64_t> getBuildVectorConstantSplat(const MachineInstr &MI,
                                              const MachineRegisterInfo &MRI);

/// Return true if the specified instruction is a G_BUILD_VECTOR or
/// G_BUILD_VECTOR_TRUNC where all of the elements are 0 or undef.
bool isBuildVectorAllZeros(const MachineInstr &MI,
                           const MachineRegisterInfo &MRI);

/// Return true if the specified instruction is a G_BUILD_VECTOR or
/// G_BUILD_VECTOR_TRUNC where all of the elements are ~0 or undef.
bool isBuildVectorAllOnes(const MachineInstr &MI,
                          const MachineRegisterInfo &MRI);

/// Returns true if given the TargetLowering's boolean contents information,
/// the value \p Val contains a true value.
bool isConstTrueVal(const TargetLowering &TLI, int64_t Val, bool IsVector,
                    bool IsFP);

/// Returns an integer representing true, as defined by the
/// TargetBooleanContents.
int64_t getICmpTrueVal(const TargetLowering &TLI, bool IsVector, bool IsFP);

/// Returns true if the given block should be optimized for size.
bool shouldOptForSize(const MachineBasicBlock &MBB, ProfileSummaryInfo *PSI,
                      BlockFrequencyInfo *BFI);
} // End namespace llvm.
#endif
