//===- llvm/CodeGen/GlobalISel/GIMatchTableExecutor.h -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file declares the GIMatchTableExecutor API, the opcodes supported
/// by the match table, and some associated data structures used by the
/// executor's implementation (see `GIMatchTableExecutorImpl.h`).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_GIMATCHTABLEEXECUTOR_H
#define LLVM_CODEGEN_GLOBALISEL_GIMATCHTABLEEXECUTOR_H

#include "llvm/ADT/Bitset.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Function.h"
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <optional>
#include <vector>

namespace llvm {

class BlockFrequencyInfo;
class CodeGenCoverage;
class MachineBasicBlock;
class ProfileSummaryInfo;
class APInt;
class APFloat;
class GISelKnownBits;
class MachineInstr;
class MachineInstrBuilder;
class MachineFunction;
class MachineOperand;
class MachineRegisterInfo;
class RegisterBankInfo;
class TargetInstrInfo;
class TargetRegisterInfo;

enum {
  GICXXPred_Invalid = 0,
  GICXXCustomAction_Invalid = 0,
};

enum {
  /// Begin a try-block to attempt a match and jump to OnFail if it is
  /// unsuccessful.
  /// - OnFail - The MatchTable entry at which to resume if the match fails.
  ///
  /// FIXME: This ought to take an argument indicating the number of try-blocks
  ///        to exit on failure. It's usually one but the last match attempt of
  ///        a block will need more. The (implemented) alternative is to tack a
  ///        GIM_Reject on the end of each try-block which is simpler but
  ///        requires an extra opcode and iteration in the interpreter on each
  ///        failed match.
  GIM_Try,

  /// Switch over the opcode on the specified instruction
  /// - InsnID - Instruction ID
  /// - LowerBound - numerically minimum opcode supported
  /// - UpperBound - numerically maximum + 1 opcode supported
  /// - Default - failure jump target
  /// - JumpTable... - (UpperBound - LowerBound) (at least 2) jump targets
  GIM_SwitchOpcode,

  /// Switch over the LLT on the specified instruction operand
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - LowerBound - numerically minimum Type ID supported
  /// - UpperBound - numerically maximum + 1 Type ID supported
  /// - Default - failure jump target
  /// - JumpTable... - (UpperBound - LowerBound) (at least 2) jump targets
  GIM_SwitchType,

  /// Record the specified instruction.
  /// The IgnoreCopies variant ignores COPY instructions.
  /// - NewInsnID - Instruction ID to define
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  GIM_RecordInsn,
  GIM_RecordInsnIgnoreCopies,

  /// Check the feature bits
  /// - Expected features
  GIM_CheckFeatures,

  /// Check the opcode on the specified instruction
  /// - InsnID - Instruction ID
  /// - Expected opcode
  GIM_CheckOpcode,

  /// Check the opcode on the specified instruction, checking 2 acceptable
  /// alternatives.
  /// - InsnID - Instruction ID
  /// - Expected opcode
  /// - Alternative expected opcode
  GIM_CheckOpcodeIsEither,

  /// Check the instruction has the right number of operands
  /// - InsnID - Instruction ID
  /// - Expected number of operands
  GIM_CheckNumOperands,
  /// Check an immediate predicate on the specified instruction
  /// - InsnID - Instruction ID
  /// - The predicate to test
  GIM_CheckI64ImmPredicate,
  /// Check an immediate predicate on the specified instruction via an APInt.
  /// - InsnID - Instruction ID
  /// - The predicate to test
  GIM_CheckAPIntImmPredicate,
  /// Check a floating point immediate predicate on the specified instruction.
  /// - InsnID - Instruction ID
  /// - The predicate to test
  GIM_CheckAPFloatImmPredicate,
  /// Check an immediate predicate on the specified instruction
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - The predicate to test
  GIM_CheckImmOperandPredicate,
  /// Check a memory operation has the specified atomic ordering.
  /// - InsnID - Instruction ID
  /// - Ordering - The AtomicOrdering value
  GIM_CheckAtomicOrdering,
  GIM_CheckAtomicOrderingOrStrongerThan,
  GIM_CheckAtomicOrderingWeakerThan,
  /// Check the size of the memory access for the given machine memory operand.
  /// - InsnID - Instruction ID
  /// - MMOIdx - MMO index
  /// - Size - The size in bytes of the memory access
  GIM_CheckMemorySizeEqualTo,

  /// Check the address space of the memory access for the given machine memory
  /// operand.
  /// - InsnID - Instruction ID
  /// - MMOIdx - MMO index
  /// - NumAddrSpace - Number of valid address spaces
  /// - AddrSpaceN - An allowed space of the memory access
  /// - AddrSpaceN+1 ...
  GIM_CheckMemoryAddressSpace,

  /// Check the minimum alignment of the memory access for the given machine
  /// memory operand.
  /// - InsnID - Instruction ID
  /// - MMOIdx - MMO index
  /// - MinAlign - Minimum acceptable alignment
  GIM_CheckMemoryAlignment,

  /// Check the size of the memory access for the given machine memory operand
  /// against the size of an operand.
  /// - InsnID - Instruction ID
  /// - MMOIdx - MMO index
  /// - OpIdx - The operand index to compare the MMO against
  GIM_CheckMemorySizeEqualToLLT,
  GIM_CheckMemorySizeLessThanLLT,
  GIM_CheckMemorySizeGreaterThanLLT,

  /// Check if this is a vector that can be treated as a vector splat
  /// constant. This is valid for both G_BUILD_VECTOR as well as
  /// G_BUILD_VECTOR_TRUNC. For AllOnes refers to individual bits, so a -1
  /// element.
  /// - InsnID - Instruction ID
  GIM_CheckIsBuildVectorAllOnes,
  GIM_CheckIsBuildVectorAllZeros,

  /// Check a trivial predicate which takes no arguments.
  /// This can be used by executors to implement custom flags that don't fit in
  /// target features.
  GIM_CheckSimplePredicate,

  /// Check a generic C++ instruction predicate
  /// - InsnID - Instruction ID
  /// - PredicateID - The ID of the predicate function to call
  GIM_CheckCxxInsnPredicate,

  /// Check if there's no use of the first result.
  /// - InsnID - Instruction ID
  GIM_CheckHasNoUse,

  /// Check the type for the specified operand
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - Expected type
  GIM_CheckType,
  /// Check the type of a pointer to any address space.
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - SizeInBits - The size of the pointer value in bits.
  GIM_CheckPointerToAny,
  /// Check the register bank for the specified operand
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - Expected register bank (specified as a register class)
  GIM_CheckRegBankForClass,

  /// Check the operand matches a complex predicate
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - RendererID - The renderer to hold the result
  /// - Complex predicate ID
  GIM_CheckComplexPattern,

  /// Check the operand is a specific integer
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - Expected integer
  GIM_CheckConstantInt,
  /// Check the operand is a specific literal integer (i.e. MO.isImm() or
  /// MO.isCImm() is true).
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - Expected integer
  GIM_CheckLiteralInt,
  /// Check the operand is a specific intrinsic ID
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - Expected Intrinsic ID
  GIM_CheckIntrinsicID,

  /// Check the operand is a specific predicate
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - Expected predicate
  GIM_CheckCmpPredicate,

  /// Check the specified operand is an MBB
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  GIM_CheckIsMBB,

  /// Check the specified operand is an Imm
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  GIM_CheckIsImm,

  /// Check if the specified operand is safe to fold into the current
  /// instruction.
  /// - InsnID - Instruction ID
  GIM_CheckIsSafeToFold,

  /// Check the specified operands are identical.
  /// The IgnoreCopies variant looks through COPY instructions before
  /// comparing the operands.
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - OtherInsnID - Other instruction ID
  /// - OtherOpIdx - Other operand index
  GIM_CheckIsSameOperand,
  GIM_CheckIsSameOperandIgnoreCopies,

  /// Predicates with 'let PredicateCodeUsesOperands = 1' need to examine some
  /// named operands that will be recorded in RecordedOperands. Names of these
  /// operands are referenced in predicate argument list. Emitter determines
  /// StoreIdx(corresponds to the order in which names appear in argument list).
  /// - InsnID - Instruction ID
  /// - OpIdx - Operand index
  /// - StoreIdx - Store location in RecordedOperands.
  GIM_RecordNamedOperand,

  /// Fail the current try-block, or completely fail to match if there is no
  /// current try-block.
  GIM_Reject,

  //=== Renderers ===

  /// Mutate an instruction
  /// - NewInsnID - Instruction ID to define
  /// - OldInsnID - Instruction ID to mutate
  /// - NewOpcode - The new opcode to use
  GIR_MutateOpcode,

  /// Build a new instruction
  /// - InsnID - Instruction ID to define
  /// - Opcode - The new opcode to use
  GIR_BuildMI,

  /// Copy an operand to the specified instruction
  /// - NewInsnID - Instruction ID to modify
  /// - OldInsnID - Instruction ID to copy from
  /// - OpIdx - The operand to copy
  GIR_Copy,

  /// Copy an operand to the specified instruction or add a zero register if the
  /// operand is a zero immediate.
  /// - NewInsnID - Instruction ID to modify
  /// - OldInsnID - Instruction ID to copy from
  /// - OpIdx - The operand to copy
  /// - ZeroReg - The zero register to use
  GIR_CopyOrAddZeroReg,
  /// Copy an operand to the specified instruction
  /// - NewInsnID - Instruction ID to modify
  /// - OldInsnID - Instruction ID to copy from
  /// - OpIdx - The operand to copy
  /// - SubRegIdx - The subregister to copy
  GIR_CopySubReg,

  /// Add an implicit register def to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - RegNum - The register to add
  /// - Flags - Register Flags
  GIR_AddImplicitDef,
  /// Add an implicit register use to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - RegNum - The register to add
  GIR_AddImplicitUse,
  /// Add an register to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - RegNum - The register to add
  GIR_AddRegister,

  /// Marks the implicit def of a register as dead.
  /// - InsnID - Instruction ID to modify
  /// - OpIdx - The implicit def operand index
  ///
  /// OpIdx starts at 0 for the first implicit def.
  GIR_SetImplicitDefDead,

  /// Add a temporary register to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - TempRegID - The temporary register ID to add
  /// - TempRegFlags - The register flags to set
  GIR_AddTempRegister,

  /// Add a temporary register to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - TempRegID - The temporary register ID to add
  /// - TempRegFlags - The register flags to set
  /// - SubRegIndex - The subregister index to set
  GIR_AddTempSubRegister,

  /// Add an immediate to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - Imm - The immediate to add
  GIR_AddImm,

  /// Add an CImm to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - Ty - Type of the constant immediate.
  /// - Imm - The immediate to add
  GIR_AddCImm,

  /// Render complex operands to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - RendererID - The renderer to call
  GIR_ComplexRenderer,
  /// Render sub-operands of complex operands to the specified instruction
  /// - InsnID - Instruction ID to modify
  /// - RendererID - The renderer to call
  /// - RenderOpID - The suboperand to render.
  GIR_ComplexSubOperandRenderer,
  /// Render subregisters of suboperands of complex operands to the
  /// specified instruction
  /// - InsnID - Instruction ID to modify
  /// - RendererID - The renderer to call
  /// - RenderOpID - The suboperand to render
  /// - SubRegIdx - The subregister to extract
  GIR_ComplexSubOperandSubRegRenderer,

  /// Render operands to the specified instruction using a custom function
  /// - InsnID - Instruction ID to modify
  /// - OldInsnID - Instruction ID to get the matched operand from
  /// - RendererFnID - Custom renderer function to call
  GIR_CustomRenderer,

  /// Calls a C++ function to perform an action when a match is complete.
  /// The MatcherState is passed to the function to allow it to modify
  /// instructions.
  /// This is less constrained than a custom renderer and can update
  /// instructions
  /// in the state.
  /// - FnID - The function to call.
  /// TODO: Remove this at some point when combiners aren't reliant on it. It's
  /// a bit of a hack.
  GIR_CustomAction,

  /// Render operands to the specified instruction using a custom function,
  /// reading from a specific operand.
  /// - InsnID - Instruction ID to modify
  /// - OldInsnID - Instruction ID to get the matched operand from
  /// - OpIdx - Operand index in OldInsnID the render function should read
  /// from..
  /// - RendererFnID - Custom renderer function to call
  GIR_CustomOperandRenderer,

  /// Render a G_CONSTANT operator as a sign-extended immediate.
  /// - NewInsnID - Instruction ID to modify
  /// - OldInsnID - Instruction ID to copy from
  /// The operand index is implicitly 1.
  GIR_CopyConstantAsSImm,

  /// Render a G_FCONSTANT operator as a sign-extended immediate.
  /// - NewInsnID - Instruction ID to modify
  /// - OldInsnID - Instruction ID to copy from
  /// The operand index is implicitly 1.
  GIR_CopyFConstantAsFPImm,

  /// Constrain an instruction operand to a register class.
  /// - InsnID - Instruction ID to modify
  /// - OpIdx - Operand index
  /// - RCEnum - Register class enumeration value
  GIR_ConstrainOperandRC,

  /// Constrain an instructions operands according to the instruction
  /// description.
  /// - InsnID - Instruction ID to modify
  GIR_ConstrainSelectedInstOperands,

  /// Merge all memory operands into instruction.
  /// - InsnID - Instruction ID to modify
  /// - MergeInsnID... - One or more Instruction ID to merge into the result.
  /// - GIU_MergeMemOperands_EndOfList - Terminates the list of instructions to
  ///                                    merge.
  GIR_MergeMemOperands,

  /// Erase from parent.
  /// - InsnID - Instruction ID to erase
  GIR_EraseFromParent,

  /// Create a new temporary register that's not constrained.
  /// - TempRegID - The temporary register ID to initialize.
  /// - Expected type
  GIR_MakeTempReg,

  /// A successful emission
  GIR_Done,

  /// Increment the rule coverage counter.
  /// - RuleID - The ID of the rule that was covered.
  GIR_Coverage,

  /// Keeping track of the number of the GI opcodes. Must be the last entry.
  GIU_NumOpcodes,
};

enum {
  /// Indicates the end of the variable-length MergeInsnID list in a
  /// GIR_MergeMemOperands opcode.
  GIU_MergeMemOperands_EndOfList = -1,
};

/// Provides the logic to execute GlobalISel match tables, which are used by the
/// instruction selector and instruction combiners as their engine to match and
/// apply MIR patterns.
class GIMatchTableExecutor {
public:
  virtual ~GIMatchTableExecutor() = default;

  CodeGenCoverage *CoverageInfo = nullptr;
  GISelKnownBits *KB = nullptr;
  MachineFunction *MF = nullptr;
  ProfileSummaryInfo *PSI = nullptr;
  BlockFrequencyInfo *BFI = nullptr;
  // For some predicates, we need to track the current MBB.
  MachineBasicBlock *CurMBB = nullptr;

  virtual void setupGeneratedPerFunctionState(MachineFunction &MF) = 0;

  /// Setup per-MF executor state.
  virtual void setupMF(MachineFunction &mf, GISelKnownBits *kb,
                       CodeGenCoverage *covinfo = nullptr,
                       ProfileSummaryInfo *psi = nullptr,
                       BlockFrequencyInfo *bfi = nullptr) {
    CoverageInfo = covinfo;
    KB = kb;
    MF = &mf;
    PSI = psi;
    BFI = bfi;
    CurMBB = nullptr;
    setupGeneratedPerFunctionState(mf);
  }

protected:
  using ComplexRendererFns =
      std::optional<SmallVector<std::function<void(MachineInstrBuilder &)>, 4>>;
  using RecordedMIVector = SmallVector<MachineInstr *, 4>;
  using NewMIVector = SmallVector<MachineInstrBuilder, 4>;

  struct MatcherState {
    std::vector<ComplexRendererFns::value_type> Renderers;
    RecordedMIVector MIs;
    DenseMap<unsigned, unsigned> TempRegisters;
    /// Named operands that predicate with 'let PredicateCodeUsesOperands = 1'
    /// referenced in its argument list. Operands are inserted at index set by
    /// emitter, it corresponds to the order in which names appear in argument
    /// list. Currently such predicates don't have more then 3 arguments.
    std::array<const MachineOperand *, 3> RecordedOperands;

    MatcherState(unsigned MaxRenderers);
  };

  bool shouldOptForSize(const MachineFunction *MF) const {
    const auto &F = MF->getFunction();
    return F.hasOptSize() || F.hasMinSize() ||
           (PSI && BFI && CurMBB && llvm::shouldOptForSize(*CurMBB, PSI, BFI));
  }

public:
  template <class PredicateBitset, class ComplexMatcherMemFn,
            class CustomRendererFn>
  struct ExecInfoTy {
    ExecInfoTy(const LLT *TypeObjects, size_t NumTypeObjects,
               const PredicateBitset *FeatureBitsets,
               const ComplexMatcherMemFn *ComplexPredicates,
               const CustomRendererFn *CustomRenderers)
        : TypeObjects(TypeObjects), FeatureBitsets(FeatureBitsets),
          ComplexPredicates(ComplexPredicates),
          CustomRenderers(CustomRenderers) {

      for (size_t I = 0; I < NumTypeObjects; ++I)
        TypeIDMap[TypeObjects[I]] = I;
    }
    const LLT *TypeObjects;
    const PredicateBitset *FeatureBitsets;
    const ComplexMatcherMemFn *ComplexPredicates;
    const CustomRendererFn *CustomRenderers;

    SmallDenseMap<LLT, unsigned, 64> TypeIDMap;
  };

protected:
  GIMatchTableExecutor();

  /// Execute a given matcher table and return true if the match was successful
  /// and false otherwise.
  template <class TgtExecutor, class PredicateBitset, class ComplexMatcherMemFn,
            class CustomRendererFn>
  bool executeMatchTable(
      TgtExecutor &Exec, NewMIVector &OutMIs, MatcherState &State,
      const ExecInfoTy<PredicateBitset, ComplexMatcherMemFn, CustomRendererFn>
          &ISelInfo,
      const int64_t *MatchTable, const TargetInstrInfo &TII,
      MachineRegisterInfo &MRI, const TargetRegisterInfo &TRI,
      const RegisterBankInfo &RBI, const PredicateBitset &AvailableFeatures,
      CodeGenCoverage *CoverageInfo,
      GISelChangeObserver *Observer = nullptr) const;

  virtual const int64_t *getMatchTable() const {
    llvm_unreachable("Should have been overridden by tablegen if used");
  }

  virtual bool testImmPredicate_I64(unsigned, int64_t) const {
    llvm_unreachable(
        "Subclasses must override this with a tablegen-erated function");
  }
  virtual bool testImmPredicate_APInt(unsigned, const APInt &) const {
    llvm_unreachable(
        "Subclasses must override this with a tablegen-erated function");
  }
  virtual bool testImmPredicate_APFloat(unsigned, const APFloat &) const {
    llvm_unreachable(
        "Subclasses must override this with a tablegen-erated function");
  }
  virtual bool testMIPredicate_MI(unsigned, const MachineInstr &,
                                  const MatcherState &State) const {
    llvm_unreachable(
        "Subclasses must override this with a tablegen-erated function");
  }

  virtual bool testSimplePredicate(unsigned) const {
    llvm_unreachable("Subclass does not implement testSimplePredicate!");
  }

  virtual void runCustomAction(unsigned, const MatcherState &State,
                               NewMIVector &OutMIs) const {
    llvm_unreachable("Subclass does not implement runCustomAction!");
  }

  bool isOperandImmEqual(const MachineOperand &MO, int64_t Value,
                         const MachineRegisterInfo &MRI,
                         bool Splat = false) const;

  /// Return true if the specified operand is a G_PTR_ADD with a G_CONSTANT on
  /// the right-hand side. GlobalISel's separation of pointer and integer types
  /// means that we don't need to worry about G_OR with equivalent semantics.
  bool isBaseWithConstantOffset(const MachineOperand &Root,
                                const MachineRegisterInfo &MRI) const;

  /// Return true if MI can obviously be folded into IntoMI.
  /// MI and IntoMI do not need to be in the same basic blocks, but MI must
  /// preceed IntoMI.
  bool isObviouslySafeToFold(MachineInstr &MI, MachineInstr &IntoMI) const;
};

} // end namespace llvm

#endif // LLVM_CODEGEN_GLOBALISEL_GIMATCHTABLEEXECUTOR_H
