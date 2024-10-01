//===- Instruction.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SANDBOXIR_INSTRUCTION_H
#define LLVM_SANDBOXIR_INSTRUCTION_H

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/SandboxIR/BasicBlock.h"
#include "llvm/SandboxIR/Constant.h"
#include "llvm/SandboxIR/User.h"

namespace llvm::sandboxir {

/// A sandboxir::User with operands, opcode and linked with previous/next
/// instructions in an instruction list.
class Instruction : public User {
public:
  enum class Opcode {
#define OP(OPC) OPC,
#define OPCODES(...) __VA_ARGS__
#define DEF_INSTR(ID, OPC, CLASS) OPC
#include "llvm/SandboxIR/SandboxIRValues.def"
  };

protected:
  Instruction(ClassID ID, Opcode Opc, llvm::Instruction *I,
              sandboxir::Context &SBCtx)
      : User(ID, I, SBCtx), Opc(Opc) {}

  Opcode Opc;

  /// A SandboxIR Instruction may map to multiple LLVM IR Instruction. This
  /// returns its topmost LLVM IR instruction.
  llvm::Instruction *getTopmostLLVMInstruction() const;
  friend class VAArgInst;          // For getTopmostLLVMInstruction().
  friend class FreezeInst;         // For getTopmostLLVMInstruction().
  friend class FenceInst;          // For getTopmostLLVMInstruction().
  friend class SelectInst;         // For getTopmostLLVMInstruction().
  friend class ExtractElementInst; // For getTopmostLLVMInstruction().
  friend class InsertElementInst;  // For getTopmostLLVMInstruction().
  friend class ShuffleVectorInst;  // For getTopmostLLVMInstruction().
  friend class ExtractValueInst;   // For getTopmostLLVMInstruction().
  friend class InsertValueInst;    // For getTopmostLLVMInstruction().
  friend class BranchInst;         // For getTopmostLLVMInstruction().
  friend class LoadInst;           // For getTopmostLLVMInstruction().
  friend class StoreInst;          // For getTopmostLLVMInstruction().
  friend class ReturnInst;         // For getTopmostLLVMInstruction().
  friend class CallInst;           // For getTopmostLLVMInstruction().
  friend class InvokeInst;         // For getTopmostLLVMInstruction().
  friend class CallBrInst;         // For getTopmostLLVMInstruction().
  friend class LandingPadInst;     // For getTopmostLLVMInstruction().
  friend class CatchPadInst;       // For getTopmostLLVMInstruction().
  friend class CleanupPadInst;     // For getTopmostLLVMInstruction().
  friend class CatchReturnInst;    // For getTopmostLLVMInstruction().
  friend class CleanupReturnInst;  // For getTopmostLLVMInstruction().
  friend class GetElementPtrInst;  // For getTopmostLLVMInstruction().
  friend class ResumeInst;         // For getTopmostLLVMInstruction().
  friend class CatchSwitchInst;    // For getTopmostLLVMInstruction().
  friend class SwitchInst;         // For getTopmostLLVMInstruction().
  friend class UnaryOperator;      // For getTopmostLLVMInstruction().
  friend class BinaryOperator;     // For getTopmostLLVMInstruction().
  friend class AtomicRMWInst;      // For getTopmostLLVMInstruction().
  friend class AtomicCmpXchgInst;  // For getTopmostLLVMInstruction().
  friend class AllocaInst;         // For getTopmostLLVMInstruction().
  friend class CastInst;           // For getTopmostLLVMInstruction().
  friend class PHINode;            // For getTopmostLLVMInstruction().
  friend class UnreachableInst;    // For getTopmostLLVMInstruction().
  friend class CmpInst;            // For getTopmostLLVMInstruction().

  /// \Returns the LLVM IR Instructions that this SandboxIR maps to in program
  /// order.
  virtual SmallVector<llvm::Instruction *, 1> getLLVMInstrs() const = 0;
  friend class EraseFromParent; // For getLLVMInstrs().

public:
  static const char *getOpcodeName(Opcode Opc);
  /// This is used by BasicBlock::iterator.
  virtual unsigned getNumOfIRInstrs() const = 0;
  /// \Returns a BasicBlock::iterator for this Instruction.
  BBIterator getIterator() const;
  /// \Returns the next sandboxir::Instruction in the block, or nullptr if at
  /// the end of the block.
  Instruction *getNextNode() const;
  /// \Returns the previous sandboxir::Instruction in the block, or nullptr if
  /// at the beginning of the block.
  Instruction *getPrevNode() const;
  /// \Returns this Instruction's opcode. Note that SandboxIR has its own opcode
  /// state to allow for new SandboxIR-specific instructions.
  Opcode getOpcode() const { return Opc; }

  const char *getOpcodeName() const { return getOpcodeName(Opc); }

  // Note that these functions below are calling into llvm::Instruction.
  // A sandbox IR instruction could introduce a new opcode that could change the
  // behavior of one of these functions. It is better that these functions are
  // only added as needed and new sandbox IR instructions must explicitly check
  // if any of these functions could have a different behavior.

  bool isTerminator() const {
    return cast<llvm::Instruction>(Val)->isTerminator();
  }
  bool isUnaryOp() const { return cast<llvm::Instruction>(Val)->isUnaryOp(); }
  bool isBinaryOp() const { return cast<llvm::Instruction>(Val)->isBinaryOp(); }
  bool isIntDivRem() const {
    return cast<llvm::Instruction>(Val)->isIntDivRem();
  }
  bool isShift() const { return cast<llvm::Instruction>(Val)->isShift(); }
  bool isCast() const { return cast<llvm::Instruction>(Val)->isCast(); }
  bool isFuncletPad() const {
    return cast<llvm::Instruction>(Val)->isFuncletPad();
  }
  bool isSpecialTerminator() const {
    return cast<llvm::Instruction>(Val)->isSpecialTerminator();
  }
  bool isOnlyUserOfAnyOperand() const {
    return cast<llvm::Instruction>(Val)->isOnlyUserOfAnyOperand();
  }
  bool isLogicalShift() const {
    return cast<llvm::Instruction>(Val)->isLogicalShift();
  }

  //===--------------------------------------------------------------------===//
  // Metadata manipulation.
  //===--------------------------------------------------------------------===//

  /// Return true if the instruction has any metadata attached to it.
  bool hasMetadata() const {
    return cast<llvm::Instruction>(Val)->hasMetadata();
  }

  /// Return true if this instruction has metadata attached to it other than a
  /// debug location.
  bool hasMetadataOtherThanDebugLoc() const {
    return cast<llvm::Instruction>(Val)->hasMetadataOtherThanDebugLoc();
  }

  /// Return true if this instruction has the given type of metadata attached.
  bool hasMetadata(unsigned KindID) const {
    return cast<llvm::Instruction>(Val)->hasMetadata(KindID);
  }

  // TODO: Implement getMetadata and getAllMetadata after sandboxir::MDNode is
  // available.

  // TODO: More missing functions

  /// Detach this from its parent BasicBlock without deleting it.
  void removeFromParent();
  /// Detach this Value from its parent and delete it.
  void eraseFromParent();
  /// Insert this detached instruction before \p BeforeI.
  void insertBefore(Instruction *BeforeI);
  /// Insert this detached instruction after \p AfterI.
  void insertAfter(Instruction *AfterI);
  /// Insert this detached instruction into \p BB at \p WhereIt.
  void insertInto(BasicBlock *BB, const BBIterator &WhereIt);
  /// Move this instruction to \p WhereIt.
  void moveBefore(BasicBlock &BB, const BBIterator &WhereIt);
  /// Move this instruction before \p Before.
  void moveBefore(Instruction *Before) {
    moveBefore(*Before->getParent(), Before->getIterator());
  }
  /// Move this instruction after \p After.
  void moveAfter(Instruction *After) {
    moveBefore(*After->getParent(), std::next(After->getIterator()));
  }
  // TODO: This currently relies on LLVM IR Instruction::comesBefore which is
  // can be linear-time.
  /// Given an instruction Other in the same basic block as this instruction,
  /// return true if this instruction comes before Other.
  bool comesBefore(const Instruction *Other) const {
    return cast<llvm::Instruction>(Val)->comesBefore(
        cast<llvm::Instruction>(Other->Val));
  }
  /// \Returns the BasicBlock containing this Instruction, or null if it is
  /// detached.
  BasicBlock *getParent() const;
  /// For isa/dyn_cast.
  static bool classof(const sandboxir::Value *From);

  /// Determine whether the no signed wrap flag is set.
  bool hasNoUnsignedWrap() const {
    return cast<llvm::Instruction>(Val)->hasNoUnsignedWrap();
  }
  /// Set or clear the nuw flag on this instruction, which must be an operator
  /// which supports this flag. See LangRef.html for the meaning of this flag.
  void setHasNoUnsignedWrap(bool B = true);
  /// Determine whether the no signed wrap flag is set.
  bool hasNoSignedWrap() const {
    return cast<llvm::Instruction>(Val)->hasNoSignedWrap();
  }
  /// Set or clear the nsw flag on this instruction, which must be an operator
  /// which supports this flag. See LangRef.html for the meaning of this flag.
  void setHasNoSignedWrap(bool B = true);
  /// Determine whether all fast-math-flags are set.
  bool isFast() const { return cast<llvm::Instruction>(Val)->isFast(); }
  /// Set or clear all fast-math-flags on this instruction, which must be an
  /// operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setFast(bool B);
  /// Determine whether the allow-reassociation flag is set.
  bool hasAllowReassoc() const {
    return cast<llvm::Instruction>(Val)->hasAllowReassoc();
  }
  /// Set or clear the reassociation flag on this instruction, which must be
  /// an operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasAllowReassoc(bool B);
  /// Determine whether the exact flag is set.
  bool isExact() const { return cast<llvm::Instruction>(Val)->isExact(); }
  /// Set or clear the exact flag on this instruction, which must be an operator
  /// which supports this flag. See LangRef.html for the meaning of this flag.
  void setIsExact(bool B = true);
  /// Determine whether the no-NaNs flag is set.
  bool hasNoNaNs() const { return cast<llvm::Instruction>(Val)->hasNoNaNs(); }
  /// Set or clear the no-nans flag on this instruction, which must be an
  /// operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasNoNaNs(bool B);
  /// Determine whether the no-infs flag is set.
  bool hasNoInfs() const { return cast<llvm::Instruction>(Val)->hasNoInfs(); }
  /// Set or clear the no-infs flag on this instruction, which must be an
  /// operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasNoInfs(bool B);
  /// Determine whether the no-signed-zeros flag is set.
  bool hasNoSignedZeros() const {
    return cast<llvm::Instruction>(Val)->hasNoSignedZeros();
  }
  /// Set or clear the no-signed-zeros flag on this instruction, which must be
  /// an operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasNoSignedZeros(bool B);
  /// Determine whether the allow-reciprocal flag is set.
  bool hasAllowReciprocal() const {
    return cast<llvm::Instruction>(Val)->hasAllowReciprocal();
  }
  /// Set or clear the allow-reciprocal flag on this instruction, which must be
  /// an operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasAllowReciprocal(bool B);
  /// Determine whether the allow-contract flag is set.
  bool hasAllowContract() const {
    return cast<llvm::Instruction>(Val)->hasAllowContract();
  }
  /// Set or clear the allow-contract flag on this instruction, which must be
  /// an operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasAllowContract(bool B);
  /// Determine whether the approximate-math-functions flag is set.
  bool hasApproxFunc() const {
    return cast<llvm::Instruction>(Val)->hasApproxFunc();
  }
  /// Set or clear the approximate-math-functions flag on this instruction,
  /// which must be an operator which supports this flag. See LangRef.html for
  /// the meaning of this flag.
  void setHasApproxFunc(bool B);
  /// Convenience function for getting all the fast-math flags, which must be an
  /// operator which supports these flags. See LangRef.html for the meaning of
  /// these flags.
  FastMathFlags getFastMathFlags() const {
    return cast<llvm::Instruction>(Val)->getFastMathFlags();
  }
  /// Convenience function for setting multiple fast-math flags on this
  /// instruction, which must be an operator which supports these flags. See
  /// LangRef.html for the meaning of these flags.
  void setFastMathFlags(FastMathFlags FMF);
  /// Convenience function for transferring all fast-math flag values to this
  /// instruction, which must be an operator which supports these flags. See
  /// LangRef.html for the meaning of these flags.
  void copyFastMathFlags(FastMathFlags FMF);

  bool isAssociative() const {
    return cast<llvm::Instruction>(Val)->isAssociative();
  }

  bool isCommutative() const {
    return cast<llvm::Instruction>(Val)->isCommutative();
  }

  bool isIdempotent() const {
    return cast<llvm::Instruction>(Val)->isIdempotent();
  }

  bool isNilpotent() const {
    return cast<llvm::Instruction>(Val)->isNilpotent();
  }

  bool mayWriteToMemory() const {
    return cast<llvm::Instruction>(Val)->mayWriteToMemory();
  }

  bool mayReadFromMemory() const {
    return cast<llvm::Instruction>(Val)->mayReadFromMemory();
  }
  bool mayReadOrWriteMemory() const {
    return cast<llvm::Instruction>(Val)->mayReadOrWriteMemory();
  }

  bool isAtomic() const { return cast<llvm::Instruction>(Val)->isAtomic(); }

  bool hasAtomicLoad() const {
    return cast<llvm::Instruction>(Val)->hasAtomicLoad();
  }

  bool hasAtomicStore() const {
    return cast<llvm::Instruction>(Val)->hasAtomicStore();
  }

  bool isVolatile() const { return cast<llvm::Instruction>(Val)->isVolatile(); }

  Type *getAccessType() const;

  bool mayThrow(bool IncludePhaseOneUnwind = false) const {
    return cast<llvm::Instruction>(Val)->mayThrow(IncludePhaseOneUnwind);
  }

  bool isFenceLike() const {
    return cast<llvm::Instruction>(Val)->isFenceLike();
  }

  bool mayHaveSideEffects() const {
    return cast<llvm::Instruction>(Val)->mayHaveSideEffects();
  }

  // TODO: Missing functions.

  bool isStackSaveOrRestoreIntrinsic() const {
    auto *I = cast<llvm::Instruction>(Val);
    return match(I,
                 PatternMatch::m_Intrinsic<llvm::Intrinsic::stackrestore>()) ||
           match(I, PatternMatch::m_Intrinsic<llvm::Intrinsic::stacksave>());
  }

  /// We consider \p I as a Memory Dependency Candidate instruction if it
  /// reads/write memory or if it has side-effects. This is used by the
  /// dependency graph.
  bool isMemDepCandidate() const {
    auto *I = cast<llvm::Instruction>(Val);
    return I->mayReadOrWriteMemory() &&
           (!isa<llvm::IntrinsicInst>(I) ||
            (cast<llvm::IntrinsicInst>(I)->getIntrinsicID() !=
                 Intrinsic::sideeffect &&
             cast<llvm::IntrinsicInst>(I)->getIntrinsicID() !=
                 Intrinsic::pseudoprobe));
  }

#ifndef NDEBUG
  void dumpOS(raw_ostream &OS) const override;
#endif
};

/// Instructions that contain a single LLVM Instruction can inherit from this.
template <typename LLVMT> class SingleLLVMInstructionImpl : public Instruction {
  SingleLLVMInstructionImpl(ClassID ID, Opcode Opc, llvm::Instruction *I,
                            sandboxir::Context &SBCtx)
      : Instruction(ID, Opc, I, SBCtx) {}

  // All instructions are friends with this so they can call the constructor.
#define DEF_INSTR(ID, OPC, CLASS) friend class CLASS;
#include "llvm/SandboxIR/SandboxIRValues.def"
  friend class UnaryInstruction;
  friend class CallBase;
  friend class FuncletPadInst;
  friend class CmpInst;

  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  SmallVector<llvm::Instruction *, 1> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }

public:
  unsigned getUseOperandNo(const Use &Use) const final {
    return getUseOperandNoDefault(Use);
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
#ifndef NDEBUG
  void verify() const final { assert(isa<LLVMT>(Val) && "Expected LLVMT!"); }
  void dumpOS(raw_ostream &OS) const override {
    dumpCommonPrefix(OS);
    dumpCommonSuffix(OS);
  }
#endif
};

class FenceInst : public SingleLLVMInstructionImpl<llvm::FenceInst> {
  FenceInst(llvm::FenceInst *FI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Fence, Opcode::Fence, FI, Ctx) {}
  friend Context; // For constructor;

public:
  static FenceInst *create(AtomicOrdering Ordering, BBIterator WhereIt,
                           BasicBlock *WhereBB, Context &Ctx,
                           SyncScope::ID SSID = SyncScope::System);
  /// Returns the ordering constraint of this fence instruction.
  AtomicOrdering getOrdering() const {
    return cast<llvm::FenceInst>(Val)->getOrdering();
  }
  /// Sets the ordering constraint of this fence instruction.  May only be
  /// Acquire, Release, AcquireRelease, or SequentiallyConsistent.
  void setOrdering(AtomicOrdering Ordering);
  /// Returns the synchronization scope ID of this fence instruction.
  SyncScope::ID getSyncScopeID() const {
    return cast<llvm::FenceInst>(Val)->getSyncScopeID();
  }
  /// Sets the synchronization scope ID of this fence instruction.
  void setSyncScopeID(SyncScope::ID SSID);
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Fence;
  }
};

class SelectInst : public SingleLLVMInstructionImpl<llvm::SelectInst> {
  /// Use Context::createSelectInst(). Don't call the
  /// constructor directly.
  SelectInst(llvm::SelectInst *CI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Select, Opcode::Select, CI, Ctx) {}
  friend Context; // for SelectInst()
  static Value *createCommon(Value *Cond, Value *True, Value *False,
                             const Twine &Name, IRBuilder<> &Builder,
                             Context &Ctx);

public:
  static Value *create(Value *Cond, Value *True, Value *False,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *Cond, Value *True, Value *False,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");

  const Value *getCondition() const { return getOperand(0); }
  const Value *getTrueValue() const { return getOperand(1); }
  const Value *getFalseValue() const { return getOperand(2); }
  Value *getCondition() { return getOperand(0); }
  Value *getTrueValue() { return getOperand(1); }
  Value *getFalseValue() { return getOperand(2); }

  void setCondition(Value *New) { setOperand(0, New); }
  void setTrueValue(Value *New) { setOperand(1, New); }
  void setFalseValue(Value *New) { setOperand(2, New); }
  void swapValues();

  /// Return a string if the specified operands are invalid for a select
  /// operation, otherwise return null.
  static const char *areInvalidOperands(Value *Cond, Value *True,
                                        Value *False) {
    return llvm::SelectInst::areInvalidOperands(Cond->Val, True->Val,
                                                False->Val);
  }

  /// For isa/dyn_cast.
  static bool classof(const Value *From);
};

class InsertElementInst final
    : public SingleLLVMInstructionImpl<llvm::InsertElementInst> {
  /// Use Context::createInsertElementInst() instead.
  InsertElementInst(llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::InsertElement, Opcode::InsertElement,
                                  I, Ctx) {}
  friend class Context; // For accessing the constructor in create*()

public:
  static Value *create(Value *Vec, Value *NewElt, Value *Idx,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *Vec, Value *NewElt, Value *Idx,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::InsertElement;
  }
  static bool isValidOperands(const Value *Vec, const Value *NewElt,
                              const Value *Idx) {
    return llvm::InsertElementInst::isValidOperands(Vec->Val, NewElt->Val,
                                                    Idx->Val);
  }
};

class ExtractElementInst final
    : public SingleLLVMInstructionImpl<llvm::ExtractElementInst> {
  /// Use Context::createExtractElementInst() instead.
  ExtractElementInst(llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::ExtractElement,
                                  Opcode::ExtractElement, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()

public:
  static Value *create(Value *Vec, Value *Idx, Instruction *InsertBefore,
                       Context &Ctx, const Twine &Name = "");
  static Value *create(Value *Vec, Value *Idx, BasicBlock *InsertAtEnd,
                       Context &Ctx, const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ExtractElement;
  }

  static bool isValidOperands(const Value *Vec, const Value *Idx) {
    return llvm::ExtractElementInst::isValidOperands(Vec->Val, Idx->Val);
  }
  Value *getVectorOperand() { return getOperand(0); }
  Value *getIndexOperand() { return getOperand(1); }
  const Value *getVectorOperand() const { return getOperand(0); }
  const Value *getIndexOperand() const { return getOperand(1); }
  VectorType *getVectorOperandType() const;
};

class ShuffleVectorInst final
    : public SingleLLVMInstructionImpl<llvm::ShuffleVectorInst> {
  /// Use Context::createShuffleVectorInst() instead.
  ShuffleVectorInst(llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::ShuffleVector, Opcode::ShuffleVector,
                                  I, Ctx) {}
  friend class Context; // For accessing the constructor in create*()

public:
  static Value *create(Value *V1, Value *V2, Value *Mask,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *V1, Value *V2, Value *Mask,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *V1, Value *V2, ArrayRef<int> Mask,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Value *V1, Value *V2, ArrayRef<int> Mask,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ShuffleVector;
  }

  /// Swap the operands and adjust the mask to preserve the semantics of the
  /// instruction.
  void commute();

  /// Return true if a shufflevector instruction can be formed with the
  /// specified operands.
  static bool isValidOperands(const Value *V1, const Value *V2,
                              const Value *Mask) {
    return llvm::ShuffleVectorInst::isValidOperands(V1->Val, V2->Val,
                                                    Mask->Val);
  }
  static bool isValidOperands(const Value *V1, const Value *V2,
                              ArrayRef<int> Mask) {
    return llvm::ShuffleVectorInst::isValidOperands(V1->Val, V2->Val, Mask);
  }

  /// Overload to return most specific vector type.
  VectorType *getType() const;

  /// Return the shuffle mask value of this instruction for the given element
  /// index. Return PoisonMaskElem if the element is undef.
  int getMaskValue(unsigned Elt) const {
    return cast<llvm::ShuffleVectorInst>(Val)->getMaskValue(Elt);
  }

  /// Convert the input shuffle mask operand to a vector of integers. Undefined
  /// elements of the mask are returned as PoisonMaskElem.
  static void getShuffleMask(const Constant *Mask,
                             SmallVectorImpl<int> &Result) {
    llvm::ShuffleVectorInst::getShuffleMask(cast<llvm::Constant>(Mask->Val),
                                            Result);
  }

  /// Return the mask for this instruction as a vector of integers. Undefined
  /// elements of the mask are returned as PoisonMaskElem.
  void getShuffleMask(SmallVectorImpl<int> &Result) const {
    cast<llvm::ShuffleVectorInst>(Val)->getShuffleMask(Result);
  }

  /// Return the mask for this instruction, for use in bitcode.
  Constant *getShuffleMaskForBitcode() const;

  static Constant *convertShuffleMaskForBitcode(ArrayRef<int> Mask,
                                                Type *ResultTy);

  void setShuffleMask(ArrayRef<int> Mask);

  ArrayRef<int> getShuffleMask() const {
    return cast<llvm::ShuffleVectorInst>(Val)->getShuffleMask();
  }

  /// Return true if this shuffle returns a vector with a different number of
  /// elements than its source vectors.
  /// Examples: shufflevector <4 x n> A, <4 x n> B, <1,2,3>
  ///           shufflevector <4 x n> A, <4 x n> B, <1,2,3,4,5>
  bool changesLength() const {
    return cast<llvm::ShuffleVectorInst>(Val)->changesLength();
  }

  /// Return true if this shuffle returns a vector with a greater number of
  /// elements than its source vectors.
  /// Example: shufflevector <2 x n> A, <2 x n> B, <1,2,3>
  bool increasesLength() const {
    return cast<llvm::ShuffleVectorInst>(Val)->increasesLength();
  }

  /// Return true if this shuffle mask chooses elements from exactly one source
  /// vector.
  /// Example: <7,5,undef,7>
  /// This assumes that vector operands (of length \p NumSrcElts) are the same
  /// length as the mask.
  static bool isSingleSourceMask(ArrayRef<int> Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isSingleSourceMask(Mask, NumSrcElts);
  }
  static bool isSingleSourceMask(const Constant *Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isSingleSourceMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts);
  }

  /// Return true if this shuffle chooses elements from exactly one source
  /// vector without changing the length of that vector.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <3,0,undef,3>
  bool isSingleSource() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isSingleSource();
  }

  /// Return true if this shuffle mask chooses elements from exactly one source
  /// vector without lane crossings. A shuffle using this mask is not
  /// necessarily a no-op because it may change the number of elements from its
  /// input vectors or it may provide demanded bits knowledge via undef lanes.
  /// Example: <undef,undef,2,3>
  static bool isIdentityMask(ArrayRef<int> Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isIdentityMask(Mask, NumSrcElts);
  }
  static bool isIdentityMask(const Constant *Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isIdentityMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts);
  }

  /// Return true if this shuffle chooses elements from exactly one source
  /// vector without lane crossings and does not change the number of elements
  /// from its input vectors.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <4,undef,6,undef>
  bool isIdentity() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isIdentity();
  }

  /// Return true if this shuffle lengthens exactly one source vector with
  /// undefs in the high elements.
  bool isIdentityWithPadding() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isIdentityWithPadding();
  }

  /// Return true if this shuffle extracts the first N elements of exactly one
  /// source vector.
  bool isIdentityWithExtract() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isIdentityWithExtract();
  }

  /// Return true if this shuffle concatenates its 2 source vectors. This
  /// returns false if either input is undefined. In that case, the shuffle is
  /// is better classified as an identity with padding operation.
  bool isConcat() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isConcat();
  }

  /// Return true if this shuffle mask chooses elements from its source vectors
  /// without lane crossings. A shuffle using this mask would be
  /// equivalent to a vector select with a constant condition operand.
  /// Example: <4,1,6,undef>
  /// This returns false if the mask does not choose from both input vectors.
  /// In that case, the shuffle is better classified as an identity shuffle.
  /// This assumes that vector operands are the same length as the mask
  /// (a length-changing shuffle can never be equivalent to a vector select).
  static bool isSelectMask(ArrayRef<int> Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isSelectMask(Mask, NumSrcElts);
  }
  static bool isSelectMask(const Constant *Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isSelectMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts);
  }

  /// Return true if this shuffle chooses elements from its source vectors
  /// without lane crossings and all operands have the same number of elements.
  /// In other words, this shuffle is equivalent to a vector select with a
  /// constant condition operand.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <undef,1,6,3>
  /// This returns false if the mask does not choose from both input vectors.
  /// In that case, the shuffle is better classified as an identity shuffle.
  bool isSelect() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isSelect();
  }

  /// Return true if this shuffle mask swaps the order of elements from exactly
  /// one source vector.
  /// Example: <7,6,undef,4>
  /// This assumes that vector operands (of length \p NumSrcElts) are the same
  /// length as the mask.
  static bool isReverseMask(ArrayRef<int> Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isReverseMask(Mask, NumSrcElts);
  }
  static bool isReverseMask(const Constant *Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isReverseMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts);
  }

  /// Return true if this shuffle swaps the order of elements from exactly
  /// one source vector.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <3,undef,1,undef>
  bool isReverse() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isReverse();
  }

  /// Return true if this shuffle mask chooses all elements with the same value
  /// as the first element of exactly one source vector.
  /// Example: <4,undef,undef,4>
  /// This assumes that vector operands (of length \p NumSrcElts) are the same
  /// length as the mask.
  static bool isZeroEltSplatMask(ArrayRef<int> Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isZeroEltSplatMask(Mask, NumSrcElts);
  }
  static bool isZeroEltSplatMask(const Constant *Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isZeroEltSplatMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts);
  }

  /// Return true if all elements of this shuffle are the same value as the
  /// first element of exactly one source vector without changing the length
  /// of that vector.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <undef,0,undef,0>
  bool isZeroEltSplat() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isZeroEltSplat();
  }

  /// Return true if this shuffle mask is a transpose mask.
  /// Transpose vector masks transpose a 2xn matrix. They read corresponding
  /// even- or odd-numbered vector elements from two n-dimensional source
  /// vectors and write each result into consecutive elements of an
  /// n-dimensional destination vector. Two shuffles are necessary to complete
  /// the transpose, one for the even elements and another for the odd elements.
  /// This description closely follows how the TRN1 and TRN2 AArch64
  /// instructions operate.
  ///
  /// For example, a simple 2x2 matrix can be transposed with:
  ///
  ///   ; Original matrix
  ///   m0 = < a, b >
  ///   m1 = < c, d >
  ///
  ///   ; Transposed matrix
  ///   t0 = < a, c > = shufflevector m0, m1, < 0, 2 >
  ///   t1 = < b, d > = shufflevector m0, m1, < 1, 3 >
  ///
  /// For matrices having greater than n columns, the resulting nx2 transposed
  /// matrix is stored in two result vectors such that one vector contains
  /// interleaved elements from all the even-numbered rows and the other vector
  /// contains interleaved elements from all the odd-numbered rows. For example,
  /// a 2x4 matrix can be transposed with:
  ///
  ///   ; Original matrix
  ///   m0 = < a, b, c, d >
  ///   m1 = < e, f, g, h >
  ///
  ///   ; Transposed matrix
  ///   t0 = < a, e, c, g > = shufflevector m0, m1 < 0, 4, 2, 6 >
  ///   t1 = < b, f, d, h > = shufflevector m0, m1 < 1, 5, 3, 7 >
  static bool isTransposeMask(ArrayRef<int> Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isTransposeMask(Mask, NumSrcElts);
  }
  static bool isTransposeMask(const Constant *Mask, int NumSrcElts) {
    return llvm::ShuffleVectorInst::isTransposeMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts);
  }

  /// Return true if this shuffle transposes the elements of its inputs without
  /// changing the length of the vectors. This operation may also be known as a
  /// merge or interleave. See the description for isTransposeMask() for the
  /// exact specification.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <0,4,2,6>
  bool isTranspose() const {
    return cast<llvm::ShuffleVectorInst>(Val)->isTranspose();
  }

  /// Return true if this shuffle mask is a splice mask, concatenating the two
  /// inputs together and then extracts an original width vector starting from
  /// the splice index.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <1,2,3,4>
  /// This assumes that vector operands (of length \p NumSrcElts) are the same
  /// length as the mask.
  static bool isSpliceMask(ArrayRef<int> Mask, int NumSrcElts, int &Index) {
    return llvm::ShuffleVectorInst::isSpliceMask(Mask, NumSrcElts, Index);
  }
  static bool isSpliceMask(const Constant *Mask, int NumSrcElts, int &Index) {
    return llvm::ShuffleVectorInst::isSpliceMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts, Index);
  }

  /// Return true if this shuffle splices two inputs without changing the length
  /// of the vectors. This operation concatenates the two inputs together and
  /// then extracts an original width vector starting from the splice index.
  /// Example: shufflevector <4 x n> A, <4 x n> B, <1,2,3,4>
  bool isSplice(int &Index) const {
    return cast<llvm::ShuffleVectorInst>(Val)->isSplice(Index);
  }

  /// Return true if this shuffle mask is an extract subvector mask.
  /// A valid extract subvector mask returns a smaller vector from a single
  /// source operand. The base extraction index is returned as well.
  static bool isExtractSubvectorMask(ArrayRef<int> Mask, int NumSrcElts,
                                     int &Index) {
    return llvm::ShuffleVectorInst::isExtractSubvectorMask(Mask, NumSrcElts,
                                                           Index);
  }
  static bool isExtractSubvectorMask(const Constant *Mask, int NumSrcElts,
                                     int &Index) {
    return llvm::ShuffleVectorInst::isExtractSubvectorMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts, Index);
  }

  /// Return true if this shuffle mask is an extract subvector mask.
  bool isExtractSubvectorMask(int &Index) const {
    return cast<llvm::ShuffleVectorInst>(Val)->isExtractSubvectorMask(Index);
  }

  /// Return true if this shuffle mask is an insert subvector mask.
  /// A valid insert subvector mask inserts the lowest elements of a second
  /// source operand into an in-place first source operand.
  /// Both the sub vector width and the insertion index is returned.
  static bool isInsertSubvectorMask(ArrayRef<int> Mask, int NumSrcElts,
                                    int &NumSubElts, int &Index) {
    return llvm::ShuffleVectorInst::isInsertSubvectorMask(Mask, NumSrcElts,
                                                          NumSubElts, Index);
  }
  static bool isInsertSubvectorMask(const Constant *Mask, int NumSrcElts,
                                    int &NumSubElts, int &Index) {
    return llvm::ShuffleVectorInst::isInsertSubvectorMask(
        cast<llvm::Constant>(Mask->Val), NumSrcElts, NumSubElts, Index);
  }

  /// Return true if this shuffle mask is an insert subvector mask.
  bool isInsertSubvectorMask(int &NumSubElts, int &Index) const {
    return cast<llvm::ShuffleVectorInst>(Val)->isInsertSubvectorMask(NumSubElts,
                                                                     Index);
  }

  /// Return true if this shuffle mask replicates each of the \p VF elements
  /// in a vector \p ReplicationFactor times.
  /// For example, the mask for \p ReplicationFactor=3 and \p VF=4 is:
  ///   <0,0,0,1,1,1,2,2,2,3,3,3>
  static bool isReplicationMask(ArrayRef<int> Mask, int &ReplicationFactor,
                                int &VF) {
    return llvm::ShuffleVectorInst::isReplicationMask(Mask, ReplicationFactor,
                                                      VF);
  }
  static bool isReplicationMask(const Constant *Mask, int &ReplicationFactor,
                                int &VF) {
    return llvm::ShuffleVectorInst::isReplicationMask(
        cast<llvm::Constant>(Mask->Val), ReplicationFactor, VF);
  }

  /// Return true if this shuffle mask is a replication mask.
  bool isReplicationMask(int &ReplicationFactor, int &VF) const {
    return cast<llvm::ShuffleVectorInst>(Val)->isReplicationMask(
        ReplicationFactor, VF);
  }

  /// Return true if this shuffle mask represents "clustered" mask of size VF,
  /// i.e. each index between [0..VF) is used exactly once in each submask of
  /// size VF.
  /// For example, the mask for \p VF=4 is:
  /// 0, 1, 2, 3, 3, 2, 0, 1 - "clustered", because each submask of size 4
  /// (0,1,2,3 and 3,2,0,1) uses indices [0..VF) exactly one time.
  /// 0, 1, 2, 3, 3, 3, 1, 0 - not "clustered", because
  ///                          element 3 is used twice in the second submask
  ///                          (3,3,1,0) and index 2 is not used at all.
  static bool isOneUseSingleSourceMask(ArrayRef<int> Mask, int VF) {
    return llvm::ShuffleVectorInst::isOneUseSingleSourceMask(Mask, VF);
  }

  /// Return true if this shuffle mask is a one-use-single-source("clustered")
  /// mask.
  bool isOneUseSingleSourceMask(int VF) const {
    return cast<llvm::ShuffleVectorInst>(Val)->isOneUseSingleSourceMask(VF);
  }

  /// Change values in a shuffle permute mask assuming the two vector operands
  /// of length InVecNumElts have swapped position.
  static void commuteShuffleMask(MutableArrayRef<int> Mask,
                                 unsigned InVecNumElts) {
    llvm::ShuffleVectorInst::commuteShuffleMask(Mask, InVecNumElts);
  }

  /// Return if this shuffle interleaves its two input vectors together.
  bool isInterleave(unsigned Factor) const {
    return cast<llvm::ShuffleVectorInst>(Val)->isInterleave(Factor);
  }

  /// Return true if the mask interleaves one or more input vectors together.
  ///
  /// I.e. <0, LaneLen, ... , LaneLen*(Factor - 1), 1, LaneLen + 1, ...>
  /// E.g. For a Factor of 2 (LaneLen=4):
  ///   <0, 4, 1, 5, 2, 6, 3, 7>
  /// E.g. For a Factor of 3 (LaneLen=4):
  ///   <4, 0, 9, 5, 1, 10, 6, 2, 11, 7, 3, 12>
  /// E.g. For a Factor of 4 (LaneLen=2):
  ///   <0, 2, 6, 4, 1, 3, 7, 5>
  ///
  /// NumInputElts is the total number of elements in the input vectors.
  ///
  /// StartIndexes are the first indexes of each vector being interleaved,
  /// substituting any indexes that were undef
  /// E.g. <4, -1, 2, 5, 1, 3> (Factor=3): StartIndexes=<4, 0, 2>
  ///
  /// Note that this does not check if the input vectors are consecutive:
  /// It will return true for masks such as
  /// <0, 4, 6, 1, 5, 7> (Factor=3, LaneLen=2)
  static bool isInterleaveMask(ArrayRef<int> Mask, unsigned Factor,
                               unsigned NumInputElts,
                               SmallVectorImpl<unsigned> &StartIndexes) {
    return llvm::ShuffleVectorInst::isInterleaveMask(Mask, Factor, NumInputElts,
                                                     StartIndexes);
  }
  static bool isInterleaveMask(ArrayRef<int> Mask, unsigned Factor,
                               unsigned NumInputElts) {
    return llvm::ShuffleVectorInst::isInterleaveMask(Mask, Factor,
                                                     NumInputElts);
  }

  /// Check if the mask is a DE-interleave mask of the given factor
  /// \p Factor like:
  ///     <Index, Index+Factor, ..., Index+(NumElts-1)*Factor>
  static bool isDeInterleaveMaskOfFactor(ArrayRef<int> Mask, unsigned Factor,
                                         unsigned &Index) {
    return llvm::ShuffleVectorInst::isDeInterleaveMaskOfFactor(Mask, Factor,
                                                               Index);
  }
  static bool isDeInterleaveMaskOfFactor(ArrayRef<int> Mask, unsigned Factor) {
    return llvm::ShuffleVectorInst::isDeInterleaveMaskOfFactor(Mask, Factor);
  }

  /// Checks if the shuffle is a bit rotation of the first operand across
  /// multiple subelements, e.g:
  ///
  /// shuffle <8 x i8> %a, <8 x i8> poison, <8 x i32> <1, 0, 3, 2, 5, 4, 7, 6>
  ///
  /// could be expressed as
  ///
  /// rotl <4 x i16> %a, 8
  ///
  /// If it can be expressed as a rotation, returns the number of subelements to
  /// group by in NumSubElts and the number of bits to rotate left in RotateAmt.
  static bool isBitRotateMask(ArrayRef<int> Mask, unsigned EltSizeInBits,
                              unsigned MinSubElts, unsigned MaxSubElts,
                              unsigned &NumSubElts, unsigned &RotateAmt) {
    return llvm::ShuffleVectorInst::isBitRotateMask(
        Mask, EltSizeInBits, MinSubElts, MaxSubElts, NumSubElts, RotateAmt);
  }
};

class InsertValueInst
    : public SingleLLVMInstructionImpl<llvm::InsertValueInst> {
  /// Use Context::createInsertValueInst(). Don't call the constructor directly.
  InsertValueInst(llvm::InsertValueInst *IVI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::InsertValue, Opcode::InsertValue,
                                  IVI, Ctx) {}
  friend Context; // for InsertValueInst()

public:
  static Value *create(Value *Agg, Value *Val, ArrayRef<unsigned> Idxs,
                       BBIterator WhereIt, BasicBlock *WhereBB, Context &Ctx,
                       const Twine &Name = "");

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::InsertValue;
  }

  using idx_iterator = llvm::InsertValueInst::idx_iterator;
  inline idx_iterator idx_begin() const {
    return cast<llvm::InsertValueInst>(Val)->idx_begin();
  }
  inline idx_iterator idx_end() const {
    return cast<llvm::InsertValueInst>(Val)->idx_end();
  }
  inline iterator_range<idx_iterator> indices() const {
    return cast<llvm::InsertValueInst>(Val)->indices();
  }

  Value *getAggregateOperand() {
    return getOperand(getAggregateOperandIndex());
  }
  const Value *getAggregateOperand() const {
    return getOperand(getAggregateOperandIndex());
  }
  static unsigned getAggregateOperandIndex() {
    return llvm::InsertValueInst::getAggregateOperandIndex();
  }

  Value *getInsertedValueOperand() {
    return getOperand(getInsertedValueOperandIndex());
  }
  const Value *getInsertedValueOperand() const {
    return getOperand(getInsertedValueOperandIndex());
  }
  static unsigned getInsertedValueOperandIndex() {
    return llvm::InsertValueInst::getInsertedValueOperandIndex();
  }

  ArrayRef<unsigned> getIndices() const {
    return cast<llvm::InsertValueInst>(Val)->getIndices();
  }

  unsigned getNumIndices() const {
    return cast<llvm::InsertValueInst>(Val)->getNumIndices();
  }

  unsigned hasIndices() const {
    return cast<llvm::InsertValueInst>(Val)->hasIndices();
  }
};

class BranchInst : public SingleLLVMInstructionImpl<llvm::BranchInst> {
  /// Use Context::createBranchInst(). Don't call the constructor directly.
  BranchInst(llvm::BranchInst *BI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Br, Opcode::Br, BI, Ctx) {}
  friend Context; // for BranchInst()

public:
  static BranchInst *create(BasicBlock *IfTrue, Instruction *InsertBefore,
                            Context &Ctx);
  static BranchInst *create(BasicBlock *IfTrue, BasicBlock *InsertAtEnd,
                            Context &Ctx);
  static BranchInst *create(BasicBlock *IfTrue, BasicBlock *IfFalse,
                            Value *Cond, Instruction *InsertBefore,
                            Context &Ctx);
  static BranchInst *create(BasicBlock *IfTrue, BasicBlock *IfFalse,
                            Value *Cond, BasicBlock *InsertAtEnd, Context &Ctx);
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  bool isUnconditional() const {
    return cast<llvm::BranchInst>(Val)->isUnconditional();
  }
  bool isConditional() const {
    return cast<llvm::BranchInst>(Val)->isConditional();
  }
  Value *getCondition() const;
  void setCondition(Value *V) { setOperand(0, V); }
  unsigned getNumSuccessors() const { return 1 + isConditional(); }
  BasicBlock *getSuccessor(unsigned SuccIdx) const;
  void setSuccessor(unsigned Idx, BasicBlock *NewSucc);
  void swapSuccessors() { swapOperandsInternal(1, 2); }

private:
  struct LLVMBBToSBBB {
    Context &Ctx;
    LLVMBBToSBBB(Context &Ctx) : Ctx(Ctx) {}
    BasicBlock *operator()(llvm::BasicBlock *BB) const;
  };

  struct ConstLLVMBBToSBBB {
    Context &Ctx;
    ConstLLVMBBToSBBB(Context &Ctx) : Ctx(Ctx) {}
    const BasicBlock *operator()(const llvm::BasicBlock *BB) const;
  };

public:
  using sb_succ_op_iterator =
      mapped_iterator<llvm::BranchInst::succ_op_iterator, LLVMBBToSBBB>;
  iterator_range<sb_succ_op_iterator> successors() {
    iterator_range<llvm::BranchInst::succ_op_iterator> LLVMRange =
        cast<llvm::BranchInst>(Val)->successors();
    LLVMBBToSBBB BBMap(Ctx);
    sb_succ_op_iterator MappedBegin = map_iterator(LLVMRange.begin(), BBMap);
    sb_succ_op_iterator MappedEnd = map_iterator(LLVMRange.end(), BBMap);
    return make_range(MappedBegin, MappedEnd);
  }

  using const_sb_succ_op_iterator =
      mapped_iterator<llvm::BranchInst::const_succ_op_iterator,
                      ConstLLVMBBToSBBB>;
  iterator_range<const_sb_succ_op_iterator> successors() const {
    iterator_range<llvm::BranchInst::const_succ_op_iterator> ConstLLVMRange =
        static_cast<const llvm::BranchInst *>(cast<llvm::BranchInst>(Val))
            ->successors();
    ConstLLVMBBToSBBB ConstBBMap(Ctx);
    const_sb_succ_op_iterator ConstMappedBegin =
        map_iterator(ConstLLVMRange.begin(), ConstBBMap);
    const_sb_succ_op_iterator ConstMappedEnd =
        map_iterator(ConstLLVMRange.end(), ConstBBMap);
    return make_range(ConstMappedBegin, ConstMappedEnd);
  }
};

/// An abstract class, parent of unary instructions.
class UnaryInstruction
    : public SingleLLVMInstructionImpl<llvm::UnaryInstruction> {
protected:
  UnaryInstruction(ClassID ID, Opcode Opc, llvm::Instruction *LLVMI,
                   Context &Ctx)
      : SingleLLVMInstructionImpl(ID, Opc, LLVMI, Ctx) {}

public:
  static bool classof(const Instruction *I) {
    return isa<LoadInst>(I) || isa<CastInst>(I) || isa<FreezeInst>(I);
  }
  static bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

class ExtractValueInst : public UnaryInstruction {
  /// Use Context::createExtractValueInst() instead.
  ExtractValueInst(llvm::ExtractValueInst *EVI, Context &Ctx)
      : UnaryInstruction(ClassID::ExtractValue, Opcode::ExtractValue, EVI,
                         Ctx) {}
  friend Context; // for ExtractValueInst()

public:
  static Value *create(Value *Agg, ArrayRef<unsigned> Idxs, BBIterator WhereIt,
                       BasicBlock *WhereBB, Context &Ctx,
                       const Twine &Name = "");

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ExtractValue;
  }

  /// Returns the type of the element that would be extracted
  /// with an extractvalue instruction with the specified parameters.
  ///
  /// Null is returned if the indices are invalid for the specified type.
  static Type *getIndexedType(Type *Agg, ArrayRef<unsigned> Idxs);

  using idx_iterator = llvm::ExtractValueInst::idx_iterator;

  inline idx_iterator idx_begin() const {
    return cast<llvm::ExtractValueInst>(Val)->idx_begin();
  }
  inline idx_iterator idx_end() const {
    return cast<llvm::ExtractValueInst>(Val)->idx_end();
  }
  inline iterator_range<idx_iterator> indices() const {
    return cast<llvm::ExtractValueInst>(Val)->indices();
  }

  Value *getAggregateOperand() {
    return getOperand(getAggregateOperandIndex());
  }
  const Value *getAggregateOperand() const {
    return getOperand(getAggregateOperandIndex());
  }
  static unsigned getAggregateOperandIndex() {
    return llvm::ExtractValueInst::getAggregateOperandIndex();
  }

  ArrayRef<unsigned> getIndices() const {
    return cast<llvm::ExtractValueInst>(Val)->getIndices();
  }

  unsigned getNumIndices() const {
    return cast<llvm::ExtractValueInst>(Val)->getNumIndices();
  }

  unsigned hasIndices() const {
    return cast<llvm::ExtractValueInst>(Val)->hasIndices();
  }
};

class VAArgInst : public UnaryInstruction {
  VAArgInst(llvm::VAArgInst *FI, Context &Ctx)
      : UnaryInstruction(ClassID::VAArg, Opcode::VAArg, FI, Ctx) {}
  friend Context; // For constructor;

public:
  static VAArgInst *create(Value *List, Type *Ty, BBIterator WhereIt,
                           BasicBlock *WhereBB, Context &Ctx,
                           const Twine &Name = "");
  Value *getPointerOperand();
  const Value *getPointerOperand() const {
    return const_cast<VAArgInst *>(this)->getPointerOperand();
  }
  static unsigned getPointerOperandIndex() {
    return llvm::VAArgInst::getPointerOperandIndex();
  }
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::VAArg;
  }
};

class FreezeInst : public UnaryInstruction {
  FreezeInst(llvm::FreezeInst *FI, Context &Ctx)
      : UnaryInstruction(ClassID::Freeze, Opcode::Freeze, FI, Ctx) {}
  friend Context; // For constructor;

public:
  static FreezeInst *create(Value *V, BBIterator WhereIt, BasicBlock *WhereBB,
                            Context &Ctx, const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Freeze;
  }
};

class LoadInst final : public UnaryInstruction {
  /// Use LoadInst::create() instead of calling the constructor.
  LoadInst(llvm::LoadInst *LI, Context &Ctx)
      : UnaryInstruction(ClassID::Load, Opcode::Load, LI, Ctx) {}
  friend Context; // for LoadInst()

public:
  /// Return true if this is a load from a volatile memory location.
  bool isVolatile() const { return cast<llvm::LoadInst>(Val)->isVolatile(); }
  /// Specify whether this is a volatile load or not.
  void setVolatile(bool V);

  static LoadInst *create(Type *Ty, Value *Ptr, MaybeAlign Align,
                          Instruction *InsertBefore, Context &Ctx,
                          const Twine &Name = "");
  static LoadInst *create(Type *Ty, Value *Ptr, MaybeAlign Align,
                          Instruction *InsertBefore, bool IsVolatile,
                          Context &Ctx, const Twine &Name = "");
  static LoadInst *create(Type *Ty, Value *Ptr, MaybeAlign Align,
                          BasicBlock *InsertAtEnd, Context &Ctx,
                          const Twine &Name = "");
  static LoadInst *create(Type *Ty, Value *Ptr, MaybeAlign Align,
                          BasicBlock *InsertAtEnd, bool IsVolatile,
                          Context &Ctx, const Twine &Name = "");

  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  Value *getPointerOperand() const;
  Align getAlign() const { return cast<llvm::LoadInst>(Val)->getAlign(); }
  bool isUnordered() const { return cast<llvm::LoadInst>(Val)->isUnordered(); }
  bool isSimple() const { return cast<llvm::LoadInst>(Val)->isSimple(); }
};

class StoreInst final : public SingleLLVMInstructionImpl<llvm::StoreInst> {
  /// Use StoreInst::create().
  StoreInst(llvm::StoreInst *SI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Store, Opcode::Store, SI, Ctx) {}
  friend Context; // for StoreInst()

public:
  /// Return true if this is a store from a volatile memory location.
  bool isVolatile() const { return cast<llvm::StoreInst>(Val)->isVolatile(); }
  /// Specify whether this is a volatile store or not.
  void setVolatile(bool V);

  static StoreInst *create(Value *V, Value *Ptr, MaybeAlign Align,
                           Instruction *InsertBefore, Context &Ctx);
  static StoreInst *create(Value *V, Value *Ptr, MaybeAlign Align,
                           Instruction *InsertBefore, bool IsVolatile,
                           Context &Ctx);
  static StoreInst *create(Value *V, Value *Ptr, MaybeAlign Align,
                           BasicBlock *InsertAtEnd, Context &Ctx);
  static StoreInst *create(Value *V, Value *Ptr, MaybeAlign Align,
                           BasicBlock *InsertAtEnd, bool IsVolatile,
                           Context &Ctx);
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  Value *getValueOperand() const;
  Value *getPointerOperand() const;
  Align getAlign() const { return cast<llvm::StoreInst>(Val)->getAlign(); }
  bool isSimple() const { return cast<llvm::StoreInst>(Val)->isSimple(); }
  bool isUnordered() const { return cast<llvm::StoreInst>(Val)->isUnordered(); }
};

class UnreachableInst final : public Instruction {
  /// Use UnreachableInst::create() instead of calling the constructor.
  UnreachableInst(llvm::UnreachableInst *I, Context &Ctx)
      : Instruction(ClassID::Unreachable, Opcode::Unreachable, I, Ctx) {}
  friend Context;
  Use getOperandUseInternal(unsigned OpIdx, bool Verify) const final {
    return getOperandUseDefault(OpIdx, Verify);
  }
  SmallVector<llvm::Instruction *, 1> getLLVMInstrs() const final {
    return {cast<llvm::Instruction>(Val)};
  }

public:
  static UnreachableInst *create(Instruction *InsertBefore, Context &Ctx);
  static UnreachableInst *create(BasicBlock *InsertAtEnd, Context &Ctx);
  static bool classof(const Value *From);
  unsigned getNumSuccessors() const { return 0; }
  unsigned getUseOperandNo(const Use &Use) const final {
    llvm_unreachable("UnreachableInst has no operands!");
  }
  unsigned getNumOfIRInstrs() const final { return 1u; }
};

class ReturnInst final : public SingleLLVMInstructionImpl<llvm::ReturnInst> {
  /// Use ReturnInst::create() instead of calling the constructor.
  ReturnInst(llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Ret, Opcode::Ret, I, Ctx) {}
  ReturnInst(ClassID SubclassID, llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(SubclassID, Opcode::Ret, I, Ctx) {}
  friend class Context; // For accessing the constructor in create*()
  static ReturnInst *createCommon(Value *RetVal, IRBuilder<> &Builder,
                                  Context &Ctx);

public:
  static ReturnInst *create(Value *RetVal, Instruction *InsertBefore,
                            Context &Ctx);
  static ReturnInst *create(Value *RetVal, BasicBlock *InsertAtEnd,
                            Context &Ctx);
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Ret;
  }
  /// \Returns null if there is no return value.
  Value *getReturnValue() const;
};

class CallBase : public SingleLLVMInstructionImpl<llvm::CallBase> {
  CallBase(ClassID ID, Opcode Opc, llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(ID, Opc, I, Ctx) {}
  friend class CallInst;   // For constructor.
  friend class InvokeInst; // For constructor.
  friend class CallBrInst; // For constructor.

public:
  static bool classof(const Value *From) {
    auto Opc = From->getSubclassID();
    return Opc == Instruction::ClassID::Call ||
           Opc == Instruction::ClassID::Invoke ||
           Opc == Instruction::ClassID::CallBr;
  }

  FunctionType *getFunctionType() const;

  op_iterator data_operands_begin() { return op_begin(); }
  const_op_iterator data_operands_begin() const {
    return const_cast<CallBase *>(this)->data_operands_begin();
  }
  op_iterator data_operands_end() {
    auto *LLVMCB = cast<llvm::CallBase>(Val);
    auto Dist = LLVMCB->data_operands_end() - LLVMCB->data_operands_begin();
    return op_begin() + Dist;
  }
  const_op_iterator data_operands_end() const {
    auto *LLVMCB = cast<llvm::CallBase>(Val);
    auto Dist = LLVMCB->data_operands_end() - LLVMCB->data_operands_begin();
    return op_begin() + Dist;
  }
  iterator_range<op_iterator> data_ops() {
    return make_range(data_operands_begin(), data_operands_end());
  }
  iterator_range<const_op_iterator> data_ops() const {
    return make_range(data_operands_begin(), data_operands_end());
  }
  bool data_operands_empty() const {
    return data_operands_end() == data_operands_begin();
  }
  unsigned data_operands_size() const {
    return std::distance(data_operands_begin(), data_operands_end());
  }
  bool isDataOperand(Use U) const {
    assert(this == U.getUser() &&
           "Only valid to query with a use of this instruction!");
    return cast<llvm::CallBase>(Val)->isDataOperand(U.LLVMUse);
  }
  unsigned getDataOperandNo(Use U) const {
    assert(isDataOperand(U) && "Data operand # out of range!");
    return cast<llvm::CallBase>(Val)->getDataOperandNo(U.LLVMUse);
  }

  /// Return the total number operands (not operand bundles) used by
  /// every operand bundle in this OperandBundleUser.
  unsigned getNumTotalBundleOperands() const {
    return cast<llvm::CallBase>(Val)->getNumTotalBundleOperands();
  }

  op_iterator arg_begin() { return op_begin(); }
  const_op_iterator arg_begin() const { return op_begin(); }
  op_iterator arg_end() {
    return data_operands_end() - getNumTotalBundleOperands();
  }
  const_op_iterator arg_end() const {
    return const_cast<CallBase *>(this)->arg_end();
  }
  iterator_range<op_iterator> args() {
    return make_range(arg_begin(), arg_end());
  }
  iterator_range<const_op_iterator> args() const {
    return make_range(arg_begin(), arg_end());
  }
  bool arg_empty() const { return arg_end() == arg_begin(); }
  unsigned arg_size() const { return arg_end() - arg_begin(); }

  Value *getArgOperand(unsigned OpIdx) const {
    assert(OpIdx < arg_size() && "Out of bounds!");
    return getOperand(OpIdx);
  }
  void setArgOperand(unsigned OpIdx, Value *NewOp) {
    assert(OpIdx < arg_size() && "Out of bounds!");
    setOperand(OpIdx, NewOp);
  }

  Use getArgOperandUse(unsigned Idx) const {
    assert(Idx < arg_size() && "Out of bounds!");
    return getOperandUse(Idx);
  }
  Use getArgOperandUse(unsigned Idx) {
    assert(Idx < arg_size() && "Out of bounds!");
    return getOperandUse(Idx);
  }

  bool isArgOperand(Use U) const {
    return cast<llvm::CallBase>(Val)->isArgOperand(U.LLVMUse);
  }
  unsigned getArgOperandNo(Use U) const {
    return cast<llvm::CallBase>(Val)->getArgOperandNo(U.LLVMUse);
  }
  bool hasArgument(const Value *V) const { return is_contained(args(), V); }

  Value *getCalledOperand() const;
  Use getCalledOperandUse() const;

  Function *getCalledFunction() const;
  bool isIndirectCall() const {
    return cast<llvm::CallBase>(Val)->isIndirectCall();
  }
  bool isCallee(Use U) const {
    return cast<llvm::CallBase>(Val)->isCallee(U.LLVMUse);
  }
  Function *getCaller();
  const Function *getCaller() const {
    return const_cast<CallBase *>(this)->getCaller();
  }
  bool isMustTailCall() const {
    return cast<llvm::CallBase>(Val)->isMustTailCall();
  }
  bool isTailCall() const { return cast<llvm::CallBase>(Val)->isTailCall(); }
  Intrinsic::ID getIntrinsicID() const {
    return cast<llvm::CallBase>(Val)->getIntrinsicID();
  }
  void setCalledOperand(Value *V) { getCalledOperandUse().set(V); }
  void setCalledFunction(Function *F);
  CallingConv::ID getCallingConv() const {
    return cast<llvm::CallBase>(Val)->getCallingConv();
  }
  bool isInlineAsm() const { return cast<llvm::CallBase>(Val)->isInlineAsm(); }
};

class CallInst final : public CallBase {
  /// Use Context::createCallInst(). Don't call the
  /// constructor directly.
  CallInst(llvm::Instruction *I, Context &Ctx)
      : CallBase(ClassID::Call, Opcode::Call, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()

public:
  static CallInst *create(FunctionType *FTy, Value *Func,
                          ArrayRef<Value *> Args, BBIterator WhereIt,
                          BasicBlock *WhereBB, Context &Ctx,
                          const Twine &NameStr = "");
  static CallInst *create(FunctionType *FTy, Value *Func,
                          ArrayRef<Value *> Args, Instruction *InsertBefore,
                          Context &Ctx, const Twine &NameStr = "");
  static CallInst *create(FunctionType *FTy, Value *Func,
                          ArrayRef<Value *> Args, BasicBlock *InsertAtEnd,
                          Context &Ctx, const Twine &NameStr = "");

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Call;
  }
};

class InvokeInst final : public CallBase {
  /// Use Context::createInvokeInst(). Don't call the
  /// constructor directly.
  InvokeInst(llvm::Instruction *I, Context &Ctx)
      : CallBase(ClassID::Invoke, Opcode::Invoke, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()

public:
  static InvokeInst *create(FunctionType *FTy, Value *Func,
                            BasicBlock *IfNormal, BasicBlock *IfException,
                            ArrayRef<Value *> Args, BBIterator WhereIt,
                            BasicBlock *WhereBB, Context &Ctx,
                            const Twine &NameStr = "");
  static InvokeInst *create(FunctionType *FTy, Value *Func,
                            BasicBlock *IfNormal, BasicBlock *IfException,
                            ArrayRef<Value *> Args, Instruction *InsertBefore,
                            Context &Ctx, const Twine &NameStr = "");
  static InvokeInst *create(FunctionType *FTy, Value *Func,
                            BasicBlock *IfNormal, BasicBlock *IfException,
                            ArrayRef<Value *> Args, BasicBlock *InsertAtEnd,
                            Context &Ctx, const Twine &NameStr = "");

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Invoke;
  }
  BasicBlock *getNormalDest() const;
  BasicBlock *getUnwindDest() const;
  void setNormalDest(BasicBlock *BB);
  void setUnwindDest(BasicBlock *BB);
  LandingPadInst *getLandingPadInst() const;
  BasicBlock *getSuccessor(unsigned SuccIdx) const;
  void setSuccessor(unsigned SuccIdx, BasicBlock *NewSucc) {
    assert(SuccIdx < 2 && "Successor # out of range for invoke!");
    if (SuccIdx == 0)
      setNormalDest(NewSucc);
    else
      setUnwindDest(NewSucc);
  }
  unsigned getNumSuccessors() const {
    return cast<llvm::InvokeInst>(Val)->getNumSuccessors();
  }
};

class CallBrInst final : public CallBase {
  /// Use Context::createCallBrInst(). Don't call the
  /// constructor directly.
  CallBrInst(llvm::Instruction *I, Context &Ctx)
      : CallBase(ClassID::CallBr, Opcode::CallBr, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()

public:
  static CallBrInst *create(FunctionType *FTy, Value *Func,
                            BasicBlock *DefaultDest,
                            ArrayRef<BasicBlock *> IndirectDests,
                            ArrayRef<Value *> Args, BBIterator WhereIt,
                            BasicBlock *WhereBB, Context &Ctx,
                            const Twine &NameStr = "");
  static CallBrInst *create(FunctionType *FTy, Value *Func,
                            BasicBlock *DefaultDest,
                            ArrayRef<BasicBlock *> IndirectDests,
                            ArrayRef<Value *> Args, Instruction *InsertBefore,
                            Context &Ctx, const Twine &NameStr = "");
  static CallBrInst *create(FunctionType *FTy, Value *Func,
                            BasicBlock *DefaultDest,
                            ArrayRef<BasicBlock *> IndirectDests,
                            ArrayRef<Value *> Args, BasicBlock *InsertAtEnd,
                            Context &Ctx, const Twine &NameStr = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::CallBr;
  }
  unsigned getNumIndirectDests() const {
    return cast<llvm::CallBrInst>(Val)->getNumIndirectDests();
  }
  Value *getIndirectDestLabel(unsigned Idx) const;
  Value *getIndirectDestLabelUse(unsigned Idx) const;
  BasicBlock *getDefaultDest() const;
  BasicBlock *getIndirectDest(unsigned Idx) const;
  SmallVector<BasicBlock *, 16> getIndirectDests() const;
  void setDefaultDest(BasicBlock *BB);
  void setIndirectDest(unsigned Idx, BasicBlock *BB);
  BasicBlock *getSuccessor(unsigned Idx) const;
  unsigned getNumSuccessors() const {
    return cast<llvm::CallBrInst>(Val)->getNumSuccessors();
  }
};

class LandingPadInst : public SingleLLVMInstructionImpl<llvm::LandingPadInst> {
  LandingPadInst(llvm::LandingPadInst *LP, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::LandingPad, Opcode::LandingPad, LP,
                                  Ctx) {}
  friend class Context; // For constructor.

public:
  static LandingPadInst *create(Type *RetTy, unsigned NumReservedClauses,
                                BBIterator WhereIt, BasicBlock *WhereBB,
                                Context &Ctx, const Twine &Name = "");
  /// Return 'true' if this landingpad instruction is a
  /// cleanup. I.e., it should be run when unwinding even if its landing pad
  /// doesn't catch the exception.
  bool isCleanup() const {
    return cast<llvm::LandingPadInst>(Val)->isCleanup();
  }
  /// Indicate that this landingpad instruction is a cleanup.
  void setCleanup(bool V);

  // TODO: We are not implementing addClause() because we have no way to revert
  // it for now.

  /// Get the value of the clause at index Idx. Use isCatch/isFilter to
  /// determine what type of clause this is.
  Constant *getClause(unsigned Idx) const;

  /// Return 'true' if the clause and index Idx is a catch clause.
  bool isCatch(unsigned Idx) const {
    return cast<llvm::LandingPadInst>(Val)->isCatch(Idx);
  }
  /// Return 'true' if the clause and index Idx is a filter clause.
  bool isFilter(unsigned Idx) const {
    return cast<llvm::LandingPadInst>(Val)->isFilter(Idx);
  }
  /// Get the number of clauses for this landing pad.
  unsigned getNumClauses() const {
    return cast<llvm::LandingPadInst>(Val)->getNumOperands();
  }
  // TODO: We are not implementing reserveClauses() because we can't revert it.
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::LandingPad;
  }
};

class FuncletPadInst : public SingleLLVMInstructionImpl<llvm::FuncletPadInst> {
  FuncletPadInst(ClassID SubclassID, Opcode Opc, llvm::Instruction *I,
                 Context &Ctx)
      : SingleLLVMInstructionImpl(SubclassID, Opc, I, Ctx) {}
  friend class CatchPadInst;   // For constructor.
  friend class CleanupPadInst; // For constructor.

public:
  /// Return the number of funcletpad arguments.
  unsigned arg_size() const {
    return cast<llvm::FuncletPadInst>(Val)->arg_size();
  }
  /// Return the outer EH-pad this funclet is nested within.
  ///
  /// Note: This returns the associated CatchSwitchInst if this FuncletPadInst
  /// is a CatchPadInst.
  Value *getParentPad() const;
  void setParentPad(Value *ParentPad);
  /// Return the Idx-th funcletpad argument.
  Value *getArgOperand(unsigned Idx) const;
  /// Set the Idx-th funcletpad argument.
  void setArgOperand(unsigned Idx, Value *V);

  // TODO: Implement missing functions: arg_operands().
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::CatchPad ||
           From->getSubclassID() == ClassID::CleanupPad;
  }
};

class CatchPadInst : public FuncletPadInst {
  CatchPadInst(llvm::CatchPadInst *CPI, Context &Ctx)
      : FuncletPadInst(ClassID::CatchPad, Opcode::CatchPad, CPI, Ctx) {}
  friend class Context; // For constructor.

public:
  CatchSwitchInst *getCatchSwitch() const;
  // TODO: We have not implemented setCatchSwitch() because we can't revert it
  // for now, as there is no CatchPadInst member function that can undo it.

  static CatchPadInst *create(Value *ParentPad, ArrayRef<Value *> Args,
                              BBIterator WhereIt, BasicBlock *WhereBB,
                              Context &Ctx, const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::CatchPad;
  }
};

class CleanupPadInst : public FuncletPadInst {
  CleanupPadInst(llvm::CleanupPadInst *CPI, Context &Ctx)
      : FuncletPadInst(ClassID::CleanupPad, Opcode::CleanupPad, CPI, Ctx) {}
  friend class Context; // For constructor.

public:
  static CleanupPadInst *create(Value *ParentPad, ArrayRef<Value *> Args,
                                BBIterator WhereIt, BasicBlock *WhereBB,
                                Context &Ctx, const Twine &Name = "");
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::CleanupPad;
  }
};

class CatchReturnInst
    : public SingleLLVMInstructionImpl<llvm::CatchReturnInst> {
  CatchReturnInst(llvm::CatchReturnInst *CRI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::CatchRet, Opcode::CatchRet, CRI,
                                  Ctx) {}
  friend class Context; // For constructor.

public:
  static CatchReturnInst *create(CatchPadInst *CatchPad, BasicBlock *BB,
                                 BBIterator WhereIt, BasicBlock *WhereBB,
                                 Context &Ctx);
  CatchPadInst *getCatchPad() const;
  void setCatchPad(CatchPadInst *CatchPad);
  BasicBlock *getSuccessor() const;
  void setSuccessor(BasicBlock *NewSucc);
  unsigned getNumSuccessors() {
    return cast<llvm::CatchReturnInst>(Val)->getNumSuccessors();
  }
  Value *getCatchSwitchParentPad() const;
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::CatchRet;
  }
};

class CleanupReturnInst
    : public SingleLLVMInstructionImpl<llvm::CleanupReturnInst> {
  CleanupReturnInst(llvm::CleanupReturnInst *CRI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::CleanupRet, Opcode::CleanupRet, CRI,
                                  Ctx) {}
  friend class Context; // For constructor.

public:
  static CleanupReturnInst *create(CleanupPadInst *CleanupPad,
                                   BasicBlock *UnwindBB, BBIterator WhereIt,
                                   BasicBlock *WhereBB, Context &Ctx);
  bool hasUnwindDest() const {
    return cast<llvm::CleanupReturnInst>(Val)->hasUnwindDest();
  }
  bool unwindsToCaller() const {
    return cast<llvm::CleanupReturnInst>(Val)->unwindsToCaller();
  }
  CleanupPadInst *getCleanupPad() const;
  void setCleanupPad(CleanupPadInst *CleanupPad);
  unsigned getNumSuccessors() const {
    return cast<llvm::CleanupReturnInst>(Val)->getNumSuccessors();
  }
  BasicBlock *getUnwindDest() const;
  void setUnwindDest(BasicBlock *NewDest);

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::CleanupRet;
  }
};

class GetElementPtrInst final
    : public SingleLLVMInstructionImpl<llvm::GetElementPtrInst> {
  /// Use Context::createGetElementPtrInst(). Don't call
  /// the constructor directly.
  GetElementPtrInst(llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::GetElementPtr, Opcode::GetElementPtr,
                                  I, Ctx) {}
  GetElementPtrInst(ClassID SubclassID, llvm::Instruction *I, Context &Ctx)
      : SingleLLVMInstructionImpl(SubclassID, Opcode::GetElementPtr, I, Ctx) {}
  friend class Context; // For accessing the constructor in
                        // create*()

public:
  static Value *create(Type *Ty, Value *Ptr, ArrayRef<Value *> IdxList,
                       BBIterator WhereIt, BasicBlock *WhereBB, Context &Ctx,
                       const Twine &NameStr = "");
  static Value *create(Type *Ty, Value *Ptr, ArrayRef<Value *> IdxList,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &NameStr = "");
  static Value *create(Type *Ty, Value *Ptr, ArrayRef<Value *> IdxList,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &NameStr = "");

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::GetElementPtr;
  }

  Type *getSourceElementType() const;
  Type *getResultElementType() const;
  unsigned getAddressSpace() const {
    return cast<llvm::GetElementPtrInst>(Val)->getAddressSpace();
  }

  inline op_iterator idx_begin() { return op_begin() + 1; }
  inline const_op_iterator idx_begin() const {
    return const_cast<GetElementPtrInst *>(this)->idx_begin();
  }
  inline op_iterator idx_end() { return op_end(); }
  inline const_op_iterator idx_end() const {
    return const_cast<GetElementPtrInst *>(this)->idx_end();
  }
  inline iterator_range<op_iterator> indices() {
    return make_range(idx_begin(), idx_end());
  }
  inline iterator_range<const_op_iterator> indices() const {
    return const_cast<GetElementPtrInst *>(this)->indices();
  }

  Value *getPointerOperand() const;
  static unsigned getPointerOperandIndex() {
    return llvm::GetElementPtrInst::getPointerOperandIndex();
  }
  Type *getPointerOperandType() const;
  unsigned getPointerAddressSpace() const {
    return cast<llvm::GetElementPtrInst>(Val)->getPointerAddressSpace();
  }
  unsigned getNumIndices() const {
    return cast<llvm::GetElementPtrInst>(Val)->getNumIndices();
  }
  bool hasIndices() const {
    return cast<llvm::GetElementPtrInst>(Val)->hasIndices();
  }
  bool hasAllConstantIndices() const {
    return cast<llvm::GetElementPtrInst>(Val)->hasAllConstantIndices();
  }
  GEPNoWrapFlags getNoWrapFlags() const {
    return cast<llvm::GetElementPtrInst>(Val)->getNoWrapFlags();
  }
  bool isInBounds() const {
    return cast<llvm::GetElementPtrInst>(Val)->isInBounds();
  }
  bool hasNoUnsignedSignedWrap() const {
    return cast<llvm::GetElementPtrInst>(Val)->hasNoUnsignedSignedWrap();
  }
  bool hasNoUnsignedWrap() const {
    return cast<llvm::GetElementPtrInst>(Val)->hasNoUnsignedWrap();
  }
  bool accumulateConstantOffset(const DataLayout &DL, APInt &Offset) const {
    return cast<llvm::GetElementPtrInst>(Val)->accumulateConstantOffset(DL,
                                                                        Offset);
  }
  // TODO: Add missing member functions.
};

class CatchSwitchInst
    : public SingleLLVMInstructionImpl<llvm::CatchSwitchInst> {
public:
  CatchSwitchInst(llvm::CatchSwitchInst *CSI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::CatchSwitch, Opcode::CatchSwitch,
                                  CSI, Ctx) {}

  static CatchSwitchInst *create(Value *ParentPad, BasicBlock *UnwindBB,
                                 unsigned NumHandlers, BBIterator WhereIt,
                                 BasicBlock *WhereBB, Context &Ctx,
                                 const Twine &Name = "");

  Value *getParentPad() const;
  void setParentPad(Value *ParentPad);

  bool hasUnwindDest() const {
    return cast<llvm::CatchSwitchInst>(Val)->hasUnwindDest();
  }
  bool unwindsToCaller() const {
    return cast<llvm::CatchSwitchInst>(Val)->unwindsToCaller();
  }
  BasicBlock *getUnwindDest() const;
  void setUnwindDest(BasicBlock *UnwindDest);

  unsigned getNumHandlers() const {
    return cast<llvm::CatchSwitchInst>(Val)->getNumHandlers();
  }

private:
  static BasicBlock *handler_helper(Value *V) { return cast<BasicBlock>(V); }
  static const BasicBlock *handler_helper(const Value *V) {
    return cast<BasicBlock>(V);
  }

public:
  using DerefFnTy = BasicBlock *(*)(Value *);
  using handler_iterator = mapped_iterator<op_iterator, DerefFnTy>;
  using handler_range = iterator_range<handler_iterator>;
  using ConstDerefFnTy = const BasicBlock *(*)(const Value *);
  using const_handler_iterator =
      mapped_iterator<const_op_iterator, ConstDerefFnTy>;
  using const_handler_range = iterator_range<const_handler_iterator>;

  handler_iterator handler_begin() {
    op_iterator It = op_begin() + 1;
    if (hasUnwindDest())
      ++It;
    return handler_iterator(It, DerefFnTy(handler_helper));
  }
  const_handler_iterator handler_begin() const {
    const_op_iterator It = op_begin() + 1;
    if (hasUnwindDest())
      ++It;
    return const_handler_iterator(It, ConstDerefFnTy(handler_helper));
  }
  handler_iterator handler_end() {
    return handler_iterator(op_end(), DerefFnTy(handler_helper));
  }
  const_handler_iterator handler_end() const {
    return const_handler_iterator(op_end(), ConstDerefFnTy(handler_helper));
  }
  handler_range handlers() {
    return make_range(handler_begin(), handler_end());
  }
  const_handler_range handlers() const {
    return make_range(handler_begin(), handler_end());
  }

  void addHandler(BasicBlock *Dest);

  // TODO: removeHandler() cannot be reverted because there is no equivalent
  // addHandler() with a handler_iterator to specify the position. So we can't
  // implement it for now.

  unsigned getNumSuccessors() const { return getNumOperands() - 1; }
  BasicBlock *getSuccessor(unsigned Idx) const {
    assert(Idx < getNumSuccessors() &&
           "Successor # out of range for catchswitch!");
    return cast<BasicBlock>(getOperand(Idx + 1));
  }
  void setSuccessor(unsigned Idx, BasicBlock *NewSucc) {
    assert(Idx < getNumSuccessors() &&
           "Successor # out of range for catchswitch!");
    setOperand(Idx + 1, NewSucc);
  }

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::CatchSwitch;
  }
};

class ResumeInst : public SingleLLVMInstructionImpl<llvm::ResumeInst> {
public:
  ResumeInst(llvm::ResumeInst *CSI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Resume, Opcode::Resume, CSI, Ctx) {}

  static ResumeInst *create(Value *Exn, BBIterator WhereIt, BasicBlock *WhereBB,
                            Context &Ctx);
  Value *getValue() const;
  unsigned getNumSuccessors() const {
    return cast<llvm::ResumeInst>(Val)->getNumSuccessors();
  }
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Resume;
  }
};

class SwitchInst : public SingleLLVMInstructionImpl<llvm::SwitchInst> {
public:
  SwitchInst(llvm::SwitchInst *SI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Switch, Opcode::Switch, SI, Ctx) {}

  static constexpr const unsigned DefaultPseudoIndex =
      llvm::SwitchInst::DefaultPseudoIndex;

  static SwitchInst *create(Value *V, BasicBlock *Dest, unsigned NumCases,
                            BasicBlock::iterator WhereIt, BasicBlock *WhereBB,
                            Context &Ctx, const Twine &Name = "");

  Value *getCondition() const;
  void setCondition(Value *V);
  BasicBlock *getDefaultDest() const;
  bool defaultDestUndefined() const {
    return cast<llvm::SwitchInst>(Val)->defaultDestUndefined();
  }
  void setDefaultDest(BasicBlock *DefaultCase);
  unsigned getNumCases() const {
    return cast<llvm::SwitchInst>(Val)->getNumCases();
  }

  using CaseHandle =
      llvm::SwitchInst::CaseHandleImpl<SwitchInst, ConstantInt, BasicBlock>;
  using ConstCaseHandle =
      llvm::SwitchInst::CaseHandleImpl<const SwitchInst, const ConstantInt,
                                       const BasicBlock>;
  using CaseIt = llvm::SwitchInst::CaseIteratorImpl<CaseHandle>;
  using ConstCaseIt = llvm::SwitchInst::CaseIteratorImpl<ConstCaseHandle>;

  /// Returns a read/write iterator that points to the first case in the
  /// SwitchInst.
  CaseIt case_begin() { return CaseIt(this, 0); }
  ConstCaseIt case_begin() const { return ConstCaseIt(this, 0); }
  /// Returns a read/write iterator that points one past the last in the
  /// SwitchInst.
  CaseIt case_end() { return CaseIt(this, getNumCases()); }
  ConstCaseIt case_end() const { return ConstCaseIt(this, getNumCases()); }
  /// Iteration adapter for range-for loops.
  iterator_range<CaseIt> cases() {
    return make_range(case_begin(), case_end());
  }
  iterator_range<ConstCaseIt> cases() const {
    return make_range(case_begin(), case_end());
  }
  CaseIt case_default() { return CaseIt(this, DefaultPseudoIndex); }
  ConstCaseIt case_default() const {
    return ConstCaseIt(this, DefaultPseudoIndex);
  }
  CaseIt findCaseValue(const ConstantInt *C) {
    return CaseIt(
        this,
        const_cast<const SwitchInst *>(this)->findCaseValue(C)->getCaseIndex());
  }
  ConstCaseIt findCaseValue(const ConstantInt *C) const {
    ConstCaseIt I = llvm::find_if(cases(), [C](const ConstCaseHandle &Case) {
      return Case.getCaseValue() == C;
    });
    if (I != case_end())
      return I;
    return case_default();
  }
  ConstantInt *findCaseDest(BasicBlock *BB);

  void addCase(ConstantInt *OnVal, BasicBlock *Dest);
  /// This method removes the specified case and its successor from the switch
  /// instruction. Note that this operation may reorder the remaining cases at
  /// index idx and above.
  /// Note:
  /// This action invalidates iterators for all cases following the one removed,
  /// including the case_end() iterator. It returns an iterator for the next
  /// case.
  CaseIt removeCase(CaseIt It);

  unsigned getNumSuccessors() const {
    return cast<llvm::SwitchInst>(Val)->getNumSuccessors();
  }
  BasicBlock *getSuccessor(unsigned Idx) const;
  void setSuccessor(unsigned Idx, BasicBlock *NewSucc);
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::Switch;
  }
};

class UnaryOperator : public UnaryInstruction {
  static Opcode getUnaryOpcode(llvm::Instruction::UnaryOps UnOp) {
    switch (UnOp) {
    case llvm::Instruction::FNeg:
      return Opcode::FNeg;
    case llvm::Instruction::UnaryOpsEnd:
      llvm_unreachable("Bad UnOp!");
    }
    llvm_unreachable("Unhandled UnOp!");
  }
  UnaryOperator(llvm::UnaryOperator *UO, Context &Ctx)
      : UnaryInstruction(ClassID::UnOp, getUnaryOpcode(UO->getOpcode()), UO,
                         Ctx) {}
  friend Context; // for constructor.
public:
  static Value *create(Instruction::Opcode Op, Value *OpV, BBIterator WhereIt,
                       BasicBlock *WhereBB, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Instruction::Opcode Op, Value *OpV,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Instruction::Opcode Op, Value *OpV,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *OpV,
                                      Value *CopyFrom, BBIterator WhereIt,
                                      BasicBlock *WhereBB, Context &Ctx,
                                      const Twine &Name = "");
  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *OpV,
                                      Value *CopyFrom,
                                      Instruction *InsertBefore, Context &Ctx,
                                      const Twine &Name = "");
  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *OpV,
                                      Value *CopyFrom, BasicBlock *InsertAtEnd,
                                      Context &Ctx, const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::UnOp;
  }
};

class BinaryOperator : public SingleLLVMInstructionImpl<llvm::BinaryOperator> {
protected:
  static Opcode getBinOpOpcode(llvm::Instruction::BinaryOps BinOp) {
    switch (BinOp) {
    case llvm::Instruction::Add:
      return Opcode::Add;
    case llvm::Instruction::FAdd:
      return Opcode::FAdd;
    case llvm::Instruction::Sub:
      return Opcode::Sub;
    case llvm::Instruction::FSub:
      return Opcode::FSub;
    case llvm::Instruction::Mul:
      return Opcode::Mul;
    case llvm::Instruction::FMul:
      return Opcode::FMul;
    case llvm::Instruction::UDiv:
      return Opcode::UDiv;
    case llvm::Instruction::SDiv:
      return Opcode::SDiv;
    case llvm::Instruction::FDiv:
      return Opcode::FDiv;
    case llvm::Instruction::URem:
      return Opcode::URem;
    case llvm::Instruction::SRem:
      return Opcode::SRem;
    case llvm::Instruction::FRem:
      return Opcode::FRem;
    case llvm::Instruction::Shl:
      return Opcode::Shl;
    case llvm::Instruction::LShr:
      return Opcode::LShr;
    case llvm::Instruction::AShr:
      return Opcode::AShr;
    case llvm::Instruction::And:
      return Opcode::And;
    case llvm::Instruction::Or:
      return Opcode::Or;
    case llvm::Instruction::Xor:
      return Opcode::Xor;
    case llvm::Instruction::BinaryOpsEnd:
      llvm_unreachable("Bad BinOp!");
    }
    llvm_unreachable("Unhandled BinOp!");
  }
  BinaryOperator(llvm::BinaryOperator *BinOp, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::BinaryOperator,
                                  getBinOpOpcode(BinOp->getOpcode()), BinOp,
                                  Ctx) {}
  friend class Context; // For constructor.

public:
  static Value *create(Instruction::Opcode Op, Value *LHS, Value *RHS,
                       BBIterator WhereIt, BasicBlock *WhereBB, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Instruction::Opcode Op, Value *LHS, Value *RHS,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Instruction::Opcode Op, Value *LHS, Value *RHS,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");

  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *LHS,
                                      Value *RHS, Value *CopyFrom,
                                      BBIterator WhereIt, BasicBlock *WhereBB,
                                      Context &Ctx, const Twine &Name = "");
  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *LHS,
                                      Value *RHS, Value *CopyFrom,
                                      Instruction *InsertBefore, Context &Ctx,
                                      const Twine &Name = "");
  static Value *createWithCopiedFlags(Instruction::Opcode Op, Value *LHS,
                                      Value *RHS, Value *CopyFrom,
                                      BasicBlock *InsertAtEnd, Context &Ctx,
                                      const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::BinaryOperator;
  }
  void swapOperands() { swapOperandsInternal(0, 1); }
};

/// An or instruction, which can be marked as "disjoint", indicating that the
/// inputs don't have a 1 in the same bit position. Meaning this instruction
/// can also be treated as an add.
class PossiblyDisjointInst : public BinaryOperator {
public:
  void setIsDisjoint(bool B);
  bool isDisjoint() const {
    return cast<llvm::PossiblyDisjointInst>(Val)->isDisjoint();
  }
  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    return isa<Instruction>(From) &&
           cast<Instruction>(From)->getOpcode() == Opcode::Or;
  }
};

class AtomicRMWInst : public SingleLLVMInstructionImpl<llvm::AtomicRMWInst> {
  AtomicRMWInst(llvm::AtomicRMWInst *Atomic, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::AtomicRMW,
                                  Instruction::Opcode::AtomicRMW, Atomic, Ctx) {
  }
  friend class Context; // For constructor.

public:
  using BinOp = llvm::AtomicRMWInst::BinOp;
  BinOp getOperation() const {
    return cast<llvm::AtomicRMWInst>(Val)->getOperation();
  }
  static StringRef getOperationName(BinOp Op) {
    return llvm::AtomicRMWInst::getOperationName(Op);
  }
  static bool isFPOperation(BinOp Op) {
    return llvm::AtomicRMWInst::isFPOperation(Op);
  }
  void setOperation(BinOp Op) {
    cast<llvm::AtomicRMWInst>(Val)->setOperation(Op);
  }
  Align getAlign() const { return cast<llvm::AtomicRMWInst>(Val)->getAlign(); }
  void setAlignment(Align Align);
  bool isVolatile() const {
    return cast<llvm::AtomicRMWInst>(Val)->isVolatile();
  }
  void setVolatile(bool V);
  AtomicOrdering getOrdering() const {
    return cast<llvm::AtomicRMWInst>(Val)->getOrdering();
  }
  void setOrdering(AtomicOrdering Ordering);
  SyncScope::ID getSyncScopeID() const {
    return cast<llvm::AtomicRMWInst>(Val)->getSyncScopeID();
  }
  void setSyncScopeID(SyncScope::ID SSID);
  Value *getPointerOperand();
  const Value *getPointerOperand() const {
    return const_cast<AtomicRMWInst *>(this)->getPointerOperand();
  }
  Value *getValOperand();
  const Value *getValOperand() const {
    return const_cast<AtomicRMWInst *>(this)->getValOperand();
  }
  unsigned getPointerAddressSpace() const {
    return cast<llvm::AtomicRMWInst>(Val)->getPointerAddressSpace();
  }
  bool isFloatingPointOperation() const {
    return cast<llvm::AtomicRMWInst>(Val)->isFloatingPointOperation();
  }
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::AtomicRMW;
  }

  static AtomicRMWInst *create(BinOp Op, Value *Ptr, Value *Val,
                               MaybeAlign Align, AtomicOrdering Ordering,
                               BBIterator WhereIt, BasicBlock *WhereBB,
                               Context &Ctx,
                               SyncScope::ID SSID = SyncScope::System,
                               const Twine &Name = "");
  static AtomicRMWInst *create(BinOp Op, Value *Ptr, Value *Val,
                               MaybeAlign Align, AtomicOrdering Ordering,
                               Instruction *InsertBefore, Context &Ctx,
                               SyncScope::ID SSID = SyncScope::System,
                               const Twine &Name = "");
  static AtomicRMWInst *create(BinOp Op, Value *Ptr, Value *Val,
                               MaybeAlign Align, AtomicOrdering Ordering,
                               BasicBlock *InsertAtEnd, Context &Ctx,
                               SyncScope::ID SSID = SyncScope::System,
                               const Twine &Name = "");
};

class AtomicCmpXchgInst
    : public SingleLLVMInstructionImpl<llvm::AtomicCmpXchgInst> {
  AtomicCmpXchgInst(llvm::AtomicCmpXchgInst *Atomic, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::AtomicCmpXchg,
                                  Instruction::Opcode::AtomicCmpXchg, Atomic,
                                  Ctx) {}
  friend class Context; // For constructor.

public:
  /// Return the alignment of the memory that is being allocated by the
  /// instruction.
  Align getAlign() const {
    return cast<llvm::AtomicCmpXchgInst>(Val)->getAlign();
  }

  void setAlignment(Align Align);
  /// Return true if this is a cmpxchg from a volatile memory
  /// location.
  bool isVolatile() const {
    return cast<llvm::AtomicCmpXchgInst>(Val)->isVolatile();
  }
  /// Specify whether this is a volatile cmpxchg.
  void setVolatile(bool V);
  /// Return true if this cmpxchg may spuriously fail.
  bool isWeak() const { return cast<llvm::AtomicCmpXchgInst>(Val)->isWeak(); }
  void setWeak(bool IsWeak);
  static bool isValidSuccessOrdering(AtomicOrdering Ordering) {
    return llvm::AtomicCmpXchgInst::isValidSuccessOrdering(Ordering);
  }
  static bool isValidFailureOrdering(AtomicOrdering Ordering) {
    return llvm::AtomicCmpXchgInst::isValidFailureOrdering(Ordering);
  }
  AtomicOrdering getSuccessOrdering() const {
    return cast<llvm::AtomicCmpXchgInst>(Val)->getSuccessOrdering();
  }
  void setSuccessOrdering(AtomicOrdering Ordering);

  AtomicOrdering getFailureOrdering() const {
    return cast<llvm::AtomicCmpXchgInst>(Val)->getFailureOrdering();
  }
  void setFailureOrdering(AtomicOrdering Ordering);
  AtomicOrdering getMergedOrdering() const {
    return cast<llvm::AtomicCmpXchgInst>(Val)->getMergedOrdering();
  }
  SyncScope::ID getSyncScopeID() const {
    return cast<llvm::AtomicCmpXchgInst>(Val)->getSyncScopeID();
  }
  void setSyncScopeID(SyncScope::ID SSID);
  Value *getPointerOperand();
  const Value *getPointerOperand() const {
    return const_cast<AtomicCmpXchgInst *>(this)->getPointerOperand();
  }

  Value *getCompareOperand();
  const Value *getCompareOperand() const {
    return const_cast<AtomicCmpXchgInst *>(this)->getCompareOperand();
  }

  Value *getNewValOperand();
  const Value *getNewValOperand() const {
    return const_cast<AtomicCmpXchgInst *>(this)->getNewValOperand();
  }

  /// Returns the address space of the pointer operand.
  unsigned getPointerAddressSpace() const {
    return cast<llvm::AtomicCmpXchgInst>(Val)->getPointerAddressSpace();
  }

  static AtomicCmpXchgInst *
  create(Value *Ptr, Value *Cmp, Value *New, MaybeAlign Align,
         AtomicOrdering SuccessOrdering, AtomicOrdering FailureOrdering,
         BBIterator WhereIt, BasicBlock *WhereBB, Context &Ctx,
         SyncScope::ID SSID = SyncScope::System, const Twine &Name = "");
  static AtomicCmpXchgInst *
  create(Value *Ptr, Value *Cmp, Value *New, MaybeAlign Align,
         AtomicOrdering SuccessOrdering, AtomicOrdering FailureOrdering,
         Instruction *InsertBefore, Context &Ctx,
         SyncScope::ID SSID = SyncScope::System, const Twine &Name = "");
  static AtomicCmpXchgInst *
  create(Value *Ptr, Value *Cmp, Value *New, MaybeAlign Align,
         AtomicOrdering SuccessOrdering, AtomicOrdering FailureOrdering,
         BasicBlock *InsertAtEnd, Context &Ctx,
         SyncScope::ID SSID = SyncScope::System, const Twine &Name = "");

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::AtomicCmpXchg;
  }
};

class AllocaInst final : public UnaryInstruction {
  AllocaInst(llvm::AllocaInst *AI, Context &Ctx)
      : UnaryInstruction(ClassID::Alloca, Instruction::Opcode::Alloca, AI,
                         Ctx) {}
  friend class Context; // For constructor.

public:
  static AllocaInst *create(Type *Ty, unsigned AddrSpace, BBIterator WhereIt,
                            BasicBlock *WhereBB, Context &Ctx,
                            Value *ArraySize = nullptr, const Twine &Name = "");
  static AllocaInst *create(Type *Ty, unsigned AddrSpace,
                            Instruction *InsertBefore, Context &Ctx,
                            Value *ArraySize = nullptr, const Twine &Name = "");
  static AllocaInst *create(Type *Ty, unsigned AddrSpace,
                            BasicBlock *InsertAtEnd, Context &Ctx,
                            Value *ArraySize = nullptr, const Twine &Name = "");

  /// Return true if there is an allocation size parameter to the allocation
  /// instruction that is not 1.
  bool isArrayAllocation() const {
    return cast<llvm::AllocaInst>(Val)->isArrayAllocation();
  }
  /// Get the number of elements allocated. For a simple allocation of a single
  /// element, this will return a constant 1 value.
  Value *getArraySize();
  const Value *getArraySize() const {
    return const_cast<AllocaInst *>(this)->getArraySize();
  }
  /// Overload to return most specific pointer type.
  PointerType *getType() const;
  /// Return the address space for the allocation.
  unsigned getAddressSpace() const {
    return cast<llvm::AllocaInst>(Val)->getAddressSpace();
  }
  /// Get allocation size in bytes. Returns std::nullopt if size can't be
  /// determined, e.g. in case of a VLA.
  std::optional<TypeSize> getAllocationSize(const DataLayout &DL) const {
    return cast<llvm::AllocaInst>(Val)->getAllocationSize(DL);
  }
  /// Get allocation size in bits. Returns std::nullopt if size can't be
  /// determined, e.g. in case of a VLA.
  std::optional<TypeSize> getAllocationSizeInBits(const DataLayout &DL) const {
    return cast<llvm::AllocaInst>(Val)->getAllocationSizeInBits(DL);
  }
  /// Return the type that is being allocated by the instruction.
  Type *getAllocatedType() const;
  /// for use only in special circumstances that need to generically
  /// transform a whole instruction (eg: IR linking and vectorization).
  void setAllocatedType(Type *Ty);
  /// Return the alignment of the memory that is being allocated by the
  /// instruction.
  Align getAlign() const { return cast<llvm::AllocaInst>(Val)->getAlign(); }
  void setAlignment(Align Align);
  /// Return true if this alloca is in the entry block of the function and is a
  /// constant size. If so, the code generator will fold it into the
  /// prolog/epilog code, so it is basically free.
  bool isStaticAlloca() const {
    return cast<llvm::AllocaInst>(Val)->isStaticAlloca();
  }
  /// Return true if this alloca is used as an inalloca argument to a call. Such
  /// allocas are never considered static even if they are in the entry block.
  bool isUsedWithInAlloca() const {
    return cast<llvm::AllocaInst>(Val)->isUsedWithInAlloca();
  }
  /// Specify whether this alloca is used to represent the arguments to a call.
  void setUsedWithInAlloca(bool V);

  static bool classof(const Value *From) {
    if (auto *I = dyn_cast<Instruction>(From))
      return I->getSubclassID() == Instruction::ClassID::Alloca;
    return false;
  }
};

class CastInst : public UnaryInstruction {
  static Opcode getCastOpcode(llvm::Instruction::CastOps CastOp) {
    switch (CastOp) {
    case llvm::Instruction::ZExt:
      return Opcode::ZExt;
    case llvm::Instruction::SExt:
      return Opcode::SExt;
    case llvm::Instruction::FPToUI:
      return Opcode::FPToUI;
    case llvm::Instruction::FPToSI:
      return Opcode::FPToSI;
    case llvm::Instruction::FPExt:
      return Opcode::FPExt;
    case llvm::Instruction::PtrToInt:
      return Opcode::PtrToInt;
    case llvm::Instruction::IntToPtr:
      return Opcode::IntToPtr;
    case llvm::Instruction::SIToFP:
      return Opcode::SIToFP;
    case llvm::Instruction::UIToFP:
      return Opcode::UIToFP;
    case llvm::Instruction::Trunc:
      return Opcode::Trunc;
    case llvm::Instruction::FPTrunc:
      return Opcode::FPTrunc;
    case llvm::Instruction::BitCast:
      return Opcode::BitCast;
    case llvm::Instruction::AddrSpaceCast:
      return Opcode::AddrSpaceCast;
    case llvm::Instruction::CastOpsEnd:
      llvm_unreachable("Bad CastOp!");
    }
    llvm_unreachable("Unhandled CastOp!");
  }
  /// Use Context::createCastInst(). Don't call the
  /// constructor directly.
  CastInst(llvm::CastInst *CI, Context &Ctx)
      : UnaryInstruction(ClassID::Cast, getCastOpcode(CI->getOpcode()), CI,
                         Ctx) {}
  friend Context; // for SBCastInstruction()

public:
  static Value *create(Type *DestTy, Opcode Op, Value *Operand,
                       BBIterator WhereIt, BasicBlock *WhereBB, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Type *DestTy, Opcode Op, Value *Operand,
                       Instruction *InsertBefore, Context &Ctx,
                       const Twine &Name = "");
  static Value *create(Type *DestTy, Opcode Op, Value *Operand,
                       BasicBlock *InsertAtEnd, Context &Ctx,
                       const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const Value *From);
  Type *getSrcTy() const;
  Type *getDestTy() const;
};

/// Instruction that can have a nneg flag (zext/uitofp).
class PossiblyNonNegInst : public CastInst {
public:
  bool hasNonNeg() const {
    return cast<llvm::PossiblyNonNegInst>(Val)->hasNonNeg();
  }
  void setNonNeg(bool B);
  /// For isa/dyn_cast.
  static bool classof(const Value *From) {
    if (auto *I = dyn_cast<Instruction>(From)) {
      switch (I->getOpcode()) {
      case Opcode::ZExt:
      case Opcode::UIToFP:
        return true;
      default:
        return false;
      }
    }
    return false;
  }
};

// Helper class to simplify stamping out CastInst subclasses.
template <Instruction::Opcode Op> class CastInstImpl : public CastInst {
public:
  static Value *create(Value *Src, Type *DestTy, BBIterator WhereIt,
                       BasicBlock *WhereBB, Context &Ctx,
                       const Twine &Name = "") {
    return CastInst::create(DestTy, Op, Src, WhereIt, WhereBB, Ctx, Name);
  }
  static Value *create(Value *Src, Type *DestTy, Instruction *InsertBefore,
                       Context &Ctx, const Twine &Name = "") {
    return create(Src, DestTy, InsertBefore->getIterator(),
                  InsertBefore->getParent(), Ctx, Name);
  }
  static Value *create(Value *Src, Type *DestTy, BasicBlock *InsertAtEnd,
                       Context &Ctx, const Twine &Name = "") {
    return create(Src, DestTy, InsertAtEnd->end(), InsertAtEnd, Ctx, Name);
  }

  static bool classof(const Value *From) {
    if (auto *I = dyn_cast<Instruction>(From))
      return I->getOpcode() == Op;
    return false;
  }
};

class TruncInst final : public CastInstImpl<Instruction::Opcode::Trunc> {};
class ZExtInst final : public CastInstImpl<Instruction::Opcode::ZExt> {};
class SExtInst final : public CastInstImpl<Instruction::Opcode::SExt> {};
class FPTruncInst final : public CastInstImpl<Instruction::Opcode::FPTrunc> {};
class FPExtInst final : public CastInstImpl<Instruction::Opcode::FPExt> {};
class UIToFPInst final : public CastInstImpl<Instruction::Opcode::UIToFP> {};
class SIToFPInst final : public CastInstImpl<Instruction::Opcode::SIToFP> {};
class FPToUIInst final : public CastInstImpl<Instruction::Opcode::FPToUI> {};
class FPToSIInst final : public CastInstImpl<Instruction::Opcode::FPToSI> {};
class IntToPtrInst final : public CastInstImpl<Instruction::Opcode::IntToPtr> {
};
class PtrToIntInst final : public CastInstImpl<Instruction::Opcode::PtrToInt> {
};
class BitCastInst final : public CastInstImpl<Instruction::Opcode::BitCast> {};
class AddrSpaceCastInst final
    : public CastInstImpl<Instruction::Opcode::AddrSpaceCast> {
public:
  /// \Returns the pointer operand.
  Value *getPointerOperand() { return getOperand(0); }
  /// \Returns the pointer operand.
  const Value *getPointerOperand() const {
    return const_cast<AddrSpaceCastInst *>(this)->getPointerOperand();
  }
  /// \Returns the operand index of the pointer operand.
  static unsigned getPointerOperandIndex() { return 0u; }
  /// \Returns the address space of the pointer operand.
  unsigned getSrcAddressSpace() const {
    return getPointerOperand()->getType()->getPointerAddressSpace();
  }
  /// \Returns the address space of the result.
  unsigned getDestAddressSpace() const {
    return getType()->getPointerAddressSpace();
  }
};

class PHINode final : public SingleLLVMInstructionImpl<llvm::PHINode> {
  /// Use Context::createPHINode(). Don't call the constructor directly.
  PHINode(llvm::PHINode *PHI, Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::PHI, Opcode::PHI, PHI, Ctx) {}
  friend Context; // for PHINode()
  /// Helper for mapped_iterator.
  struct LLVMBBToBB {
    Context &Ctx;
    LLVMBBToBB(Context &Ctx) : Ctx(Ctx) {}
    BasicBlock *operator()(llvm::BasicBlock *LLVMBB) const;
  };

public:
  static PHINode *create(Type *Ty, unsigned NumReservedValues,
                         Instruction *InsertBefore, Context &Ctx,
                         const Twine &Name = "");
  /// For isa/dyn_cast.
  static bool classof(const Value *From);

  using const_block_iterator =
      mapped_iterator<llvm::PHINode::const_block_iterator, LLVMBBToBB>;

  const_block_iterator block_begin() const {
    LLVMBBToBB BBGetter(Ctx);
    return const_block_iterator(cast<llvm::PHINode>(Val)->block_begin(),
                                BBGetter);
  }
  const_block_iterator block_end() const {
    LLVMBBToBB BBGetter(Ctx);
    return const_block_iterator(cast<llvm::PHINode>(Val)->block_end(),
                                BBGetter);
  }
  iterator_range<const_block_iterator> blocks() const {
    return make_range(block_begin(), block_end());
  }

  op_range incoming_values() { return operands(); }

  const_op_range incoming_values() const { return operands(); }

  unsigned getNumIncomingValues() const {
    return cast<llvm::PHINode>(Val)->getNumIncomingValues();
  }
  Value *getIncomingValue(unsigned Idx) const;
  void setIncomingValue(unsigned Idx, Value *V);
  static unsigned getOperandNumForIncomingValue(unsigned Idx) {
    return llvm::PHINode::getOperandNumForIncomingValue(Idx);
  }
  static unsigned getIncomingValueNumForOperand(unsigned Idx) {
    return llvm::PHINode::getIncomingValueNumForOperand(Idx);
  }
  BasicBlock *getIncomingBlock(unsigned Idx) const;
  BasicBlock *getIncomingBlock(const Use &U) const;

  void setIncomingBlock(unsigned Idx, BasicBlock *BB);

  void addIncoming(Value *V, BasicBlock *BB);

  Value *removeIncomingValue(unsigned Idx);
  Value *removeIncomingValue(BasicBlock *BB);

  int getBasicBlockIndex(const BasicBlock *BB) const;
  Value *getIncomingValueForBlock(const BasicBlock *BB) const;

  Value *hasConstantValue() const;

  bool hasConstantOrUndefValue() const {
    return cast<llvm::PHINode>(Val)->hasConstantOrUndefValue();
  }
  bool isComplete() const { return cast<llvm::PHINode>(Val)->isComplete(); }
  void replaceIncomingBlockWith(const BasicBlock *Old, BasicBlock *New);
  void removeIncomingValueIf(function_ref<bool(unsigned)> Predicate);
  // TODO: Implement
  // void copyIncomingBlocks(iterator_range<const_block_iterator> BBRange,
  //                         uint32_t ToIdx = 0)
};

// Wraps a static function that takes a single Predicate parameter
// LLVMValType should be the type of the wrapped class
#define WRAP_STATIC_PREDICATE(FunctionName)                                    \
  static auto FunctionName(Predicate P) { return LLVMValType::FunctionName(P); }
// Wraps a member function that takes no parameters
// LLVMValType should be the type of the wrapped class
#define WRAP_MEMBER(FunctionName)                                              \
  auto FunctionName() const { return cast<LLVMValType>(Val)->FunctionName(); }
// Wraps both--a common idiom in the CmpInst classes
#define WRAP_BOTH(FunctionName)                                                \
  WRAP_STATIC_PREDICATE(FunctionName)                                          \
  WRAP_MEMBER(FunctionName)

class CmpInst : public SingleLLVMInstructionImpl<llvm::CmpInst> {
protected:
  using LLVMValType = llvm::CmpInst;
  /// Use Context::createCmpInst(). Don't call the constructor directly.
  CmpInst(llvm::CmpInst *CI, Context &Ctx, ClassID Id, Opcode Opc)
      : SingleLLVMInstructionImpl(Id, Opc, CI, Ctx) {}
  friend Context; // for CmpInst()
  static Value *createCommon(Value *Cond, Value *True, Value *False,
                             const Twine &Name, IRBuilder<> &Builder,
                             Context &Ctx);

public:
  using Predicate = llvm::CmpInst::Predicate;

  static CmpInst *create(Predicate Pred, Value *S1, Value *S2,
                         Instruction *InsertBefore, Context &Ctx,
                         const Twine &Name = "");
  static CmpInst *createWithCopiedFlags(Predicate Pred, Value *S1, Value *S2,
                                        const Instruction *FlagsSource,
                                        Instruction *InsertBefore, Context &Ctx,
                                        const Twine &Name = "");
  void setPredicate(Predicate P);
  void swapOperands();

  WRAP_MEMBER(getPredicate);
  WRAP_BOTH(isFPPredicate);
  WRAP_BOTH(isIntPredicate);
  WRAP_STATIC_PREDICATE(getPredicateName);
  WRAP_BOTH(getInversePredicate);
  WRAP_BOTH(getOrderedPredicate);
  WRAP_BOTH(getUnorderedPredicate);
  WRAP_BOTH(getSwappedPredicate);
  WRAP_BOTH(isStrictPredicate);
  WRAP_BOTH(isNonStrictPredicate);
  WRAP_BOTH(getStrictPredicate);
  WRAP_BOTH(getNonStrictPredicate);
  WRAP_BOTH(getFlippedStrictnessPredicate);
  WRAP_MEMBER(isCommutative);
  WRAP_BOTH(isEquality);
  WRAP_BOTH(isRelational);
  WRAP_BOTH(isSigned);
  WRAP_BOTH(getSignedPredicate);
  WRAP_BOTH(getUnsignedPredicate);
  WRAP_BOTH(getFlippedSignednessPredicate);
  WRAP_BOTH(isTrueWhenEqual);
  WRAP_BOTH(isFalseWhenEqual);
  WRAP_BOTH(isUnsigned);
  WRAP_STATIC_PREDICATE(isOrdered);
  WRAP_STATIC_PREDICATE(isUnordered);

  static bool isImpliedTrueByMatchingCmp(Predicate Pred1, Predicate Pred2) {
    return llvm::CmpInst::isImpliedTrueByMatchingCmp(Pred1, Pred2);
  }
  static bool isImpliedFalseByMatchingCmp(Predicate Pred1, Predicate Pred2) {
    return llvm::CmpInst::isImpliedFalseByMatchingCmp(Pred1, Pred2);
  }

  /// Method for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ICmp ||
           From->getSubclassID() == ClassID::FCmp;
  }

  /// Create a result type for fcmp/icmp
  static Type *makeCmpResultType(Type *OpndType);

#ifndef NDEBUG
  void dumpOS(raw_ostream &OS) const override;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

class ICmpInst : public CmpInst {
  /// Use Context::createICmpInst(). Don't call the constructor directly.
  ICmpInst(llvm::ICmpInst *CI, Context &Ctx)
      : CmpInst(CI, Ctx, ClassID::ICmp, Opcode::ICmp) {}
  friend class Context; // For constructor.
  using LLVMValType = llvm::ICmpInst;

public:
  void swapOperands();

  WRAP_BOTH(getSignedPredicate);
  WRAP_BOTH(getUnsignedPredicate);
  WRAP_BOTH(isEquality);
  WRAP_MEMBER(isCommutative);
  WRAP_MEMBER(isRelational);
  WRAP_STATIC_PREDICATE(isGT);
  WRAP_STATIC_PREDICATE(isLT);
  WRAP_STATIC_PREDICATE(isGE);
  WRAP_STATIC_PREDICATE(isLE);

  static auto predicates() { return llvm::ICmpInst::predicates(); }
  static bool compare(const APInt &LHS, const APInt &RHS,
                      ICmpInst::Predicate Pred) {
    return llvm::ICmpInst::compare(LHS, RHS, Pred);
  }

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::ICmp;
  }
};

class FCmpInst : public CmpInst {
  /// Use Context::createFCmpInst(). Don't call the constructor directly.
  FCmpInst(llvm::FCmpInst *CI, Context &Ctx)
      : CmpInst(CI, Ctx, ClassID::FCmp, Opcode::FCmp) {}
  friend class Context; // For constructor.
  using LLVMValType = llvm::FCmpInst;

public:
  void swapOperands();

  WRAP_BOTH(isEquality);
  WRAP_MEMBER(isCommutative);
  WRAP_MEMBER(isRelational);

  static auto predicates() { return llvm::FCmpInst::predicates(); }
  static bool compare(const APFloat &LHS, const APFloat &RHS,
                      FCmpInst::Predicate Pred) {
    return llvm::FCmpInst::compare(LHS, RHS, Pred);
  }

  static bool classof(const Value *From) {
    return From->getSubclassID() == ClassID::FCmp;
  }
};

#undef WRAP_STATIC_PREDICATE
#undef WRAP_MEMBER
#undef WRAP_BOTH

/// An LLLVM Instruction that has no SandboxIR equivalent class gets mapped to
/// an OpaqueInstr.
class OpaqueInst : public SingleLLVMInstructionImpl<llvm::Instruction> {
  OpaqueInst(llvm::Instruction *I, sandboxir::Context &Ctx)
      : SingleLLVMInstructionImpl(ClassID::Opaque, Opcode::Opaque, I, Ctx) {}
  OpaqueInst(ClassID SubclassID, llvm::Instruction *I, sandboxir::Context &Ctx)
      : SingleLLVMInstructionImpl(SubclassID, Opcode::Opaque, I, Ctx) {}
  friend class Context; // For constructor.

public:
  static bool classof(const sandboxir::Value *From) {
    return From->getSubclassID() == ClassID::Opaque;
  }
};

} // namespace llvm::sandboxir

#endif // LLVM_SANDBOXIR_INSTRUCTION_H
