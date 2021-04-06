//===-- llvm/Instruction.h - Instruction class definition -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Instruction class, which is the
// base class for all of the LLVM instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INSTRUCTION_H
#define LLVM_IR_INSTRUCTION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Bitfields.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/SymbolTableListTraits.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <utility>

namespace llvm {

class BasicBlock;
class FastMathFlags;
class MDNode;
class Module;
struct AAMDNodes;

template <> struct ilist_alloc_traits<Instruction> {
  static inline void deleteNode(Instruction *V);
};

class Instruction : public User,
                    public ilist_node_with_parent<Instruction, BasicBlock> {
  BasicBlock *Parent;
  DebugLoc DbgLoc;                         // 'dbg' Metadata cache.

  /// Relative order of this instruction in its parent basic block. Used for
  /// O(1) local dominance checks between instructions.
  mutable unsigned Order = 0;

protected:
  // The 15 first bits of `Value::SubclassData` are available for subclasses of
  // `Instruction` to use.
  using OpaqueField = Bitfield::Element<uint16_t, 0, 15>;

  // Template alias so that all Instruction storing alignment use the same
  // definiton.
  // Valid alignments are powers of two from 2^0 to 2^MaxAlignmentExponent =
  // 2^29. We store them as Log2(Alignment), so we need 5 bits to encode the 30
  // possible values.
  template <unsigned Offset>
  using AlignmentBitfieldElementT =
      typename Bitfield::Element<unsigned, Offset, 5,
                                 Value::MaxAlignmentExponent>;

  template <unsigned Offset>
  using BoolBitfieldElementT = typename Bitfield::Element<bool, Offset, 1>;

  template <unsigned Offset>
  using AtomicOrderingBitfieldElementT =
      typename Bitfield::Element<AtomicOrdering, Offset, 3,
                                 AtomicOrdering::LAST>;

private:
  // The last bit is used to store whether the instruction has metadata attached
  // or not.
  using HasMetadataField = Bitfield::Element<bool, 15, 1>;

protected:
  ~Instruction(); // Use deleteValue() to delete a generic Instruction.

public:
  Instruction(const Instruction &) = delete;
  Instruction &operator=(const Instruction &) = delete;

  /// Specialize the methods defined in Value, as we know that an instruction
  /// can only be used by other instructions.
  Instruction       *user_back()       { return cast<Instruction>(*user_begin());}
  const Instruction *user_back() const { return cast<Instruction>(*user_begin());}

  inline const BasicBlock *getParent() const { return Parent; }
  inline       BasicBlock *getParent()       { return Parent; }

  /// Return the module owning the function this instruction belongs to
  /// or nullptr it the function does not have a module.
  ///
  /// Note: this is undefined behavior if the instruction does not have a
  /// parent, or the parent basic block does not have a parent function.
  const Module *getModule() const;
  Module *getModule() {
    return const_cast<Module *>(
                           static_cast<const Instruction *>(this)->getModule());
  }

  /// Return the function this instruction belongs to.
  ///
  /// Note: it is undefined behavior to call this on an instruction not
  /// currently inserted into a function.
  const Function *getFunction() const;
  Function *getFunction() {
    return const_cast<Function *>(
                         static_cast<const Instruction *>(this)->getFunction());
  }

  /// This method unlinks 'this' from the containing basic block, but does not
  /// delete it.
  void removeFromParent();

  /// This method unlinks 'this' from the containing basic block and deletes it.
  ///
  /// \returns an iterator pointing to the element after the erased one
  SymbolTableList<Instruction>::iterator eraseFromParent();

  /// Insert an unlinked instruction into a basic block immediately before
  /// the specified instruction.
  void insertBefore(Instruction *InsertPos);

  /// Insert an unlinked instruction into a basic block immediately after the
  /// specified instruction.
  void insertAfter(Instruction *InsertPos);

  /// Unlink this instruction from its current basic block and insert it into
  /// the basic block that MovePos lives in, right before MovePos.
  void moveBefore(Instruction *MovePos);

  /// Unlink this instruction and insert into BB before I.
  ///
  /// \pre I is a valid iterator into BB.
  void moveBefore(BasicBlock &BB, SymbolTableList<Instruction>::iterator I);

  /// Unlink this instruction from its current basic block and insert it into
  /// the basic block that MovePos lives in, right after MovePos.
  void moveAfter(Instruction *MovePos);

  /// Given an instruction Other in the same basic block as this instruction,
  /// return true if this instruction comes before Other. In this worst case,
  /// this takes linear time in the number of instructions in the block. The
  /// results are cached, so in common cases when the block remains unmodified,
  /// it takes constant time.
  bool comesBefore(const Instruction *Other) const;

  //===--------------------------------------------------------------------===//
  // Subclass classification.
  //===--------------------------------------------------------------------===//

  /// Returns a member of one of the enums like Instruction::Add.
  unsigned getOpcode() const { return getValueID() - InstructionVal; }

  const char *getOpcodeName() const { return getOpcodeName(getOpcode()); }
  bool isTerminator() const { return isTerminator(getOpcode()); }
  bool isUnaryOp() const { return isUnaryOp(getOpcode()); }
  bool isBinaryOp() const { return isBinaryOp(getOpcode()); }
  bool isIntDivRem() const { return isIntDivRem(getOpcode()); }
  bool isShift() const { return isShift(getOpcode()); }
  bool isCast() const { return isCast(getOpcode()); }
  bool isFuncletPad() const { return isFuncletPad(getOpcode()); }
  bool isExceptionalTerminator() const {
    return isExceptionalTerminator(getOpcode());
  }
  bool isIndirectTerminator() const {
    return isIndirectTerminator(getOpcode());
  }

  static const char* getOpcodeName(unsigned OpCode);

  static inline bool isTerminator(unsigned OpCode) {
    return OpCode >= TermOpsBegin && OpCode < TermOpsEnd;
  }

  static inline bool isUnaryOp(unsigned Opcode) {
    return Opcode >= UnaryOpsBegin && Opcode < UnaryOpsEnd;
  }
  static inline bool isBinaryOp(unsigned Opcode) {
    return Opcode >= BinaryOpsBegin && Opcode < BinaryOpsEnd;
  }

  static inline bool isIntDivRem(unsigned Opcode) {
    return Opcode == UDiv || Opcode == SDiv || Opcode == URem || Opcode == SRem;
  }

  /// Determine if the Opcode is one of the shift instructions.
  static inline bool isShift(unsigned Opcode) {
    return Opcode >= Shl && Opcode <= AShr;
  }

  /// Return true if this is a logical shift left or a logical shift right.
  inline bool isLogicalShift() const {
    return getOpcode() == Shl || getOpcode() == LShr;
  }

  /// Return true if this is an arithmetic shift right.
  inline bool isArithmeticShift() const {
    return getOpcode() == AShr;
  }

  /// Determine if the Opcode is and/or/xor.
  static inline bool isBitwiseLogicOp(unsigned Opcode) {
    return Opcode == And || Opcode == Or || Opcode == Xor;
  }

  /// Return true if this is and/or/xor.
  inline bool isBitwiseLogicOp() const {
    return isBitwiseLogicOp(getOpcode());
  }

  /// Determine if the OpCode is one of the CastInst instructions.
  static inline bool isCast(unsigned OpCode) {
    return OpCode >= CastOpsBegin && OpCode < CastOpsEnd;
  }

  /// Determine if the OpCode is one of the FuncletPadInst instructions.
  static inline bool isFuncletPad(unsigned OpCode) {
    return OpCode >= FuncletPadOpsBegin && OpCode < FuncletPadOpsEnd;
  }

  /// Returns true if the OpCode is a terminator related to exception handling.
  static inline bool isExceptionalTerminator(unsigned OpCode) {
    switch (OpCode) {
    case Instruction::CatchSwitch:
    case Instruction::CatchRet:
    case Instruction::CleanupRet:
    case Instruction::Invoke:
    case Instruction::Resume:
      return true;
    default:
      return false;
    }
  }

  /// Returns true if the OpCode is a terminator with indirect targets.
  static inline bool isIndirectTerminator(unsigned OpCode) {
    switch (OpCode) {
    case Instruction::IndirectBr:
    case Instruction::CallBr:
      return true;
    default:
      return false;
    }
  }

  //===--------------------------------------------------------------------===//
  // Metadata manipulation.
  //===--------------------------------------------------------------------===//

  /// Return true if this instruction has any metadata attached to it.
  bool hasMetadata() const { return DbgLoc || Value::hasMetadata(); }

  /// Return true if this instruction has metadata attached to it other than a
  /// debug location.
  bool hasMetadataOtherThanDebugLoc() const { return Value::hasMetadata(); }

  /// Return true if this instruction has the given type of metadata attached.
  bool hasMetadata(unsigned KindID) const {
    return getMetadata(KindID) != nullptr;
  }

  /// Return true if this instruction has the given type of metadata attached.
  bool hasMetadata(StringRef Kind) const {
    return getMetadata(Kind) != nullptr;
  }

  /// Get the metadata of given kind attached to this Instruction.
  /// If the metadata is not found then return null.
  MDNode *getMetadata(unsigned KindID) const {
    if (!hasMetadata()) return nullptr;
    return getMetadataImpl(KindID);
  }

  /// Get the metadata of given kind attached to this Instruction.
  /// If the metadata is not found then return null.
  MDNode *getMetadata(StringRef Kind) const {
    if (!hasMetadata()) return nullptr;
    return getMetadataImpl(Kind);
  }

  /// Get all metadata attached to this Instruction. The first element of each
  /// pair returned is the KindID, the second element is the metadata value.
  /// This list is returned sorted by the KindID.
  void
  getAllMetadata(SmallVectorImpl<std::pair<unsigned, MDNode *>> &MDs) const {
    if (hasMetadata())
      getAllMetadataImpl(MDs);
  }

  /// This does the same thing as getAllMetadata, except that it filters out the
  /// debug location.
  void getAllMetadataOtherThanDebugLoc(
      SmallVectorImpl<std::pair<unsigned, MDNode *>> &MDs) const {
    Value::getAllMetadata(MDs);
  }

  /// Fills the AAMDNodes structure with AA metadata from this instruction.
  /// When Merge is true, the existing AA metadata is merged with that from this
  /// instruction providing the most-general result.
  void getAAMetadata(AAMDNodes &N, bool Merge = false) const;

  /// Set the metadata of the specified kind to the specified node. This updates
  /// or replaces metadata if already present, or removes it if Node is null.
  void setMetadata(unsigned KindID, MDNode *Node);
  void setMetadata(StringRef Kind, MDNode *Node);

  /// Copy metadata from \p SrcInst to this instruction. \p WL, if not empty,
  /// specifies the list of meta data that needs to be copied. If \p WL is
  /// empty, all meta data will be copied.
  void copyMetadata(const Instruction &SrcInst,
                    ArrayRef<unsigned> WL = ArrayRef<unsigned>());

  /// If the instruction has "branch_weights" MD_prof metadata and the MDNode
  /// has three operands (including name string), swap the order of the
  /// metadata.
  void swapProfMetadata();

  /// Drop all unknown metadata except for debug locations.
  /// @{
  /// Passes are required to drop metadata they don't understand. This is a
  /// convenience method for passes to do so.
  void dropUnknownNonDebugMetadata(ArrayRef<unsigned> KnownIDs);
  void dropUnknownNonDebugMetadata() {
    return dropUnknownNonDebugMetadata(None);
  }
  void dropUnknownNonDebugMetadata(unsigned ID1) {
    return dropUnknownNonDebugMetadata(makeArrayRef(ID1));
  }
  void dropUnknownNonDebugMetadata(unsigned ID1, unsigned ID2) {
    unsigned IDs[] = {ID1, ID2};
    return dropUnknownNonDebugMetadata(IDs);
  }
  /// @}

  /// Adds an !annotation metadata node with \p Annotation to this instruction.
  /// If this instruction already has !annotation metadata, append \p Annotation
  /// to the existing node.
  void addAnnotationMetadata(StringRef Annotation);

  /// Sets the metadata on this instruction from the AAMDNodes structure.
  void setAAMetadata(const AAMDNodes &N);

  /// Retrieve the raw weight values of a conditional branch or select.
  /// Returns true on success with profile weights filled in.
  /// Returns false if no metadata or invalid metadata was found.
  bool extractProfMetadata(uint64_t &TrueVal, uint64_t &FalseVal) const;

  /// Retrieve total raw weight values of a branch.
  /// Returns true on success with profile total weights filled in.
  /// Returns false if no metadata was found.
  bool extractProfTotalWeight(uint64_t &TotalVal) const;

  /// Set the debug location information for this instruction.
  void setDebugLoc(DebugLoc Loc) { DbgLoc = std::move(Loc); }

  /// Return the debug location for this node as a DebugLoc.
  const DebugLoc &getDebugLoc() const { return DbgLoc; }

  /// Set or clear the nuw flag on this instruction, which must be an operator
  /// which supports this flag. See LangRef.html for the meaning of this flag.
  void setHasNoUnsignedWrap(bool b = true);

  /// Set or clear the nsw flag on this instruction, which must be an operator
  /// which supports this flag. See LangRef.html for the meaning of this flag.
  void setHasNoSignedWrap(bool b = true);

  /// Set or clear the exact flag on this instruction, which must be an operator
  /// which supports this flag. See LangRef.html for the meaning of this flag.
  void setIsExact(bool b = true);

  /// Determine whether the no unsigned wrap flag is set.
  bool hasNoUnsignedWrap() const;

  /// Determine whether the no signed wrap flag is set.
  bool hasNoSignedWrap() const;

  /// Drops flags that may cause this instruction to evaluate to poison despite
  /// having non-poison inputs.
  void dropPoisonGeneratingFlags();

  /// Determine whether the exact flag is set.
  bool isExact() const;

  /// Set or clear all fast-math-flags on this instruction, which must be an
  /// operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setFast(bool B);

  /// Set or clear the reassociation flag on this instruction, which must be
  /// an operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasAllowReassoc(bool B);

  /// Set or clear the no-nans flag on this instruction, which must be an
  /// operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasNoNaNs(bool B);

  /// Set or clear the no-infs flag on this instruction, which must be an
  /// operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasNoInfs(bool B);

  /// Set or clear the no-signed-zeros flag on this instruction, which must be
  /// an operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasNoSignedZeros(bool B);

  /// Set or clear the allow-reciprocal flag on this instruction, which must be
  /// an operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasAllowReciprocal(bool B);

  /// Set or clear the allow-contract flag on this instruction, which must be
  /// an operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasAllowContract(bool B);

  /// Set or clear the approximate-math-functions flag on this instruction,
  /// which must be an operator which supports this flag. See LangRef.html for
  /// the meaning of this flag.
  void setHasApproxFunc(bool B);

  /// Convenience function for setting multiple fast-math flags on this
  /// instruction, which must be an operator which supports these flags. See
  /// LangRef.html for the meaning of these flags.
  void setFastMathFlags(FastMathFlags FMF);

  /// Convenience function for transferring all fast-math flag values to this
  /// instruction, which must be an operator which supports these flags. See
  /// LangRef.html for the meaning of these flags.
  void copyFastMathFlags(FastMathFlags FMF);

  /// Determine whether all fast-math-flags are set.
  bool isFast() const;

  /// Determine whether the allow-reassociation flag is set.
  bool hasAllowReassoc() const;

  /// Determine whether the no-NaNs flag is set.
  bool hasNoNaNs() const;

  /// Determine whether the no-infs flag is set.
  bool hasNoInfs() const;

  /// Determine whether the no-signed-zeros flag is set.
  bool hasNoSignedZeros() const;

  /// Determine whether the allow-reciprocal flag is set.
  bool hasAllowReciprocal() const;

  /// Determine whether the allow-contract flag is set.
  bool hasAllowContract() const;

  /// Determine whether the approximate-math-functions flag is set.
  bool hasApproxFunc() const;

  /// Convenience function for getting all the fast-math flags, which must be an
  /// operator which supports these flags. See LangRef.html for the meaning of
  /// these flags.
  FastMathFlags getFastMathFlags() const;

  /// Copy I's fast-math flags
  void copyFastMathFlags(const Instruction *I);

  /// Convenience method to copy supported exact, fast-math, and (optionally)
  /// wrapping flags from V to this instruction.
  void copyIRFlags(const Value *V, bool IncludeWrapFlags = true);

  /// Logical 'and' of any supported wrapping, exact, and fast-math flags of
  /// V and this instruction.
  void andIRFlags(const Value *V);

  /// Merge 2 debug locations and apply it to the Instruction. If the
  /// instruction is a CallIns, we need to traverse the inline chain to find
  /// the common scope. This is not efficient for N-way merging as each time
  /// you merge 2 iterations, you need to rebuild the hashmap to find the
  /// common scope. However, we still choose this API because:
  ///  1) Simplicity: it takes 2 locations instead of a list of locations.
  ///  2) In worst case, it increases the complexity from O(N*I) to
  ///     O(2*N*I), where N is # of Instructions to merge, and I is the
  ///     maximum level of inline stack. So it is still linear.
  ///  3) Merging of call instructions should be extremely rare in real
  ///     applications, thus the N-way merging should be in code path.
  /// The DebugLoc attached to this instruction will be overwritten by the
  /// merged DebugLoc.
  void applyMergedLocation(const DILocation *LocA, const DILocation *LocB);

  /// Updates the debug location given that the instruction has been hoisted
  /// from a block to a predecessor of that block.
  /// Note: it is undefined behavior to call this on an instruction not
  /// currently inserted into a function.
  void updateLocationAfterHoist();

  /// Drop the instruction's debug location. This does not guarantee removal
  /// of the !dbg source location attachment, as it must set a line 0 location
  /// with scope information attached on call instructions. To guarantee
  /// removal of the !dbg attachment, use the \ref setDebugLoc() API.
  /// Note: it is undefined behavior to call this on an instruction not
  /// currently inserted into a function.
  void dropLocation();

private:
  // These are all implemented in Metadata.cpp.
  MDNode *getMetadataImpl(unsigned KindID) const;
  MDNode *getMetadataImpl(StringRef Kind) const;
  void
  getAllMetadataImpl(SmallVectorImpl<std::pair<unsigned, MDNode *>> &) const;

public:
  //===--------------------------------------------------------------------===//
  // Predicates and helper methods.
  //===--------------------------------------------------------------------===//

  /// Return true if the instruction is associative:
  ///
  ///   Associative operators satisfy:  x op (y op z) === (x op y) op z
  ///
  /// In LLVM, the Add, Mul, And, Or, and Xor operators are associative.
  ///
  bool isAssociative() const LLVM_READONLY;
  static bool isAssociative(unsigned Opcode) {
    return Opcode == And || Opcode == Or || Opcode == Xor ||
           Opcode == Add || Opcode == Mul;
  }

  /// Return true if the instruction is commutative:
  ///
  ///   Commutative operators satisfy: (x op y) === (y op x)
  ///
  /// In LLVM, these are the commutative operators, plus SetEQ and SetNE, when
  /// applied to any type.
  ///
  bool isCommutative() const LLVM_READONLY;
  static bool isCommutative(unsigned Opcode) {
    switch (Opcode) {
    case Add: case FAdd:
    case Mul: case FMul:
    case And: case Or: case Xor:
      return true;
    default:
      return false;
  }
  }

  /// Return true if the instruction is idempotent:
  ///
  ///   Idempotent operators satisfy:  x op x === x
  ///
  /// In LLVM, the And and Or operators are idempotent.
  ///
  bool isIdempotent() const { return isIdempotent(getOpcode()); }
  static bool isIdempotent(unsigned Opcode) {
    return Opcode == And || Opcode == Or;
  }

  /// Return true if the instruction is nilpotent:
  ///
  ///   Nilpotent operators satisfy:  x op x === Id,
  ///
  ///   where Id is the identity for the operator, i.e. a constant such that
  ///     x op Id === x and Id op x === x for all x.
  ///
  /// In LLVM, the Xor operator is nilpotent.
  ///
  bool isNilpotent() const { return isNilpotent(getOpcode()); }
  static bool isNilpotent(unsigned Opcode) {
    return Opcode == Xor;
  }

  /// Return true if this instruction may modify memory.
  bool mayWriteToMemory() const;

  /// Return true if this instruction may read memory.
  bool mayReadFromMemory() const;

  /// Return true if this instruction may read or write memory.
  bool mayReadOrWriteMemory() const {
    return mayReadFromMemory() || mayWriteToMemory();
  }

  /// Return true if this instruction has an AtomicOrdering of unordered or
  /// higher.
  bool isAtomic() const;

  /// Return true if this atomic instruction loads from memory.
  bool hasAtomicLoad() const;

  /// Return true if this atomic instruction stores to memory.
  bool hasAtomicStore() const;

  /// Return true if this instruction has a volatile memory access.
  bool isVolatile() const;

  /// Return true if this instruction may throw an exception.
  bool mayThrow() const;

  /// Return true if this instruction behaves like a memory fence: it can load
  /// or store to memory location without being given a memory location.
  bool isFenceLike() const {
    switch (getOpcode()) {
    default:
      return false;
    // This list should be kept in sync with the list in mayWriteToMemory for
    // all opcodes which don't have a memory location.
    case Instruction::Fence:
    case Instruction::CatchPad:
    case Instruction::CatchRet:
    case Instruction::Call:
    case Instruction::Invoke:
      return true;
    }
  }

  /// Return true if the instruction may have side effects.
  ///
  /// Note that this does not consider malloc and alloca to have side
  /// effects because the newly allocated memory is completely invisible to
  /// instructions which don't use the returned value.  For cases where this
  /// matters, isSafeToSpeculativelyExecute may be more appropriate.
  bool mayHaveSideEffects() const { return mayWriteToMemory() || mayThrow(); }

  /// Return true if the instruction can be removed if the result is unused.
  ///
  /// When constant folding some instructions cannot be removed even if their
  /// results are unused. Specifically terminator instructions and calls that
  /// may have side effects cannot be removed without semantically changing the
  /// generated program.
  bool isSafeToRemove() const;

  /// Return true if the instruction will return (unwinding is considered as
  /// a form of returning control flow here).
  bool willReturn() const;

  /// Return true if the instruction is a variety of EH-block.
  bool isEHPad() const {
    switch (getOpcode()) {
    case Instruction::CatchSwitch:
    case Instruction::CatchPad:
    case Instruction::CleanupPad:
    case Instruction::LandingPad:
      return true;
    default:
      return false;
    }
  }

  /// Return true if the instruction is a llvm.lifetime.start or
  /// llvm.lifetime.end marker.
  bool isLifetimeStartOrEnd() const;

  /// Return true if the instruction is a llvm.launder.invariant.group or
  /// llvm.strip.invariant.group.
  bool isLaunderOrStripInvariantGroup() const;

  /// Return true if the instruction is a DbgInfoIntrinsic or PseudoProbeInst.
  bool isDebugOrPseudoInst() const;

  /// Return a pointer to the next non-debug instruction in the same basic
  /// block as 'this', or nullptr if no such instruction exists. Skip any pseudo
  /// operations if \c SkipPseudoOp is true.
  const Instruction *
  getNextNonDebugInstruction(bool SkipPseudoOp = false) const;
  Instruction *getNextNonDebugInstruction(bool SkipPseudoOp = false) {
    return const_cast<Instruction *>(
        static_cast<const Instruction *>(this)->getNextNonDebugInstruction(
            SkipPseudoOp));
  }

  /// Return a pointer to the previous non-debug instruction in the same basic
  /// block as 'this', or nullptr if no such instruction exists. Skip any pseudo
  /// operations if \c SkipPseudoOp is true.
  const Instruction *
  getPrevNonDebugInstruction(bool SkipPseudoOp = false) const;
  Instruction *getPrevNonDebugInstruction(bool SkipPseudoOp = false) {
    return const_cast<Instruction *>(
        static_cast<const Instruction *>(this)->getPrevNonDebugInstruction(
            SkipPseudoOp));
  }

  /// Create a copy of 'this' instruction that is identical in all ways except
  /// the following:
  ///   * The instruction has no parent
  ///   * The instruction has no name
  ///
  Instruction *clone() const;

  /// Return true if the specified instruction is exactly identical to the
  /// current one. This means that all operands match and any extra information
  /// (e.g. load is volatile) agree.
  bool isIdenticalTo(const Instruction *I) const;

  /// This is like isIdenticalTo, except that it ignores the
  /// SubclassOptionalData flags, which may specify conditions under which the
  /// instruction's result is undefined.
  bool isIdenticalToWhenDefined(const Instruction *I) const;

  /// When checking for operation equivalence (using isSameOperationAs) it is
  /// sometimes useful to ignore certain attributes.
  enum OperationEquivalenceFlags {
    /// Check for equivalence ignoring load/store alignment.
    CompareIgnoringAlignment = 1<<0,
    /// Check for equivalence treating a type and a vector of that type
    /// as equivalent.
    CompareUsingScalarTypes = 1<<1
  };

  /// This function determines if the specified instruction executes the same
  /// operation as the current one. This means that the opcodes, type, operand
  /// types and any other factors affecting the operation must be the same. This
  /// is similar to isIdenticalTo except the operands themselves don't have to
  /// be identical.
  /// @returns true if the specified instruction is the same operation as
  /// the current one.
  /// Determine if one instruction is the same operation as another.
  bool isSameOperationAs(const Instruction *I, unsigned flags = 0) const;

  /// Return true if there are any uses of this instruction in blocks other than
  /// the specified block. Note that PHI nodes are considered to evaluate their
  /// operands in the corresponding predecessor block.
  bool isUsedOutsideOfBlock(const BasicBlock *BB) const;

  /// Return the number of successors that this instruction has. The instruction
  /// must be a terminator.
  unsigned getNumSuccessors() const;

  /// Return the specified successor. This instruction must be a terminator.
  BasicBlock *getSuccessor(unsigned Idx) const;

  /// Update the specified successor to point at the provided block. This
  /// instruction must be a terminator.
  void setSuccessor(unsigned Idx, BasicBlock *BB);

  /// Replace specified successor OldBB to point at the provided block.
  /// This instruction must be a terminator.
  void replaceSuccessorWith(BasicBlock *OldBB, BasicBlock *NewBB);

  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static bool classof(const Value *V) {
    return V->getValueID() >= Value::InstructionVal;
  }

  //----------------------------------------------------------------------
  // Exported enumerations.
  //
  enum TermOps {       // These terminate basic blocks
#define  FIRST_TERM_INST(N)             TermOpsBegin = N,
#define HANDLE_TERM_INST(N, OPC, CLASS) OPC = N,
#define   LAST_TERM_INST(N)             TermOpsEnd = N+1
#include "llvm/IR/Instruction.def"
  };

  enum UnaryOps {
#define  FIRST_UNARY_INST(N)             UnaryOpsBegin = N,
#define HANDLE_UNARY_INST(N, OPC, CLASS) OPC = N,
#define   LAST_UNARY_INST(N)             UnaryOpsEnd = N+1
#include "llvm/IR/Instruction.def"
  };

  enum BinaryOps {
#define  FIRST_BINARY_INST(N)             BinaryOpsBegin = N,
#define HANDLE_BINARY_INST(N, OPC, CLASS) OPC = N,
#define   LAST_BINARY_INST(N)             BinaryOpsEnd = N+1
#include "llvm/IR/Instruction.def"
  };

  enum MemoryOps {
#define  FIRST_MEMORY_INST(N)             MemoryOpsBegin = N,
#define HANDLE_MEMORY_INST(N, OPC, CLASS) OPC = N,
#define   LAST_MEMORY_INST(N)             MemoryOpsEnd = N+1
#include "llvm/IR/Instruction.def"
  };

  enum CastOps {
#define  FIRST_CAST_INST(N)             CastOpsBegin = N,
#define HANDLE_CAST_INST(N, OPC, CLASS) OPC = N,
#define   LAST_CAST_INST(N)             CastOpsEnd = N+1
#include "llvm/IR/Instruction.def"
  };

  enum FuncletPadOps {
#define  FIRST_FUNCLETPAD_INST(N)             FuncletPadOpsBegin = N,
#define HANDLE_FUNCLETPAD_INST(N, OPC, CLASS) OPC = N,
#define   LAST_FUNCLETPAD_INST(N)             FuncletPadOpsEnd = N+1
#include "llvm/IR/Instruction.def"
  };

  enum OtherOps {
#define  FIRST_OTHER_INST(N)             OtherOpsBegin = N,
#define HANDLE_OTHER_INST(N, OPC, CLASS) OPC = N,
#define   LAST_OTHER_INST(N)             OtherOpsEnd = N+1
#include "llvm/IR/Instruction.def"
  };

private:
  friend class SymbolTableListTraits<Instruction>;
  friend class BasicBlock; // For renumbering.

  // Shadow Value::setValueSubclassData with a private forwarding method so that
  // subclasses cannot accidentally use it.
  void setValueSubclassData(unsigned short D) {
    Value::setValueSubclassData(D);
  }

  unsigned short getSubclassDataFromValue() const {
    return Value::getSubclassDataFromValue();
  }

  void setParent(BasicBlock *P);

protected:
  // Instruction subclasses can stick up to 15 bits of stuff into the
  // SubclassData field of instruction with these members.

  template <typename BitfieldElement>
  typename BitfieldElement::Type getSubclassData() const {
    static_assert(
        std::is_same<BitfieldElement, HasMetadataField>::value ||
            !Bitfield::isOverlapping<BitfieldElement, HasMetadataField>(),
        "Must not overlap with the metadata bit");
    return Bitfield::get<BitfieldElement>(getSubclassDataFromValue());
  }

  template <typename BitfieldElement>
  void setSubclassData(typename BitfieldElement::Type Value) {
    static_assert(
        std::is_same<BitfieldElement, HasMetadataField>::value ||
            !Bitfield::isOverlapping<BitfieldElement, HasMetadataField>(),
        "Must not overlap with the metadata bit");
    auto Storage = getSubclassDataFromValue();
    Bitfield::set<BitfieldElement>(Storage, Value);
    setValueSubclassData(Storage);
  }

  Instruction(Type *Ty, unsigned iType, Use *Ops, unsigned NumOps,
              Instruction *InsertBefore = nullptr);
  Instruction(Type *Ty, unsigned iType, Use *Ops, unsigned NumOps,
              BasicBlock *InsertAtEnd);

private:
  /// Create a copy of this instruction.
  Instruction *cloneImpl() const;
};

inline void ilist_alloc_traits<Instruction>::deleteNode(Instruction *V) {
  V->deleteValue();
}

} // end namespace llvm

#endif // LLVM_IR_INSTRUCTION_H
