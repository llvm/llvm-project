//===-BlockGenerators.h - Helper to generate code for statements-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the BlockGenerator and VectorBlockGenerator classes, which
// generate sequential code and vectorized code for a polyhedral statement,
// respectively.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_BLOCK_GENERATORS_H
#define POLLY_BLOCK_GENERATORS_H

#include "polly/CodeGen/IRBuilder.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "isl/isl-noexceptions.h"

namespace polly {
using llvm::AllocaInst;
using llvm::ArrayRef;
using llvm::AssertingVH;
using llvm::BasicBlock;
using llvm::BinaryOperator;
using llvm::CmpInst;
using llvm::DataLayout;
using llvm::DenseMap;
using llvm::DominatorTree;
using llvm::Function;
using llvm::Instruction;
using llvm::LoadInst;
using llvm::Loop;
using llvm::LoopInfo;
using llvm::LoopToScevMapT;
using llvm::MapVector;
using llvm::PHINode;
using llvm::ScalarEvolution;
using llvm::SetVector;
using llvm::SmallVector;
using llvm::StoreInst;
using llvm::StringRef;
using llvm::Type;
using llvm::UnaryInstruction;
using llvm::Value;

class MemoryAccess;
class ScopArrayInfo;
class IslExprBuilder;

/// Generate a new basic block for a polyhedral statement.
class BlockGenerator {
public:
  typedef llvm::SmallVector<ValueMapT, 8> VectorValueMapT;

  /// Map types to resolve scalar dependences.
  ///
  ///@{
  using AllocaMapTy = DenseMap<const ScopArrayInfo *, AssertingVH<AllocaInst>>;

  /// Simple vector of instructions to store escape users.
  using EscapeUserVectorTy = SmallVector<Instruction *, 4>;

  /// Map type to resolve escaping users for scalar instructions.
  ///
  /// @see The EscapeMap member.
  using EscapeUsersAllocaMapTy =
      MapVector<Instruction *,
                std::pair<AssertingVH<Value>, EscapeUserVectorTy>>;

  ///@}

  /// Create a generator for basic blocks.
  ///
  /// @param Builder     The LLVM-IR Builder used to generate the statement. The
  ///                    code is generated at the location, the Builder points
  ///                    to.
  /// @param LI          The loop info for the current function
  /// @param SE          The scalar evolution info for the current function
  /// @param DT          The dominator tree of this function.
  /// @param ScalarMap   Map from scalars to their demoted location.
  /// @param EscapeMap   Map from scalars to their escape users and locations.
  /// @param GlobalMap   A mapping from llvm::Values used in the original scop
  ///                    region to a new set of llvm::Values. Each reference to
  ///                    an original value appearing in this mapping is replaced
  ///                    with the new value it is mapped to.
  /// @param ExprBuilder An expression builder to generate new access functions.
  /// @param StartBlock  The first basic block after the RTC.
  BlockGenerator(PollyIRBuilder &Builder, LoopInfo &LI, ScalarEvolution &SE,
                 DominatorTree &DT, AllocaMapTy &ScalarMap,
                 EscapeUsersAllocaMapTy &EscapeMap, ValueMapT &GlobalMap,
                 IslExprBuilder *ExprBuilder, BasicBlock *StartBlock);

  /// Copy the basic block.
  ///
  /// This copies the entire basic block and updates references to old values
  /// with references to new values, as defined by GlobalMap.
  ///
  /// @param Stmt        The block statement to code generate.
  /// @param LTS         A map from old loops to new induction variables as
  ///                    SCEVs.
  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  void copyStmt(ScopStmt &Stmt, LoopToScevMapT &LTS,
                isl_id_to_ast_expr *NewAccesses);

  /// Remove a ScopArrayInfo's allocation from the ScalarMap.
  ///
  /// This function allows to remove values from the ScalarMap. This is useful
  /// if the corresponding alloca instruction will be deleted (or moved into
  /// another module), as without removing these values the underlying
  /// AssertingVH will trigger due to us still keeping reference to this
  /// scalar.
  ///
  /// @param Array The array for which the alloca was generated.
  void freeScalarAlloc(ScopArrayInfo *Array) { ScalarMap.erase(Array); }

  /// Return the alloca for @p Access.
  ///
  /// If no alloca was mapped for @p Access a new one is created.
  ///
  /// @param Access    The memory access for which to generate the alloca.
  ///
  /// @returns The alloca for @p Access or a replacement value taken from
  ///          GlobalMap.
  Value *getOrCreateAlloca(const MemoryAccess &Access);

  /// Return the alloca for @p Array.
  ///
  /// If no alloca was mapped for @p Array a new one is created.
  ///
  /// @param Array The array for which to generate the alloca.
  ///
  /// @returns The alloca for @p Array or a replacement value taken from
  ///          GlobalMap.
  Value *getOrCreateAlloca(const ScopArrayInfo *Array);

  /// Finalize the code generation for the SCoP @p S.
  ///
  /// This will initialize and finalize the scalar variables we demoted during
  /// the code generation.
  ///
  /// @see createScalarInitialization(Scop &)
  /// @see createScalarFinalization(Region &)
  void finalizeSCoP(Scop &S);

  /// An empty destructor
  virtual ~BlockGenerator() {}

  BlockGenerator(const BlockGenerator &) = default;

protected:
  PollyIRBuilder &Builder;
  LoopInfo &LI;
  ScalarEvolution &SE;
  IslExprBuilder *ExprBuilder;

  /// The dominator tree of this function.
  DominatorTree &DT;

  /// The entry block of the current function.
  BasicBlock *EntryBB;

  /// Map to resolve scalar dependences for PHI operands and scalars.
  ///
  /// When translating code that contains scalar dependences as they result from
  /// inter-block scalar dependences (including the use of data carrying PHI
  /// nodes), we do not directly regenerate in-register SSA code, but instead
  /// allocate some stack memory through which these scalar values are passed.
  /// Only a later pass of -mem2reg will then (re)introduce in-register
  /// computations.
  ///
  /// To keep track of the memory location(s) used to store the data computed by
  /// a given SSA instruction, we use the map 'ScalarMap'. ScalarMap maps a
  /// given ScopArrayInfo to the junk of stack allocated memory, that is
  /// used for code generation.
  ///
  /// Up to two different ScopArrayInfo objects are associated with each
  /// llvm::Value:
  ///
  /// MemoryType::Value objects are used for normal scalar dependences that go
  /// from a scalar definition to its use. Such dependences are lowered by
  /// directly writing the value an instruction computes into the corresponding
  /// chunk of memory and reading it back from this chunk of memory right before
  /// every use of this original scalar value. The memory allocations for
  /// MemoryType::Value objects end with '.s2a'.
  ///
  /// MemoryType::PHI (and MemoryType::ExitPHI) objects are used to model PHI
  /// nodes. For each PHI nodes we introduce, besides the Array of type
  /// MemoryType::Value, a second chunk of memory into which we write at the end
  /// of each basic block preceding the PHI instruction the value passed
  /// through this basic block. At the place where the PHI node is executed, we
  /// replace the PHI node with a load from the corresponding MemoryType::PHI
  /// memory location. The memory allocations for MemoryType::PHI end with
  /// '.phiops'.
  ///
  /// Example:
  ///
  ///                              Input C Code
  ///                              ============
  ///
  ///                 S1:      x1 = ...
  ///                          for (i=0...N) {
  ///                 S2:           x2 = phi(x1, add)
  ///                 S3:           add = x2 + 42;
  ///                          }
  ///                 S4:      print(x1)
  ///                          print(x2)
  ///                          print(add)
  ///
  ///
  ///        Unmodified IR                         IR After expansion
  ///        =============                         ==================
  ///
  /// S1:   x1 = ...                     S1:    x1 = ...
  ///                                           x1.s2a = s1
  ///                                           x2.phiops = s1
  ///        |                                    |
  ///        |   <--<--<--<--<                    |   <--<--<--<--<
  ///        | /              \                   | /              \     .
  ///        V V               \                  V V               \    .
  /// S2:  x2 = phi (x1, add)   |        S2:    x2 = x2.phiops       |
  ///                           |               x2.s2a = x2          |
  ///                           |                                    |
  /// S3:  add = x2 + 42        |        S3:    add = x2 + 42        |
  ///                           |               add.s2a = add        |
  ///                           |               x2.phiops = add      |
  ///        | \               /                  | \               /
  ///        |  \             /                   |  \             /
  ///        |   >-->-->-->-->                    |   >-->-->-->-->
  ///        V                                    V
  ///
  ///                                    S4:    x1 = x1.s2a
  /// S4:  ... = x1                             ... = x1
  ///                                           x2 = x2.s2a
  ///      ... = x2                             ... = x2
  ///                                           add = add.s2a
  ///      ... = add                            ... = add
  ///
  ///      ScalarMap = { x1:Value -> x1.s2a, x2:Value -> x2.s2a,
  ///                    add:Value -> add.s2a, x2:PHI -> x2.phiops }
  ///
  ///  ??? Why does a PHI-node require two memory chunks ???
  ///
  ///  One may wonder why a PHI node requires two memory chunks and not just
  ///  all data is stored in a single location. The following example tries
  ///  to store all data in .s2a and drops the .phiops location:
  ///
  ///      S1:    x1 = ...
  ///             x1.s2a = s1
  ///             x2.s2a = s1             // use .s2a instead of .phiops
  ///               |
  ///               |   <--<--<--<--<
  ///               | /              \    .
  ///               V V               \   .
  ///      S2:    x2 = x2.s2a          |  // value is same as above, but read
  ///                                  |  // from .s2a
  ///                                  |
  ///             x2.s2a = x2          |  // store into .s2a as normal
  ///                                  |
  ///      S3:    add = x2 + 42        |
  ///             add.s2a = add        |
  ///             x2.s2a = add         |  // use s2a instead of .phiops
  ///               | \               /   // !!! This is wrong, as x2.s2a now
  ///               |   >-->-->-->-->     // contains add instead of x2.
  ///               V
  ///
  ///      S4:    x1 = x1.s2a
  ///             ... = x1
  ///             x2 = x2.s2a             // !!! We now read 'add' instead of
  ///             ... = x2                // 'x2'
  ///             add = add.s2a
  ///             ... = add
  ///
  ///  As visible in the example, the SSA value of the PHI node may still be
  ///  needed _after_ the basic block, which could conceptually branch to the
  ///  PHI node, has been run and has overwritten the PHI's old value. Hence, a
  ///  single memory location is not enough to code-generate a PHI node.
  ///
  /// Memory locations used for the special PHI node modeling.
  AllocaMapTy &ScalarMap;

  /// Map from instructions to their escape users as well as the alloca.
  EscapeUsersAllocaMapTy &EscapeMap;

  /// A map from llvm::Values referenced in the old code to a new set of
  ///        llvm::Values, which is used to replace these old values during
  ///        code generation.
  ValueMapT &GlobalMap;

  /// The first basic block after the RTC.
  BasicBlock *StartBlock;

  /// Split @p BB to create a new one we can use to clone @p BB in.
  BasicBlock *splitBB(BasicBlock *BB);

  /// Copy the given basic block.
  ///
  /// @param Stmt      The statement to code generate.
  /// @param BB        The basic block to code generate.
  /// @param BBMap     A mapping from old values to their new values in this
  /// block.
  /// @param LTS         A map from old loops to new induction variables as
  ///                    SCEVs.
  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  ///
  /// @returns The copy of the basic block.
  BasicBlock *copyBB(ScopStmt &Stmt, BasicBlock *BB, ValueMapT &BBMap,
                     LoopToScevMapT &LTS, isl_id_to_ast_expr *NewAccesses);

  /// Copy the given basic block.
  ///
  /// @param Stmt      The statement to code generate.
  /// @param BB        The basic block to code generate.
  /// @param BBCopy    The new basic block to generate code in.
  /// @param BBMap     A mapping from old values to their new values in this
  /// block.
  /// @param LTS         A map from old loops to new induction variables as
  ///                    SCEVs.
  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  void copyBB(ScopStmt &Stmt, BasicBlock *BB, BasicBlock *BBCopy,
              ValueMapT &BBMap, LoopToScevMapT &LTS,
              isl_id_to_ast_expr *NewAccesses);

  /// Generate reload of scalars demoted to memory and needed by @p Stmt.
  ///
  /// @param Stmt  The statement we generate code for.
  /// @param LTS   A mapping from loops virtual canonical induction
  ///              variable to their new values.
  /// @param BBMap A mapping from old values to their new values in this block.
  /// @param NewAccesses A map from memory access ids to new ast expressions.
  void generateScalarLoads(ScopStmt &Stmt, LoopToScevMapT &LTS,
                           ValueMapT &BBMap,
                           __isl_keep isl_id_to_ast_expr *NewAccesses);

  /// When statement tracing is enabled, build the print instructions for
  /// printing the current statement instance.
  ///
  /// The printed output looks like:
  ///
  ///     Stmt1(0)
  ///
  /// If printing of scalars is enabled, it also appends the value of each
  /// scalar to the line:
  ///
  ///     Stmt1(0) %i=1 %sum=5
  ///
  /// @param Stmt  The statement we generate code for.
  /// @param LTS   A mapping from loops virtual canonical induction
  ///              variable to their new values.
  /// @param BBMap A mapping from old values to their new values in this block.
  void generateBeginStmtTrace(ScopStmt &Stmt, LoopToScevMapT &LTS,
                              ValueMapT &BBMap);

  /// Generate instructions that compute whether one instance of @p Set is
  /// executed.
  ///
  /// @param Stmt      The statement we generate code for.
  /// @param Subdomain A set in the space of @p Stmt's domain. Elements not in
  ///                  @p Stmt's domain are ignored.
  ///
  /// @return An expression of type i1, generated into the current builder
  ///         position, that evaluates to 1 if the executed instance is part of
  ///         @p Set.
  Value *buildContainsCondition(ScopStmt &Stmt, const isl::set &Subdomain);

  /// Generate code that executes in a subset of @p Stmt's domain.
  ///
  /// @param Stmt        The statement we generate code for.
  /// @param Subdomain   The condition for some code to be executed.
  /// @param Subject     A name for the code that is executed
  ///                    conditionally. Used to name new basic blocks and
  ///                    instructions.
  /// @param GenThenFunc Callback which generates the code to be executed
  ///                    when the current executed instance is in @p Set. The
  ///                    IRBuilder's position is moved to within the block that
  ///                    executes conditionally for this callback.
  void generateConditionalExecution(ScopStmt &Stmt, const isl::set &Subdomain,
                                    StringRef Subject,
                                    const std::function<void()> &GenThenFunc);

  /// Generate the scalar stores for the given statement.
  ///
  /// After the statement @p Stmt was copied all inner-SCoP scalar dependences
  /// starting in @p Stmt (hence all scalar write accesses in @p Stmt) need to
  /// be demoted to memory.
  ///
  /// @param Stmt  The statement we generate code for.
  /// @param LTS   A mapping from loops virtual canonical induction
  ///              variable to their new values
  ///              (for values recalculated in the new ScoP, but not
  ///               within this basic block)
  /// @param BBMap A mapping from old values to their new values in this block.
  /// @param NewAccesses A map from memory access ids to new ast expressions.
  virtual void generateScalarStores(ScopStmt &Stmt, LoopToScevMapT &LTS,
                                    ValueMapT &BBMap,
                                    __isl_keep isl_id_to_ast_expr *NewAccesses);

  /// Handle users of @p Array outside the SCoP.
  ///
  /// @param S         The current SCoP.
  /// @param Inst      The ScopArrayInfo to handle.
  void handleOutsideUsers(const Scop &S, ScopArrayInfo *Array);

  /// Find scalar statements that have outside users.
  ///
  /// We register these scalar values to later update subsequent scalar uses of
  /// these values to either use the newly computed value from within the scop
  /// (if the scop was executed) or the unchanged original code (if the run-time
  /// check failed).
  ///
  /// @param S The scop for which to find the outside users.
  void findOutsideUsers(Scop &S);

  /// Initialize the memory of demoted scalars.
  ///
  /// @param S The scop for which to generate the scalar initializers.
  void createScalarInitialization(Scop &S);

  /// Create exit PHI node merges for PHI nodes with more than two edges
  ///        from inside the scop.
  ///
  /// For scops which have a PHI node in the exit block that has more than two
  /// incoming edges from inside the scop region, we require some special
  /// handling to understand which of the possible values will be passed to the
  /// PHI node from inside the optimized version of the scop. To do so ScopInfo
  /// models the possible incoming values as write accesses of the ScopStmts.
  ///
  /// This function creates corresponding code to reload the computed outgoing
  /// value from the stack slot it has been stored into and to pass it on to the
  /// PHI node in the original exit block.
  ///
  /// @param S The scop for which to generate the exiting PHI nodes.
  void createExitPHINodeMerges(Scop &S);

  /// Promote the values of demoted scalars after the SCoP.
  ///
  /// If a scalar value was used outside the SCoP we need to promote the value
  /// stored in the memory cell allocated for that scalar and combine it with
  /// the original value in the non-optimized SCoP.
  void createScalarFinalization(Scop &S);

  /// Try to synthesize a new value
  ///
  /// Given an old value, we try to synthesize it in a new context from its
  /// original SCEV expression. We start from the original SCEV expression,
  /// then replace outdated parameter and loop references, and finally
  /// expand it to code that computes this updated expression.
  ///
  /// @param Stmt      The statement to code generate
  /// @param Old       The old Value
  /// @param BBMap     A mapping from old values to their new values
  ///                  (for values recalculated within this basic block)
  /// @param LTS       A mapping from loops virtual canonical induction
  ///                  variable to their new values
  ///                  (for values recalculated in the new ScoP, but not
  ///                   within this basic block)
  /// @param L         The loop that surrounded the instruction that referenced
  ///                  this value in the original code. This loop is used to
  ///                  evaluate the scalar evolution at the right scope.
  ///
  /// @returns  o A newly synthesized value.
  ///           o NULL, if synthesizing the value failed.
  Value *trySynthesizeNewValue(ScopStmt &Stmt, Value *Old, ValueMapT &BBMap,
                               LoopToScevMapT &LTS, Loop *L) const;

  /// Get the new version of a value.
  ///
  /// Given an old value, we first check if a new version of this value is
  /// available in the BBMap or GlobalMap. In case it is not and the value can
  /// be recomputed using SCEV, we do so. If we can not recompute a value
  /// using SCEV, but we understand that the value is constant within the scop,
  /// we return the old value.  If the value can still not be derived, this
  /// function will assert.
  ///
  /// @param Stmt      The statement to code generate.
  /// @param Old       The old Value.
  /// @param BBMap     A mapping from old values to their new values
  ///                  (for values recalculated within this basic block).
  /// @param LTS       A mapping from loops virtual canonical induction
  ///                  variable to their new values
  ///                  (for values recalculated in the new ScoP, but not
  ///                   within this basic block).
  /// @param L         The loop that surrounded the instruction that referenced
  ///                  this value in the original code. This loop is used to
  ///                  evaluate the scalar evolution at the right scope.
  ///
  /// @returns  o The old value, if it is still valid.
  ///           o The new value, if available.
  ///           o NULL, if no value is found.
  Value *getNewValue(ScopStmt &Stmt, Value *Old, ValueMapT &BBMap,
                     LoopToScevMapT &LTS, Loop *L) const;

  void copyInstScalar(ScopStmt &Stmt, Instruction *Inst, ValueMapT &BBMap,
                      LoopToScevMapT &LTS);

  /// Get the innermost loop that surrounds the statement @p Stmt.
  Loop *getLoopForStmt(const ScopStmt &Stmt) const;

  /// Generate the operand address
  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  Value *generateLocationAccessed(ScopStmt &Stmt, MemAccInst Inst,
                                  ValueMapT &BBMap, LoopToScevMapT &LTS,
                                  isl_id_to_ast_expr *NewAccesses);

  /// Generate the operand address.
  ///
  /// @param Stmt         The statement to generate code for.
  /// @param L            The innermost loop that surrounds the statement.
  /// @param Pointer      If the access expression is not changed (ie. not found
  ///                     in @p LTS), use this Pointer from the original code
  ///                     instead.
  /// @param BBMap        A mapping from old values to their new values.
  /// @param LTS          A mapping from loops virtual canonical induction
  ///                     variable to their new values.
  /// @param NewAccesses  Ahead-of-time generated access expressions.
  /// @param Id           Identifier of the MemoryAccess to generate.
  /// @param ExpectedType The type the returned value should have.
  ///
  /// @return The generated address.
  Value *generateLocationAccessed(ScopStmt &Stmt, Loop *L, Value *Pointer,
                                  ValueMapT &BBMap, LoopToScevMapT &LTS,
                                  isl_id_to_ast_expr *NewAccesses,
                                  __isl_take isl_id *Id, Type *ExpectedType);

  /// Generate the pointer value that is accesses by @p Access.
  ///
  /// For write accesses, generate the target address. For read accesses,
  /// generate the source address.
  /// The access can be either an array access or a scalar access. In the first
  /// case, the returned address will point to an element into that array. In
  /// the scalar case, an alloca is used.
  /// If a new AccessRelation is set for the MemoryAccess, the new relation will
  /// be used.
  ///
  /// @param Access      The access to generate a pointer for.
  /// @param L           The innermost loop that surrounds the statement.
  /// @param LTS         A mapping from loops virtual canonical induction
  ///                    variable to their new values.
  /// @param BBMap       A mapping from old values to their new values.
  /// @param NewAccesses A map from memory access ids to new ast expressions.
  ///
  /// @return The generated address.
  Value *getImplicitAddress(MemoryAccess &Access, Loop *L, LoopToScevMapT &LTS,
                            ValueMapT &BBMap,
                            __isl_keep isl_id_to_ast_expr *NewAccesses);

  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  Value *generateArrayLoad(ScopStmt &Stmt, LoadInst *load, ValueMapT &BBMap,
                           LoopToScevMapT &LTS,
                           isl_id_to_ast_expr *NewAccesses);

  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  void generateArrayStore(ScopStmt &Stmt, StoreInst *store, ValueMapT &BBMap,
                          LoopToScevMapT &LTS, isl_id_to_ast_expr *NewAccesses);

  /// Copy a single PHI instruction.
  ///
  /// The implementation in the BlockGenerator is trivial, however it allows
  /// subclasses to handle PHIs different.
  virtual void copyPHIInstruction(ScopStmt &, PHINode *, ValueMapT &,
                                  LoopToScevMapT &) {}

  /// Copy a single Instruction.
  ///
  /// This copies a single Instruction and updates references to old values
  /// with references to new values, as defined by GlobalMap and BBMap.
  ///
  /// @param Stmt        The statement to code generate.
  /// @param Inst        The instruction to copy.
  /// @param BBMap       A mapping from old values to their new values
  ///                    (for values recalculated within this basic block).
  /// @param GlobalMap   A mapping from old values to their new values
  ///                    (for values recalculated in the new ScoP, but not
  ///                    within this basic block).
  /// @param LTS         A mapping from loops virtual canonical induction
  ///                    variable to their new values
  ///                    (for values recalculated in the new ScoP, but not
  ///                     within this basic block).
  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  void copyInstruction(ScopStmt &Stmt, Instruction *Inst, ValueMapT &BBMap,
                       LoopToScevMapT &LTS, isl_id_to_ast_expr *NewAccesses);

  /// Helper to determine if @p Inst can be synthesized in @p Stmt.
  ///
  /// @returns false, iff @p Inst can be synthesized in @p Stmt.
  bool canSyntheziseInStmt(ScopStmt &Stmt, Instruction *Inst);

  /// Remove dead instructions generated for BB
  ///
  /// @param BB The basic block code for which code has been generated.
  /// @param BBMap A local map from old to new instructions.
  void removeDeadInstructions(BasicBlock *BB, ValueMapT &BBMap);

  /// Invalidate the scalar evolution expressions for a scop.
  ///
  /// This function invalidates the scalar evolution results for all
  /// instructions that are part of a given scop, and the loops
  /// surrounding the users of merge blocks. This is necessary to ensure that
  /// later scops do not obtain scalar evolution expressions that reference
  /// values that earlier dominated the later scop, but have been moved in the
  /// conditional part of an earlier scop and consequently do not any more
  /// dominate the later scop.
  ///
  /// @param S The scop to invalidate.
  void invalidateScalarEvolution(Scop &S);
};

/// Generate a new vector basic block for a polyhedral statement.
///
/// The only public function exposed is generate().
class VectorBlockGenerator : BlockGenerator {
public:
  /// Generate a new vector basic block for a ScoPStmt.
  ///
  /// This code generation is similar to the normal, scalar code generation,
  /// except that each instruction is code generated for several vector lanes
  /// at a time. If possible instructions are issued as actual vector
  /// instructions, but e.g. for address calculation instructions we currently
  /// generate scalar instructions for each vector lane.
  ///
  /// @param BlockGen    A block generator object used as parent.
  /// @param Stmt        The statement to code generate.
  /// @param VLTS        A mapping from loops virtual canonical induction
  ///                    variable to their new values
  ///                    (for values recalculated in the new ScoP, but not
  ///                     within this basic block), one for each lane.
  /// @param Schedule    A map from the statement to a schedule where the
  ///                    innermost dimension is the dimension of the innermost
  ///                    loop containing the statement.
  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  static void generate(BlockGenerator &BlockGen, ScopStmt &Stmt,
                       std::vector<LoopToScevMapT> &VLTS,
                       __isl_keep isl_map *Schedule,
                       __isl_keep isl_id_to_ast_expr *NewAccesses) {
    VectorBlockGenerator Generator(BlockGen, VLTS, Schedule);
    Generator.copyStmt(Stmt, NewAccesses);
  }

private:
  // This is a vector of loop->scev maps.  The first map is used for the first
  // vector lane, ...
  // Each map, contains information about Instructions in the old ScoP, which
  // are recalculated in the new SCoP. When copying the basic block, we replace
  // all references to the old instructions with their recalculated values.
  //
  // For example, when the code generator produces this AST:
  //
  //   for (int c1 = 0; c1 <= 1023; c1 += 1)
  //     for (int c2 = 0; c2 <= 1023; c2 += VF)
  //       for (int lane = 0; lane <= VF; lane += 1)
  //         Stmt(c2 + lane + 3, c1);
  //
  // VLTS[lane] contains a map:
  //   "outer loop in the old loop nest" -> SCEV("c2 + lane + 3"),
  //   "inner loop in the old loop nest" -> SCEV("c1").
  std::vector<LoopToScevMapT> &VLTS;

  // A map from the statement to a schedule where the innermost dimension is the
  // dimension of the innermost loop containing the statement.
  isl_map *Schedule;

  VectorBlockGenerator(BlockGenerator &BlockGen,
                       std::vector<LoopToScevMapT> &VLTS,
                       __isl_keep isl_map *Schedule);

  int getVectorWidth();

  Value *getVectorValue(ScopStmt &Stmt, Value *Old, ValueMapT &VectorMap,
                        VectorValueMapT &ScalarMaps, Loop *L);

  /// Load a vector from a set of adjacent scalars
  ///
  /// In case a set of scalars is known to be next to each other in memory,
  /// create a vector load that loads those scalars
  ///
  /// %vector_ptr= bitcast double* %p to <4 x double>*
  /// %vec_full = load <4 x double>* %vector_ptr
  ///
  /// @param Stmt           The statement to code generate.
  /// @param NegativeStride This is used to indicate a -1 stride. In such
  ///                       a case we load the end of a base address and
  ///                       shuffle the accesses in reverse order into the
  ///                       vector. By default we would do only positive
  ///                       strides.
  ///
  /// @param NewAccesses    A map from memory access ids to new ast
  ///                       expressions, which may contain new access
  ///                       expressions for certain memory accesses.
  Value *generateStrideOneLoad(ScopStmt &Stmt, LoadInst *Load,
                               VectorValueMapT &ScalarMaps,
                               __isl_keep isl_id_to_ast_expr *NewAccesses,
                               bool NegativeStride);

  /// Load a vector initialized from a single scalar in memory
  ///
  /// In case all elements of a vector are initialized to the same
  /// scalar value, this value is loaded and shuffled into all elements
  /// of the vector.
  ///
  /// %splat_one = load <1 x double>* %p
  /// %splat = shufflevector <1 x double> %splat_one, <1 x
  ///       double> %splat_one, <4 x i32> zeroinitializer
  ///
  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  Value *generateStrideZeroLoad(ScopStmt &Stmt, LoadInst *Load,
                                ValueMapT &BBMap,
                                __isl_keep isl_id_to_ast_expr *NewAccesses);

  /// Load a vector from scalars distributed in memory
  ///
  /// In case some scalars a distributed randomly in memory. Create a vector
  /// by loading each scalar and by inserting one after the other into the
  /// vector.
  ///
  /// %scalar_1= load double* %p_1
  /// %vec_1 = insertelement <2 x double> undef, double %scalar_1, i32 0
  /// %scalar 2 = load double* %p_2
  /// %vec_2 = insertelement <2 x double> %vec_1, double %scalar_1, i32 1
  ///
  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  Value *generateUnknownStrideLoad(ScopStmt &Stmt, LoadInst *Load,
                                   VectorValueMapT &ScalarMaps,
                                   __isl_keep isl_id_to_ast_expr *NewAccesses);

  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  void generateLoad(ScopStmt &Stmt, LoadInst *Load, ValueMapT &VectorMap,
                    VectorValueMapT &ScalarMaps,
                    __isl_keep isl_id_to_ast_expr *NewAccesses);

  void copyUnaryInst(ScopStmt &Stmt, UnaryInstruction *Inst,
                     ValueMapT &VectorMap, VectorValueMapT &ScalarMaps);

  void copyBinaryInst(ScopStmt &Stmt, BinaryOperator *Inst,
                      ValueMapT &VectorMap, VectorValueMapT &ScalarMaps);

  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  void copyStore(ScopStmt &Stmt, StoreInst *Store, ValueMapT &VectorMap,
                 VectorValueMapT &ScalarMaps,
                 __isl_keep isl_id_to_ast_expr *NewAccesses);

  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  void copyInstScalarized(ScopStmt &Stmt, Instruction *Inst,
                          ValueMapT &VectorMap, VectorValueMapT &ScalarMaps,
                          __isl_keep isl_id_to_ast_expr *NewAccesses);

  bool extractScalarValues(const Instruction *Inst, ValueMapT &VectorMap,
                           VectorValueMapT &ScalarMaps);

  bool hasVectorOperands(const Instruction *Inst, ValueMapT &VectorMap);

  /// Generate vector loads for scalars.
  ///
  /// @param Stmt           The scop statement for which to generate the loads.
  /// @param VectorBlockMap A map that will be updated to relate the original
  ///                       values with the newly generated vector loads.
  void generateScalarVectorLoads(ScopStmt &Stmt, ValueMapT &VectorBlockMap);

  /// Verify absence of scalar stores.
  ///
  /// @param Stmt The scop statement to check for scalar stores.
  void verifyNoScalarStores(ScopStmt &Stmt);

  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  void copyInstruction(ScopStmt &Stmt, Instruction *Inst, ValueMapT &VectorMap,
                       VectorValueMapT &ScalarMaps,
                       __isl_keep isl_id_to_ast_expr *NewAccesses);

  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  void copyStmt(ScopStmt &Stmt, __isl_keep isl_id_to_ast_expr *NewAccesses);
};

/// Generator for new versions of polyhedral region statements.
class RegionGenerator : public BlockGenerator {
public:
  /// Create a generator for regions.
  ///
  /// @param BlockGen A generator for basic blocks.
  RegionGenerator(BlockGenerator &BlockGen) : BlockGenerator(BlockGen) {}

  virtual ~RegionGenerator() {}

  /// Copy the region statement @p Stmt.
  ///
  /// This copies the entire region represented by @p Stmt and updates
  /// references to old values with references to new values, as defined by
  /// GlobalMap.
  ///
  /// @param Stmt      The statement to code generate.
  /// @param LTS       A map from old loops to new induction variables as SCEVs.
  void copyStmt(ScopStmt &Stmt, LoopToScevMapT &LTS,
                __isl_keep isl_id_to_ast_expr *IdToAstExp);

private:
  /// A map from old to the first new block in the region, that was created to
  /// model the old basic block.
  DenseMap<BasicBlock *, BasicBlock *> StartBlockMap;

  /// A map from old to the last new block in the region, that was created to
  /// model the old basic block.
  DenseMap<BasicBlock *, BasicBlock *> EndBlockMap;

  /// The "BBMaps" for the whole region (one for each block). In case a basic
  /// block is code generated to multiple basic blocks (e.g., for partial
  /// writes), the StartBasic is used as index for the RegionMap.
  DenseMap<BasicBlock *, ValueMapT> RegionMaps;

  /// Mapping to remember PHI nodes that still need incoming values.
  using PHINodePairTy = std::pair<PHINode *, PHINode *>;
  DenseMap<BasicBlock *, SmallVector<PHINodePairTy, 4>> IncompletePHINodeMap;

  /// Repair the dominance tree after we created a copy block for @p BB.
  ///
  /// @returns The immediate dominator in the DT for @p BBCopy if in the region.
  BasicBlock *repairDominance(BasicBlock *BB, BasicBlock *BBCopy);

  /// Add the new operand from the copy of @p IncomingBB to @p PHICopy.
  ///
  /// PHI nodes, which may have (multiple) edges that enter from outside the
  /// non-affine subregion and even from outside the scop, are code generated as
  /// follows:
  ///
  /// # Original
  ///
  ///   Region: %A-> %exit
  ///   NonAffine Stmt: %nonaffB -> %D (includes %nonaffB, %nonaffC)
  ///
  ///     pre:
  ///       %val = add i64 1, 1
  ///
  ///     A:
  ///      br label %nonaff
  ///
  ///     nonaffB:
  ///       %phi = phi i64 [%val, %A], [%valC, %nonAffC], [%valD, %D]
  ///       %cmp = <nonaff>
  ///       br i1 %cmp, label %C, label %nonaffC
  ///
  ///     nonaffC:
  ///       %valC = add i64 1, 1
  ///       br i1 undef, label %D, label %nonaffB
  ///
  ///     D:
  ///       %valD = ...
  ///       %exit_cond = <loopexit>
  ///       br i1 %exit_cond, label %nonaffB, label %exit
  ///
  ///     exit:
  ///       ...
  ///
  ///  - %start and %C enter from outside the non-affine region.
  ///  - %nonaffC enters from within the non-affine region.
  ///
  ///  # New
  ///
  ///    polly.A:
  ///       store i64 %val, i64* %phi.phiops
  ///       br label %polly.nonaffA.entry
  ///
  ///    polly.nonaffB.entry:
  ///       %phi.phiops.reload = load i64, i64* %phi.phiops
  ///       br label %nonaffB
  ///
  ///    polly.nonaffB:
  ///       %polly.phi = [%phi.phiops.reload, %nonaffB.entry],
  ///                    [%p.valC, %polly.nonaffC]
  ///
  ///    polly.nonaffC:
  ///       %p.valC = add i64 1, 1
  ///       br i1 undef, label %polly.D, label %polly.nonaffB
  ///
  ///    polly.D:
  ///        %p.valD = ...
  ///        store i64 %p.valD, i64* %phi.phiops
  ///        %p.exit_cond = <loopexit>
  ///        br i1 %p.exit_cond, label %polly.nonaffB, label %exit
  ///
  /// Values that enter the PHI from outside the non-affine region are stored
  /// into the stack slot %phi.phiops by statements %polly.A and %polly.D and
  /// reloaded in %polly.nonaffB.entry, a basic block generated before the
  /// actual non-affine region.
  ///
  /// When generating the PHI node of the non-affine region in %polly.nonaffB,
  /// incoming edges from outside the region are combined into a single branch
  /// from %polly.nonaffB.entry which has as incoming value the value reloaded
  /// from the %phi.phiops stack slot. Incoming edges from within the region
  /// refer to the copied instructions (%p.valC) and basic blocks
  /// (%polly.nonaffC) of the non-affine region.
  ///
  /// @param Stmt       The statement to code generate.
  /// @param PHI        The original PHI we copy.
  /// @param PHICopy    The copy of @p PHI.
  /// @param IncomingBB An incoming block of @p PHI.
  /// @param LTS        A map from old loops to new induction variables as
  /// SCEVs.
  void addOperandToPHI(ScopStmt &Stmt, PHINode *PHI, PHINode *PHICopy,
                       BasicBlock *IncomingBB, LoopToScevMapT &LTS);

  /// Create a PHI that combines the incoming values from all incoming blocks
  /// that are in the subregion.
  ///
  /// PHIs in the subregion's exit block can have incoming edges from within and
  /// outside the subregion. This function combines the incoming values from
  /// within the subregion to appear as if there is only one incoming edge from
  /// the subregion (an additional exit block is created by RegionGenerator).
  /// This is to avoid that a value is written to the .phiops location without
  /// leaving the subregion because the exiting block as an edge back into the
  /// subregion.
  ///
  /// @param MA    The WRITE of MemoryKind::PHI/MemoryKind::ExitPHI for a PHI in
  ///              the subregion's exit block.
  /// @param LTS   Virtual induction variable mapping.
  /// @param BBMap A mapping from old values to their new values in this block.
  /// @param L     Loop surrounding this region statement.
  ///
  /// @returns The constructed PHI node.
  PHINode *buildExitPHI(MemoryAccess *MA, LoopToScevMapT &LTS, ValueMapT &BBMap,
                        Loop *L);

  /// @param Return the new value of a scalar write, creating a PHINode if
  ///        necessary.
  ///
  /// @param MA    A scalar WRITE MemoryAccess.
  /// @param LTS   Virtual induction variable mapping.
  /// @param BBMap A mapping from old values to their new values in this block.
  ///
  /// @returns The effective value of @p MA's written value when leaving the
  ///          subregion.
  /// @see buildExitPHI
  Value *getExitScalar(MemoryAccess *MA, LoopToScevMapT &LTS, ValueMapT &BBMap);

  /// Generate the scalar stores for the given statement.
  ///
  /// After the statement @p Stmt was copied all inner-SCoP scalar dependences
  /// starting in @p Stmt (hence all scalar write accesses in @p Stmt) need to
  /// be demoted to memory.
  ///
  /// @param Stmt  The statement we generate code for.
  /// @param LTS   A mapping from loops virtual canonical induction variable to
  ///              their new values (for values recalculated in the new ScoP,
  ///              but not within this basic block)
  /// @param BBMap A mapping from old values to their new values in this block.
  /// @param LTS   A mapping from loops virtual canonical induction variable to
  /// their new values.
  virtual void
  generateScalarStores(ScopStmt &Stmt, LoopToScevMapT &LTS, ValueMapT &BBMAp,
                       __isl_keep isl_id_to_ast_expr *NewAccesses) override;

  /// Copy a single PHI instruction.
  ///
  /// This copies a single PHI instruction and updates references to old values
  /// with references to new values, as defined by GlobalMap and BBMap.
  ///
  /// @param Stmt      The statement to code generate.
  /// @param PHI       The PHI instruction to copy.
  /// @param BBMap     A mapping from old values to their new values
  ///                  (for values recalculated within this basic block).
  /// @param LTS       A map from old loops to new induction variables as SCEVs.
  virtual void copyPHIInstruction(ScopStmt &Stmt, PHINode *Inst,
                                  ValueMapT &BBMap,
                                  LoopToScevMapT &LTS) override;
};
} // namespace polly
#endif
