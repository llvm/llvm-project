//===-BlockGenerators.h - Helper to generate code for statements-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "isl/map.h"

struct isl_ast_build;
struct isl_id_to_ast_expr;

namespace llvm {
class Pass;
class Region;
class ScalarEvolution;
}

namespace polly {
using namespace llvm;
class ScopStmt;
class MemoryAccess;
class ScopArrayInfo;
class IslExprBuilder;

/// @brief Generate a new basic block for a polyhedral statement.
class BlockGenerator {
public:
  typedef llvm::SmallVector<ValueMapT, 8> VectorValueMapT;

  /// @brief Map types to resolve scalar dependences.
  ///
  ///@{

  /// @see The ScalarMap and PHIOpMap member.
  using ScalarAllocaMapTy = DenseMap<AssertingVH<Value>, AssertingVH<Value>>;

  /// @brief Simple vector of instructions to store escape users.
  using EscapeUserVectorTy = SmallVector<Instruction *, 4>;

  /// @brief Map type to resolve escaping users for scalar instructions.
  ///
  /// @see The EscapeMap member.
  using EscapeUsersAllocaMapTy =
      DenseMap<Instruction *,
               std::pair<AssertingVH<Value>, EscapeUserVectorTy>>;

  ///@}

  /// @brief Create a generator for basic blocks.
  ///
  /// @param Builder     The LLVM-IR Builder used to generate the statement. The
  ///                    code is generated at the location, the Builder points
  ///                    to.
  /// @param LI          The loop info for the current function
  /// @param SE          The scalar evolution info for the current function
  /// @param DT          The dominator tree of this function.
  /// @param ScalarMap   Map from scalars to their demoted location.
  /// @param PHIOpMap    Map from PHIs to their demoted operand location.
  /// @param EscapeMap   Map from scalars to their escape users and locations.
  /// @param GlobalMap   A mapping from llvm::Values used in the original scop
  ///                    region to a new set of llvm::Values. Each reference to
  ///                    an original value appearing in this mapping is replaced
  ///                    with the new value it is mapped to.
  /// @param ExprBuilder An expression builder to generate new access functions.
  BlockGenerator(PollyIRBuilder &Builder, LoopInfo &LI, ScalarEvolution &SE,
                 DominatorTree &DT, ScalarAllocaMapTy &ScalarMap,
                 ScalarAllocaMapTy &PHIOpMap, EscapeUsersAllocaMapTy &EscapeMap,
                 ValueMapT &GlobalMap, IslExprBuilder *ExprBuilder = nullptr);

  /// @brief Copy the basic block.
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

  /// @brief Return the scalar alloca for @p ScalarBase
  ///
  /// If no alloca was mapped to @p ScalarBase a new one is created.
  ///
  /// @param ScalarBase The demoted scalar value.
  /// @param GlobalMap  A mapping from Allocas to other memory locations that
  ///                   can be used to replace the original alloca locations
  ///                   with new memory locations, e.g. when passing values to
  ///                   subfunctions while offloading parallel sections.
  ///
  /// @returns The alloca for @p ScalarBase or a replacement value taken from
  ///          GlobalMap.
  Value *getOrCreateScalarAlloca(Value *ScalarBase);

  /// @brief Return the PHi-node alloca for @p ScalarBase
  ///
  /// If no alloca was mapped to @p ScalarBase a new one is created.
  ///
  /// @param ScalarBase The demoted scalar value.
  ///
  /// @returns The alloca for @p ScalarBase or a replacement value taken from
  ///          GlobalMap.
  Value *getOrCreatePHIAlloca(Value *ScalarBase);

  /// @brief Return the alloca for @p Access
  ///
  /// If no alloca was mapped for @p Access a new one is created.
  ///
  /// @param Access    The memory access for which to generate the alloca
  ///
  /// @returns The alloca for @p Access or a replacement value taken from
  ///          GlobalMap.
  Value *getOrCreateAlloca(MemoryAccess &Access);

  /// @brief Return the alloca for @p Array
  ///
  /// If no alloca was mapped for @p Array a new one is created.
  ///
  /// @param Array The array for which to generate the alloca
  ///
  /// @returns The alloca for @p Array or a replacement value taken from
  ///          GlobalMap.
  Value *getOrCreateAlloca(const ScopArrayInfo *Array);

  /// @brief Finalize the code generation for the SCoP @p S.
  ///
  /// This will initialize and finalize the scalar variables we demoted during
  /// the code generation.
  ///
  /// @see createScalarInitialization(Scop &)
  /// @see createScalarFinalization(Region &)
  void finalizeSCoP(Scop &S);

  /// @brief An empty destructor
  virtual ~BlockGenerator(){};

  BlockGenerator(const BlockGenerator &) = default;

protected:
  PollyIRBuilder &Builder;
  LoopInfo &LI;
  ScalarEvolution &SE;
  IslExprBuilder *ExprBuilder;

  /// @brief The dominator tree of this function.
  DominatorTree &DT;

  /// @brief The entry block of the current function.
  BasicBlock *EntryBB;

  /// @brief Maps to resolve scalar dependences for PHI operands and scalars.
  ///
  /// When translating code that contains scalar dependences as they result from
  /// inter-block scalar dependences (including the use of data carrying
  /// PHI nodes), we do not directly regenerate in-register SSA code, but
  /// instead allocate some stack memory through which these scalar values are
  /// passed. Only a later pass of -mem2reg will then (re)introduce in-register
  /// computations.
  ///
  /// To keep track of the memory location(s) used to store the data computed by
  /// a given SSA instruction, we use the maps 'ScalarMap' and 'PHIOpMap'. Each
  /// maps a given scalar value to a junk of stack allocated memory.
  ///
  /// 'ScalarMap' is used for normal scalar dependences that go from a scalar
  /// definition to its use. Such dependences are lowered by directly writing
  /// the value an instruction computes into the corresponding chunk of memory
  /// and reading it back from this chunk of memory right before every use of
  /// this original scalar value. The memory locations in 'ScalarMap' end with
  /// '.s2a'.
  ///
  /// 'PHIOpMap' is used to model PHI nodes. For each PHI nodes we introduce,
  /// besides the memory in 'ScalarMap', a second chunk of memory into which we
  /// write at the end of each basic block preceeding the PHI instruction the
  /// value passed through this basic block. At the place where the PHI node is
  /// executed, we replace the PHI node with a load from the corresponding
  /// memory location in the 'PHIOpMap' table. The memory locations in
  /// 'PHIOpMap' end with '.phiops'.
  ///
  /// The ScopArrayInfo objects of accesses that belong to a PHI node may have
  /// identical base pointers, even though they refer to two different memory
  /// locations, the normal '.s2a' locations and the special '.phiops'
  /// locations. For historic reasons we keep such accesses in two maps
  /// 'ScalarMap' and 'PHIOpMap', index by the BasePointer. An alternative
  /// implemenation, could use a single map that uses the ScopArrayInfo object
  /// as index.
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
  ///      ScalarMap = { x1 -> x1.s2a, x2 -> x2.s2a, add -> add.s2a }
  ///      PHIOpMap =  { x2 -> x2.phiops }
  ///
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
  ///{
  ///
  /// @brief Memory locations used for the special PHI node modeling.
  ScalarAllocaMapTy &PHIOpMap;

  /// @brief Memory locations used to model scalar dependences.
  ScalarAllocaMapTy &ScalarMap;
  ///}

  /// @brief Map from instructions to their escape users as well as the alloca.
  EscapeUsersAllocaMapTy &EscapeMap;

  /// @brief A map from llvm::Values referenced in the old code to a new set of
  ///        llvm::Values, which is used to replace these old values during
  ///        code generation.
  ValueMapT &GlobalMap;

  /// @brief Split @p BB to create a new one we can use to clone @p BB in.
  BasicBlock *splitBB(BasicBlock *BB);

  /// @brief Copy the given basic block.
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

  /// @brief Copy the given basic block.
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

  /// @brief Return the alloca for @p ScalarBase in @p Map.
  ///
  /// If no alloca was mapped to @p ScalarBase in @p Map a new one is created
  /// and named after @p ScalarBase with the suffix @p NameExt.
  ///
  /// @param ScalarBase The demoted scalar value.
  /// @param Map        The map we should look for a mapped alloca value.
  /// @param NameExt    The suffix we add to the name of a new created alloca.
  ///
  /// @returns The alloca for @p ScalarBase.
  Value *getOrCreateAlloca(Value *ScalarBase, ScalarAllocaMapTy &Map,
                           const char *NameExt);

  /// @brief Generate reload of scalars demoted to memory and needed by @p Stmt.
  ///
  /// @param Stmt  The statement we generate code for.
  /// @param BBMap A mapping from old values to their new values in this block.
  void generateScalarLoads(ScopStmt &Stmt, ValueMapT &BBMap);

  /// @brief Generate the scalar stores for the given statement.
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
  virtual void generateScalarStores(ScopStmt &Stmt, LoopToScevMapT &LTS,
                                    ValueMapT &BBMap);

  /// @brief Handle users of @p Inst outside the SCoP.
  ///
  /// @param R         The current SCoP region.
  /// @param Inst      The current instruction we check.
  /// @param Address   If given it is used as the escape address for @p Inst.
  void handleOutsideUsers(const Region &R, Instruction *Inst,
                          Value *Address = nullptr);

  /// @brief Find scalar statements that have outside users.
  ///
  /// We register these scalar values to later update subsequent scalar uses of
  /// these values to either use the newly computed value from within the scop
  /// (if the scop was executed) or the unchanged original code (if the run-time
  /// check failed).
  ///
  /// @param S The scop for which to find the outside users.
  void findOutsideUsers(Scop &S);

  /// @brief Initialize the memory of demoted scalars.
  ///
  /// @param S The scop for which to generate the scalar initializers.
  void createScalarInitialization(Scop &S);

  /// @brief Create exit PHI node merges for PHI nodes with more than two edges
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

  /// @brief Promote the values of demoted scalars after the SCoP.
  ///
  /// If a scalar value was used outside the SCoP we need to promote the value
  /// stored in the memory cell allocated for that scalar and combine it with
  /// the original value in the non-optimized SCoP.
  void createScalarFinalization(Region &R);

  /// @brief Try to synthesize a new value
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

  /// @brief Get the new version of a value.
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

  /// @brief Get the innermost loop that surrounds an instruction.
  ///
  /// @param Inst The instruction for which we get the loop.
  /// @return The innermost loop that surrounds the instruction.
  Loop *getLoopForInst(const Instruction *Inst);

  /// @brief Generate the operand address
  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  Value *generateLocationAccessed(ScopStmt &Stmt, const Instruction *Inst,
                                  Value *Pointer, ValueMapT &BBMap,
                                  LoopToScevMapT &LTS,
                                  isl_id_to_ast_expr *NewAccesses);

  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  Value *generateScalarLoad(ScopStmt &Stmt, LoadInst *load, ValueMapT &BBMap,
                            LoopToScevMapT &LTS,
                            isl_id_to_ast_expr *NewAccesses);

  /// @param NewAccesses A map from memory access ids to new ast expressions,
  ///                    which may contain new access expressions for certain
  ///                    memory accesses.
  void generateScalarStore(ScopStmt &Stmt, StoreInst *store, ValueMapT &BBMap,
                           LoopToScevMapT &LTS,
                           isl_id_to_ast_expr *NewAccesses);

  /// @brief Copy a single PHI instruction.
  ///
  /// The implementation in the BlockGenerator is trivial, however it allows
  /// subclasses to handle PHIs different.
  ///
  /// @returns The nullptr as the BlockGenerator does not copy PHIs.
  virtual Value *copyPHIInstruction(ScopStmt &, PHINode *, ValueMapT &,
                                    LoopToScevMapT &) {
    return nullptr;
  }

  /// @brief Copy a single Instruction.
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

  /// @brief Helper to determine if @p Inst can be synthezised in @p Stmt.
  ///
  /// @returns false, iff @p Inst can be synthesized in @p Stmt.
  bool canSyntheziseInStmt(ScopStmt &Stmt, Instruction *Inst);
};

/// @brief Generate a new vector basic block for a polyhedral statement.
///
/// The only public function exposed is generate().
class VectorBlockGenerator : BlockGenerator {
public:
  /// @brief Generate a new vector basic block for a ScoPStmt.
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
  ///                    loop containing the statemenet.
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
  // all referenes to the old instructions with their recalculated values.
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
  // dimension of the innermost loop containing the statemenet.
  isl_map *Schedule;

  VectorBlockGenerator(BlockGenerator &BlockGen,
                       std::vector<LoopToScevMapT> &VLTS,
                       __isl_keep isl_map *Schedule);

  int getVectorWidth();

  Value *getVectorValue(ScopStmt &Stmt, Value *Old, ValueMapT &VectorMap,
                        VectorValueMapT &ScalarMaps, Loop *L);

  Type *getVectorPtrTy(const Value *V, int Width);

  /// @brief Load a vector from a set of adjacent scalars
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

  /// @brief Load a vector initialized from a single scalar in memory
  ///
  /// In case all elements of a vector are initialized to the same
  /// scalar value, this value is loaded and shuffeled into all elements
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

  /// @brief Load a vector from scalars distributed in memory
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

  /// @brief Generate vector loads for scalars.
  ///
  /// @param Stmt           The scop statement for which to generate the loads.
  /// @param VectorBlockMap A map that will be updated to relate the original
  ///                       values with the newly generated vector loads.
  void generateScalarVectorLoads(ScopStmt &Stmt, ValueMapT &VectorBlockMap);

  /// @brief Verify absence of scalar stores.
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

/// @brief Generator for new versions of polyhedral region statements.
class RegionGenerator : public BlockGenerator {
public:
  /// @brief Create a generator for regions.
  ///
  /// @param BlockGen A generator for basic blocks.
  RegionGenerator(BlockGenerator &BlockGen) : BlockGenerator(BlockGen) {}

  virtual ~RegionGenerator(){};

  /// @brief Copy the region statement @p Stmt.
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
  /// @brief A map from old to new blocks in the region.
  DenseMap<BasicBlock *, BasicBlock *> BlockMap;

  /// @brief The "BBMaps" for the whole region (one for each block).
  DenseMap<BasicBlock *, ValueMapT> RegionMaps;

  /// @brief Mapping to remember PHI nodes that still need incoming values.
  using PHINodePairTy = std::pair<const PHINode *, PHINode *>;
  DenseMap<BasicBlock *, SmallVector<PHINodePairTy, 4>> IncompletePHINodeMap;

  /// @brief Repair the dominance tree after we created a copy block for @p BB.
  ///
  /// @returns The immediate dominator in the DT for @p BBCopy if in the region.
  BasicBlock *repairDominance(BasicBlock *BB, BasicBlock *BBCopy);

  /// @brief Add the new operand from the copy of @p IncomingBB to @p PHICopy.
  ///
  /// @param Stmt       The statement to code generate.
  /// @param PHI        The original PHI we copy.
  /// @param PHICopy    The copy of @p PHI.
  /// @param IncomingBB An incoming block of @p PHI.
  /// @param LTS        A map from old loops to new induction variables as
  /// SCEVs.
  void addOperandToPHI(ScopStmt &Stmt, const PHINode *PHI, PHINode *PHICopy,
                       BasicBlock *IncomingBB, LoopToScevMapT &LTS);

  /// @brief Generate the scalar stores for the given statement.
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
  virtual void generateScalarStores(ScopStmt &Stmt, LoopToScevMapT &LTS,
                                    ValueMapT &BBMAp) override;

  /// @brief Copy a single PHI instruction.
  ///
  /// This copies a single PHI instruction and updates references to old values
  /// with references to new values, as defined by GlobalMap and BBMap.
  ///
  /// @param Stmt      The statement to code generate.
  /// @param PHI       The PHI instruction to copy.
  /// @param BBMap     A mapping from old values to their new values
  ///                  (for values recalculated within this basic block).
  /// @param LTS       A map from old loops to new induction variables as SCEVs.
  ///
  /// @returns The copied instruction or nullptr if no copy was made.
  virtual Value *copyPHIInstruction(ScopStmt &Stmt, PHINode *Inst,
                                    ValueMapT &BBMap,
                                    LoopToScevMapT &LTS) override;
};
}
#endif
