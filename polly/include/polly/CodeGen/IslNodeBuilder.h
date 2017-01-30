//===------ IslNodeBuilder.cpp - Translate an isl AST into a LLVM-IR AST---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file contains the IslNodeBuilder, a class to translate an isl AST into
// a LLVM-IR AST.
//===----------------------------------------------------------------------===//

#ifndef POLLY_ISL_NODE_BUILDER_H
#define POLLY_ISL_NODE_BUILDER_H

#include "polly/CodeGen/BlockGenerators.h"
#include "polly/CodeGen/IslExprBuilder.h"
#include "polly/CodeGen/LoopGenerators.h"
#include "polly/ScopInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "isl/ctx.h"
#include "isl/union_map.h"
#include <utility>
#include <vector>

using namespace polly;
using namespace llvm;

struct isl_ast_node;
struct isl_ast_build;
struct isl_union_map;

struct SubtreeReferences {
  LoopInfo &LI;
  ScalarEvolution &SE;
  Scop &S;
  ValueMapT &GlobalMap;
  SetVector<Value *> &Values;
  SetVector<const SCEV *> &SCEVs;
  BlockGenerator &BlockGen;
};

/// Extract the out-of-scop values and SCEVs referenced from a ScopStmt.
///
/// This includes the SCEVUnknowns referenced by the SCEVs used in the
/// statement and the base pointers of the memory accesses. For scalar
/// statements we force the generation of alloca memory locations and list
/// these locations in the set of out-of-scop values as well.
///
/// @param Stmt             The statement for which to extract the information.
/// @param UserPtr          A void pointer that can be casted to a
///                         SubtreeReferences structure.
/// @param CreateScalarRefs Should the result include allocas of scalar
///                         references?
isl_stat addReferencesFromStmt(const ScopStmt *Stmt, void *UserPtr,
                               bool CreateScalarRefs = true);

class IslNodeBuilder {
public:
  IslNodeBuilder(PollyIRBuilder &Builder, ScopAnnotator &Annotator, Pass *P,
                 const DataLayout &DL, LoopInfo &LI, ScalarEvolution &SE,
                 DominatorTree &DT, Scop &S, BasicBlock *StartBlock)
      : S(S), Builder(Builder), Annotator(Annotator),
        ExprBuilder(S, Builder, IDToValue, ValueMap, DL, SE, DT, LI,
                    StartBlock),
        BlockGen(Builder, LI, SE, DT, ScalarMap, EscapeMap, ValueMap,
                 &ExprBuilder, StartBlock),
        RegionGen(BlockGen), P(P), DL(DL), LI(LI), SE(SE), DT(DT),
        StartBlock(StartBlock) {}

  virtual ~IslNodeBuilder() = default;

  void addParameters(__isl_take isl_set *Context);

  /// Generate code that evaluates @p Condition at run-time.
  ///
  /// This function is typically called to generate the LLVM-IR for the
  /// run-time condition of the scop, that verifies that all the optimistic
  /// assumptions we have taken during scop modeling and transformation
  /// hold at run-time.
  ///
  /// @param Condition The condition to evaluate
  ///
  /// @result An llvm::Value that is true if the condition holds and false
  ///         otherwise.
  Value *createRTC(isl_ast_expr *Condition);

  void create(__isl_take isl_ast_node *Node);

  /// Allocate memory for all new arrays created by Polly.
  void allocateNewArrays();

  /// Preload all memory loads that are invariant.
  bool preloadInvariantLoads();

  /// Finalize code generation.
  ///
  /// @see BlockGenerator::finalizeSCoP(Scop &S)
  virtual void finalize() { BlockGen.finalizeSCoP(S); }

  IslExprBuilder &getExprBuilder() { return ExprBuilder; }

  /// Get the associated block generator.
  ///
  /// @return A referecne to the associated block generator.
  BlockGenerator &getBlockGenerator() { return BlockGen; }

  /// Return the parallel subfunctions that have been created.
  const ArrayRef<Function *> getParallelSubfunctions() const {
    return ParallelSubfunctions;
  }

protected:
  Scop &S;
  PollyIRBuilder &Builder;
  ScopAnnotator &Annotator;

  IslExprBuilder ExprBuilder;

  /// Maps used by the block and region generator to demote scalars.
  ///
  ///@{

  /// See BlockGenerator::ScalarMap.
  BlockGenerator::AllocaMapTy ScalarMap;

  /// See BlockGenerator::EscapeMap.
  BlockGenerator::EscapeUsersAllocaMapTy EscapeMap;

  ///@}

  /// The generator used to copy a basic block.
  BlockGenerator BlockGen;

  /// The generator used to copy a non-affine region.
  RegionGenerator RegionGen;

  Pass *const P;
  const DataLayout &DL;
  LoopInfo &LI;
  ScalarEvolution &SE;
  DominatorTree &DT;
  BasicBlock *StartBlock;

  /// The current iteration of out-of-scop loops
  ///
  /// This map provides for a given loop a llvm::Value that contains the current
  /// loop iteration.
  LoopToScevMapT OutsideLoopIterations;

  // This maps an isl_id* to the Value* it has in the generated program. For now
  // on, the only isl_ids that are stored here are the newly calculated loop
  // ivs.
  IslExprBuilder::IDToValueTy IDToValue;

  /// A collection of all parallel subfunctions that have been created.
  SmallVector<Function *, 8> ParallelSubfunctions;

  /// Generate code for a given SCEV*
  ///
  /// This function generates code for a given SCEV expression. It generated
  /// code is emitted at the end of the basic block our Builder currently
  /// points to and the resulting value is returned.
  ///
  /// @param Expr The expression to code generate.
  llvm::Value *generateSCEV(const SCEV *Expr);

  /// A set of Value -> Value remappings to apply when generating new code.
  ///
  /// When generating new code for a ScopStmt this map is used to map certain
  /// llvm::Values to new llvm::Values.
  ValueMapT ValueMap;

  /// Materialize code for @p Id if it was not done before.
  ///
  /// @returns False, iff a problem occured and the value was not materialized.
  bool materializeValue(__isl_take isl_id *Id);

  /// Materialize parameters of @p Set.
  ///
  /// @param All If not set only parameters referred to by the constraints in
  ///            @p Set will be materialized, otherwise all.
  ///
  /// @returns False, iff a problem occurred and the value was not materialized.
  bool materializeParameters(__isl_take isl_set *Set, bool All);

  // Extract the upper bound of this loop
  //
  // The isl code generation can generate arbitrary expressions to check if the
  // upper bound of a loop is reached, but it provides an option to enforce
  // 'atomic' upper bounds. An 'atomic upper bound is always of the form
  // iv <= expr, where expr is an (arbitrary) expression not containing iv.
  //
  // This function extracts 'atomic' upper bounds. Polly, in general, requires
  // atomic upper bounds for the following reasons:
  //
  // 1. An atomic upper bound is loop invariant
  //
  //    It must not be calculated at each loop iteration and can often even be
  //    hoisted out further by the loop invariant code motion.
  //
  // 2. OpenMP needs a loop invariant upper bound to calculate the number
  //    of loop iterations.
  //
  // 3. With the existing code, upper bounds have been easier to implement.
  __isl_give isl_ast_expr *getUpperBound(__isl_keep isl_ast_node *For,
                                         CmpInst::Predicate &Predicate);

  /// Return non-negative number of iterations in case of the following form
  /// of a loop and -1 otherwise.
  ///
  /// for (i = 0; i <= NumIter; i++) {
  ///   loop body;
  /// }
  ///
  /// NumIter is a non-negative integer value. Condition can have
  /// isl_ast_op_lt type.
  int getNumberOfIterations(__isl_keep isl_ast_node *For);

  /// Compute the values and loops referenced in this subtree.
  ///
  /// This function looks at all ScopStmts scheduled below the provided For node
  /// and finds the llvm::Value[s] and llvm::Loops[s] which are referenced but
  /// not locally defined.
  ///
  /// Values that can be synthesized or that are available as globals are
  /// considered locally defined.
  ///
  /// Loops that contain the scop or that are part of the scop are considered
  /// locally defined. Loops that are before the scop, but do not contain the
  /// scop itself are considered not locally defined.
  ///
  /// @param For    The node defining the subtree.
  /// @param Values A vector that will be filled with the Values referenced in
  ///               this subtree.
  /// @param Loops  A vector that will be filled with the Loops referenced in
  ///               this subtree.
  void getReferencesInSubtree(__isl_keep isl_ast_node *For,
                              SetVector<Value *> &Values,
                              SetVector<const Loop *> &Loops);

  /// Change the llvm::Value(s) used for code generation.
  ///
  /// When generating code certain values (e.g., references to induction
  /// variables or array base pointers) in the original code may be replaced by
  /// new values. This function allows to (partially) update the set of values
  /// used. A typical use case for this function is the case when we continue
  /// code generation in a subfunction/kernel function and need to explicitly
  /// pass down certain values.
  ///
  /// @param NewValues A map that maps certain llvm::Values to new llvm::Values.
  void updateValues(ValueMapT &NewValues);

  /// Generate code for a marker now.
  ///
  /// For mark nodes with an unknown name, we just forward the code generation
  /// to its child. This is currently the only behavior implemented, as there is
  /// currently not special handling for marker nodes implemented.
  ///
  /// @param Mark The node we generate code for.
  virtual void createMark(__isl_take isl_ast_node *Marker);
  virtual void createFor(__isl_take isl_ast_node *For);

  /// Set to remember materialized invariant loads.
  ///
  /// An invariant load is identified by its pointer (the SCEV) and its type.
  SmallSet<std::pair<const SCEV *, Type *>, 16> PreloadedPtrs;

  /// Preload the memory access at @p AccessRange with @p Build.
  ///
  /// @returns The preloaded value casted to type @p Ty
  Value *preloadUnconditionally(__isl_take isl_set *AccessRange,
                                isl_ast_build *Build, Instruction *AccInst);

  /// Preload the memory load access @p MA.
  ///
  /// If @p MA is not always executed it will be conditionally loaded and
  /// merged with undef from the same type. Hence, if @p MA is executed only
  /// under condition C then the preload code will look like this:
  ///
  /// MA_preload = undef;
  /// if (C)
  ///   MA_preload = load MA;
  /// use MA_preload
  Value *preloadInvariantLoad(const MemoryAccess &MA,
                              __isl_take isl_set *Domain);

  /// Preload the invariant access equivalence class @p IAClass
  ///
  /// This function will preload the representing load from @p IAClass and
  /// map all members of @p IAClass to that preloaded value, potentially casted
  /// to the required type.
  ///
  /// @returns False, iff a problem occurred and the load was not preloaded.
  bool preloadInvariantEquivClass(InvariantEquivClassTy &IAClass);

  void createForVector(__isl_take isl_ast_node *For, int VectorWidth);
  void createForSequential(__isl_take isl_ast_node *For, bool KnownParallel);

  /// Create LLVM-IR that executes a for node thread parallel.
  ///
  /// @param For The FOR isl_ast_node for which code is generated.
  void createForParallel(__isl_take isl_ast_node *For);

  /// Create new access functions for modified memory accesses.
  ///
  /// In case the access function of one of the memory references in the Stmt
  /// has been modified, we generate a new isl_ast_expr that reflects the
  /// newly modified access function and return a map that maps from the
  /// individual memory references in the statement (identified by their id)
  /// to these newly generated ast expressions.
  ///
  /// @param Stmt  The statement for which to (possibly) generate new access
  ///              functions.
  /// @param Node  The ast node corresponding to the statement for us to extract
  ///              the local schedule from.
  /// @return A new hash table that contains remappings from memory ids to new
  ///         access expressions.
  __isl_give isl_id_to_ast_expr *
  createNewAccesses(ScopStmt *Stmt, __isl_keep isl_ast_node *Node);

  /// Generate LLVM-IR that computes the values of the original induction
  /// variables in function of the newly generated loop induction variables.
  ///
  /// Example:
  ///
  ///   // Original
  ///   for i
  ///     for j
  ///       S(i)
  ///
  ///   Schedule: [i,j] -> [i+j, j]
  ///
  ///   // New
  ///   for c0
  ///     for c1
  ///       S(c0 - c1, c1)
  ///
  /// Assuming the original code consists of two loops which are
  /// transformed according to a schedule [i,j] -> [c0=i+j,c1=j]. The resulting
  /// ast models the original statement as a call expression where each argument
  /// is an expression that computes the old induction variables from the new
  /// ones, ordered such that the first argument computes the value of induction
  /// variable that was outermost in the original code.
  ///
  /// @param Expr The call expression that represents the statement.
  /// @param Stmt The statement that is called.
  /// @param LTS  The loop to SCEV map in which the mapping from the original
  ///             loop to a SCEV representing the new loop iv is added. This
  ///             mapping does not require an explicit induction variable.
  ///             Instead, we think in terms of an implicit induction variable
  ///             that counts the number of times a loop is executed. For each
  ///             original loop this count, expressed in function of the new
  ///             induction variables, is added to the LTS map.
  void createSubstitutions(__isl_take isl_ast_expr *Expr, ScopStmt *Stmt,
                           LoopToScevMapT &LTS);
  void createSubstitutionsVector(__isl_take isl_ast_expr *Expr, ScopStmt *Stmt,
                                 std::vector<LoopToScevMapT> &VLTS,
                                 std::vector<Value *> &IVS,
                                 __isl_take isl_id *IteratorID);
  virtual void createIf(__isl_take isl_ast_node *If);
  void createUserVector(__isl_take isl_ast_node *User,
                        std::vector<Value *> &IVS,
                        __isl_take isl_id *IteratorID,
                        __isl_take isl_union_map *Schedule);
  virtual void createUser(__isl_take isl_ast_node *User);
  virtual void createBlock(__isl_take isl_ast_node *Block);

  /// Get the schedule for a given AST node.
  ///
  /// This information is used to reason about parallelism of loops or the
  /// locality of memory accesses under a given schedule.
  ///
  /// @param Node The node we want to obtain the schedule for.
  /// @return Return an isl_union_map that maps from the statements executed
  ///         below this ast node to the scheduling vectors used to enumerate
  ///         them.
  ///
  virtual __isl_give isl_union_map *
  getScheduleForAstNode(__isl_take isl_ast_node *Node);

private:
  /// Create code for a copy statement.
  ///
  /// A copy statement is expected to have one read memory access and one write
  /// memory access (in this very order). Data is loaded from the location
  /// described by the read memory access and written to the location described
  /// by the write memory access. @p NewAccesses contains for each access
  /// the isl ast expression that describes the location accessed.
  ///
  /// @param Stmt The copy statement that contains the accesses.
  /// @param NewAccesses The hash table that contains remappings from memory
  ///                    ids to new access expressions.
  void generateCopyStmt(ScopStmt *Stmt,
                        __isl_keep isl_id_to_ast_expr *NewAccesses);
};

#endif // POLLY_ISL_NODE_BUILDER_H
