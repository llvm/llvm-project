//===- LoopGenerators.h - IR helper to create loops -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create scalar and OpenMP parallel loops
// as LLVM-IR.
//
//===----------------------------------------------------------------------===//
#ifndef POLLY_LOOP_GENERATORS_H
#define POLLY_LOOP_GENERATORS_H

#include "polly/CodeGen/IRBuilder.h"
#include "polly/Support/ScopHelper.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/ValueMap.h"

namespace llvm {
class Value;
class Pass;
class BasicBlock;
} // namespace llvm

namespace polly {
using namespace llvm;

/// Create a scalar do/for-style loop.
///
/// @param LowerBound         The starting value of the induction variable.
/// @param UpperBound         The upper bound of the induction variable.
/// @param Stride             The value by which the induction variable
///                           is incremented.
///
/// @param Builder            The builder used to create the loop.
/// @param P                  A pointer to the pass that uses this function.
///                           It is used to update analysis information.
/// @param LI                 The loop info for the current function
/// @param DT                 The dominator tree we need to update
/// @param ExitBlock          The block the loop will exit to.
/// @param Predicate          The predicate used to generate the upper loop
///                           bound.
/// @param Annotator          This function can (optionally) take
///                           a ScopAnnotator which
///                           annotates loops and alias information in the SCoP.
/// @param Parallel           If this loop should be marked parallel in
///                           the Annotator.
/// @param UseGuard           Create a guard in front of the header to check if
///                           the loop is executed at least once, otherwise just
///                           assume it.
/// @param LoopVectDisabled   If the Loop vectorizer should be disabled for this
///                           loop.
///
/// @return Value*    The newly created induction variable for this loop.
Value *createLoop(Value *LowerBound, Value *UpperBound, Value *Stride,
                  PollyIRBuilder &Builder, LoopInfo &LI, DominatorTree &DT,
                  BasicBlock *&ExitBlock, ICmpInst::Predicate Predicate,
                  ScopAnnotator *Annotator = NULL, bool Parallel = false,
                  bool UseGuard = true, bool LoopVectDisabled = false);

/// The ParallelLoopGenerator allows to create parallelized loops
///
/// To parallelize a loop, we perform the following steps:
///   o  Generate a subfunction which will hold the loop body.
///   o  Create a struct to hold all outer values needed in the loop body.
///   o  Create calls to a runtime library to achieve the actual parallelism.
///      These calls will spawn and join threads, define how the work (here the
///      iterations) are distributed between them and make sure each has access
///      to the struct holding all needed values.
///
/// At the moment we support only one parallel runtime, OpenMP.
///
/// If we parallelize the outer loop of the following loop nest,
///
///   S0;
///   for (int i = 0; i < N; i++)
///     for (int j = 0; j < M; j++)
///       S1(i, j);
///   S2;
///
/// we will generate the following code (with different runtime function names):
///
///   S0;
///   auto *values = storeValuesIntoStruct();
///   // Execute subfunction with multiple threads
///   spawn_threads(subfunction, values);
///   join_threads();
///   S2;
///
///  // This function is executed in parallel by different threads
///   void subfunction(values) {
///     while (auto *WorkItem = getWorkItem()) {
///       int LB = WorkItem.begin();
///       int UB = WorkItem.end();
///       for (int i = LB; i < UB; i++)
///         for (int j = 0; j < M; j++)
///           S1(i, j);
///     }
///     cleanup_thread();
///   }
class ParallelLoopGenerator {
public:
  /// Create a parallel loop generator for the current function.
  ParallelLoopGenerator(PollyIRBuilder &Builder, LoopInfo &LI,
                        DominatorTree &DT, const DataLayout &DL)
      : Builder(Builder), LI(LI), DT(DT),
        LongType(
            Type::getIntNTy(Builder.getContext(), DL.getPointerSizeInBits())),
        M(Builder.GetInsertBlock()->getParent()->getParent()) {}

  /// Create a parallel loop.
  ///
  /// This function is the main function to automatically generate a parallel
  /// loop with all its components.
  ///
  /// @param LB        The lower bound for the loop we parallelize.
  /// @param UB        The upper bound for the loop we parallelize.
  /// @param Stride    The stride of the loop we parallelize.
  /// @param Values    A set of LLVM-IR Values that should be available in
  ///                  the new loop body.
  /// @param VMap      A map to allow outside access to the new versions of
  ///                  the values in @p Values.
  /// @param LoopBody  A pointer to an iterator that is set to point to the
  ///                  body of the created loop. It should be used to insert
  ///                  instructions that form the actual loop body.
  ///
  /// @return The newly created induction variable for this loop.
  Value *createParallelLoop(Value *LB, Value *UB, Value *Stride,
                            SetVector<Value *> &Values, ValueMapT &VMap,
                            BasicBlock::iterator *LoopBody);

private:
  /// The IR builder we use to create instructions.
  PollyIRBuilder &Builder;

  /// The loop info of the current function we need to update.
  LoopInfo &LI;

  /// The dominance tree of the current function we need to update.
  DominatorTree &DT;

  /// The type of a "long" on this hardware used for backend calls.
  Type *LongType;

  /// The current module
  Module *M;

public:
  /// The functions below can be used if one does not want to generate a
  /// specific OpenMP parallel loop, but generate individual parts of it
  /// (e.g., the subfunction definition).

  /// Create a runtime library call to spawn the worker threads.
  ///
  /// @param SubFn      The subfunction which holds the loop body.
  /// @param SubFnParam The parameter for the subfunction (basically the struct
  ///                   filled with the outside values).
  /// @param LB         The lower bound for the loop we parallelize.
  /// @param UB         The upper bound for the loop we parallelize.
  /// @param Stride     The stride of the loop we parallelize.
  void createCallSpawnThreads(Value *SubFn, Value *SubFnParam, Value *LB,
                              Value *UB, Value *Stride);

  /// Create a runtime library call to join the worker threads.
  void createCallJoinThreads();

  /// Create a runtime library call to get the next work item.
  ///
  /// @param LBPtr A pointer value to store the work item begin in.
  /// @param UBPtr A pointer value to store the work item end in.
  ///
  /// @returns A true value if the work item is not empty.
  Value *createCallGetWorkItem(Value *LBPtr, Value *UBPtr);

  /// Create a runtime library call to allow cleanup of the thread.
  ///
  /// @note This function is called right before the thread will exit the
  ///       subfunction and only if the runtime system depends on it.
  void createCallCleanupThread();

  /// Create a struct for all @p Values and store them in there.
  ///
  /// @param Values The values which should be stored in the struct.
  ///
  /// @return The created struct.
  AllocaInst *storeValuesIntoStruct(SetVector<Value *> &Values);

  /// Extract all values from the @p Struct and construct the mapping.
  ///
  /// @param Values The values which were stored in the struct.
  /// @param Struct The struct holding all the values in @p Values.
  /// @param VMap   A map to associate every element of @p Values with the
  ///               new llvm value loaded from the @p Struct.
  void extractValuesFromStruct(SetVector<Value *> Values, Type *Ty,
                               Value *Struct, ValueMapT &VMap);

  /// Create the definition of the parallel subfunction.
  Function *createSubFnDefinition();

  /// Create the parallel subfunction.
  ///
  /// @param Stride The induction variable increment.
  /// @param Struct A struct holding all values in @p Values.
  /// @param Values A set of LLVM-IR Values that should be available in
  ///               the new loop body.
  /// @param VMap   A map to allow outside access to the new versions of
  ///               the values in @p Values.
  /// @param SubFn  The newly created subfunction is returned here.
  ///
  /// @return The newly created induction variable.
  Value *createSubFn(Value *Stride, AllocaInst *Struct,
                     SetVector<Value *> UsedValues, ValueMapT &VMap,
                     Function **SubFn);
};
} // end namespace polly
#endif
