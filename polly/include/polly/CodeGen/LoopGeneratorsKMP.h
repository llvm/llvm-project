//===- LoopGeneratorsKMP.h - IR helper to create loops ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to create scalar and OpenMP parallel loops
// as LLVM-IR.
//
//===----------------------------------------------------------------------===//
#ifndef POLLY_LOOP_GENERATORS_KMP_H
#define POLLY_LOOP_GENERATORS_KMP_H

#include "polly/CodeGen/IRBuilder.h"
#include "polly/CodeGen/LoopGenerators.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/SetVector.h"

namespace polly {
using llvm::GlobalValue;
using llvm::GlobalVariable;

/// This ParallelLoopGenerator subclass handles the generation of parallelized
/// code, utilizing the LLVM OpenMP library.
class ParallelLoopGeneratorKMP final : public ParallelLoopGenerator {
public:
  /// Create a parallel loop generator for the current function.
  ParallelLoopGeneratorKMP(PollyIRBuilder &Builder, LoopInfo &LI,
                           DominatorTree &DT, const DataLayout &DL)
      : ParallelLoopGenerator(Builder, LI, DT, DL) {
    SourceLocationInfo = createSourceLocation();
  }

protected:
  /// The source location struct of this loop.
  /// ident_t = type { i32, i32, i32, i32, i8* }
  GlobalValue *SourceLocationInfo;

  /// Convert the combination of given chunk size and scheduling type (which
  /// might have been set via the command line) into the corresponding
  /// scheduling type. This may result (e.g.) in a 'change' from
  /// "static chunked" scheduling to "static non-chunked" (regarding the
  /// provided and returned scheduling types).
  ///
  /// @param ChunkSize    The chunk size, set via command line or its default.
  /// @param Scheduling   The scheduling, set via command line or its default.
  ///
  /// @return The corresponding OMPGeneralSchedulingType.
  OMPGeneralSchedulingType
  getSchedType(int ChunkSize, OMPGeneralSchedulingType Scheduling) const;

  /// Returns True if 'LongType' is 64bit wide, otherwise: False.
  bool is64BitArch();

public:
  // The functions below may be used if one does not want to generate a
  // specific OpenMP parallel loop, but generate individual parts of it
  // (e.g. the subfunction definition).

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

  void deployParallelExecution(Function *SubFn, Value *SubFnParam, Value *LB,
                               Value *UB, Value *Stride) override;

  Function *prepareSubFnDefinition(Function *F) const override;

  std::tuple<Value *, Function *> createSubFn(Value *Stride, AllocaInst *Struct,
                                              SetVector<Value *> UsedValues,
                                              ValueMapT &VMap) override;

  /// Create a runtime library call to get the current global thread number.
  ///
  /// @return A Value ref which holds the current global thread number.
  Value *createCallGlobalThreadNum();

  /// Create a runtime library call to request a number of threads.
  /// Which will be used in the next OpenMP section (by the next fork).
  ///
  /// @param GlobalThreadID   The global thread ID.
  /// @param NumThreads       The number of threads to use.
  void createCallPushNumThreads(Value *GlobalThreadID, Value *NumThreads);

  /// Create a runtime library call to prepare the OpenMP runtime.
  /// For dynamically scheduled loops, saving the loop arguments.
  ///
  /// @param GlobalThreadID   The global thread ID.
  /// @param LB               The loop's lower bound.
  /// @param UB               The loop's upper bound.
  /// @param Inc              The loop increment.
  /// @param ChunkSize        The chunk size of the parallel loop.
  void createCallDispatchInit(Value *GlobalThreadID, Value *LB, Value *UB,
                              Value *Inc, Value *ChunkSize);

  /// Create a runtime library call to retrieve the next (dynamically)
  /// allocated chunk of work for this thread.
  ///
  /// @param GlobalThreadID   The global thread ID.
  /// @param IsLastPtr        Pointer to a flag, which is set to 1 if this is
  ///                         the last chunk of work, or 0 otherwise.
  /// @param LBPtr            Pointer to the lower bound for the next chunk.
  /// @param UBPtr            Pointer to the upper bound for the next chunk.
  /// @param StridePtr        Pointer to the stride for the next chunk.
  ///
  /// @return A Value which holds 1 if there is work to be done, 0 otherwise.
  Value *createCallDispatchNext(Value *GlobalThreadID, Value *IsLastPtr,
                                Value *LBPtr, Value *UBPtr, Value *StridePtr);

  /// Create a runtime library call to prepare the OpenMP runtime.
  /// For statically scheduled loops, saving the loop arguments.
  ///
  /// @param GlobalThreadID   The global thread ID.
  /// @param IsLastPtr        Pointer to a flag, which is set to 1 if this is
  ///                         the last chunk of work, or 0 otherwise.
  /// @param LBPtr            Pointer to the lower bound for the next chunk.
  /// @param UBPtr            Pointer to the upper bound for the next chunk.
  /// @param StridePtr        Pointer to the stride for the next chunk.
  /// @param ChunkSize        The chunk size of the parallel loop.
  void createCallStaticInit(Value *GlobalThreadID, Value *IsLastPtr,
                            Value *LBPtr, Value *UBPtr, Value *StridePtr,
                            Value *ChunkSize);

  /// Create a runtime library call to mark the end of
  /// a statically scheduled loop.
  ///
  /// @param GlobalThreadID   The global thread ID.
  void createCallStaticFini(Value *GlobalThreadID);

  /// Create the current source location.
  ///
  /// TODO: Generates only(!) dummy values.
  GlobalVariable *createSourceLocation();
};
} // end namespace polly
#endif
