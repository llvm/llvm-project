//===-- llvm/Analysis/Passes.h - Constructors for analyses ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the analysis libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_PASSES_H
#define LLVM_ANALYSIS_PASSES_H

namespace llvm {
  class FunctionPass;
  class ImmutablePass;
  class LoopPass;
  class ModulePass;
  class Pass;
  class PassInfo;
  class LibCallInfo;

  //===--------------------------------------------------------------------===//
  //
  // createGlobalsModRefPass - This pass provides alias and mod/ref info for
  // global values that do not have their addresses taken.
  //
  Pass *createGlobalsModRefPass();

  //===--------------------------------------------------------------------===//
  //
  // createAliasDebugger - This pass helps debug clients of AA
  //
  Pass *createAliasDebugger();

  //===--------------------------------------------------------------------===//
  //
  // createAliasAnalysisCounterPass - This pass counts alias queries and how the
  // alias analysis implementation responds.
  //
  ModulePass *createAliasAnalysisCounterPass();

  //===--------------------------------------------------------------------===//
  //
  // createAAEvalPass - This pass implements a simple N^2 alias analysis
  // accuracy evaluator.
  //
  FunctionPass *createAAEvalPass();

  //===--------------------------------------------------------------------===//
  //
  // createNoAAPass - This pass implements a "I don't know" alias analysis.
  //
  ImmutablePass *createNoAAPass();

  //===--------------------------------------------------------------------===//
  //
  // createBasicAliasAnalysisPass - This pass implements the stateless alias
  // analysis.
  //
  ImmutablePass *createBasicAliasAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  /// createLibCallAliasAnalysisPass - Create an alias analysis pass that knows
  /// about the semantics of a set of libcalls specified by LCI.  The newly
  /// constructed pass takes ownership of the pointer that is provided.
  ///
  FunctionPass *createLibCallAliasAnalysisPass(LibCallInfo *LCI);

  //===--------------------------------------------------------------------===//
  //
  // createScalarEvolutionAliasAnalysisPass - This pass implements a simple
  // alias analysis using ScalarEvolution queries.
  //
  FunctionPass *createScalarEvolutionAliasAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  // createTypeBasedAliasAnalysisPass - This pass implements metadata-based
  // type-based alias analysis.
  //
  ImmutablePass *createTypeBasedAliasAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  // createObjCARCAliasAnalysisPass - This pass implements ObjC-ARC-based
  // alias analysis.
  //
  ImmutablePass *createObjCARCAliasAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  // createProfileLoaderPass - This pass loads information from a profile dump
  // file.
  //
  ModulePass *createProfileLoaderPass();
  extern char &ProfileLoaderPassID;

  //===--------------------------------------------------------------------===//
  //
  // createProfileMetadataLoaderPass - This pass loads information from a
  // profile dump file and sets branch weight metadata.
  //
  ModulePass *createProfileMetadataLoaderPass();
  extern char &ProfileMetadataLoaderPassID;

  //===--------------------------------------------------------------------===//
  //
  // createNoProfileInfoPass - This pass implements the default "no profile".
  //
  ImmutablePass *createNoProfileInfoPass();

  //===--------------------------------------------------------------------===//
  //
  // createProfileEstimatorPass - This pass estimates profiling information
  // instead of loading it from a previous run.
  //
  FunctionPass *createProfileEstimatorPass();
  extern char &ProfileEstimatorPassID;

  //===--------------------------------------------------------------------===//
  //
  // createProfileVerifierPass - This pass verifies profiling information.
  //
  FunctionPass *createProfileVerifierPass();

  //===--------------------------------------------------------------------===//
  //
  // createPathProfileLoaderPass - This pass loads information from a path
  // profile dump file.
  //
  ModulePass *createPathProfileLoaderPass();
  extern char &PathProfileLoaderPassID;

  //===--------------------------------------------------------------------===//
  //
  // createNoPathProfileInfoPass - This pass implements the default
  // "no path profile".
  //
  ImmutablePass *createNoPathProfileInfoPass();

  //===--------------------------------------------------------------------===//
  //
  // createPathProfileVerifierPass - This pass verifies path profiling
  // information.
  //
  ModulePass *createPathProfileVerifierPass();

  //===--------------------------------------------------------------------===//
  //
  // createDSAAPass - This pass implements simple context sensitive alias
  // analysis.
  //
  ModulePass *createDSAAPass();

  //===--------------------------------------------------------------------===//
  //
  // createDSOptPass - This pass uses DSA to do a series of simple
  // optimizations.
  //
  ModulePass *createDSOptPass();

  //===--------------------------------------------------------------------===//
  //
  // createSteensgaardPass - This pass uses the data structure graphs to do a
  // simple context insensitive alias analysis.
  //
  ModulePass *createSteensgaardPass();

  //===--------------------------------------------------------------------===//
  //
  /// createLazyValueInfoPass - This creates an instance of the LazyValueInfo
  /// pass.
  FunctionPass *createLazyValueInfoPass();

  //===--------------------------------------------------------------------===//
  //
  // createDependenceAnalysisPass - This creates an instance of the
  // DependenceAnalysis pass.
  //
  FunctionPass *createDependenceAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  // createCostModelAnalysisPass - This creates an instance of the
  // CostModelAnalysis pass.
  //
  FunctionPass *createCostModelAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  // Minor pass prototypes, allowing us to expose them through bugpoint and
  // analyze.
  FunctionPass *createInstCountPass();

  // print debug info intrinsics in human readable form
  FunctionPass *createDbgInfoPrinterPass();

  //===--------------------------------------------------------------------===//
  //
  // createRegionInfoPass - This pass finds all single entry single exit regions
  // in a function and builds the region hierarchy.
  //
  FunctionPass *createRegionInfoPass();

  // Print module-level debug info metadata in human-readable form.
  ModulePass *createModuleDebugInfoPrinterPass();

  //===--------------------------------------------------------------------===//
  //
  // createMemDepPrinter - This pass exhaustively collects all memdep
  // information and prints it with -analyze.
  //
  FunctionPass *createMemDepPrinter();
}

#endif
