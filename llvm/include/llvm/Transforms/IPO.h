//===- llvm/Transforms/IPO.h - Interprocedural Transformations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the IPO transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_H
#define LLVM_TRANSFORMS_IPO_H

#include "llvm/ADT/SmallVector.h"
#include <functional>
#include <vector>

namespace llvm {

struct InlineParams;
class ModulePass;
class Pass;
class BasicBlock;
class GlobalValue;
class raw_ostream;

//===----------------------------------------------------------------------===//
//
// This pass adds !annotation metadata to entries in the
// @llvm.global.annotations global constant.
//
ModulePass *createAnnotation2MetadataLegacyPass();

//===----------------------------------------------------------------------===//
/// createConstantMergePass - This function returns a new pass that merges
/// duplicate global constants together into a single constant that is shared.
/// This is useful because some passes (ie TraceValues) insert a lot of string
/// constants into the program, regardless of whether or not they duplicate an
/// existing string.
///
ModulePass *createConstantMergePass();

//===----------------------------------------------------------------------===//
/// createGlobalOptimizerPass - This function returns a new pass that optimizes
/// non-address taken internal globals.
///
ModulePass *createGlobalOptimizerPass();

//===----------------------------------------------------------------------===//
/// createGlobalDCEPass - This transform is designed to eliminate unreachable
/// internal globals (functions or global variables)
///
ModulePass *createGlobalDCEPass();

//===----------------------------------------------------------------------===//
/// This transform is designed to eliminate available external globals
/// (functions or global variables)
///
ModulePass *createEliminateAvailableExternallyPass();

//===----------------------------------------------------------------------===//
/// createGVExtractionPass - If deleteFn is true, this pass deletes
/// the specified global values. Otherwise, it deletes as much of the module as
/// possible, except for the global values specified. If keepConstInit is true,
/// the initializers of global constants are not deleted even if they are
/// unused.
///
ModulePass *createGVExtractionPass(std::vector<GlobalValue*>& GVs, bool
                                  deleteFn = false, bool keepConstInit = false);

//===----------------------------------------------------------------------===//
/// createDeadArgEliminationPass - This pass removes arguments from functions
/// which are not used by the body of the function.
///
ModulePass *createDeadArgEliminationPass();

/// DeadArgHacking pass - Same as DAE, but delete arguments of external
/// functions as well.  This is definitely not safe, and should only be used by
/// bugpoint.
ModulePass *createDeadArgHackingPass();

//===----------------------------------------------------------------------===//
/// createIPSCCPPass - This pass propagates constants from call sites into the
/// bodies of functions, and keeps track of whether basic blocks are executable
/// in the process.
///
ModulePass *createIPSCCPPass();

//===----------------------------------------------------------------------===//
//
/// createLoopExtractorPass - This pass extracts all natural loops from the
/// program into a function if it can.
///
Pass *createLoopExtractorPass();

/// createSingleLoopExtractorPass - This pass extracts one natural loop from the
/// program into a function if it can.  This is used by bugpoint.
///
Pass *createSingleLoopExtractorPass();

//===----------------------------------------------------------------------===//
/// createBarrierNoopPass - This pass is purely a module pass barrier in a pass
/// manager.
ModulePass *createBarrierNoopPass();

/// What to do with the summary when running passes that operate on it.
enum class PassSummaryAction {
  None,   ///< Do nothing.
  Import, ///< Import information from summary.
  Export, ///< Export information to summary.
};

/// This pass export CFI checks for use by external modules.
ModulePass *createCrossDSOCFIPass();

/// This pass splits globals into pieces for the benefit of whole-program
/// devirtualization and control-flow integrity.
ModulePass *createGlobalSplitPass();

} // End llvm namespace

#endif
