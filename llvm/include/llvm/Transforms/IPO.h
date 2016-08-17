//===- llvm/Transforms/IPO.h - Interprocedural Transformations --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the IPO transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_H
#define LLVM_TRANSFORMS_IPO_H

#include <functional>
#include <vector>

namespace llvm {

struct InlineParams;
class StringRef;
class ModuleSummaryIndex;
class ModulePass;
class Pass;
class Function;
class BasicBlock;
class GlobalValue;

//===----------------------------------------------------------------------===//
//
// These functions removes symbols from functions and modules.  If OnlyDebugInfo
// is true, only debugging information is removed from the module.
//
ModulePass *createStripSymbolsPass(bool OnlyDebugInfo = false);

//===----------------------------------------------------------------------===//
//
// These functions strips symbols from functions and modules.
// Only debugging information is not stripped.
//
ModulePass *createStripNonDebugSymbolsPass();

//===----------------------------------------------------------------------===//
//
// These pass removes llvm.dbg.declare intrinsics.
ModulePass *createStripDebugDeclarePass();

//===----------------------------------------------------------------------===//
//
// These pass removes unused symbols' debug info.
ModulePass *createStripDeadDebugInfoPass();

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
/// possible, except for the global values specified.
///
ModulePass *createGVExtractionPass(std::vector<GlobalValue*>& GVs, bool
                                   deleteFn = false);

//===----------------------------------------------------------------------===//
/// This pass performs iterative function importing from other modules.
Pass *createFunctionImportPass(const ModuleSummaryIndex *Index = nullptr);

//===----------------------------------------------------------------------===//
/// createFunctionInliningPass - Return a new pass object that uses a heuristic
/// to inline direct function calls to small functions.
///
/// The Threshold can be passed directly, or asked to be computed from the
/// given optimization and size optimization arguments.
///
/// The -inline-threshold command line option takes precedence over the
/// threshold given here.
Pass *createFunctionInliningPass();
Pass *createFunctionInliningPass(int Threshold);
Pass *createFunctionInliningPass(unsigned OptLevel, unsigned SizeOptLevel);
Pass *createFunctionInliningPass(InlineParams &Params);

//===----------------------------------------------------------------------===//
/// createPruneEHPass - Return a new pass object which transforms invoke
/// instructions into calls, if the callee can _not_ unwind the stack.
///
Pass *createPruneEHPass();

//===----------------------------------------------------------------------===//
/// createInternalizePass - This pass loops over all of the functions in the
/// input module, internalizing all globals (functions and variables) it can.
////
/// Before internalizing a symbol, the callback \p MustPreserveGV is invoked and
/// gives to the client the ability to prevent internalizing specific symbols.
///
/// The symbol in DSOList are internalized if it is safe to drop them from
/// the symbol table.
///
/// Note that commandline options that are used with the above function are not
/// used now!
ModulePass *
createInternalizePass(std::function<bool(const GlobalValue &)> MustPreserveGV);

/// createInternalizePass - Same as above, but with an empty exportList.
ModulePass *createInternalizePass();

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
/// createArgumentPromotionPass - This pass promotes "by reference" arguments to
/// be passed by value if the number of elements passed is smaller or
/// equal to maxElements (maxElements == 0 means always promote).
///
Pass *createArgumentPromotionPass(unsigned maxElements = 3);

//===----------------------------------------------------------------------===//
/// createIPConstantPropagationPass - This pass propagates constants from call
/// sites into the bodies of functions.
///
ModulePass *createIPConstantPropagationPass();

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

/// createBlockExtractorPass - This pass extracts all blocks (except those
/// specified in the argument list) from the functions in the module.
///
ModulePass *createBlockExtractorPass();

/// createStripDeadPrototypesPass - This pass removes any function declarations
/// (prototypes) that are not used.
ModulePass *createStripDeadPrototypesPass();

//===----------------------------------------------------------------------===//
/// createReversePostOrderFunctionAttrsPass - This pass walks SCCs of the call
/// graph in RPO to deduce and propagate function attributes. Currently it
/// only handles synthesizing norecurse attributes.
///
Pass *createReversePostOrderFunctionAttrsPass();

//===----------------------------------------------------------------------===//
/// createMergeFunctionsPass - This pass discovers identical functions and
/// collapses them.
///
ModulePass *createMergeFunctionsPass();

//===----------------------------------------------------------------------===//
/// createPartialInliningPass - This pass inlines parts of functions.
///
ModulePass *createPartialInliningPass();

//===----------------------------------------------------------------------===//
// createMetaRenamerPass - Rename everything with metasyntatic names.
//
ModulePass *createMetaRenamerPass();

//===----------------------------------------------------------------------===//
/// createBarrierNoopPass - This pass is purely a module pass barrier in a pass
/// manager.
ModulePass *createBarrierNoopPass();

/// \brief This pass lowers type metadata and the llvm.type.test intrinsic to
/// bitsets.
ModulePass *createLowerTypeTestsPass();

/// \brief This pass export CFI checks for use by external modules.
ModulePass *createCrossDSOCFIPass();

/// \brief This pass implements whole-program devirtualization using type
/// metadata.
ModulePass *createWholeProgramDevirtPass();

//===----------------------------------------------------------------------===//
// SampleProfilePass - Loads sample profile data from disk and generates
// IR metadata to reflect the profile.
ModulePass *createSampleProfileLoaderPass();
ModulePass *createSampleProfileLoaderPass(StringRef Name);

} // End llvm namespace

#endif
