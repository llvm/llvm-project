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

#include <vector>

namespace llvm {

class FunctionPass;
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
// These functions removes symbols from functions and modules.  
// Only debugging information is not removed.
//
ModulePass *createStripNonDebugSymbolsPass();

//===----------------------------------------------------------------------===//
/// createLowerSetJmpPass - This function lowers the setjmp/longjmp intrinsics
/// to invoke/unwind instructions.  This should really be part of the C/C++
/// front-end, but it's so much easier to write transformations in LLVM proper.
///
ModulePass *createLowerSetJmpPass();

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
/// createRaiseAllocationsPass - Return a new pass that transforms malloc and
/// free function calls into malloc and free instructions.
///
ModulePass *createRaiseAllocationsPass();


//===----------------------------------------------------------------------===//
/// createDeadTypeEliminationPass - Return a new pass that eliminates symbol
/// table entries for types that are never used.
///
ModulePass *createDeadTypeEliminationPass();


//===----------------------------------------------------------------------===//
/// createGlobalDCEPass - This transform is designed to eliminate unreachable
/// internal globals (functions or global variables)
///
ModulePass *createGlobalDCEPass();


//===----------------------------------------------------------------------===//
/// createGVExtractionPass - If deleteFn is true, this pass deletes as
/// the specified global values. Otherwise, it deletes as much of the module as
/// possible, except for the global values specified.
///
ModulePass *createGVExtractionPass(std::vector<GlobalValue*>& GVs, bool 
                                   deleteFn = false, 
                                   bool relinkCallees = false);

//===----------------------------------------------------------------------===//
/// createFunctionInliningPass - Return a new pass object that uses a heuristic
/// to inline direct function calls to small functions.
///
Pass *createFunctionInliningPass();
Pass *createFunctionInliningPass(int Threshold);

//===----------------------------------------------------------------------===//
/// createAlwaysInlinerPass - Return a new pass object that inlines only 
/// functions that are marked as "always_inline".
Pass *createAlwaysInlinerPass();

//===----------------------------------------------------------------------===//
/// createPruneEHPass - Return a new pass object which transforms invoke
/// instructions into calls, if the callee can _not_ unwind the stack.
///
Pass *createPruneEHPass();

//===----------------------------------------------------------------------===//
/// createInternalizePass - This pass loops over all of the functions in the
/// input module, internalizing all globals (functions and variables) not part
/// of the api.  If a list of symbols is specified with the
/// -internalize-public-api-* command line options, those symbols are not
/// internalized and all others are.  Otherwise if AllButMain is set and the
/// main function is found, all other globals are marked as internal. If no api
/// is supplied and AllButMain is not set, or no main function is found, nothing
/// is internalized.
///
ModulePass *createInternalizePass(bool AllButMain);

/// createInternalizePass - This pass loops over all of the functions in the
/// input module, internalizing all globals (functions and variables) not in the
/// given exportList.
///
/// Note that commandline options that are used with the above function are not
/// used now! Also, when exportList is empty, nothing is internalized.
ModulePass *createInternalizePass(const std::vector<const char *> &exportList);

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
Pass *createStructRetPromotionPass();

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
FunctionPass *createLoopExtractorPass();

/// createSingleLoopExtractorPass - This pass extracts one natural loop from the
/// program into a function if it can.  This is used by bugpoint.
///
FunctionPass *createSingleLoopExtractorPass();

/// createBlockExtractorPass - This pass extracts all blocks (except those
/// specified in the argument list) from the functions in the module.
///
ModulePass *createBlockExtractorPass(const std::vector<BasicBlock*> &BTNE);

/// createIndMemRemPass - This pass removes potential indirect calls of
/// malloc and free
ModulePass *createIndMemRemPass();

/// createStripDeadPrototypesPass - This pass removes any function declarations
/// (prototypes) that are not used.
ModulePass *createStripDeadPrototypesPass();

//===----------------------------------------------------------------------===//
/// createPartialSpecializationPass - This pass specializes functions for
/// constant arguments.
///
ModulePass *createPartialSpecializationPass();

//===----------------------------------------------------------------------===//
/// createFunctionAttrsPass - This pass discovers functions that do not access
/// memory, or only read memory, and gives them the readnone/readonly attribute.
/// It also discovers function arguments that are not captured by the function
/// and marks them with the nocapture attribute.
///
Pass *createFunctionAttrsPass();

//===----------------------------------------------------------------------===//
/// createMergeFunctionsPass - This pass discovers identical functions and
/// collapses them.
///
ModulePass *createMergeFunctionsPass();

} // End llvm namespace

#endif
