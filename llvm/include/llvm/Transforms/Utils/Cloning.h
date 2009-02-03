//===- Cloning.h - Clone various parts of LLVM programs ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines various functions that are used to clone chunks of LLVM
// code for various purposes.  This varies from copying whole modules into new
// modules, to cloning functions with different arguments, to inlining
// functions, to copying basic blocks to support loop unrolling or superblock
// formation, etc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_CLONING_H
#define LLVM_TRANSFORMS_UTILS_CLONING_H

#include <vector>
#include "llvm/ADT/DenseMap.h"

namespace llvm {

class Module;
class Function;
class Pass;
class LPPassManager;
class BasicBlock;
class Value;
class CallInst;
class InvokeInst;
class ReturnInst;
class CallSite;
class Trace;
class CallGraph;
class TargetData;
class LoopInfo;
template<class N> class LoopBase;
typedef LoopBase<BasicBlock> Loop;

/// CloneModule - Return an exact copy of the specified module
///
Module *CloneModule(const Module *M);
Module *CloneModule(const Module *M, DenseMap<const Value*, Value*> &ValueMap);

/// ClonedCodeInfo - This struct can be used to capture information about code
/// being cloned, while it is being cloned.
struct ClonedCodeInfo {
  /// ContainsCalls - This is set to true if the cloned code contains a normal
  /// call instruction.
  bool ContainsCalls;
  
  /// ContainsUnwinds - This is set to true if the cloned code contains an
  /// unwind instruction.
  bool ContainsUnwinds;
  
  /// ContainsDynamicAllocas - This is set to true if the cloned code contains
  /// a 'dynamic' alloca.  Dynamic allocas are allocas that are either not in
  /// the entry block or they are in the entry block but are not a constant
  /// size.
  bool ContainsDynamicAllocas;
  
  ClonedCodeInfo() {
    ContainsCalls = false;
    ContainsUnwinds = false;
    ContainsDynamicAllocas = false;
  }
};


/// CloneBasicBlock - Return a copy of the specified basic block, but without
/// embedding the block into a particular function.  The block returned is an
/// exact copy of the specified basic block, without any remapping having been
/// performed.  Because of this, this is only suitable for applications where
/// the basic block will be inserted into the same function that it was cloned
/// from (loop unrolling would use this, for example).
///
/// Also, note that this function makes a direct copy of the basic block, and
/// can thus produce illegal LLVM code.  In particular, it will copy any PHI
/// nodes from the original block, even though there are no predecessors for the
/// newly cloned block (thus, phi nodes will have to be updated).  Also, this
/// block will branch to the old successors of the original block: these
/// successors will have to have any PHI nodes updated to account for the new
/// incoming edges.
///
/// The correlation between instructions in the source and result basic blocks
/// is recorded in the ValueMap map.
///
/// If you have a particular suffix you'd like to use to add to any cloned
/// names, specify it as the optional third parameter.
///
/// If you would like the basic block to be auto-inserted into the end of a
/// function, you can specify it as the optional fourth parameter.
///
/// If you would like to collect additional information about the cloned
/// function, you can specify a ClonedCodeInfo object with the optional fifth
/// parameter.
///
BasicBlock *CloneBasicBlock(const BasicBlock *BB,
                            DenseMap<const Value*, Value*> &ValueMap,
                            const char *NameSuffix = "", Function *F = 0,
                            ClonedCodeInfo *CodeInfo = 0);


/// CloneLoop - Clone Loop. Clone dominator info for loop insiders. Populate ValueMap
/// using old blocks to new blocks mapping.
Loop *CloneLoop(Loop *L, LPPassManager  *LPM, LoopInfo *LI, 
                DenseMap<const Value *, Value *> &ValueMap, Pass *P);

/// CloneFunction - Return a copy of the specified function, but without
/// embedding the function into another module.  Also, any references specified
/// in the ValueMap are changed to refer to their mapped value instead of the
/// original one.  If any of the arguments to the function are in the ValueMap,
/// the arguments are deleted from the resultant function.  The ValueMap is
/// updated to include mappings from all of the instructions and basicblocks in
/// the function from their old to new values.  The final argument captures
/// information about the cloned code if non-null.
///
Function *CloneFunction(const Function *F,
                        DenseMap<const Value*, Value*> &ValueMap,
                        ClonedCodeInfo *CodeInfo = 0);

/// CloneFunction - Version of the function that doesn't need the ValueMap.
///
inline Function *CloneFunction(const Function *F, ClonedCodeInfo *CodeInfo = 0){
  DenseMap<const Value*, Value*> ValueMap;
  return CloneFunction(F, ValueMap, CodeInfo);
}

/// Clone OldFunc into NewFunc, transforming the old arguments into references
/// to ArgMap values.  Note that if NewFunc already has basic blocks, the ones
/// cloned into it will be added to the end of the function.  This function
/// fills in a list of return instructions, and can optionally append the
/// specified suffix to all values cloned.
///
void CloneFunctionInto(Function *NewFunc, const Function *OldFunc,
                       DenseMap<const Value*, Value*> &ValueMap,
                       std::vector<ReturnInst*> &Returns,
                       const char *NameSuffix = "", 
                       ClonedCodeInfo *CodeInfo = 0);

/// CloneAndPruneFunctionInto - This works exactly like CloneFunctionInto,
/// except that it does some simple constant prop and DCE on the fly.  The
/// effect of this is to copy significantly less code in cases where (for
/// example) a function call with constant arguments is inlined, and those
/// constant arguments cause a significant amount of code in the callee to be
/// dead.  Since this doesn't produce an exactly copy of the input, it can't be
/// used for things like CloneFunction or CloneModule.
void CloneAndPruneFunctionInto(Function *NewFunc, const Function *OldFunc,
                               DenseMap<const Value*, Value*> &ValueMap,
                               std::vector<ReturnInst*> &Returns,
                               const char *NameSuffix = "", 
                               ClonedCodeInfo *CodeInfo = 0,
                               const TargetData *TD = 0);


/// CloneTraceInto - Clone T into NewFunc. Original<->clone mapping is
/// saved in ValueMap.
///
void CloneTraceInto(Function *NewFunc, Trace &T,
                    DenseMap<const Value*, Value*> &ValueMap,
                    const char *NameSuffix);

/// CloneTrace - Returns a copy of the specified trace.
/// It takes a vector of basic blocks clones the basic blocks, removes internal
/// phi nodes, adds it to the same function as the original (although there is
/// no jump to it) and returns the new vector of basic blocks.
std::vector<BasicBlock *> CloneTrace(const std::vector<BasicBlock*> &origTrace);

/// InlineFunction - This function inlines the called function into the basic
/// block of the caller.  This returns false if it is not possible to inline
/// this call.  The program is still in a well defined state if this occurs
/// though.
///
/// Note that this only does one level of inlining.  For example, if the
/// instruction 'call B' is inlined, and 'B' calls 'C', then the call to 'C' now
/// exists in the instruction stream.  Similiarly this will inline a recursive
/// function by one level.
///
/// If a non-null callgraph pointer is provided, these functions update the
/// CallGraph to represent the program after inlining.
///
bool InlineFunction(CallInst *C, CallGraph *CG = 0, const TargetData *TD = 0);
bool InlineFunction(InvokeInst *II, CallGraph *CG = 0, const TargetData *TD =0);
bool InlineFunction(CallSite CS, CallGraph *CG = 0, const TargetData *TD = 0);

} // End llvm namespace

#endif
