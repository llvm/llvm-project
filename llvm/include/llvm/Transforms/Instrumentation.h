//===- Transforms/Instrumentation.h - Instrumentation passes ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines constructor functions for instrumentation passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_H

namespace llvm {

class ModulePass;
class FunctionPass;

// Insert function profiling instrumentation
ModulePass *createFunctionProfilerPass();

// Insert block profiling instrumentation
ModulePass *createBlockProfilerPass();

// Insert edge profiling instrumentation
ModulePass *createEdgeProfilerPass();

// Random Sampling Profiling Framework
ModulePass* createNullProfilerRSPass();
FunctionPass* createRSProfilingPass();

} // End llvm namespace

#endif
