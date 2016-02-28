//===- LoopPassManager.cpp - Loop pass management -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopPassManager.h"

using namespace llvm;

// Explicit instantiations for core typedef'ed templates.
namespace llvm {
template class PassManager<Loop>;
template class AnalysisManager<Loop>;
template class InnerAnalysisManagerProxy<LoopAnalysisManager, Function>;
template class AnalysisBase<LoopAnalysisManagerFunctionProxy>;
template class OuterAnalysisManagerProxy<FunctionAnalysisManager, Loop>;
}
