//===- InlineCost.cpp - Cost analysis for inliner ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements heuristics for inlining decisions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_INLINECOST_H
#define LLVM_TRANSFORMS_UTILS_INLINECOST_H

#include "llvm/ADT/SmallPtrSet.h"
#include <cassert>
#include <map>
#include <vector>

namespace llvm {

  class Value;
  class Function;
  class CallSite;

  /// InlineCost - Represent the cost of inlining a function. This
  /// supports special values for functions which should "always" or
  /// "never" be inlined. Otherwise, the cost represents a unitless
  /// amount; smaller values increase the likelyhood of the function
  /// being inlined.
  class InlineCost {
    enum Kind {
      Value,
      Always,
      Never
    };

    int Cost : 30;
    unsigned Type :  2;

    InlineCost(int C, int T) : Cost(C), Type(T) {
      assert(Cost == C && "Cost exceeds InlineCost precision");
    }
  public:
    static InlineCost get(int Cost) { return InlineCost(Cost, Value); }
    static InlineCost getAlways() { return InlineCost(0, Always); }
    static InlineCost getNever() { return InlineCost(0, Never); } 

    bool isVariable() const { return Type == Value; }
    bool isAlways() const { return Type == Always; }
    bool isNever() const { return Type == Never; }

    /// getValue() - Return a "variable" inline cost's amount. It is
    /// an error to call this on an "always" or "never" InlineCost.
    int getValue() const { 
      assert(Type == Value && "Invalid access of InlineCost");
      return Cost;
    }
  };
  
  /// InlineCostAnalyzer - Cost analyzer used by inliner.
  class InlineCostAnalyzer {
    struct ArgInfo {
    public:
      unsigned ConstantWeight;
      unsigned AllocaWeight;
      
      ArgInfo(unsigned CWeight, unsigned AWeight)
        : ConstantWeight(CWeight), AllocaWeight(AWeight) {}
    };
    
    // FunctionInfo - For each function, calculate the size of it in blocks and
    // instructions.
    struct FunctionInfo {
      /// NeverInline - True if this callee should never be inlined into a
      /// caller.
      bool NeverInline;
      
      /// usesDynamicAlloca - True if this function calls alloca (in the C sense).
      bool usesDynamicAlloca;

      /// NumInsts, NumBlocks - Keep track of how large each function is, which
      /// is used to estimate the code size cost of inlining it.
      unsigned NumInsts, NumBlocks;

      /// NumVectorInsts - Keep track of how many instructions produce vector
      /// values.  The inliner is being more aggressive with inlining vector
      /// kernels.
      unsigned NumVectorInsts;
      
      /// ArgumentWeights - Each formal argument of the function is inspected to
      /// see if it is used in any contexts where making it a constant or alloca
      /// would reduce the code size.  If so, we add some value to the argument
      /// entry here.
      std::vector<ArgInfo> ArgumentWeights;
      
      FunctionInfo() : NeverInline(false), usesDynamicAlloca(false), NumInsts(0),
                       NumBlocks(0), NumVectorInsts(0) {}
      
      /// analyzeFunction - Fill in the current structure with information
      /// gleaned from the specified function.
      void analyzeFunction(Function *F);

      /// CountCodeReductionForConstant - Figure out an approximation for how
      /// many instructions will be constant folded if the specified value is
      /// constant.
      unsigned CountCodeReductionForConstant(Value *V);
      
      /// CountCodeReductionForAlloca - Figure out an approximation of how much
      /// smaller the function will be if it is inlined into a context where an
      /// argument becomes an alloca.
      ///
      unsigned CountCodeReductionForAlloca(Value *V);
    };

    std::map<const Function *, FunctionInfo> CachedFunctionInfo;

  public:

    /// getInlineCost - The heuristic used to determine if we should inline the
    /// function call or not.
    ///
    InlineCost getInlineCost(CallSite CS,
                             SmallPtrSet<const Function *, 16> &NeverInline);

    /// getInlineFudgeFactor - Return a > 1.0 factor if the inliner should use a
    /// higher threshold to determine if the function call should be inlined.
    float getInlineFudgeFactor(CallSite CS);

    /// resetCachedFunctionInfo - erase any cached cost info for this function.
    void resetCachedCostInfo(Function* Caller) {
      CachedFunctionInfo[Caller].NumBlocks = 0;
    }
  };
}

#endif
