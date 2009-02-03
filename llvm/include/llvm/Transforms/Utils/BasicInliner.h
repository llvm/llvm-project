//===- BasicInliner.h - Basic function level inliner ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a simple function based inliner that does not use
// call graph information. 
//
//===----------------------------------------------------------------------===//

#ifndef BASICINLINER_H
#define BASICINLINER_H

#include "llvm/Transforms/Utils/InlineCost.h"

namespace llvm {

  class Function;
  class TargetData;
  struct BasicInlinerImpl;

  /// BasicInliner - BasicInliner provides function level inlining interface.
  /// Clients provide list of functions which are inline without using
  /// module level call graph information. Note that the BasicInliner is
  /// free to delete a function if it is inlined into all call sites.
  class BasicInliner {
  public:
    
    explicit BasicInliner(TargetData *T = NULL);
    ~BasicInliner();

    /// addFunction - Add function into the list of functions to process.
    /// All functions must be inserted using this interface before invoking
    /// inlineFunctions().
    void addFunction(Function *F);

    /// neverInlineFunction - Sometimes a function is never to be inlined 
    /// because of one or other reason. 
    void neverInlineFunction(Function *F);

    /// inlineFuctions - Walk all call sites in all functions supplied by
    /// client. Inline as many call sites as possible. Delete completely
    /// inlined functions.
    void inlineFunctions();

  private:
    BasicInlinerImpl *Impl;
  };
}

#endif
