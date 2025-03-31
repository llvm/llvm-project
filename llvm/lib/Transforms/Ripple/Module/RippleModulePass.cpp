//===--------- RippleModulePass.cpp - Expand Ripple intrinsics ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands Ripple intrinsics.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Ripple/RippleModulePass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist_iterator.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Ripple/Preprocess/RippleFPExtFPTrunc.h"
#include "llvm/Transforms/Ripple/Preprocess/RippleFPExtFPTruncRevert.h"
#include "llvm/Transforms/Ripple/Preprocess/RippleSESE.h"
#include "llvm/Transforms/Ripple/Ripple.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "ripple"

PreservedAnalyses RippleModulePass::run(Module &M, ModuleAnalysisManager &MAM) {
  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

  // Request PassInstrumentation from analysis manager, will use it to run
  // instrumenting callbacks for the passes later.
  PassInstrumentation PI = MAM.getResult<PassInstrumentationAnalysis>(M);

  Ripple::ProcessingStatus PS;
  DenseSet<AssertingVH<Function>> SpecializationsPendingProcessing;
  DenseSet<AssertingVH<Function>> SpecializationsAvailable;

  FunctionPassManager FPM;
  FPM.addPass(RippleSESEPass(TM, PS, SpecializationsPendingProcessing,
                             SpecializationsAvailable));
  FPM.addPass(RipplePass(TM, PS, SpecializationsPendingProcessing,
                         SpecializationsAvailable));

  PreservedAnalyses PA = PreservedAnalyses::all();

  auto runRipplePassOnFunction = [&](Function *F) -> void {
    LLVM_DEBUG(dbgs() << "Running ripple module pass on " << F->getName()
                      << "\n");
    // Check the PassInstrumentation's BeforePass callbacks before running
    // the pass, skip its execution completely if asked to (callback returns
    // false).
    if (!PI.runBeforePass<Function>(FPM, *F))
      return;

    PreservedAnalyses PassPA = FPM.run(*F, FAM);

    // We know that the function pass couldn't have invalidated any other
    // function's analyses (that's the contract of a function pass), so
    // directly handle the function analysis manager's invalidation here.
    FAM.invalidate(*F, PassPA);

    PI.runAfterPass(FPM, F, PassPA);

    // Then intersect the preserved set so that invalidation of module
    // analyses will eventually occur when the module pass completes.
    PA.intersect(std::move(PassPA));
  };

  FunctionPassManager PreProcessPasses;
  PreProcessPasses.addPass(RippleFPExtFPTruncRevertPass());
  for (auto &F : M) {
    if (!F.isDeclaration()) {
      PreservedAnalyses PassPA = PreProcessPasses.run(F, FAM);
      FAM.invalidate(F, PassPA);
      PI.runAfterPass(PreProcessPasses, F, PassPA);
      PA.intersect(std::move(PassPA));
    }
  }

  DenseSet<AssertingVH<Function>> AlreadyProcessed;

  CallGraph CG(M);
  ReversePostOrderTraversal<const CallGraph *> RPOT(&CG);
  // Processing in reverse post-order as well as processing specialization
  // first is required:
  // - to specialize we need to clone a function un-processed by ripple (to
  // preserve the ripple constructs that may affect the return shape)
  // - specialization may request specializations through function call so we
  // need to clone non-specialized functions.

  // There is another approach that I though about later:
  // clone all the non-specialization functions being processed by ripple so
  // that we have the original at hand when needed
  // We continue until we processed all specializations and functions

  // The number of specialization requests is finite because of the broadcast
  // rule of shape propagation: you can only bcast or reduce the tensor shape
  // starting from constants
  bool SpecializationProgress;
  // When we encounter an error, we process all the functions at least once so
  // that we report as many errors as possible at once
  bool EncounteredErrors = false;
  while (true) {

    // We are progressing if we process some specialization or "normal
    // functions"
    SpecializationProgress = SpecializationsPendingProcessing.empty();
    for (auto &Spec : make_early_inc_range(SpecializationsPendingProcessing)) {

      unsigned NumberPendingBefore = SpecializationsPendingProcessing.size();
      runRipplePassOnFunction(Spec);
      bool HasNewSpecialization =
          SpecializationsPendingProcessing.size() > NumberPendingBefore;

      if (PS == Ripple::ProcessingStatus::Success) {
        Function *ToRemove = &*Spec;
        SpecializationsPendingProcessing.erase(Spec);
        ToRemove->eraseFromParent();
        SpecializationProgress = true;
      } else if (PS == Ripple::ProcessingStatus::ShapePropagationFailure ||
                 PS == Ripple::ProcessingStatus::SemanticsCheckFailure)
        EncounteredErrors = true;

      // Insertions invalidate the iterator so we need to re-enter
      if (HasNewSpecialization) {
        SpecializationProgress = true;
        break;
      }
    }
    // Break when we are not progressing (specialization cycle) but continue on
    // Errors so that we print as many errors as possible before exiting the
    // pass
    if (!SpecializationProgress && !EncounteredErrors)
      break;

    // We process all specializations first. The reason is that specializations
    // may request other specializations and we need to clone the original
    // functions before processing them.
    if (!SpecializationsPendingProcessing.empty() && !EncounteredErrors)
      continue;

    // Process functions in the module in a caller->callee order (reverse post
    // order on the call graph). This is required so that we don't process a
    // callee before being able to clone it to create a ripple specialization it
    // for the caller.
    for (auto *CGN : RPOT) {
      Function *F = CGN->getFunction();
      if (F && !F->isDeclaration() && !AlreadyProcessed.contains(F)) {
        runRipplePassOnFunction(F);
        // Either the function was processed successfully
        if (PS == Ripple::ProcessingStatus::Success) {
          AlreadyProcessed.insert(F);
        } else if (PS == Ripple::ProcessingStatus::WaitingForSpecialization) {
          // Or the function has requested a specialization that we need to
          // process first, before the other functions in RPOT
          assert(!SpecializationsPendingProcessing.empty());
          if (!EncounteredErrors)
            break;
        } else if (PS == Ripple::ProcessingStatus::ShapePropagationFailure ||
                   PS == Ripple::ProcessingStatus::SemanticsCheckFailure) {
          // Or there was an error, we continue processing to report as many
          // errors as possible
          EncounteredErrors = true;
        }
      }
    }

    // We are done processing on errors or when everyone was processed!
    if (EncounteredErrors || SpecializationsPendingProcessing.empty())
      break;
  }

  // Cycle error reporting
  if (!EncounteredErrors && !SpecializationProgress) {
    LLVM_DEBUG(dbgs() << "Checking for cycles!\n");
    for (auto F : SpecializationsPendingProcessing) {
      SmallPtrSet<Function *, 8> VisitedFunctions;
      // Check for cycles staring with the specialization
      std::function<bool(const CallGraphNode *)> visitRecursively;
      visitRecursively = [&](const CallGraphNode *CGN) -> bool {
        LLVM_DEBUG(dbgs() << "Visiting " << CGN->getFunction()->getName()
                          << "\n");
        if (VisitedFunctions.contains(CGN->getFunction()))
          return true;
        // Find the base function of this specialization
        VisitedFunctions.insert(CGN->getFunction());

        for (auto &[_, Callee] : *CGN) {
          if (Callee)
            if (visitRecursively(Callee))
              return true;
        }

        VisitedFunctions.erase(CGN->getFunction());

        return false;
      };
      Function *OriginalFunction = Ripple::specializationOriginalFunction(*F);
      assert(OriginalFunction);
      if (visitRecursively(CG[OriginalFunction])) {
        // We found a cycle
        std::string ErrorMsg;
        {
          raw_string_ostream ErrStream(ErrorMsg);
          ErrStream
              << "Ripple encountered a cycle with "
                 "the following functions and cannot reliably propagate tensor "
                 "shapes in this case:";
          // Get consistent error messages
          std::vector<Function *> FVector(VisitedFunctions.begin(),
                                          VisitedFunctions.end());
          std::sort(FVector.begin(), FVector.end(),
                    [](Function *L, Function *R) {
                      return L->getName() < R->getName();
                    });
          for (size_t Idx = 0, E = FVector.size(); Idx < E; ++Idx) {
            if (Idx)
              ErrStream << ",";
            ErrStream << " " << FVector[Idx]->getName();
          }
        }
        M.getContext().diagnose(DiagnosticInfoGeneric(
            nullptr, ErrorMsg, DiagnosticSeverity::DS_Error));
        return PreservedAnalyses::none();
      }
    }
    M.getContext().diagnose(
        DiagnosticInfoGeneric(nullptr,
                              "Ripple entered an non-progress state "
                              "due to specialization cycles",
                              DiagnosticSeverity::DS_Error));
    return PreservedAnalyses::none();
  }

  FunctionPassManager PostProcessPasses;
  PostProcessPasses.addPass(RippleFPExtFPTruncPass());
  PostProcessPasses.addPass(InstCombinePass());
  for (auto &F : M) {
    Ripple::eraseFunctionSpecializationRelatedMetadata(F);
    if (!F.isDeclaration()) {
      PreservedAnalyses PassPA = PostProcessPasses.run(F, FAM);
      FAM.invalidate(F, PassPA);
      PI.runAfterPass(PostProcessPasses, F, PassPA);
      PA.intersect(std::move(PassPA));
    }
  }

  return PA;
}
