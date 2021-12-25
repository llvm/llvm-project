//===------ ManualOptimizer.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Handle pragma/metadata-directed transformations.
//
//===----------------------------------------------------------------------===//

#include "polly/ManualOptimizer.h"
#include "polly/DependenceInfo.h"
#include "polly/Options.h"
#include "polly/ScheduleTreeTransform.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#define DEBUG_TYPE "polly-opt-manual"

using namespace polly;
using namespace llvm;

namespace {

static cl::opt<bool> IgnoreDepcheck(
    "polly-pragma-ignore-depcheck",
    cl::desc("Skip the dependency check for pragma-based transformations"),
    cl::init(false), cl::ZeroOrMore, cl::cat(PollyCategory));

/// Same as llvm::hasUnrollTransformation(), but takes a LoopID as argument
/// instead of a Loop.
static TransformationMode hasUnrollTransformation(MDNode *LoopID) {
  if (getBooleanLoopAttribute(LoopID, "llvm.loop.unroll.disable"))
    return TM_SuppressedByUser;

  Optional<int> Count =
      getOptionalIntLoopAttribute(LoopID, "llvm.loop.unroll.count");
  if (Count.hasValue())
    return Count.getValue() == 1 ? TM_SuppressedByUser : TM_ForcedByUser;

  if (getBooleanLoopAttribute(LoopID, "llvm.loop.unroll.enable"))
    return TM_ForcedByUser;

  if (getBooleanLoopAttribute(LoopID, "llvm.loop.unroll.full"))
    return TM_ForcedByUser;

  if (hasDisableAllTransformsHint(LoopID))
    return TM_Disable;

  return TM_Unspecified;
}

// Return the first DebugLoc in the list.
static DebugLoc findFirstDebugLoc(MDNode *MD) {
  if (MD) {
    for (const MDOperand &X : drop_begin(MD->operands(), 1)) {
      Metadata *A = X.get();
      if (!isa<DILocation>(A))
        continue;
      return cast<DILocation>(A);
    }
  }

  return {};
}

static DebugLoc findTransformationDebugLoc(MDNode *LoopMD, StringRef Name) {
  // First find dedicated transformation location
  // (such as the location of #pragma clang loop)
  MDNode *MD = findOptionMDForLoopID(LoopMD, Name);
  if (DebugLoc K = findFirstDebugLoc(MD))
    return K;

  // Otherwise, fall back to the location of the loop itself
  return findFirstDebugLoc(LoopMD);
}

/// Apply full or partial unrolling.
static isl::schedule applyLoopUnroll(MDNode *LoopMD,
                                     isl::schedule_node BandToUnroll) {
  TransformationMode UnrollMode = ::hasUnrollTransformation(LoopMD);
  if (UnrollMode & TM_Disable)
    return {};

  assert(!BandToUnroll.is_null());
  // TODO: Isl's codegen also supports unrolling by isl_ast_build via
  // isl_schedule_node_band_set_ast_build_options({ unroll[x] }) which would be
  // more efficient because the content duplication is delayed. However, the
  // unrolled loop could be input of another loop transformation which expects
  // the explicit schedule nodes. That is, we would need this explicit expansion
  // anyway and using the ISL codegen option is a compile-time optimization.
  int64_t Factor = getOptionalIntLoopAttribute(LoopMD, "llvm.loop.unroll.count")
                       .getValueOr(0);
  bool Full = getBooleanLoopAttribute(LoopMD, "llvm.loop.unroll.full");
  assert((!Full || !(Factor > 0)) &&
         "Cannot unroll fully and partially at the same time");

  if (Full)
    return applyFullUnroll(BandToUnroll);

  if (Factor > 0)
    return applyPartialUnroll(BandToUnroll, Factor);

  // For heuristic unrolling, fall back to LLVM's LoopUnroll pass.
  return {};
}

static isl::schedule applyLoopFission(MDNode *LoopMD,
                                      isl::schedule_node BandToFission) {
  // TODO: Make it possible to selectively fission substatements.
  // TODO: Apply followup loop properties.
  // TODO: Instead of fission every statement, find the maximum set that does
  // not cause a dependency violation.
  return applyMaxFission(BandToFission);
}

// Return the properties from a LoopID. Scalar properties are ignored.
static auto getLoopMDProps(MDNode *LoopMD) {
  return map_range(
      make_filter_range(
          drop_begin(LoopMD->operands(), 1),
          [](const MDOperand &MDOp) { return isa<MDNode>(MDOp.get()); }),
      [](const MDOperand &MDOp) { return cast<MDNode>(MDOp.get()); });
}

/// Recursively visit all nodes in a schedule, loop for loop-transformations
/// metadata and apply the first encountered.
class SearchTransformVisitor
    : public RecursiveScheduleTreeVisitor<SearchTransformVisitor> {
private:
  using BaseTy = RecursiveScheduleTreeVisitor<SearchTransformVisitor>;
  BaseTy &getBase() { return *this; }
  const BaseTy &getBase() const { return *this; }

  polly::Scop *S;
  const Dependences *D;
  OptimizationRemarkEmitter *ORE;

  // Set after a transformation is applied. Recursive search must be aborted
  // once this happens to ensure that any new followup transformation is
  // transformed in innermost-first order.
  isl::schedule Result;

  /// Check wether a schedule after a  transformation is legal. Return the old
  /// schedule without the transformation.
  isl::schedule
  checkDependencyViolation(llvm::MDNode *LoopMD, llvm::Value *CodeRegion,
                           const isl::schedule_node &OrigBand,
                           StringRef DebugLocAttr, StringRef TransPrefix,
                           StringRef RemarkName, StringRef TransformationName) {
    if (D->isValidSchedule(*S, Result))
      return Result;

    LLVMContext &Ctx = LoopMD->getContext();
    LLVM_DEBUG(dbgs() << "Dependency violation detected\n");

    DebugLoc TransformLoc = findTransformationDebugLoc(LoopMD, DebugLocAttr);

    if (IgnoreDepcheck) {
      LLVM_DEBUG(dbgs() << "Still accepting transformation due to "
                           "-polly-pragma-ignore-depcheck\n");
      if (ORE) {
        ORE->emit(
            OptimizationRemark(DEBUG_TYPE, RemarkName, TransformLoc, CodeRegion)
            << (Twine("Could not verify dependencies for ") +
                TransformationName +
                "; still applying because of -polly-pragma-ignore-depcheck")
                   .str());
      }
      return Result;
    }

    LLVM_DEBUG(dbgs() << "Rolling back transformation\n");

    if (ORE) {
      ORE->emit(DiagnosticInfoOptimizationFailure(DEBUG_TYPE, RemarkName,
                                                  TransformLoc, CodeRegion)
                << (Twine("not applying ") + TransformationName +
                    ": cannot ensure semantic equivalence due to possible "
                    "dependency violations")
                       .str());
    }

    // If illegal, revert and remove the transformation to not risk re-trying
    // indefintely.
    MDNode *NewLoopMD =
        makePostTransformationMetadata(Ctx, LoopMD, {TransPrefix}, {});
    BandAttr *Attr = getBandAttr(OrigBand);
    Attr->Metadata = NewLoopMD;

    // Roll back old schedule.
    return OrigBand.get_schedule();
  }

public:
  SearchTransformVisitor(polly::Scop *S, const Dependences *D,
                         OptimizationRemarkEmitter *ORE)
      : S(S), D(D), ORE(ORE) {}

  static isl::schedule applyOneTransformation(polly::Scop *S,
                                              const Dependences *D,
                                              OptimizationRemarkEmitter *ORE,
                                              const isl::schedule &Sched) {
    SearchTransformVisitor Transformer(S, D, ORE);
    Transformer.visit(Sched);
    return Transformer.Result;
  }

  void visitBand(isl::schedule_node_band Band) {
    // Transform inner loops first (depth-first search).
    getBase().visitBand(Band);
    if (!Result.is_null())
      return;

    // Since it is (currently) not possible to have a BandAttr marker that is
    // specific to each loop in a band, we only support single-loop bands.
    if (isl_schedule_node_band_n_member(Band.get()) != 1)
      return;

    BandAttr *Attr = getBandAttr(Band);
    if (!Attr) {
      // Band has no attribute.
      return;
    }

    // CodeRegion used but ORE to determine code hotness.
    // TODO: Works only for original loop; for transformed loops, should track
    // where the loop's body code comes from.
    Loop *Loop = Attr->OriginalLoop;
    Value *CodeRegion = nullptr;
    if (Loop)
      CodeRegion = Loop->getHeader();

    MDNode *LoopMD = Attr->Metadata;
    if (!LoopMD)
      return;

    // Iterate over loop properties to find the first transformation.
    // FIXME: If there are more than one transformation in the LoopMD (making
    // the order of transformations ambiguous), all others are silently ignored.
    for (MDNode *MD : getLoopMDProps(LoopMD)) {
      auto *NameMD = dyn_cast<MDString>(MD->getOperand(0).get());
      if (!NameMD)
        continue;
      StringRef AttrName = NameMD->getString();

      // Honor transformation order; transform the first transformation in the
      // list first.
      if (AttrName == "llvm.loop.unroll.enable" ||
          AttrName == "llvm.loop.unroll.count" ||
          AttrName == "llvm.loop.unroll.full") {
        Result = applyLoopUnroll(LoopMD, Band);
        if (!Result.is_null())
          return;
      } else if (AttrName == "llvm.loop.distribute.enable") {
        Result = applyLoopFission(LoopMD, Band);
        if (!Result.is_null())
          Result = checkDependencyViolation(
              LoopMD, CodeRegion, Band, "llvm.loop.distribute.loc",
              "llvm.loop.distribute.", "FailedRequestedFission",
              "loop fission/distribution");
        if (!Result.is_null())
          return;
      }

      // not a loop transformation; look for next property
    }
  }

  void visitNode(isl::schedule_node Other) {
    if (!Result.is_null())
      return;
    getBase().visitNode(Other);
  }
};

} // namespace

isl::schedule
polly::applyManualTransformations(Scop *S, isl::schedule Sched,
                                  const Dependences &D,
                                  OptimizationRemarkEmitter *ORE) {
  // Search the loop nest for transformations until fixpoint.
  while (true) {
    isl::schedule Result =
        SearchTransformVisitor::applyOneTransformation(S, &D, ORE, Sched);
    if (Result.is_null()) {
      // No (more) transformation has been found.
      break;
    }

    // Use transformed schedule and look for more transformations.
    Sched = Result;
  }

  return Sched;
}
