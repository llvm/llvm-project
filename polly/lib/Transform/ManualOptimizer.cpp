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
#include "polly/ScheduleTreeTransform.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#define DEBUG_TYPE "polly-opt-manual"

using namespace polly;
using namespace llvm;

namespace {
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

  // Set after a transformation is applied. Recursive search must be aborted
  // once this happens to ensure that any new followup transformation is
  // transformed in innermost-first order.
  isl::schedule Result;

public:
  static isl::schedule applyOneTransformation(const isl::schedule &Sched) {
    SearchTransformVisitor Transformer;
    Transformer.visit(Sched);
    return Transformer.Result;
  }

  void visitBand(const isl::schedule_node &Band) {
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
      }

      // not a loop transformation; look for next property
      continue;
    }
  }

  void visitNode(const isl::schedule_node &Other) {
    if (!Result.is_null())
      return;
    getBase().visitNode(Other);
  }
};

} // namespace

isl::schedule polly::applyManualTransformations(Scop *S, isl::schedule Sched) {
  // Search the loop nest for transformations until fixpoint.
  while (true) {
    isl::schedule Result =
        SearchTransformVisitor::applyOneTransformation(Sched);
    if (Result.is_null()) {
      // No (more) transformation has been found.
      break;
    }

    // Use transformed schedule and look for more transformations.
    Sched = Result;
  }

  return Sched;
}
