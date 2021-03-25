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

#define DEBUG_TYPE "polly-opt-manual"

using namespace polly;
using namespace llvm;

namespace {
/// Extract an integer property from an LoopID metadata node.
static llvm::Optional<int64_t> findOptionalIntOperand(MDNode *LoopMD,
                                                      StringRef Name) {
  Metadata *AttrMD = findMetadataOperand(LoopMD, Name).getValueOr(nullptr);
  if (!AttrMD)
    return None;

  ConstantInt *IntMD = mdconst::extract_or_null<ConstantInt>(AttrMD);
  if (!IntMD)
    return None;

  return IntMD->getSExtValue();
}

/// Extract boolean property from an LoopID metadata node.
static llvm::Optional<bool> findOptionalBoolOperand(MDNode *LoopMD,
                                                    StringRef Name) {
  auto MD = findOptionMDForLoopID(LoopMD, Name);
  if (!MD)
    return None;

  switch (MD->getNumOperands()) {
  case 1:
    // When the value is absent it is interpreted as 'attribute set'.
    return true;
  case 2:
    ConstantInt *IntMD =
        mdconst::extract_or_null<ConstantInt>(MD->getOperand(1).get());
    return IntMD->getZExtValue() != 0;
  }
  llvm_unreachable("unexpected number of options");
}

/// Apply full or partial unrolling.
static isl::schedule applyLoopUnroll(MDNode *LoopMD,
                                     isl::schedule_node BandToUnroll) {
  assert(BandToUnroll);
  // TODO: Isl's codegen also supports unrolling by isl_ast_build via
  // isl_schedule_node_band_set_ast_build_options({ unroll[x] }) which would be
  // more efficient because the content duplication is delayed. However, the
  // unrolled loop could be input of another loop transformation which expects
  // the explicit schedule nodes. That is, we would need this explicit expansion
  // anyway and using the ISL codegen option is a compile-time optimization.
  int64_t Factor =
      findOptionalIntOperand(LoopMD, "llvm.loop.unroll.count").getValueOr(0);
  bool Full = findOptionalBoolOperand(LoopMD, "llvm.loop.unroll.full")
                  .getValueOr(false);
  assert((!Full || !(Factor > 0)) &&
         "Cannot unroll fully and partially at the same time");

  if (Full)
    return applyFullUnroll(BandToUnroll);

  if (Factor > 0)
    return applyPartialUnroll(BandToUnroll, Factor);

  llvm_unreachable("Negative unroll factor");
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
    if (Result)
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

      if (AttrName == "llvm.loop.unroll.enable") {
        // TODO: Handle disabling like llvm::hasUnrollTransformation().
        Result = applyLoopUnroll(LoopMD, Band);
      } else {
        // not a loop transformation; look for next property
        continue;
      }

      assert(Result && "expecting applied transformation");
      return;
    }
  }

  void visitNode(const isl::schedule_node &Other) {
    if (Result)
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
    if (!Result) {
      // No (more) transformation has been found.
      break;
    }

    // Use transformed schedule and look for more transformations.
    Sched = Result;
  }

  return Sched;
}
