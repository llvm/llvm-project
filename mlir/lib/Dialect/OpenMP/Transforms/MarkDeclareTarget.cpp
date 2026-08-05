//===- MarkDeclareTarget.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mark functions called from explicit target code as implicitly declare target.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace omp {

#define GEN_PASS_DEF_MARKDECLARETARGETPASS
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"

} // namespace omp
} // namespace mlir

using namespace mlir;

/// Check whether the given operation is located inside of an \c omp.target.
static bool isInTargetRegion(Operation &op) {
  // TODO: Detection of callees inside of a target region might need an update
  // once reverse offloading is implemented.
  // Reverse offload target regions would then have to propagate the "host"
  // device type.
  return op.getParentOfType<omp::TargetOp>();
}

/// Add to \c callees all names of the functions called from regions owned by
/// \c op. If \c targetCallees is provided, split non-target and target uses
/// between these two output sets.
static void gatherNestedCallees(Operation &op, llvm::StringSet<> &callees,
                                llvm::StringSet<> *targetCallees = nullptr) {
  op.walk([&](CallOpInterface callOp) {
    CallInterfaceCallable callable = callOp.getCallableForCallee();
    if (auto callableSymRef = dyn_cast<SymbolRefAttr>(callable)) {
      StringRef callee = callableSymRef.getLeafReference();
      if (targetCallees && isInTargetRegion(*callOp))
        targetCallees->insert(callee);
      else
        callees.insert(callee);
    }
  });
}

/// Extract from \c arrayAttr and into \c syms the list of symbol names stored
/// in the attribute.
static void gatherSymsFromAttr(ArrayAttr arrayAttr, llvm::StringSet<> &syms) {
  if (!arrayAttr)
    return;

  for (Attribute attr : arrayAttr)
    if (auto symbolRefAttr = dyn_cast<SymbolRefAttr>(attr))
      syms.insert(symbolRefAttr.getLeafReference());
}

/// Go through all OpenMP dialect operations located in regions owned by \c op
/// looking for symbol references to \c accomp::RecipeInterface or
/// \c FunctionOpInterface operations and, based on whether they are located
/// within a nested \c omp.target region, add them to the corresponding output
/// \c StringSet.
static void gatherNestedSymbolUses(Operation &op,
                                   llvm::StringSet<> &nestedRecipeUses,
                                   llvm::StringSet<> &targetRecipeUses,
                                   llvm::StringSet<> &nestedFunctionUses,
                                   llvm::StringSet<> &targetFunctionUses) {
  op.walk([&](Operation *op) {
    bool inTarget = isInTargetRegion(*op);
    llvm::StringSet<> &recipeUses =
        inTarget ? targetRecipeUses : nestedRecipeUses;
    llvm::StringSet<> &functionUses =
        inTarget ? targetFunctionUses : nestedFunctionUses;

    // Handle each op holding clauses linked to a recipe op separately. This
    // must be kept in sync with dialect changes.
    llvm::TypeSwitch<Operation &>(*op)
        .Case([&](omp::DistributeOp op) {
          gatherSymsFromAttr(op.getPrivateSymsAttr(), recipeUses);
        })
        .Case([&](omp::LoopOp op) {
          gatherSymsFromAttr(op.getPrivateSymsAttr(), recipeUses);
          gatherSymsFromAttr(op.getReductionSymsAttr(), recipeUses);
        })
        .Case([&](omp::MapInfoOp op) {
          if (FlatSymbolRefAttr mapperAttr = op.getMapperIdAttr())
            recipeUses.insert(mapperAttr.getValue());
        })
        .Case([&](omp::ParallelOp op) {
          gatherSymsFromAttr(op.getPrivateSymsAttr(), recipeUses);
          gatherSymsFromAttr(op.getReductionSymsAttr(), recipeUses);
        })
        .Case([&](omp::ScopeOp op) {
          gatherSymsFromAttr(op.getPrivateSymsAttr(), recipeUses);
          gatherSymsFromAttr(op.getReductionSymsAttr(), recipeUses);
        })
        .Case([&](omp::SectionsOp op) {
          gatherSymsFromAttr(op.getPrivateSymsAttr(), recipeUses);
          gatherSymsFromAttr(op.getReductionSymsAttr(), recipeUses);
        })
        .Case([&](omp::SimdOp op) {
          gatherSymsFromAttr(op.getPrivateSymsAttr(), recipeUses);
          gatherSymsFromAttr(op.getReductionSymsAttr(), recipeUses);
        })
        .Case([&](omp::SingleOp op) {
          gatherSymsFromAttr(op.getPrivateSymsAttr(), recipeUses);
          // This goes directly to the called functions, as it's pointing to a
          // function, not a recipe op.
          gatherSymsFromAttr(op.getCopyprivateSymsAttr(), functionUses);
        })
        .Case([&](omp::TargetOp op) {
          // omp.private is inlined inside of the target region, hence we need
          // to add it with the target uses rather than base it on context.
          // TODO: The reverse-offload case would require adding it to
          // nestedRecipeUses.
          gatherSymsFromAttr(op.getPrivateSymsAttr(), targetRecipeUses);
          gatherSymsFromAttr(op.getInReductionSymsAttr(), recipeUses);
        })
        .Case([&](omp::TaskgroupOp op) {
          gatherSymsFromAttr(op.getTaskReductionSymsAttr(), recipeUses);
        })
        .Case([&](omp::TaskloopContextOp op) {
          gatherSymsFromAttr(op.getPrivateSymsAttr(), recipeUses);
          gatherSymsFromAttr(op.getReductionSymsAttr(), recipeUses);
          gatherSymsFromAttr(op.getInReductionSymsAttr(), recipeUses);
        })
        .Case([&](omp::TaskOp op) {
          gatherSymsFromAttr(op.getPrivateSymsAttr(), recipeUses);
          gatherSymsFromAttr(op.getInReductionSymsAttr(), recipeUses);
        })
        .Case([&](omp::TeamsOp op) {
          gatherSymsFromAttr(op.getPrivateSymsAttr(), recipeUses);
          gatherSymsFromAttr(op.getReductionSymsAttr(), recipeUses);
        })
        .Case([&](omp::WsloopOp op) {
          gatherSymsFromAttr(op.getPrivateSymsAttr(), recipeUses);
          gatherSymsFromAttr(op.getReductionSymsAttr(), recipeUses);
        });
  });
}

namespace {

// If this pass runs more than once, something like this can happen:
// - 1st run: The pass marks an external function as declare_target with
//   device_type(nohost) based on there being a single call from an omp.target.
// - Somewhere in between: New calls to that external function are added to the
//   host part of the application (e.g. it is part of a standard library).
// - 2nd run: The pass doesn't update the function after seeing it's reachable
//   by the host because it's unable to tell that the declare_target information
//   wasn't explicitly added by the user.
// TODO: This can be fixed by adding a discardable attribute only used by this
// pass or by extending the DeclareTargetInterface to also store whether it is
// implicit or explicit.
class MarkDeclareTargetPass
    : public omp::impl::MarkDeclareTargetPassBase<MarkDeclareTargetPass> {

  // This pass executes on mlir::ModuleOp, marking functions contained within
  // as implicitly declare target if they are called from within an explicitly
  // marked declare target function or a target region (TargetOp), or
  // transitively through recipe ops (e.g. omp.declare_reduction, omp.private)
  // or other function calls.
  void runOnOperation() override {
    // Illegal as an MLIR symbol name to avoid collisions. Used to gather all
    // calls from within omp.target regions as a single "function".
    constexpr const static ::llvm::StringLiteral kTargetRegionsSymName =
        "omp targets";

    ModuleOp modOp = getOperation();

    // Gather and store the set of called functions by each recipe.
    // TODO: This doesn't currently support recipe ops holding references to
    // other recipe ops.
    llvm::StringMap<llvm::StringSet<>> calls;
    for (auto recipeOp : modOp.getOps<accomp::RecipeInterface>()) {
      StringAttr recipeSymName;
      if (auto symOp = dyn_cast<SymbolOpInterface>(*recipeOp))
        recipeSymName = symOp.getNameAttr();
      else if (auto privateOp = dyn_cast<omp::PrivateClauseOp>(*recipeOp))
        recipeSymName = privateOp.getSymNameAttr();

      if (recipeSymName) {
        llvm::StringSet<> recipeCalls;
        gatherNestedCallees(*recipeOp, recipeCalls);
        calls[recipeSymName] = recipeCalls;
      }
    }

    // Gather and store the set of called functions by each function.
    for (auto funcOp : modOp.getOps<FunctionOpInterface>()) {
      llvm::StringSet<> functionCalls, targetCalls;
      gatherNestedCallees(*funcOp, functionCalls, &targetCalls);

      // Transitively include functions called from recipe op users, as if
      // inlined.
      llvm::StringSet<> recipeUses, targetRecipeUses;
      gatherNestedSymbolUses(*funcOp, recipeUses, targetRecipeUses,
                             functionCalls, targetCalls);
      for (auto &recipe : recipeUses) {
        const llvm::StringSet<> &recipeCalls = calls.at(recipe.getKey());
        functionCalls.insert_range(recipeCalls);
      }
      for (auto &recipe : targetRecipeUses) {
        const llvm::StringSet<> &recipeCalls = calls.at(recipe.getKey());
        targetCalls.insert_range(recipeCalls);
      }

      calls[funcOp.getName()] = functionCalls;
      calls[kTargetRegionsSymName].insert_range(targetCalls);
    }

    // Create worklist with all functions that are directly reachable from
    // declare_target functions or target regions.
    llvm::SmallVector<std::pair<StringRef, omp::DeclareTargetDeviceType>>
        worklist;
    for (auto funcOp : getOperation().getOps<FunctionOpInterface>()) {
      auto declareTargetOp =
          llvm::dyn_cast<omp::DeclareTargetInterface>(funcOp.getOperation());

      if (!declareTargetOp || !declareTargetOp.isDeclareTarget())
        continue;

      // Add to the worklist all called functions with the declare_target
      // information of this one, so it gets propagated.
      for (auto &callee : calls[funcOp.getName()])
        worklist.push_back(
            {callee.getKey(), declareTargetOp.getDeclareTargetDeviceType()});
    }

    // Add to the worklist all functions reached from target regions.
    for (auto &callee : calls[kTargetRegionsSymName])
      worklist.push_back(
          {callee.getKey(), omp::DeclareTargetDeviceType::nohost});

    // Process the work list storing intermediate declare target information
    // separately to avoid mixing up explicit declare_target functions with
    // implicitly propagated information.
    llvm::StringMap<omp::DeclareTargetDeviceType> intermediateInfos;
    while (!worklist.empty()) {
      std::pair<StringRef, omp::DeclareTargetDeviceType> workItem =
          worklist.pop_back_val();
      auto funcOp = modOp.lookupSymbol<FunctionOpInterface>(workItem.first);
      assert(funcOp && "a work item must point to an existing function");

      // Skip if the function is explicitly marked as declare_target or if it
      // doesn't support the interface. We only want to propagate implicit
      // declare_target information to functions for which the user hasn't
      // specified an explicit behavior.
      if (auto declareTargetOp =
              dyn_cast<omp::DeclareTargetInterface>(*funcOp)) {
        if (declareTargetOp.isDeclareTarget())
          continue;
      } else {
        continue;
      }

      omp::DeclareTargetDeviceType changedDeviceType;
      if (!intermediateInfos.contains(workItem.first)) {
        // Prevent public and external functions from being restricted to a
        // device. We don't have visibility over all their uses.
        if (funcOp.isPublic() || funcOp.isExternal())
          changedDeviceType = omp::DeclareTargetDeviceType::any;
        else
          changedDeviceType = workItem.second;

        intermediateInfos.try_emplace(workItem.first, changedDeviceType);
      } else {
        omp::DeclareTargetDeviceType &currentDeviceType =
            intermediateInfos[workItem.first];

        // Skip the update (and adding callees to the worklist) if the added
        // info doesn't change anything.
        if (currentDeviceType == omp::DeclareTargetDeviceType::any ||
            currentDeviceType == workItem.second) {
          continue;
        }

        // Update intermediate information about this function. By the previous
        // check, we know it's host + nohost = any.
        changedDeviceType = currentDeviceType =
            omp::DeclareTargetDeviceType::any;
      }

      // Add callees to the worklist to propagate the update.
      for (auto &callee : calls[workItem.first])
        worklist.push_back({callee.getKey(), changedDeviceType});
    }

    // Apply the final intermediate results to the corresponding operations.
    for (auto &[funcName, deviceType] : intermediateInfos) {
      auto declareTargetOp =
          modOp.lookupSymbol<omp::DeclareTargetInterface>(funcName);
      assert(declareTargetOp &&
             "declare_target info attached to incompatible operation");
      declareTargetOp.setDeclareTarget(deviceType,
                                       omp::DeclareTargetCaptureClause::to,
                                       /*automap=*/false);
    }
  }
};
} // namespace
