//===- ACCImplicitDeclare.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass applies implicit `acc declare` actions to global variables
// referenced in OpenACC compute regions and routine functions.
//
// Overview:
// ---------
// Global references in an acc regions (for globals not marked with `acc
// declare` by the user) can be handled in one of two ways:
// - Mapped through data clauses
// - Implicitly marked as `acc declare` (this pass)
//
// Thus, the OpenACC specification focuses solely on implicit data mapping rules
// whose implementation is captured in `ACCImplicitData` pass.
//
// However, it is both advantageous and required for certain cases to
// use implicit `acc declare` instead:
// - Any functions that are implicitly marked as `acc routine` through
//   `ACCImplicitRoutine` may reference globals. Since data mapping
//   is only possible for compute regions, such globals can only be
//   made available on device through `acc declare`.
// - Compiler can generate and use globals for cases needed in IR
//   representation such as type descriptors or various names needed for
//   runtime calls and error reporting - such cases often are introduced
//   after a frontend semantic checking is done since it is related to
//   implementation detail. Thus, such compiler generated globals would
//   not have been visible for a user to mark with `acc declare`.
// - Constant globals such as filename strings or data initialization values
//   are values that do not get mutated but are still needed for appropriate
//   runtime execution. If a kernel is launched 1000 times, it is not a
//   good idea to map such a global 1000 times. Therefore, such globals
//   benefit from being marked with `acc declare`.
//
// This pass automatically
// marks global variables with the `acc.declare` attribute when they are
// referenced in OpenACC compute constructs or routine functions and meet
// the criteria noted above, ensuring
// they are properly handled for device execution.
//
// The pass performs two main optimizations:
//
// 1. Hoisting: For non-constant globals referenced in compute regions, the
//    pass hoists the address-of operation out of the region when possible,
//    allowing them to be implicitly mapped through normal data clause
//    mechanisms rather than requiring declare marking.
//
// 2. Declaration: For globals that must be available on the device (constants,
//    globals in routines, globals in recipe operations), the pass adds the
//    `acc.declare` attribute with the copyin data clause.
//
// Requirements:
// -------------
// To use this pass in a pipeline, the following requirements must be met:
//
// 1. Operation Interface Implementation: Operations that compute addresses
//    of global variables must implement the `acc::AddressOfGlobalOpInterface`
//    and those that represent globals must implement the
//    `acc::GlobalOpInterface`. Additionally, any operations that indirectly
//    access globals must implement the `acc::IndirectGlobalAccessOpInterface`.
//
// 2. Analysis Registration (Optional): If custom behavior is needed for
//    determining if a symbol use is valid within GPU regions, the dialect
//    should pre-register the `acc::OpenACCSupport` analysis.
//
// Examples:
// ---------
//
// Example 1: Non-constant global in compute region (hoisted)
//
// Before:
//   memref.global @g_scalar : memref<f32> = dense<0.0>
//   func.func @test() {
//     acc.serial {
//       %addr = memref.get_global @g_scalar : memref<f32>
//       %val = memref.load %addr[] : memref<f32>
//       acc.yield
//     }
//   }
//
// After:
//   memref.global @g_scalar : memref<f32> = dense<0.0>
//   func.func @test() {
//     %addr = memref.get_global @g_scalar : memref<f32>
//     acc.serial {
//       %val = memref.load %addr[] : memref<f32>
//       acc.yield
//     }
//   }
//
// Example 2: Constant global in compute region (declared)
//
// Before:
//   memref.global constant @g_const : memref<f32> = dense<1.0>
//   func.func @test() {
//     acc.serial {
//       %addr = memref.get_global @g_const : memref<f32>
//       %val = memref.load %addr[] : memref<f32>
//       acc.yield
//     }
//   }
//
// After:
//   memref.global constant @g_const : memref<f32> = dense<1.0>
//       {acc.declare = #acc.declare<dataClause = acc_copyin>}
//   func.func @test() {
//     acc.serial {
//       %addr = memref.get_global @g_const : memref<f32>
//       %val = memref.load %addr[] : memref<f32>
//       acc.yield
//     }
//   }
//
// Example 3: Global in acc routine (declared)
//
// Before:
//   memref.global @g_data : memref<f32> = dense<0.0>
//   acc.routine @routine_0 func(@device_func)
//   func.func @device_func() attributes {acc.routine_info = ...} {
//     %addr = memref.get_global @g_data : memref<f32>
//     %val = memref.load %addr[] : memref<f32>
//   }
//
// After:
//   memref.global @g_data : memref<f32> = dense<0.0>
//       {acc.declare = #acc.declare<dataClause = acc_copyin>}
//   acc.routine @routine_0 func(@device_func)
//   func.func @device_func() attributes {acc.routine_info = ...} {
//     %addr = memref.get_global @g_data : memref<f32>
//     %val = memref.load %addr[] : memref<f32>
//   }
//
// Example 4: Global in private recipe (declared if recipe is used)
//
// Before:
//   memref.global @g_init : memref<f32> = dense<0.0>
//   acc.private.recipe @priv_recipe : memref<f32> init {
//   ^bb0(%arg0: memref<f32>):
//     %alloc = memref.alloc() : memref<f32>
//     %global = memref.get_global @g_init : memref<f32>
//     %val = memref.load %global[] : memref<f32>
//     memref.store %val, %alloc[] : memref<f32>
//     acc.yield %alloc : memref<f32>
//   } destroy { ... }
//   func.func @test() {
//     %var = memref.alloc() : memref<f32>
//     %priv = acc.private varPtr(%var : memref<f32>)
//               recipe(@priv_recipe) -> memref<f32>
//     acc.parallel private(%priv : memref<f32>) { ... }
//   }
//
// After:
//   memref.global @g_init : memref<f32> = dense<0.0>
//       {acc.declare = #acc.declare<dataClause = acc_copyin>}
//   acc.private.recipe @priv_recipe : memref<f32> init {
//   ^bb0(%arg0: memref<f32>):
//     %alloc = memref.alloc() : memref<f32>
//     %global = memref.get_global @g_init : memref<f32>
//     %val = memref.load %global[] : memref<f32>
//     memref.store %val, %alloc[] : memref<f32>
//     acc.yield %alloc : memref<f32>
//   } destroy { ... }
//   func.func @test() {
//     %var = memref.alloc() : memref<f32>
//     %priv = acc.private varPtr(%var : memref<f32>)
//               recipe(@priv_recipe) -> memref<f32>
//     acc.parallel private(%priv : memref<f32>) { ... }
//   }
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCIMPLICITDECLARE
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-implicit-declare"

using namespace mlir;

namespace {

using GlobalOpSetT = llvm::SmallSetVector<Operation *, 16>;

/// Checks whether a use of the requested `globalOp` should be considered
/// for hoisting out of acc region due to avoid `acc declare`ing something
/// that instead should be implicitly mapped.
static bool isGlobalUseCandidateForHoisting(Operation *globalOp,
                                            Operation *user,
                                            SymbolRefAttr symbol,
                                            acc::OpenACCSupport &accSupport) {
  // This symbol is valid in GPU region. This means semantics
  // would change if moved to host - therefore it is not a candidate.
  if (accSupport.isValidSymbolUse(user, symbol))
    return false;

  bool isConstant = false;
  bool isFunction = false;

  if (auto globalVarOp = dyn_cast<acc::GlobalVariableOpInterface>(globalOp))
    isConstant = globalVarOp.isConstant();

  if (isa<FunctionOpInterface>(globalOp))
    isFunction = true;

  // Constants should be kept in device code to ensure they are duplicated.
  // Function references should be kept in device code to ensure their device
  // addresses are computed. Everything else should be hoisted since we already
  // proved they are not valid symbols in GPU region.
  return !isConstant && !isFunction;
}

/// Checks whether it is valid to use acc.declare marking on the global.
bool isValidForAccDeclare(Operation *globalOp) {
  // For functions - we use acc.routine marking instead.
  return !isa<FunctionOpInterface>(globalOp);
}

/// Checks whether a recipe operation has meaningful use of its symbol that
/// justifies processing its regions for global references. Returns false if:
/// 1. The recipe has no symbol uses at all, or
/// 2. The only symbol use is the recipe's own symbol definition
template <typename RecipeOpT>
static bool hasRelevantRecipeUse(RecipeOpT &recipeOp, ModuleOp &mod) {
  std::optional<SymbolTable::UseRange> symbolUses = recipeOp.getSymbolUses(mod);

  // No recipe symbol uses.
  if (!symbolUses.has_value() || symbolUses->empty())
    return false;

  // If more than one use, assume it's used.
  auto begin = symbolUses->begin();
  auto end = symbolUses->end();
  if (begin != end && std::next(begin) != end)
    return true;

  // If single use, check if the use is the recipe itself.
  const SymbolTable::SymbolUse &use = *symbolUses->begin();
  return use.getUser() != recipeOp.getOperation();
}

// Hoists addr_of operations for non-constant globals out of OpenACC regions.
// This way - they are implicitly mapped instead of being considered for
// implicit declare.
template <typename AccConstructT>
static void hoistNonConstantDirectUses(AccConstructT accOp,
                                       acc::OpenACCSupport &accSupport) {
  accOp.walk([&](acc::AddressOfGlobalOpInterface addrOfOp) {
    SymbolRefAttr symRef = addrOfOp.getSymbol();
    if (symRef) {
      Operation *globalOp =
          SymbolTable::lookupNearestSymbolFrom(addrOfOp, symRef);
      if (isGlobalUseCandidateForHoisting(globalOp, addrOfOp, symRef,
                                          accSupport)) {
        addrOfOp->moveBefore(accOp);
        LLVM_DEBUG(
            llvm::dbgs() << "Hoisted:\n\t" << addrOfOp << "\n\tfrom:\n\t";
            accOp->print(llvm::dbgs(),
                         OpPrintingFlags{}.skipRegions().enableDebugInfo());
            llvm::dbgs() << "\n");
      }
    }
  });
}

// Collects the globals referenced in a device region
static void collectGlobalsFromDeviceRegion(Region &region,
                                           GlobalOpSetT &globals,
                                           acc::OpenACCSupport &accSupport,
                                           SymbolTable &symTab) {
  region.walk([&](Operation *op) {
    // 1) Only consider relevant operations which use symbols
    auto addrOfOp = dyn_cast<acc::AddressOfGlobalOpInterface>(op);
    if (addrOfOp) {
      SymbolRefAttr symRef = addrOfOp.getSymbol();
      // 2) Found an operation which uses the symbol. Next determine if it
      //    is a candidate for `acc declare`. Some of the criteria considered
      //    is whether this symbol is not already a device one (either because
      //    acc declare is already used or this is a CUF global).
      Operation *globalOp = nullptr;
      bool isCandidate = !accSupport.isValidSymbolUse(op, symRef, &globalOp);
      // 3) Add the candidate to the set of globals to be `acc declare`d.
      if (isCandidate && globalOp && isValidForAccDeclare(globalOp))
        globals.insert(globalOp);
    } else if (auto indirectAccessOp =
                   dyn_cast<acc::IndirectGlobalAccessOpInterface>(op)) {
      // Process operations that indirectly access globals
      llvm::SmallVector<SymbolRefAttr> symbols;
      indirectAccessOp.getReferencedSymbols(symbols, &symTab);
      for (SymbolRefAttr symRef : symbols)
        if (Operation *globalOp = symTab.lookup(symRef.getLeafReference()))
          if (isValidForAccDeclare(globalOp))
            globals.insert(globalOp);
    }
  });
}

// Adds the declare attribute to the operation `op`.
static void addDeclareAttr(MLIRContext *context, Operation *op,
                           acc::DataClause clause) {
  op->setAttr(acc::getDeclareAttrName(),
              acc::DeclareAttr::get(context,
                                    acc::DataClauseAttr::get(context, clause)));
}

// This pass applies implicit declare actions for globals referenced in
// OpenACC compute and routine regions.
class ACCImplicitDeclare
    : public acc::impl::ACCImplicitDeclareBase<ACCImplicitDeclare> {
public:
  using ACCImplicitDeclareBase<ACCImplicitDeclare>::ACCImplicitDeclareBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *context = &getContext();
    acc::OpenACCSupport &accSupport = getAnalysis<acc::OpenACCSupport>();

    // 1) Start off by hoisting any AddressOf operations out of acc region
    // for any cases we do not want to `acc declare`. This is because we can
    // rely on implicit data mapping in majority of cases without uselessly
    // polluting the device globals.
    mod.walk([&](Operation *op) {
      TypeSwitch<Operation *, void>(op)
          .Case<ACC_COMPUTE_CONSTRUCT_OPS, acc::KernelEnvironmentOp>(
              [&](auto accOp) {
                hoistNonConstantDirectUses(accOp, accSupport);
              });
    });

    // 2) Collect global symbols which need to be `acc declare`d. Do it for
    // compute regions, acc routine, and existing globals with the declare
    // attribute.
    SymbolTable symTab(mod);
    GlobalOpSetT globalsToAccDeclare;
    mod.walk([&](Operation *op) {
      TypeSwitch<Operation *, void>(op)
          .Case<ACC_COMPUTE_CONSTRUCT_OPS, acc::KernelEnvironmentOp>(
              [&](auto accOp) {
                collectGlobalsFromDeviceRegion(
                    accOp.getRegion(), globalsToAccDeclare, accSupport, symTab);
              })
          .Case([&](FunctionOpInterface func) {
            if ((acc::isAccRoutine(func) ||
                 acc::isSpecializedAccRoutine(func)) &&
                !func.isExternal())
              collectGlobalsFromDeviceRegion(func.getFunctionBody(),
                                             globalsToAccDeclare, accSupport,
                                             symTab);
          })
          .Case([&](acc::GlobalVariableOpInterface globalVarOp) {
            if (globalVarOp->getAttr(acc::getDeclareAttrName()))
              if (Region *initRegion = globalVarOp.getInitRegion())
                collectGlobalsFromDeviceRegion(*initRegion, globalsToAccDeclare,
                                               accSupport, symTab);
          })
          .Case([&](acc::PrivateRecipeOp privateRecipe) {
            if (hasRelevantRecipeUse(privateRecipe, mod)) {
              collectGlobalsFromDeviceRegion(privateRecipe.getInitRegion(),
                                             globalsToAccDeclare, accSupport,
                                             symTab);
              collectGlobalsFromDeviceRegion(privateRecipe.getDestroyRegion(),
                                             globalsToAccDeclare, accSupport,
                                             symTab);
            }
          })
          .Case([&](acc::FirstprivateRecipeOp firstprivateRecipe) {
            if (hasRelevantRecipeUse(firstprivateRecipe, mod)) {
              collectGlobalsFromDeviceRegion(firstprivateRecipe.getInitRegion(),
                                             globalsToAccDeclare, accSupport,
                                             symTab);
              collectGlobalsFromDeviceRegion(
                  firstprivateRecipe.getDestroyRegion(), globalsToAccDeclare,
                  accSupport, symTab);
              collectGlobalsFromDeviceRegion(firstprivateRecipe.getCopyRegion(),
                                             globalsToAccDeclare, accSupport,
                                             symTab);
            }
          })
          .Case([&](acc::ReductionRecipeOp reductionRecipe) {
            if (hasRelevantRecipeUse(reductionRecipe, mod)) {
              collectGlobalsFromDeviceRegion(reductionRecipe.getInitRegion(),
                                             globalsToAccDeclare, accSupport,
                                             symTab);
              collectGlobalsFromDeviceRegion(
                  reductionRecipe.getCombinerRegion(), globalsToAccDeclare,
                  accSupport, symTab);
            }
          });
    });

    // 3) Finally, generate the appropriate declare actions needed to ensure
    // this is considered for device global.
    for (Operation *globalOp : globalsToAccDeclare) {
      LLVM_DEBUG(
          llvm::dbgs() << "Global is being `acc declare copyin`d: ";
          globalOp->print(llvm::dbgs(),
                          OpPrintingFlags{}.skipRegions().enableDebugInfo());
          llvm::dbgs() << "\n");

      // Mark it as declare copyin.
      addDeclareAttr(context, globalOp, acc::DataClause::acc_copyin);

      // TODO: May need to create the global constructor which does the mapping
      // action. It is not yet clear if this is needed yet (since the globals
      // might just end up in the GPU image without requiring mapping via
      // runtime).
    }
  }
};

} // namespace
