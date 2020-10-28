//===- NormalizeMemRefs.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an interprocedural pass to normalize memrefs to have
// identity layout maps.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#define DEBUG_TYPE "normalize-memrefs"

using namespace mlir;

namespace {

/// All memrefs passed across functions with non-trivial layout maps are
/// converted to ones with trivial identity layout ones.

// Input :-
// #tile = affine_map<(i) -> (i floordiv 4, i mod 4)>
// func @matmul(%A: memref<16xf64, #tile>, %B: index, %C: memref<16xf64>) ->
// (memref<16xf64, #tile>) {
//   affine.for %arg3 = 0 to 16 {
//         %a = affine.load %A[%arg3] : memref<16xf64, #tile>
//         %p = mulf %a, %a : f64
//         affine.store %p, %A[%arg3] : memref<16xf64, #tile>
//   }
//   %c = alloc() : memref<16xf64, #tile>
//   %d = affine.load %c[0] : memref<16xf64, #tile>
//   return %A: memref<16xf64, #tile>
// }

// Output :-
//   func @matmul(%arg0: memref<4x4xf64>, %arg1: index, %arg2: memref<16xf64>)
//   -> memref<4x4xf64> {
//     affine.for %arg3 = 0 to 16 {
//       %2 = affine.load %arg0[%arg3 floordiv 4, %arg3 mod 4] : memref<4x4xf64>
//       %3 = mulf %2, %2 : f64
//       affine.store %3, %arg0[%arg3 floordiv 4, %arg3 mod 4] : memref<4x4xf64>
//     }
//     %0 = alloc() : memref<16xf64, #map0>
//     %1 = affine.load %0[0] : memref<16xf64, #map0>
//     return %arg0 : memref<4x4xf64>
//   }

struct NormalizeMemRefs : public NormalizeMemRefsBase<NormalizeMemRefs> {
  void runOnOperation() override;
  void runOnFunction(FuncOp funcOp);
  bool areMemRefsNormalizable(FuncOp funcOp);
  void updateFunctionSignature(FuncOp funcOp);
};

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createNormalizeMemRefsPass() {
  return std::make_unique<NormalizeMemRefs>();
}

void NormalizeMemRefs::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  // We traverse each function within the module in order to normalize the
  // memref type arguments.
  // TODO: Handle external functions.
  moduleOp.walk([&](FuncOp funcOp) {
    if (areMemRefsNormalizable(funcOp))
      runOnFunction(funcOp);
  });
}

// Return true if this operation dereferences one or more memref's.
// TODO: Temporary utility, will be replaced when this is modeled through
// side-effects/op traits.
static bool isMemRefDereferencingOp(Operation &op) {
  return isa<AffineReadOpInterface, AffineWriteOpInterface, AffineDmaStartOp,
             AffineDmaWaitOp>(op);
}

// Check whether all the uses of oldMemRef are either dereferencing uses or the
// op is of type : DeallocOp, CallOp. Only if these constraints are satisfied
// will the value become a candidate for replacement.
static bool isMemRefNormalizable(Value::user_range opUsers) {
  if (llvm::any_of(opUsers, [](Operation *op) {
        if (isMemRefDereferencingOp(*op))
          return false;
        return !isa<DeallocOp, CallOp>(*op);
      }))
    return false;
  return true;
}

// Check whether all the uses of AllocOps, CallOps and function arguments of a
// function are either of dereferencing type or of type: DeallocOp, CallOp. Only
// if these constraints are satisfied will the function become a candidate for
// normalization.
bool NormalizeMemRefs::areMemRefsNormalizable(FuncOp funcOp) {
  if (funcOp
          .walk([&](AllocOp allocOp) -> WalkResult {
            Value oldMemRef = allocOp.getResult();
            if (!isMemRefNormalizable(oldMemRef.getUsers()))
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return false;

  if (funcOp
          .walk([&](CallOp callOp) -> WalkResult {
            for (unsigned resIndex :
                 llvm::seq<unsigned>(0, callOp.getNumResults())) {
              Value oldMemRef = callOp.getResult(resIndex);
              if (oldMemRef.getType().isa<MemRefType>())
                if (!isMemRefNormalizable(oldMemRef.getUsers()))
                  return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted())
    return false;

  for (unsigned argIndex : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {
    BlockArgument oldMemRef = funcOp.getArgument(argIndex);
    if (oldMemRef.getType().isa<MemRefType>())
      if (!isMemRefNormalizable(oldMemRef.getUsers()))
        return false;
  }

  return true;
}

// Fetch the updated argument list and result of the function and update the
// function signature.
void NormalizeMemRefs::updateFunctionSignature(FuncOp funcOp) {
  FunctionType functionType = funcOp.getType();
  SmallVector<Type, 8> argTypes;
  SmallVector<Type, 4> resultTypes;

  for (const auto &arg : llvm::enumerate(funcOp.getArguments()))
    argTypes.push_back(arg.value().getType());

  resultTypes = llvm::to_vector<4>(functionType.getResults());
  // We create a new function type and modify the function signature with this
  // new type.
  FunctionType newFuncType = FunctionType::get(/*inputs=*/argTypes,
                                               /*results=*/resultTypes,
                                               /*context=*/&getContext());

  // TODO: Handle ReturnOps to update function results the caller site.
  funcOp.setType(newFuncType);
}

void NormalizeMemRefs::runOnFunction(FuncOp funcOp) {
  // Turn memrefs' non-identity layouts maps into ones with identity. Collect
  // alloc ops first and then process since normalizeMemRef replaces/erases ops
  // during memref rewriting.
  SmallVector<AllocOp, 4> allocOps;
  funcOp.walk([&](AllocOp op) { allocOps.push_back(op); });
  for (AllocOp allocOp : allocOps)
    normalizeMemRef(allocOp);

  // We use this OpBuilder to create new memref layout later.
  OpBuilder b(funcOp);

  // Walk over each argument of a function to perform memref normalization (if
  // any).
  for (unsigned argIndex : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {
    Type argType = funcOp.getArgument(argIndex).getType();
    MemRefType memrefType = argType.dyn_cast<MemRefType>();
    // Check whether argument is of MemRef type. Any other argument type can
    // simply be part of the final function signature.
    if (!memrefType)
      continue;
    // Fetch a new memref type after normalizing the old memref to have an
    // identity map layout.
    MemRefType newMemRefType = normalizeMemRefType(memrefType, b,
                                                   /*numSymbolicOperands=*/0);
    if (newMemRefType == memrefType) {
      // Either memrefType already had an identity map or the map couldn't be
      // transformed to an identity map.
      continue;
    }

    // Insert a new temporary argument with the new memref type.
    BlockArgument newMemRef =
        funcOp.front().insertArgument(argIndex, newMemRefType);
    BlockArgument oldMemRef = funcOp.getArgument(argIndex + 1);
    AffineMap layoutMap = memrefType.getAffineMaps().front();
    // Replace all uses of the old memref.
    if (failed(replaceAllMemRefUsesWith(oldMemRef, /*newMemRef=*/newMemRef,
                                        /*extraIndices=*/{},
                                        /*indexRemap=*/layoutMap,
                                        /*extraOperands=*/{},
                                        /*symbolOperands=*/{},
                                        /*domInstFilter=*/nullptr,
                                        /*postDomInstFilter=*/nullptr,
                                        /*allowNonDereferencingOps=*/true,
                                        /*handleDeallocOp=*/true))) {
      // If it failed (due to escapes for example), bail out. Removing the
      // temporary argument inserted previously.
      funcOp.front().eraseArgument(argIndex);
      continue;
    }

    // All uses for the argument with old memref type were replaced
    // successfully. So we remove the old argument now.
    funcOp.front().eraseArgument(argIndex + 1);
  }

  updateFunctionSignature(funcOp);
}
