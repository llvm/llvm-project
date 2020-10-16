//===- CallInterfaces.cpp - ControlFlow Interfaces ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/CallInterfaces.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// CallOpInterface
//===----------------------------------------------------------------------===//

/// Resolve the callable operation for given callee to a CallableOpInterface, or
/// nullptr if a valid callable was not resolved. `symbolTable` is an optional
/// parameter that will allow for using a cached symbol table for symbol lookups
/// instead of performing an O(N) scan.
Operation *
CallOpInterface::resolveCallable(SymbolTableCollection *symbolTable) {
  CallInterfaceCallable callable = getCallableForCallee();
  if (auto symbolVal = callable.dyn_cast<Value>())
    return symbolVal.getDefiningOp();

  // If the callable isn't a value, lookup the symbol reference.
  auto symbolRef = callable.get<SymbolRefAttr>();
  if (symbolTable)
    return symbolTable->lookupNearestSymbolFrom(getOperation(), symbolRef);
  return SymbolTable::lookupNearestSymbolFrom(getOperation(), symbolRef);
}

//===----------------------------------------------------------------------===//
// CallInterfaces
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/CallInterfaces.cpp.inc"
