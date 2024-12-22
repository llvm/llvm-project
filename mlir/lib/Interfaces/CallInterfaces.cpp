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

Operation *
call_interface_impl::resolveCallable(CallOpInterface call,
                                     SymbolTableCollection *symbolTable) {
  CallInterfaceCallable callable = call.getCallableForCallee();
  if (auto symbolVal = dyn_cast<Value>(callable))
    return symbolVal.getDefiningOp();

  // If the callable isn't a value, lookup the symbol reference.
  auto symbolRef = callable.get<SymbolRefAttr>();
  if (symbolTable)
    return symbolTable->lookupNearestSymbolFrom(call.getOperation(), symbolRef);
  return SymbolTable::lookupNearestSymbolFrom(call.getOperation(), symbolRef);
}

//===----------------------------------------------------------------------===//
// CallInterfaces
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/CallInterfaces.cpp.inc"
