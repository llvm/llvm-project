//===- ODSDialectHook.h - Dialect customization hooks into ODS --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines ODS customization hooks for dialects to programmatically
// emit dialect specific contents in ODS C++ code emission.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_ODSDIALECTHOOK_H_
#define MLIR_TABLEGEN_ODSDIALECTHOOK_H_

#include <functional>

namespace llvm {
class StringRef;
}

namespace mlir {
namespace tblgen {
class Operator;
class OpClass;

// The emission function for dialect specific content. It takes in an Operator
// and updates the OpClass accordingly.
using DialectEmitFunction =
    std::function<void(const Operator &srcOp, OpClass &emitClass)>;

// ODSDialectHookRegistration provides a global initializer that registers a
// dialect specific content emission function.
struct ODSDialectHookRegistration {
  ODSDialectHookRegistration(llvm::StringRef dialectName,
                             DialectEmitFunction emitFn);
};
} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_ODSDIALECTHOOK_H_
