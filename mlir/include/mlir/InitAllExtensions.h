//===- InitAllExtensions.h - MLIR Extension Registration --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to trigger the registration of all dialect
// extensions to the system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INITALLEXTENSIONS_H_
#define MLIR_INITALLEXTENSIONS_H_

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"

#include <cstdlib>

namespace mlir {

/// This function may be called to register all MLIR dialect extensions with the
/// provided registry.
/// If you're building a compiler, you generally shouldn't use this: you would
/// individually register the specific extensions that are useful for the
/// pipelines and transformations you are using.
inline void registerAllExtensions(DialectRegistry &registry) {
  func::registerAllExtensions(registry);
}

} // namespace mlir

#endif // MLIR_INITALLEXTENSIONS_H_
