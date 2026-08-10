//===- StandaloneTypes.h - Standalone dialect types -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_STANDALONETYPES_H
#define STANDALONE_STANDALONETYPES_H

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Standalone/StandaloneOpsTypes.h.inc"

#endif // STANDALONE_STANDALONETYPES_H
