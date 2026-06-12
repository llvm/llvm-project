//===- AIRDialect.cpp - AIR dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AIR/IR/AIRDialect.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::air;

#include "mlir/Dialect/AIR/IR/AIROpsDialect.cpp.inc"

void AIRDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/AIR/IR/AIROpsAttributes.cpp.inc"
      >();
  declarePromisedInterface<mlir::gpu::TargetAttrInterface,
                           mlir::air::AIRTargetAttr>();
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/AIR/IR/AIROpsAttributes.cpp.inc"
