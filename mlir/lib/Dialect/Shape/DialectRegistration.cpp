//===- DialectRegistration.cpp - Register shape dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Shape/IR/Shape.h"
using namespace mlir;

// Static initialization for shape dialect registration.
static DialectRegistration<shape::ShapeDialect> ShapeOps;
