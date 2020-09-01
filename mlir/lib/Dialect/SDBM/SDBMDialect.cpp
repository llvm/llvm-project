//===- SDBMDialect.cpp - MLIR SDBM Dialect --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SDBM/SDBMDialect.h"
#include "SDBMExprDetail.h"

using namespace mlir;

SDBMDialect::SDBMDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<SDBMDialect>()) {
  uniquer.registerParametricStorageType<detail::SDBMBinaryExprStorage>();
  uniquer.registerParametricStorageType<detail::SDBMConstantExprStorage>();
  uniquer.registerParametricStorageType<detail::SDBMDiffExprStorage>();
  uniquer.registerParametricStorageType<detail::SDBMNegExprStorage>();
  uniquer.registerParametricStorageType<detail::SDBMTermExprStorage>();
}

SDBMDialect::~SDBMDialect() = default;
