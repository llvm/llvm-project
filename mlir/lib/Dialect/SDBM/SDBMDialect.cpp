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
  uniquer.registerStorageType(TypeID::get<detail::SDBMBinaryExprStorage>());
  uniquer.registerStorageType(TypeID::get<detail::SDBMConstantExprStorage>());
  uniquer.registerStorageType(TypeID::get<detail::SDBMDiffExprStorage>());
  uniquer.registerStorageType(TypeID::get<detail::SDBMNegExprStorage>());
  uniquer.registerStorageType(TypeID::get<detail::SDBMTermExprStorage>());
}

SDBMDialect::~SDBMDialect() = default;
