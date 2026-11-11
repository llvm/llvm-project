//===- ShardingInterfaceImpl.h - Linalg sharding models ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_EXTENSIONS_SHARDINGINTERFACEIMPL_H_
#define MLIR_DIALECT_LINALG_EXTENSIONS_SHARDINGINTERFACEIMPL_H_

namespace mlir {
class DialectRegistry;

namespace linalg {
void registerShardingInterfaceExternalModels(DialectRegistry &registry);
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_EXTENSIONS_SHARDINGINTERFACEIMPL_H_
