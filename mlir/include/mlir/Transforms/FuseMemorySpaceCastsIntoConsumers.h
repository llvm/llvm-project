//===-- FuseMemorySpaceCastsIntoConsumers.h - Cast fusion patterns -C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_FUSEMEMORYSPACECASTSINTOCONSUMERS_H
#define MLIR_TRANSFORMS_FUSEMEMORYSPACECASTSINTOCONSUMERS_H

namespace mlir {
class RewritePatternSet;
/// Collect a set of patterns to fuse memory-space cast operations into
/// consumers.
void populateFuseMemorySpaceCastIntoConsumersPatterns(
    RewritePatternSet &patterns);
} // namespace mlir

#endif // MLIR_TRANSFORMS_FUSEMEMORYSPACECASTSINTOCONSUMERS_H
