//===- mlir-config.h - MLIR configuration ------------------------*- C -*-===*//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* This file enumerates variables from the MLIR configuration so that they
   can be in exported headers and won't override package specific directives.
   This is a C header that can be included in the mlir-c headers. */

#ifndef MLIR_CONFIG_H
#define MLIR_CONFIG_H

/* Enable expensive checks to detect invalid pattern API usage. Failed checks
   manifest as fatal errors or invalid memory accesses (e.g., accessing
   deallocated memory) that cause a crash. Running with ASAN is recommended for
   easier debugging. */
#cmakedefine01 MLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS

/* If set, greedy pattern application is randomized: ops on the worklist are
   chosen at random. For testing/debugging purposes only. This feature can be
   used to ensure that lowering pipelines work correctly regardless of the order
   in which ops are processed by the GreedyPatternRewriteDriver. This flag is
   numeric seed that is passed to the random number generator. */
#cmakedefine MLIR_GREEDY_REWRITE_RANDOMIZER_SEED ${MLIR_GREEDY_REWRITE_RANDOMIZER_SEED}

#endif
