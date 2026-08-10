//===-- PassUtil.td - Utilities for rewrite passes ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains several utilities for passes that utilize rewrite
// patterns.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REWRITE_PASSUTIL_TD_
#define MLIR_REWRITE_PASSUTIL_TD_

include "mlir/Pass/PassBase.td"

def RewritePassUtils {
  // A set of options commonly options used for pattern rewrites.
  list<Option> options = [
    // These two options provide filtering for which patterns are applied. These
    // should be passed directly to the FrozenRewritePatternSet when it is
    // created.
    ListOption<"disabledPatterns", "disable-patterns", "std::string",
               "Labels of patterns that should be filtered out during"
               " application">,
    ListOption<"enabledPatterns", "enable-patterns", "std::string",
               "Labels of patterns that should be used during"
               " application, all other patterns are filtered out">,
  ];
}

#endif // MLIR_REWRITE_PASSUTIL_TD_
