//===--- SourceMgrUtils.h - SourceMgr LSP Utils -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an array of generally useful SourceMgr utilities for
// interacting with LSP components.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_MLIR_TOOLS_LSPSERVERSUPPORT_TRANSPORT_H_
#define LIB_MLIR_TOOLS_LSPSERVERSUPPORT_TRANSPORT_H_

#include "Protocol.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
namespace lsp {

/// Returns the range of a lexical token given a SMLoc corresponding to the
/// start of an token location. The range is computed heuristically, and
/// supports identifier-like tokens, strings, etc.
SMRange convertTokenLocToRange(SMLoc loc);

} // namespace lsp
} // namespace mlir

#endif
