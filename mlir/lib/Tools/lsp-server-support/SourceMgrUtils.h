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
#include <optional>

namespace mlir {
namespace lsp {
//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Returns the range of a lexical token given a SMLoc corresponding to the
/// start of an token location. The range is computed heuristically, and
/// supports identifier-like tokens, strings, etc.
SMRange convertTokenLocToRange(SMLoc loc);

/// Extract a documentation comment for the given location within the source
/// manager. Returns std::nullopt if no comment could be computed.
std::optional<std::string> extractSourceDocComment(llvm::SourceMgr &sourceMgr,
                                                   SMLoc loc);

/// Returns true if the given range contains the given source location. Note
/// that this has different behavior than SMRange because it is inclusive of the
/// end location.
bool contains(SMRange range, SMLoc loc);

//===----------------------------------------------------------------------===//
// SourceMgrInclude
//===----------------------------------------------------------------------===//

/// This class represents a single include within a root file.
struct SourceMgrInclude {
  SourceMgrInclude(const lsp::URIForFile &uri, const lsp::Range &range)
      : uri(uri), range(range) {}

  /// Build a hover for the current include file.
  Hover buildHover() const;

  /// The URI of the file that is included.
  lsp::URIForFile uri;

  /// The range of the include directive.
  lsp::Range range;
};

/// Given a source manager, gather all of the processed include files. These are
/// assumed to be all of the files other than the main root file.
void gatherIncludeFiles(llvm::SourceMgr &sourceMgr,
                        SmallVectorImpl<SourceMgrInclude> &includes);

} // namespace lsp
} // namespace mlir

#endif
