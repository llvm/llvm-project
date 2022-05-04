//===- MarkupFilter.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares a filter that replaces symbolizer markup with
/// human-readable expressions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_SYMBOLIZE_MARKUPFILTER_H
#define LLVM_DEBUGINFO_SYMBOLIZE_MARKUPFILTER_H

#include "Markup.h"

#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace symbolize {

/// Filter to convert parsed log symbolizer markup elements into human-readable
/// text.
class MarkupFilter {
public:
  MarkupFilter(raw_ostream &OS, Optional<bool> ColorsEnabled = llvm::None);

  /// Begins a logical \p Line of markup.
  ///
  /// This must be called for each line of the input stream before calls to
  /// filter() for elements of that line. The provided \p Line must be the same
  /// one that was passed to parseLine() to produce the elements to be later
  /// passed to filter().
  ///
  /// This informs the filter that a new line is beginning and establishes a
  /// context for error location reporting.
  void beginLine(StringRef Line);

  /// Handle a \p Node of symbolizer markup.
  ///
  /// If the node is a recognized, valid markup element, it is replaced with a
  /// human-readable string. If the node isn't an element or the element isn't
  /// recognized, it is output verbatim. If the element is recognized but isn't
  /// valid, it is omitted from the output.
  void filter(const MarkupNode &Node);

private:
  bool trySGR(const MarkupNode &Node);

  void highlight();
  void restoreColor();
  void resetColor();

  bool checkTag(const MarkupNode &Node) const;
  bool checkNumFields(const MarkupNode &Node, size_t Size) const;

  void reportTypeError(StringRef Str, StringRef TypeName) const;
  void reportLocation(StringRef::iterator Loc) const;

  raw_ostream &OS;
  const bool ColorsEnabled;

  StringRef Line;

  Optional<raw_ostream::Colors> Color;
  bool Bold = false;
};

} // end namespace symbolize
} // end namespace llvm

#endif // LLVM_DEBUGINFO_SYMBOLIZE_MARKUPFILTER_H
