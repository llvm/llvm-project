//===- Markup.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the log symbolizer markup data model and parser.
///
/// \todo Add a link to the reference documentation once added.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_SYMBOLIZE_MARKUP_H
#define LLVM_DEBUGINFO_SYMBOLIZE_MARKUP_H

#include <iostream>

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Regex.h"

namespace llvm {
namespace symbolize {

/// A node of symbolizer markup.
///
/// If only the Text field is set, this represents a region of text outside a
/// markup element. ANSI SGR control codes are also reported this way; if
/// detected, then the control code will be the entirety of the Text field, and
/// any surrounding text will be reported as preceding and following nodes.
struct MarkupNode {
  /// The full text of this node in the input.
  StringRef Text;

  /// If this represents an element, the tag. Otherwise, empty.
  StringRef Tag;

  /// If this represents an element with fields, a list of the field contents.
  /// Otherwise, empty.
  SmallVector<StringRef> Fields;

  bool operator==(const MarkupNode &Other) const {
    return Text == Other.Text && Tag == Other.Tag && Fields == Other.Fields;
  }
  bool operator!=(const MarkupNode &Other) const { return !(*this == Other); }
};

/// Parses a log containing symbolizer markup into a sequence of nodes.
class MarkupParser {
public:
  MarkupParser();

  /// Parses an individual \p Line of input.
  ///
  /// Nodes from the previous parseLine() call that haven't yet been extracted
  /// by nextNode() are discarded. The nodes returned by nextNode() may
  /// reference the input string, so it must be retained by the caller until the
  /// last use.
  void parseLine(StringRef Line);

  /// Returns the next node from the most recent parseLine() call.
  ///
  /// Calling nextNode() may invalidate the contents of the node returned by the
  /// previous call.
  ///
  /// \returns the next markup node or None if none remain.
  Optional<MarkupNode> nextNode() {
    if (!NextIdx)
      NextIdx = 0;
    if (*NextIdx == Buffer.size()) {
      NextIdx.reset();
      Buffer.clear();
      return None;
    }
    return std::move(Buffer[(*NextIdx)++]);
  }

private:
  Optional<MarkupNode> parseElement(StringRef Line);
  void parseTextOutsideMarkup(StringRef Text);

  // Buffer for nodes parsed from the current line.
  SmallVector<MarkupNode> Buffer;

  // Next buffer index to return or None if nextNode has not yet been called.
  Optional<size_t> NextIdx;

  // Regular expression matching supported ANSI SGR escape sequences.
  const Regex SGRSyntax;
};

} // end namespace symbolize
} // end namespace llvm

#endif // LLVM_DEBUGINFO_SYMBOLIZE_MARKUP_H
