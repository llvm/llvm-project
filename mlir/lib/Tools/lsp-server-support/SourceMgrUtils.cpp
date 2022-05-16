//===--- SourceMgrUtils.cpp - SourceMgr LSP Utils -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SourceMgrUtils.h"

using namespace mlir;
using namespace mlir::lsp;

/// Find the end of a string whose contents start at the given `curPtr`. Returns
/// the position at the end of the string, after a terminal or invalid character
/// (e.g. `"` or `\0`).
static const char *lexLocStringTok(const char *curPtr) {
  while (char c = *curPtr++) {
    // Check for various terminal characters.
    if (StringRef("\"\n\v\f").contains(c))
      return curPtr;

    // Check for escape sequences.
    if (c == '\\') {
      // Check a few known escapes and \xx hex digits.
      if (*curPtr == '"' || *curPtr == '\\' || *curPtr == 'n' || *curPtr == 't')
        ++curPtr;
      else if (llvm::isHexDigit(*curPtr) && llvm::isHexDigit(curPtr[1]))
        curPtr += 2;
      else
        return curPtr;
    }
  }

  // If we hit this point, we've reached the end of the buffer. Update the end
  // pointer to not point past the buffer.
  return curPtr - 1;
}

SMRange lsp::convertTokenLocToRange(SMLoc loc) {
  if (!loc.isValid())
    return SMRange();
  const char *curPtr = loc.getPointer();

  // Check if this is a string token.
  if (*curPtr == '"') {
    curPtr = lexLocStringTok(curPtr + 1);

    // Otherwise, default to handling an identifier.
  } else {
    // Return if the given character is a valid identifier character.
    auto isIdentifierChar = [](char c) {
      return isalnum(c) || c == '$' || c == '.' || c == '_' || c == '-';
    };

    while (*curPtr && isIdentifierChar(*(++curPtr)))
      continue;
  }

  return SMRange(loc, SMLoc::getFromPointer(curPtr));
}
