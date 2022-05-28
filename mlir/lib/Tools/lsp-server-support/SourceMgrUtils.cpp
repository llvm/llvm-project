//===--- SourceMgrUtils.cpp - SourceMgr LSP Utils -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SourceMgrUtils.h"
#include "llvm/Support/Path.h"

using namespace mlir;
using namespace mlir::lsp;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// SourceMgrInclude
//===----------------------------------------------------------------------===//

Hover SourceMgrInclude::buildHover() const {
  Hover hover(range);
  {
    llvm::raw_string_ostream hoverOS(hover.contents.value);
    hoverOS << "`" << llvm::sys::path::filename(uri.file()) << "`\n***\n"
            << uri.file();
  }
  return hover;
}

void lsp::gatherIncludeFiles(llvm::SourceMgr &sourceMgr,
                             SmallVectorImpl<SourceMgrInclude> &includes) {
  for (unsigned i = 1, e = sourceMgr.getNumBuffers(); i < e; ++i) {
    // Check to see if this file was included by the main file.
    SMLoc includeLoc = sourceMgr.getBufferInfo(i + 1).IncludeLoc;
    if (!includeLoc.isValid() || sourceMgr.FindBufferContainingLoc(
                                     includeLoc) != sourceMgr.getMainFileID())
      continue;

    // Try to build a URI for this file path.
    auto *buffer = sourceMgr.getMemoryBuffer(i + 1);
    llvm::SmallString<256> path(buffer->getBufferIdentifier());
    llvm::sys::path::remove_dots(path, /*remove_dot_dot=*/true);

    llvm::Expected<URIForFile> includedFileURI = URIForFile::fromFile(path);
    if (!includedFileURI)
      continue;

    // Find the end of the include token.
    const char *includeStart = includeLoc.getPointer() - 2;
    while (*(--includeStart) != '\"')
      continue;

    // Push this include.
    SMRange includeRange(SMLoc::getFromPointer(includeStart), includeLoc);
    includes.emplace_back(*includedFileURI, Range(sourceMgr, includeRange));
  }
}
