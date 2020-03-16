//===-- llvm/Support/FormattedStream.cpp - Formatted streams ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of formatted_raw_ostream.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Locale.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

using namespace llvm;

/// UpdatePosition - Examine the given char sequence and figure out which
/// column we end up in after output, and how many line breaks are contained.
/// This assumes that the input string is well-formed UTF-8, and takes into
/// account unicode characters which render as multiple columns wide.
static void UpdatePosition(std::pair<unsigned, unsigned> &Position,
                           const char *Ptr, size_t Size) {
  unsigned &Column = Position.first;
  unsigned &Line = Position.second;

  // Keep track of the current column and line by scanning the string for
  // special characters.
  unsigned NumBytes;
  for (const char *End = Ptr + Size; Ptr < End; Ptr += NumBytes) {
    NumBytes = getNumBytesForUTF8(*Ptr);

    // The string should never end part way through a multi-byte sequence.
    assert((Ptr + NumBytes) <= End && "Malformed multi-byte sequence");

    int Width = sys::locale::columnWidth(StringRef(Ptr, NumBytes));
    // columnWidth returns -1 for non-printing characters.
    if (Width != -1)
      Column += Width;

    // If this is the final byte of a multi-byte sequence, it can't be any of
    // the special whitespace characters below.
    if (NumBytes > 1)
      continue;

    switch (*Ptr) {
    case '\n':
      Line += 1;
      LLVM_FALLTHROUGH;
    case '\r':
      Column = 0;
      break;
    case '\t':
      // Assumes tab stop = 8 characters.
      Column += (8 - (Column & 0x7)) & 0x7;
      break;
    }
  }
}

/// ComputePosition - Examine the current output and update line and column
/// counts.
void formatted_raw_ostream::ComputePosition(const char *Ptr, size_t Size) {
  // If our previous scan pointer is inside the buffer, assume we already
  // scanned those bytes. This depends on raw_ostream to not change our buffer
  // in unexpected ways.
  if (Ptr <= Scanned && Scanned <= Ptr + Size)
    // Scan all characters added since our last scan to determine the new
    // column.
    UpdatePosition(Position, Scanned, Size - (Scanned - Ptr));
  else
    UpdatePosition(Position, Ptr, Size);

  // Update the scanning pointer.
  Scanned = Ptr + Size;
}

/// PadToColumn - Align the output to some column number.
///
/// \param NewCol - The column to move to.
///
formatted_raw_ostream &formatted_raw_ostream::PadToColumn(unsigned NewCol) {
  // Figure out what's in the buffer and add it to the column count.
  ComputePosition(getBufferStart(), GetNumBytesInBuffer());

  // Output spaces until we reach the desired column.
  indent(std::max(int(NewCol - getColumn()), 1));
  return *this;
}

void formatted_raw_ostream::write_impl(const char *Ptr, size_t Size) {
  // Figure out what's in the buffer and add it to the column count.
  ComputePosition(Ptr, Size);

  // Write the data to the underlying stream (which is unbuffered, so
  // the data will be immediately written out).
  TheStream->write(Ptr, Size);

  // Reset the scanning pointer.
  Scanned = nullptr;
}

/// fouts() - This returns a reference to a formatted_raw_ostream for
/// standard output.  Use it like: fouts() << "foo" << "bar";
formatted_raw_ostream &llvm::fouts() {
  static formatted_raw_ostream S(outs());
  return S;
}

/// ferrs() - This returns a reference to a formatted_raw_ostream for
/// standard error.  Use it like: ferrs() << "foo" << "bar";
formatted_raw_ostream &llvm::ferrs() {
  static formatted_raw_ostream S(errs());
  return S;
}

/// fdbgs() - This returns a reference to a formatted_raw_ostream for
/// the debug stream.  Use it like: fdbgs() << "foo" << "bar";
formatted_raw_ostream &llvm::fdbgs() {
  static formatted_raw_ostream S(dbgs());
  return S;
}
