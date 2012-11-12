//===-- llvm/Support/FormattedStream.cpp - Formatted streams ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of formatted_raw_ostream.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormattedStream.h"
#include <algorithm>

using namespace llvm;

/// CountColumns - Examine the given char sequence and figure out which
/// column we end up in after output.
///
static unsigned CountColumns(unsigned Column, const char *Ptr, size_t Size) {
  // Keep track of the current column by scanning the string for
  // special characters

  for (const char *End = Ptr + Size; Ptr != End; ++Ptr) {
    ++Column;
    if (*Ptr == '\n' || *Ptr == '\r')
      Column = 0;
    else if (*Ptr == '\t')
      // Assumes tab stop = 8 characters.
      Column += (8 - (Column & 0x7)) & 0x7;
  }

  return Column;
}

/// ComputeColumn - Examine the current output and figure out which
/// column we end up in after output.
void formatted_raw_ostream::ComputeColumn(const char *Ptr, size_t Size) {
  // If our previous scan pointer is inside the buffer, assume we already
  // scanned those bytes. This depends on raw_ostream to not change our buffer
  // in unexpected ways.
  if (Ptr <= Scanned && Scanned <= Ptr + Size) {
    // Scan all characters added since our last scan to determine the new
    // column.
    ColumnScanned = CountColumns(ColumnScanned, Scanned, 
                                 Size - (Scanned - Ptr));
  } else
    ColumnScanned = CountColumns(ColumnScanned, Ptr, Size);

  // Update the scanning pointer.
  Scanned = Ptr + Size;
}

/// PadToColumn - Align the output to some column number.
///
/// \param NewCol - The column to move to.
///
formatted_raw_ostream &formatted_raw_ostream::PadToColumn(unsigned NewCol) { 
  // Figure out what's in the buffer and add it to the column count.
  ComputeColumn(getBufferStart(), GetNumBytesInBuffer());

  // Output spaces until we reach the desired column.
  indent(std::max(int(NewCol - ColumnScanned), 1));
  return *this;
}

void formatted_raw_ostream::write_impl(const char *Ptr, size_t Size) {
  // Figure out what's in the buffer and add it to the column count.
  ComputeColumn(Ptr, Size);

  // Write the data to the underlying stream (which is unbuffered, so
  // the data will be immediately written out).
  TheStream->write(Ptr, Size);

  // Reset the scanning pointer.
  Scanned = 0;
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
