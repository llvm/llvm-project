//===-- SourceFile.cpp - SourceFile implementation for the debugger -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SourceFile class for the LLVM debugger.
//
//===----------------------------------------------------------------------===//

#include "llvm/Debugger/SourceFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cassert>
using namespace llvm;

static const char EmptyFile = 0;

SourceFile::SourceFile(const std::string &fn, const GlobalVariable *Desc)
  : Filename(fn), Descriptor(Desc) {
  File.reset(MemoryBuffer::getFileOrSTDIN(fn));
    
  // On error, return an empty buffer.
  if (File == 0)
    File.reset(MemoryBuffer::getMemBuffer(&EmptyFile, &EmptyFile));
}

SourceFile::~SourceFile() {
}


/// calculateLineOffsets - Compute the LineOffset vector for the current file.
///
void SourceFile::calculateLineOffsets() const {
  assert(LineOffset.empty() && "Line offsets already computed!");
  const char *BufPtr = File->getBufferStart();
  const char *FileStart = BufPtr;
  const char *FileEnd = File->getBufferEnd();
  do {
    LineOffset.push_back(BufPtr-FileStart);

    // Scan until we get to a newline.
    while (BufPtr != FileEnd && *BufPtr != '\n' && *BufPtr != '\r')
      ++BufPtr;

    if (BufPtr != FileEnd) {
      ++BufPtr;               // Skip over the \n or \r
      if (BufPtr[-1] == '\r' && BufPtr != FileEnd && BufPtr[0] == '\n')
        ++BufPtr;   // Skip over dos/windows style \r\n's
    }
  } while (BufPtr != FileEnd);
}


/// getSourceLine - Given a line number, return the start and end of the line
/// in the file.  If the line number is invalid, or if the file could not be
/// loaded, null pointers are returned for the start and end of the file. Note
/// that line numbers start with 0, not 1.
void SourceFile::getSourceLine(unsigned LineNo, const char *&LineStart,
                               const char *&LineEnd) const {
  LineStart = LineEnd = 0;
  if (LineOffset.empty()) calculateLineOffsets();

  // Asking for an out-of-range line number?
  if (LineNo >= LineOffset.size()) return;

  // Otherwise, they are asking for a valid line, which we can fulfill.
  LineStart = File->getBufferStart()+LineOffset[LineNo];

  if (LineNo+1 < LineOffset.size())
    LineEnd = File->getBufferStart()+LineOffset[LineNo+1];
  else
    LineEnd = File->getBufferEnd();

  // If the line ended with a newline, strip it off.
  while (LineEnd != LineStart && (LineEnd[-1] == '\n' || LineEnd[-1] == '\r'))
    --LineEnd;

  assert(LineEnd >= LineStart && "We somehow got our pointers swizzled!");
}
