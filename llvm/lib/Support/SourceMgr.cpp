//===- SourceMgr.cpp - Manager for Simple Source Buffers & Diagnostics ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SourceMgr class.  This class is used as a simple
// substrate for diagnostics, #include handling, and other low level things for
// simple parsers.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
using namespace llvm;

namespace {
  struct LineNoCacheTy {
    int LastQueryBufferID;
    const char *LastQuery;
    unsigned LineNoOfQuery;
  };
}

static LineNoCacheTy *getCache(void *Ptr) {
  return (LineNoCacheTy*)Ptr;
}


SourceMgr::~SourceMgr() {
  // Delete the line # cache if allocated.
  if (LineNoCacheTy *Cache = getCache(LineNoCache))
    delete Cache;

  while (!Buffers.empty()) {
    delete Buffers.back().Buffer;
    Buffers.pop_back();
  }
}

/// AddIncludeFile - Search for a file with the specified name in the current
/// directory or in one of the IncludeDirs.  If no file is found, this returns
/// ~0, otherwise it returns the buffer ID of the stacked file.
unsigned SourceMgr::AddIncludeFile(const std::string &Filename,
                                   SMLoc IncludeLoc,
                                   std::string &IncludedFile) {
  OwningPtr<MemoryBuffer> NewBuf;
  IncludedFile = Filename;
  MemoryBuffer::getFile(IncludedFile.c_str(), NewBuf);

  // If the file didn't exist directly, see if it's in an include path.
  for (unsigned i = 0, e = IncludeDirectories.size(); i != e && !NewBuf; ++i) {
    IncludedFile = IncludeDirectories[i] + "/" + Filename;
    MemoryBuffer::getFile(IncludedFile.c_str(), NewBuf);
  }

  if (NewBuf == 0) return ~0U;

  return AddNewSourceBuffer(NewBuf.take(), IncludeLoc);
}


/// FindBufferContainingLoc - Return the ID of the buffer containing the
/// specified location, returning -1 if not found.
int SourceMgr::FindBufferContainingLoc(SMLoc Loc) const {
  for (unsigned i = 0, e = Buffers.size(); i != e; ++i)
    if (Loc.getPointer() >= Buffers[i].Buffer->getBufferStart() &&
        // Use <= here so that a pointer to the null at the end of the buffer
        // is included as part of the buffer.
        Loc.getPointer() <= Buffers[i].Buffer->getBufferEnd())
      return i;
  return -1;
}

/// getLineAndColumn - Find the line and column number for the specified
/// location in the specified file.  This is not a fast method.
std::pair<unsigned, unsigned>
SourceMgr::getLineAndColumn(SMLoc Loc, int BufferID) const {
  if (BufferID == -1) BufferID = FindBufferContainingLoc(Loc);
  assert(BufferID != -1 && "Invalid Location!");

  MemoryBuffer *Buff = getBufferInfo(BufferID).Buffer;

  // Count the number of \n's between the start of the file and the specified
  // location.
  unsigned LineNo = 1;

  const char *BufStart = Buff->getBufferStart();
  const char *Ptr = BufStart;

  // If we have a line number cache, and if the query is to a later point in the
  // same file, start searching from the last query location.  This optimizes
  // for the case when multiple diagnostics come out of one file in order.
  if (LineNoCacheTy *Cache = getCache(LineNoCache))
    if (Cache->LastQueryBufferID == BufferID &&
        Cache->LastQuery <= Loc.getPointer()) {
      Ptr = Cache->LastQuery;
      LineNo = Cache->LineNoOfQuery;
    }

  // Scan for the location being queried, keeping track of the number of lines
  // we see.
  for (; SMLoc::getFromPointer(Ptr) != Loc; ++Ptr)
    if (*Ptr == '\n') ++LineNo;

  // Allocate the line number cache if it doesn't exist.
  if (LineNoCache == 0)
    LineNoCache = new LineNoCacheTy();

  // Update the line # cache.
  LineNoCacheTy &Cache = *getCache(LineNoCache);
  Cache.LastQueryBufferID = BufferID;
  Cache.LastQuery = Ptr;
  Cache.LineNoOfQuery = LineNo;
  
  size_t NewlineOffs = StringRef(BufStart, Ptr-BufStart).find_last_of("\n\r");
  if (NewlineOffs == StringRef::npos) NewlineOffs = ~(size_t)0;
  return std::make_pair(LineNo, Ptr-BufStart-NewlineOffs);
}

void SourceMgr::PrintIncludeStack(SMLoc IncludeLoc, raw_ostream &OS) const {
  if (IncludeLoc == SMLoc()) return;  // Top of stack.

  int CurBuf = FindBufferContainingLoc(IncludeLoc);
  assert(CurBuf != -1 && "Invalid or unspecified location!");

  PrintIncludeStack(getBufferInfo(CurBuf).IncludeLoc, OS);

  OS << "Included from "
     << getBufferInfo(CurBuf).Buffer->getBufferIdentifier()
     << ":" << FindLineNumber(IncludeLoc, CurBuf) << ":\n";
}


/// GetMessage - Return an SMDiagnostic at the specified location with the
/// specified string.
///
/// @param Type - If non-null, the kind of message (e.g., "error") which is
/// prefixed to the message.
SMDiagnostic SourceMgr::GetMessage(SMLoc Loc, SourceMgr::DiagKind Kind,
                                   const Twine &Msg,
                                   ArrayRef<SMRange> Ranges) const {

  // First thing to do: find the current buffer containing the specified
  // location to pull out the source line.
  SmallVector<std::pair<unsigned, unsigned>, 4> ColRanges;
  std::pair<unsigned, unsigned> LineAndCol;
  const char *BufferID = "<unknown>";
  std::string LineStr;
  
  if (Loc.isValid()) {
    int CurBuf = FindBufferContainingLoc(Loc);
    assert(CurBuf != -1 && "Invalid or unspecified location!");

    MemoryBuffer *CurMB = getBufferInfo(CurBuf).Buffer;
    BufferID = CurMB->getBufferIdentifier();
    
    // Scan backward to find the start of the line.
    const char *LineStart = Loc.getPointer();
    const char *BufStart = CurMB->getBufferStart();
    while (LineStart != BufStart && LineStart[-1] != '\n' &&
           LineStart[-1] != '\r')
      --LineStart;

    // Get the end of the line.
    const char *LineEnd = Loc.getPointer();
    const char *BufEnd = CurMB->getBufferEnd();
    while (LineEnd != BufEnd && LineEnd[0] != '\n' && LineEnd[0] != '\r')
      ++LineEnd;
    LineStr = std::string(LineStart, LineEnd);

    // Convert any ranges to column ranges that only intersect the line of the
    // location.
    for (unsigned i = 0, e = Ranges.size(); i != e; ++i) {
      SMRange R = Ranges[i];
      if (!R.isValid()) continue;
      
      // If the line doesn't contain any part of the range, then ignore it.
      if (R.Start.getPointer() > LineEnd || R.End.getPointer() < LineStart)
        continue;
     
      // Ignore pieces of the range that go onto other lines.
      if (R.Start.getPointer() < LineStart)
        R.Start = SMLoc::getFromPointer(LineStart);
      if (R.End.getPointer() > LineEnd)
        R.End = SMLoc::getFromPointer(LineEnd);
      
      // Translate from SMLoc ranges to column ranges.
      ColRanges.push_back(std::make_pair(R.Start.getPointer()-LineStart,
                                         R.End.getPointer()-LineStart));
    }

    LineAndCol = getLineAndColumn(Loc, CurBuf);
  }
    
  return SMDiagnostic(*this, Loc, BufferID, LineAndCol.first,
                      LineAndCol.second-1, Kind, Msg.str(),
                      LineStr, ColRanges);
}

void SourceMgr::PrintMessage(SMLoc Loc, SourceMgr::DiagKind Kind,
                             const Twine &Msg, ArrayRef<SMRange> Ranges,
                             bool ShowColors) const {
  SMDiagnostic Diagnostic = GetMessage(Loc, Kind, Msg, Ranges);
  
  // Report the message with the diagnostic handler if present.
  if (DiagHandler) {
    DiagHandler(Diagnostic, DiagContext);
    return;
  }

  raw_ostream &OS = errs();

  if (Loc != SMLoc()) {
    int CurBuf = FindBufferContainingLoc(Loc);
    assert(CurBuf != -1 && "Invalid or unspecified location!");
    PrintIncludeStack(getBufferInfo(CurBuf).IncludeLoc, OS);
  }

  Diagnostic.print(0, OS, ShowColors);
}

//===----------------------------------------------------------------------===//
// SMDiagnostic Implementation
//===----------------------------------------------------------------------===//

SMDiagnostic::SMDiagnostic(const SourceMgr &sm, SMLoc L, const std::string &FN,
                           int Line, int Col, SourceMgr::DiagKind Kind,
                           const std::string &Msg,
                           const std::string &LineStr,
                           ArrayRef<std::pair<unsigned,unsigned> > Ranges)
  : SM(&sm), Loc(L), Filename(FN), LineNo(Line), ColumnNo(Col), Kind(Kind),
    Message(Msg), LineContents(LineStr), Ranges(Ranges.vec()) {
}


void SMDiagnostic::print(const char *ProgName, raw_ostream &S,
                         bool ShowColors) const {
  // Display colors only if OS supports colors.
  ShowColors &= S.has_colors();

  if (ShowColors)
    S.changeColor(raw_ostream::SAVEDCOLOR, true);

  if (ProgName && ProgName[0])
    S << ProgName << ": ";

  if (!Filename.empty()) {
    if (Filename == "-")
      S << "<stdin>";
    else
      S << Filename;

    if (LineNo != -1) {
      S << ':' << LineNo;
      if (ColumnNo != -1)
        S << ':' << (ColumnNo+1);
    }
    S << ": ";
  }

  switch (Kind) {
  case SourceMgr::DK_Error:
    if (ShowColors)
      S.changeColor(raw_ostream::RED, true);
    S << "error: ";
    break;
  case SourceMgr::DK_Warning:
    if (ShowColors)
      S.changeColor(raw_ostream::MAGENTA, true);
    S << "warning: ";
    break;
  case SourceMgr::DK_Note:
    if (ShowColors)
      S.changeColor(raw_ostream::BLACK, true);
    S << "note: ";
    break;
  }

  if (ShowColors) {
    S.resetColor();
    S.changeColor(raw_ostream::SAVEDCOLOR, true);
  }

  S << Message << '\n';

  if (ShowColors)
    S.resetColor();

  if (LineNo == -1 || ColumnNo == -1)
    return;

  // Build the line with the caret and ranges.
  std::string CaretLine(LineContents.size()+1, ' ');
  
  // Expand any ranges.
  for (unsigned r = 0, e = Ranges.size(); r != e; ++r) {
    std::pair<unsigned, unsigned> R = Ranges[r];
    for (unsigned i = R.first,
         e = std::min(R.second, (unsigned)LineContents.size())+1; i != e; ++i)
      CaretLine[i] = '~';
  }
    
  // Finally, plop on the caret.
  if (unsigned(ColumnNo) <= LineContents.size())
    CaretLine[ColumnNo] = '^';
  else 
    CaretLine[LineContents.size()] = '^';
  
  // ... and remove trailing whitespace so the output doesn't wrap for it.  We
  // know that the line isn't completely empty because it has the caret in it at
  // least.
  CaretLine.erase(CaretLine.find_last_not_of(' ')+1);
  
  // Print out the source line one character at a time, so we can expand tabs.
  for (unsigned i = 0, e = LineContents.size(), OutCol = 0; i != e; ++i) {
    if (LineContents[i] != '\t') {
      S << LineContents[i];
      ++OutCol;
      continue;
    }
    
    // If we have a tab, emit at least one space, then round up to 8 columns.
    do {
      S << ' ';
      ++OutCol;
    } while (OutCol & 7);
  }
  S << '\n';

  if (ShowColors)
    S.changeColor(raw_ostream::GREEN, true);

  // Print out the caret line, matching tabs in the source line.
  for (unsigned i = 0, e = CaretLine.size(), OutCol = 0; i != e; ++i) {
    if (i >= LineContents.size() || LineContents[i] != '\t') {
      S << CaretLine[i];
      ++OutCol;
      continue;
    }
    
    // Okay, we have a tab.  Insert the appropriate number of characters.
    do {
      S << CaretLine[i];
      ++OutCol;
    } while (OutCol & 7);
  }

  if (ShowColors)
    S.resetColor();
  
  S << '\n';
}
