//===- GCOVr.cpp - LLVM coverage tool -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// GCOV implements the interface to read and write coverage files that use 
// 'gcov' format.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/GCOV.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/system_error.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// GCOVFile implementation.

/// ~GCOVFile - Delete GCOVFile and its content.
GCOVFile::~GCOVFile() {
  DeleteContainerPointers(Functions);
}

/// isGCDAFile - Return true if Format identifies a .gcda file.
static bool isGCDAFile(GCOV::GCOVFormat Format) {
  return Format == GCOV::GCDA_402 || Format == GCOV::GCDA_404;
}

/// isGCNOFile - Return true if Format identifies a .gcno file.
static bool isGCNOFile(GCOV::GCOVFormat Format) {
  return Format == GCOV::GCNO_402 || Format == GCOV::GCNO_404;
}

/// read - Read GCOV buffer.
bool GCOVFile::read(GCOVBuffer &Buffer) {
  GCOV::GCOVFormat Format = Buffer.readGCOVFormat();
  if (Format == GCOV::InvalidGCOV)
    return false;

  unsigned i = 0;
  while (1) {
    GCOVFunction *GFun = NULL;
    if (isGCDAFile(Format)) {
      // Use existing function while reading .gcda file.
      assert(i < Functions.size() && ".gcda data does not match .gcno data");
      GFun = Functions[i];
    } else if (isGCNOFile(Format)){
      GFun = new GCOVFunction();
      Functions.push_back(GFun);
    }
    if (!GFun || !GFun->read(Buffer, Format))
      break;
    ++i;
  }
  return true;
}

/// dump - Dump GCOVFile content on standard out for debugging purposes.
void GCOVFile::dump() {
  for (SmallVector<GCOVFunction *, 16>::iterator I = Functions.begin(),
         E = Functions.end(); I != E; ++I)
    (*I)->dump();
}

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.
void GCOVFile::collectLineCounts(FileInfo &FI) {
  for (SmallVector<GCOVFunction *, 16>::iterator I = Functions.begin(),
         E = Functions.end(); I != E; ++I) 
    (*I)->collectLineCounts(FI);
  FI.print();
}

//===----------------------------------------------------------------------===//
// GCOVFunction implementation.

/// ~GCOVFunction - Delete GCOVFunction and its content.
GCOVFunction::~GCOVFunction() {
  DeleteContainerPointers(Blocks);
}

/// read - Read a aunction from the buffer. Return false if buffer cursor
/// does not point to a function tag.
bool GCOVFunction::read(GCOVBuffer &Buff, GCOV::GCOVFormat Format) {
  if (!Buff.readFunctionTag())
    return false;

  Buff.readInt(); // Function header length
  Ident = Buff.readInt(); 
  Buff.readInt(); // Checksum #1
  if (Format != GCOV::GCNO_402)
    Buff.readInt(); // Checksum #2

  Name = Buff.readString();
  if (Format == GCOV::GCNO_402 || Format == GCOV::GCNO_404)
    Filename = Buff.readString();

  if (Format == GCOV::GCDA_402 || Format == GCOV::GCDA_404) {
    Buff.readArcTag();
    uint32_t Count = Buff.readInt() / 2;
    for (unsigned i = 0, e = Count; i != e; ++i) {
      Blocks[i]->addCount(Buff.readInt64());
    }
    return true;
  }

  LineNumber = Buff.readInt();

  // read blocks.
  bool BlockTagFound = Buff.readBlockTag();
  (void)BlockTagFound;
  assert(BlockTagFound && "Block Tag not found!");
  uint32_t BlockCount = Buff.readInt();
  for (int i = 0, e = BlockCount; i != e; ++i) {
    Buff.readInt(); // Block flags;
    Blocks.push_back(new GCOVBlock(i));
  }

  // read edges.
  while (Buff.readEdgeTag()) {
    uint32_t EdgeCount = (Buff.readInt() - 1) / 2;
    uint32_t BlockNo = Buff.readInt();
    assert(BlockNo < BlockCount && "Unexpected Block number!");
    for (int i = 0, e = EdgeCount; i != e; ++i) {
      Blocks[BlockNo]->addEdge(Buff.readInt());
      Buff.readInt(); // Edge flag
    }
  }

  // read line table.
  while (Buff.readLineTag()) {
    uint32_t LineTableLength = Buff.readInt();
    uint32_t Size = Buff.getCursor() + LineTableLength*4;
    uint32_t BlockNo = Buff.readInt();
    assert(BlockNo < BlockCount && "Unexpected Block number!");
    GCOVBlock *Block = Blocks[BlockNo];
    Buff.readInt(); // flag
    while (Buff.getCursor() != (Size - 4)) {
      StringRef Filename = Buff.readString();
      if (Buff.getCursor() == (Size - 4)) break;
      while (uint32_t L = Buff.readInt())
        Block->addLine(Filename, L);
    }
    Buff.readInt(); // flag
  }
  return true;
}

/// dump - Dump GCOVFunction content on standard out for debugging purposes.
void GCOVFunction::dump() {
  outs() <<  "===== " << Name << " @ " << Filename << ":" << LineNumber << "\n";
  for (SmallVector<GCOVBlock *, 16>::iterator I = Blocks.begin(),
         E = Blocks.end(); I != E; ++I)
    (*I)->dump();
}

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.
void GCOVFunction::collectLineCounts(FileInfo &FI) {
  for (SmallVector<GCOVBlock *, 16>::iterator I = Blocks.begin(),
         E = Blocks.end(); I != E; ++I)
    (*I)->collectLineCounts(FI);
}

//===----------------------------------------------------------------------===//
// GCOVBlock implementation.

/// ~GCOVBlock - Delete GCOVBlock and its content.
GCOVBlock::~GCOVBlock() {
  Edges.clear();
  DeleteContainerSeconds(Lines);
}

void GCOVBlock::addLine(StringRef Filename, uint32_t LineNo) {
  GCOVLines *&LinesForFile = Lines[Filename];
  if (!LinesForFile)
    LinesForFile = new GCOVLines();
  LinesForFile->add(LineNo);
}

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.
void GCOVBlock::collectLineCounts(FileInfo &FI) {
  for (StringMap<GCOVLines *>::iterator I = Lines.begin(),
         E = Lines.end(); I != E; ++I)
    I->second->collectLineCounts(FI, I->first(), Counter);
}

/// dump - Dump GCOVBlock content on standard out for debugging purposes.
void GCOVBlock::dump() {
  outs() << "Block : " << Number << " Counter : " << Counter << "\n";
  if (!Edges.empty()) {
    outs() << "\tEdges : ";
    for (SmallVector<uint32_t, 16>::iterator I = Edges.begin(), E = Edges.end();
         I != E; ++I)
      outs() << (*I) << ",";
    outs() << "\n";
  }
  if (!Lines.empty()) {
    outs() << "\tLines : ";
    for (StringMap<GCOVLines *>::iterator LI = Lines.begin(),
           LE = Lines.end(); LI != LE; ++LI) {
      outs() << LI->first() << " -> ";
      LI->second->dump();
      outs() << "\n";
    }
  }
}

//===----------------------------------------------------------------------===//
// GCOVLines implementation.

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.
void GCOVLines::collectLineCounts(FileInfo &FI, StringRef Filename, 
                                  uint32_t Count) {
  for (SmallVector<uint32_t, 16>::iterator I = Lines.begin(),
         E = Lines.end(); I != E; ++I)
    FI.addLineCount(Filename, *I, Count);
}

/// dump - Dump GCOVLines content on standard out for debugging purposes.
void GCOVLines::dump() {
  for (SmallVector<uint32_t, 16>::iterator I = Lines.begin(),
         E = Lines.end(); I != E; ++I)
    outs() << (*I) << ",";
}

//===----------------------------------------------------------------------===//
// FileInfo implementation.

/// addLineCount - Add line count for the given line number in a file.
void FileInfo::addLineCount(StringRef Filename, uint32_t Line, uint32_t Count) {
  if (LineInfo.find(Filename) == LineInfo.end()) {
    OwningPtr<MemoryBuffer> Buff;
    if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename, Buff)) {
      errs() << Filename << ": " << ec.message() << "\n";
      return;
    }
    StringRef AllLines = Buff.take()->getBuffer();
    LineCounts L(AllLines.count('\n')+2);
    L[Line-1] = Count;
    LineInfo[Filename] = L;
    return;
  }
  LineCounts &L = LineInfo[Filename];
  L[Line-1] = Count;
}

/// print -  Print source files with collected line count information.
void FileInfo::print() {
  for (StringMap<LineCounts>::iterator I = LineInfo.begin(), E = LineInfo.end();
       I != E; ++I) {
    StringRef Filename = I->first();
    outs() << Filename << "\n";
    LineCounts &L = LineInfo[Filename];
    OwningPtr<MemoryBuffer> Buff;
    if (error_code ec = MemoryBuffer::getFileOrSTDIN(Filename, Buff)) {
      errs() << Filename << ": " << ec.message() << "\n";
      return;
    }
    StringRef AllLines = Buff.take()->getBuffer();
    for (unsigned i = 0, e = L.size(); i != e; ++i) {
      if (L[i])
        outs() << L[i] << ":\t";
      else
        outs() << " :\t";
      std::pair<StringRef, StringRef> P = AllLines.split('\n');
      if (AllLines != P.first)
        outs() << P.first;
      outs() << "\n";
      AllLines = P.second;
    }
  }
}


