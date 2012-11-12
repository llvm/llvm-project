//===-- llvm/Support/GCOV.h - LLVM coverage tool ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header provides the interface to read and write coverage files that 
// use 'gcov' format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_GCOV_H
#define LLVM_GCOV_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class GCOVFunction;
class GCOVBlock;
class GCOVLines;
class FileInfo;

namespace GCOV {
  enum GCOVFormat {
    InvalidGCOV,
    GCNO_402,
    GCNO_404,
    GCDA_402,
    GCDA_404
  };
} // end GCOV namespace

/// GCOVBuffer - A wrapper around MemoryBuffer to provide GCOV specific
/// read operations.
class GCOVBuffer {
public:
  GCOVBuffer(MemoryBuffer *B) : Buffer(B), Cursor(0) {}
  
  /// readGCOVFormat - Read GCOV signature at the beginning of buffer.
  GCOV::GCOVFormat readGCOVFormat() {
    StringRef Magic = Buffer->getBuffer().slice(0, 12);
    Cursor = 12;
    if (Magic == "oncg*404MVLL")
      return GCOV::GCNO_404;
    else if (Magic == "oncg*204MVLL")
      return GCOV::GCNO_402;
    else if (Magic == "adcg*404MVLL")
      return GCOV::GCDA_404;
    else if (Magic == "adcg*204MVLL")
      return GCOV::GCDA_402;
    
    Cursor = 0;
    return GCOV::InvalidGCOV;
  }

  /// readFunctionTag - If cursor points to a function tag then increment the
  /// cursor and return true otherwise return false.
  bool readFunctionTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor+4);
    if (Tag.empty() || 
        Tag[0] != '\0' || Tag[1] != '\0' ||
        Tag[2] != '\0' || Tag[3] != '\1') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readBlockTag - If cursor points to a block tag then increment the
  /// cursor and return true otherwise return false.
  bool readBlockTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor+4);
    if (Tag.empty() || 
        Tag[0] != '\0' || Tag[1] != '\0' ||
        Tag[2] != '\x41' || Tag[3] != '\x01') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readEdgeTag - If cursor points to an edge tag then increment the
  /// cursor and return true otherwise return false.
  bool readEdgeTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor+4);
    if (Tag.empty() || 
        Tag[0] != '\0' || Tag[1] != '\0' ||
        Tag[2] != '\x43' || Tag[3] != '\x01') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readLineTag - If cursor points to a line tag then increment the
  /// cursor and return true otherwise return false.
  bool readLineTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor+4);
    if (Tag.empty() || 
        Tag[0] != '\0' || Tag[1] != '\0' ||
        Tag[2] != '\x45' || Tag[3] != '\x01') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  /// readArcTag - If cursor points to an gcda arc tag then increment the
  /// cursor and return true otherwise return false.
  bool readArcTag() {
    StringRef Tag = Buffer->getBuffer().slice(Cursor, Cursor+4);
    if (Tag.empty() || 
        Tag[0] != '\0' || Tag[1] != '\0' ||
        Tag[2] != '\xa1' || Tag[3] != '\1') {
      return false;
    }
    Cursor += 4;
    return true;
  }

  uint32_t readInt() {
    uint32_t Result;
    StringRef Str = Buffer->getBuffer().slice(Cursor, Cursor+4);
    assert (Str.empty() == false && "Unexpected memory buffer end!");
    Cursor += 4;
    Result = *(const uint32_t *)(Str.data());
    return Result;
  }

  uint64_t readInt64() {
    uint64_t Lo = readInt();
    uint64_t Hi = readInt();
    uint64_t Result = Lo | (Hi << 32);
    return Result;
  }

  StringRef readString() {
    uint32_t Len = readInt() * 4;
    StringRef Str = Buffer->getBuffer().slice(Cursor, Cursor+Len);
    Cursor += Len;
    return Str;
  }

  uint64_t getCursor() const { return Cursor; }
private:
  MemoryBuffer *Buffer;
  uint64_t Cursor;
};

/// GCOVFile - Collects coverage information for one pair of coverage file
/// (.gcno and .gcda).
class GCOVFile {
public:
  GCOVFile() {}
  ~GCOVFile();
  bool read(GCOVBuffer &Buffer);
  void dump();
  void collectLineCounts(FileInfo &FI);
private:
  SmallVector<GCOVFunction *, 16> Functions;
};

/// GCOVFunction - Collects function information.
class GCOVFunction {
public:
  GCOVFunction() : Ident(0), LineNumber(0) {}
  ~GCOVFunction();
  bool read(GCOVBuffer &Buffer, GCOV::GCOVFormat Format);
  void dump();
  void collectLineCounts(FileInfo &FI);
private:
  uint32_t Ident;
  uint32_t LineNumber;
  StringRef Name;
  StringRef Filename;
  SmallVector<GCOVBlock *, 16> Blocks;
};

/// GCOVBlock - Collects block information.
class GCOVBlock {
public:
  GCOVBlock(uint32_t N) : Number(N), Counter(0) {}
  ~GCOVBlock();
  void addEdge(uint32_t N) { Edges.push_back(N); }
  void addLine(StringRef Filename, uint32_t LineNo);
  void addCount(uint64_t N) { Counter = N; }
  void dump();
  void collectLineCounts(FileInfo &FI);
private:
  uint32_t Number;
  uint64_t Counter;
  SmallVector<uint32_t, 16> Edges;
  StringMap<GCOVLines *> Lines;
};

/// GCOVLines - A wrapper around a vector of int to keep track of line nos.
class GCOVLines {
public:
  ~GCOVLines() { Lines.clear(); }
  void add(uint32_t N) { Lines.push_back(N); }
  void collectLineCounts(FileInfo &FI, StringRef Filename, uint32_t Count);
  void dump();

private:
  SmallVector<uint32_t, 4> Lines;
};

typedef SmallVector<uint32_t, 16> LineCounts;
class FileInfo {
public:
  void addLineCount(StringRef Filename, uint32_t Line, uint32_t Count);
  void print();
private:
  StringMap<LineCounts> LineInfo;
};

}

#endif
