//===- GCOV.h - LLVM coverage tool ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header provides the interface to read and write coverage files that
// use 'gcov' format.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_GCOV_H
#define LLVM_PROFILEDATA_GCOV_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>

namespace llvm {

class GCOVFunction;
class GCOVBlock;
class FileInfo;

namespace GCOV {

enum GCOVVersion { V304, V407, V408, V800, V900 };

/// A struct for passing gcov options between functions.
struct Options {
  Options(bool A, bool B, bool C, bool F, bool P, bool U, bool L, bool N,
          bool T, bool X)
      : AllBlocks(A), BranchInfo(B), BranchCount(C), FuncCoverage(F),
        PreservePaths(P), UncondBranch(U), LongFileNames(L), NoOutput(N),
        UseStdout(T), HashFilenames(X) {}

  bool AllBlocks;
  bool BranchInfo;
  bool BranchCount;
  bool FuncCoverage;
  bool PreservePaths;
  bool UncondBranch;
  bool LongFileNames;
  bool NoOutput;
  bool UseStdout;
  bool HashFilenames;
};

} // end namespace GCOV

/// GCOVBuffer - A wrapper around MemoryBuffer to provide GCOV specific
/// read operations.
class GCOVBuffer {
public:
  GCOVBuffer(MemoryBuffer *B) : Buffer(B) {}
  ~GCOVBuffer() { consumeError(cursor.takeError()); }

  /// readGCNOFormat - Check GCNO signature is valid at the beginning of buffer.
  bool readGCNOFormat() {
    StringRef buf = Buffer->getBuffer();
    StringRef magic = buf.substr(0, 4);
    if (magic == "gcno") {
      de = DataExtractor(buf.substr(4), false, 0);
    } else if (magic == "oncg") {
      de = DataExtractor(buf.substr(4), true, 0);
    } else {
      errs() << "unexpected magic: " << magic << "\n";
      return false;
    }
    return true;
  }

  /// readGCDAFormat - Check GCDA signature is valid at the beginning of buffer.
  bool readGCDAFormat() {
    StringRef buf = Buffer->getBuffer();
    StringRef magic = buf.substr(0, 4);
    if (magic == "gcda") {
      de = DataExtractor(buf.substr(4), false, 0);
    } else if (magic == "adcg") {
      de = DataExtractor(buf.substr(4), true, 0);
    } else {
      errs() << "unexpected file type: " << magic << "\n";
      return false;
    }
    return true;
  }

  /// readGCOVVersion - Read GCOV version.
  bool readGCOVVersion(GCOV::GCOVVersion &Version) {
    std::string str(de.getBytes(cursor, 4));
    if (str.size() != 4)
      return false;
    if (de.isLittleEndian())
      std::reverse(str.begin(), str.end());
    int ver = str[0] >= 'A'
                  ? (str[0] - 'A') * 100 + (str[1] - '0') * 10 + str[2] - '0'
                  : (str[0] - '0') * 10 + str[2] - '0';
    if (ver >= 90) {
      // PR gcov-profile/84846, r269678
      Version = GCOV::V900;
      return true;
    } else if (ver >= 80) {
      // PR gcov-profile/48463
      Version = GCOV::V800;
      return true;
    } else if (ver >= 48) {
      // r189778: the exit block moved from the last to the second.
      Version = GCOV::V408;
      return true;
    } else if (ver >= 47) {
      // r173147: split checksum into cfg checksum and line checksum.
      Version = GCOV::V407;
      return true;
    } else if (ver >= 34) {
      Version = GCOV::V304;
      return true;
    }
    errs() << "unexpected version: " << str << "\n";
    return false;
  }

  uint32_t getWord() { return de.getU32(cursor); }
  StringRef getString() {
    uint32_t len;
    if (!readInt(len) || len == 0)
      return {};
    return de.getBytes(cursor, len * 4).split('\0').first;
  }

  bool readInt(uint32_t &Val) {
    if (cursor.tell() + 4 > de.size()) {
      Val = 0;
      errs() << "unexpected end of memory buffer: " << cursor.tell() << "\n";
      return false;
    }
    Val = de.getU32(cursor);
    return true;
  }

  bool readInt64(uint64_t &Val) {
    uint32_t Lo, Hi;
    if (!readInt(Lo) || !readInt(Hi))
      return false;
    Val = ((uint64_t)Hi << 32) | Lo;
    return true;
  }

  bool readString(StringRef &Str) {
    uint32_t len;
    if (!readInt(len) || len == 0)
      return false;
    Str = de.getBytes(cursor, len * 4).split('\0').first;
    return bool(cursor);
  }

  DataExtractor de{ArrayRef<uint8_t>{}, false, 0};
  DataExtractor::Cursor cursor{0};

private:
  MemoryBuffer *Buffer;
};

/// GCOVFile - Collects coverage information for one pair of coverage file
/// (.gcno and .gcda).
class GCOVFile {
public:
  GCOVFile() = default;

  bool readGCNO(GCOVBuffer &Buffer);
  bool readGCDA(GCOVBuffer &Buffer);
  GCOV::GCOVVersion getVersion() const { return Version; }
  uint32_t getChecksum() const { return Checksum; }
  void print(raw_ostream &OS) const;
  void dump() const;
  void collectLineCounts(FileInfo &FI);

private:
  bool GCNOInitialized = false;
  GCOV::GCOVVersion Version;
  uint32_t Checksum = 0;
  StringRef cwd;
  SmallVector<std::unique_ptr<GCOVFunction>, 16> Functions;
  std::map<uint32_t, GCOVFunction *> IdentToFunction;
  uint32_t RunCount = 0;
  uint32_t ProgramCount = 0;
};

struct GCOVArc {
  GCOVArc(GCOVBlock &src, GCOVBlock &dst, bool fallthrough)
      : src(src), dst(dst), fallthrough(fallthrough) {}

  GCOVBlock &src;
  GCOVBlock &dst;
  bool fallthrough;
  uint64_t Count = 0;
  uint64_t CyclesCount = 0;
};

/// GCOVFunction - Collects function information.
class GCOVFunction {
public:
  using BlockIterator = pointee_iterator<
      SmallVectorImpl<std::unique_ptr<GCOVBlock>>::const_iterator>;

  GCOVFunction(GCOVFile &P) {}

  StringRef getName() const { return Name; }
  StringRef getFilename() const { return Filename; }
  size_t getNumBlocks() const { return Blocks.size(); }
  uint64_t getEntryCount() const;
  uint64_t getExitCount() const;

  BlockIterator block_begin() const { return Blocks.begin(); }
  BlockIterator block_end() const { return Blocks.end(); }
  iterator_range<BlockIterator> blocks() const {
    return make_range(block_begin(), block_end());
  }

  void print(raw_ostream &OS) const;
  void dump() const;
  void collectLineCounts(FileInfo &FI);

  uint32_t ident = 0;
  uint32_t linenoChecksum;
  uint32_t cfgChecksum = 0;
  uint32_t startLine = 0;
  uint32_t startColumn = 0;
  uint32_t endLine = 0;
  uint32_t endColumn = 0;
  uint8_t artificial = 0;
  StringRef Name;
  StringRef Filename;
  SmallVector<std::unique_ptr<GCOVBlock>, 0> Blocks;
  SmallVector<std::unique_ptr<GCOVArc>, 0> arcs, treeArcs;
};

/// GCOVBlock - Collects block information.
class GCOVBlock {
  struct EdgeWeight {
    EdgeWeight(GCOVBlock *D) : Dst(D) {}

    GCOVBlock *Dst;
    uint64_t Count = 0;
  };

public:
  using EdgeIterator = SmallVectorImpl<GCOVArc *>::const_iterator;
  using BlockVector = SmallVector<const GCOVBlock *, 4>;
  using BlockVectorLists = SmallVector<BlockVector, 4>;
  using Edges = SmallVector<GCOVArc *, 4>;

  GCOVBlock(GCOVFunction &P, uint32_t N) : Parent(P), Number(N) {}

  const GCOVFunction &getParent() const { return Parent; }
  void addLine(uint32_t N) { Lines.push_back(N); }
  uint32_t getLastLine() const { return Lines.back(); }
  uint64_t getCount() const { return Counter; }

  void addSrcEdge(GCOVArc *Edge) { pred.push_back(Edge); }

  void addDstEdge(GCOVArc *Edge) { succ.push_back(Edge); }

  size_t getNumSrcEdges() const { return pred.size(); }
  size_t getNumDstEdges() const { return succ.size(); }

  iterator_range<EdgeIterator> srcs() const {
    return make_range(pred.begin(), pred.end());
  }

  iterator_range<EdgeIterator> dsts() const {
    return make_range(succ.begin(), succ.end());
  }

  void print(raw_ostream &OS) const;
  void dump() const;
  void collectLineCounts(FileInfo &FI);

  static uint64_t getCycleCount(const Edges &Path);
  static void unblock(const GCOVBlock *U, BlockVector &Blocked,
                      BlockVectorLists &BlockLists);
  static bool lookForCircuit(const GCOVBlock *V, const GCOVBlock *Start,
                             Edges &Path, BlockVector &Blocked,
                             BlockVectorLists &BlockLists,
                             const BlockVector &Blocks, uint64_t &Count);
  static void getCyclesCount(const BlockVector &Blocks, uint64_t &Count);
  static uint64_t getLineCount(const BlockVector &Blocks);

public:
  GCOVFunction &Parent;
  uint32_t Number;
  uint64_t Counter = 0;
  SmallVector<GCOVArc *, 2> pred;
  SmallVector<GCOVArc *, 2> succ;
  SmallVector<uint32_t, 16> Lines;
};

class FileInfo {
protected:
  // It is unlikely--but possible--for multiple functions to be on the same
  // line.
  // Therefore this typedef allows LineData.Functions to store multiple
  // functions
  // per instance. This is rare, however, so optimize for the common case.
  using FunctionVector = SmallVector<const GCOVFunction *, 1>;
  using FunctionLines = DenseMap<uint32_t, FunctionVector>;
  using BlockVector = SmallVector<const GCOVBlock *, 4>;
  using BlockLines = DenseMap<uint32_t, BlockVector>;

  struct LineData {
    LineData() = default;

    BlockLines Blocks;
    FunctionLines Functions;
    uint32_t LastLine = 0;
  };

  struct GCOVCoverage {
    GCOVCoverage(StringRef Name) : Name(Name) {}

    StringRef Name;

    uint32_t LogicalLines = 0;
    uint32_t LinesExec = 0;

    uint32_t Branches = 0;
    uint32_t BranchesExec = 0;
    uint32_t BranchesTaken = 0;
  };

public:
  FileInfo(const GCOV::Options &Options) : Options(Options) {}

  void addBlockLine(StringRef Filename, uint32_t Line, const GCOVBlock *Block) {
    if (Line > LineInfo[Filename].LastLine)
      LineInfo[Filename].LastLine = Line;
    LineInfo[Filename].Blocks[Line - 1].push_back(Block);
  }

  void addFunctionLine(StringRef Filename, uint32_t Line,
                       const GCOVFunction *Function) {
    if (Line > LineInfo[Filename].LastLine)
      LineInfo[Filename].LastLine = Line;
    LineInfo[Filename].Functions[Line - 1].push_back(Function);
  }

  void setRunCount(uint32_t Runs) { RunCount = Runs; }
  void setProgramCount(uint32_t Programs) { ProgramCount = Programs; }
  void print(raw_ostream &OS, StringRef MainFilename, StringRef GCNOFile,
             StringRef GCDAFile, GCOV::GCOVVersion Version);

protected:
  std::string getCoveragePath(StringRef Filename, StringRef MainFilename);
  std::unique_ptr<raw_ostream> openCoveragePath(StringRef CoveragePath);
  void printFunctionSummary(raw_ostream &OS, const FunctionVector &Funcs) const;
  void printBlockInfo(raw_ostream &OS, const GCOVBlock &Block,
                      uint32_t LineIndex, uint32_t &BlockNo) const;
  void printBranchInfo(raw_ostream &OS, const GCOVBlock &Block,
                       GCOVCoverage &Coverage, uint32_t &EdgeNo);
  void printUncondBranchInfo(raw_ostream &OS, uint32_t &EdgeNo,
                             uint64_t Count) const;

  void printCoverage(raw_ostream &OS, const GCOVCoverage &Coverage) const;
  void printFuncCoverage(raw_ostream &OS) const;
  void printFileCoverage(raw_ostream &OS) const;

  const GCOV::Options &Options;
  StringMap<LineData> LineInfo;
  uint32_t RunCount = 0;
  uint32_t ProgramCount = 0;

  using FileCoverageList = SmallVector<std::pair<std::string, GCOVCoverage>, 4>;
  using FuncCoverageMap = MapVector<const GCOVFunction *, GCOVCoverage>;

  FileCoverageList FileCoverages;
  FuncCoverageMap FuncCoverages;
};

} // end namespace llvm

#endif // LLVM_SUPPORT_GCOV_H
