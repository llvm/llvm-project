//===- GCOV.cpp - LLVM coverage tool --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// GCOV implements the interface to read and write coverage files that use
// 'gcov' format.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/GCOV.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <system_error>

using namespace llvm;

enum : uint32_t {
  GCOV_ARC_ON_TREE = 1 << 0,
  GCOV_ARC_FALLTHROUGH = 1 << 2,

  GCOV_TAG_FUNCTION = 0x01000000,
  GCOV_TAG_BLOCKS = 0x01410000,
  GCOV_TAG_ARCS = 0x01430000,
  GCOV_TAG_LINES = 0x01450000,
  GCOV_TAG_COUNTER_ARCS = 0x01a10000,
  // GCOV_TAG_OBJECT_SUMMARY superseded GCOV_TAG_PROGRAM_SUMMARY in GCC 9.
  GCOV_TAG_OBJECT_SUMMARY = 0xa1000000,
  GCOV_TAG_PROGRAM_SUMMARY = 0xa3000000,
};

//===----------------------------------------------------------------------===//
// GCOVFile implementation.

/// readGCNO - Read GCNO buffer.
bool GCOVFile::readGCNO(GCOVBuffer &buf) {
  if (!buf.readGCNOFormat())
    return false;
  if (!buf.readGCOVVersion(Version))
    return false;

  Checksum = buf.getWord();
  if (Version >= GCOV::V900)
    cwd = buf.getString();
  if (Version >= GCOV::V800)
    buf.getWord(); // hasUnexecutedBlocks

  uint32_t tag, length;
  GCOVFunction *fn;
  while ((tag = buf.getWord())) {
    if (!buf.readInt(length))
      return false;
    if (tag == GCOV_TAG_FUNCTION) {
      Functions.push_back(std::make_unique<GCOVFunction>(*this));
      fn = Functions.back().get();
      fn->ident = buf.getWord();
      fn->linenoChecksum = buf.getWord();
      if (Version >= GCOV::V407)
        fn->cfgChecksum = buf.getWord();
      buf.readString(fn->Name);
      StringRef filename;
      if (Version < GCOV::V800) {
        filename = buf.getString();
        fn->startLine = buf.getWord();
      } else {
        fn->artificial = buf.getWord();
        filename = buf.getString();
        fn->startLine = buf.getWord();
        fn->startColumn = buf.getWord();
        fn->endLine = buf.getWord();
        if (Version >= GCOV::V900)
          fn->endColumn = buf.getWord();
      }
      auto r = filenameToIdx.try_emplace(filename, filenameToIdx.size());
      if (r.second)
        filenames.emplace_back(filename);
      fn->srcIdx = r.first->second;
      IdentToFunction[fn->ident] = fn;
    } else if (tag == GCOV_TAG_BLOCKS && fn) {
      if (Version < GCOV::V800) {
        for (uint32_t i = 0; i != length; ++i) {
          buf.getWord(); // Ignored block flags
          fn->Blocks.push_back(std::make_unique<GCOVBlock>(*fn, i));
        }
      } else {
        uint32_t num = buf.getWord();
        for (uint32_t i = 0; i != num; ++i)
          fn->Blocks.push_back(std::make_unique<GCOVBlock>(*fn, i));
      }
    } else if (tag == GCOV_TAG_ARCS && fn) {
      uint32_t srcNo = buf.getWord();
      if (srcNo >= fn->Blocks.size()) {
        errs() << "unexpected block number: " << srcNo << " (in "
               << fn->Blocks.size() << ")\n";
        return false;
      }
      GCOVBlock *src = fn->Blocks[srcNo].get();
      for (uint32_t i = 0, e = (length - 1) / 2; i != e; ++i) {
        uint32_t dstNo = buf.getWord(), flags = buf.getWord();
        GCOVBlock *dst = fn->Blocks[dstNo].get();
        auto arc =
            std::make_unique<GCOVArc>(*src, *dst, flags & GCOV_ARC_FALLTHROUGH);
        src->addDstEdge(arc.get());
        dst->addSrcEdge(arc.get());
        if (flags & GCOV_ARC_ON_TREE)
          fn->treeArcs.push_back(std::move(arc));
        else
          fn->arcs.push_back(std::move(arc));
      }
    } else if (tag == GCOV_TAG_LINES && fn) {
      uint32_t srcNo = buf.getWord();
      if (srcNo >= fn->Blocks.size()) {
        errs() << "unexpected block number: " << srcNo << " (in "
               << fn->Blocks.size() << ")\n";
        return false;
      }
      GCOVBlock &Block = *fn->Blocks[srcNo];
      for (;;) {
        uint32_t line = buf.getWord();
        if (line)
          Block.addLine(line);
        else {
          StringRef filename = buf.getString();
          if (filename.empty())
            break;
          // TODO Unhandled
        }
      }
    }
  }

  GCNOInitialized = true;
  return true;
}

/// readGCDA - Read GCDA buffer. It is required that readGCDA() can only be
/// called after readGCNO().
bool GCOVFile::readGCDA(GCOVBuffer &buf) {
  assert(GCNOInitialized && "readGCDA() can only be called after readGCNO()");
  if (!buf.readGCDAFormat())
    return false;
  GCOV::GCOVVersion GCDAVersion;
  if (!buf.readGCOVVersion(GCDAVersion))
    return false;
  if (Version != GCDAVersion) {
    errs() << "GCOV versions do not match.\n";
    return false;
  }

  uint32_t GCDAChecksum;
  if (!buf.readInt(GCDAChecksum))
    return false;
  if (Checksum != GCDAChecksum) {
    errs() << "File checksums do not match: " << Checksum
           << " != " << GCDAChecksum << ".\n";
    return false;
  }
  uint32_t dummy, tag, length;
  uint32_t ident;
  GCOVFunction *fn = nullptr;
  while ((tag = buf.getWord())) {
    if (!buf.readInt(length))
      return false;
    uint32_t pos = buf.cursor.tell();
    if (tag == GCOV_TAG_OBJECT_SUMMARY) {
      buf.readInt(RunCount);
      buf.readInt(dummy);
    } else if (tag == GCOV_TAG_PROGRAM_SUMMARY) {
      buf.readInt(dummy);
      buf.readInt(dummy);
      buf.readInt(RunCount);
      ++ProgramCount;
    } else if (tag == GCOV_TAG_FUNCTION) {
      if (length == 0) // Placeholder
        continue;
      // As of GCC 10, GCOV_TAG_FUNCTION_LENGTH has never been larger than 3.
      if ((length != 2 && length != 3) || !buf.readInt(ident))
        return false;
      auto It = IdentToFunction.find(ident);
      uint32_t linenoChecksum, cfgChecksum = 0;
      buf.readInt(linenoChecksum);
      if (Version >= GCOV::V407)
        buf.readInt(cfgChecksum);
      if (It != IdentToFunction.end()) {
        fn = It->second;
        if (linenoChecksum != fn->linenoChecksum ||
            cfgChecksum != fn->cfgChecksum) {
          errs() << fn->Name
                 << format(": checksum mismatch, (%u, %u) != (%u, %u)\n",
                           linenoChecksum, cfgChecksum, fn->linenoChecksum,
                           fn->cfgChecksum);
          return false;
        }
      }
    } else if (tag == GCOV_TAG_COUNTER_ARCS && fn) {
      if (length != 2 * fn->arcs.size()) {
        errs() << fn->Name
               << format(
                      ": GCOV_TAG_COUNTER_ARCS mismatch, got %u, expected %u\n",
                      length, unsigned(2 * fn->arcs.size()));
        return false;
      }
      for (std::unique_ptr<GCOVArc> &arc : fn->arcs) {
        if (!buf.readInt64(arc->Count))
          return false;
        // FIXME Fix counters
        arc->src.Counter += arc->Count;
        if (arc->dst.succ.empty())
          arc->dst.Counter += arc->Count;
      }
    }
    pos += 4 * length;
    if (pos < buf.cursor.tell())
      return false;
    buf.de.skip(buf.cursor, pos - buf.cursor.tell());
  }

  return true;
}

void GCOVFile::print(raw_ostream &OS) const {
  for (const GCOVFunction &f : *this)
    f.print(OS);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// dump - Dump GCOVFile content to dbgs() for debugging purposes.
LLVM_DUMP_METHOD void GCOVFile::dump() const { print(dbgs()); }
#endif

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.
void GCOVFile::collectLineCounts(FileInfo &fi) {
  assert(fi.sources.empty());
  for (StringRef filename : filenames)
    fi.sources.emplace_back(filename);
  for (GCOVFunction &f : *this) {
    f.collectLineCounts(fi);
    fi.sources[f.srcIdx].functions.push_back(&f);
  }
  fi.setRunCount(RunCount);
  fi.setProgramCount(ProgramCount);
}

//===----------------------------------------------------------------------===//
// GCOVFunction implementation.

StringRef GCOVFunction::getFilename() const { return file.filenames[srcIdx]; }

/// getEntryCount - Get the number of times the function was called by
/// retrieving the entry block's count.
uint64_t GCOVFunction::getEntryCount() const {
  return Blocks.front()->getCount();
}

/// getExitCount - Get the number of times the function returned by retrieving
/// the exit block's count.
uint64_t GCOVFunction::getExitCount() const {
  return Blocks.back()->getCount();
}

void GCOVFunction::print(raw_ostream &OS) const {
  OS << "===== " << Name << " (" << ident << ") @ " << getFilename() << ":"
     << startLine << "\n";
  for (const auto &Block : Blocks)
    Block->print(OS);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// dump - Dump GCOVFunction content to dbgs() for debugging purposes.
LLVM_DUMP_METHOD void GCOVFunction::dump() const { print(dbgs()); }
#endif

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.
void GCOVFunction::collectLineCounts(FileInfo &FI) {
  // If the line number is zero, this is a function that doesn't actually appear
  // in the source file, so there isn't anything we can do with it.
  if (startLine == 0)
    return;

  for (const auto &Block : Blocks)
    Block->collectLineCounts(FI);
  FI.addFunctionLine(getFilename(), startLine, this);
}

//===----------------------------------------------------------------------===//
// GCOVBlock implementation.

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.
void GCOVBlock::collectLineCounts(FileInfo &FI) {
  for (uint32_t N : Lines)
    FI.addBlockLine(Parent.getFilename(), N, this);
}

void GCOVBlock::print(raw_ostream &OS) const {
  OS << "Block : " << Number << " Counter : " << Counter << "\n";
  if (!pred.empty()) {
    OS << "\tSource Edges : ";
    for (const GCOVArc *Edge : pred)
      OS << Edge->src.Number << " (" << Edge->Count << "), ";
    OS << "\n";
  }
  if (!succ.empty()) {
    OS << "\tDestination Edges : ";
    for (const GCOVArc *Edge : succ)
      OS << Edge->dst.Number << " (" << Edge->Count << "), ";
    OS << "\n";
  }
  if (!Lines.empty()) {
    OS << "\tLines : ";
    for (uint32_t N : Lines)
      OS << (N) << ",";
    OS << "\n";
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// dump - Dump GCOVBlock content to dbgs() for debugging purposes.
LLVM_DUMP_METHOD void GCOVBlock::dump() const { print(dbgs()); }
#endif

//===----------------------------------------------------------------------===//
// Cycles detection
//
// The algorithm in GCC is based on the algorithm by Hawick & James:
//   "Enumerating Circuits and Loops in Graphs with Self-Arcs and Multiple-Arcs"
//   http://complexity.massey.ac.nz/cstn/013/cstn-013.pdf.

/// Get the count for the detected cycle.
uint64_t GCOVBlock::getCycleCount(const Edges &Path) {
  uint64_t CycleCount = std::numeric_limits<uint64_t>::max();
  for (auto E : Path) {
    CycleCount = std::min(E->CyclesCount, CycleCount);
  }
  for (auto E : Path) {
    E->CyclesCount -= CycleCount;
  }
  return CycleCount;
}

/// Unblock a vertex previously marked as blocked.
void GCOVBlock::unblock(const GCOVBlock *U, BlockVector &Blocked,
                        BlockVectorLists &BlockLists) {
  auto it = find(Blocked, U);
  if (it == Blocked.end()) {
    return;
  }

  const size_t index = it - Blocked.begin();
  Blocked.erase(it);

  const BlockVector ToUnblock(BlockLists[index]);
  BlockLists.erase(BlockLists.begin() + index);
  for (auto GB : ToUnblock) {
    GCOVBlock::unblock(GB, Blocked, BlockLists);
  }
}

bool GCOVBlock::lookForCircuit(const GCOVBlock *V, const GCOVBlock *Start,
                               Edges &Path, BlockVector &Blocked,
                               BlockVectorLists &BlockLists,
                               const BlockVector &Blocks, uint64_t &Count) {
  Blocked.push_back(V);
  BlockLists.emplace_back(BlockVector());
  bool FoundCircuit = false;

  for (auto E : V->dsts()) {
    const GCOVBlock *W = &E->dst;
    if (W < Start || find(Blocks, W) == Blocks.end()) {
      continue;
    }

    Path.push_back(E);

    if (W == Start) {
      // We've a cycle.
      Count += GCOVBlock::getCycleCount(Path);
      FoundCircuit = true;
    } else if (find(Blocked, W) == Blocked.end() && // W is not blocked.
               GCOVBlock::lookForCircuit(W, Start, Path, Blocked, BlockLists,
                                         Blocks, Count)) {
      FoundCircuit = true;
    }

    Path.pop_back();
  }

  if (FoundCircuit) {
    GCOVBlock::unblock(V, Blocked, BlockLists);
  } else {
    for (auto E : V->dsts()) {
      const GCOVBlock *W = &E->dst;
      if (W < Start || find(Blocks, W) == Blocks.end()) {
        continue;
      }
      const size_t index = find(Blocked, W) - Blocked.begin();
      BlockVector &List = BlockLists[index];
      if (find(List, V) == List.end()) {
        List.push_back(V);
      }
    }
  }

  return FoundCircuit;
}

/// Get the count for the list of blocks which lie on the same line.
void GCOVBlock::getCyclesCount(const BlockVector &Blocks, uint64_t &Count) {
  for (auto Block : Blocks) {
    Edges Path;
    BlockVector Blocked;
    BlockVectorLists BlockLists;

    GCOVBlock::lookForCircuit(Block, Block, Path, Blocked, BlockLists, Blocks,
                              Count);
  }
}

/// Get the count for the list of blocks which lie on the same line.
uint64_t GCOVBlock::getLineCount(const BlockVector &Blocks) {
  uint64_t Count = 0;

  for (auto Block : Blocks) {
    if (Block->getNumSrcEdges() == 0) {
      // The block has no predecessors and a non-null counter
      // (can be the case with entry block in functions).
      Count += Block->getCount();
    } else {
      // Add counts from predecessors that are not on the same line.
      for (auto E : Block->srcs()) {
        const GCOVBlock *W = &E->src;
        if (find(Blocks, W) == Blocks.end()) {
          Count += E->Count;
        }
      }
    }
    for (auto E : Block->dsts()) {
      E->CyclesCount = E->Count;
    }
  }

  GCOVBlock::getCyclesCount(Blocks, Count);

  return Count;
}

//===----------------------------------------------------------------------===//
// FileInfo implementation.

// Safe integer division, returns 0 if numerator is 0.
static uint32_t safeDiv(uint64_t Numerator, uint64_t Divisor) {
  if (!Numerator)
    return 0;
  return Numerator / Divisor;
}

// This custom division function mimics gcov's branch ouputs:
//   - Round to closest whole number
//   - Only output 0% or 100% if it's exactly that value
static uint32_t branchDiv(uint64_t Numerator, uint64_t Divisor) {
  if (!Numerator)
    return 0;
  if (Numerator == Divisor)
    return 100;

  uint8_t Res = (Numerator * 100 + Divisor / 2) / Divisor;
  if (Res == 0)
    return 1;
  if (Res == 100)
    return 99;
  return Res;
}

namespace {
struct formatBranchInfo {
  formatBranchInfo(const GCOV::Options &Options, uint64_t Count, uint64_t Total)
      : Options(Options), Count(Count), Total(Total) {}

  void print(raw_ostream &OS) const {
    if (!Total)
      OS << "never executed";
    else if (Options.BranchCount)
      OS << "taken " << Count;
    else
      OS << "taken " << branchDiv(Count, Total) << "%";
  }

  const GCOV::Options &Options;
  uint64_t Count;
  uint64_t Total;
};

static raw_ostream &operator<<(raw_ostream &OS, const formatBranchInfo &FBI) {
  FBI.print(OS);
  return OS;
}

class LineConsumer {
  std::unique_ptr<MemoryBuffer> Buffer;
  StringRef Remaining;

public:
  LineConsumer(StringRef Filename) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
        MemoryBuffer::getFileOrSTDIN(Filename);
    if (std::error_code EC = BufferOrErr.getError()) {
      errs() << Filename << ": " << EC.message() << "\n";
      Remaining = "";
    } else {
      Buffer = std::move(BufferOrErr.get());
      Remaining = Buffer->getBuffer();
    }
  }
  bool empty() { return Remaining.empty(); }
  void printNext(raw_ostream &OS, uint32_t LineNum) {
    StringRef Line;
    if (empty())
      Line = "/*EOF*/";
    else
      std::tie(Line, Remaining) = Remaining.split("\n");
    OS << format("%5u:", LineNum) << Line << "\n";
  }
};
} // end anonymous namespace

/// Convert a path to a gcov filename. If PreservePaths is true, this
/// translates "/" to "#", ".." to "^", and drops ".", to match gcov.
static std::string mangleCoveragePath(StringRef Filename, bool PreservePaths) {
  if (!PreservePaths)
    return sys::path::filename(Filename).str();

  // This behaviour is defined by gcov in terms of text replacements, so it's
  // not likely to do anything useful on filesystems with different textual
  // conventions.
  llvm::SmallString<256> Result("");
  StringRef::iterator I, S, E;
  for (I = S = Filename.begin(), E = Filename.end(); I != E; ++I) {
    if (*I != '/')
      continue;

    if (I - S == 1 && *S == '.') {
      // ".", the current directory, is skipped.
    } else if (I - S == 2 && *S == '.' && *(S + 1) == '.') {
      // "..", the parent directory, is replaced with "^".
      Result.append("^#");
    } else {
      if (S < I)
        // Leave other components intact,
        Result.append(S, I);
      // And separate with "#".
      Result.push_back('#');
    }
    S = I + 1;
  }

  if (S < I)
    Result.append(S, I);
  return std::string(Result.str());
}

std::string FileInfo::getCoveragePath(StringRef Filename,
                                      StringRef MainFilename) {
  if (Options.NoOutput)
    // This is probably a bug in gcov, but when -n is specified, paths aren't
    // mangled at all, and the -l and -p options are ignored. Here, we do the
    // same.
    return std::string(Filename);

  std::string CoveragePath;
  if (Options.LongFileNames && !Filename.equals(MainFilename))
    CoveragePath =
        mangleCoveragePath(MainFilename, Options.PreservePaths) + "##";
  CoveragePath += mangleCoveragePath(Filename, Options.PreservePaths);
  if (Options.HashFilenames) {
    MD5 Hasher;
    MD5::MD5Result Result;
    Hasher.update(Filename.str());
    Hasher.final(Result);
    CoveragePath += "##" + std::string(Result.digest());
  }
  CoveragePath += ".gcov";
  return CoveragePath;
}

std::unique_ptr<raw_ostream>
FileInfo::openCoveragePath(StringRef CoveragePath) {
  std::error_code EC;
  auto OS =
      std::make_unique<raw_fd_ostream>(CoveragePath, EC, sys::fs::OF_Text);
  if (EC) {
    errs() << EC.message() << "\n";
    return std::make_unique<raw_null_ostream>();
  }
  return std::move(OS);
}

/// print -  Print source files with collected line count information.
void FileInfo::print(raw_ostream &InfoOS, StringRef MainFilename,
                     StringRef GCNOFile, StringRef GCDAFile, GCOVFile &file) {
  SmallVector<StringRef, 4> Filenames;
  for (const auto &LI : LineInfo)
    Filenames.push_back(LI.first());
  llvm::sort(Filenames);

  for (StringRef Filename : Filenames) {
    auto AllLines = LineConsumer(Filename);

    std::string CoveragePath = getCoveragePath(Filename, MainFilename);
    std::unique_ptr<raw_ostream> CovStream;
    if (Options.NoOutput)
      CovStream = std::make_unique<raw_null_ostream>();
    else if (!Options.UseStdout)
      CovStream = openCoveragePath(CoveragePath);
    raw_ostream &CovOS =
        !Options.NoOutput && Options.UseStdout ? llvm::outs() : *CovStream;

    CovOS << "        -:    0:Source:" << Filename << "\n";
    CovOS << "        -:    0:Graph:" << GCNOFile << "\n";
    CovOS << "        -:    0:Data:" << GCDAFile << "\n";
    CovOS << "        -:    0:Runs:" << RunCount << "\n";
    if (file.getVersion() < GCOV::V900)
      CovOS << "        -:    0:Programs:" << ProgramCount << "\n";

    const LineData &Line = LineInfo[Filename];
    GCOVCoverage FileCoverage(Filename);
    for (uint32_t LineIndex = 0; LineIndex < Line.LastLine || !AllLines.empty();
         ++LineIndex) {
      if (Options.BranchInfo) {
        FunctionLines::const_iterator FuncsIt = Line.Functions.find(LineIndex);
        if (FuncsIt != Line.Functions.end())
          printFunctionSummary(CovOS, FuncsIt->second);
      }

      BlockLines::const_iterator BlocksIt = Line.Blocks.find(LineIndex);
      if (BlocksIt == Line.Blocks.end()) {
        // No basic blocks are on this line. Not an executable line of code.
        CovOS << "        -:";
        AllLines.printNext(CovOS, LineIndex + 1);
      } else {
        const BlockVector &Blocks = BlocksIt->second;

        // Add up the block counts to form line counts.
        DenseMap<const GCOVFunction *, bool> LineExecs;
        for (const GCOVBlock *Block : Blocks) {
          if (Options.FuncCoverage) {
            // This is a slightly convoluted way to most accurately gather line
            // statistics for functions. Basically what is happening is that we
            // don't want to count a single line with multiple blocks more than
            // once. However, we also don't simply want to give the total line
            // count to every function that starts on the line. Thus, what is
            // happening here are two things:
            // 1) Ensure that the number of logical lines is only incremented
            //    once per function.
            // 2) If there are multiple blocks on the same line, ensure that the
            //    number of lines executed is incremented as long as at least
            //    one of the blocks are executed.
            const GCOVFunction *Function = &Block->getParent();
            if (FuncCoverages.find(Function) == FuncCoverages.end()) {
              std::pair<const GCOVFunction *, GCOVCoverage> KeyValue(
                  Function, GCOVCoverage(Function->getName()));
              FuncCoverages.insert(KeyValue);
            }
            GCOVCoverage &FuncCoverage = FuncCoverages.find(Function)->second;

            if (LineExecs.find(Function) == LineExecs.end()) {
              if (Block->getCount()) {
                ++FuncCoverage.LinesExec;
                LineExecs[Function] = true;
              } else {
                LineExecs[Function] = false;
              }
              ++FuncCoverage.LogicalLines;
            } else if (!LineExecs[Function] && Block->getCount()) {
              ++FuncCoverage.LinesExec;
              LineExecs[Function] = true;
            }
          }
        }

        const uint64_t LineCount = GCOVBlock::getLineCount(Blocks);
        if (LineCount == 0)
          CovOS << "    #####:";
        else {
          CovOS << format("%9" PRIu64 ":", LineCount);
          ++FileCoverage.LinesExec;
        }
        ++FileCoverage.LogicalLines;

        AllLines.printNext(CovOS, LineIndex + 1);

        uint32_t BlockNo = 0;
        uint32_t EdgeNo = 0;
        for (const GCOVBlock *Block : Blocks) {
          // Only print block and branch information at the end of the block.
          if (Block->getLastLine() != LineIndex + 1)
            continue;
          if (Options.AllBlocks)
            printBlockInfo(CovOS, *Block, LineIndex, BlockNo);
          if (Options.BranchInfo) {
            size_t NumEdges = Block->getNumDstEdges();
            if (NumEdges > 1)
              printBranchInfo(CovOS, *Block, FileCoverage, EdgeNo);
            else if (Options.UncondBranch && NumEdges == 1)
              printUncondBranchInfo(CovOS, EdgeNo, Block->succ[0]->Count);
          }
        }
      }
    }
    SourceInfo &source = sources[file.filenameToIdx.find(Filename)->second];
    source.name = CoveragePath;
    source.coverage = FileCoverage;
  }

  if (!Options.UseStdout) {
    // FIXME: There is no way to detect calls given current instrumentation.
    if (Options.FuncCoverage)
      printFuncCoverage(InfoOS);
    printFileCoverage(InfoOS);
  }
}

/// printFunctionSummary - Print function and block summary.
void FileInfo::printFunctionSummary(raw_ostream &OS,
                                    const FunctionVector &Funcs) const {
  for (const GCOVFunction *Func : Funcs) {
    uint64_t EntryCount = Func->getEntryCount();
    uint32_t BlocksExec = 0;
    for (const GCOVBlock &Block : Func->blocks())
      if (Block.getNumDstEdges() && Block.getCount())
        ++BlocksExec;

    OS << "function " << Func->getName() << " called " << EntryCount
       << " returned " << safeDiv(Func->getExitCount() * 100, EntryCount)
       << "% blocks executed "
       << safeDiv(BlocksExec * 100, Func->getNumBlocks() - 1) << "%\n";
  }
}

/// printBlockInfo - Output counts for each block.
void FileInfo::printBlockInfo(raw_ostream &OS, const GCOVBlock &Block,
                              uint32_t LineIndex, uint32_t &BlockNo) const {
  if (Block.getCount() == 0)
    OS << "    $$$$$:";
  else
    OS << format("%9" PRIu64 ":", Block.getCount());
  OS << format("%5u-block %2u\n", LineIndex + 1, BlockNo++);
}

/// printBranchInfo - Print conditional branch probabilities.
void FileInfo::printBranchInfo(raw_ostream &OS, const GCOVBlock &Block,
                               GCOVCoverage &Coverage, uint32_t &EdgeNo) {
  SmallVector<uint64_t, 16> BranchCounts;
  uint64_t TotalCounts = 0;
  for (const GCOVArc *Edge : Block.dsts()) {
    BranchCounts.push_back(Edge->Count);
    TotalCounts += Edge->Count;
    if (Block.getCount())
      ++Coverage.BranchesExec;
    if (Edge->Count)
      ++Coverage.BranchesTaken;
    ++Coverage.Branches;

    if (Options.FuncCoverage) {
      const GCOVFunction *Function = &Block.getParent();
      GCOVCoverage &FuncCoverage = FuncCoverages.find(Function)->second;
      if (Block.getCount())
        ++FuncCoverage.BranchesExec;
      if (Edge->Count)
        ++FuncCoverage.BranchesTaken;
      ++FuncCoverage.Branches;
    }
  }

  for (uint64_t N : BranchCounts)
    OS << format("branch %2u ", EdgeNo++)
       << formatBranchInfo(Options, N, TotalCounts) << "\n";
}

/// printUncondBranchInfo - Print unconditional branch probabilities.
void FileInfo::printUncondBranchInfo(raw_ostream &OS, uint32_t &EdgeNo,
                                     uint64_t Count) const {
  OS << format("unconditional %2u ", EdgeNo++)
     << formatBranchInfo(Options, Count, Count) << "\n";
}

// printCoverage - Print generic coverage info used by both printFuncCoverage
// and printFileCoverage.
void FileInfo::printCoverage(raw_ostream &OS,
                             const GCOVCoverage &Coverage) const {
  OS << format("Lines executed:%.2f%% of %u\n",
               double(Coverage.LinesExec) * 100 / Coverage.LogicalLines,
               Coverage.LogicalLines);
  if (Options.BranchInfo) {
    if (Coverage.Branches) {
      OS << format("Branches executed:%.2f%% of %u\n",
                   double(Coverage.BranchesExec) * 100 / Coverage.Branches,
                   Coverage.Branches);
      OS << format("Taken at least once:%.2f%% of %u\n",
                   double(Coverage.BranchesTaken) * 100 / Coverage.Branches,
                   Coverage.Branches);
    } else {
      OS << "No branches\n";
    }
    OS << "No calls\n"; // to be consistent with gcov
  }
}

// printFuncCoverage - Print per-function coverage info.
void FileInfo::printFuncCoverage(raw_ostream &OS) const {
  for (const auto &FC : FuncCoverages) {
    const GCOVCoverage &Coverage = FC.second;
    OS << "Function '" << Coverage.Name << "'\n";
    printCoverage(OS, Coverage);
    OS << "\n";
  }
}

// printFileCoverage - Print per-file coverage info.
void FileInfo::printFileCoverage(raw_ostream &OS) const {
  for (const SourceInfo &source : sources) {
    const GCOVCoverage &Coverage = source.coverage;
    OS << "File '" << Coverage.Name << "'\n";
    printCoverage(OS, Coverage);
    if (!Options.NoOutput)
      OS << "Creating '" << source.name << "'\n";
    OS << "\n";
  }
}
