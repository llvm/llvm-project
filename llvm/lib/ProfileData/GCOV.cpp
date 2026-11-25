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
#include "llvm/ADT/SmallSet.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <optional>
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

namespace {
struct Summary {
  Summary(StringRef Name) : Name(Name) {}

  StringRef Name;
  uint64_t lines = 0;
  uint64_t linesExec = 0;
  uint64_t branches = 0;
  uint64_t branchesExec = 0;
  uint64_t branchesTaken = 0;
};

struct LineInfo {
  SmallVector<const GCOVBlock *, 1> blocks;
  uint64_t count = 0;
  bool exists = false;
};

struct SourceInfo {
  StringRef filename;
  SmallString<0> displayName;
  std::vector<std::vector<const GCOVFunction *>> startLineToFunctions;
  std::vector<LineInfo> lines;
  bool ignored = false;
  SourceInfo(StringRef filename) : filename(filename) {}
};

class Context {
public:
  Context(const GCOV::Options &Options) : options(Options) {}
  void print(StringRef filename, StringRef gcno, StringRef gcda,
             GCOVFile &file);

private:
  std::string getCoveragePath(StringRef filename, StringRef mainFilename) const;
  void printFunctionDetails(const GCOVFunction &f, raw_ostream &os) const;
  void printBranchInfo(const GCOVBlock &Block, uint32_t &edgeIdx,
                       raw_ostream &OS) const;
  void printSummary(const Summary &summary, raw_ostream &os) const;

  void collectFunction(GCOVFunction &f, Summary &summary);
  void collectSourceLine(SourceInfo &si, Summary *summary, LineInfo &line,
                         size_t lineNum) const;
  void collectSource(SourceInfo &si, Summary &summary) const;
  void annotateSource(SourceInfo &si, const GCOVFile &file, StringRef gcno,
                      StringRef gcda, raw_ostream &os) const;
  void printSourceToIntermediate(const SourceInfo &si, raw_ostream &os) const;

  const GCOV::Options &options;
  std::vector<SourceInfo> sources;
};
} // namespace

//===----------------------------------------------------------------------===//
// GCOVFile implementation.

static Error makeGCOVError(GCOVBuffer &Buf, const Twine &Msg) {
  const std::error_code EC = make_error_code(errc::illegal_byte_sequence);
  if (Error E = Buf.cursor.takeError()) {
    std::string CursorErr = toString(std::move(E));
    return createStringError(EC, Msg + " (" + CursorErr + ")");
  }
  return createStringError(EC, Msg);
}

/// readGCNO - Read GCNO buffer.
Error GCOVFile::readGCNO(GCOVBuffer &buf) {
  if (!buf.readGCNOFormat())
    return makeGCOVError(buf, "invalid GCOV notes file magic");
  if (!buf.readGCOVVersion(version))
    return makeGCOVError(buf, "invalid or unsupported GCOV version");

  if (!buf.readInt(checksum))
    return makeGCOVError(buf, "truncated GCOV notes header");
  if (version >= GCOV::V900 && !buf.readString(cwd))
    return makeGCOVError(buf, "malformed working directory in GCOV header");
  uint32_t hasUnexecutedBlocks = 0;
  if (version >= GCOV::V800 && !buf.readInt(hasUnexecutedBlocks))
    return makeGCOVError(buf, "truncated GCOV notes header");

  uint32_t tag = 0, length = 0;
  GCOVFunction *fn = nullptr;
  uint64_t totalBlocks = 0;
  const uint64_t maxBlocks = uint64_t(buf.de.size()) + 1024;
  while (!buf.de.eof(buf.cursor)) {
    if (!buf.readInt(tag))
      return makeGCOVError(buf, "truncated GCOV record tag");
    if (tag == 0)
      break;
    if (!buf.readInt(length))
      return makeGCOVError(buf, "truncated GCOV record length");
    const uint64_t pos = buf.cursor.tell();
    const uint64_t byteLen =
        version >= GCOV::V1200 ? uint64_t(length) : uint64_t(length) * 4;
    if (byteLen > buf.de.size() - pos)
      return makeGCOVError(buf, "GCOV record length exceeds buffer size");
    const uint64_t endPos = pos + byteLen;
    if (tag == GCOV_TAG_FUNCTION) {
      fn = nullptr;
      auto newFn = std::make_unique<GCOVFunction>(*this);
      if (!buf.readInt(newFn->ident) || !buf.readInt(newFn->linenoChecksum) ||
          (version >= GCOV::V407 && !buf.readInt(newFn->cfgChecksum)) ||
          !buf.readString(newFn->Name, /*AllowEmpty=*/true))
        return makeGCOVError(buf, "malformed GCOV_TAG_FUNCTION record");
      StringRef filename;
      if (version < GCOV::V800) {
        if (!buf.readString(filename) || !buf.readInt(newFn->startLine))
          return makeGCOVError(buf, "malformed GCOV_TAG_FUNCTION record");
      } else {
        uint32_t artificial = 0;
        if (!buf.readInt(artificial) || !buf.readString(filename) ||
            !buf.readInt(newFn->startLine) ||
            !buf.readInt(newFn->startColumn) || !buf.readInt(newFn->endLine) ||
            (version >= GCOV::V900 && !buf.readInt(newFn->endColumn)))
          return makeGCOVError(buf, "malformed GCOV_TAG_FUNCTION record");
        newFn->artificial = artificial;
      }
      if (endPos < buf.cursor.tell())
        return makeGCOVError(buf, "GCOV_TAG_FUNCTION overran record length");
      newFn->srcIdx = addNormalizedPathToMap(filename);
      fn = newFn.get();
      identToFunction[fn->ident] = fn;
      functions.push_back(std::move(newFn));
    } else if (tag == GCOV_TAG_BLOCKS && fn) {
      if (!fn->blocks.empty())
        return makeGCOVError(buf, "duplicate GCOV_TAG_BLOCKS record");
      if (version < GCOV::V800) {
        if (length > maxBlocks - totalBlocks)
          return makeGCOVError(buf, "GCOV block count exceeds file limit");
        totalBlocks += length;
        fn->blocks.reserve(length);
        for (uint32_t i = 0; i != length; ++i) {
          uint32_t ignoredFlags = 0;
          if (!buf.readInt(ignoredFlags))
            return makeGCOVError(buf, "malformed GCOV_TAG_BLOCKS record");
          fn->blocks.push_back(std::make_unique<GCOVBlock>(i));
        }
      } else {
        uint32_t num = 0;
        if (byteLen < 4 || !buf.readInt(num))
          return makeGCOVError(buf, "malformed GCOV_TAG_BLOCKS record");
        if (num > maxBlocks - totalBlocks)
          return makeGCOVError(buf, "GCOV block count exceeds file limit");
        totalBlocks += num;
        fn->blocks.reserve(num);
        for (uint32_t i = 0; i != num; ++i)
          fn->blocks.push_back(std::make_unique<GCOVBlock>(i));
      }
    } else if (tag == GCOV_TAG_ARCS && fn) {
      uint32_t srcNo = 0;
      if (byteLen < 4 || !buf.readInt(srcNo))
        return makeGCOVError(buf, "malformed GCOV_TAG_ARCS record");
      if (srcNo >= fn->blocks.size())
        return makeGCOVError(buf, "unexpected block number: " + Twine(srcNo) +
                                      " (in " + Twine(fn->blocks.size()) + ")");
      GCOVBlock *src = fn->blocks[srcNo].get();
      const uint64_t arcsBytes = byteLen - 4;
      if (arcsBytes % 8 != 0)
        return makeGCOVError(buf, "malformed GCOV_TAG_ARCS record");
      const uint64_t e = arcsBytes / 8;
      for (uint64_t i = 0; i != e; ++i) {
        uint32_t dstNo = 0, flags = 0;
        if (!buf.readInt(dstNo) || !buf.readInt(flags))
          return makeGCOVError(buf, "malformed GCOV_TAG_ARCS record");
        if (dstNo >= fn->blocks.size())
          return makeGCOVError(buf, "unexpected block number: " + Twine(dstNo) +
                                        " (in " + Twine(fn->blocks.size()) +
                                        ")");
        GCOVBlock *dst = fn->blocks[dstNo].get();
        auto arc = std::make_unique<GCOVArc>(*src, *dst, flags);
        src->addDstEdge(arc.get());
        dst->addSrcEdge(arc.get());
        if (arc->onTree())
          fn->treeArcs.push_back(std::move(arc));
        else
          fn->arcs.push_back(std::move(arc));
      }
    } else if (tag == GCOV_TAG_LINES && fn) {
      uint32_t srcNo = 0;
      if (byteLen < 4 || !buf.readInt(srcNo))
        return makeGCOVError(buf, "malformed GCOV_TAG_LINES record");
      if (srcNo >= fn->blocks.size())
        return makeGCOVError(buf, "unexpected block number: " + Twine(srcNo) +
                                      " (in " + Twine(fn->blocks.size()) + ")");
      GCOVBlock &Block = *fn->blocks[srcNo];
      while (buf.cursor.tell() < endPos) {
        uint32_t line = 0;
        if (!buf.readInt(line))
          return makeGCOVError(buf, "malformed GCOV_TAG_LINES record");
        if (line != 0) {
          if (Block.locations.empty())
            return makeGCOVError(
                buf, "line number before initial filename in GCOV_TAG_LINES");
          Block.addLine(line);
        } else {
          if (buf.cursor.tell() >= endPos)
            return makeGCOVError(buf, "truncated GCOV_TAG_LINES filename");
          StringRef filename;
          if (!buf.readString(filename, /*AllowEmpty=*/true))
            return makeGCOVError(buf, "malformed GCOV_TAG_LINES filename");
          if (filename.empty())
            break;
          Block.addFile(addNormalizedPathToMap(filename));
        }
      }
    }
    if (!buf.cursor || endPos < buf.cursor.tell())
      return makeGCOVError(buf, "GCOV record overran declared length");
    if (uint64_t rem = endPos - buf.cursor.tell()) {
      buf.de.skip(buf.cursor, rem);
      if (!buf.cursor)
        return makeGCOVError(buf, "truncated GCOV record");
    }
  }

  GCNOInitialized = true;
  return Error::success();
}

/// readGCDA - Read GCDA buffer. It is required that readGCDA() can only be
/// called after readGCNO().
Error GCOVFile::readGCDA(GCOVBuffer &buf) {
  assert(GCNOInitialized && "readGCDA() can only be called after readGCNO()");
  if (!buf.readGCDAFormat())
    return makeGCOVError(buf, "invalid GCOV data file magic");
  GCOV::GCOVVersion GCDAVersion;
  if (!buf.readGCOVVersion(GCDAVersion))
    return makeGCOVError(buf, "invalid or unsupported GCOV version");
  if (version != GCDAVersion)
    return makeGCOVError(buf, "GCOV versions do not match");

  uint32_t GCDAChecksum = 0;
  if (!buf.readInt(GCDAChecksum))
    return makeGCOVError(buf, "truncated GCOV data header");
  if (checksum != GCDAChecksum)
    return makeGCOVError(buf,
                         "file checksums do not match: " + Twine(checksum) +
                             " != " + Twine(GCDAChecksum));

  uint32_t dummy = 0, tag = 0, length = 0;
  uint32_t ident = 0;
  GCOVFunction *fn = nullptr;
  while (!buf.de.eof(buf.cursor)) {
    if (!buf.readInt(tag))
      return makeGCOVError(buf, "truncated GCOV record tag");
    if (tag == 0)
      break;
    if (!buf.readInt(length))
      return makeGCOVError(buf, "truncated GCOV record length");
    const uint64_t pos = buf.cursor.tell();
    const uint64_t byteLen =
        version >= GCOV::V1200 ? uint64_t(length) : uint64_t(length) * 4;
    if (byteLen > buf.de.size() - pos)
      return makeGCOVError(buf, "GCOV record length exceeds buffer size");
    const uint64_t endPos = pos + byteLen;
    if (tag == GCOV_TAG_OBJECT_SUMMARY) {
      if (byteLen < 8 || !buf.readInt(runCount) || !buf.readInt(dummy))
        return makeGCOVError(buf, "malformed GCOV_TAG_OBJECT_SUMMARY record");
    } else if (tag == GCOV_TAG_PROGRAM_SUMMARY) {
      if (byteLen < 12 || !buf.readInt(dummy) || !buf.readInt(dummy) ||
          !buf.readInt(runCount))
        return makeGCOVError(buf, "malformed GCOV_TAG_PROGRAM_SUMMARY record");
      ++programCount;
    } else if (tag == GCOV_TAG_FUNCTION) {
      fn = nullptr;
      if (length == 0) // Placeholder
        continue;
      const uint64_t minFuncBytes = version >= GCOV::V407 ? 12 : 8;
      uint32_t linenoChecksum = 0, cfgChecksum = 0;
      if (byteLen < minFuncBytes || !buf.readInt(ident) ||
          !buf.readInt(linenoChecksum) ||
          (version >= GCOV::V407 && !buf.readInt(cfgChecksum)))
        return makeGCOVError(buf, "malformed GCOV_TAG_FUNCTION record");
      auto It = identToFunction.find(ident);
      if (It != identToFunction.end()) {
        fn = It->second;
        if (linenoChecksum != fn->linenoChecksum ||
            cfgChecksum != fn->cfgChecksum)
          return makeGCOVError(buf, fn->Name + ": checksum mismatch, (" +
                                        Twine(linenoChecksum) + ", " +
                                        Twine(cfgChecksum) + ") != (" +
                                        Twine(fn->linenoChecksum) + ", " +
                                        Twine(fn->cfgChecksum) + ")");
      }
    } else if (tag == GCOV_TAG_COUNTER_ARCS && fn) {
      const uint64_t expectedBytes = uint64_t(fn->arcs.size()) * 8;
      if (byteLen != expectedBytes) {
        const uint64_t expectedLength =
            version >= GCOV::V1200 ? expectedBytes : expectedBytes / 4;
        return makeGCOVError(
            buf, fn->Name + ": GCOV_TAG_COUNTER_ARCS mismatch, got " +
                     Twine(length) + ", expected " + Twine(expectedLength));
      }
      for (std::unique_ptr<GCOVArc> &arc : fn->arcs) {
        if (!buf.readInt64(arc->count))
          return makeGCOVError(buf, "truncated GCOV_TAG_COUNTER_ARCS record");
        arc->src.count += arc->count;
      }

      if (fn->blocks.size() >= 2) {
        GCOVBlock &src = *fn->blocks[0];
        GCOVBlock &sink =
            version < GCOV::V408 ? *fn->blocks.back() : *fn->blocks[1];
        auto arc = std::make_unique<GCOVArc>(sink, src, GCOV_ARC_ON_TREE);
        sink.addDstEdge(arc.get());
        src.addSrcEdge(arc.get());
        fn->treeArcs.push_back(std::move(arc));

        for (GCOVBlock &block : fn->blocksRange())
          fn->propagateCounts(block, nullptr);
        for (size_t i = fn->treeArcs.size() - 1; i; --i)
          fn->treeArcs[i - 1]->src.count += fn->treeArcs[i - 1]->count;
      }
      fn = nullptr;
    }
    if (!buf.cursor || endPos < buf.cursor.tell())
      return makeGCOVError(buf, "GCOV record overran declared length");
    if (uint64_t rem = endPos - buf.cursor.tell()) {
      buf.de.skip(buf.cursor, rem);
      if (!buf.cursor)
        return makeGCOVError(buf, "truncated GCOV record");
    }
  }

  return Error::success();
}

void GCOVFile::print(raw_ostream &OS) const {
  for (const GCOVFunction &f : *this)
    f.print(OS);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// dump - Dump GCOVFile content to dbgs() for debugging purposes.
LLVM_DUMP_METHOD void GCOVFile::dump() const { print(dbgs()); }
#endif

unsigned GCOVFile::addNormalizedPathToMap(StringRef filename) {
  // unify filename, as the same path can have different form
  SmallString<256> P(filename);
  sys::path::remove_dots(P, true);
  filename = P.str();

  auto r = filenameToIdx.try_emplace(filename, filenameToIdx.size());
  if (r.second)
    filenames.emplace_back(filename);

  return r.first->second;
}

bool GCOVArc::onTree() const { return flags & GCOV_ARC_ON_TREE; }

//===----------------------------------------------------------------------===//
// GCOVFunction implementation.

StringRef GCOVFunction::getName(bool demangle) const {
  if (!demangle)
    return Name;
  if (demangled.empty()) {
    do {
      if (Name.starts_with("_Z")) {
        if (char *res = itaniumDemangle(Name)) {
          demangled = res;
          free(res);
          break;
        }
      }
      demangled = Name;
    } while (false);
  }
  return demangled;
}
StringRef GCOVFunction::getFilename() const { return file.filenames[srcIdx]; }

/// getEntryCount - Get the number of times the function was called by
/// retrieving the entry block's count.
uint64_t GCOVFunction::getEntryCount() const {
  return blocks.front()->getCount();
}

GCOVBlock &GCOVFunction::getExitBlock() const {
  return file.getVersion() < GCOV::V408 ? *blocks.back() : *blocks[1];
}

// For each basic block, the sum of incoming edge counts equals the sum of
// outgoing edge counts by Kirchoff's circuit law. If the unmeasured arcs form a
// spanning tree, the count for each unmeasured arc (GCOV_ARC_ON_TREE) can be
// uniquely identified. Use an iterative algorithm to decrease stack usage for
// library users in threads. See the edge propagation algorithm in Optimally
// Profiling and Tracing Programs, ACM Transactions on Programming Languages and
// Systems, 1994.
void GCOVFunction::propagateCounts(const GCOVBlock &v, GCOVArc *pred) {
  struct Elem {
    const GCOVBlock &v;
    GCOVArc *pred;
    bool inDst;
    size_t i = 0;
    uint64_t excess = 0;
  };

  SmallVector<Elem, 0> stack;
  stack.push_back({v, pred, false});
  for (;;) {
    Elem &u = stack.back();
    // If GCOV_ARC_ON_TREE edges do form a tree, visited is not needed;
    // otherwise, this prevents infinite recursion for bad input.
    if (u.i == 0 && !visited.insert(&u.v).second) {
      stack.pop_back();
      if (stack.empty())
        break;
      continue;
    }
    if (u.i < u.v.pred.size()) {
      GCOVArc *e = u.v.pred[u.i++];
      if (e != u.pred) {
        if (e->onTree())
          stack.push_back({e->src, e, /*inDst=*/false});
        else
          u.excess += e->count;
      }
    } else if (u.i < u.v.pred.size() + u.v.succ.size()) {
      GCOVArc *e = u.v.succ[u.i++ - u.v.pred.size()];
      if (e != u.pred) {
        if (e->onTree())
          stack.push_back({e->dst, e, /*inDst=*/true});
        else
          u.excess -= e->count;
      }
    } else {
      uint64_t excess = u.excess;
      if (static_cast<int64_t>(excess) < 0)
        excess = -excess;
      if (u.pred)
        u.pred->count = excess;
      bool inDst = u.inDst;
      stack.pop_back();
      if (stack.empty())
        break;
      stack.back().excess += inDst ? -excess : excess;
    }
  }
}

void GCOVFunction::print(raw_ostream &OS) const {
  OS << "===== " << Name << " (" << ident << ") @ " << getFilename() << ":"
     << startLine << "\n";
  for (const auto &Block : blocks)
    Block->print(OS);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// dump - Dump GCOVFunction content to dbgs() for debugging purposes.
LLVM_DUMP_METHOD void GCOVFunction::dump() const { print(dbgs()); }
#endif

/// collectLineCounts - Collect line counts. This must be used after
/// reading .gcno and .gcda files.

//===----------------------------------------------------------------------===//
// GCOVBlock implementation.

void GCOVBlock::print(raw_ostream &OS) const {
  OS << "Block : " << number << " Counter : " << count << "\n";
  if (!pred.empty()) {
    OS << "\tSource Edges : ";
    for (const GCOVArc *Edge : pred)
      OS << Edge->src.number << " (" << Edge->count << "), ";
    OS << "\n";
  }
  if (!succ.empty()) {
    OS << "\tDestination Edges : ";
    for (const GCOVArc *Edge : succ) {
      if (Edge->flags & GCOV_ARC_ON_TREE)
        OS << '*';
      OS << Edge->dst.number << " (" << Edge->count << "), ";
    }
    OS << "\n";
  }
  if (!locations.empty()) {
    for (const GCOVBlockLocation &loc : locations) {
      OS << "\tFile: " << loc.srcIdx << ": ";
      for (uint32_t N : loc.lines)
        OS << (N) << ",";
      OS << "\n";
    }
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// dump - Dump GCOVBlock content to dbgs() for debugging purposes.
LLVM_DUMP_METHOD void GCOVBlock::dump() const { print(dbgs()); }
#endif

uint64_t
GCOVBlock::augmentOneCycle(GCOVBlock *src,
                           std::vector<std::pair<GCOVBlock *, size_t>> &stack) {
  GCOVBlock *u;
  size_t i;
  stack.clear();
  stack.emplace_back(src, 0);
  src->incoming = (GCOVArc *)1; // Mark u available for cycle detection
  for (;;) {
    std::tie(u, i) = stack.back();
    if (i == u->succ.size()) {
      u->traversable = false;
      stack.pop_back();
      if (stack.empty())
        break;
      continue;
    }
    ++stack.back().second;
    GCOVArc *succ = u->succ[i];
    // Ignore saturated arcs (cycleCount has been reduced to 0) and visited
    // blocks. Ignore self arcs to guard against bad input (.gcno has no
    // self arcs).
    if (succ->cycleCount == 0 || !succ->dst.traversable || &succ->dst == u)
      continue;
    if (succ->dst.incoming == nullptr) {
      succ->dst.incoming = succ;
      stack.emplace_back(&succ->dst, 0);
      continue;
    }
    uint64_t minCount = succ->cycleCount;
    for (GCOVBlock *v = u;;) {
      minCount = std::min(minCount, v->incoming->cycleCount);
      v = &v->incoming->src;
      if (v == &succ->dst)
        break;
    }
    succ->cycleCount -= minCount;
    for (GCOVBlock *v = u;;) {
      v->incoming->cycleCount -= minCount;
      v = &v->incoming->src;
      if (v == &succ->dst)
        break;
    }
    return minCount;
  }
  return 0;
}

// Get the total execution count of loops among blocks on the same line.
// Assuming a reducible flow graph, the count is the sum of back edge counts.
// Identifying loops is complex, so we simply find cycles and perform cycle
// cancelling iteratively.
uint64_t GCOVBlock::getCyclesCount(const BlockVector &blocks) {
  std::vector<std::pair<GCOVBlock *, size_t>> stack;
  uint64_t count = 0, d;
  for (;;) {
    // Make blocks on the line traversable and try finding a cycle.
    for (const auto *b : blocks) {
      const_cast<GCOVBlock *>(b)->traversable = true;
      const_cast<GCOVBlock *>(b)->incoming = nullptr;
    }
    d = 0;
    for (const auto *block : blocks) {
      auto *b = const_cast<GCOVBlock *>(block);
      if (b->traversable && (d = augmentOneCycle(b, stack)) > 0)
        break;
    }
    if (d == 0)
      break;
    count += d;
  }
  // If there is no more loop, all traversable bits should have been cleared.
  // This property is needed by subsequent calls.
  for (const auto *b : blocks) {
    assert(!b->traversable);
    (void)b;
  }
  return count;
}

//===----------------------------------------------------------------------===//
// FileInfo implementation.

// Format dividend/divisor as a percentage. Return 1 if the result is greater
// than 0% and less than 1%.
static uint32_t formatPercentage(uint64_t dividend, uint64_t divisor) {
  if (!dividend || !divisor)
    return 0;
  dividend *= 100;
  return dividend < divisor ? 1 : dividend / divisor;
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
  LineConsumer() = default;
  LineConsumer(StringRef Filename) {
    // Open source files without requiring a NUL terminator. The concurrent
    // modification may nullify the NUL terminator condition.
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
        MemoryBuffer::getFileOrSTDIN(Filename, /*IsText=*/false,
                                     /*RequiresNullTerminator=*/false);
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
  return std::string(Result);
}

std::string Context::getCoveragePath(StringRef filename,
                                     StringRef mainFilename) const {
  if (options.NoOutput)
    // This is probably a bug in gcov, but when -n is specified, paths aren't
    // mangled at all, and the -l and -p options are ignored. Here, we do the
    // same.
    return std::string(filename);

  std::string CoveragePath;
  if (options.LongFileNames && filename != mainFilename)
    CoveragePath =
        mangleCoveragePath(mainFilename, options.PreservePaths) + "##";
  CoveragePath += mangleCoveragePath(filename, options.PreservePaths);
  if (options.HashFilenames) {
    MD5 Hasher;
    MD5::MD5Result Result;
    Hasher.update(filename.str());
    Hasher.final(Result);
    CoveragePath += "##" + std::string(Result.digest());
  }
  CoveragePath += ".gcov";
  return CoveragePath;
}

void Context::collectFunction(GCOVFunction &f, Summary &summary) {
  SourceInfo &si = sources[f.srcIdx];
  if (f.startLine >= si.startLineToFunctions.size())
    si.startLineToFunctions.resize(f.startLine + 1);
  si.startLineToFunctions[f.startLine].push_back(&f);
  SmallSet<uint32_t, 16> lines;
  SmallSet<uint32_t, 16> linesExec;
  for (const GCOVBlock &b : f.blocksRange()) {
    if (b.locations.empty())
      continue;
    for (const GCOVBlockLocation &loc : b.locations) {
      SourceInfo &locSource = sources[loc.srcIdx];
      uint32_t maxLineNum = *llvm::max_element(loc.lines);
      if (maxLineNum >= locSource.lines.size())
        locSource.lines.resize(maxLineNum + 1);
      for (uint32_t lineNum : loc.lines) {
        LineInfo &line = locSource.lines[lineNum];
        line.exists = true;
        line.count += b.count;
        line.blocks.push_back(&b);
        if (f.srcIdx == loc.srcIdx) {
          if (lines.insert(lineNum).second)
            ++summary.lines;
          if (b.count && linesExec.insert(lineNum).second)
            ++summary.linesExec;
        }
      }
    }
  }
}

void Context::collectSourceLine(SourceInfo &si, Summary *summary,
                                LineInfo &line, size_t lineNum) const {
  uint64_t count = 0;
  for (const GCOVBlock *b : line.blocks) {
    if (b->number == 0) {
      // For nonstandard control flows, arcs into the exit block may be
      // duplicately counted (fork) or not be counted (abnormal exit), and thus
      // the (exit,entry) counter may be inaccurate. Count the entry block with
      // the outgoing arcs.
      for (const GCOVArc *arc : b->succ)
        count += arc->count;
    } else {
      // Add counts from predecessors that are not on the same line.
      for (const GCOVArc *arc : b->pred)
        if (!llvm::is_contained(line.blocks, &arc->src))
          count += arc->count;
    }
    for (GCOVArc *arc : b->succ)
      arc->cycleCount = arc->count;
  }

  count += GCOVBlock::getCyclesCount(line.blocks);
  line.count = count;
  if (line.exists) {
    ++summary->lines;
    if (line.count != 0)
      ++summary->linesExec;
  }

  if (options.BranchInfo)
    for (const GCOVBlock *b : line.blocks) {
      if (b->getLastLine() != lineNum)
        continue;
      int branches = 0, execBranches = 0, takenBranches = 0;
      for (const GCOVArc *arc : b->succ) {
        ++branches;
        if (count != 0)
          ++execBranches;
        if (arc->count != 0)
          ++takenBranches;
      }
      if (branches > 1) {
        summary->branches += branches;
        summary->branchesExec += execBranches;
        summary->branchesTaken += takenBranches;
      }
    }
}

void Context::collectSource(SourceInfo &si, Summary &summary) const {
  size_t lineNum = 0;
  for (LineInfo &line : si.lines) {
    collectSourceLine(si, &summary, line, lineNum);
    ++lineNum;
  }
}

void Context::annotateSource(SourceInfo &si, const GCOVFile &file,
                             StringRef gcno, StringRef gcda,
                             raw_ostream &os) const {
  auto source =
      options.Intermediate ? LineConsumer() : LineConsumer(si.filename);

  os << "        -:    0:Source:" << si.displayName << '\n';
  os << "        -:    0:Graph:" << gcno << '\n';
  os << "        -:    0:Data:" << gcda << '\n';
  os << "        -:    0:Runs:" << file.runCount << '\n';
  if (file.version < GCOV::V900)
    os << "        -:    0:Programs:" << file.programCount << '\n';

  for (size_t lineNum = 1; !source.empty(); ++lineNum) {
    if (lineNum >= si.lines.size()) {
      os << "        -:";
      source.printNext(os, lineNum);
      continue;
    }

    const LineInfo &line = si.lines[lineNum];
    if (options.BranchInfo && lineNum < si.startLineToFunctions.size())
      for (const auto *f : si.startLineToFunctions[lineNum])
        printFunctionDetails(*f, os);
    if (!line.exists)
      os << "        -:";
    else if (line.count == 0)
      os << "    #####:";
    else
      os << format("%9" PRIu64 ":", line.count);
    source.printNext(os, lineNum);

    uint32_t blockIdx = 0, edgeIdx = 0;
    for (const GCOVBlock *b : line.blocks) {
      if (b->getLastLine() != lineNum)
        continue;
      if (options.AllBlocks) {
        if (b->getCount() == 0)
          os << "    $$$$$:";
        else
          os << format("%9" PRIu64 ":", b->count);
        os << format("%5u-block %2u\n", lineNum, blockIdx++);
      }
      if (options.BranchInfo) {
        size_t NumEdges = b->succ.size();
        if (NumEdges > 1)
          printBranchInfo(*b, edgeIdx, os);
        else if (options.UncondBranch && NumEdges == 1) {
          uint64_t count = b->succ[0]->count;
          os << format("unconditional %2u ", edgeIdx++)
             << formatBranchInfo(options, count, count) << '\n';
        }
      }
    }
  }
}

void Context::printSourceToIntermediate(const SourceInfo &si,
                                        raw_ostream &os) const {
  os << "file:" << si.filename << '\n';
  for (const auto &fs : si.startLineToFunctions)
    for (const GCOVFunction *f : fs)
      os << "function:" << f->startLine << ',' << f->getEntryCount() << ','
         << f->getName(options.Demangle) << '\n';
  for (size_t lineNum = 1, size = si.lines.size(); lineNum < size; ++lineNum) {
    const LineInfo &line = si.lines[lineNum];
    if (line.blocks.empty())
      continue;
    // GCC 8 (r254259) added third third field for Ada:
    // lcount:<line>,<count>,<has_unexecuted_blocks>
    // We don't need the third field.
    os << "lcount:" << lineNum << ',' << line.count << '\n';

    if (!options.BranchInfo)
      continue;
    for (const GCOVBlock *b : line.blocks) {
      if (b->succ.size() < 2 || b->getLastLine() != lineNum)
        continue;
      for (const GCOVArc *arc : b->succ) {
        const char *type =
            b->getCount() ? arc->count ? "taken" : "nottaken" : "notexec";
        os << "branch:" << lineNum << ',' << type << '\n';
      }
    }
  }
}

void Context::print(StringRef filename, StringRef gcno, StringRef gcda,
                    GCOVFile &file) {
  for (StringRef filename : file.filenames) {
    sources.emplace_back(filename);
    SourceInfo &si = sources.back();
    si.displayName = si.filename;
    if (!options.SourcePrefix.empty() &&
        sys::path::replace_path_prefix(si.displayName, options.SourcePrefix,
                                       "") &&
        !si.displayName.empty()) {
      // TODO replace_path_prefix may strip the prefix even if the remaining
      // part does not start with a separator.
      if (sys::path::is_separator(si.displayName[0]))
        si.displayName.erase(si.displayName.begin());
      else
        si.displayName = si.filename;
    }
    if (options.RelativeOnly && sys::path::is_absolute(si.displayName))
      si.ignored = true;
  }

  raw_ostream &os = llvm::outs();
  for (GCOVFunction &f : make_pointee_range(file.functions)) {
    Summary summary(f.getName(options.Demangle));
    collectFunction(f, summary);
    if (options.FuncCoverage && !options.UseStdout) {
      os << "Function '" << summary.Name << "'\n";
      printSummary(summary, os);
      os << '\n';
    }
  }

  for (SourceInfo &si : sources) {
    if (si.ignored)
      continue;
    Summary summary(si.displayName);
    collectSource(si, summary);

    // Print file summary unless -t is specified.
    std::string gcovName = getCoveragePath(si.filename, filename);
    if (!options.UseStdout) {
      os << "File '" << summary.Name << "'\n";
      printSummary(summary, os);
      if (!options.NoOutput && !options.Intermediate)
        os << "Creating '" << gcovName << "'\n";
      os << '\n';
    }

    if (options.NoOutput || options.Intermediate)
      continue;
    std::optional<raw_fd_ostream> os;
    if (!options.UseStdout) {
      std::error_code ec;
      os.emplace(gcovName, ec, sys::fs::OF_TextWithCRLF);
      if (ec) {
        errs() << ec.message() << '\n';
        continue;
      }
    }
    annotateSource(si, file, gcno, gcda,
                   options.UseStdout ? llvm::outs() : *os);
  }

  if (options.Intermediate && !options.NoOutput) {
    // gcov 7.* unexpectedly create multiple .gcov files, which was fixed in 8.0
    // (PR GCC/82702). We create just one file.
    std::string outputPath(sys::path::filename(filename));
    std::error_code ec;
    raw_fd_ostream os(outputPath + ".gcov", ec, sys::fs::OF_TextWithCRLF);
    if (ec) {
      errs() << ec.message() << '\n';
      return;
    }

    for (const SourceInfo &si : sources)
      printSourceToIntermediate(si, os);
  }
}

void Context::printFunctionDetails(const GCOVFunction &f,
                                   raw_ostream &os) const {
  const uint64_t entryCount = f.getEntryCount();
  uint32_t blocksExec = 0;
  const GCOVBlock &exitBlock = f.getExitBlock();
  uint64_t exitCount = 0;
  for (const GCOVArc *arc : exitBlock.pred)
    exitCount += arc->count;
  for (const GCOVBlock &b : f.blocksRange())
    if (b.number != 0 && &b != &exitBlock && b.getCount())
      ++blocksExec;

  os << "function " << f.getName(options.Demangle) << " called " << entryCount
     << " returned " << formatPercentage(exitCount, entryCount)
     << "% blocks executed "
     << formatPercentage(blocksExec, f.blocks.size() - 2) << "%\n";
}

/// printBranchInfo - Print conditional branch probabilities.
void Context::printBranchInfo(const GCOVBlock &Block, uint32_t &edgeIdx,
                              raw_ostream &os) const {
  uint64_t total = 0;
  for (const GCOVArc *arc : Block.dsts())
    total += arc->count;
  for (const GCOVArc *arc : Block.dsts())
    os << format("branch %2u ", edgeIdx++)
       << formatBranchInfo(options, arc->count, total) << '\n';
}

void Context::printSummary(const Summary &summary, raw_ostream &os) const {
  os << format("Lines executed:%.2f%% of %" PRIu64 "\n",
               double(summary.linesExec) * 100 / summary.lines, summary.lines);
  if (options.BranchInfo) {
    if (summary.branches == 0) {
      os << "No branches\n";
    } else {
      os << format("Branches executed:%.2f%% of %" PRIu64 "\n",
                   double(summary.branchesExec) * 100 / summary.branches,
                   summary.branches);
      os << format("Taken at least once:%.2f%% of %" PRIu64 "\n",
                   double(summary.branchesTaken) * 100 / summary.branches,
                   summary.branches);
    }
    os << "No calls\n";
  }
}

void llvm::gcovOneInput(const GCOV::Options &options, StringRef filename,
                        StringRef gcno, StringRef gcda, GCOVFile &file) {
  Context fi(options);
  fi.print(filename, gcno, gcda, file);
}
