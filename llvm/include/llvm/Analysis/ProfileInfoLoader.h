//===- ProfileInfoLoader.h - Load & convert profile information -*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The ProfileInfoLoader class is used to load and represent profiling
// information read in from the dump file.  If conversions between formats are
// needed, it can also do this.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_PROFILEINFOLOADER_H
#define LLVM_ANALYSIS_PROFILEINFOLOADER_H

#include <vector>
#include <string>
#include <utility>

namespace llvm {

class Module;
class Function;
class BasicBlock;

class ProfileInfoLoader {
  Module &M;
  std::vector<std::string> CommandLines;
  std::vector<unsigned>    FunctionCounts;
  std::vector<unsigned>    BlockCounts;
  std::vector<unsigned>    EdgeCounts;
  std::vector<unsigned>    BBTrace;
public:
  // ProfileInfoLoader ctor - Read the specified profiling data file, exiting
  // the program if the file is invalid or broken.
  ProfileInfoLoader(const char *ToolName, const std::string &Filename,
                    Module &M);

  unsigned getNumExecutions() const { return CommandLines.size(); }
  const std::string &getExecution(unsigned i) const { return CommandLines[i]; }

  // getFunctionCounts - This method is used by consumers of function counting
  // information.  If we do not directly have function count information, we
  // compute it from other, more refined, types of profile information.
  //
  void getFunctionCounts(std::vector<std::pair<Function*, unsigned> > &Counts);

  // hasAccurateBlockCounts - Return true if we can synthesize accurate block
  // frequency information from whatever we have.
  //
  bool hasAccurateBlockCounts() const {
    return !BlockCounts.empty() || !EdgeCounts.empty();
  }

  // hasAccurateEdgeCounts - Return true if we can synthesize accurate edge
  // frequency information from whatever we have.
  //
  bool hasAccurateEdgeCounts() const {
    return !EdgeCounts.empty();
  }

  // getBlockCounts - This method is used by consumers of block counting
  // information.  If we do not directly have block count information, we
  // compute it from other, more refined, types of profile information.
  //
  void getBlockCounts(std::vector<std::pair<BasicBlock*, unsigned> > &Counts);

  // getEdgeCounts - This method is used by consumers of edge counting
  // information.  If we do not directly have edge count information, we compute
  // it from other, more refined, types of profile information.
  //
  // Edges are represented as a pair, where the first element is the basic block
  // and the second element is the successor number.
  //
  typedef std::pair<BasicBlock*, unsigned> Edge;
  void getEdgeCounts(std::vector<std::pair<Edge, unsigned> > &Counts);

  // getBBTrace - This method is used by consumers of basic-block trace
  // information.
  //
  void getBBTrace(std::vector<BasicBlock *> &Trace);
};

} // End llvm namespace

#endif
