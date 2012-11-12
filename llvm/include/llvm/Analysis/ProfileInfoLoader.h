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
  const std::string &Filename;
  std::vector<std::string> CommandLines;
  std::vector<unsigned>    FunctionCounts;
  std::vector<unsigned>    BlockCounts;
  std::vector<unsigned>    EdgeCounts;
  std::vector<unsigned>    OptimalEdgeCounts;
  std::vector<unsigned>    BBTrace;
public:
  // ProfileInfoLoader ctor - Read the specified profiling data file, exiting
  // the program if the file is invalid or broken.
  ProfileInfoLoader(const char *ToolName, const std::string &Filename);

  static const unsigned Uncounted;

  unsigned getNumExecutions() const { return CommandLines.size(); }
  const std::string &getExecution(unsigned i) const { return CommandLines[i]; }

  const std::string &getFileName() const { return Filename; }

  // getRawFunctionCounts - This method is used by consumers of function
  // counting information.
  //
  const std::vector<unsigned> &getRawFunctionCounts() const {
    return FunctionCounts;
  }

  // getRawBlockCounts - This method is used by consumers of block counting
  // information.
  //
  const std::vector<unsigned> &getRawBlockCounts() const {
    return BlockCounts;
  }

  // getEdgeCounts - This method is used by consumers of edge counting
  // information.
  //
  const std::vector<unsigned> &getRawEdgeCounts() const {
    return EdgeCounts;
  }

  // getEdgeOptimalCounts - This method is used by consumers of optimal edge 
  // counting information.
  //
  const std::vector<unsigned> &getRawOptimalEdgeCounts() const {
    return OptimalEdgeCounts;
  }

};

} // End llvm namespace

#endif
