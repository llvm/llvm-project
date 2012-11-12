//===- ProfileDataLoader.h - Load & convert profile info ----*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The ProfileDataLoader class is used to load profiling data from a dump file.
// The ProfileDataT<FType, BType> class is used to store the mapping of this
// data to control flow edges.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_PROFILEDATALOADER_H
#define LLVM_ANALYSIS_PROFILEDATALOADER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <string>

namespace llvm {

class ModulePass;
class Function;
class BasicBlock;

// Helper for dumping edges to dbgs().
raw_ostream& operator<<(raw_ostream &O, std::pair<const BasicBlock *,
                                                  const BasicBlock *> E);

/// \brief The ProfileDataT<FType, BType> class is used to store the mapping of
/// profiling data to control flow edges.
///
/// An edge is defined by its source and sink basic blocks.
template<class FType, class BType>
class ProfileDataT {
public:
  // The profiling information defines an Edge by its source and sink basic
  // blocks.
  typedef std::pair<const BType*, const BType*> Edge;

private:
  typedef DenseMap<Edge, unsigned> EdgeWeights;

  /// \brief Count the number of times a transition between two blocks is
  /// executed.
  ///
  /// As a special case, we also hold an edge from the null BasicBlock to the
  /// entry block to indicate how many times the function was entered.
  DenseMap<const FType*, EdgeWeights> EdgeInformation;

public:
  /// getFunction() - Returns the Function for an Edge.
  static const FType *getFunction(Edge e) {
    // e.first may be NULL
    assert(((!e.first) || (e.first->getParent() == e.second->getParent()))
           && "A ProfileData::Edge can not be between two functions");
    assert(e.second && "A ProfileData::Edge must have a real sink");
    return e.second->getParent();
  }

  /// getEdge() - Creates an Edge between two BasicBlocks.
  static Edge getEdge(const BType *Src, const BType *Dest) {
    return Edge(Src, Dest);
  }

  /// getEdgeWeight - Return the number of times that a given edge was
  /// executed.
  unsigned getEdgeWeight(Edge e) const {
    const FType *f = getFunction(e);
    assert((EdgeInformation.find(f) != EdgeInformation.end())
           && "No profiling information for function");
    EdgeWeights weights = EdgeInformation.find(f)->second;

    assert((weights.find(e) != weights.end())
           && "No profiling information for edge");
    return weights.find(e)->second;
  }

  /// addEdgeWeight - Add 'weight' to the already stored execution count for
  /// this edge.
  void addEdgeWeight(Edge e, unsigned weight) {
    EdgeInformation[getFunction(e)][e] += weight;
  }
};

typedef ProfileDataT<Function, BasicBlock> ProfileData;
//typedef ProfileDataT<MachineFunction, MachineBasicBlock> MachineProfileData;

/// The ProfileDataLoader class is used to load raw profiling data from the
/// dump file.
class ProfileDataLoader {
private:
  /// The name of the file where the raw profiling data is stored.
  const std::string &Filename;

  /// A vector of the command line arguments used when the target program was
  /// run to generate profiling data.  One entry per program run.
  SmallVector<std::string, 1> CommandLines;

  /// The raw values for how many times each edge was traversed, values from
  /// multiple program runs are accumulated.
  SmallVector<unsigned, 32> EdgeCounts;

public:
  /// ProfileDataLoader ctor - Read the specified profiling data file, exiting
  /// the program if the file is invalid or broken.
  ProfileDataLoader(const char *ToolName, const std::string &Filename);

  /// A special value used to represent the weight of an edge which has not
  /// been counted yet.
  static const unsigned Uncounted;

  /// getNumExecutions - Return the number of times the target program was run
  /// to generate this profiling data.
  unsigned getNumExecutions() const { return CommandLines.size(); }

  /// getExecution - Return the command line parameters used to generate the
  /// i'th set of profiling data.
  const std::string &getExecution(unsigned i) const { return CommandLines[i]; }

  const std::string &getFileName() const { return Filename; }

  /// getRawEdgeCounts - Return the raw profiling data, this is just a list of
  /// numbers with no mappings to edges.
  ArrayRef<unsigned> getRawEdgeCounts() const { return EdgeCounts; }
};

/// createProfileMetadataLoaderPass - This function returns a Pass that loads
/// the profiling information for the module from the specified filename.
ModulePass *createProfileMetadataLoaderPass(const std::string &Filename);

} // End llvm namespace

#endif
