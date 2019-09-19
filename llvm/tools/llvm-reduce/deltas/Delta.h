//===- Delta.h - Delta Debugging Algorithm Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation for the Delta Debugging Algorithm:
// it splits a given set of Targets (i.e. Functions, Instructions, BBs, etc.)
// into chunks and tries to reduce the number chunks that are interesting.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMREDUCE_LLVMREDUCE_DELTA_H
#define LLVM_TOOLS_LLVMREDUCE_LLVMREDUCE_DELTA_H

#include "TestRunner.h"
#include <vector>
#include <utility>
#include <functional>

namespace llvm {

struct Chunk {
  int begin;
  int end;

  /// Helper function to verify if a given Target-index is inside the Chunk
  bool contains(int Index) const { return Index >= begin && Index <= end; }

  void print() const {
    errs() << "[" << begin;
    if (end - begin != 0)
      errs() << "," << end;
    errs() << "]";
  }

  /// Operator when populating CurrentChunks in Generic Delta Pass
  friend bool operator!=(const Chunk &C1, const Chunk &C2) {
    return C1.begin != C2.begin || C1.end != C2.end;
  }

  /// Operator used for sets
  friend bool operator<(const Chunk &C1, const Chunk &C2) {
    return std::tie(C1.begin, C1.end) < std::tie(C2.begin, C2.end);
  }
};

/// This function implements the Delta Debugging algorithm, it receives a
/// number of Targets (e.g. Functions, Instructions, Basic Blocks, etc.) and
/// splits them in half; these chunks of targets are then tested while ignoring
/// one chunk, if a chunk is proven to be uninteresting (i.e. fails the test)
/// it is removed from consideration. The algorithm will attempt to split the
/// Chunks in half and start the process again until it can't split chunks
/// anymore.
///
/// This function is intended to be called by each specialized delta pass (e.g.
/// RemoveFunctions) and receives three key parameters:
/// * Test: The main TestRunner instance which is used to run the provided
/// interesting-ness test, as well as to store and access the reduced Program.
/// * Targets: The amount of Targets that are going to be reduced by the
/// algorithm, for example, the RemoveGlobalVars pass would send the amount of
/// initialized GVs.
/// * ExtractChunksFromModule: A function used to tailor the main program so it
/// only contains Targets that are inside Chunks of the given iteration.
/// Note: This function is implemented by each specialized Delta pass
///
/// Other implementations of the Delta Debugging algorithm can also be found in
/// the CReduce, Delta, and Lithium projects.
void runDeltaPass(TestRunner &Test, int Targets,
                  std::function<void(const std::vector<Chunk> &, Module *)>
                      ExtractChunksFromModule);
} // namespace llvm

#endif
