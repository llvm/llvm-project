//===- Delta.cpp - Delta Debugging Algorithm Implementation ---------------===//
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

#include "Delta.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <fstream>
#include <set>

using namespace llvm;

static cl::opt<bool> AbortOnInvalidReduction(
    "abort-on-invalid-reduction",
    cl::desc("Abort if any reduction results in invalid IR"));

void writeOutput(llvm::Module &M, llvm::StringRef Message);

bool isReduced(Module &M, TestRunner &Test, SmallString<128> &CurrentFilepath) {
  // Write Module to tmp file
  int FD;
  std::error_code EC =
      sys::fs::createTemporaryFile("llvm-reduce", "ll", FD, CurrentFilepath);
  if (EC) {
    errs() << "Error making unique filename: " << EC.message() << "!\n";
    exit(1);
  }

  ToolOutputFile Out(CurrentFilepath, FD);
  M.print(Out.os(), /*AnnotationWriter=*/nullptr);
  Out.os().close();
  if (Out.os().has_error()) {
    errs() << "Error emitting bitcode to file '" << CurrentFilepath << "'!\n";
    exit(1);
  }

  // Current Chunks aren't interesting
  return Test.run(CurrentFilepath);
}

/// Counts the amount of lines for a given file
static int getLines(StringRef Filepath) {
  int Lines = 0;
  std::string CurrLine;
  std::ifstream FileStream{std::string(Filepath)};

  while (std::getline(FileStream, CurrLine))
    ++Lines;

  return Lines;
}

/// Splits Chunks in half and prints them.
/// If unable to split (when chunk size is 1) returns false.
static bool increaseGranularity(std::vector<Chunk> &Chunks) {
  errs() << "Increasing granularity...";
  std::vector<Chunk> NewChunks;
  bool SplitOne = false;

  for (auto &C : Chunks) {
    if (C.End - C.Begin == 0)
      NewChunks.push_back(C);
    else {
      int Half = (C.Begin + C.End) / 2;
      NewChunks.push_back({C.Begin, Half});
      NewChunks.push_back({Half + 1, C.End});
      SplitOne = true;
    }
  }
  if (SplitOne) {
    Chunks = NewChunks;
    errs() << "Success! New Chunks:\n";
    for (auto C : Chunks) {
      errs() << '\t';
      C.print();
      errs() << '\n';
    }
  }
  return SplitOne;
}

/// Runs the Delta Debugging algorithm, splits the code into chunks and
/// reduces the amount of chunks that are considered interesting by the
/// given test.
void llvm::runDeltaPass(
    TestRunner &Test, int Targets,
    function_ref<void(Oracle &, Module &)> ExtractChunksFromModule) {
  assert(Targets >= 0);
  if (!Targets) {
    errs() << "\nNothing to reduce\n";
    return;
  }

  SmallString<128> CurrentFilepath;
  if (!isReduced(Test.getProgram(), Test, CurrentFilepath)) {
    errs() << "\nInput isn't interesting! Verify interesting-ness test\n";
    exit(1);
  }

  assert(!verifyModule(Test.getProgram(), &errs()) &&
         "input module is broken before making changes");

  std::vector<Chunk> ChunksStillConsideredInteresting = {{1, Targets}};
  std::unique_ptr<Module> ReducedProgram;

  bool FoundAtLeastOneNewUninterestingChunkWithCurrentGranularity;
  do {
    FoundAtLeastOneNewUninterestingChunkWithCurrentGranularity = false;

    std::set<Chunk> UninterestingChunks;
    for (Chunk &ChunkToCheckForUninterestingness :
         reverse(ChunksStillConsideredInteresting)) {
      // Take all of ChunksStillConsideredInteresting chunks, except those we've
      // already deemed uninteresting (UninterestingChunks) but didn't remove
      // from ChunksStillConsideredInteresting yet, and additionally ignore
      // ChunkToCheckForUninterestingness chunk.
      std::vector<Chunk> CurrentChunks;
      CurrentChunks.reserve(ChunksStillConsideredInteresting.size() -
                            UninterestingChunks.size() - 1);
      copy_if(ChunksStillConsideredInteresting,
              std::back_inserter(CurrentChunks), [&](const Chunk &C) {
                return !UninterestingChunks.count(C) &&
                       C != ChunkToCheckForUninterestingness;
              });

      // Clone module before hacking it up..
      std::unique_ptr<Module> Clone = CloneModule(Test.getProgram());
      // Generate Module with only Targets inside Current Chunks
      Oracle O(CurrentChunks);
      ExtractChunksFromModule(O, *Clone);

      // Some reductions may result in invalid IR. Skip such reductions.
      if (verifyModule(*Clone, &errs())) {
        if (AbortOnInvalidReduction) {
          errs() << "Invalid reduction\n";
          exit(1);
        }
        errs() << " **** WARNING | reduction resulted in invalid module, "
                  "skipping\n";
        continue;
      }

      errs() << "Ignoring: ";
      ChunkToCheckForUninterestingness.print();
      for (const Chunk &C : UninterestingChunks)
        C.print();

      SmallString<128> CurrentFilepath;
      if (!isReduced(*Clone, Test, CurrentFilepath)) {
        // Program became non-reduced, so this chunk appears to be interesting.
        errs() << "\n";
        continue;
      }

      FoundAtLeastOneNewUninterestingChunkWithCurrentGranularity = true;
      UninterestingChunks.insert(ChunkToCheckForUninterestingness);
      ReducedProgram = std::move(Clone);
      errs() << " **** SUCCESS | lines: " << getLines(CurrentFilepath) << "\n";
      writeOutput(*ReducedProgram, "Saved new best reduction to ");
    }
    // Delete uninteresting chunks
    erase_if(ChunksStillConsideredInteresting,
             [&UninterestingChunks](const Chunk &C) {
               return UninterestingChunks.count(C);
             });
  } while (!ChunksStillConsideredInteresting.empty() &&
           (FoundAtLeastOneNewUninterestingChunkWithCurrentGranularity ||
            increaseGranularity(ChunksStillConsideredInteresting)));

  // If we reduced the testcase replace it
  if (ReducedProgram)
    Test.setProgram(std::move(ReducedProgram));
  errs() << "Couldn't increase anymore.\n";
}
