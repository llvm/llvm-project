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
#include "ReducerWorkItem.h"
#include "Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"
#include <fstream>
#include <set>

using namespace llvm;

extern cl::OptionCategory LLVMReduceOptions;

static cl::opt<bool> AbortOnInvalidReduction(
    "abort-on-invalid-reduction",
    cl::desc("Abort if any reduction results in invalid IR"),
    cl::cat(LLVMReduceOptions));

static cl::opt<unsigned int> StartingGranularityLevel(
    "starting-granularity-level",
    cl::desc("Number of times to divide chunks prior to first test"),
    cl::cat(LLVMReduceOptions));

static cl::opt<bool> TmpFilesAsBitcode(
    "write-tmp-files-as-bitcode",
    cl::desc("Always write temporary files as bitcode instead of textual IR"),
    cl::init(false), cl::cat(LLVMReduceOptions));

#ifdef LLVM_ENABLE_THREADS
static cl::opt<unsigned> NumJobs(
    "j",
    cl::desc("Maximum number of threads to use to process chunks. Set to 1 to "
             "disables parallelism."),
    cl::init(1), cl::cat(LLVMReduceOptions));
#else
unsigned NumJobs = 1;
#endif

void writeBitcode(ReducerWorkItem &M, raw_ostream &OutStream);

void readBitcode(ReducerWorkItem &M, MemoryBufferRef Data, LLVMContext &Ctx,
                 const char *ToolName);

bool isReduced(ReducerWorkItem &M, const TestRunner &Test) {
  const bool UseBitcode = Test.inputIsBitcode() || TmpFilesAsBitcode;

  SmallString<128> CurrentFilepath;

  // Write ReducerWorkItem to tmp file
  int FD;
  std::error_code EC = sys::fs::createTemporaryFile(
      "llvm-reduce", M.isMIR() ? "mir" : (UseBitcode ? "bc" : "ll"), FD,
      CurrentFilepath,
      UseBitcode && !M.isMIR() ? sys::fs::OF_None : sys::fs::OF_Text);
  if (EC) {
    errs() << "Error making unique filename: " << EC.message() << "!\n";
    exit(1);
  }

  ToolOutputFile Out(CurrentFilepath, FD);

  if (TmpFilesAsBitcode)
    writeBitcode(M, Out.os());
  else
    M.print(Out.os(), /*AnnotationWriter=*/nullptr);

  Out.os().close();
  if (Out.os().has_error()) {
    errs() << "Error emitting bitcode to file '" << CurrentFilepath
           << "': " << Out.os().error().message();
    exit(1);
  }

  // Current Chunks aren't interesting
  return Test.run(CurrentFilepath);
}

/// Splits Chunks in half and prints them.
/// If unable to split (when chunk size is 1) returns false.
static bool increaseGranularity(std::vector<Chunk> &Chunks) {
  if (Verbose)
    errs() << "Increasing granularity...";
  std::vector<Chunk> NewChunks;
  bool SplitAny = false;

  for (Chunk C : Chunks) {
    if (C.End - C.Begin == 0)
      NewChunks.push_back(C);
    else {
      int Half = (C.Begin + C.End) / 2;
      NewChunks.push_back({C.Begin, Half});
      NewChunks.push_back({Half + 1, C.End});
      SplitAny = true;
    }
  }
  if (SplitAny) {
    Chunks = NewChunks;
    if (Verbose) {
      errs() << "Success! " << NewChunks.size() << " New Chunks:\n";
      for (auto C : Chunks) {
        errs() << '\t';
        C.print();
        errs() << '\n';
      }
    }
  }
  return SplitAny;
}

// Check if \p ChunkToCheckForUninterestingness is interesting. Returns the
// modified module if the chunk resulted in a reduction.
static std::unique_ptr<ReducerWorkItem>
CheckChunk(const Chunk &ChunkToCheckForUninterestingness,
           std::unique_ptr<ReducerWorkItem> Clone, const TestRunner &Test,
           ReductionFunc ExtractChunksFromModule,
           const DenseSet<Chunk> &UninterestingChunks,
           const std::vector<Chunk> &ChunksStillConsideredInteresting) {
  // Take all of ChunksStillConsideredInteresting chunks, except those we've
  // already deemed uninteresting (UninterestingChunks) but didn't remove
  // from ChunksStillConsideredInteresting yet, and additionally ignore
  // ChunkToCheckForUninterestingness chunk.
  std::vector<Chunk> CurrentChunks;
  CurrentChunks.reserve(ChunksStillConsideredInteresting.size() -
                        UninterestingChunks.size() - 1);
  copy_if(ChunksStillConsideredInteresting, std::back_inserter(CurrentChunks),
          [&](const Chunk &C) {
            return C != ChunkToCheckForUninterestingness &&
                   !UninterestingChunks.count(C);
          });

  // Generate Module with only Targets inside Current Chunks
  Oracle O(CurrentChunks);
  ExtractChunksFromModule(O, *Clone);

  // Some reductions may result in invalid IR. Skip such reductions.
  if (verifyReducerWorkItem(*Clone, &errs())) {
    if (AbortOnInvalidReduction) {
      errs() << "Invalid reduction, aborting.\n";
      Clone->print(errs());
      exit(1);
    }
    if (Verbose) {
      errs() << " **** WARNING | reduction resulted in invalid module, "
                "skipping\n";
    }
    return nullptr;
  }

  if (Verbose) {
    errs() << "Ignoring: ";
    ChunkToCheckForUninterestingness.print();
    for (const Chunk &C : UninterestingChunks)
      C.print();
    errs() << "\n";
  }

  if (!isReduced(*Clone, Test)) {
    // Program became non-reduced, so this chunk appears to be interesting.
    if (Verbose)
      errs() << "\n";
    return nullptr;
  }
  return Clone;
}

static SmallString<0> ProcessChunkFromSerializedBitcode(
    Chunk &ChunkToCheckForUninterestingness, TestRunner &Test,
    ReductionFunc ExtractChunksFromModule, DenseSet<Chunk> &UninterestingChunks,
    std::vector<Chunk> &ChunksStillConsideredInteresting,
    SmallString<0> &OriginalBC, std::atomic<bool> &AnyReduced) {
  LLVMContext Ctx;
  auto CloneMMM = std::make_unique<ReducerWorkItem>();
  MemoryBufferRef Data(StringRef(OriginalBC), "<bc file>");
  readBitcode(*CloneMMM, Data, Ctx, Test.getToolName());

  SmallString<0> Result;
  if (std::unique_ptr<ReducerWorkItem> ChunkResult =
          CheckChunk(ChunkToCheckForUninterestingness, std::move(CloneMMM),
                     Test, ExtractChunksFromModule, UninterestingChunks,
                     ChunksStillConsideredInteresting)) {
    raw_svector_ostream BCOS(Result);
    writeBitcode(*ChunkResult, BCOS);
    // Communicate that the task reduced a chunk.
    AnyReduced = true;
  }
  return Result;
}

/// Runs the Delta Debugging algorithm, splits the code into chunks and
/// reduces the amount of chunks that are considered interesting by the
/// given test. The number of chunks is determined by a preliminary run of the
/// reduction pass where no change must be made to the module.
void llvm::runDeltaPass(TestRunner &Test, ReductionFunc ExtractChunksFromModule,
                        StringRef Message) {
  assert(!verifyReducerWorkItem(Test.getProgram(), &errs()) &&
         "input module is broken before making changes");
  errs() << "*** " << Message << "...\n";

  int Targets;
  {
    // Count the number of chunks by counting the number of calls to
    // Oracle::shouldKeep() but always returning true so no changes are
    // made.
    std::vector<Chunk> AllChunks = {{0, INT_MAX}};
    Oracle Counter(AllChunks);
    ExtractChunksFromModule(Counter, Test.getProgram());
    Targets = Counter.count();

    assert(!verifyReducerWorkItem(Test.getProgram(), &errs()) &&
           "input module is broken after counting chunks");
    assert(isReduced(Test.getProgram(), Test) &&
           "input module no longer interesting after counting chunks");

#ifndef NDEBUG
    // Make sure that the number of chunks does not change as we reduce.
    std::vector<Chunk> NoChunks = {{0, INT_MAX}};
    Oracle NoChunksCounter(NoChunks);
    std::unique_ptr<ReducerWorkItem> Clone =
        cloneReducerWorkItem(Test.getProgram(), Test.getTargetMachine());
    ExtractChunksFromModule(NoChunksCounter, *Clone);
    assert(Targets == NoChunksCounter.count() &&
           "number of chunks changes when reducing");
#endif
  }
  if (!Targets) {
    if (Verbose)
      errs() << "\nNothing to reduce\n";
    errs() << "----------------------------\n";
    return;
  }

  std::vector<Chunk> ChunksStillConsideredInteresting = {{0, Targets - 1}};
  std::unique_ptr<ReducerWorkItem> ReducedProgram;

  for (unsigned int Level = 0; Level < StartingGranularityLevel; Level++) {
    increaseGranularity(ChunksStillConsideredInteresting);
  }

  std::atomic<bool> AnyReduced;
  std::unique_ptr<ThreadPool> ChunkThreadPoolPtr;
  if (NumJobs > 1)
    ChunkThreadPoolPtr =
        std::make_unique<ThreadPool>(hardware_concurrency(NumJobs));

  bool FoundAtLeastOneNewUninterestingChunkWithCurrentGranularity;
  do {
    FoundAtLeastOneNewUninterestingChunkWithCurrentGranularity = false;

    DenseSet<Chunk> UninterestingChunks;

    // When running with more than one thread, serialize the original bitcode
    // to OriginalBC.
    SmallString<0> OriginalBC;
    if (NumJobs > 1) {
      raw_svector_ostream BCOS(OriginalBC);
      writeBitcode(Test.getProgram(), BCOS);
    }

    std::deque<std::shared_future<SmallString<0>>> TaskQueue;
    for (auto I = ChunksStillConsideredInteresting.rbegin(),
              E = ChunksStillConsideredInteresting.rend();
         I != E; ++I) {
      std::unique_ptr<ReducerWorkItem> Result = nullptr;
      unsigned WorkLeft = std::distance(I, E);

      // Run in parallel mode, if the user requested more than one thread and
      // there are at least a few chunks to process.
      if (NumJobs > 1 && WorkLeft > 1) {
        unsigned NumInitialTasks = std::min(WorkLeft, unsigned(NumJobs));
        unsigned NumChunksProcessed = 0;

        ThreadPool &ChunkThreadPool = *ChunkThreadPoolPtr;
        TaskQueue.clear();

        AnyReduced = false;
        // Queue jobs to process NumInitialTasks chunks in parallel using
        // ChunkThreadPool. When the tasks are added to the pool, parse the
        // original module from OriginalBC with a fresh LLVMContext object. This
        // ensures that the cloned module of each task uses an independent
        // LLVMContext object. If a task reduces the input, serialize the result
        // back in the corresponding Result element.
        for (unsigned J = 0; J < NumInitialTasks; ++J) {
          TaskQueue.emplace_back(ChunkThreadPool.async(
              [J, I, &Test, &ExtractChunksFromModule, &UninterestingChunks,
               &ChunksStillConsideredInteresting, &OriginalBC, &AnyReduced]() {
                return ProcessChunkFromSerializedBitcode(
                    *(I + J), Test, ExtractChunksFromModule,
                    UninterestingChunks, ChunksStillConsideredInteresting,
                    OriginalBC, AnyReduced);
              }));
        }

        // Start processing results of the queued tasks. We wait for the first
        // task in the queue to finish. If it reduced a chunk, we parse the
        // result and exit the loop.
        //  Otherwise we will try to schedule a new task, if
        //  * no other pending job reduced a chunk and
        //  * we have not reached the end of the chunk.
        while (!TaskQueue.empty()) {
          auto &Future = TaskQueue.front();
          Future.wait();

          NumChunksProcessed++;
          SmallString<0> Res = Future.get();
          TaskQueue.pop_front();
          if (Res.empty()) {
            unsigned NumScheduledTasks = NumChunksProcessed + TaskQueue.size();
            if (!AnyReduced && I + NumScheduledTasks != E) {
              Chunk &ChunkToCheck = *(I + NumScheduledTasks);
              TaskQueue.emplace_back(ChunkThreadPool.async(
                  [&Test, &ExtractChunksFromModule, &UninterestingChunks,
                   &ChunksStillConsideredInteresting, &OriginalBC,
                   &ChunkToCheck, &AnyReduced]() {
                    return ProcessChunkFromSerializedBitcode(
                        ChunkToCheck, Test, ExtractChunksFromModule,
                        UninterestingChunks, ChunksStillConsideredInteresting,
                        OriginalBC, AnyReduced);
                  }));
            }
            continue;
          }

          Result = std::make_unique<ReducerWorkItem>();
          MemoryBufferRef Data(StringRef(Res), "<bc file>");
          readBitcode(*Result, Data, Test.getProgram().M->getContext(),
                      Test.getToolName());
          break;
        }
        // Forward I to the last chunk processed in parallel.
        I += NumChunksProcessed - 1;
      } else {
        Result = CheckChunk(
            *I,
            cloneReducerWorkItem(Test.getProgram(), Test.getTargetMachine()),
            Test, ExtractChunksFromModule, UninterestingChunks,
            ChunksStillConsideredInteresting);
      }

      if (!Result)
        continue;

      Chunk &ChunkToCheckForUninterestingness = *I;
      FoundAtLeastOneNewUninterestingChunkWithCurrentGranularity = true;
      UninterestingChunks.insert(ChunkToCheckForUninterestingness);
      ReducedProgram = std::move(Result);

      // FIXME: Report meaningful progress info
      Test.writeOutput(" **** SUCCESS | Saved new best reduction to ");
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
  if (Verbose)
    errs() << "Couldn't increase anymore.\n";
  errs() << "----------------------------\n";
}
