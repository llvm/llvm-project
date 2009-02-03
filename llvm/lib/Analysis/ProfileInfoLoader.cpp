//===- ProfileInfoLoad.cpp - Load profile information from disk -----------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The ProfileInfoLoader class is used to load and represent profiling
// information read in from the dump file.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ProfileInfoLoader.h"
#include "llvm/Analysis/ProfileInfoTypes.h"
#include "llvm/Module.h"
#include "llvm/InstrTypes.h"
#include "llvm/Support/Streams.h"
#include <cstdio>
#include <cstdlib>
#include <map>
using namespace llvm;

// ByteSwap - Byteswap 'Var' if 'Really' is true.
//
static inline unsigned ByteSwap(unsigned Var, bool Really) {
  if (!Really) return Var;
  return ((Var & (255<< 0)) << 24) |
         ((Var & (255<< 8)) <<  8) |
         ((Var & (255<<16)) >>  8) |
         ((Var & (255<<24)) >> 24);
}

static void ReadProfilingBlock(const char *ToolName, FILE *F,
                               bool ShouldByteSwap,
                               std::vector<unsigned> &Data) {
  // Read the number of entries...
  unsigned NumEntries;
  if (fread(&NumEntries, sizeof(unsigned), 1, F) != 1) {
    cerr << ToolName << ": data packet truncated!\n";
    perror(0);
    exit(1);
  }
  NumEntries = ByteSwap(NumEntries, ShouldByteSwap);

  // Read the counts...
  std::vector<unsigned> TempSpace(NumEntries);

  // Read in the block of data...
  if (fread(&TempSpace[0], sizeof(unsigned)*NumEntries, 1, F) != 1) {
    cerr << ToolName << ": data packet truncated!\n";
    perror(0);
    exit(1);
  }

  // Make sure we have enough space...
  if (Data.size() < NumEntries)
    Data.resize(NumEntries);

  // Accumulate the data we just read into the data.
  if (!ShouldByteSwap) {
    for (unsigned i = 0; i != NumEntries; ++i)
      Data[i] += TempSpace[i];
  } else {
    for (unsigned i = 0; i != NumEntries; ++i)
      Data[i] += ByteSwap(TempSpace[i], true);
  }
}

// ProfileInfoLoader ctor - Read the specified profiling data file, exiting the
// program if the file is invalid or broken.
//
ProfileInfoLoader::ProfileInfoLoader(const char *ToolName,
                                     const std::string &Filename,
                                     Module &TheModule) : M(TheModule) {
  FILE *F = fopen(Filename.c_str(), "r");
  if (F == 0) {
    cerr << ToolName << ": Error opening '" << Filename << "': ";
    perror(0);
    exit(1);
  }

  // Keep reading packets until we run out of them.
  unsigned PacketType;
  while (fread(&PacketType, sizeof(unsigned), 1, F) == 1) {
    // If the low eight bits of the packet are zero, we must be dealing with an
    // endianness mismatch.  Byteswap all words read from the profiling
    // information.
    bool ShouldByteSwap = (char)PacketType == 0;
    PacketType = ByteSwap(PacketType, ShouldByteSwap);

    switch (PacketType) {
    case ArgumentInfo: {
      unsigned ArgLength;
      if (fread(&ArgLength, sizeof(unsigned), 1, F) != 1) {
        cerr << ToolName << ": arguments packet truncated!\n";
        perror(0);
        exit(1);
      }
      ArgLength = ByteSwap(ArgLength, ShouldByteSwap);

      // Read in the arguments...
      std::vector<char> Chars(ArgLength+4);

      if (ArgLength)
        if (fread(&Chars[0], (ArgLength+3) & ~3, 1, F) != 1) {
          cerr << ToolName << ": arguments packet truncated!\n";
          perror(0);
          exit(1);
        }
      CommandLines.push_back(std::string(&Chars[0], &Chars[ArgLength]));
      break;
    }

    case FunctionInfo:
      ReadProfilingBlock(ToolName, F, ShouldByteSwap, FunctionCounts);
      break;

    case BlockInfo:
      ReadProfilingBlock(ToolName, F, ShouldByteSwap, BlockCounts);
      break;

    case EdgeInfo:
      ReadProfilingBlock(ToolName, F, ShouldByteSwap, EdgeCounts);
      break;

    case BBTraceInfo:
      ReadProfilingBlock(ToolName, F, ShouldByteSwap, BBTrace);
      break;

    default:
      cerr << ToolName << ": Unknown packet type #" << PacketType << "!\n";
      exit(1);
    }
  }

  fclose(F);
}


// getFunctionCounts - This method is used by consumers of function counting
// information.  If we do not directly have function count information, we
// compute it from other, more refined, types of profile information.
//
void ProfileInfoLoader::getFunctionCounts(std::vector<std::pair<Function*,
                                                      unsigned> > &Counts) {
  if (FunctionCounts.empty()) {
    if (hasAccurateBlockCounts()) {
      // Synthesize function frequency information from the number of times
      // their entry blocks were executed.
      std::vector<std::pair<BasicBlock*, unsigned> > BlockCounts;
      getBlockCounts(BlockCounts);

      for (unsigned i = 0, e = BlockCounts.size(); i != e; ++i)
        if (&BlockCounts[i].first->getParent()->getEntryBlock() ==
            BlockCounts[i].first)
          Counts.push_back(std::make_pair(BlockCounts[i].first->getParent(),
                                          BlockCounts[i].second));
    } else {
      cerr << "Function counts are not available!\n";
    }
    return;
  }

  unsigned Counter = 0;
  for (Module::iterator I = M.begin(), E = M.end();
       I != E && Counter != FunctionCounts.size(); ++I)
    if (!I->isDeclaration())
      Counts.push_back(std::make_pair(I, FunctionCounts[Counter++]));
}

// getBlockCounts - This method is used by consumers of block counting
// information.  If we do not directly have block count information, we
// compute it from other, more refined, types of profile information.
//
void ProfileInfoLoader::getBlockCounts(std::vector<std::pair<BasicBlock*,
                                                         unsigned> > &Counts) {
  if (BlockCounts.empty()) {
    if (hasAccurateEdgeCounts()) {
      // Synthesize block count information from edge frequency information.
      // The block execution frequency is equal to the sum of the execution
      // frequency of all outgoing edges from a block.
      //
      // If a block has no successors, this will not be correct, so we have to
      // special case it. :(
      std::vector<std::pair<Edge, unsigned> > EdgeCounts;
      getEdgeCounts(EdgeCounts);

      std::map<BasicBlock*, unsigned> InEdgeFreqs;

      BasicBlock *LastBlock = 0;
      TerminatorInst *TI = 0;
      for (unsigned i = 0, e = EdgeCounts.size(); i != e; ++i) {
        if (EdgeCounts[i].first.first != LastBlock) {
          LastBlock = EdgeCounts[i].first.first;
          TI = LastBlock->getTerminator();
          Counts.push_back(std::make_pair(LastBlock, 0));
        }
        Counts.back().second += EdgeCounts[i].second;
        unsigned SuccNum = EdgeCounts[i].first.second;
        if (SuccNum >= TI->getNumSuccessors()) {
          static bool Warned = false;
          if (!Warned) {
            cerr << "WARNING: profile info doesn't seem to match"
                 << " the program!\n";
            Warned = true;
          }
        } else {
          // If this successor has no successors of its own, we will never
          // compute an execution count for that block.  Remember the incoming
          // edge frequencies to add later.
          BasicBlock *Succ = TI->getSuccessor(SuccNum);
          if (Succ->getTerminator()->getNumSuccessors() == 0)
            InEdgeFreqs[Succ] += EdgeCounts[i].second;
        }
      }

      // Now we have to accumulate information for those blocks without
      // successors into our table.
      for (std::map<BasicBlock*, unsigned>::iterator I = InEdgeFreqs.begin(),
             E = InEdgeFreqs.end(); I != E; ++I) {
        unsigned i = 0;
        for (; i != Counts.size() && Counts[i].first != I->first; ++i)
          /*empty*/;
        if (i == Counts.size()) Counts.push_back(std::make_pair(I->first, 0));
        Counts[i].second += I->second;
      }

    } else {
      cerr << "Block counts are not available!\n";
    }
    return;
  }

  unsigned Counter = 0;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      Counts.push_back(std::make_pair(BB, BlockCounts[Counter++]));
      if (Counter == BlockCounts.size())
        return;
    }
}

// getEdgeCounts - This method is used by consumers of edge counting
// information.  If we do not directly have edge count information, we compute
// it from other, more refined, types of profile information.
//
void ProfileInfoLoader::getEdgeCounts(std::vector<std::pair<Edge,
                                                  unsigned> > &Counts) {
  if (EdgeCounts.empty()) {
    cerr << "Edge counts not available, and no synthesis "
         << "is implemented yet!\n";
    return;
  }

  unsigned Counter = 0;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F)
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      for (unsigned i = 0, e = BB->getTerminator()->getNumSuccessors();
           i != e; ++i) {
        Counts.push_back(std::make_pair(Edge(BB, i), EdgeCounts[Counter++]));
        if (Counter == EdgeCounts.size())
          return;
      }
}

// getBBTrace - This method is used by consumers of basic-block trace
// information.
//
void ProfileInfoLoader::getBBTrace(std::vector<BasicBlock *> &Trace) {
  if (BBTrace.empty ()) {
    cerr << "Basic block trace is not available!\n";
    return;
  }
  cerr << "Basic block trace loading is not implemented yet!\n";
}
