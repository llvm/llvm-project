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
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
#include <cstdlib>
using namespace llvm;

// ByteSwap - Byteswap 'Var' if 'Really' is true.
//
static inline unsigned ByteSwap(unsigned Var, bool Really) {
  if (!Really) return Var;
  return ((Var & (255U<< 0U)) << 24U) |
         ((Var & (255U<< 8U)) <<  8U) |
         ((Var & (255U<<16U)) >>  8U) |
         ((Var & (255U<<24U)) >> 24U);
}

static unsigned AddCounts(unsigned A, unsigned B) {
  // If either value is undefined, use the other.
  if (A == ProfileInfoLoader::Uncounted) return B;
  if (B == ProfileInfoLoader::Uncounted) return A;
  return A + B;
}

static void ReadProfilingBlock(const char *ToolName, FILE *F,
                               bool ShouldByteSwap,
                               std::vector<unsigned> &Data) {
  // Read the number of entries...
  unsigned NumEntries;
  if (fread(&NumEntries, sizeof(unsigned), 1, F) != 1) {
    errs() << ToolName << ": data packet truncated!\n";
    perror(0);
    exit(1);
  }
  NumEntries = ByteSwap(NumEntries, ShouldByteSwap);

  // Read the counts...
  std::vector<unsigned> TempSpace(NumEntries);

  // Read in the block of data...
  if (fread(&TempSpace[0], sizeof(unsigned)*NumEntries, 1, F) != 1) {
    errs() << ToolName << ": data packet truncated!\n";
    perror(0);
    exit(1);
  }

  // Make sure we have enough space... The space is initialised to -1 to
  // facitiltate the loading of missing values for OptimalEdgeProfiling.
  if (Data.size() < NumEntries)
    Data.resize(NumEntries, ProfileInfoLoader::Uncounted);

  // Accumulate the data we just read into the data.
  if (!ShouldByteSwap) {
    for (unsigned i = 0; i != NumEntries; ++i) {
      Data[i] = AddCounts(TempSpace[i], Data[i]);
    }
  } else {
    for (unsigned i = 0; i != NumEntries; ++i) {
      Data[i] = AddCounts(ByteSwap(TempSpace[i], true), Data[i]);
    }
  }
}

const unsigned ProfileInfoLoader::Uncounted = ~0U;

// ProfileInfoLoader ctor - Read the specified profiling data file, exiting the
// program if the file is invalid or broken.
//
ProfileInfoLoader::ProfileInfoLoader(const char *ToolName,
                                     const std::string &Filename)
  : Filename(Filename) {
  FILE *F = fopen(Filename.c_str(), "rb");
  if (F == 0) {
    errs() << ToolName << ": Error opening '" << Filename << "': ";
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
        errs() << ToolName << ": arguments packet truncated!\n";
        perror(0);
        exit(1);
      }
      ArgLength = ByteSwap(ArgLength, ShouldByteSwap);

      // Read in the arguments...
      std::vector<char> Chars(ArgLength+4);

      if (ArgLength)
        if (fread(&Chars[0], (ArgLength+3) & ~3, 1, F) != 1) {
          errs() << ToolName << ": arguments packet truncated!\n";
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

    case OptEdgeInfo:
      ReadProfilingBlock(ToolName, F, ShouldByteSwap, OptimalEdgeCounts);
      break;

    case BBTraceInfo:
      ReadProfilingBlock(ToolName, F, ShouldByteSwap, BBTrace);
      break;

    default:
      errs() << ToolName << ": Unknown packet type #" << PacketType << "!\n";
      exit(1);
    }
  }

  fclose(F);
}

