//===-- BasicBlockSectionsProfileReader.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the basic block sections profile reader pass. It parses
// and stores the basic block sections profile file (which is specified via the
// `-basic-block-sections` flag).
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/BasicBlockSectionsProfileReader.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;

char BasicBlockSectionsProfileReader::ID = 0;
INITIALIZE_PASS(BasicBlockSectionsProfileReader, "bbsections-profile-reader",
                "Reads and parses a basic block sections profile.", false,
                false)

bool BasicBlockSectionsProfileReader::isFunctionHot(StringRef FuncName) const {
  return getBBClusterInfoForFunction(FuncName).first;
}

std::pair<bool, SmallVector<BBClusterInfo>>
BasicBlockSectionsProfileReader::getBBClusterInfoForFunction(
    StringRef FuncName) const {
  auto R = ProgramBBClusterInfo.find(getAliasName(FuncName));
  return R != ProgramBBClusterInfo.end()
             ? std::pair(true, R->second)
             : std::pair(false, SmallVector<BBClusterInfo>{});
}

// Basic Block Sections can be enabled for a subset of machine basic blocks.
// This is done by passing a file containing names of functions for which basic
// block sections are desired.  Additionally, machine basic block ids of the
// functions can also be specified for a finer granularity. Moreover, a cluster
// of basic blocks could be assigned to the same section.
// A file with basic block sections for all of function main and three blocks
// for function foo (of which 1 and 2 are placed in a cluster) looks like this:
// ----------------------------
// list.txt:
// !main
// !foo
// !!1 2
// !!4
static Error getBBClusterInfo(const MemoryBuffer *MBuf,
                              ProgramBBClusterInfoMapTy &ProgramBBClusterInfo,
                              StringMap<StringRef> &FuncAliasMap) {
  assert(MBuf);
  line_iterator LineIt(*MBuf, /*SkipBlanks=*/true, /*CommentMarker=*/'#');

  auto invalidProfileError = [&](auto Message) {
    return make_error<StringError>(
        Twine("Invalid profile " + MBuf->getBufferIdentifier() + " at line " +
              Twine(LineIt.line_number()) + ": " + Message),
        inconvertibleErrorCode());
  };

  auto FI = ProgramBBClusterInfo.end();

  // Current cluster ID corresponding to this function.
  unsigned CurrentCluster = 0;
  // Current position in the current cluster.
  unsigned CurrentPosition = 0;

  // Temporary set to ensure every basic block ID appears once in the clusters
  // of a function.
  SmallSet<unsigned, 4> FuncBBIDs;

  for (; !LineIt.is_at_eof(); ++LineIt) {
    StringRef S(*LineIt);
    if (S[0] == '@')
      continue;
    // Check for the leading "!"
    if (!S.consume_front("!") || S.empty())
      break;
    // Check for second "!" which indicates a cluster of basic blocks.
    if (S.consume_front("!")) {
      if (FI == ProgramBBClusterInfo.end())
        return invalidProfileError(
            "Cluster list does not follow a function name specifier.");
      SmallVector<StringRef, 4> BBIDs;
      S.split(BBIDs, ' ');
      // Reset current cluster position.
      CurrentPosition = 0;
      for (auto BBIDStr : BBIDs) {
        unsigned long long BBID;
        if (getAsUnsignedInteger(BBIDStr, 10, BBID))
          return invalidProfileError(Twine("Unsigned integer expected: '") +
                                     BBIDStr + "'.");
        if (!FuncBBIDs.insert(BBID).second)
          return invalidProfileError(Twine("Duplicate basic block id found '") +
                                     BBIDStr + "'.");
        if (BBID == 0 && CurrentPosition)
          return invalidProfileError("Entry BB (0) does not begin a cluster.");

        FI->second.emplace_back(
            BBClusterInfo{((unsigned)BBID), CurrentCluster, CurrentPosition++});
      }
      CurrentCluster++;
    } else { // This is a function name specifier.
      // Function aliases are separated using '/'. We use the first function
      // name for the cluster info mapping and delegate all other aliases to
      // this one.
      SmallVector<StringRef, 4> Aliases;
      S.split(Aliases, '/');
      for (size_t i = 1; i < Aliases.size(); ++i)
        FuncAliasMap.try_emplace(Aliases[i], Aliases.front());

      // Prepare for parsing clusters of this function name.
      // Start a new cluster map for this function name.
      FI = ProgramBBClusterInfo.try_emplace(Aliases.front()).first;
      CurrentCluster = 0;
      FuncBBIDs.clear();
    }
  }
  return Error::success();
}

void BasicBlockSectionsProfileReader::initializePass() {
  if (!MBuf)
    return;
  if (auto Err = getBBClusterInfo(MBuf, ProgramBBClusterInfo, FuncAliasMap))
    report_fatal_error(std::move(Err));
}

ImmutablePass *
llvm::createBasicBlockSectionsProfileReaderPass(const MemoryBuffer *Buf) {
  return new BasicBlockSectionsProfileReader(Buf);
}
