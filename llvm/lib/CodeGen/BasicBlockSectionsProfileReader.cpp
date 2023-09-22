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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Pass.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <llvm/ADT/STLExtras.h>

using namespace llvm;

char BasicBlockSectionsProfileReader::ID = 0;
INITIALIZE_PASS(BasicBlockSectionsProfileReader, "bbsections-profile-reader",
                "Reads and parses a basic block sections profile.", false,
                false)

Expected<ProfileBBID>
BasicBlockSectionsProfileReader::parseProfileBBID(StringRef S) const {
  SmallVector<StringRef, 2> Parts;
  S.split(Parts, '.');
  if (Parts.size() > 2)
    return createProfileParseError(Twine("unable to parse basic block id: '") +
                                   S + "'");
  unsigned long long BBID;
  if (getAsUnsignedInteger(Parts[0], 10, BBID))
    return createProfileParseError(
        Twine("unable to parse BB id: '" + Parts[0]) +
        "': unsigned integer expected");
  unsigned long long CloneID = 0;
  if (Parts.size() > 1 && getAsUnsignedInteger(Parts[1], 10, CloneID))
    return createProfileParseError(Twine("unable to parse clone id: '") +
                                   Parts[1] + "': unsigned integer expected");
  return ProfileBBID{static_cast<unsigned>(BBID),
                     static_cast<unsigned>(CloneID)};
}

bool BasicBlockSectionsProfileReader::isFunctionHot(StringRef FuncName) const {
  return getRawProfileForFunction(FuncName).first;
}

std::pair<bool, RawFunctionProfile>
BasicBlockSectionsProfileReader::getRawProfileForFunction(
    StringRef FuncName) const {
  auto R = RawProgramProfile.find(getAliasName(FuncName));
  return R != RawProgramProfile.end() ? std::pair(true, R->second)
                                      : std::pair(false, RawFunctionProfile());
}

// Reads the version 1 basic block sections profile. Profile for each function
// is encoded as follows:
//   m <module_name>
//   f <function_name_1> <function_name_2> ...
//   c <bb_id_1> <bb_id_2> <bb_id_3>
//   c <bb_id_4> <bb_id_5>
//   ...
// Module name specifier (starting with 'm') is optional and allows
// distinguishing profile for internal-linkage functions with the same name. If
// not specified, it will apply to any function with the same name. Function
// name specifier (starting with 'f') can specify multiple function name
// aliases. Basic block clusters are specified by 'c' and specify the cluster of
// basic blocks, and the internal order in which they must be placed in the same
// section.
// This profile can also specify cloning paths which instruct the compiler to
// clone basic blocks along a path. The cloned blocks are then specified in the
// cluster information.
// The following profile lists two cloning paths (starting with 'p') for
// function bar and places the total 11 blocks within two clusters. Each cloned
// block is identified by its original block id, along with its clone id. A
// block cloned multiple times (2 in this example) appears with distinct clone
// ids (2.1 and 2.2).
// ---------------------------
//
// f main
// f bar
// p 1 2 3
// p 4 2 5
// c 2 3 5 6 7
// c 1 2.1 3.1 4 2.2 5.1
Error BasicBlockSectionsProfileReader::ReadV1Profile() {
  auto FI = RawProgramProfile.end();

  // Current cluster ID corresponding to this function.
  unsigned CurrentCluster = 0;
  // Current position in the current cluster.
  unsigned CurrentPosition = 0;

  // Temporary set to ensure every basic block ID appears once in the clusters
  // of a function.
  DenseSet<ProfileBBID> FuncBBIDs;

  // Debug-info-based module filename for the current function. Empty string
  // means no filename.
  StringRef DIFilename;

  for (; !LineIt.is_at_eof(); ++LineIt) {
    StringRef S(*LineIt);
    char Specifier = S[0];
    S = S.drop_front().trim();
    SmallVector<StringRef, 4> Values;
    S.split(Values, ' ');
    switch (Specifier) {
    case '@':
      break;
    case 'm': // Module name speicifer.
      if (Values.size() != 1) {
        return createProfileParseError(Twine("invalid module name value: '") +
                                       S + "'");
      }
      DIFilename = sys::path::remove_leading_dotslash(Values[0]);
      continue;
    case 'f': { // Function names specifier.
      bool FunctionFound = any_of(Values, [&](StringRef Alias) {
        auto It = FunctionNameToDIFilename.find(Alias);
        // No match if this function name is not found in this module.
        if (It == FunctionNameToDIFilename.end())
          return false;
        // Return a match if debug-info-filename is not specified. Otherwise,
        // check for equality.
        return DIFilename.empty() || It->second.equals(DIFilename);
      });
      if (!FunctionFound) {
        // Skip the following profile by setting the profile iterator (FI) to
        // the past-the-end element.
        FI = RawProgramProfile.end();
        DIFilename = "";
        continue;
      }
      for (size_t i = 1; i < Values.size(); ++i)
        FuncAliasMap.try_emplace(Values[i], Values.front());

      // Prepare for parsing clusters of this function name.
      // Start a new cluster map for this function name.
      auto R = RawProgramProfile.try_emplace(Values.front());
      // Report error when multiple profiles have been specified for the same
      // function.
      if (!R.second)
        return createProfileParseError("duplicate profile for function '" +
                                       Values.front() + "'");
      FI = R.first;
      CurrentCluster = 0;
      FuncBBIDs.clear();
      // We won't need DIFilename anymore. Clean it up to avoid its application
      // on the next function.
      DIFilename = "";
      continue;
    }
    case 'c': // Basic block cluster specifier.
      // Skip the profile when we the profile iterator (FI) refers to the
      // past-the-end element.
      if (FI == RawProgramProfile.end())
        break;
      // Reset current cluster position.
      CurrentPosition = 0;
      for (auto BasicBlockIDStr : Values) {
        auto BasicBlockID = parseProfileBBID(BasicBlockIDStr);
        if (!BasicBlockID)
          return BasicBlockID.takeError();
        if (!FuncBBIDs.insert(*BasicBlockID).second)
          return createProfileParseError(
              Twine("duplicate basic block id found '") + BasicBlockIDStr +
              "'");

        if (!BasicBlockID->BBID && CurrentPosition)
          return createProfileParseError(
              "entry BB (0) does not begin a cluster.");

        FI->second.RawBBProfiles.emplace_back(BBProfile<ProfileBBID>{
            *std::move(BasicBlockID), CurrentCluster, CurrentPosition++});
      }
      CurrentCluster++;
      continue;
    case 'p': { // Basic block cloning path specifier.
      SmallSet<unsigned, 5> BBsInPath;
      FI->second.ClonePaths.push_back({});
      for (size_t I = 0; I < Values.size(); ++I) {
        auto BBIDStr = Values[I];
        unsigned long long BBID = 0;
        if (getAsUnsignedInteger(BBIDStr, 10, BBID))
          return createProfileParseError(Twine("unsigned integer expected: '") +
                                         BBIDStr + "'");
        if (I != 0 && !BBsInPath.insert(BBID).second)
          return createProfileParseError(
              Twine("duplicate cloned block in path: '") + BBIDStr + "'");
        FI->second.ClonePaths.back().push_back(BBID);
      }
      continue;
    }
    default:
      return createProfileParseError(Twine("invalid specifier: '") +
                                     Twine(Specifier) + "'");
    }
  }
  return Error::success();
}

Error BasicBlockSectionsProfileReader::ReadV0Profile() {
  auto FI = RawProgramProfile.end();
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
      // Skip the profile when we the profile iterator (FI) refers to the
      // past-the-end element.
      if (FI == RawProgramProfile.end())
        continue;
      SmallVector<StringRef, 4> BBIDs;
      S.split(BBIDs, ' ');
      // Reset current cluster position.
      CurrentPosition = 0;
      for (auto BBIDStr : BBIDs) {
        unsigned long long BBID;
        if (getAsUnsignedInteger(BBIDStr, 10, BBID))
          return createProfileParseError(Twine("unsigned integer expected: '") +
                                         BBIDStr + "'");
        if (!FuncBBIDs.insert(BBID).second)
          return createProfileParseError(
              Twine("duplicate basic block id found '") + BBIDStr + "'");
        if (BBID == 0 && CurrentPosition)
          return createProfileParseError(
              "entry BB (0) does not begin a cluster");

        FI->second.RawBBProfiles.emplace_back(
            BBProfile<ProfileBBID>({{static_cast<unsigned>(BBID), 0},
                                    CurrentCluster,
                                    CurrentPosition++}));
      }
      CurrentCluster++;
    } else {
      // This is a function name specifier. It may include a debug info filename
      // specifier starting with `M=`.
      auto [AliasesStr, DIFilenameStr] = S.split(' ');
      SmallString<128> DIFilename;
      if (DIFilenameStr.startswith("M=")) {
        DIFilename =
            sys::path::remove_leading_dotslash(DIFilenameStr.substr(2));
        if (DIFilename.empty())
          return createProfileParseError("empty module name specifier");
      } else if (!DIFilenameStr.empty()) {
        return createProfileParseError("unknown string found: '" +
                                       DIFilenameStr + "'");
      }
      // Function aliases are separated using '/'. We use the first function
      // name for the cluster info mapping and delegate all other aliases to
      // this one.
      SmallVector<StringRef, 4> Aliases;
      AliasesStr.split(Aliases, '/');
      bool FunctionFound = any_of(Aliases, [&](StringRef Alias) {
        auto It = FunctionNameToDIFilename.find(Alias);
        // No match if this function name is not found in this module.
        if (It == FunctionNameToDIFilename.end())
          return false;
        // Return a match if debug-info-filename is not specified. Otherwise,
        // check for equality.
        return DIFilename.empty() || It->second.equals(DIFilename);
      });
      if (!FunctionFound) {
        // Skip the following profile by setting the profile iterator (FI) to
        // the past-the-end element.
        FI = RawProgramProfile.end();
        continue;
      }
      for (size_t i = 1; i < Aliases.size(); ++i)
        FuncAliasMap.try_emplace(Aliases[i], Aliases.front());

      // Prepare for parsing clusters of this function name.
      // Start a new cluster map for this function name.
      auto R = RawProgramProfile.try_emplace(Aliases.front());
      // Report error when multiple profiles have been specified for the same
      // function.
      if (!R.second)
        return createProfileParseError("duplicate profile for function '" +
                                       Aliases.front() + "'");
      FI = R.first;
      CurrentCluster = 0;
      FuncBBIDs.clear();
    }
  }
  return Error::success();
}

// Basic Block Sections can be enabled for a subset of machine basic blocks.
// This is done by passing a file containing names of functions for which basic
// block sections are desired. Additionally, machine basic block ids of the
// functions can also be specified for a finer granularity. Moreover, a cluster
// of basic blocks could be assigned to the same section.
// Optionally, a debug-info filename can be specified for each function to allow
// distinguishing internal-linkage functions of the same name.
// A file with basic block sections for all of function main and three blocks
// for function foo (of which 1 and 2 are placed in a cluster) looks like this:
// (Profile for function foo is only loaded when its debug-info filename
// matches 'path/to/foo_file.cc').
// ----------------------------
// list.txt:
// !main
// !foo M=path/to/foo_file.cc
// !!1 2
// !!4
Error BasicBlockSectionsProfileReader::ReadProfile() {
  assert(MBuf);

  unsigned long long Version = 0;
  StringRef FirstLine(*LineIt);
  if (FirstLine.consume_front("v")) {
    if (getAsUnsignedInteger(FirstLine, 10, Version)) {
      return createProfileParseError(Twine("version number expected: '") +
                                     FirstLine + "'");
    }
    if (Version > 1) {
      return createProfileParseError(Twine("invalid profile version: ") +
                                     Twine(Version));
    }
    ++LineIt;
  }

  switch (Version) {
  case 0:
    // TODO: Deprecate V0 once V1 is fully integrated downstream.
    return ReadV0Profile();
  case 1:
    return ReadV1Profile();
  default:
    llvm_unreachable("Invalid profile version.");
  }
}

bool BasicBlockSectionsProfileReader::doInitialization(Module &M) {
  if (!MBuf)
    return false;
  // Get the function name to debug info filename mapping.
  FunctionNameToDIFilename.clear();
  for (const Function &F : M) {
    SmallString<128> DIFilename;
    if (F.isDeclaration())
      continue;
    DISubprogram *Subprogram = F.getSubprogram();
    if (Subprogram) {
      llvm::DICompileUnit *CU = Subprogram->getUnit();
      if (CU)
        DIFilename = sys::path::remove_leading_dotslash(CU->getFilename());
    }
    [[maybe_unused]] bool inserted =
        FunctionNameToDIFilename.try_emplace(F.getName(), DIFilename).second;
    assert(inserted);
  }
  if (auto Err = ReadProfile())
    report_fatal_error(std::move(Err));
  return false;
}

ImmutablePass *
llvm::createBasicBlockSectionsProfileReaderPass(const MemoryBuffer *Buf) {
  return new BasicBlockSectionsProfileReader(Buf);
}
