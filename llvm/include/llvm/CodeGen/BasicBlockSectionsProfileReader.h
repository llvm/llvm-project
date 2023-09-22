//===-- BasicBlockSectionsProfileReader.h - BB sections profile reader pass ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass creates the basic block cluster info by reading the basic block
// sections profile. The cluster info will be used by the basic-block-sections
// pass to arrange basic blocks in their sections.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_BASICBLOCKSECTIONSPROFILEREADER_H
#define LLVM_CODEGEN_BASICBLOCKSECTIONSPROFILEREADER_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

namespace llvm {

// This structure represents a unique ID for every block specified in the
// profile.
struct ProfileBBID {
  unsigned BBID;
  unsigned CloneID;
};

// Provides DenseMapInfo for ProfileBBID.
template <> struct DenseMapInfo<ProfileBBID> {
  static inline ProfileBBID getEmptyKey() {
    unsigned EmptyKey = DenseMapInfo<unsigned>::getEmptyKey();
    return ProfileBBID{EmptyKey, EmptyKey};
  }
  static inline ProfileBBID getTombstoneKey() {
    unsigned TombstoneKey = DenseMapInfo<unsigned>::getTombstoneKey();
    return ProfileBBID{TombstoneKey, TombstoneKey};
  }
  static unsigned getHashValue(const ProfileBBID &Val) {
    std::pair<unsigned, unsigned> PairVal =
        std::make_pair(Val.BBID, Val.CloneID);
    return DenseMapInfo<std::pair<unsigned, unsigned>>::getHashValue(PairVal);
  }
  static bool isEqual(const ProfileBBID &LHS, const ProfileBBID &RHS) {
    return DenseMapInfo<unsigned>::isEqual(LHS.BBID, RHS.BBID) &&
           DenseMapInfo<unsigned>::isEqual(LHS.CloneID, RHS.CloneID);
  }
};

// This struct represents the cluster information for a machine basic block,
// which is specifed by a unique ID.
template <typename BBIDType> struct BBProfile {
  // Basic block ID.
  BBIDType BasicBlockID;
  // Cluster ID this basic block belongs to.
  unsigned ClusterID;
  // Position of basic block within the cluster.
  unsigned PositionInCluster;
};

// This represents the profile for one function.
struct RawFunctionProfile {
  // BB Cluster information specified by `ProfileBBID`s (before cloning).
  SmallVector<BBProfile<ProfileBBID>> RawBBProfiles;
  // Paths to clone. A path a -> b -> c -> d implies cloning b, c, and d along
  // the edge a -> b.
  SmallVector<SmallVector<unsigned>> ClonePaths;
};

class BasicBlockSectionsProfileReader : public ImmutablePass {
public:
  static char ID;

  BasicBlockSectionsProfileReader(const MemoryBuffer *Buf)
      : ImmutablePass(ID), MBuf(Buf),
        LineIt(*Buf, /*SkipBlanks=*/true, /*CommentMarker=*/'#') {
    initializeBasicBlockSectionsProfileReaderPass(
        *PassRegistry::getPassRegistry());
  };

  BasicBlockSectionsProfileReader() : ImmutablePass(ID) {
    initializeBasicBlockSectionsProfileReaderPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Basic Block Sections Profile Reader";
  }

  // Returns true if basic block sections profile exist for function \p
  // FuncName.
  bool isFunctionHot(StringRef FuncName) const;

  // Returns a pair with first element representing whether basic block sections
  // profile exist for the function \p FuncName, and the second element
  // representing the basic block sections profile (cluster info) for this
  // function. If the first element is true and the second element is empty, it
  // means unique basic block sections are desired for all basic blocks of the
  // function.
  std::pair<bool, RawFunctionProfile>
  getRawProfileForFunction(StringRef FuncName) const;

  // Initializes the FunctionNameToDIFilename map for the current module and
  // then reads the profile for matching functions.
  bool doInitialization(Module &M) override;

private:
  StringRef getAliasName(StringRef FuncName) const {
    auto R = FuncAliasMap.find(FuncName);
    return R == FuncAliasMap.end() ? FuncName : R->second;
  }

  // Returns a profile parsing error for the current line.
  Error createProfileParseError(Twine Message) const {
    return make_error<StringError>(
        Twine("invalid profile " + MBuf->getBufferIdentifier() + " at line " +
              Twine(LineIt.line_number()) + ": " + Message),
        inconvertibleErrorCode());
  }

  // Parses a `ProfileBBID` from `S`.
  Expected<ProfileBBID> parseProfileBBID(StringRef S) const;

  // Reads the basic block sections profile for functions in this module.
  Error ReadProfile();

  // Reads version 0 profile.
  // TODO: Remove this function once version 0 is deprecated.
  Error ReadV0Profile();

  // Reads version 1 profile.
  Error ReadV1Profile();

  // This contains the basic-block-sections profile.
  const MemoryBuffer *MBuf = nullptr;

  // Iterator to the line being parsed.
  line_iterator LineIt;

  // Map from every function name in the module to its debug info filename or
  // empty string if no debug info is available.
  StringMap<SmallString<128>> FunctionNameToDIFilename;

  // This encapsulates the BB cluster information for the whole program.
  //
  // For every function name, it contains the cloning and cluster information
  // for (all or some of) its basic blocks. The cluster information for every
  // basic block includes its cluster ID along with the position of the basic
  // block in that cluster.
  StringMap<RawFunctionProfile> RawProgramProfile;

  // Some functions have alias names. We use this map to find the main alias
  // name for which we have mapping in ProgramBBClusterInfo.
  StringMap<StringRef> FuncAliasMap;
};

// Creates a BasicBlockSectionsProfileReader pass to parse the basic block
// sections profile. \p Buf is a memory buffer that contains the list of
// functions and basic block ids to selectively enable basic block sections.
ImmutablePass *
createBasicBlockSectionsProfileReaderPass(const MemoryBuffer *Buf);

} // namespace llvm
#endif // LLVM_CODEGEN_BASICBLOCKSECTIONSPROFILEREADER_H
