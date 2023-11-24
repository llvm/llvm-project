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
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
using namespace llvm;

namespace llvm {

// This struct represents the cluster information for a machine basic block,
// which is specifed by a unique ID (`MachineBasicBlock::BBID`).
struct BBClusterInfo {
  // Basic block ID.
  UniqueBBID BBID;
  // Cluster ID this basic block belongs to.
  unsigned ClusterID;
  // Position of basic block within the cluster.
  unsigned PositionInCluster;
};

// This represents the raw input profile for one function.
struct FunctionPathAndClusterInfo {
  // BB Cluster information specified by `UniqueBBID`s.
  SmallVector<BBClusterInfo> ClusterInfo;
  // Paths to clone. A path a -> b -> c -> d implies cloning b, c, and d along
  // the edge a -> b (a is not cloned). The index of the path in this vector
  // determines the `UniqueBBID::CloneID` of the cloned blocks in that path.
  SmallVector<SmallVector<unsigned>> ClonePaths;
};

// Provides DenseMapInfo for UniqueBBID.
template <> struct DenseMapInfo<UniqueBBID> {
  static inline UniqueBBID getEmptyKey() {
    unsigned EmptyKey = DenseMapInfo<unsigned>::getEmptyKey();
    return UniqueBBID{EmptyKey, EmptyKey};
  }
  static inline UniqueBBID getTombstoneKey() {
    unsigned TombstoneKey = DenseMapInfo<unsigned>::getTombstoneKey();
    return UniqueBBID{TombstoneKey, TombstoneKey};
  }
  static unsigned getHashValue(const UniqueBBID &Val) {
    std::pair<unsigned, unsigned> PairVal =
        std::make_pair(Val.BaseID, Val.CloneID);
    return DenseMapInfo<std::pair<unsigned, unsigned>>::getHashValue(PairVal);
  }
  static bool isEqual(const UniqueBBID &LHS, const UniqueBBID &RHS) {
    return DenseMapInfo<unsigned>::isEqual(LHS.BaseID, RHS.BaseID) &&
           DenseMapInfo<unsigned>::isEqual(LHS.CloneID, RHS.CloneID);
  }
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
  std::pair<bool, SmallVector<BBClusterInfo>>
  getClusterInfoForFunction(StringRef FuncName) const;

  // Returns the path clonings for the given function.
  SmallVector<SmallVector<unsigned>>
  getClonePathsForFunction(StringRef FuncName) const;

  // Initializes the FunctionNameToDIFilename map for the current module and
  // then reads the profile for the matching functions.
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

  // Parses a `UniqueBBID` from `S`. `S` must be in the form "<bbid>"
  // (representing an original block) or "<bbid>.<cloneid>" (representing a
  // cloned block) where bbid is a non-negative integer and cloneid is a
  // positive integer.
  Expected<UniqueBBID> parseUniqueBBID(StringRef S) const;

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

  // This contains the BB cluster information for the whole program.
  //
  // For every function name, it contains the cloning and cluster information
  // for (all or some of) its basic blocks. The cluster information for every
  // basic block includes its cluster ID along with the position of the basic
  // block in that cluster.
  StringMap<FunctionPathAndClusterInfo> ProgramPathAndClusterInfo;

  // Some functions have alias names. We use this map to find the main alias
  // name which appears in ProgramPathAndClusterInfo as a key.
  StringMap<StringRef> FuncAliasMap;
};

// Creates a BasicBlockSectionsProfileReader pass to parse the basic block
// sections profile. \p Buf is a memory buffer that contains the list of
// functions and basic block ids to selectively enable basic block sections.
ImmutablePass *
createBasicBlockSectionsProfileReaderPass(const MemoryBuffer *Buf);

} // namespace llvm
#endif // LLVM_CODEGEN_BASICBLOCKSECTIONSPROFILEREADER_H
