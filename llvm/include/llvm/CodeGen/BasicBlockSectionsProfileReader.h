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
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/UniqueBBID.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

// This struct represents the cluster information for a machine basic block,
// which is specifed by a unique basic block ID.
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
  // Node counts for each basic block.
  DenseMap<UniqueBBID, uint64_t> NodeCounts;
  // Edge counts for each edge, stored as a nested map.
  DenseMap<UniqueBBID, DenseMap<UniqueBBID, uint64_t>> EdgeCounts;
  // Hash for each basic block. The Hashes are stored for every original block
  // (not cloned blocks), hence the map key being unsigned instead of
  // UniqueBBID.
  DenseMap<unsigned, uint64_t> BBHashes;
};

class BasicBlockSectionsProfileReader {
public:
  friend class BasicBlockSectionsProfileReaderWrapperPass;
  BasicBlockSectionsProfileReader(const MemoryBuffer *Buf)
      : MBuf(Buf), LineIt(*Buf, /*SkipBlanks=*/true, /*CommentMarker=*/'#'){};

  BasicBlockSectionsProfileReader() = default;

  // Returns true if function \p FuncName is hot based on the basic block
  // section profile.
  bool isFunctionHot(StringRef FuncName) const;

  // Returns the cluster info for the function \p FuncName. Returns an empty
  // vector if function has no cluster info.
  SmallVector<BBClusterInfo>
  getClusterInfoForFunction(StringRef FuncName) const;

  // Returns the path clonings for the given function.
  SmallVector<SmallVector<unsigned>>
  getClonePathsForFunction(StringRef FuncName) const;

  // Returns the profile count for the edge from `SrcBBID` to `SinkBBID` in
  // function `FuncName` or zero if it does not exist.
  uint64_t getEdgeCount(StringRef FuncName, const UniqueBBID &SrcBBID,
                        const UniqueBBID &SinkBBID) const;

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
createBasicBlockSectionsProfileReaderWrapperPass(const MemoryBuffer *Buf);

/// Analysis pass providing the \c BasicBlockSectionsProfileReader.
///
/// Note that this pass's result cannot be invalidated, it is immutable for the
/// life of the module.
class BasicBlockSectionsProfileReaderAnalysis
    : public AnalysisInfoMixin<BasicBlockSectionsProfileReaderAnalysis> {

public:
  static AnalysisKey Key;
  typedef BasicBlockSectionsProfileReader Result;
  BasicBlockSectionsProfileReaderAnalysis(const TargetMachine &TM) : TM(&TM) {}

  Result run(Function &F, FunctionAnalysisManager &AM);

private:
  const TargetMachine *TM;
};

class BasicBlockSectionsProfileReaderWrapperPass : public ImmutablePass {
public:
  static char ID;
  BasicBlockSectionsProfileReader BBSPR;

  BasicBlockSectionsProfileReaderWrapperPass(const MemoryBuffer *Buf)
      : ImmutablePass(ID), BBSPR(BasicBlockSectionsProfileReader(Buf)) {
    initializeBasicBlockSectionsProfileReaderWrapperPassPass(
        *PassRegistry::getPassRegistry());
  };

  BasicBlockSectionsProfileReaderWrapperPass()
      : ImmutablePass(ID), BBSPR(BasicBlockSectionsProfileReader()) {
    initializeBasicBlockSectionsProfileReaderWrapperPassPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Basic Block Sections Profile Reader";
  }

  bool isFunctionHot(StringRef FuncName) const;

  SmallVector<BBClusterInfo>
  getClusterInfoForFunction(StringRef FuncName) const;

  SmallVector<SmallVector<unsigned>>
  getClonePathsForFunction(StringRef FuncName) const;

  uint64_t getEdgeCount(StringRef FuncName, const UniqueBBID &SrcBBID,
                        const UniqueBBID &DestBBID) const;

  // Initializes the FunctionNameToDIFilename map for the current module and
  // then reads the profile for the matching functions.
  bool doInitialization(Module &M) override;

  BasicBlockSectionsProfileReader &getBBSPR();
};

} // namespace llvm
#endif // LLVM_CODEGEN_BASICBLOCKSECTIONSPROFILEREADER_H
