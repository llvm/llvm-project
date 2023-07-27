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

#include "llvm/ADT/SmallSet.h"
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
using namespace llvm;

namespace llvm {

// The cluster information for a machine basic block.
struct BBClusterInfo {
  // Unique ID for this basic block.
  unsigned BBID;
  // Cluster ID this basic block belongs to.
  unsigned ClusterID;
  // Position of basic block within the cluster.
  unsigned PositionInCluster;
};

using ProgramBBClusterInfoMapTy = StringMap<SmallVector<BBClusterInfo>>;

class BasicBlockSectionsProfileReader : public ImmutablePass {
public:
  static char ID;

  BasicBlockSectionsProfileReader(const MemoryBuffer *Buf)
      : ImmutablePass(ID), MBuf(Buf) {
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
  getBBClusterInfoForFunction(StringRef FuncName) const;

  // Initializes the FunctionNameToDIFilename map for the current module and
  // then reads the profile for matching functions.
  bool doInitialization(Module &M) override;

private:
  StringRef getAliasName(StringRef FuncName) const {
    auto R = FuncAliasMap.find(FuncName);
    return R == FuncAliasMap.end() ? FuncName : R->second;
  }

  // Reads the basic block sections profile for functions in this module.
  Error ReadProfile();

  // This contains the basic-block-sections profile.
  const MemoryBuffer *MBuf = nullptr;

  // Map from every function name in the module to its debug info filename or
  // empty string if no debug info is available.
  StringMap<SmallString<128>> FunctionNameToDIFilename;

  // This encapsulates the BB cluster information for the whole program.
  //
  // For every function name, it contains the cluster information for (all or
  // some of) its basic blocks. The cluster information for every basic block
  // includes its cluster ID along with the position of the basic block in that
  // cluster.
  ProgramBBClusterInfoMapTy ProgramBBClusterInfo;

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
