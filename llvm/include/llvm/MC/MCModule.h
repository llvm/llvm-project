//===-- llvm/MC/MCModule.h - MCModule class ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCModule class, which is used to
// represent a complete, disassembled object file or executable.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCMODULE_H
#define LLVM_MC_MCMODULE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class MCAtom;

/// MCModule - This class represent a completely disassembled object file or
/// executable.  It comprises a list of MCAtom's, and a branch target table.
/// Each atom represents a contiguous range of either instructions or data.
class MCModule {
  /// AtomAllocationTracker - An MCModule owns its component MCAtom's, so it
  /// must track them in order to ensure they are properly freed as atoms are
  /// merged or otherwise manipulated.
  SmallPtrSet<MCAtom*, 8> AtomAllocationTracker;

  /// OffsetMap - Efficiently maps offset ranges to MCAtom's.
  IntervalMap<uint64_t, MCAtom*> OffsetMap;

  /// BranchTargetMap - Maps offsets that are determined to be branches and
  /// can be statically resolved to their target offsets.
  DenseMap<uint64_t, MCAtom*> BranchTargetMap;

  friend class MCAtom;

  /// remap - Update the interval mapping for an MCAtom.
  void remap(MCAtom *Atom, uint64_t NewBegin, uint64_t NewEnd);

public:
  MCModule(IntervalMap<uint64_t, MCAtom*>::Allocator &A) : OffsetMap(A) { }

  /// createAtom - Creates a new MCAtom covering the specified offset range.
  MCAtom *createAtom(MCAtom::AtomType Type, uint64_t Begin, uint64_t End);
};

}

#endif

