//===---- llvm/CodeGen/DebugLoc.h - Debug Location Information --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a number of light weight data structures used by the code
// generator to describe and track debug location information.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DEBUGLOC_H
#define LLVM_CODEGEN_DEBUGLOC_H

#include "llvm/ADT/DenseMap.h"
#include <vector>

namespace llvm {

  /// DebugLocTuple - Debug location tuple of filename id, line and column.
  ///
  struct DebugLocTuple {
    unsigned Src, Line, Col;

    DebugLocTuple(unsigned s, unsigned l, unsigned c)
      : Src(s), Line(l), Col(c) {};
  };

  /// DebugLoc - Debug location id. This is carried by SDNode and MachineInstr
  /// to index into a vector of unique debug location tuples.
  class DebugLoc {
    unsigned Idx;

  public:
    DebugLoc() : Idx(~0U) {}  // Defaults to invalid.

    static DebugLoc getUnknownLoc()   { DebugLoc L; L.Idx = 0;   return L; }
    static DebugLoc get(unsigned idx) { DebugLoc L; L.Idx = idx; return L; }

    /// isInvalid - Return true if the DebugLoc is invalid.
    bool isInvalid() const { return Idx == ~0U; }

    /// isUnknown - Return true if there is no debug info for the SDNode /
    /// MachineInstr.
    bool isUnknown() const { return Idx == 0; }
  };

  // Partially specialize DenseMapInfo for DebugLocTyple.
  template<>  struct DenseMapInfo<DebugLocTuple> {
    static inline DebugLocTuple getEmptyKey() {
      return DebugLocTuple(~0U, ~0U, ~0U);
    }
    static inline DebugLocTuple getTombstoneKey() {
      return DebugLocTuple(~1U, ~1U, ~1U);
    }
    static unsigned getHashValue(const DebugLocTuple &Val) {
      return DenseMapInfo<unsigned>::getHashValue(Val.Src) ^
             DenseMapInfo<unsigned>::getHashValue(Val.Line) ^
             DenseMapInfo<unsigned>::getHashValue(Val.Col);
    }
    static bool isEqual(const DebugLocTuple &LHS, const DebugLocTuple &RHS) {
      return LHS.Src  == RHS.Src &&
             LHS.Line == RHS.Line &&
             LHS.Col  == RHS.Col;
    }

    static bool isPod() { return true; }
  };

  /// DebugLocTracker - This class tracks debug location information.
  ///
  struct DebugLocTracker {
    /// DebugLocations - A vector of unique DebugLocTuple's.
    ///
    std::vector<DebugLocTuple> DebugLocations;

    /// DebugIdsMap - This maps DebugLocTuple's to indices into DebugLocations
    /// vector.
    DenseMap<DebugLocTuple, unsigned> DebugIdMap;

    DebugLocTracker() {}

    ~DebugLocTracker() {
      DebugLocations.clear();
      DebugIdMap.clear();
    }
  };
  
} // end namespace llvm

#endif /* LLVM_CODEGEN_DEBUGLOC_H */
