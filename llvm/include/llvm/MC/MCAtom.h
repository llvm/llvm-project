//===-- llvm/MC/MCAtom.h - MCAtom class ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCAtom class, which is used to
// represent a contiguous region in a decoded object that is uniformly data or
// instructions;
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCATOM_H
#define LLVM_MC_MCATOM_H

#include "llvm/MC/MCInst.h"
#include "llvm/Support/DataTypes.h"
#include <vector>

namespace llvm {

class MCModule;

/// MCData - An entry in a data MCAtom.
// NOTE: This may change to a more complex type in the future.
typedef uint8_t MCData;

/// MCAtom - Represents a contiguous range of either instructions (a TextAtom)
/// or data (a DataAtom).  Address ranges are expressed as _closed_ intervals.
class MCAtom {
  friend class MCModule;
  typedef enum { TextAtom, DataAtom } AtomType;

  AtomType Type;
  MCModule *Parent;
  uint64_t Begin, End;

  std::vector<std::pair<uint64_t, MCInst> > Text;
  std::vector<MCData> Data;

  // Private constructor - only callable by MCModule
  MCAtom(AtomType T, MCModule *P, uint64_t B, uint64_t E)
    : Type(T), Parent(P), Begin(B), End(E) { }

public:
  bool isTextAtom() { return Type == TextAtom; }
  bool isDataAtom() { return Type == DataAtom; }

  void addInst(const MCInst &I, uint64_t Address, unsigned Size);
  void addData(const MCData &D);

  /// split - Splits the atom in two at a given address, which must align with
  /// and instruction boundary if this is a TextAtom.  Returns the newly created
  /// atom representing the high part of the split.
  MCAtom *split(uint64_t SplitPt);

  /// truncate - Truncates an atom so that TruncPt is the last byte address
  /// contained in the atom.
  void truncate(uint64_t TruncPt);
};

}

#endif

