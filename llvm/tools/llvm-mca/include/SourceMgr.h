//===--------------------- SourceMgr.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements class SourceMgr. Class SourceMgr abstracts the input
/// code sequence (a sequence of MCInst), and assings unique identifiers to
/// every instruction in the sequence.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_SOURCEMGR_H
#define LLVM_TOOLS_LLVM_MCA_SOURCEMGR_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCInst.h"
#include <vector>

namespace mca {

typedef std::pair<unsigned, const llvm::MCInst &> SourceRef;

class SourceMgr {
  llvm::ArrayRef<llvm::MCInst> Sequence;
  unsigned Current;
  const unsigned Iterations;
  static const unsigned DefaultIterations = 100;

public:
  SourceMgr(llvm::ArrayRef<llvm::MCInst> MCInstSequence, unsigned NumIterations)
      : Sequence(MCInstSequence), Current(0),
        Iterations(NumIterations ? NumIterations : DefaultIterations) {}

  unsigned getNumIterations() const { return Iterations; }
  unsigned size() const { return Sequence.size(); }
  bool hasNext() const { return Current < (Iterations * Sequence.size()); }
  void updateNext() { ++Current; }

  const SourceRef peekNext() const {
    assert(hasNext() && "Already at end of sequence!");
    return SourceRef(Current, Sequence[Current % Sequence.size()]);
  }

  using const_iterator = llvm::ArrayRef<llvm::MCInst>::const_iterator;
  const_iterator begin() const { return Sequence.begin(); }
  const_iterator end() const { return Sequence.end(); }

  bool isEmpty() const { return size() == 0; }
};
} // namespace mca

#endif
