//===--------- ScopPass.h - Pass for Static Control Parts --------*-C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ScopPass class.  ScopPasses are just RegionPasses,
// except they operate on Polly IR (Scop and ScopStmt) built by ScopInfo Pass.
// Because they operate on Polly IR, not the LLVM IR, ScopPasses are not allowed
// to modify the LLVM IR. Due to this limitation, the ScopPass class takes
// care of declaring that no LLVM passes are invalidated.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_SCOP_PASS_H
#define POLLY_SCOP_PASS_H

#include "llvm/Analysis/RegionPass.h"

using namespace llvm;

struct isl_ctx;

namespace polly {
class Scop;

/// ScopPass - This class adapts the RegionPass interface to allow convenient
/// creation of passes that operate on the Polly IR. Instead of overriding
/// runOnRegion, subclasses override runOnScop.
class ScopPass : public RegionPass {
  Scop *S;

protected:
  explicit ScopPass(char &ID) : RegionPass(ID), S(0) {}

  /// runOnScop - This method must be overloaded to perform the
  /// desired Polyhedral transformation or analysis.
  ///
  virtual bool runOnScop(Scop &S) = 0;

  /// getAnalysisUsage - Subclasses that override getAnalysisUsage
  /// must call this.
  ///
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

public:
  Scop &getCurScop() const {
    assert(S && "Not on a Scop!");
    return *S;
  }

private:
  virtual bool runOnRegion(Region *R, RGPassManager &RGM);
  void print(raw_ostream &OS, const Module *) const;
  virtual void printScop(raw_ostream &OS) const {}
};

} // End llvm namespace

#endif
