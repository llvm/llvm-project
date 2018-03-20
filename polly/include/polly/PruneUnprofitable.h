//===- PruneUnprofitable.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Mark a SCoP as unfeasible if not deemed profitable to optimize.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_PRUNEUNPROFITABLE_H
#define POLLY_PRUNEUNPROFITABLE_H

namespace llvm {

class Pass;
class PassRegistry;

void initializePruneUnprofitablePass(PassRegistry &);
} // namespace llvm

namespace polly {

llvm::Pass *createPruneUnprofitablePass();
} // namespace polly

#endif // POLLY_PRUNEUNPROFITABLE_H
