//===------ ForwardOpTree.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Move instructions between statements.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_FORWARDOPTREE_H
#define POLLY_FORWARDOPTREE_H

namespace polly {
class ScopPass;

ScopPass *createForwardOpTreePass();
} // namespace polly

namespace llvm {
class PassRegistry;
void initializeForwardOpTreePass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_FORWARDOPTREE_H */
