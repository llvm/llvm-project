//===- ForwardOpTree.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Move instructions between statements.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_FORWARDOPTREE_H
#define POLLY_FORWARDOPTREE_H

namespace llvm {

class PassRegistry;

void initializeForwardOpTreePass(PassRegistry &);
} // namespace llvm

namespace polly {

class ScopPass;

ScopPass *createForwardOpTreePass();
} // namespace polly

#endif // POLLY_FORWARDOPTREE_H
