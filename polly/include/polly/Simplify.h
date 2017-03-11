//===------ Simplify.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Simplify a SCoP by removing unnecessary statements and accesses.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_TRANSFORM_SIMPLIFY_H
#define POLLY_TRANSFORM_SIMPLIFY_H

namespace llvm {
class PassRegistry;
class Pass;
} // namespace llvm

namespace polly {
llvm::Pass *createSimplifyPass();
} // namespace polly

namespace llvm {
void initializeSimplifyPass(llvm::PassRegistry &);
} // namespace llvm

#endif /* POLLY_TRANSFORM_SIMPLIFY_H */
