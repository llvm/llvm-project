//===- SafepointIRVerifier.h - Checks for GC relocation problems *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a verifier which is useful for enforcing the relocation
// properties required by a relocating GC.  Specifically, it looks for uses of
// the unrelocated value of pointer SSA values after a possible safepoint. It
// attempts to report no false negatives, but may end up reporting false
// positives in rare cases (see the note at the top of the corresponding cpp
// file.)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_SAFEPOINT_IR_VERIFIER
#define LLVM_IR_SAFEPOINT_IR_VERIFIER

namespace llvm {

class Function;
class FunctionPass;

/// Run the safepoint verifier over a single function.  Crashes on failure.
void verifySafepointIR(Function &F);

/// Create an instance of the safepoint verifier pass which can be added to
/// a pass pipeline to check for relocation bugs.
FunctionPass *createSafepointIRVerifierPass();
}

#endif // LLVM_IR_SAFEPOINT_IR_VERIFIER
