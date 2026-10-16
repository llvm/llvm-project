//===- InternalizeLinkOnceODR.h - Clone linkonce_odr functions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_INTERNALIZELINKONCEODR_H
#define LLVM_TRANSFORMS_IPO_INTERNALIZELINKONCEODR_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Module;

/// Clone and internalize linkonce_odr functions in the current TU (when asked):
/// * -enable-linkonce-odr-internalization=off ... default, nothing
/// * -enable-linkonce-odr-internalization=likely-module-local ... when marked
///   with "frontend-hint-likely-module-local" function attribute
/// * -enable-linkonce-odr-internalization=all ... on all linkonce_odr functions
///
/// Changing calls in the current TU to instead use a private copy of a
/// linkonce_odr function is allowed, as long as nothing relies on the address
/// of the function, so only direct calls are redirected, other uses keep
/// referring to the original. When all uses of a function are direct calls, the
/// internalization is done in-place.
class InternalizeLinkOnceODRPass
    : public OptionalPassInfoMixin<InternalizeLinkOnceODRPass> {
public:
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_INTERNALIZELINKONCEODR_H
