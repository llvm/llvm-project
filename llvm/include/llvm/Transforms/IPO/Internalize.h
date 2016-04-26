//====- Internalize.h - Internalization API ---------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_INTERNALIZE_H
#define LLVM_TRANSFORMS_IPO_INTERNALIZE_H

#include "llvm/IR/GlobalValue.h"

#include <functional>

namespace llvm {
class Module;
class CallGraph;

bool internalizeModule(
    Module &TheModule,
    const std::function<bool(const GlobalValue &)> &MustPreserveGV,
    CallGraph *CG = nullptr);
}

#endif // LLVM_TRANSFORMS_IPO_INTERNALIZE_H
