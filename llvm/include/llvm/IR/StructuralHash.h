//===- llvm/IR/StructuralHash.h - IR Hash for expensive checks --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides hashing of the LLVM IR structure to be used to check
// Passes modification status.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_STRUCTURALHASH_H
#define LLVM_IR_STRUCTURALHASH_H

#ifdef EXPENSIVE_CHECKS

#include <cstdint>

// This header is only meant to be used when -DEXPENSIVE_CHECKS is set
namespace llvm {

class Function;
class Module;

uint64_t StructuralHash(const Function &F);
uint64_t StructuralHash(const Module &M);

} // end namespace llvm

#endif

#endif // LLVM_IR_STRUCTURALHASH_H
