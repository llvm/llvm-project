//===- PruneUnprofitable.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mark a SCoP as unfeasible if not deemed profitable to optimize.
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_PRUNEUNPROFITABLE_H
#define POLLY_PRUNEUNPROFITABLE_H

namespace polly {
class Scop;

bool runPruneUnprofitable(Scop &S);
} // namespace polly

#endif // POLLY_PRUNEUNPROFITABLE_H
