//===- AArch64SchedPredAmpere.td - AArch64 Sched Preds -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines scheduling predicate definitions that are used by the
// AArch64 Ampere Computing processors.
//
//===----------------------------------------------------------------------===//

// Auxiliary predicates.

// Check for a LSL shift <= 4
def AmpereCheapLSL : MCSchedPredicate<
                                CheckAny<[CheckShiftBy0,
                                 CheckAll<
                                   [CheckShiftLSL,
                                    CheckAny<
                                      [CheckShiftBy1,
                                       CheckShiftBy2,
                                       CheckShiftBy3,
                                       CheckShiftBy4]>]>]>>;
