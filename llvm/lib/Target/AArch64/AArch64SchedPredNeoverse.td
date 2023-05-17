//===- AArch64SchedPredNeoverse.td - AArch64 Sched Preds -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines scheduling predicate definitions that are used by the
// AArch64 Neoverse processors.
//
//===----------------------------------------------------------------------===//

// Auxiliary predicates.

// Check for LSL shift <= 4
def NeoverseCheapLSL : MCSchedPredicate<
                         CheckAll<
                           [CheckShiftLSL,
                            CheckAny<
                              [CheckShiftBy0,
                               CheckShiftBy1,
                               CheckShiftBy2,
                               CheckShiftBy3,
                               CheckShiftBy4]>]>>;

// Check for LSL shift == 0
def NeoverseNoLSL : MCSchedPredicate<
                      CheckAll<[CheckShiftLSL,
                                CheckShiftBy0]>>;
