//===-- X86ShuffleMatch.h - X86 Shuffle Pattern Matching --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines shared shuffle pattern matching functions that can be used
// by both SelectionDAG and GlobalISel lowering.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86SHUFFLEMATCH_H
#define LLVM_LIB_TARGET_X86_X86SHUFFLEMATCH_H

#include "llvm/ADT/ArrayRef.h"

namespace llvm {
namespace X86 {

/// Check if a shuffle mask matches a SHUFPS/SHUFPD pattern.
/// Returns true if the mask can be implemented with SHUFP, filling in Imm
/// with the immediate value and Swap with whether operands should be swapped.
bool matchShufpMask(ArrayRef<int> Mask, unsigned NumElts, unsigned NumSrcElts,
                    unsigned EltSize, bool SingleSource, unsigned &Imm,
                    bool &Swap);

/// Check if a shuffle mask is a simple broadcast pattern.
bool isBroadcastMask(ArrayRef<int> Mask);

/// Check if a shuffle mask matches a blend pattern where each element comes
/// from either src1[i] or src2[i].
bool matchBlendMask(ArrayRef<int> Mask, unsigned NumElts, unsigned NumSrcElts);

/// Check if a shuffle mask matches UNPCKL pattern.
bool matchUnpackLowMask(ArrayRef<int> Mask, unsigned NumElts,
                        unsigned NumSrcElts, unsigned EltSize, bool &Swap);

/// Check if a shuffle mask matches UNPCKH pattern.
bool matchUnpackHighMask(ArrayRef<int> Mask, unsigned NumElts,
                         unsigned NumSrcElts, unsigned EltSize, bool &Swap);

/// Check if a shuffle mask can be implemented with PSHUFD.
/// Returns true and fills in Imm if successful.
bool matchPshufdMask(ArrayRef<int> Mask, unsigned NumElts, unsigned &Imm);

/// Check if a shuffle mask can be implemented with PSHUFB.
/// Returns true if the mask is byte-aligned and doesn't cross 128-bit lanes.
bool matchPshufbMask(ArrayRef<int> Mask, unsigned NumElts, unsigned EltSize);

/// Check if shuffle mask can be implemented as VPERMILPS/VPERMILPD.
/// Returns true and fills in Imm for immediate form, or returns true with
/// Imm=-1 for variable mask form.
bool matchVPermilMask(ArrayRef<int> Mask, unsigned NumElts, unsigned EltSize,
                      int &Imm);

/// Check if shuffle mask can be implemented as VPERMQ/VPERMPD (AVX2 cross-lane).
bool matchVPermiMask(ArrayRef<int> Mask, unsigned NumElts, unsigned &Imm);

} // namespace X86
} // namespace llvm

#endif
