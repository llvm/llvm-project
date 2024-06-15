//===--- SipHash.h - An ABI-stable string SipHash ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// An implementation of SipHash, a hash function optimized for speed on
// short inputs. Based on the SipHash reference implementation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SIPHASH_H
#define LLVM_SUPPORT_SIPHASH_H

#include <cstdint>

namespace llvm {

template <typename T> class ArrayRef;

/// Computes a SipHash-2-4 64-bit result.
void getSipHash_2_4_64(ArrayRef<uint8_t> In, const uint8_t (&K)[16],
                       uint8_t (&Out)[8]);

/// Computes a SipHash-2-4 128-bit result.
void getSipHash_2_4_128(ArrayRef<uint8_t> In, const uint8_t (&K)[16],
                        uint8_t (&Out)[16]);

} // end namespace llvm

#endif
