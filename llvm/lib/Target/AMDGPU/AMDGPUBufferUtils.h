//===- AMDGPUBufferUtils.h - Buffer related helper functions ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUBUFFERUTILS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUBUFFERUTILS_H

#include <cstdint>

namespace llvm {

class DataLayout;
class GCNSubtarget;
class Value;

namespace AMDGPU {

/// Returns true if the hardware range check of a non-volatile raw buffer
/// access, applied to its first \p Payload bytes, provably fails. With a
/// \p Payload of 1, the whole access is out of bounds, so a load returns zero
/// and a store is dropped. Returns false unless \p Rsrc is an
/// llvm.amdgcn.make.buffer.rsrc call with constant stride, num_records and
/// flags.
bool isRawBufferAccessDefinitelyOOB(const GCNSubtarget &ST, Value *Rsrc,
                                    Value *Offset, Value *SOffset, Value *Aux,
                                    uint64_t Payload, const DataLayout &DL);

} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUBUFFERUTILS_H
