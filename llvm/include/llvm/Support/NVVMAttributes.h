//===--- NVVMAttributes.h - NVVM IR attribute names -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Canonical string constants for NVVM function and parameter attributes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_NVVMATTRIBUTES_H
#define LLVM_SUPPORT_NVVMATTRIBUTES_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
namespace NVVMAttr {

constexpr StringLiteral MaxNTID("nvvm.maxntid");
constexpr StringLiteral ReqNTID("nvvm.reqntid");
constexpr StringLiteral ClusterDim("nvvm.cluster_dim");
constexpr StringLiteral MaxClusterRank("nvvm.maxclusterrank");
constexpr StringLiteral MinCTASm("nvvm.minctasm");
constexpr StringLiteral MaxNReg("nvvm.maxnreg");
constexpr StringLiteral BlocksAreClusters("nvvm.blocksareclusters");
constexpr StringLiteral GridConstant("nvvm.grid_constant");

// Custom ABI attributes: the number of registers a function preserves across a
// call. Valid on function definitions and declarations, and on callsites for
// indirect calls.
// Ref:
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#performance-tuning-directives-abi-preserve
constexpr StringLiteral PreserveNData("nvvm.preserve_n_data");
constexpr StringLiteral PreserveNControl("nvvm.preserve_n_control");

} // namespace NVVMAttr
} // namespace llvm

#endif // LLVM_SUPPORT_NVVMATTRIBUTES_H
