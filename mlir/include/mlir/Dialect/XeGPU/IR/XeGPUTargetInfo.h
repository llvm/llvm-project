//===- XeGPUTargetInfo.h - Target constants ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_IR_XEGPUTARGETINFO_H_
#define MLIR_DIALECT_XEGPU_IR_XEGPUTARGETINFO_H_

namespace mlir {
namespace xegpu {
/// HW dependent constants.
/// TODO: These constants should be queried from the target information.
namespace targetinfo {
constexpr unsigned subgroupSize = 16; // How many lanes in a subgroup.
/// If DPAS A or B operands have low precision element types they must be packed
/// according to the following sizes.
constexpr unsigned packedSizeInBitsForDefault =
    16; // Minimum packing size per register for DPAS A.
constexpr unsigned packedSizeInBitsForDpasB =
    32; // Minimum packing size per register for DPAS B.
constexpr unsigned packedSizeInBitsForGatherScatter =
    32; // Minimum packing size per register for Gather and Scatter ops.
} // namespace targetinfo
} // namespace xegpu
} // namespace mlir

#endif // MLIR_DIALECT_XEGPU_IR_XEGPUTARGETINFO_H_
