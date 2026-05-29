//===- MetalAddressSpaces.h - Metal address-space constants -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Single source of truth for the AIR address-space numbering used by every
/// pass and writer in this target.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_METAL_METALADDRESSSPACES_H
#define LLVM_LIB_TARGET_METAL_METALADDRESSSPACES_H

namespace llvm {
namespace metal {
namespace AS {

constexpr unsigned Default = 0;
constexpr unsigned Device = 1;
constexpr unsigned Constant = 2;
constexpr unsigned Threadgroup = 3;

} // namespace AS
} // namespace metal
} // namespace llvm

#endif // LLVM_LIB_TARGET_METAL_METALADDRESSSPACES_H
