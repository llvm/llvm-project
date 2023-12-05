//===---------------- AMDGPUAddrSpace.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// AMDGPU address space definition
///
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_AMDGPUADDRSPACE_H
#define LLVM_SUPPORT_AMDGPUADDRSPACE_H

namespace llvm {
namespace AMDGPU {
enum class AddrSpace {
  Generic = 0,
  Global = 1,
  Local = 3,
  Constant = 4,
  Private = 5
};

} // end namespace AMDGPU
} // end namespace llvm

#endif // LLVM_SUPPORT_AMDGPUADDRSPACE_H
