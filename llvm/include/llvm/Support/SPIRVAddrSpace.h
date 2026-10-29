//===---------------- SPIRVAddrSpace.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// SPIR-V address space definition
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SPIRVADDRSPACE_H
#define LLVM_SUPPORT_SPIRVADDRSPACE_H

namespace llvm {
namespace SPIRVAS {

/// LLVM IR address space numbers used by the SPIR-V target. Each address
/// space maps to the SPIR-V storage class it is named after. This is kept
/// separate from SPIRV::StorageClass, which is tablegen'd inside the SPIR-V
/// backend, so that it stays dependency-free and usable from clang and other
/// non-backend code.
enum AddressSpace : unsigned {
  Function = 0,
  CrossWorkgroup = 1,
  UniformConstant = 2,
  Workgroup = 3,
  Generic = 4,
  // TODO: DeviceOnlyINTEL and HostOnlyINTEL back Clang's global_device and
  // global_host address spaces, which are deprecated (#203569). Remove these
  // entries once those address spaces are removed.
  DeviceOnlyINTEL = 5,
  HostOnlyINTEL = 6,
  Input = 7,
  Output = 8,
  CodeSectionINTEL = 9,
  Private = 10,
  StorageBuffer = 11,
  Uniform = 12,
  PushConstant = 13,
};

} // end namespace SPIRVAS
} // end namespace llvm

#endif // LLVM_SUPPORT_SPIRVADDRSPACE_H
