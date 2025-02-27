//===- AttrToSPIRVConverter.cpp - GPU attributes conversion to SPIR-V - C++===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mlir/Conversion/GPUCommon/AttrToSPIRVConverter.h>

namespace mlir {
spirv::StorageClass addressSpaceToStorageClass(gpu::AddressSpace addressSpace) {
  switch (addressSpace) {
  case gpu::AddressSpace::Global:
    return spirv::StorageClass::CrossWorkgroup;
  case gpu::AddressSpace::Workgroup:
    return spirv::StorageClass::Workgroup;
  case gpu::AddressSpace::Private:
    return spirv::StorageClass::Private;
  }
  llvm_unreachable("Unhandled storage class");
}
} // namespace mlir
