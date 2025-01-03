//===- AttrToLLVMConverter.cpp - SPIR-V attributes conversion to LLVM -C++ ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mlir/Conversion/SPIRVCommon/AttrToLLVMConverter.h>

namespace mlir {
namespace {

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

constexpr unsigned defaultAddressSpace = 0;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

static unsigned
storageClassToOCLAddressSpace(spirv::StorageClass storageClass) {
  // Based on
  // https://registry.khronos.org/SPIR-V/specs/unified1/OpenCL.ExtendedInstructionSet.100.html#_binary_form
  // and clang/lib/Basic/Targets/SPIR.h.
  switch (storageClass) {
  case spirv::StorageClass::Function:
    return 0;
  case spirv::StorageClass::Input:
  case spirv::StorageClass::CrossWorkgroup:
    return 1;
  case spirv::StorageClass::UniformConstant:
    return 2;
  case spirv::StorageClass::Workgroup:
    return 3;
  case spirv::StorageClass::Generic:
    return 4;
  case spirv::StorageClass::DeviceOnlyINTEL:
    return 5;
  case spirv::StorageClass::HostOnlyINTEL:
    return 6;
  default:
    return defaultAddressSpace;
  }
}
} // namespace

unsigned storageClassToAddressSpace(spirv::ClientAPI clientAPI,
                                    spirv::StorageClass storageClass) {
  switch (clientAPI) {
  case spirv::ClientAPI::OpenCL:
    return storageClassToOCLAddressSpace(storageClass);
  default:
    return defaultAddressSpace;
  }
}
} // namespace mlir
