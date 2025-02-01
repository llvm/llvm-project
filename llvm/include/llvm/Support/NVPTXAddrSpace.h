//===---------------- NVPTXAddrSpace.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// NVPTX address space definition
///
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_NVPTXADDRSPACE_H
#define LLVM_SUPPORT_NVPTXADDRSPACE_H

namespace llvm {
namespace NVPTXAS {
enum AddressSpace : unsigned {
  ADDRESS_SPACE_GENERIC = 0,
  ADDRESS_SPACE_GLOBAL = 1,
  ADDRESS_SPACE_SHARED = 3,
  ADDRESS_SPACE_CONST = 4,
  ADDRESS_SPACE_LOCAL = 5,
  ADDRESS_SPACE_TENSOR = 6,

  ADDRESS_SPACE_PARAM = 101,
};
} // end namespace NVPTXAS

} // end namespace llvm

#endif // LLVM_SUPPORT_NVPTXADDRSPACE_H
