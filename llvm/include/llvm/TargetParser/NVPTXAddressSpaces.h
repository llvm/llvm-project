//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Address space definitions for NVPTX target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_NVPTX_ADDRESS_SPACES_H
#define LLVM_TARGETPARSER_NVPTX_ADDRESS_SPACES_H

namespace llvm::NVPTX {

using AddressSpaceUnderlyingType = unsigned int;
enum AddressSpace : AddressSpaceUnderlyingType {
  Generic = 0,
  Global = 1,
  Shared = 3,
  Const = 4,
  Local = 5,
  SharedCluster = 7,

  // NVPTX Backend Private:
  Param = 101
};

} // namespace llvm::NVPTX

#endif // LLVM_TARGETPARSER_NVPTX_ADDRESS_SPACES_H
