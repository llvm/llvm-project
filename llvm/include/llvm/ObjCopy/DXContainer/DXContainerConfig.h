//===- DXContainerConfig.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJCOPY_DXCONTAINER_DXCONTAINERCONFIG_H
#define LLVM_OBJCOPY_DXCONTAINER_DXCONTAINERCONFIG_H

namespace llvm {
namespace objcopy {

// DXContainer specific configuration for copying/stripping a single file. This
// is defined, following convention elsewhere, as the return type of
// `getDXContainerConfig`, which reports an error for an unsupported option.
struct DXContainerConfig {};

} // namespace objcopy
} // namespace llvm

#endif // LLVM_OBJCOPY_DXCONTAINER_DXCONTAINERCONFIG_H
