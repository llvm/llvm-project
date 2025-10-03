//===- HLSLResource.h - HLSL Resource helper objects ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects for working with HLSL Resources.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_HLSL_HLSLRESOURCE_H
#define LLVM_FRONTEND_HLSL_HLSLRESOURCE_H

#include "llvm/Support/DXILABI.h"

namespace llvm {
namespace hlsl {

// For now we use DXIL ABI enum values directly. This may change in the future.
using dxil::ResourceClass;

const unsigned CBufferRowSizeInBytes = 16U;

} // namespace hlsl
} // namespace llvm

#endif // LLVM_FRONTEND_HLSL_HLSLRESOURCE_H
