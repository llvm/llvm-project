//===- DXILConstants.h - Essential DXIL constants -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains essential DXIL constants.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DIRECTX_DXILCONSTANTS_H
#define LLVM_LIB_TARGET_DIRECTX_DXILCONSTANTS_H

namespace llvm {
namespace dxil {

enum class OpCode : unsigned {
#define DXIL_OPCODE(Op, Name) Name = Op,
#include "DXILOperation.inc"
};

enum class OpCodeClass : unsigned {
#define DXIL_OPCLASS(Name) Name,
#include "DXILOperation.inc"
};

enum class OpParamType : unsigned {
#define DXIL_OP_PARAM_TYPE(Name) Name,
#include "DXILOperation.inc"
};

} // namespace dxil
} // namespace llvm

#endif
