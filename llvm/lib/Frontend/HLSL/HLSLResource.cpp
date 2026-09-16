//===- HLSLResource.cpp - HLSL Resource helper objects --------------------===//
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

#include "llvm/Frontend/HLSL/HLSLResource.h"
#include "llvm/IR/DerivedTypes.h"

using namespace llvm;
using namespace llvm::hlsl;

dxil::ElementType hlsl::getDXILElementType(Type *Ty, bool IsSigned) {
  // TODO: Handle unorm, snorm, and packed.
  Ty = Ty->getScalarType();

  if (Ty->isIntegerTy()) {
    switch (Ty->getIntegerBitWidth()) {
    case 16:
      return IsSigned ? dxil::ElementType::I16 : dxil::ElementType::U16;
    case 32:
      return IsSigned ? dxil::ElementType::I32 : dxil::ElementType::U32;
    case 64:
      return IsSigned ? dxil::ElementType::I64 : dxil::ElementType::U64;
    case 1:
    default:
      return dxil::ElementType::Invalid;
    }
  } else if (Ty->isFloatTy()) {
    return dxil::ElementType::F32;
  } else if (Ty->isDoubleTy()) {
    return dxil::ElementType::F64;
  } else if (Ty->isHalfTy()) {
    return dxil::ElementType::F16;
  }

  return dxil::ElementType::Invalid;
}
