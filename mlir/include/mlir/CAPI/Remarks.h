//===- Remarks.h - C API Utils for MLIR Remarks -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the conversion between the C API remark type and its C++
// counterpart.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_REMARKS_H
#define MLIR_CAPI_REMARKS_H

#include "mlir-c/Remarks.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Remarks.h"

DEFINE_C_API_PTR_METHODS(MlirRemark, const mlir::remark::detail::Remark)

#endif // MLIR_CAPI_REMARKS_H
