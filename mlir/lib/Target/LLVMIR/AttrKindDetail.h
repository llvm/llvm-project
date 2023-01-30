//===- AttrKindDetail.h - AttrKind conversion details -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ATTRKINDDETAIL_H_
#define ATTRKINDDETAIL_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/IR/Attributes.h"

namespace mlir {
namespace LLVM {
namespace detail {

/// Returns a list of pairs that each hold a mapping from LLVM attribute kinds
/// to their corresponding string name in LLVM IR dialect.
llvm::ArrayRef<std::pair<llvm::Attribute::AttrKind, llvm::StringRef>>
getAttrKindToNameMapping();

} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // ATTRKINDDETAIL_H_
