//===- llvm/CodeGen/GlobalISel/Types.h - Types used by GISel ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes high level types that are used by several passes or
/// APIs involved in the GlobalISel pipeline.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_TYPES_H
#define LLVM_CODEGEN_GLOBALISEL_TYPES_H

#include "llvm/ADT/DenseMap.h"

namespace llvm {

class Value;

/// Map a value to a virtual register.
/// For now, we chose to map aggregate types to on single virtual
/// register. This might be revisited if it turns out to be inefficient.
/// PR26161 tracks that.
/// Note: We need to expose this type to the target hooks for thing like
/// ABI lowering that would be used during IRTranslation.
using ValueToVReg = DenseMap<const Value *, unsigned>;

} // end namespace llvm

#endif // LLVM_CODEGEN_GLOBALISEL_TYPES_H
