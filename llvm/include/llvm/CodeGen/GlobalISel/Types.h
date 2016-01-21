//===-- llvm/CodeGen/GlobalISel/Types.h - Types used by GISel ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Value.h"

namespace llvm {

/// Map a value to virtual registers.
/// We must support several virtual registers for a value.
/// Indeed each virtual register is mapped to one EVT, but a value
/// may span over several EVT when it is a type representing a structure.
/// In that case the value will be break into EVTs.
/// Note: We need to expose this type to the target hooks for thing like
/// ABI lowering that would be used during IRTranslation.
typedef DenseMap<const Value *, SmallVector<unsigned, 1>> ValueToVRegs;

} // End namespace llvm.
#endif
