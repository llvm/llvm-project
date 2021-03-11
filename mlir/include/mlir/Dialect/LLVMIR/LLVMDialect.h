//===- LLVMDialect.h - MLIR LLVM IR dialect ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LLVM IR dialect in MLIR, containing LLVM operations and
// LLVM type system.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_
#define MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.h.inc"
#include "mlir/Dialect/LLVMIR/LLVMOpsInterfaces.h.inc"

namespace llvm {
class Type;
class LLVMContext;
namespace sys {
template <bool mt_only>
class SmartMutex;
} // end namespace sys
} // end namespace llvm

namespace mlir {
namespace LLVM {
class LLVMDialect;
class LoopOptionsAttrBuilder;

namespace detail {
struct LLVMTypeStorage;
struct LLVMDialectImpl;
} // namespace detail
} // namespace LLVM
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.h.inc"

///// Ops /////
#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOps.h.inc"

#include "mlir/Dialect/LLVMIR/LLVMOpsDialect.h.inc"

namespace mlir {
namespace LLVM {
/// Create an LLVM global containing the string "value" at the module containing
/// surrounding the insertion point of builder. Obtain the address of that
/// global and use it to compute the address of the first character in the
/// string (operations inserted at the builder insertion point).
Value createGlobalString(Location loc, OpBuilder &builder, StringRef name,
                         StringRef value, Linkage linkage);

/// LLVM requires some operations to be inside of a Module operation. This
/// function confirms that the Operation has the desired properties.
bool satisfiesLLVMModule(Operation *op);

/// Builder class for LoopOptionsAttr. This helper class allows to progressively
/// build a LoopOptionsAttr one option at a time, and pay the price of attribute
/// creation once all the options are in place.
class LoopOptionsAttrBuilder {
public:
  /// Construct a empty builder.
  LoopOptionsAttrBuilder() = default;

  /// Construct a builder with an initial list of options from an existing
  /// LoopOptionsAttr.
  LoopOptionsAttrBuilder(LoopOptionsAttr attr);

  /// Set the `disable_licm` option to the provided value. If no value
  /// is provided the option is deleted.
  LoopOptionsAttrBuilder &setDisableLICM(Optional<bool> value);

  /// Set the `interleave_count` option to the provided value. If no value
  /// is provided the option is deleted.
  LoopOptionsAttrBuilder &setInterleaveCount(Optional<uint64_t> count);

  /// Set the `disable_unroll` option to the provided value. If no value
  /// is provided the option is deleted.
  LoopOptionsAttrBuilder &setDisableUnroll(Optional<bool> value);

  /// Returns true if any option has been set.
  bool empty() { return options.empty(); }

private:
  template <typename T>
  LoopOptionsAttrBuilder &setOption(LoopOptionCase tag, Optional<T> value);

  friend class LoopOptionsAttr;
  SmallVector<LoopOptionsAttr::OptionValuePair> options;
};

} // end namespace LLVM
} // end namespace mlir

#endif // MLIR_DIALECT_LLVMIR_LLVMDIALECT_H_
