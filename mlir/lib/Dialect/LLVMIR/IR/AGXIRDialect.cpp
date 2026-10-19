//===- AGXIRDialect.cpp - AGXIR IR Ops and Dialect registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The AGXIR dialect: operations wrapping the llvm.agx.* intrinsics and the
// #agxir.target attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/AGXIRDialect.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::AGXIR;

#include "mlir/Dialect/LLVMIR/AGXIROpsDialect.cpp.inc"
#include "mlir/Dialect/LLVMIR/AGXIROpsEnums.cpp.inc"

namespace {
struct AGXIRInlinerInterface final : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void AGXIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/AGXIROps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/AGXIROpsAttributes.cpp.inc"
      >();

  addInterfaces<AGXIRInlinerInterface>();
  declarePromisedInterface<gpu::TargetAttrInterface, AGXIRTargetAttr>();
}

LogicalResult AGXIRDialect::verifyOperationAttribute(Operation *op,
                                                     NamedAttribute attr) {
  if (kernelAttrName.getName() == attr.getName() &&
      !isa<LLVM::LLVMFuncOp>(op))
    return op->emitError() << "'" << kernelAttrName.getName()
                           << "' attribute attached to unexpected op";
  return success();
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

/// Floating point reductions take f32 values, integer and bitwise ones i32.
static LogicalResult verifyReduceKind(Operation *op, ReduceKind kind,
                                      Type type) {
  bool isFloatKind = kind == ReduceKind::fadd || kind == ReduceKind::fmul ||
                     kind == ReduceKind::fmin || kind == ReduceKind::fmax;
  if (isFloatKind != isa<FloatType>(type))
    return op->emitOpError() << "reduction '" << stringifyReduceKind(kind)
                             << "' does not apply to " << type;
  return success();
}

LogicalResult SIMDReduceOp::verify() {
  return verifyReduceKind(*this, getKind(), getType());
}
LogicalResult SIMDPrefixOp::verify() {
  return verifyReduceKind(*this, getKind(), getType());
}
LogicalResult QuadReduceOp::verify() {
  return verifyReduceKind(*this, getKind(), getType());
}
LogicalResult QuadPrefixOp::verify() {
  return verifyReduceKind(*this, getKind(), getType());
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/AGXIROps.cpp.inc"

//===----------------------------------------------------------------------===//
// Target attribute
//===----------------------------------------------------------------------===//

LogicalResult
AGXIRTargetAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        int optLevel, StringRef triple, StringRef chip,
                        StringRef features, ArrayAttr files) {
  if (optLevel < 0 || optLevel > 3)
    return emitError() << "the optimization level must be a number between 0 "
                          "and 3";
  if (triple.empty())
    return emitError() << "the target triple cannot be empty";
  if (!chip.starts_with("agx-") && chip != "generic")
    return emitError() << "the target chip must be an AGX processor, such as "
                          "agx-g14g";
  if (files && !llvm::all_of(files, [](Attribute attr) {
        return attr && mlir::isa<StringAttr>(attr);
      }))
    return emitError() << "all the elements in the `link` array must be "
                          "strings";
  return success();
}

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/AGXIROpsAttributes.cpp.inc"
