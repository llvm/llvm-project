//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors in the
// shape transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
class ConversionTarget;
class TypeConverter;
} // namespace mlir

namespace mlir {
/// Creates an instance of the ShapeToShapeLowering pass that legalizes Shape
/// dialect to be convertible to Standard. For example, `shape.num_elements` get
/// transformed to `shape.reduce`, which can be lowered to SCF and Standard.
std::unique_ptr<Pass> createShapeToShapeLowering();

/// Collects a set of patterns to rewrite ops within the Shape dialect.
void populateShapeRewritePatterns(RewritePatternSet &patterns);

// Collects a set of patterns to replace all constraints with passing witnesses.
// This is intended to then allow all ShapeConstraint related ops and data to
// have no effects and allow them to be freely removed such as through
// canonicalization and dead code elimination.
//
// After this pass, no cstr_ operations exist.
void populateRemoveShapeConstraintsPatterns(RewritePatternSet &patterns);
std::unique_ptr<FunctionPass> createRemoveShapeConstraintsPass();

/// Populates patterns for shape dialect structural type conversions and sets up
/// the provided ConversionTarget with the appropriate legality configuration
/// for the ops to get converted properly.
///
/// A "structural" type conversion is one where the underlying ops are
/// completely agnostic to the actual types involved and simply need to update
/// their types consistently. An example of this is shape.assuming -- the
/// shape.assuming op and the corresponding shape.assuming_yield op need to have
/// consistent types, but the exact types don't matter. So all that we need to
/// do for a structural type conversion is to update both of their types
/// consistently to the new types prescribed by the TypeConverter.
void populateShapeStructuralTypeConversionsAndLegality(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target);

// Bufferizes shape dialect ops.
//
// Note that most shape dialect ops must be converted to std before
// bufferization happens, as they are intended to be bufferized at the std
// level.
std::unique_ptr<FunctionPass> createShapeBufferizePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/Shape/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_SHAPE_TRANSFORMS_PASSES_H_
