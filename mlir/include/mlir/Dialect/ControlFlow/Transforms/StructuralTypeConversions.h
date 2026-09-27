//===- StructuralTypeConversions.h - CF Type Conversions --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CONTROL_FLOW_TRANSFORMS_STRUCTURAL_TYPE_CONVERSIONS_H
#define MLIR_DIALECT_CONTROL_FLOW_TRANSFORMS_STRUCTURAL_TYPE_CONVERSIONS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir {

class ConversionTarget;
class TypeConverter;

namespace cf {

/// Populates patterns for CF structural type conversions and sets up the
/// provided ConversionTarget with the appropriate legality configuration for
/// the ops to get converted properly.
///
/// A "structural" type conversion is one where the underlying ops are
/// completely agnostic to the actual types involved and simply need to update
/// their types. An example of this is cf.br -- the cf.br op needs to update
/// its types accordingly to the TypeConverter, but otherwise does not care
/// what type conversions are happening.
void populateCFStructuralTypeConversionsAndLegality(
    const TypeConverter &typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target, PatternBenefit benefit = 1);

/// Similar to `populateCFStructuralTypeConversionsAndLegality` but does not
/// populate the conversion target.
void populateCFStructuralTypeConversions(const TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         PatternBenefit benefit = 1);

/// Updates the ConversionTarget with dynamic legality of CF operations based
/// on the provided type converter.
void populateCFStructuralTypeConversionTarget(
    const TypeConverter &typeConverter, ConversionTarget &target);

} // namespace cf
} // namespace mlir

#endif // MLIR_DIALECT_CONTROL_FLOW_TRANSFORMS_STRUCTURAL_TYPE_CONVERSIONS_H
