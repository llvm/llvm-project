//===- AsyncToLLVM.h - Convert Async to LLVM dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ASYNCTOLLVM_ASYNCTOLLVM_H
#define MLIR_CONVERSION_ASYNCTOLLVM_ASYNCTOLLVM_H

#include <memory>

namespace mlir {

class ConversionTarget;
class ModuleOp;
template <typename T>
class OperationPass;
class MLIRContext;
class TypeConverter;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

/// Create a pass to convert Async operations to the LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertAsyncToLLVMPass();

/// Populates patterns for async structural type conversions.
///
/// A "structural" type conversion is one where the underlying ops are
/// completely agnostic to the actual types involved and simply need to update
/// their types. An example of this is async.execute -- the async.execute op and
/// the corresponding async.yield ops need to update their types accordingly to
/// the TypeConverter, but otherwise don't care what type conversions are
/// happening.
void populateAsyncStructuralTypeConversionsAndLegality(
    TypeConverter &typeConverter, OwningRewritePatternList &patterns,
    ConversionTarget &target);

} // namespace mlir

#endif // MLIR_CONVERSION_ASYNCTOLLVM_ASYNCTOLLVM_H
