//===- MemRefToEmitC.h - Convert MemRef to EmitC --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITC_H
#define MLIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITC_H

constexpr const char *alignedAllocFunctionName = "aligned_alloc";
constexpr const char *mallocFunctionName = "malloc";
constexpr const char *memcpyFunctionName = "memcpy";
constexpr const char *cppStandardLibraryHeader = "cstdlib";
constexpr const char *cStandardLibraryHeader = "stdlib.h";
constexpr const char *cppStringLibraryHeader = "cstring";
constexpr const char *cStringLibraryHeader = "string.h";

namespace mlir {
class DialectRegistry;
class RewritePatternSet;
class TypeConverter;

void populateMemRefToEmitCTypeConversion(TypeConverter &typeConverter);

void populateMemRefToEmitCConversionPatterns(RewritePatternSet &patterns,
                                             const TypeConverter &converter);

void registerConvertMemRefToEmitCInterface(DialectRegistry &registry);
} // namespace mlir

#endif // MLIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITC_H
