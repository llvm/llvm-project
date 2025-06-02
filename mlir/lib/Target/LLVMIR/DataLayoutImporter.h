//===- DataLayoutImporter.h - LLVM to MLIR data layout conversion -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the translation between the LLVMIR data layout and the
// corresponding MLIR representation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_TARGET_LLVMIR_DATALAYOUTIMPORTER_H_
#define MLIR_LIB_TARGET_LLVMIR_DATALAYOUTIMPORTER_H_

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/MapVector.h"

namespace llvm {
class StringRef;
class DataLayout;
} // namespace llvm

namespace mlir {
class FloatType;
class MLIRContext;
class Operation;

namespace LLVM {
class LLVMFuncOp;

namespace detail {

/// Returns a supported MLIR floating point type of the given bit width or
/// null if the bit width is not supported.
FloatType getFloatType(MLIRContext *context, unsigned width);

/// Helper class that translates an LLVM data layout to an MLIR data layout
/// specification. Only integer, float, pointer, alloca memory space, stack
/// alignment, and endianness entries are translated. The class also returns all
/// entries from the default data layout specification found in the language
/// reference (https://llvm.org/docs/LangRef.html#data-layout) if they are not
/// overwritten by the provided data layout.
class DataLayoutImporter {
public:
  DataLayoutImporter(MLIRContext *context,
                     const llvm::DataLayout &llvmDataLayout)
      : context(context) {
    translateDataLayout(llvmDataLayout);
  }

  /// Returns the MLIR data layout specification translated from the LLVM
  /// data layout.
  DataLayoutSpecInterface getDataLayout() const { return dataLayout; }

  /// Returns the last data layout token that has been processed before
  /// the data layout translation failed.
  StringRef getLastToken() const { return lastToken; }

  /// Returns the data layout tokens that have not been handled during the
  /// data layout translation.
  ArrayRef<StringRef> getUnhandledTokens() const { return unhandledTokens; }

private:
  /// Translates the LLVM `dataLayout` to an MLIR data layout specification.
  void translateDataLayout(const llvm::DataLayout &llvmDataLayout);

  /// Tries to parse the letter only prefix that identifies the specification
  /// and removes the consumed characters from the beginning of the string.
  FailureOr<StringRef> tryToParseAlphaPrefix(StringRef &token) const;

  /// Tries to parse an integer parameter and removes the integer from the
  /// beginning of the string.
  FailureOr<uint64_t> tryToParseInt(StringRef &token) const;

  /// Tries to parse an integer parameter array.
  FailureOr<SmallVector<uint64_t>> tryToParseIntList(StringRef token) const;

  /// Tries to parse the parameters of a type alignment entry.
  FailureOr<DenseIntElementsAttr> tryToParseAlignment(StringRef token) const;

  /// Tries to parse the parameters of a pointer alignment entry.
  FailureOr<DenseIntElementsAttr>
  tryToParsePointerAlignment(StringRef token) const;

  /// Adds a type alignment entry if there is none yet.
  LogicalResult tryToEmplaceAlignmentEntry(Type type, StringRef token);

  /// Adds a pointer alignment entry if there is none yet.
  LogicalResult tryToEmplacePointerAlignmentEntry(LLVMPointerType type,
                                                  StringRef token);

  /// Adds an endianness entry if there is none yet.
  LogicalResult tryToEmplaceEndiannessEntry(StringRef endianness,
                                            StringRef token);

  /// Adds an alloca address space entry if there is none yet.
  LogicalResult tryToEmplaceAddrSpaceEntry(StringRef token,
                                           llvm::StringLiteral spaceKey);

  /// Adds an mangling mode entry if there is none yet.
  LogicalResult tryToEmplaceManglingModeEntry(StringRef token,
                                              llvm::StringLiteral manglingKey);

  /// Adds a stack alignment entry if there is none yet.
  LogicalResult tryToEmplaceStackAlignmentEntry(StringRef token);

  /// Adds a function pointer alignment entry if there is none yet.
  LogicalResult
  tryToEmplaceFunctionPointerAlignmentEntry(StringRef fnPtrAlignEntry,
                                            StringRef token);

  std::string layoutStr = {};
  StringRef lastToken = {};
  SmallVector<StringRef> unhandledTokens;
  llvm::MapVector<StringAttr, DataLayoutEntryInterface> keyEntries;
  llvm::MapVector<TypeAttr, DataLayoutEntryInterface> typeEntries;
  MLIRContext *context;
  DataLayoutSpecInterface dataLayout;
};

} // namespace detail
} // namespace LLVM
} // namespace mlir

#endif // MLIR_LIB_TARGET_LLVMIR_DATALAYOUTIMPORTER_H_
