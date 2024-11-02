//===- DataLayoutImporter.cpp - LLVM to MLIR data layout conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DataLayoutImporter.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/IR/DataLayout.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

/// The default data layout used during the translation.
static constexpr StringRef kDefaultDataLayout =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-"
    "f16:16:16-f64:64:64-f128:128:128";

FloatType mlir::LLVM::detail::getFloatType(MLIRContext *context,
                                           unsigned width) {
  switch (width) {
  case 16:
    return FloatType::getF16(context);
  case 32:
    return FloatType::getF32(context);
  case 64:
    return FloatType::getF64(context);
  case 80:
    return FloatType::getF80(context);
  case 128:
    return FloatType::getF128(context);
  default:
    return {};
  }
}

FailureOr<StringRef>
DataLayoutImporter::tryToParseAlphaPrefix(StringRef &token) const {
  if (token.empty())
    return failure();

  StringRef prefix = token.take_while(isalpha);
  if (prefix.empty())
    return failure();

  token.consume_front(prefix);
  return prefix;
}

FailureOr<uint64_t> DataLayoutImporter::tryToParseInt(StringRef &token) const {
  uint64_t parameter;
  if (token.consumeInteger(/*Radix=*/10, parameter))
    return failure();
  return parameter;
}

FailureOr<SmallVector<uint64_t>>
DataLayoutImporter::tryToParseIntList(StringRef token) const {
  SmallVector<StringRef> tokens;
  token.consume_front(":");
  token.split(tokens, ':');

  // Parse an integer list.
  SmallVector<uint64_t> results(tokens.size());
  for (auto [result, token] : llvm::zip(results, tokens))
    if (token.getAsInteger(/*Radix=*/10, result))
      return failure();
  return results;
}

FailureOr<DenseIntElementsAttr>
DataLayoutImporter::tryToParseAlignment(StringRef token) const {
  FailureOr<SmallVector<uint64_t>> alignment = tryToParseIntList(token);
  if (failed(alignment))
    return failure();
  if (alignment->empty() || alignment->size() > 2)
    return failure();

  // Alignment specifications (such as 32 or 32:64) are of the
  // form <abi>[:<pref>], where abi specifies the minimal alignment and pref the
  // optional preferred alignment. The preferred alignment is set to the minimal
  // alignment if not available.
  uint64_t minimal = (*alignment)[0];
  uint64_t preferred = alignment->size() == 1 ? minimal : (*alignment)[1];
  return DenseIntElementsAttr::get(
      VectorType::get({2}, IntegerType::get(context, 64)),
      {minimal, preferred});
}

FailureOr<DenseIntElementsAttr>
DataLayoutImporter::tryToParsePointerAlignment(StringRef token) const {
  FailureOr<SmallVector<uint64_t>> alignment = tryToParseIntList(token);
  if (failed(alignment))
    return failure();
  if (alignment->size() < 2 || alignment->size() > 4)
    return failure();

  // Pointer alignment specifications (such as 64:32:64:32 or 32:32) are of
  // the form <size>:<abi>[:<pref>][:<idx>], where size is the pointer size, abi
  // specifies the minimal alignment, pref the optional preferred alignment, and
  // idx the optional index computation bit width. The preferred alignment is
  // set to the minimal alignment if not available and the index computation
  // width is set to the pointer size if not available.
  uint64_t size = (*alignment)[0];
  uint64_t minimal = (*alignment)[1];
  uint64_t preferred = alignment->size() < 3 ? minimal : (*alignment)[2];
  uint64_t idx = alignment->size() < 4 ? size : (*alignment)[3];
  return DenseIntElementsAttr::get<uint64_t>(
      VectorType::get({4}, IntegerType::get(context, 64)),
      {size, minimal, preferred, idx});
}

LogicalResult DataLayoutImporter::tryToEmplaceAlignmentEntry(Type type,
                                                             StringRef token) {
  auto key = TypeAttr::get(type);
  if (typeEntries.count(key))
    return success();

  FailureOr<DenseIntElementsAttr> params = tryToParseAlignment(token);
  if (failed(params))
    return failure();

  typeEntries.try_emplace(key, DataLayoutEntryAttr::get(type, *params));
  return success();
}

LogicalResult
DataLayoutImporter::tryToEmplacePointerAlignmentEntry(LLVMPointerType type,
                                                      StringRef token) {
  auto key = TypeAttr::get(type);
  if (typeEntries.count(key))
    return success();

  FailureOr<DenseIntElementsAttr> params = tryToParsePointerAlignment(token);
  if (failed(params))
    return failure();

  typeEntries.try_emplace(key, DataLayoutEntryAttr::get(type, *params));
  return success();
}

LogicalResult
DataLayoutImporter::tryToEmplaceEndiannessEntry(StringRef endianness,
                                                StringRef token) {
  auto key = StringAttr::get(context, DLTIDialect::kDataLayoutEndiannessKey);
  if (keyEntries.count(key))
    return success();

  if (!token.empty())
    return failure();

  keyEntries.try_emplace(
      key, DataLayoutEntryAttr::get(key, StringAttr::get(context, endianness)));
  return success();
}

LogicalResult
DataLayoutImporter::tryToEmplaceAddrSpaceEntry(StringRef token,
                                               llvm::StringLiteral spaceKey) {
  auto key = StringAttr::get(context, spaceKey);
  if (keyEntries.count(key))
    return success();

  FailureOr<uint64_t> space = tryToParseInt(token);
  if (failed(space))
    return failure();

  // Only store the address space if it has a non-default value.
  if (*space == 0)
    return success();
  OpBuilder builder(context);
  keyEntries.try_emplace(
      key,
      DataLayoutEntryAttr::get(
          key, builder.getIntegerAttr(
                   builder.getIntegerType(64, /*isSigned=*/false), *space)));
  return success();
}

LogicalResult
DataLayoutImporter::tryToEmplaceStackAlignmentEntry(StringRef token) {
  auto key =
      StringAttr::get(context, DLTIDialect::kDataLayoutStackAlignmentKey);
  if (keyEntries.count(key))
    return success();

  FailureOr<uint64_t> alignment = tryToParseInt(token);
  if (failed(alignment))
    return failure();

  // Only store the stack alignment if it has a non-default value.
  if (*alignment == 0)
    return success();
  OpBuilder builder(context);
  keyEntries.try_emplace(key, DataLayoutEntryAttr::get(
                                  key, builder.getI64IntegerAttr(*alignment)));
  return success();
}

void DataLayoutImporter::translateDataLayout(
    const llvm::DataLayout &llvmDataLayout) {
  dataLayout = {};

  // Transform the data layout to its string representation and append the
  // default data layout string specified in the language reference
  // (https://llvm.org/docs/LangRef.html#data-layout). The translation then
  // parses the string and ignores the default value if a specific kind occurs
  // in both strings. Additionally, the following default values exist:
  // - non-default address space pointer specifications default to the default
  //   address space pointer specification
  // - the alloca address space defaults to the default address space.
  layoutStr = llvmDataLayout.getStringRepresentation();
  if (!layoutStr.empty())
    layoutStr += "-";
  layoutStr += kDefaultDataLayout;
  StringRef layout(layoutStr);

  // Split the data layout string into tokens separated by a dash.
  SmallVector<StringRef> tokens;
  layout.split(tokens, '-');

  for (StringRef token : tokens) {
    lastToken = token;
    FailureOr<StringRef> prefix = tryToParseAlphaPrefix(token);
    if (failed(prefix))
      return;

    // Parse the endianness.
    if (*prefix == "e") {
      if (failed(tryToEmplaceEndiannessEntry(
              DLTIDialect::kDataLayoutEndiannessLittle, token)))
        return;
      continue;
    }
    if (*prefix == "E") {
      if (failed(tryToEmplaceEndiannessEntry(
              DLTIDialect::kDataLayoutEndiannessBig, token)))
        return;
      continue;
    }
    // Parse the program address space.
    if (*prefix == "P") {
      if (failed(tryToEmplaceAddrSpaceEntry(
              token, DLTIDialect::kDataLayoutProgramMemorySpaceKey)))
        return;
      continue;
    }
    // Parse the global address space.
    if (*prefix == "G") {
      if (failed(tryToEmplaceAddrSpaceEntry(
              token, DLTIDialect::kDataLayoutGlobalMemorySpaceKey)))
        return;
      continue;
    }
    // Parse the alloca address space.
    if (*prefix == "A") {
      if (failed(tryToEmplaceAddrSpaceEntry(
              token, DLTIDialect::kDataLayoutAllocaMemorySpaceKey)))
        return;
      continue;
    }
    // Parse the stack alignment.
    if (*prefix == "S") {
      if (failed(tryToEmplaceStackAlignmentEntry(token)))
        return;
      continue;
    }
    // Parse integer alignment specifications.
    if (*prefix == "i") {
      FailureOr<uint64_t> width = tryToParseInt(token);
      if (failed(width))
        return;

      Type type = IntegerType::get(context, *width);
      if (failed(tryToEmplaceAlignmentEntry(type, token)))
        return;
      continue;
    }
    // Parse float alignment specifications.
    if (*prefix == "f") {
      FailureOr<uint64_t> width = tryToParseInt(token);
      if (failed(width))
        return;

      Type type = getFloatType(context, *width);
      if (failed(tryToEmplaceAlignmentEntry(type, token)))
        return;
      continue;
    }
    // Parse pointer alignment specifications.
    if (*prefix == "p") {
      FailureOr<uint64_t> space =
          token.starts_with(":") ? 0 : tryToParseInt(token);
      if (failed(space))
        return;

      auto type = LLVMPointerType::get(context, *space);
      if (failed(tryToEmplacePointerAlignmentEntry(type, token)))
        return;
      continue;
    }

    // Store all tokens that have not been handled.
    unhandledTokens.push_back(lastToken);
  }

  // Assemble all entries to a data layout specification.
  SmallVector<DataLayoutEntryInterface> entries;
  entries.reserve(typeEntries.size() + keyEntries.size());
  for (const auto &it : typeEntries)
    entries.push_back(it.second);
  for (const auto &it : keyEntries)
    entries.push_back(it.second);
  dataLayout = DataLayoutSpecAttr::get(context, entries);
}

DataLayoutSpecInterface
mlir::translateDataLayout(const llvm::DataLayout &dataLayout,
                          MLIRContext *context) {
  return DataLayoutImporter(context, dataLayout).getDataLayout();
}
