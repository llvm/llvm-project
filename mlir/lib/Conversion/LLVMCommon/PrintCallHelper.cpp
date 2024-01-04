//===- PrintCallHelper.cpp - Helper to emit runtime print calls -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/PrintCallHelper.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace llvm;

static std::string ensureSymbolNameIsUnique(ModuleOp moduleOp,
                                            StringRef symbolName) {
  static int counter = 0;
  std::string uniqueName = std::string(symbolName);
  while (moduleOp.lookupSymbol(uniqueName)) {
    uniqueName = std::string(symbolName) + "_" + std::to_string(counter++);
  }
  return uniqueName;
}

void mlir::LLVM::createPrintStrCall(
    OpBuilder &builder, Location loc, ModuleOp moduleOp, StringRef symbolName,
    StringRef string, const LLVMTypeConverter &typeConverter, bool addNewline,
    std::optional<StringRef> runtimeFunctionName) {
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(moduleOp.getBody());
  MLIRContext *ctx = builder.getContext();

  // Create a zero-terminated byte representation and allocate global symbol.
  SmallVector<uint8_t> elementVals;
  elementVals.append(string.begin(), string.end());
  if (addNewline)
    elementVals.push_back('\n');
  elementVals.push_back('\0');
  auto dataAttrType = RankedTensorType::get(
      {static_cast<int64_t>(elementVals.size())}, builder.getI8Type());
  auto dataAttr =
      DenseElementsAttr::get(dataAttrType, llvm::ArrayRef(elementVals));
  auto arrayTy =
      LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8), elementVals.size());
  auto globalOp = builder.create<LLVM::GlobalOp>(
      loc, arrayTy, /*constant=*/true, LLVM::Linkage::Private,
      ensureSymbolNameIsUnique(moduleOp, symbolName), dataAttr);

  auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
  // Emit call to `printStr` in runtime library.
  builder.restoreInsertionPoint(ip);
  auto msgAddr =
      builder.create<LLVM::AddressOfOp>(loc, ptrTy, globalOp.getName());
  SmallVector<LLVM::GEPArg> indices(1, 0);
  Value gep =
      builder.create<LLVM::GEPOp>(loc, ptrTy, arrayTy, msgAddr, indices);
  Operation *printer =
      LLVM::lookupOrCreatePrintStringFn(moduleOp, runtimeFunctionName);
  builder.create<LLVM::CallOp>(loc, TypeRange(), SymbolRefAttr::get(printer),
                               gep);
}
