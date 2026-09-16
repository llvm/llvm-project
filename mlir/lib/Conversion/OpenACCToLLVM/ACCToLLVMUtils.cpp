//===- ACCToLLVMUtils.cpp - OpenACC to LLVM helpers -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToLLVM/ACCToLLVMUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"

#include <iterator>

using namespace mlir;
using namespace mlir::acc;

Location acc::unfuseLoc(Location loc) {
  while (auto fusedLoc = dyn_cast<FusedLoc>(loc))
    loc = fusedLoc.getLocations().back();
  return loc;
}

std::optional<FileLineColLoc>
acc::getFileLineColLoc(Location loc, bool errorOnInvalidLocation) {
  Location unfusedLoc = unfuseLoc(loc);

  if (auto fileLoc = dyn_cast<FileLineColLoc>(unfusedLoc))
    return fileLoc;

  if (auto callSiteLoc = dyn_cast<CallSiteLoc>(unfusedLoc)) {
    if (auto calleeFileLoc = getFileLineColLoc(callSiteLoc.getCallee(), false))
      return calleeFileLoc;
    if (auto callerFileLoc =
            getFileLineColLoc(callSiteLoc.getCaller(), errorOnInvalidLocation))
      return callerFileLoc;
  }

  if (errorOnInvalidLocation)
    llvm_unreachable(
        "cannot get file:line information: invalid Location information");
  return std::nullopt;
}

StringRef acc::getParentFunctionName(Operation *op) {
  if (!op)
    return "";
  auto funcOp = dyn_cast<FunctionOpInterface>(op);
  if (!funcOp)
    funcOp = op->getParentOfType<FunctionOpInterface>();
  return funcOp ? funcOp.getName() : StringRef();
}

StringRef acc::getParentFunctionName(Value value) {
  if (auto *op = value.getDefiningOp())
    return getParentFunctionName(op);
  return "";
}

StringRef acc::getParentFunctionName(ValueRange values) {
  for (Value value : values) {
    if (StringRef name = getParentFunctionName(value); !name.empty())
      return name;
  }
  return "";
}

std::string acc::getInternalGlobalName(StringRef kind, StringRef detail) {
  SmallString<64> name("acc.");
  name += kind;
  if (detail.empty())
    return name.str().str();

  name += '.';
  for (unsigned char c : detail)
    name += (llvm::isAlnum(c) || c == '_' || c == '$' || c == '.') ? c : '_';
  return name.str().str();
}

/// Creates or reuses a module-internal null-terminated string global and
/// returns the GlobalOp.
static LLVM::GlobalOp getOrCreateGlobalStringOp(Location loc,
                                                OpBuilder &builder,
                                                StringRef name, StringRef value,
                                                Region &globalSymbolRegion,
                                                SymbolTable *symbolTable) {
  SmallString<32> nullTermStr(value);
  nullTermStr.push_back('\0');
  StringAttr valueAttr = builder.getStringAttr(nullTermStr);

  // A name says what the global holds, so a global of that name is the one to
  // reuse as long as it holds this very value. Names come out of detail the
  // source spells and are not one to one with it, so a name can also lead to a
  // global holding something else, which a suffix then keeps apart.
  SmallString<64> uniqueName(name);
  if (symbolTable) {
    unsigned suffix = 0;
    while (Operation *taken = symbolTable->lookup(uniqueName)) {
      auto global = dyn_cast<LLVM::GlobalOp>(taken);
      if (global && global.getValueOrNull() == valueAttr)
        return global;
      uniqueName.clear();
      (name + "." + Twine(suffix++)).toVector(uniqueName);
    }
  }

  // Materialize the global through the incoming builder so that it stays
  // tracked when the caller is a dialect conversion rewriter.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&globalSymbolRegion.front());
  auto arrayTy = LLVM::LLVMArrayType::get(builder.getI8Type(),
                                          nullTermStr.size_in_bytes());
  auto global =
      LLVM::GlobalOp::create(builder, loc, arrayTy, /*isConstant=*/true,
                             LLVM::Linkage::Internal, uniqueName, valueAttr,
                             /*alignment=*/0);
  if (symbolTable)
    symbolTable->insert(global);
  return global;
}

Value acc::getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                   StringRef name, StringRef value,
                                   Region &globalSymbolRegion,
                                   SymbolTable *symbolTable) {
  Type i64Ty = builder.getI64Type();
  Type ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
  LLVM::GlobalOp global = getOrCreateGlobalStringOp(
      loc, builder, name, value, globalSymbolRegion, symbolTable);

  Value globalPtr = LLVM::AddressOfOp::create(builder, loc, global);
  Value cst0 = LLVM::ConstantOp::create(builder, loc, i64Ty,
                                        builder.getI64IntegerAttr(0));
  return LLVM::GEPOp::create(builder, loc, ptrTy, global.getType(), globalPtr,
                             ArrayRef<Value>({cst0, cst0}));
}

Value acc::createIdent(Location loc, StringRef functionName, OpBuilder &builder,
                       Region &globalSymbolRegion,
                       const ACCRuntimeCallConfig &config,
                       SymbolTable *symbolTable) {
  MLIRContext *ctx = builder.getContext();
  Type i32Ty = builder.getI32Type();
  Type i64Ty = builder.getI64Type();
  Type ptrTy = LLVM::LLVMPointerType::get(ctx);
  Type structTy = LLVM::LLVMStructType::getLiteral(
      ctx, {i32Ty, i32Ty, i32Ty, i32Ty, ptrTy});

  std::string source;
  // The position the globals describe is what names them, so a global found
  // under such a name holds this very source string or ident.
  std::string position;
  if (auto fileLineColLoc =
          getFileLineColLoc(loc, /*errorOnInvalidLocation=*/false)) {
    std::string filename = fileLineColLoc->getFilename().str();
    std::string line = std::to_string(fileLineColLoc->getLine());
    std::string column = std::to_string(fileLineColLoc->getColumn());
    std::string functionDisplayName =
        functionName.empty() ? std::string()
                             : config.getFunctionDisplayName(functionName);
    source = ";";
    source += filename + ";";
    source += functionDisplayName + ";";
    source += line + ";";
    source += column + ";";
    source += ";";
    position = line + "." + column + ".";
    position += std::to_string(static_cast<uint64_t>(llvm::hash_value(source)));
  } else {
    source = ";unknown;unknown;0;0;;";
    position = "unknown";
  }

  std::string identGlobalName = getInternalGlobalName("ident", position);
  LLVM::GlobalOp identGlobal =
      symbolTable ? symbolTable->lookup<LLVM::GlobalOp>(identGlobalName)
                  : LLVM::GlobalOp();
  if (!identGlobal) {
    LLVM::GlobalOp sourceGlobal = getOrCreateGlobalStringOp(
        loc, builder, getInternalGlobalName("loc", position), source,
        globalSymbolRegion, symbolTable);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(sourceGlobal);
    identGlobal = LLVM::GlobalOp::create(
        builder, loc, structTy, /*isConstant=*/true, LLVM::Linkage::Internal,
        identGlobalName, /*value=*/Attribute(), /*alignment=*/0);
    if (symbolTable)
      symbolTable->insert(identGlobal);

    Block *block = builder.createBlock(&identGlobal.getInitializerRegion());
    builder.setInsertionPointToStart(block);
    Value ident = LLVM::ZeroOp::create(builder, loc, structTy);
    Value sourceBase = LLVM::AddressOfOp::create(builder, loc, sourceGlobal);
    Value cst0 = LLVM::ConstantOp::create(builder, loc, i64Ty,
                                          builder.getI64IntegerAttr(0));
    Value sourcePtr =
        LLVM::GEPOp::create(builder, loc, ptrTy, sourceGlobal.getType(),
                            sourceBase, ArrayRef<Value>({cst0, cst0}));
    ident = LLVM::InsertValueOp::create(builder, loc, structTy, ident,
                                        sourcePtr, ArrayRef<int64_t>{4});
    LLVM::ReturnOp::create(builder, loc, ident);
  }

  return LLVM::AddressOfOp::create(builder, loc, identGlobal);
}

Value acc::castToI64(Location loc, Value value, OpBuilder &builder) {
  Type i64Ty = builder.getI64Type();
  if (value.getType() == i64Ty)
    return value;
  if (isa<IndexType>(value.getType()))
    return arith::IndexCastOp::create(builder, loc, i64Ty, value);
  unsigned bitwidth = value.getType().getIntOrFloatBitWidth();
  if (bitwidth > 64)
    return arith::TruncIOp::create(builder, loc, i64Ty, value);
  return arith::ExtSIOp::create(builder, loc, i64Ty, value);
}

Value acc::getAsyncQueue(Location loc, Value asyncOperand, bool asyncOnly,
                         OpBuilder &builder,
                         const ACCRuntimeCallConfig &config) {
  Type i64Ty = builder.getI64Type();
  if (asyncOnly)
    return LLVM::ConstantOp::create(builder, loc, i64Ty,
                                    config.getAsyncNoValueRuntimeValue());
  if (asyncOperand)
    return castToI64(loc, asyncOperand, builder);
  return LLVM::ConstantOp::create(builder, loc, i64Ty,
                                  config.getAsyncSyncRuntimeValue());
}

LogicalResult acc::emitWaitCall(Location loc, ValueRange waitOperands,
                                Value asyncQueue, OpBuilder &builder,
                                Region &globalSymbolRegion,
                                SymbolTable &symbolTable,
                                const ACCRuntimeCallConfig &config) {
  Type i32Ty = builder.getI32Type();
  Type i64Ty = builder.getI64Type();
  Type ptrTy = LLVM::LLVMPointerType::get(builder.getContext());

  SmallVector<Value> queues;
  queues.reserve(waitOperands.size());
  for (Value waitOperand : waitOperands)
    queues.push_back(castToI64(loc, waitOperand, builder));

  unsigned size = queues.size();
  Value waitNum = LLVM::ConstantOp::create(builder, loc, i32Ty, size);
  Value waitList;
  if (size == 0) {
    waitList = LLVM::ZeroOp::create(builder, loc, ptrTy);
  } else {
    waitList = LLVM::AllocaOp::create(builder, loc, ptrTy, i64Ty, waitNum);
    for (auto [index, queue] : llvm::enumerate(queues)) {
      Value idx = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                           static_cast<int64_t>(index));
      Value elementPtr = LLVM::GEPOp::create(builder, loc, ptrTy, i64Ty,
                                             waitList, ArrayRef<Value>{idx});
      LLVM::StoreOp::create(builder, loc, queue, elementPtr);
    }
  }

  StringRef functionName = getParentFunctionName(waitOperands);
  if (functionName.empty())
    if (Block *block = builder.getInsertionBlock())
      functionName = getParentFunctionName(block->getParentOp());
  Value ident = createIdent(loc, functionName, builder, globalSymbolRegion,
                            config, &symbolTable);
  Value flags = LLVM::ConstantOp::create(builder, loc, i64Ty, 0);
  Value deviceType = LLVM::ConstantOp::create(
      builder, loc, i64Ty, config.getDeviceTypeRuntimeValue(DeviceType::None));
  Value deviceNum = LLVM::ConstantOp::create(builder, loc, i32Ty, 0);

  return createRuntimeCall(
      loc, builder, globalSymbolRegion, symbolTable,
      RuntimeFunction::ACCRTL_tgt_acc_wait, config,
      {ident, flags, deviceType, deviceNum, waitNum, waitList, asyncQueue});
}

SmallVector<DeviceType, 3>
acc::getDeviceTypesByPrecedence(DeviceType deviceType) {
  SmallVector<DeviceType, 3> deviceTypes;
  if (deviceType != DeviceType::None && deviceType != DeviceType::Star)
    deviceTypes.push_back(deviceType);
  deviceTypes.push_back(DeviceType::Star);
  deviceTypes.push_back(DeviceType::None);
  return deviceTypes;
}

LogicalResult acc::emitGuardedByIfCond(Location loc, Value ifCond,
                                       RewriterBase &rewriter,
                                       function_ref<LogicalResult()> emitFn) {
  if (!ifCond)
    return emitFn();

  Block *parentBlock = rewriter.getInsertionBlock();
  Block *continueBlock =
      rewriter.splitBlock(parentBlock, rewriter.getInsertionPoint());
  Block *thenBlock = rewriter.createBlock(
      parentBlock->getParent(), std::next(Region::iterator(parentBlock)));

  rewriter.setInsertionPointToEnd(parentBlock);
  LLVM::CondBrOp::create(rewriter, loc, ifCond, thenBlock, ValueRange{},
                         continueBlock, ValueRange{});

  rewriter.setInsertionPointToStart(thenBlock);
  LogicalResult result = emitFn();
  rewriter.setInsertionPointToEnd(thenBlock);
  LLVM::BrOp::create(rewriter, loc, ValueRange{}, continueBlock);
  rewriter.setInsertionPointToStart(continueBlock);
  return result;
}
