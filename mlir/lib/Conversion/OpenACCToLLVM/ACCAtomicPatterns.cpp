//===- ACCAtomicPatterns.cpp - ACC atomic to LLVM patterns ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers OpenACC atomic operations (read, write, update, capture) to LLVM
// dialect atomicrmw / cmpxchg sequences.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/OpenACCToLLVM/ACCToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsType.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <optional>
#include <set>
#include <utility>

#define DEBUG_TYPE "acc-atomic-patterns"

using namespace mlir;
using namespace mlir::acc;

namespace {

constexpr uint64_t kBitsInByte = 8;

template <typename AtomicOpTy>
class ACCAtomicOpConversion : public ConvertOpToLLVMPattern<AtomicOpTy> {
  using typename ConvertOpToLLVMPattern<AtomicOpTy>::OpAdaptor;

public:
  ACCAtomicOpConversion(const LLVMTypeConverter &typeConverter,
                        OpenACCSupport &accSupport,
                        const ACCAtomicLoadAddressCallback &getLoadAddress)
      : ConvertOpToLLVMPattern<AtomicOpTy>(typeConverter),
        accSupport(accSupport), getLoadAddress(getLoadAddress) {}

  LogicalResult
  matchAndRewrite(AtomicOpTy op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

private:
  OpenACCSupport &accSupport;
  ACCAtomicLoadAddressCallback getLoadAddress;

  size_t getComplexStructElementSizeInBits(Type ty) const {
    auto structType = dyn_cast<LLVM::LLVMStructType>(ty);
    if (!structType || structType.getBody().size() != 2 ||
        structType.getBody()[0] != structType.getBody()[1] ||
        !structType.getBody()[0].isIntOrFloat())
      return 0;
    size_t elementSizeInBits = structType.getBody()[0].getIntOrFloatBitWidth();
    if (elementSizeInBits == 0 || elementSizeInBits > 64)
      llvm_unreachable("unexpected complex type");
    return elementSizeInBits;
  }

  /// Serialize a complex value into an integer.
  Value serializeExpr(Value expr, ConversionPatternRewriter &rewriter) const {
    size_t elementSizeInBits =
        getComplexStructElementSizeInBits(expr.getType());
    if (!elementSizeInBits)
      return expr;

    Location loc = expr.getLoc();
    Value firstValue = LLVM::ExtractValueOp::create(rewriter, loc, expr, 0);
    Value secondValue = LLVM::ExtractValueOp::create(rewriter, loc, expr, 1);
    MLIRContext *context = rewriter.getContext();
    Type intCastTy = IntegerType::get(context, elementSizeInBits);
    Type intTy = IntegerType::get(context, elementSizeInBits * 2);
    firstValue = LLVM::BitcastOp::create(rewriter, loc, intCastTy, firstValue);
    secondValue =
        LLVM::BitcastOp::create(rewriter, loc, intCastTy, secondValue);
    firstValue = LLVM::ZExtOp::create(rewriter, loc, intTy, firstValue);
    secondValue = LLVM::ZExtOp::create(rewriter, loc, intTy, secondValue);
    Value shlVal =
        LLVM::ConstantOp::create(rewriter, loc, intTy, elementSizeInBits);
    Value result = LLVM::ShlOp::create(rewriter, loc, secondValue, shlVal);
    return LLVM::OrOp::create(rewriter, loc, result, firstValue);
  }

  /// Deserialize an integer into a complex value.
  Value deserializeExpr(Value expr, Type origTy,
                        ConversionPatternRewriter &rewriter) const {
    size_t elementSizeInBits = getComplexStructElementSizeInBits(origTy);
    if (!elementSizeInBits)
      return expr;

    auto intTy = dyn_cast<IntegerType>(expr.getType());
    if (!intTy || intTy.getWidth() != elementSizeInBits * 2)
      return expr;

    Location loc = expr.getLoc();
    MLIRContext *context = rewriter.getContext();
    Type elemTy = IntegerType::get(context, elementSizeInBits);
    Value low = LLVM::TruncOp::create(rewriter, loc, elemTy, expr);
    Value shiftAmount =
        LLVM::ConstantOp::create(rewriter, loc, intTy, elementSizeInBits);
    Value highFull = LLVM::LShrOp::create(rewriter, loc, expr, shiftAmount);
    Value high = LLVM::TruncOp::create(rewriter, loc, elemTy, highFull);
    Type origElemTy = cast<LLVM::LLVMStructType>(origTy).getBody()[0];
    if (origElemTy != elemTy) {
      low = LLVM::BitcastOp::create(rewriter, loc, origElemTy, low);
      high = LLVM::BitcastOp::create(rewriter, loc, origElemTy, high);
    }
    Value undefStruct = LLVM::UndefOp::create(rewriter, loc, origTy);
    Value structWithLow = LLVM::InsertValueOp::create(
        rewriter, loc, origTy, undefStruct, low, ArrayRef<int64_t>{0});
    return LLVM::InsertValueOp::create(rewriter, loc, origTy, structWithLow,
                                       high, ArrayRef<int64_t>{1});
  }

  uint64_t getAtomicSizeInBytes(Type originalTy, Type convertedTy,
                                ModuleOp module) const {
    std::optional<TypeSizeAndAlignment> sizeAndAlignment =
        acc::getTypeSizeAndAlignment(originalTy, module, &accSupport);
    if (!sizeAndAlignment)
      sizeAndAlignment =
          acc::getTypeSizeAndAlignment(convertedTy, module, &accSupport);
    assert(sizeAndAlignment && "atomic type size is not computable");
    return sizeAndAlignment->first.getFixedValue();
  }

  Type getAtomicType(Type originalTy, Type convertedTy, ModuleOp module) const {
    if (convertedTy.isIntOrFloat())
      return convertedTy;
    return IntegerType::get(
        convertedTy.getContext(),
        getAtomicSizeInBytes(originalTy, convertedTy, module) * kBitsInByte);
  }

  Type getReferencedElementType(Value ref, ModuleOp module = nullptr) const {
    auto ptr = dyn_cast<PointerLikeType>(ref.getType());
    if (!ptr)
      llvm_unreachable("unexpected type");
    Type elementTy = ptr.getElementType();
    Type convertedTy = this->getTypeConverter()->convertType(elementTy);
    if (module)
      return getAtomicType(elementTy, convertedTy, module);
    return convertedTy;
  }

  /// Memrefs are converted to descriptors rather than bare pointers, so the
  /// element pointer must be recomputed from the descriptor.
  Value getAtomicPointer(Value originalRef, Value convertedPtr, Location loc,
                         ConversionPatternRewriter &rewriter) const {
    auto memrefTy = dyn_cast<MemRefType>(originalRef.getType());
    if (!memrefTy)
      return convertedPtr;

    // Extract aligned base pointer (index 1) and offset (index 2).
    Value alignedPtr =
        LLVM::ExtractValueOp::create(rewriter, loc, convertedPtr, 1);
    Value offset = LLVM::ExtractValueOp::create(rewriter, loc, convertedPtr, 2);
    Type elemPtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    return LLVM::GEPOp::create(
        rewriter, loc, elemPtrType,
        this->getTypeConverter()->convertType(memrefTy.getElementType()),
        alignedPtr, offset);
  }

  Block *constructCmpxchgLoop(Value ptr, Type type, Value expr,
                              ConversionPatternRewriter &rewriter) const;

  Value genUpdateCmpxchgLoop(AtomicUpdateOp update,
                             ConversionPatternRewriter &rewriter) const;
};

template <>
LogicalResult ACCAtomicOpConversion<AtomicReadOp>::matchAndRewrite(
    AtomicReadOp read, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = read.getLoc();
  Value xRef = read.getX();
  Value xPtr = getAtomicPointer(xRef, adaptor.getX(), loc, rewriter);
  ModuleOp mod = read->getParentOfType<ModuleOp>();
  Type xType = getReferencedElementType(xRef, mod);

  auto ordering = LLVM::AtomicOrdering::monotonic;
  Value storeVal;
  if (xType.isSignlessInteger()) {
    Value zero = LLVM::ConstantOp::create(rewriter, loc, xType, 0);
    storeVal = LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::_or,
                                         xPtr, zero, ordering);
  } else {
    unsigned bitWidth = xType.getIntOrFloatBitWidth();
    Type intType = IntegerType::get(rewriter.getContext(), bitWidth);
    Value zero = LLVM::ConstantOp::create(rewriter, loc, intType, 0);
    Value intVal = LLVM::AtomicRMWOp::create(
        rewriter, loc, LLVM::AtomicBinOp::_or, xPtr, zero, ordering);
    storeVal = LLVM::BitcastOp::create(rewriter, loc, xType, intVal);
  }

  Value vRef = read.getV();
  Value vPtr = getAtomicPointer(vRef, adaptor.getV(), loc, rewriter);
  Type vType = getReferencedElementType(vRef, mod);

  if (xType != vType) {
    // Convert `x` if the types do not match.
    auto vPtrType = cast<PointerLikeType>(vRef.getType());
    storeVal = vPtrType.genCast(rewriter, loc, storeVal, vType);
    if (!storeVal) {
      return rewriter.notifyMatchFailure(
          read, "failed to convert the loaded value to the destination type");
    }
  }
  auto storeOp = LLVM::StoreOp::create(rewriter, loc, storeVal, vPtr);
  rewriter.replaceOp(read, storeOp);
  return success();
}

template <>
LogicalResult ACCAtomicOpConversion<AtomicWriteOp>::matchAndRewrite(
    AtomicWriteOp write, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Location loc = write.getLoc();
  Value expr = serializeExpr(adaptor.getExpr(), rewriter);
  Value xRef = write.getX();
  Value xPtr = getAtomicPointer(xRef, adaptor.getX(), loc, rewriter);
  ModuleOp mod = write->getParentOfType<ModuleOp>();
  Type xType = getReferencedElementType(xRef, mod);

  auto ordering = LLVM::AtomicOrdering::monotonic;
  if (!xType.isSignlessInteger()) {
    unsigned bitWidth = xType.getIntOrFloatBitWidth();
    Type intType = IntegerType::get(rewriter.getContext(), bitWidth);
    expr = LLVM::BitcastOp::create(rewriter, loc, intType, expr);
  }
  LLVM::AtomicRMWOp::create(rewriter, loc, LLVM::AtomicBinOp::xchg, xPtr, expr,
                            ordering);
  rewriter.eraseOp(write);
  return success();
}

// Generate a cmpxchg loop based on the GenericAtomicRMWOpLowering algorithm.
template <typename AtomicOpTy>
Block *ACCAtomicOpConversion<AtomicOpTy>::constructCmpxchgLoop(
    Value ptr, Type type, Value expr,
    ConversionPatternRewriter &rewriter) const {
  // Split the block into initial, loop, and ending parts.
  Location loc = rewriter.getInsertionPoint()->getLoc();
  Block *initBlock = rewriter.getInsertionBlock();
  Block *loopBlock =
      rewriter.splitBlock(initBlock, rewriter.getInsertionPoint());
  loopBlock->addArgument(type, loc);
  Block *endBlock =
      rewriter.splitBlock(loopBlock, rewriter.getInsertionPoint());

  // Compute the loaded value and branch to the loop block.
  rewriter.setInsertionPointToEnd(initBlock);
  Value init = LLVM::LoadOp::create(rewriter, loc, type, ptr);
  LLVM::BrOp::create(rewriter, loc, init, loopBlock);

  // Prepare the body of the loop block.
  rewriter.setInsertionPointToStart(loopBlock);

  Value loopArgument = loopBlock->getArgument(0);
  Value result = serializeExpr(rewriter.getRemappedValue(expr), rewriter);
  ModuleOp mod = initBlock->getParent()->getParentOfType<ModuleOp>();
  Type convertedExprType =
      this->getTypeConverter()->convertType(expr.getType());
  Type exprType = getAtomicType(expr.getType(), convertedExprType, mod);

  // Cast to an integer type.
  if (!exprType.isSignlessInteger()) {
    Type tmpType = IntegerType::get(
        rewriter.getContext(),
        getAtomicSizeInBytes(expr.getType(), convertedExprType, mod) *
            kBitsInByte);
    result = LLVM::BitcastOp::create(rewriter, loc, tmpType, result);
    loopArgument =
        LLVM::BitcastOp::create(rewriter, loc, tmpType, loopArgument);
  }

  // Prepare the epilog of the loop block.
  // Append the cmpxchg op to the end of the loop block.
  auto successOrdering = LLVM::AtomicOrdering::acq_rel;
  auto failureOrdering = LLVM::AtomicOrdering::monotonic;
  auto cmpxchg =
      LLVM::AtomicCmpXchgOp::create(rewriter, loc, ptr, loopArgument, result,
                                    successOrdering, failureOrdering);
  // Extract the %new_loaded and %ok values from the pair.
  Value newLoaded = LLVM::ExtractValueOp::create(rewriter, loc, cmpxchg, 0);
  Value ok = LLVM::ExtractValueOp::create(rewriter, loc, cmpxchg, 1);

  // Cast back to the original type.
  if (!exprType.isSignlessInteger())
    newLoaded = LLVM::BitcastOp::create(rewriter, loc, exprType, newLoaded);

  // Conditionally branch to the end or back to the loop depending on %ok.
  LLVM::CondBrOp::create(rewriter, loc, ok, endBlock, ArrayRef<Value>(),
                         loopBlock, newLoaded);

  return loopBlock;
}

static Value skipUnrealizedConversionOp(Value v) {
  if (auto convOp =
          dyn_cast_or_null<UnrealizedConversionCastOp>(v.getDefiningOp()))
    return skipUnrealizedConversionOp(convOp.getOperand(0));
  return v;
}

/// Obtain the foremost value of addr to indicate the origin of the storage.
static Value getBaseStorage(Value addr, ConversionPatternRewriter &rewriter) {
  Operation *op = skipUnrealizedConversionOp(addr).getDefiningOp();
  if (auto gepOp = dyn_cast_or_null<LLVM::GEPOp>(op)) {
    addr = gepOp.getBase();
    op = addr.getDefiningOp();
  }
  if (auto extractOp = dyn_cast_or_null<LLVM::ExtractValueOp>(op)) {
    // addr is in a struct. Get the inserted value corresponding to the
    // extraction.
    ArrayRef extractingPosition = extractOp.getPosition();
    Value container = skipUnrealizedConversionOp(extractOp.getContainer());
    Value remappedContainer = rewriter.getRemappedValue(container);
    op = remappedContainer ? remappedContainer.getDefiningOp() : nullptr;
    bool inserted = false;
    while (auto insertValueOp = dyn_cast_or_null<LLVM::InsertValueOp>(op)) {
      ArrayRef insertingPosition = insertValueOp.getPosition();
      if (insertingPosition == extractingPosition) {
        addr = insertValueOp.getValue();
        inserted = true;
        break;
      }
      op = insertValueOp.getContainer().getDefiningOp();
    }
    // The aggregate holding addr is not built by insertions, so the value it
    // is converted from identifies the storage.
    if (!inserted)
      addr = container;
  }
  // op might be alloca already, or addr might be an argument.
  return skipUnrealizedConversionOp(addr);
}

/// Include flow dependency (v -> expr) in the generated loop of
/// `{ atomic.read, atomic.write/update }`.
static void moveDependency(Value vRef, Value vPtr, Value expr,
                           Value loopArgument, Operation &loopHead,
                           ConversionPatternRewriter &rewriter,
                           const ACCAtomicLoadAddressCallback &getLoadAddress) {
  Value vStorage = getBaseStorage(vPtr, rewriter);
  // A dependency may be reached before it is itself converted, in which case
  // its address is still expressed in terms of `vRef`.
  Value vStorageRef = skipUnrealizedConversionOp(vRef);
  LLVM_DEBUG({
    llvm::dbgs() << "[acc-atomic] moveDependency\n";
    llvm::dbgs() << "  vPtr        = " << vPtr << "\n";
    llvm::dbgs() << "  vStorage    = " << vStorage << "\n";
    llvm::dbgs() << "  vStorageRef = " << vStorageRef << "\n";
    llvm::dbgs() << "  expr        = " << expr << "\n";
  });
  llvm::DenseMap<Value, Value> remappedToOriginal;
  std::set<std::pair<Operation *, std::set<Operation *>>> worklist;
  std::set<Operation *> included;
  Value mappedExpr = rewriter.getRemappedValue(expr);
  remappedToOriginal[mappedExpr] = expr;
  if (Operation *exprDef = mappedExpr.getDefiningOp())
    worklist.insert(std::pair{exprDef, std::set<Operation *>{}});

  while (!worklist.empty()) {
    auto [dep, post] = worklist.extract(worklist.begin()).value();
    if (!dep)
      continue;
    LLVM_DEBUG(llvm::dbgs() << "  visit dep   = " << *dep << "\n");
    if (vPtr.getDefiningOp() &&
        !vPtr.getDefiningOp()->getParentOp()->isAncestor(dep)) {
      // Outside the parental region. No load found in this flow.
      LLVM_DEBUG(llvm::dbgs() << "    -> skipped: not in parental region\n");
      continue;
    }

    Value addr;
    if (auto load = dyn_cast<LLVM::LoadOp>(dep)) {
      addr = load.getAddr();
    } else if (auto load = dyn_cast<memref::LoadOp>(dep)) {
      addr = load.getMemref();
    } else if (getLoadAddress) {
      addr = getLoadAddress(dep);
    }
    if (addr) {
      Value baseStorage = getBaseStorage(addr, rewriter);
      LLVM_DEBUG({
        llvm::dbgs() << "    addr        = " << addr << "\n";
        llvm::dbgs() << "    baseStorage = " << baseStorage << "\n";
        llvm::dbgs() << "    remapped    = "
                     << rewriter.getRemappedValue(baseStorage) << "\n";
        llvm::dbgs() << "    matchRef=" << (baseStorage == vStorageRef)
                     << " matchConv="
                     << (rewriter.getRemappedValue(baseStorage) == vStorage)
                     << "\n";
      });
      if (baseStorage == vStorageRef ||
          rewriter.getRemappedValue(baseStorage) == vStorage) {
        // Found the load of `v`. Include this flow in dependency.
        Value load = dep->getResult(0);
        auto replaceUses = [&](Operation *op) {
          rewriter.modifyOpInPlace(op, [&] {
            op->replaceUsesOfWith(remappedToOriginal[load], loopArgument);
          });
        };
        for (Operation *p : post)
          replaceUses(p);
        replaceUses(&loopHead);
        included.insert(post.begin(), post.end());
      }
      continue;
    }
    post.insert(dep);
    for (Value operand : dep->getOperands()) {
      Value mappedOperand = rewriter.getRemappedValue(operand);
      if (auto *d = mappedOperand.getDefiningOp()) {
        if (dep == d) {
          if (auto *op = operand.getDefiningOp()) {
            remappedToOriginal[op->getResult(0)] = operand;
            worklist.insert(std::pair{op, post});
          }
        } else {
          remappedToOriginal[mappedOperand] = operand;
          worklist.insert(std::pair{d, post});
        }
      }
    }
  }

  // Include dependency.
  SmallVector<Operation *> includedInOrder;
  vPtr.getParentRegion()->walk([&](Operation *op) {
    if (included.find(op) != included.end())
      includedInOrder.push_back(op);
  });
  for (Operation *d : includedInOrder)
    d->moveBefore(&loopHead);
}

/// Generate a cmpxchg loop for update and return a stored value.
template <typename AtomicOpTy>
Value ACCAtomicOpConversion<AtomicOpTy>::genUpdateCmpxchgLoop(
    AtomicUpdateOp update, ConversionPatternRewriter &rewriter) const {
  Location loc = update.getLoc();
  Value xRef = update.getX();
  Value xPtr =
      getAtomicPointer(xRef, rewriter.getRemappedValue(xRef), loc, rewriter);
  ModuleOp mod = update->getParentOfType<ModuleOp>();
  Type xType = getReferencedElementType(xRef, mod);
  Type xTypeOrig = getReferencedElementType(xRef);

  Block &updateBlock = update.getRegion().front();
  Value updateArgument = updateBlock.getArgument(0);
  Operation *terminator = updateBlock.getTerminator();
  Value expr = terminator->getOperand(0);

  Block *loopBlock = constructCmpxchgLoop(xPtr, xType, expr, rewriter);
  Value loopArgument = loopBlock->getArgument(0);
  Operation &loopHead = loopBlock->front();

  rewriter.setInsertionPointToStart(loopBlock);
  loopArgument = deserializeExpr(loopArgument, xTypeOrig, rewriter);

  // Move in and out flow dependency (x -> expr). Some computation might be
  // outside atomic regions.
  moveDependency(xRef, xPtr, expr, loopArgument, loopHead, rewriter,
                 getLoadAddress);
  // Move out the residue.
  rewriter.replaceAllUsesWith(cast<BlockArgument>(updateArgument),
                              {loopArgument});

  updateBlock.walk([&](Operation *op) {
    if (!op->hasTrait<OpTrait::IsTerminator>())
      rewriter.moveOpBefore(op, &loopHead);
  });

  if (auto cmpxchg = dyn_cast<LLVM::AtomicCmpXchgOp>(loopHead))
    return cmpxchg.getVal();
  if (auto bitcast = dyn_cast<LLVM::BitcastOp>(loopHead))
    // Handling a non-integer type.
    return bitcast.getArg();
  if (auto extract = dyn_cast<LLVM::ExtractValueOp>(loopHead))
    // Handling a complex type.
    return extract.getContainer();
  llvm_unreachable("invalid cmpxchg loop");
}

/// Generate llvm.atomicrmw or an llvm.cmpxchg loop.
template <>
LogicalResult ACCAtomicOpConversion<AtomicUpdateOp>::matchAndRewrite(
    AtomicUpdateOp update, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  Block &updateBlock = update.getRegion().front();
  Value updateArgument = updateBlock.getArgument(0);

  // Collect operations that depend on the update argument.
  std::set<Operation *> dependents;
  SmallVector<Value> worklist;
  worklist.push_back(updateArgument);
  while (!worklist.empty()) {
    Value value = worklist.back();
    worklist.pop_back();
    for (OpOperand &use : value.getUses()) {
      Operation *useOp = use.getOwner();
      dependents.insert(useOp);
      if (useOp->getNumResults() == 1)
        worklist.push_back(useOp->getResult(0));
    }
  }

  // Move independent operations out of the update block.
  SmallVector<Operation *> independent;
  for (Operation &op : updateBlock.getOperations()) {
    if (dependents.find(&op) == dependents.end()) {
      if (op.hasTrait<OpTrait::IsTerminator>())
        llvm_unreachable("invalid update operation");
      independent.push_back(&op);
    }
  }
  for (Operation *op : independent)
    rewriter.moveOpBefore(op, update);

  // Map arith op to atomicrmw kind.
  //
  // In the current version of LLVM, these are the equivalences for float
  // min/max across the different instructions intrinsics and operations -
  // confusingly they have slightly different and not very descriptive names.
  //
  // | MLIR       | atomicrmw inst | llvm intrinsic    |
  // |------------+----------------+-------------------|
  // | -          | fmax           | llvm.maxnum.*     |
  // | MaximumFOp | fmaximum       | llvm.maximum.*    |
  // | MaxNumFOp  | fmaximumnum    | llvm.maximumnum.* |
  //
  // Sources:
  // https://llvm.org/docs/LangRef.html#id236
  // https://llvm.org/docs/LangRef.html#floating-point-min-max-intrinsics-comparison
  // https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmaximumf-arithmaximumfop
  // https://mlir.llvm.org/docs/Dialects/ArithOps/#arithmaxnumf-arithmaxnumfop
  auto getAtomicBinOp =
      [](Operation *op, bool updateIsLhs) -> std::optional<LLVM::AtomicBinOp> {
    return TypeSwitch<Operation *, std::optional<LLVM::AtomicBinOp>>(op)
        .Case<arith::AddFOp>([](auto) { return LLVM::AtomicBinOp::fadd; })
        .Case<arith::AddIOp>([](auto) { return LLVM::AtomicBinOp::add; })
        .Case<arith::SubFOp>(
            [updateIsLhs](auto) -> std::optional<LLVM::AtomicBinOp> {
              // atomicrmw fsub is always `*ptr = *ptr - val`.
              if (!updateIsLhs)
                return std::nullopt;
              return LLVM::AtomicBinOp::fsub;
            })
        .Case<arith::SubIOp>(
            [updateIsLhs](auto) -> std::optional<LLVM::AtomicBinOp> {
              // atomicrmw sub is always `*ptr = *ptr - val`.
              if (!updateIsLhs)
                return std::nullopt;
              return LLVM::AtomicBinOp::sub;
            })
        .Case<arith::AndIOp>([](auto) { return LLVM::AtomicBinOp::_and; })
        .Case<arith::OrIOp>([](auto) { return LLVM::AtomicBinOp::_or; })
        .Case<arith::XOrIOp>([](auto) { return LLVM::AtomicBinOp::_xor; })
        .Case<arith::MaxSIOp>([](auto) { return LLVM::AtomicBinOp::max; })
        .Case<arith::MinSIOp>([](auto) { return LLVM::AtomicBinOp::min; })
        .Case<arith::MaxUIOp>([](auto) { return LLVM::AtomicBinOp::umax; })
        .Case<arith::MinUIOp>([](auto) { return LLVM::AtomicBinOp::umin; })
        .Case<arith::MaximumFOp>(
            [](auto) { return LLVM::AtomicBinOp::fmaximum; })
        .Case<arith::MinimumFOp>(
            [](auto) { return LLVM::AtomicBinOp::fminimum; })
        .Case<arith::MaxNumFOp>(
            [](auto) { return LLVM::AtomicBinOp::fmaximumnum; })
        .Case<arith::MinNumFOp>(
            [](auto) { return LLVM::AtomicBinOp::fminimumnum; })
        .Default([](Operation *) { return std::nullopt; });
  };

  // Select the kind and the val of atomicrmw.
  std::optional<Value> val = std::nullopt;
  std::optional<LLVM::AtomicBinOp> kind = std::nullopt;

  auto &ops = updateBlock.getOperations();
  Operation &firstOp = ops.front();
  Operation &yield = ops.back();

  if (dependents.size() == 2 && firstOp.getResult(0) == yield.getOperand(0)) {
    bool updateIsLhs = firstOp.getOperand(0) == updateArgument;
    kind = getAtomicBinOp(&firstOp, updateIsLhs);
    if (kind)
      val = firstOp.getOperand(updateIsLhs ? 1 : 0);
  }

  // Per-component atomicrmw info for complex type decomposition.
  // Decomposed complex ops (complex.re/im + arith binop + complex.create)
  // produce per-component binary ops that each need a separate atomicrmw.
  struct ComponentAtomic {
    LLVM::AtomicBinOp kind;
    Value val;
    int32_t fieldIdx;
  };
  SmallVector<ComponentAtomic, 2> componentAtomics;

  if (!val || !kind) {
    Type convertedArgTy =
        this->getTypeConverter()->convertType(updateArgument.getType());
    if (auto structTy = dyn_cast<LLVM::LLVMStructType>(convertedArgTy)) {
      if (structTy.getBody().size() == 2 &&
          structTy.getBody()[0] == structTy.getBody()[1] &&
          structTy.getBody()[0].isIntOrFloat() &&
          structTy.getBody()[0].getIntOrFloatBitWidth() > 32) {
        for (Operation &op : updateBlock.getOperations()) {
          if (op.hasTrait<OpTrait::IsTerminator>() || op.getNumOperands() < 2)
            continue;
          int32_t fieldIdx = -1;
          Value externalVal = nullptr;
          bool updateIsLhs = false;
          for (unsigned i = 0; i < 2; ++i) {
            Value operand = op.getOperand(i);
            if (auto reOp = operand.getDefiningOp<complex::ReOp>()) {
              if (reOp.getOperand() == updateArgument) {
                fieldIdx = 0;
                externalVal = op.getOperand(1 - i);
                updateIsLhs = i == 0;
              }
            } else if (auto imOp = operand.getDefiningOp<complex::ImOp>()) {
              if (imOp.getOperand() == updateArgument) {
                fieldIdx = 1;
                externalVal = op.getOperand(1 - i);
                updateIsLhs = i == 0;
              }
            }
          }
          if (fieldIdx < 0)
            continue;
          auto componentKind = getAtomicBinOp(&op, updateIsLhs);
          if (!componentKind)
            continue;
          componentAtomics.push_back({*componentKind, externalVal, fieldIdx});
        }
      }
    }
  }

  Location loc = update.getLoc();
  Value xPtr = getAtomicPointer(update.getX(), adaptor.getX(), loc, rewriter);

  // Require distinct real/imag lanes; duplicate fieldIdx values must fall back
  // to cmpxchg rather than emitting two atomicrmw ops on the same component.
  bool hasDistinctComplexLanes = false;
  if (componentAtomics.size() == 2) {
    unsigned lanes = 0;
    for (const ComponentAtomic &ca : componentAtomics) {
      if (ca.fieldIdx == 0 || ca.fieldIdx == 1)
        lanes |= 1u << ca.fieldIdx;
    }
    hasDistinctComplexLanes = lanes == 0b11;
  }

  if (val && kind) {
    auto ordering = LLVM::AtomicOrdering::monotonic;
    LLVM::AtomicRMWOp::create(rewriter, loc, *kind, xPtr,
                              rewriter.getRemappedValue(*val), ordering);
  } else if (hasDistinctComplexLanes) {
    auto structTy = cast<LLVM::LLVMStructType>(
        this->getTypeConverter()->convertType(updateArgument.getType()));
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto ordering = LLVM::AtomicOrdering::monotonic;
    for (ComponentAtomic &ca : componentAtomics) {
      Value elemPtr =
          LLVM::GEPOp::create(rewriter, loc, ptrType, structTy, xPtr,
                              ArrayRef<LLVM::GEPArg>{0, ca.fieldIdx});
      LLVM::AtomicRMWOp::create(rewriter, loc, ca.kind, elemPtr,
                                rewriter.getRemappedValue(ca.val), ordering);
    }
  } else {
    // Fallback to the llvm.cmpxchg loop generation.
    genUpdateCmpxchgLoop(update, rewriter);
  }
  rewriter.eraseOp(update);
  return success();
}

/// Generate an llvm.cmpxchg loop.
template <>
LogicalResult ACCAtomicOpConversion<AtomicCaptureOp>::matchAndRewrite(
    AtomicCaptureOp capture, OpAdaptor /*adaptor*/,
    ConversionPatternRewriter &rewriter) const {
  Operation *firstOp = capture.getFirstOp();
  Operation *secondOp = capture.getSecondOp();
  Value vPtr = nullptr;
  Value storeVal = nullptr;
  if (auto firstReadStmt = dyn_cast<AtomicReadOp>(firstOp)) {
    Location loc = capture.getLoc();
    Value xRef = firstReadStmt.getX();
    Value xPtr =
        getAtomicPointer(xRef, rewriter.getRemappedValue(xRef), loc, rewriter);
    ModuleOp mod = capture->getParentOfType<ModuleOp>();
    Type xType = getReferencedElementType(xRef, mod);
    Type xTypeOrig = getReferencedElementType(xRef);
    Value vRef = firstReadStmt.getV();
    vPtr =
        getAtomicPointer(vRef, rewriter.getRemappedValue(vRef), loc, rewriter);

    Value expr = nullptr;
    if (auto secondWriteStmt = dyn_cast<AtomicWriteOp>(secondOp)) {
      // 1. `{ atomic.read, atomic.write }` pattern
      expr = secondWriteStmt.getExpr();
    } else if (auto secondUpdateStmt = dyn_cast<AtomicUpdateOp>(secondOp)) {
      // 2. `{ atomic.read, atomic.update }` pattern
      Block &updateBlock = secondUpdateStmt.getRegion().front();
      Operation *terminator = updateBlock.getTerminator();
      expr = terminator->getOperand(0);
    }

    Block *loopBlock = constructCmpxchgLoop(xPtr, xType, expr, rewriter);
    Value loopArgument = loopBlock->getArgument(0);
    Operation &loopHead = loopBlock->front();
    auto condBr = cast<LLVM::CondBrOp>(loopBlock->back());
    storeVal = condBr.getFalseDestOperands()[0];

    rewriter.setInsertionPointToStart(loopBlock);
    loopArgument = deserializeExpr(loopArgument, xTypeOrig, rewriter);

    // Include flow dependency (v -> expr).
    if (auto secondUpdateStmt = dyn_cast<AtomicUpdateOp>(secondOp)) {
      Block &updateBlock = secondUpdateStmt.getRegion().front();
      Value updateArgument = updateBlock.getArgument(0);
      updateArgument.replaceAllUsesWith(loopArgument);
      updateBlock.walk([&](Operation *op) {
        if (!op->hasTrait<OpTrait::IsTerminator>())
          rewriter.moveOpBefore(op, &loopHead);
      });
    }
    moveDependency(vRef, vPtr, expr, loopArgument, loopHead, rewriter,
                   getLoadAddress);
  } else if (auto firstUpdateStmt = dyn_cast<AtomicUpdateOp>(firstOp)) {
    if (auto secondReadStmt = dyn_cast<AtomicReadOp>(secondOp)) {
      // 3. `{ atomic.update, atomic.read }` pattern
      storeVal = genUpdateCmpxchgLoop(firstUpdateStmt, rewriter);

      Value vRef = secondReadStmt.getV();
      vPtr = getAtomicPointer(vRef, rewriter.getRemappedValue(vRef),
                              capture.getLoc(), rewriter);
    }
  }
  // Generate `v = x`.
  rewriter.setInsertionPoint(capture);
  rewriter.replaceOpWithNewOp<LLVM::StoreOp>(capture, storeVal, vPtr);
  return success();
}

} // namespace

namespace mlir {

void configureACCAtomicConversionLegality(ConversionTarget &target) {
  target.addIllegalOp<AtomicReadOp, AtomicWriteOp, AtomicUpdateOp,
                      AtomicCaptureOp>();
}

void populateACCAtomicPatterns(const LLVMTypeConverter &converter,
                               RewritePatternSet &patterns,
                               acc::OpenACCSupport &accSupport,
                               ACCAtomicLoadAddressCallback getLoadAddress) {
  patterns.add<ACCAtomicOpConversion<AtomicReadOp>,
               ACCAtomicOpConversion<AtomicWriteOp>,
               ACCAtomicOpConversion<AtomicUpdateOp>,
               ACCAtomicOpConversion<AtomicCaptureOp>>(converter, accSupport,
                                                       getLoadAddress);
}

} // namespace mlir
