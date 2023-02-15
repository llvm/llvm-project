//===- SparseTensorCodegen.cpp - Sparse tensor primitives conversion ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass that converts sparse tensor types and primitives to actual compiler
// visible buffers and actual compiler IR that implements these primitives on
// the selected sparse tensor storage schemes. This pass provides an alternative
// to the SparseTensorConversion pass, eliminating the dependence on a runtime
// support library, and providing much more opportunities for subsequent
// compiler optimization of the generated code.
//
//===----------------------------------------------------------------------===//

#include "CodegenUtils.h"
#include "SparseTensorStorageLayout.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SparseTensor/IR/Enums.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"

#include <optional>

using namespace mlir;
using namespace mlir::sparse_tensor;

namespace {

using FuncGeneratorType =
    function_ref<void(OpBuilder &, ModuleOp, func::FuncOp, RankedTensorType)>;

static constexpr const char kInsertFuncNamePrefix[] = "_insert_";

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

/// Flatten a list of operands that may contain sparse tensors.
static void flattenOperands(ValueRange operands,
                            SmallVectorImpl<Value> &flattened) {
  // In case of
  // sparse_tensor, c, sparse_tensor
  // ==>
  // memref ..., c, memref ...
  for (auto operand : operands) {
    if (getSparseTensorEncoding(operand.getType())) {
      auto tuple = getTuple(operand);
      // An unrealized_conversion_cast will be inserted by type converter to
      // inter-mix the gap between 1:N conversion between sparse tensors and
      // fields. In this case, take the operands in the cast and replace the
      // sparse tensor output with the flattened type array.
      flattened.append(tuple.getOperands().begin(), tuple.getOperands().end());
    } else {
      flattened.push_back(operand);
    }
  }
}

/// Adds index conversions where needed.
static Value toType(OpBuilder &builder, Location loc, Value value, Type tp) {
  if (value.getType() != tp)
    return builder.create<arith::IndexCastOp>(loc, tp, value);
  return value;
}

/// Generates a load with proper index typing.
static Value genLoad(OpBuilder &builder, Location loc, Value mem, Value idx) {
  idx = toType(builder, loc, idx, builder.getIndexType());
  return builder.create<memref::LoadOp>(loc, mem, idx);
}

/// Generates a store with proper index typing and (for indices) proper value.
static void genStore(OpBuilder &builder, Location loc, Value val, Value mem,
                     Value idx) {
  idx = toType(builder, loc, idx, builder.getIndexType());
  val = toType(builder, loc, val,
               mem.getType().cast<ShapedType>().getElementType());
  builder.create<memref::StoreOp>(loc, val, mem, idx);
}

/// Creates a straightforward counting for-loop.
static scf::ForOp createFor(OpBuilder &builder, Location loc, Value upper,
                            MutableArrayRef<Value> fields,
                            Value lower = Value()) {
  Type indexType = builder.getIndexType();
  if (!lower)
    lower = constantZero(builder, loc, indexType);
  Value one = constantOne(builder, loc, indexType);
  scf::ForOp forOp = builder.create<scf::ForOp>(loc, lower, upper, one, fields);
  for (unsigned i = 0, e = fields.size(); i < e; i++)
    fields[i] = forOp.getRegionIterArg(i);
  builder.setInsertionPointToStart(forOp.getBody());
  return forOp;
}

/// Gets the dimension size for the given sparse tensor at the given
/// original dimension 'dim'.
static Value sizeFromTensorAtDim(OpBuilder &builder, Location loc,
                                 SparseTensorDescriptor desc, Dimension dim) {
  const SparseTensorType stt(desc.getRankedTensorType());
  // Access into static dimension can query original type directly.
  // Note that this is typically already done by DimOp's folding.
  if (auto sz = stt.getStaticDimSize(dim))
    return constantIndex(builder, loc, *sz);

  // Any other query can consult the dimSizes array at field DimSizesIdx,
  // accounting for the reordering applied to the sparse storage.
  // FIXME: `toStoredDim` is deprecated.
  const Level lvl = toStoredDim(stt, dim);
  // FIXME: this method seems to get *level* sizes, but the name is confusing
  return desc.getDimSize(builder, loc, lvl);
}

// Gets the dimension size at the given stored level 'lvl', either as a
// constant for a static size, or otherwise dynamically through memSizes.
static Value sizeFromTensorAtLvl(OpBuilder &builder, Location loc,
                                 SparseTensorDescriptor desc, Level lvl) {
  // FIXME: `toOrigDim` is deprecated.
  return sizeFromTensorAtDim(builder, loc, desc,
                             toOrigDim(desc.getRankedTensorType(), lvl));
}

static void createPushback(OpBuilder &builder, Location loc,
                           MutSparseTensorDescriptor desc,
                           SparseTensorFieldKind kind, std::optional<Level> lvl,
                           Value value, Value repeat = Value()) {
  Type etp = desc.getMemRefElementType(kind, lvl);
  Value field = desc.getMemRefField(kind, lvl);
  StorageSpecifierKind specFieldKind = toSpecifierKind(kind);

  auto pushBackOp = builder.create<PushBackOp>(
      loc, desc.getSpecifierField(builder, loc, specFieldKind, lvl), field,
      toType(builder, loc, value, etp), repeat);

  desc.setMemRefField(kind, lvl, pushBackOp.getOutBuffer());
  desc.setSpecifierField(builder, loc, specFieldKind, lvl,
                         pushBackOp.getNewSize());
}

/// Generates code that allocates a sparse storage scheme for given rank.
static void allocSchemeForRank(OpBuilder &builder, Location loc,
                               MutSparseTensorDescriptor desc, Level startLvl) {
  const SparseTensorType stt(desc.getRankedTensorType());
  Value linear = constantIndex(builder, loc, 1);
  const Level lvlRank = stt.getLvlRank();
  for (Level l = startLvl; l < lvlRank; l++) {
    const auto dlt = stt.getLvlType(l);
    if (isCompressedDLT(dlt)) {
      // Append linear x pointers, initialized to zero. Since each compressed
      // dimension initially already has a single zero entry, this maintains
      // the desired "linear + 1" length property at all times.
      Type ptrType = stt.getPointerType();
      Value ptrZero = constantZero(builder, loc, ptrType);
      createPushback(builder, loc, desc, SparseTensorFieldKind::PtrMemRef, l,
                     ptrZero, linear);
      return;
    }
    if (isSingletonDLT(dlt)) {
      return; // nothing to do
    }
    // Keep compounding the size, but nothing needs to be initialized
    // at this level. We will eventually reach a compressed level or
    // otherwise the values array for the from-here "all-dense" case.
    assert(isDenseDLT(dlt));
    Value size = sizeFromTensorAtLvl(builder, loc, desc, l);
    linear = builder.create<arith::MulIOp>(loc, linear, size);
  }
  // Reached values array so prepare for an insertion.
  Value valZero = constantZero(builder, loc, stt.getElementType());
  createPushback(builder, loc, desc, SparseTensorFieldKind::ValMemRef,
                 std::nullopt, valZero, linear);
}

/// Creates allocation operation.
static Value createAllocation(OpBuilder &builder, Location loc,
                              MemRefType memRefType, Value sz,
                              bool enableInit) {
  Value buffer = builder.create<memref::AllocOp>(loc, memRefType, sz);
  Type elemType = memRefType.getElementType();
  if (enableInit) {
    Value fillValue = constantZero(builder, loc, elemType);
    builder.create<linalg::FillOp>(loc, fillValue, buffer);
  }
  return buffer;
}

/// Creates allocation for each field in sparse tensor type. Note that
/// for all dynamic memrefs, the memory size is really the capacity of
/// the "vector", while the actual size resides in the sizes array.
///
/// TODO: for efficiency, we will need heuristics to make educated guesses
///       on the required capacities (see heuristic variable).
///
static void createAllocFields(OpBuilder &builder, Location loc,
                              SparseTensorType stt, ValueRange dynSizes,
                              bool enableInit, SmallVectorImpl<Value> &fields,
                              Value sizeHint) {
  // Build original sizes.
  assert((dynSizes.size() == static_cast<size_t>(stt.getNumDynamicDims())) &&
         "Got wrong number of dynamic sizes");
  const Dimension dimRank = stt.getDimRank();
  SmallVector<Value> dimSizes;
  dimSizes.reserve(dimRank);
  unsigned i = 0; // cumulative index into `dynSizes`.
  for (const DynSize sh : stt.getDimShape())
    dimSizes.push_back(ShapedType::isDynamic(sh)
                           ? dynSizes[i++]
                           : constantIndex(builder, loc, sh));

  // Set up some heuristic sizes. We try to set the initial
  // size based on available information. Otherwise we just
  // initialize a few elements to start the reallocation chain.
  // TODO: refine this
  Value ptrHeuristic, idxHeuristic, valHeuristic;
  if (stt.isAllDense()) {
    valHeuristic = dimSizes[0];
    for (const Value sz : ArrayRef<Value>{dimSizes}.drop_front())
      valHeuristic = builder.create<arith::MulIOp>(loc, valHeuristic, sz);
  } else if (sizeHint) {
    if (getCOOStart(stt.getEncoding()) == 0) {
      ptrHeuristic = constantIndex(builder, loc, 2);
      idxHeuristic = builder.create<arith::MulIOp>(
          loc, constantIndex(builder, loc, dimRank), sizeHint); // AOS
    } else if (dimRank == 2 && stt.isDenseLvl(0) && stt.isCompressedLvl(1)) {
      ptrHeuristic = builder.create<arith::AddIOp>(
          loc, sizeHint, constantIndex(builder, loc, 1));
      idxHeuristic = sizeHint;
    } else {
      ptrHeuristic = idxHeuristic = constantIndex(builder, loc, 16);
    }
    valHeuristic = sizeHint;
  } else {
    ptrHeuristic = idxHeuristic = valHeuristic =
        constantIndex(builder, loc, 16);
  }

  foreachFieldAndTypeInSparseTensor(
      stt,
      [&builder, &fields, stt, loc, ptrHeuristic, idxHeuristic, valHeuristic,
       enableInit](Type fType, FieldIndex fIdx, SparseTensorFieldKind fKind,
                   Level /*lvl*/, DimLevelType /*dlt*/) -> bool {
        assert(fields.size() == fIdx);
        Value field;
        switch (fKind) {
        case SparseTensorFieldKind::StorageSpec:
          field = SparseTensorSpecifier::getInitValue(builder, loc, stt);
          break;
        case SparseTensorFieldKind::PtrMemRef:
        case SparseTensorFieldKind::IdxMemRef:
        case SparseTensorFieldKind::ValMemRef:
          field = createAllocation(
              builder, loc, fType.cast<MemRefType>(),
              (fKind == SparseTensorFieldKind::PtrMemRef)   ? ptrHeuristic
              : (fKind == SparseTensorFieldKind::IdxMemRef) ? idxHeuristic
                                                            : valHeuristic,
              enableInit);
          break;
        }
        assert(field);
        fields.push_back(field);
        // Returns true to continue the iteration.
        return true;
      });

  MutSparseTensorDescriptor desc(stt, fields);

  // Initialize the storage scheme to an empty tensor. Initialized memSizes
  // to all zeros, sets the dimSizes to known values and gives all pointer
  // fields an initial zero entry, so that it is easier to maintain the
  // "linear + 1" length property.
  Value ptrZero = constantZero(builder, loc, stt.getPointerType());
  for (Level lvlRank = stt.getLvlRank(), l = 0; l < lvlRank; l++) {
    // Fills dim sizes array.
    // FIXME: this method seems to set *level* sizes, but the name is confusing
    // FIXME: `toOrigDim` is deprecated.
    desc.setDimSize(builder, loc, l, dimSizes[toOrigDim(stt, l)]);
    // Pushes a leading zero to pointers memref.
    if (stt.isCompressedLvl(l))
      createPushback(builder, loc, desc, SparseTensorFieldKind::PtrMemRef, l,
                     ptrZero);
  }
  allocSchemeForRank(builder, loc, desc, /*rank=*/0);
}

/// Helper method that generates block specific to compressed case:
///
///  plo = pointers[l][pos[l-1]]
///  phi = pointers[l][pos[l-1]+1]
///  msz = indices[l].size()
///  if (plo < phi) {
///    present = indices[l][phi-1] == i[l]
///  } else { // first insertion
///    present = false
///    pointers[l][pos[l-1]] = msz
///  }
///  if (present) { // index already present
///    next = phi-1
///  } else {
///    indices[l].push_back(i[l])
///    pointers[l][pos[l-1]+1] = msz+1
///    next = msz
///    <prepare level l + 1>
///  }
///  pos[l] = next
static Value genCompressed(OpBuilder &builder, Location loc,
                           MutSparseTensorDescriptor desc, ValueRange indices,
                           Value value, Value pos, Level lvl) {
  const SparseTensorType stt(desc.getRankedTensorType());
  const Level lvlRank = stt.getLvlRank();
  assert(lvl < lvlRank && "Level is out of bounds");
  assert(indices.size() == static_cast<size_t>(lvlRank) &&
         "Level-rank mismatch");
  SmallVector<Type> types;
  Type indexType = builder.getIndexType();
  Type boolType = builder.getIntegerType(1);
  unsigned idxIndex;
  unsigned idxStride;
  std::tie(idxIndex, idxStride) = desc.getIdxMemRefIndexAndStride(lvl);
  Value one = constantIndex(builder, loc, 1);
  Value pp1 = builder.create<arith::AddIOp>(loc, pos, one);
  Value plo = genLoad(builder, loc, desc.getPtrMemRef(lvl), pos);
  Value phi = genLoad(builder, loc, desc.getPtrMemRef(lvl), pp1);
  Value msz = desc.getIdxMemSize(builder, loc, lvl);
  Value idxStrideC;
  if (idxStride > 1) {
    idxStrideC = constantIndex(builder, loc, idxStride);
    msz = builder.create<arith::DivUIOp>(loc, msz, idxStrideC);
  }
  Value phim1 = builder.create<arith::SubIOp>(
      loc, toType(builder, loc, phi, indexType), one);
  // Conditional expression.
  Value lt =
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, plo, phi);
  types.push_back(boolType);
  scf::IfOp ifOp1 = builder.create<scf::IfOp>(loc, types, lt, /*else*/ true);
  types.pop_back();
  builder.setInsertionPointToStart(&ifOp1.getThenRegion().front());
  Value crd = genLoad(
      builder, loc, desc.getMemRefField(idxIndex),
      idxStride > 1 ? builder.create<arith::MulIOp>(loc, phim1, idxStrideC)
                    : phim1);
  Value eq = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                           toType(builder, loc, crd, indexType),
                                           indices[lvl]);
  builder.create<scf::YieldOp>(loc, eq);
  builder.setInsertionPointToStart(&ifOp1.getElseRegion().front());
  if (lvl > 0)
    genStore(builder, loc, msz, desc.getPtrMemRef(lvl), pos);
  builder.create<scf::YieldOp>(loc, constantI1(builder, loc, false));
  builder.setInsertionPointAfter(ifOp1);
  // If present construct. Note that for a non-unique dimension level, we
  // simply set the condition to false and rely on CSE/DCE to clean up the IR.
  //
  // TODO: generate less temporary IR?
  //
  for (unsigned i = 0, e = desc.getNumFields(); i < e; i++)
    types.push_back(desc.getField(i).getType());
  types.push_back(indexType);
  const Value p = stt.isUniqueLvl(lvl) ? ifOp1.getResult(0)
                                       : constantI1(builder, loc, false);
  scf::IfOp ifOp2 = builder.create<scf::IfOp>(loc, types, p, /*else*/ true);
  // If present (fields unaffected, update next to phim1).
  builder.setInsertionPointToStart(&ifOp2.getThenRegion().front());

  // FIXME: This does not looks like a clean way, but probably the most
  // efficient way.
  desc.getFields().push_back(phim1);
  builder.create<scf::YieldOp>(loc, desc.getFields());
  desc.getFields().pop_back();

  // If !present (changes fields, update next).
  builder.setInsertionPointToStart(&ifOp2.getElseRegion().front());
  Value mszp1 = builder.create<arith::AddIOp>(loc, msz, one);
  genStore(builder, loc, mszp1, desc.getPtrMemRef(lvl), pp1);
  createPushback(builder, loc, desc, SparseTensorFieldKind::IdxMemRef, lvl,
                 indices[lvl]);
  // Prepare the next dimension "as needed".
  if ((lvl + 1) < lvlRank)
    allocSchemeForRank(builder, loc, desc, lvl + 1);

  desc.getFields().push_back(msz);
  builder.create<scf::YieldOp>(loc, desc.getFields());
  desc.getFields().pop_back();

  // Update fields and return next pos.
  builder.setInsertionPointAfter(ifOp2);
  unsigned o = 0;
  for (unsigned i = 0, e = desc.getNumFields(); i < e; i++)
    desc.setField(i, ifOp2.getResult(o++));
  return ifOp2.getResult(o);
}

/// Generates code along an insertion path without the need for a "cursor".
/// This current insertion strategy comes at the expense of some testing
/// overhead for each insertion. The strategy will be optimized later for
/// common insertion patterns. The current insertion strategy also assumes
/// insertions occur in "a reasonable order" that enables building the
/// storage scheme in an appending/inserting kind of fashion (i.e. no
/// in-between insertions that need data movement). The implementation
/// relies on CSE/DCE to clean up all bookkeeping that is not needed.
///
/// TODO: better unord/not-unique; also generalize, optimize, specialize!
///
static void genInsertBody(OpBuilder &builder, ModuleOp module,
                          func::FuncOp func, RankedTensorType rtp) {
  const OpBuilder::InsertionGuard insertionGuard(builder);
  Block *const entryBlock = func.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  const ValueRange args = entryBlock->getArguments();
  const Location loc = func.getLoc();
  const SparseTensorType stt(rtp);
  const Level lvlRank = stt.getLvlRank();

  // Construct fields and indices arrays from parameters.
  SmallVector<Value> fields = llvm::to_vector(args.drop_back(lvlRank + 1));
  MutSparseTensorDescriptor desc(rtp, fields);
  const SmallVector<Value> indices =
      llvm::to_vector(args.take_back(lvlRank + 1).drop_back());
  Value value = args.back();
  Value pos = constantZero(builder, loc, builder.getIndexType());
  // Generate code for every level.
  for (Level l = 0; l < lvlRank; l++) {
    const auto dlt = stt.getLvlType(l);
    if (isCompressedDLT(dlt)) {
      // Create:
      //   if (!present) {
      //     indices[l].push_back(i[l])
      //     <update pointers and prepare level l + 1>
      //   }
      //   pos[l] = indices.size() - 1
      //   <insert @ pos[l] at next level l + 1>
      pos = genCompressed(builder, loc, desc, indices, value, pos, l);
    } else if (isSingletonDLT(dlt)) {
      // Create:
      //   indices[l].push_back(i[l])
      //   pos[l] = pos[l-1]
      //   <insert @ pos[l] at next level l + 1>
      createPushback(builder, loc, desc, SparseTensorFieldKind::IdxMemRef, l,
                     indices[l]);
    } else {
      assert(isDenseDLT(dlt));
      // Construct the new position as:
      //   pos[l] = size * pos[l-1] + i[l]
      //   <insert @ pos[l] at next level l + 1>
      Value size = sizeFromTensorAtLvl(builder, loc, desc, l);
      Value mult = builder.create<arith::MulIOp>(loc, size, pos);
      pos = builder.create<arith::AddIOp>(loc, mult, indices[l]);
    }
  }
  // Reached the actual value append/insert.
  if (!stt.isDenseLvl(lvlRank - 1))
    createPushback(builder, loc, desc, SparseTensorFieldKind::ValMemRef,
                   std::nullopt, value);
  else
    genStore(builder, loc, value, desc.getValMemRef(), pos);
  builder.create<func::ReturnOp>(loc, fields);
}

/// Generates a call to a function to perform an insertion operation. If the
/// function doesn't exist yet, call `createFunc` to generate the function.
static void genInsertionCallHelper(OpBuilder &builder,
                                   MutSparseTensorDescriptor desc,
                                   SmallVectorImpl<Value> &indices, Value value,
                                   func::FuncOp insertPoint,
                                   StringRef namePrefix,
                                   FuncGeneratorType createFunc) {
  // The mangled name of the function has this format:
  //   <namePrefix>_<DLT>_<shape>_<ordering>_<eltType>
  //     _<indexBitWidth>_<pointerBitWidth>
  const SparseTensorType stt(desc.getRankedTensorType());
  SmallString<32> nameBuffer;
  llvm::raw_svector_ostream nameOstream(nameBuffer);
  nameOstream << namePrefix;
  assert(static_cast<size_t>(stt.getLvlRank()) == indices.size());
  const Level lvlRank = stt.getLvlRank();
  for (Level l = 0; l < lvlRank; l++)
    nameOstream << toMLIRString(stt.getLvlType(l)) << "_";
  // Static dim sizes are used in the generated code while dynamic sizes are
  // loaded from the dimSizes buffer. This is the reason for adding the shape
  // to the function name.
  for (const auto sh : stt.getDimShape())
    nameOstream << sh << "_";
  // Permutation information is also used in generating insertion.
  if (!stt.isIdentity())
    nameOstream << stt.getDimToLvlMap() << "_";
  nameOstream << stt.getElementType() << "_";
  nameOstream << stt.getIndexBitWidth() << "_" << stt.getPointerBitWidth();

  // Look up the function.
  ModuleOp module = insertPoint->getParentOfType<ModuleOp>();
  MLIRContext *context = module.getContext();
  auto result = SymbolRefAttr::get(context, nameOstream.str());
  auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());

  // Construct parameters for fields and indices.
  SmallVector<Value> operands = llvm::to_vector(desc.getFields());
  operands.append(indices);
  operands.push_back(value);
  Location loc = insertPoint.getLoc();

  if (!func) {
    // Create the function.
    OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPoint(insertPoint);

    func = builder.create<func::FuncOp>(
        loc, nameOstream.str(),
        FunctionType::get(context, ValueRange(operands).getTypes(),
                          ValueRange(desc.getFields()).getTypes()));
    func.setPrivate();
    createFunc(builder, module, func, stt);
  }

  // Generate a call to perform the insertion and update `fields` with values
  // returned from the call.
  func::CallOp call = builder.create<func::CallOp>(loc, func, operands);
  for (size_t i = 0, e = desc.getNumFields(); i < e; i++) {
    desc.getFields()[i] = call.getResult(i);
  }
}

/// Generations insertion finalization code.
static void genEndInsert(OpBuilder &builder, Location loc,
                         SparseTensorDescriptor desc) {
  const SparseTensorType stt(desc.getRankedTensorType());
  const Level lvlRank = stt.getLvlRank();
  for (Level l = 0; l < lvlRank; l++) {
    const auto dlt = stt.getLvlType(l);
    if (isCompressedDLT(dlt)) {
      // Compressed dimensions need a pointer cleanup for all entries
      // that were not visited during the insertion pass.
      //
      // TODO: avoid cleanup and keep compressed scheme consistent at all
      // times?
      //
      if (l > 0) {
        Type ptrType = stt.getPointerType();
        Value ptrMemRef = desc.getPtrMemRef(l);
        Value hi = desc.getPtrMemSize(builder, loc, l);
        Value zero = constantIndex(builder, loc, 0);
        Value one = constantIndex(builder, loc, 1);
        // Vector of only one, but needed by createFor's prototype.
        SmallVector<Value, 1> inits{genLoad(builder, loc, ptrMemRef, zero)};
        scf::ForOp loop = createFor(builder, loc, hi, inits, one);
        Value i = loop.getInductionVar();
        Value oldv = loop.getRegionIterArg(0);
        Value newv = genLoad(builder, loc, ptrMemRef, i);
        Value ptrZero = constantZero(builder, loc, ptrType);
        Value cond = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, newv, ptrZero);
        scf::IfOp ifOp = builder.create<scf::IfOp>(loc, TypeRange(ptrType),
                                                   cond, /*else*/ true);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        genStore(builder, loc, oldv, ptrMemRef, i);
        builder.create<scf::YieldOp>(loc, oldv);
        builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
        builder.create<scf::YieldOp>(loc, newv);
        builder.setInsertionPointAfter(ifOp);
        builder.create<scf::YieldOp>(loc, ifOp.getResult(0));
        builder.setInsertionPointAfter(loop);
      }
    } else {
      assert(isDenseDLT(dlt) || isSingletonDLT(dlt));
    }
  }
}

/// Returns a memref that fits the requested length (reallocates if requested
/// length is larger, or creates a subview if it is smaller).
static Value reallocOrSubView(OpBuilder &builder, Location loc, int64_t len,
                              Value buffer) {
  MemRefType memTp = getMemRefType(buffer);
  auto retTp = MemRefType::get(ArrayRef{len}, memTp.getElementType());

  Value targetLen = constantIndex(builder, loc, len);
  Value bufferLen = linalg::createOrFoldDimOp(builder, loc, buffer, 0);
  Value reallocP = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                                 targetLen, bufferLen);
  scf::IfOp ifOp = builder.create<scf::IfOp>(loc, retTp, reallocP, true);
  // If targetLen > bufferLen, reallocate to get enough sparse to return.
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  Value reallocBuf = builder.create<memref::ReallocOp>(loc, retTp, buffer);
  builder.create<scf::YieldOp>(loc, reallocBuf);
  // Else, return a subview to fit the size.
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  Value subViewBuf = builder.create<memref::SubViewOp>(
      loc, retTp, buffer, /*offset=*/ArrayRef<int64_t>{0},
      /*size=*/ArrayRef<int64_t>{len},
      /*stride=*/ArrayRef<int64_t>{1});
  builder.create<scf::YieldOp>(loc, subViewBuf);
  // Resets insertion point.
  builder.setInsertionPointAfter(ifOp);
  return ifOp.getResult(0);
}

//===----------------------------------------------------------------------===//
// Codegen rules.
//===----------------------------------------------------------------------===//

/// Sparse tensor storage conversion rule for returns.
class SparseReturnConverter : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> flattened;
    flattenOperands(adaptor.getOperands(), flattened);
    // Create a return with the flattened value extracted from sparse tensors.
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, flattened);
    return success();
  }
};

/// Sparse tensor storage conversion rule for calls.
class SparseCallConverter : public OpConversionPattern<func::CallOp> {
public:
  // The default CallOp converter can not handle 1:N type conversion.
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // In case of:
    //  sparse_tensor, f, sparse_tensor = call @foo(...)
    // ==>
    //  memref..., f, memref = call @foo(...) replace with
    //  cast(memref...)->sparse_tensor, f, cast(memref...)->sparse_tensor
    SmallVector<Type> finalRetTy;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), finalRetTy)))
      return failure();

    // (1) Genereates new call with flattened return value.
    SmallVector<Value> flattened;
    flattenOperands(adaptor.getOperands(), flattened);
    auto newCall = rewriter.create<func::CallOp>(loc, op.getCallee(),
                                                 finalRetTy, flattened);
    // (2) Create cast operation for sparse tensor returns.
    SmallVector<Value> castedRet;
    // Tracks the offset of current return value (of the orignal call)
    // relative to the new call (after sparse tensor flattening);
    unsigned retOffset = 0;
    // Temporal buffer to hold the flattened list of type for
    // a sparse tensor.
    SmallVector<Type> sparseFlat;
    for (auto ret : op.getResults()) {
      assert(retOffset < newCall.getNumResults());
      auto retType = ret.getType();
      if (failed(typeConverter->convertType(retType, sparseFlat)))
        // This should never happen.
        llvm_unreachable("Failed to convert type in sparse tensor codegen");

      // Converted types can not be empty when the type conversion succeed.
      assert(!sparseFlat.empty());
      if (sparseFlat.size() > 1) {
        auto flatSize = sparseFlat.size();
        ValueRange fields(iterator_range<ResultRange::iterator>(
            newCall.result_begin() + retOffset,
            newCall.result_begin() + retOffset + flatSize));
        castedRet.push_back(genTuple(rewriter, loc, retType, fields));
        retOffset += flatSize;
      } else {
        // If this is an 1:1 conversion, no need for casting.
        castedRet.push_back(newCall.getResult(retOffset));
        retOffset++;
      }
      sparseFlat.clear();
    }

    assert(castedRet.size() == op.getNumResults());
    rewriter.replaceOp(op, castedRet);
    return success();
  }
};

/// Sparse codegen rule for dimension accesses.
class SparseDimOpConverter : public OpConversionPattern<tensor::DimOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::DimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<int64_t> index = op.getConstantIndex();
    if (!index || !getSparseTensorEncoding(adaptor.getSource().getType()))
      return failure();

    auto desc = getDescriptorFromTensorTuple(adaptor.getSource());
    auto sz = sizeFromTensorAtDim(rewriter, op.getLoc(), desc, *index);

    rewriter.replaceOp(op, sz);
    return success();
  }
};

/// Sparse codegen rule for trivial tensor casts.
class SparseCastConverter : public OpConversionPattern<tensor::CastOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only rewrite identically annotated source/dest.
    auto encDst = getSparseTensorEncoding(op.getType());
    auto encSrc = getSparseTensorEncoding(op.getSource().getType());
    if (!encDst || encDst != encSrc)
      return failure();
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

/// Sparse codgen rule for the alloc operator.
class SparseTensorAllocConverter
    : public OpConversionPattern<bufferization::AllocTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  SparseTensorAllocConverter(TypeConverter &typeConverter, MLIRContext *context,
                             bool enableInit)
      : OpConversionPattern(typeConverter, context),
        enableBufferInitialization(enableInit) {}

  LogicalResult
  matchAndRewrite(bufferization::AllocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto resType = getSparseTensorType(op);
    if (!resType.hasEncoding())
      return failure();
    if (op.getCopy())
      return rewriter.notifyMatchFailure(op, "tensor copy not implemented");

    // Construct allocation for each field.
    const Location loc = op.getLoc();
    const Value sizeHint = op.getSizeHint();
    const ValueRange dynSizes = adaptor.getDynamicSizes();
    const size_t found = dynSizes.size();
    const int64_t expected = resType.getNumDynamicDims();
    if (found != static_cast<size_t>(expected))
      return rewriter.notifyMatchFailure(
          op, llvm::formatv(
                  "Got wrong number of dynamic sizes: Found={0}, Expected={1}",
                  found, expected));
    SmallVector<Value> fields;
    createAllocFields(rewriter, loc, resType, dynSizes,
                      enableBufferInitialization, fields, sizeHint);
    // Replace operation with resulting memrefs.
    rewriter.replaceOp(op, genTuple(rewriter, loc, resType, fields));
    return success();
  }

private:
  bool enableBufferInitialization;
};

/// Sparse codegen rule for the dealloc operator.
class SparseTensorDeallocConverter
    : public OpConversionPattern<bufferization::DeallocTensorOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(bufferization::DeallocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto enc = getSparseTensorEncoding(op.getTensor().getType());
    if (!enc)
      return failure();

    // Replace the sparse tensor deallocation with field deallocations.
    Location loc = op.getLoc();
    auto desc = getDescriptorFromTensorTuple(adaptor.getTensor());
    for (auto input : desc.getMemRefFields())
      // Deallocate every buffer used to store the sparse tensor handler.
      rewriter.create<memref::DeallocOp>(loc, input);

    rewriter.eraseOp(op);
    return success();
  }
};

/// Sparse codegen rule for tensor rematerialization.
class SparseTensorLoadConverter : public OpConversionPattern<LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prepare descriptor.
    auto desc = getDescriptorFromTensorTuple(adaptor.getTensor());
    // Generate optional insertion finalization code.
    if (op.getHasInserts())
      genEndInsert(rewriter, op.getLoc(), desc);
    // Replace operation with resulting memrefs.
    rewriter.replaceOp(op, genTuple(rewriter, op.getLoc(), desc));
    return success();
  }
};

/// Sparse codegen rule for the expand op.
class SparseExpandConverter : public OpConversionPattern<ExpandOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ExpandOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!getSparseTensorEncoding(op.getTensor().getType()))
      return failure();
    Location loc = op->getLoc();
    auto desc = getDescriptorFromTensorTuple(adaptor.getTensor());
    const auto srcType = getSparseTensorType(op.getTensor());
    Type eltType = srcType.getElementType();
    Type boolType = rewriter.getIntegerType(1);
    Type idxType = rewriter.getIndexType();
    // All initialization should be done on entry of the loop nest.
    rewriter.setInsertionPointAfter(op.getTensor().getDefiningOp());
    // Determine the size for access expansion (always the innermost stored
    // dimension size, translated back to original dimension). Note that we
    // recursively rewrite the new DimOp on the **original** tensor.
    // FIXME: `toOrigDim` is deprecated.
    const Dimension innerDim = toOrigDim(srcType, srcType.getLvlRank() - 1);
    const auto sz = sizeFromTensorAtDim(rewriter, loc, desc, innerDim);
    // Generate a memref for `sz` elements of type `t`.
    const auto genAlloc = [&](Type t) {
      const auto memTp = MemRefType::get({ShapedType::kDynamic}, t);
      return rewriter.create<memref::AllocOp>(loc, memTp, ValueRange{sz});
    };
    // Allocate temporary buffers for values/filled-switch and added.
    // We do not use stack buffers for this, since the expanded size may
    // be rather large (as it envelops a single expanded dense dimension).
    Value values = genAlloc(eltType);
    Value filled = genAlloc(boolType);
    Value added = genAlloc(idxType);
    Value zero = constantZero(rewriter, loc, idxType);
    // Reset the values/filled-switch to all-zero/false. Note that this
    // introduces an O(N) operation into the computation, but this reset
    // operation is amortized over the innermost loops for the access
    // pattern expansion. As noted in the operation doc, we would like
    // to amortize this setup cost even between kernels.
    rewriter.create<linalg::FillOp>(
        loc, ValueRange{constantZero(rewriter, loc, eltType)},
        ValueRange{values});
    rewriter.create<linalg::FillOp>(
        loc, ValueRange{constantZero(rewriter, loc, boolType)},
        ValueRange{filled});
    // Replace expansion op with these buffers and initial index.
    assert(op.getNumResults() == 4);
    rewriter.replaceOp(op, {values, filled, added, zero});
    return success();
  }
};

/// Sparse codegen rule for the compress operator.
class SparseCompressConverter : public OpConversionPattern<CompressOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    SmallVector<Value> fields;
    auto desc = getMutDescriptorFromTensorTuple(adaptor.getTensor(), fields);
    Value values = adaptor.getValues();
    Value filled = adaptor.getFilled();
    Value added = adaptor.getAdded();
    Value count = adaptor.getCount();
    const SparseTensorType dstType(desc.getRankedTensorType());
    Type eltType = dstType.getElementType();
    // Prepare indices.
    SmallVector<Value> indices(adaptor.getIndices());
    // If the innermost level is ordered, we need to sort the indices
    // in the "added" array prior to applying the compression.
    if (dstType.isOrderedLvl(dstType.getLvlRank() - 1))
      rewriter.create<SortOp>(loc, count, ValueRange{added}, ValueRange{},
                              SparseTensorSortKind::HybridQuickSort);
    // While performing the insertions, we also need to reset the elements
    // of the values/filled-switch by only iterating over the set elements,
    // to ensure that the runtime complexity remains proportional to the
    // sparsity of the expanded access pattern.
    //
    // Generate
    //    out_memrefs = for (i = 0; i < count; i++)(in_memrefs) {
    //      index = added[i];
    //      value = values[index];
    //      insert({prev_indices, index}, value);
    //      new_memrefs = insert(in_memrefs, {prev_indices, index}, value);
    //      values[index] = 0;
    //      filled[index] = false;
    //      yield new_memrefs
    //    }
    scf::ForOp loop = createFor(rewriter, loc, count, desc.getFields());
    Value i = loop.getInductionVar();
    Value index = genLoad(rewriter, loc, added, i);
    Value value = genLoad(rewriter, loc, values, index);
    indices.push_back(index);
    // TODO: faster for subsequent insertions?
    auto insertPoint = op->template getParentOfType<func::FuncOp>();
    genInsertionCallHelper(rewriter, desc, indices, value, insertPoint,
                           kInsertFuncNamePrefix, genInsertBody);
    genStore(rewriter, loc, constantZero(rewriter, loc, eltType), values,
             index);
    genStore(rewriter, loc, constantI1(rewriter, loc, false), filled, index);
    rewriter.create<scf::YieldOp>(loc, desc.getFields());
    rewriter.setInsertionPointAfter(loop);
    Value result = genTuple(rewriter, loc, dstType, loop->getResults());
    // Deallocate the buffers on exit of the full loop nest.
    Operation *parent = getTop(op);
    rewriter.setInsertionPointAfter(parent);
    rewriter.create<memref::DeallocOp>(loc, values);
    rewriter.create<memref::DeallocOp>(loc, filled);
    rewriter.create<memref::DeallocOp>(loc, added);
    // Replace operation with resulting memrefs.
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Sparse codegen rule for the insert operator.
class SparseInsertConverter : public OpConversionPattern<InsertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> fields;
    auto desc = getMutDescriptorFromTensorTuple(adaptor.getTensor(), fields);
    // Prepare and indices.
    SmallVector<Value> indices(adaptor.getIndices());
    // Generate insertion.
    Value value = adaptor.getValue();
    auto insertPoint = op->template getParentOfType<func::FuncOp>();
    genInsertionCallHelper(rewriter, desc, indices, value, insertPoint,
                           kInsertFuncNamePrefix, genInsertBody);

    // Replace operation with resulting memrefs.
    rewriter.replaceOp(op, genTuple(rewriter, op.getLoc(), desc));
    return success();
  }
};

/// Sparse codegen rule for pointer accesses.
class SparseToPointersConverter : public OpConversionPattern<ToPointersOp> {
public:
  using OpAdaptor = typename ToPointersOp::Adaptor;
  using OpConversionPattern<ToPointersOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToPointersOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the requested pointer access with corresponding field.
    // The cast_op is inserted by type converter to intermix 1:N type
    // conversion.
    auto desc = getDescriptorFromTensorTuple(adaptor.getTensor());
    uint64_t dim = op.getDimension().getZExtValue();
    rewriter.replaceOp(op, desc.getPtrMemRef(dim));
    return success();
  }
};

/// Sparse codegen rule for index accesses.
class SparseToIndicesConverter : public OpConversionPattern<ToIndicesOp> {
public:
  using OpAdaptor = typename ToIndicesOp::Adaptor;
  using OpConversionPattern<ToIndicesOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToIndicesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the requested pointer access with corresponding field.
    // The cast_op is inserted by type converter to intermix 1:N type
    // conversion.
    Location loc = op.getLoc();
    auto desc = getDescriptorFromTensorTuple(adaptor.getTensor());
    uint64_t dim = op.getDimension().getZExtValue();
    Value field = desc.getIdxMemRefOrView(rewriter, loc, dim);

    // Insert a cast to bridge the actual type to the user expected type. If the
    // actual type and the user expected type aren't compatible, the compiler or
    // the runtime will issue an error.
    Type resType = op.getResult().getType();
    if (resType != field.getType())
      field = rewriter.create<memref::CastOp>(loc, resType, field);
    rewriter.replaceOp(op, field);

    return success();
  }
};

/// Sparse codegen rule for accessing the linear indices buffer.
class SparseToIndicesBufferConverter
    : public OpConversionPattern<ToIndicesBufferOp> {
public:
  using OpAdaptor = typename ToIndicesBufferOp::Adaptor;
  using OpConversionPattern<ToIndicesBufferOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToIndicesBufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the requested pointer access with corresponding field.
    // The cast_op is inserted by type converter to intermix 1:N type
    // conversion.
    auto desc = getDescriptorFromTensorTuple(adaptor.getTensor());
    rewriter.replaceOp(op, desc.getAOSMemRef());

    return success();
  }
};

/// Sparse codegen rule for value accesses.
class SparseToValuesConverter : public OpConversionPattern<ToValuesOp> {
public:
  using OpAdaptor = typename ToValuesOp::Adaptor;
  using OpConversionPattern<ToValuesOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToValuesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the requested pointer access with corresponding field.
    // The cast_op is inserted by type converter to intermix 1:N type
    // conversion.
    auto desc = getDescriptorFromTensorTuple(adaptor.getTensor());
    rewriter.replaceOp(op, desc.getValMemRef());
    return success();
  }
};

/// Sparse codegen rule for the convert operator.
class SparseConvertConverter : public OpConversionPattern<ConvertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SparseTensorEncodingAttr encDst = getSparseTensorEncoding(op.getType());
    SparseTensorEncodingAttr encSrc =
        getSparseTensorEncoding(op.getSource().getType());
    if (encDst != encSrc) {
      // This should be handled by rewriting before codegen.
      return failure();
    }
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

class SparseExtractSliceCoverter
    : public OpConversionPattern<tensor::ExtractSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcEnc = getSparseTensorEncoding(op.getSourceType());
    auto dstEnc = getSparseTensorEncoding(op.getResult().getType());
    if (!srcEnc && !dstEnc)
      return failure();

    // TODO: We should check these in ExtractSliceOp::verify.
    assert(srcEnc && dstEnc && dstEnc.isSlice());
    assert(srcEnc.getDimLevelType() == dstEnc.getDimLevelType());
    assert(srcEnc.getDimOrdering() == dstEnc.getDimOrdering());
    assert(srcEnc.getHigherOrdering() == dstEnc.getHigherOrdering());
    assert(srcEnc.getPointerBitWidth() == dstEnc.getPointerBitWidth());
    assert(srcEnc.getIndexBitWidth() == dstEnc.getIndexBitWidth());

    // TODO: support dynamic slices.
    for (int i = 0, e = op.getSourceType().getRank(); i < e; i++) {
      assert(op.getStaticStrides()[i] == dstEnc.getStaticDimSliceStride(i));
      assert(op.getStaticOffsets()[i] == dstEnc.getStaticDimSliceOffset(i));
      assert(op.getStaticSizes()[i] == dstEnc.getStaticDimSliceSize(i));
    }

    // TODO: create a new specifer for slices (need to encode slice metadata).
    // It does not matter now because only constant offset/stride are allowed.
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

/// Sparse codegen rule for number of entries operator.
class SparseNumberOfEntriesConverter
    : public OpConversionPattern<NumberOfEntriesOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NumberOfEntriesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Query memSizes for the actually stored values.
    rewriter.replaceOp(
        op, genValMemSize(rewriter, op.getLoc(), adaptor.getTensor()));
    return success();
  }
};

struct SparsePackOpConverter : public OpConversionPattern<PackOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    const auto rtp = getRankedTensorType(op.getResult());
    assert(isUniqueCOOType(rtp));

    SmallVector<Value> fields;
    Location loc = op.getLoc();

    foreachFieldAndTypeInSparseTensor(
        rtp,
        [&rewriter, &fields, &op, rtp,
         loc](Type fType, FieldIndex fIdx, SparseTensorFieldKind fKind,
              Level /*lvl*/, DimLevelType /*dlt*/) -> bool {
          assert(fields.size() == fIdx);
          auto enc = getSparseTensorEncoding(rtp);
          Value field;
          switch (fKind) {
          case SparseTensorFieldKind::StorageSpec:
            field = SparseTensorSpecifier::getInitValue(rewriter, loc, rtp);
            break;
          case SparseTensorFieldKind::PtrMemRef: {
            // TACO-style COO starts with a PtrBuffer
            // By creating a constant value for it, we avoid the complexity of
            // memory management.
            auto tensorType = RankedTensorType::get({2}, enc.getPointerType());
            auto memrefType = MemRefType::get(tensorType.getShape(),
                                              tensorType.getElementType());
            auto cstPtr = rewriter.create<arith::ConstantOp>(
                loc, tensorType,
                DenseElementsAttr::get(
                    tensorType,
                    ArrayRef<Attribute>{
                        IntegerAttr::get(enc.getPointerType(), 0),
                        IntegerAttr::get(
                            enc.getPointerType(),
                            op.getData().getType().getShape()[0])}));
            field = rewriter.create<bufferization::ToMemrefOp>(loc, memrefType,
                                                               cstPtr);
            break;
          }
          case SparseTensorFieldKind::IdxMemRef: {
            auto tensorType = op.getIndices().getType();
            auto memrefType = MemRefType::get(tensorType.getShape(),
                                              tensorType.getElementType());
            auto idxMemRef = rewriter.create<bufferization::ToMemrefOp>(
                op->getLoc(), memrefType, op.getIndices());
            ReassociationIndices reassociation;
            for (int i = 0, e = tensorType.getRank(); i < e; i++)
              reassociation.push_back(i);

            // Flattened the indices buffer to rank 1.
            field = rewriter.create<memref::CollapseShapeOp>(
                loc, idxMemRef, ArrayRef<ReassociationIndices>(reassociation));
            break;
          }
          case SparseTensorFieldKind::ValMemRef: {
            auto tensorType = op.getData().getType();
            auto memrefType = MemRefType::get(tensorType.getShape(),
                                              tensorType.getElementType());
            field = rewriter.create<bufferization::ToMemrefOp>(
                op->getLoc(), memrefType, op.getData());
            break;
          }
          }

          assert(field);
          if (fType != field.getType())
            field = rewriter.create<memref::CastOp>(loc, fType, field);
          fields.push_back(field);
          // Returns true to continue the iteration.
          return true;
        });

    MutSparseTensorDescriptor desc(rtp, fields);
    auto noe = linalg::createOrFoldDimOp(rewriter, loc, op.getData(), 0);
    for (unsigned i = 0, e = rtp.getRank(); i < e; i++) {
      int dim = rtp.getShape()[i];
      assert(!ShapedType::isDynamic(dim));
      desc.setDimSize(rewriter, loc, i, constantIndex(rewriter, loc, dim));
      if (i == 0)
        desc.setPtrMemSize(rewriter, loc, i, constantIndex(rewriter, loc, 2));

      desc.setIdxMemSize(rewriter, loc, i, noe);
    }
    desc.setValMemSize(rewriter, loc, noe);

    rewriter.replaceOp(op, genTuple(rewriter, loc, desc));
    return success();
  }
};

struct SparseUnpackOpConverter : public OpConversionPattern<UnpackOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnpackOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto desc = getDescriptorFromTensorTuple(adaptor.getTensor());
    Location loc = op.getLoc();
    int64_t rank = op.getTensor().getType().getRank();

    assert(isUniqueCOOType(op.getTensor().getType()) &&
           desc.getFields().size() == 4);

    Value flatBuf = rank == 1 ? desc.getIdxMemRefOrView(rewriter, loc, 0)
                              : desc.getAOSMemRef();
    Value dataBuf = desc.getValMemRef();

    // If frontend requests a static buffer, we reallocate the data/indices
    // to ensure that we meet their need.
    TensorType dataTp = op.getData().getType();
    if (dataTp.hasStaticShape()) {
      dataBuf = reallocOrSubView(rewriter, loc, dataTp.getShape()[0], dataBuf);
    }

    TensorType indicesTp = op.getIndices().getType();
    if (indicesTp.hasStaticShape()) {
      auto len = indicesTp.getShape()[0] * indicesTp.getShape()[1];
      flatBuf = reallocOrSubView(rewriter, loc, len, flatBuf);
    }

    Value idxBuf = rewriter.create<memref::ExpandShapeOp>(
        loc, MemRefType::get(indicesTp.getShape(), indicesTp.getElementType()),
        flatBuf, ArrayRef{ReassociationIndices{0, 1}});

    // Converts MemRefs back to Tensors.
    Value data = rewriter.create<bufferization::ToTensorOp>(loc, dataBuf);
    Value indices = rewriter.create<bufferization::ToTensorOp>(loc, idxBuf);
    Value nnz = toType(rewriter, loc, desc.getValMemSize(rewriter, loc),
                       op.getNnz().getType());

    rewriter.replaceOp(op, {data, indices, nnz});
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public method for populating conversion rules.
//===----------------------------------------------------------------------===//

/// Populates the given patterns list with conversion rules required for
/// the sparsification of linear algebra operations.
void mlir::populateSparseTensorCodegenPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    bool enableBufferInitialization) {
  patterns.add<SparsePackOpConverter, SparseUnpackOpConverter,
               SparseReturnConverter, SparseCallConverter, SparseDimOpConverter,
               SparseCastConverter, SparseTensorDeallocConverter,
               SparseExtractSliceCoverter, SparseTensorLoadConverter,
               SparseExpandConverter, SparseCompressConverter,
               SparseInsertConverter, SparseToPointersConverter,
               SparseToIndicesConverter, SparseToIndicesBufferConverter,
               SparseToValuesConverter, SparseConvertConverter,
               SparseNumberOfEntriesConverter>(typeConverter,
                                               patterns.getContext());
  patterns.add<SparseTensorAllocConverter>(typeConverter, patterns.getContext(),
                                           enableBufferInitialization);
}
