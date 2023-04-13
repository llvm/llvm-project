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

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
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

/// Generates a load with proper `index` typing.
static Value genLoad(OpBuilder &builder, Location loc, Value mem, Value idx) {
  idx = genCast(builder, loc, idx, builder.getIndexType());
  return builder.create<memref::LoadOp>(loc, mem, idx);
}

/// Generates a store with proper `index` typing and proper value.
static void genStore(OpBuilder &builder, Location loc, Value val, Value mem,
                     Value idx) {
  idx = genCast(builder, loc, idx, builder.getIndexType());
  val = genCast(builder, loc, val,
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
  return desc.getLvlSize(builder, loc, lvl);
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
      genCast(builder, loc, value, etp), repeat);

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
      // Append linear x positions, initialized to zero. Since each compressed
      // dimension initially already has a single zero entry, this maintains
      // the desired "linear + 1" length property at all times.
      Value posZero = constantZero(builder, loc, stt.getPosType());
      createPushback(builder, loc, desc, SparseTensorFieldKind::PosMemRef, l,
                     posZero, linear);
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
  Value posHeuristic, crdHeuristic, valHeuristic;
  if (stt.isAllDense()) {
    valHeuristic = dimSizes[0];
    for (const Value sz : ArrayRef<Value>{dimSizes}.drop_front())
      valHeuristic = builder.create<arith::MulIOp>(loc, valHeuristic, sz);
  } else if (sizeHint) {
    if (getCOOStart(stt.getEncoding()) == 0) {
      posHeuristic = constantIndex(builder, loc, 2);
      crdHeuristic = builder.create<arith::MulIOp>(
          loc, constantIndex(builder, loc, dimRank), sizeHint); // AOS
    } else if (dimRank == 2 && stt.isDenseLvl(0) && stt.isCompressedLvl(1)) {
      posHeuristic = builder.create<arith::AddIOp>(
          loc, sizeHint, constantIndex(builder, loc, 1));
      crdHeuristic = sizeHint;
    } else {
      posHeuristic = crdHeuristic = constantIndex(builder, loc, 16);
    }
    valHeuristic = sizeHint;
  } else {
    posHeuristic = crdHeuristic = valHeuristic =
        constantIndex(builder, loc, 16);
  }

  foreachFieldAndTypeInSparseTensor(
      stt,
      [&builder, &fields, stt, loc, posHeuristic, crdHeuristic, valHeuristic,
       enableInit](Type fType, FieldIndex fIdx, SparseTensorFieldKind fKind,
                   Level /*lvl*/, DimLevelType /*dlt*/) -> bool {
        assert(fields.size() == fIdx);
        Value field;
        switch (fKind) {
        case SparseTensorFieldKind::StorageSpec:
          field = SparseTensorSpecifier::getInitValue(builder, loc, stt);
          break;
        case SparseTensorFieldKind::PosMemRef:
        case SparseTensorFieldKind::CrdMemRef:
        case SparseTensorFieldKind::ValMemRef:
          field = createAllocation(
              builder, loc, fType.cast<MemRefType>(),
              (fKind == SparseTensorFieldKind::PosMemRef)   ? posHeuristic
              : (fKind == SparseTensorFieldKind::CrdMemRef) ? crdHeuristic
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
  // to all zeros, sets the dimSizes to known values and gives all position
  // fields an initial zero entry, so that it is easier to maintain the
  // "linear + 1" length property.
  Value posZero = constantZero(builder, loc, stt.getPosType());
  for (Level lvlRank = stt.getLvlRank(), l = 0; l < lvlRank; l++) {
    // Fills dim sizes array.
    // FIXME: `toOrigDim` is deprecated.
    desc.setLvlSize(builder, loc, l, dimSizes[toOrigDim(stt, l)]);
    // Pushes a leading zero to positions memref.
    if (stt.isCompressedLvl(l))
      createPushback(builder, loc, desc, SparseTensorFieldKind::PosMemRef, l,
                     posZero);
  }
  allocSchemeForRank(builder, loc, desc, /*rank=*/0);
}

/// Helper method that generates block specific to compressed case:
///
///  // given: parentPos = posCursor[lvl-1]
///  pstart = desc.positions[lvl][parentPos]
///  pstop = desc.positions[lvl][parentPos+1]
///  plast = pstop - 1
///  msz = desc.coordinates[lvl].size()
///  if (pstart < pstop) {
///    isPresent = (desc.coordinates[lvl][plast] == lvlCoords[lvl])
///  } else { // first insertion
///    isPresent = false
///    desc.positions[lvl][parentPos] = msz
///  }
///  if (isPresent) { // coordinate is already present
///    pnext = plast
///  } else {
///    desc.coordinates[lvl].push_back(lvlCoords[lvl])
///    desc.positions[lvl][parentPos+1] = msz+1
///    pnext = msz
///    <prepare level lvl+1>
///  }
///  posCursor[lvl] = pnext
static Value genCompressed(OpBuilder &builder, Location loc,
                           MutSparseTensorDescriptor desc, ValueRange lvlCoords,
                           Value /*unused*/, Value parentPos, Level lvl) {
  const SparseTensorType stt(desc.getRankedTensorType());
  const Level lvlRank = stt.getLvlRank();
  assert(lvl < lvlRank && "Level is out of bounds");
  assert(lvlCoords.size() == static_cast<size_t>(lvlRank) &&
         "Level-rank mismatch");
  SmallVector<Type> types;
  Type indexType = builder.getIndexType();
  Type boolType = builder.getIntegerType(1);
  unsigned crdFidx;
  unsigned crdStride;
  std::tie(crdFidx, crdStride) = desc.getCrdMemRefIndexAndStride(lvl);
  const Value one = constantIndex(builder, loc, 1);
  const Value pp1 = builder.create<arith::AddIOp>(loc, parentPos, one);
  const Value positionsAtLvl = desc.getPosMemRef(lvl);
  const Value pstart = genLoad(builder, loc, positionsAtLvl, parentPos);
  const Value pstop = genLoad(builder, loc, positionsAtLvl, pp1);
  const Value crdMsz = desc.getCrdMemSize(builder, loc, lvl);
  const Value crdStrideC =
      crdStride > 1 ? constantIndex(builder, loc, crdStride) : Value();
  const Value msz =
      crdStrideC ? builder.create<arith::DivUIOp>(loc, crdMsz, crdStrideC)
                 : crdMsz;
  const Value plast = builder.create<arith::SubIOp>(
      loc, genCast(builder, loc, pstop, indexType), one);
  // Conditional expression.
  Value lt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult,
                                           pstart, pstop);
  types.push_back(boolType);
  scf::IfOp ifOp1 = builder.create<scf::IfOp>(loc, types, lt, /*else*/ true);
  types.pop_back();
  builder.setInsertionPointToStart(&ifOp1.getThenRegion().front());
  Value crd =
      genLoad(builder, loc, desc.getMemRefField(crdFidx),
              crdStrideC ? builder.create<arith::MulIOp>(loc, plast, crdStrideC)
                         : plast);
  Value eq = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, genCast(builder, loc, crd, indexType),
      lvlCoords[lvl]);
  builder.create<scf::YieldOp>(loc, eq);
  builder.setInsertionPointToStart(&ifOp1.getElseRegion().front());
  if (lvl > 0)
    genStore(builder, loc, msz, positionsAtLvl, parentPos);
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
  // If present (fields unaffected, update pnext to plast).
  builder.setInsertionPointToStart(&ifOp2.getThenRegion().front());

  // FIXME: This does not looks like a clean way, but probably the most
  // efficient way.
  desc.getFields().push_back(plast);
  builder.create<scf::YieldOp>(loc, desc.getFields());
  desc.getFields().pop_back();

  // If !present (changes fields, update pnext).
  builder.setInsertionPointToStart(&ifOp2.getElseRegion().front());
  Value mszp1 = builder.create<arith::AddIOp>(loc, msz, one);
  genStore(builder, loc, mszp1, positionsAtLvl, pp1);
  createPushback(builder, loc, desc, SparseTensorFieldKind::CrdMemRef, lvl,
                 lvlCoords[lvl]);
  // Prepare the next level "as needed".
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

  // Extract fields and coordinates from args.
  SmallVector<Value> fields = llvm::to_vector(args.drop_back(lvlRank + 1));
  MutSparseTensorDescriptor desc(rtp, fields);
  const SmallVector<Value> coords =
      llvm::to_vector(args.take_back(lvlRank + 1).drop_back());
  Value value = args.back();
  Value parentPos = constantZero(builder, loc, builder.getIndexType());
  // Generate code for every level.
  for (Level l = 0; l < lvlRank; l++) {
    const auto dlt = stt.getLvlType(l);
    if (isCompressedDLT(dlt)) {
      // Create:
      //   if (!present) {
      //     coordinates[l].push_back(coords[l])
      //     <update positions and prepare level l + 1>
      //   }
      //   positions[l] = coordinates.size() - 1
      //   <insert @ positions[l] at next level l + 1>
      parentPos =
          genCompressed(builder, loc, desc, coords, value, parentPos, l);
    } else if (isSingletonDLT(dlt)) {
      // Create:
      //   coordinates[l].push_back(coords[l])
      //   positions[l] = positions[l-1]
      //   <insert @ positions[l] at next level l + 1>
      createPushback(builder, loc, desc, SparseTensorFieldKind::CrdMemRef, l,
                     coords[l]);
    } else {
      assert(isDenseDLT(dlt));
      // Construct the new position as:
      //   positions[l] = size * positions[l-1] + coords[l]
      //   <insert @ positions[l] at next level l + 1>
      Value size = sizeFromTensorAtLvl(builder, loc, desc, l);
      Value mult = builder.create<arith::MulIOp>(loc, size, parentPos);
      parentPos = builder.create<arith::AddIOp>(loc, mult, coords[l]);
    }
  }
  // Reached the actual value append/insert.
  if (!stt.isDenseLvl(lvlRank - 1))
    createPushback(builder, loc, desc, SparseTensorFieldKind::ValMemRef,
                   std::nullopt, value);
  else
    genStore(builder, loc, value, desc.getValMemRef(), parentPos);
  builder.create<func::ReturnOp>(loc, fields);
}

/// Generates a call to a function to perform an insertion operation. If the
/// function doesn't exist yet, call `createFunc` to generate the function.
static void genInsertionCallHelper(OpBuilder &builder,
                                   MutSparseTensorDescriptor desc,
                                   SmallVectorImpl<Value> &lcvs, Value value,
                                   func::FuncOp insertPoint,
                                   StringRef namePrefix,
                                   FuncGeneratorType createFunc) {
  // The mangled name of the function has this format:
  //   <namePrefix>_<DLT>_<shape>_<ordering>_<eltType>_<crdWidth>_<posWidth>
  const SparseTensorType stt(desc.getRankedTensorType());
  SmallString<32> nameBuffer;
  llvm::raw_svector_ostream nameOstream(nameBuffer);
  nameOstream << namePrefix;
  const Level lvlRank = stt.getLvlRank();
  assert(lcvs.size() == static_cast<size_t>(lvlRank));
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
  nameOstream << stt.getCrdWidth() << "_" << stt.getPosWidth();

  // Look up the function.
  ModuleOp module = insertPoint->getParentOfType<ModuleOp>();
  MLIRContext *context = module.getContext();
  auto result = SymbolRefAttr::get(context, nameOstream.str());
  auto func = module.lookupSymbol<func::FuncOp>(result.getAttr());

  // Construct operands: fields, coords, and value.
  SmallVector<Value> operands = llvm::to_vector(desc.getFields());
  operands.append(lcvs);
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
      // Compressed dimensions need a position cleanup for all entries
      // that were not visited during the insertion pass.
      //
      // TODO: avoid cleanup and keep compressed scheme consistent at all
      // times?
      //
      if (l > 0) {
        Type posType = stt.getPosType();
        Value posMemRef = desc.getPosMemRef(l);
        Value hi = desc.getPosMemSize(builder, loc, l);
        Value zero = constantIndex(builder, loc, 0);
        Value one = constantIndex(builder, loc, 1);
        // Vector of only one, but needed by createFor's prototype.
        SmallVector<Value, 1> inits{genLoad(builder, loc, posMemRef, zero)};
        scf::ForOp loop = createFor(builder, loc, hi, inits, one);
        Value i = loop.getInductionVar();
        Value oldv = loop.getRegionIterArg(0);
        Value newv = genLoad(builder, loc, posMemRef, i);
        Value posZero = constantZero(builder, loc, posType);
        Value cond = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, newv, posZero);
        scf::IfOp ifOp = builder.create<scf::IfOp>(loc, TypeRange(posType),
                                                   cond, /*else*/ true);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        genStore(builder, loc, oldv, posMemRef, i);
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
  // Reallocates if target length is greater than the actual buffer len.
  Value reallocP = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
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
    std::optional<int64_t> dim = op.getConstantIndex();
    if (!dim || !getSparseTensorEncoding(adaptor.getSource().getType()))
      return failure();

    auto desc = getDescriptorFromTensorTuple(adaptor.getSource());
    auto sz = sizeFromTensorAtDim(rewriter, op.getLoc(), desc, *dim);

    rewriter.replaceOp(op, sz);
    return success();
  }
};

template <typename Op, StorageSpecifierKind kind>
class SparseSliceGetterOpConverter : public OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Simply lowers to specifer.get <field> operation.
    auto desc = getDescriptorFromTensorTuple(adaptor.getSlice());
    auto v = desc.getSpecifierField(rewriter, op.getLoc(), kind,
                                    op.getDim().getZExtValue());

    rewriter.replaceOp(op, v);
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

    // Construct allocation for each field.
    const Location loc = op.getLoc();
    if (op.getCopy()) {
      auto desc = getDescriptorFromTensorTuple(adaptor.getCopy());
      SmallVector<Value> fields;
      fields.reserve(desc.getNumFields());
      // Memcpy on memref fields.
      for (auto field : desc.getMemRefFields()) {
        auto memrefTp = field.getType().cast<MemRefType>();
        auto size = rewriter.create<memref::DimOp>(loc, field, 0);
        auto copied =
            rewriter.create<memref::AllocOp>(loc, memrefTp, ValueRange{size});
        rewriter.create<memref::CopyOp>(loc, field, copied);
        fields.push_back(copied);
      }
      // Reuses specifier.
      fields.push_back(desc.getSpecifier());
      assert(fields.size() == desc.getNumFields());
      rewriter.replaceOp(op, genTuple(rewriter, loc, resType, fields));
      return success();
    }

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
  SparseTensorDeallocConverter(TypeConverter &typeConverter,
                               MLIRContext *context, bool createDeallocs)
      : OpConversionPattern(typeConverter, context),
        createDeallocs(createDeallocs) {}

  LogicalResult
  matchAndRewrite(bufferization::DeallocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto enc = getSparseTensorEncoding(op.getTensor().getType());
    if (!enc)
      return failure();

    // If user requests not to deallocate sparse tensors, simply erase the
    // operation.
    if (createDeallocs) {
      // Replace the sparse tensor deallocation with field deallocations.
      Location loc = op.getLoc();
      auto desc = getDescriptorFromTensorTuple(adaptor.getTensor());
      for (auto input : desc.getMemRefFields())
        // Deallocate every buffer used to store the sparse tensor handler.
        rewriter.create<memref::DeallocOp>(loc, input);
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  bool createDeallocs;
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
    // level size, translated back to original dimension). Note that we
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
    // Replace expansion op with these buffers and initial coordinate.
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
    // Prepare level-coords.
    SmallVector<Value> lcvs(adaptor.getLvlCoords());
    // If the innermost level is ordered, we need to sort the coordinates
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
    //      crd = added[i];
    //      value = values[crd];
    //      insert({lvlCoords, crd}, value);
    //      new_memrefs = insert(in_memrefs, {lvlCoords, crd}, value);
    //      values[crd] = 0;
    //      filled[crd] = false;
    //      yield new_memrefs
    //    }
    scf::ForOp loop = createFor(rewriter, loc, count, desc.getFields());
    Value i = loop.getInductionVar();
    Value crd = genLoad(rewriter, loc, added, i);
    Value value = genLoad(rewriter, loc, values, crd);
    lcvs.push_back(crd);
    // TODO: faster for subsequent insertions?
    auto insertPoint = op->template getParentOfType<func::FuncOp>();
    genInsertionCallHelper(rewriter, desc, lcvs, value, insertPoint,
                           kInsertFuncNamePrefix, genInsertBody);
    genStore(rewriter, loc, constantZero(rewriter, loc, eltType), values, crd);
    genStore(rewriter, loc, constantI1(rewriter, loc, false), filled, crd);
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
    SmallVector<Value> lcvs(adaptor.getLvlCoords());
    // Generate insertion.
    Value value = adaptor.getValue();
    auto insertPoint = op->template getParentOfType<func::FuncOp>();
    genInsertionCallHelper(rewriter, desc, lcvs, value, insertPoint,
                           kInsertFuncNamePrefix, genInsertBody);

    // Replace operation with resulting memrefs.
    rewriter.replaceOp(op, genTuple(rewriter, op.getLoc(), desc));
    return success();
  }
};

/// Sparse codegen rule for position accesses.
class SparseToPositionsConverter : public OpConversionPattern<ToPositionsOp> {
public:
  using OpAdaptor = typename ToPositionsOp::Adaptor;
  using OpConversionPattern<ToPositionsOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToPositionsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the requested position access with corresponding field.
    // The cast_op is inserted by type converter to intermix 1:N type
    // conversion.
    auto desc = getDescriptorFromTensorTuple(adaptor.getTensor());
    rewriter.replaceOp(op, desc.getPosMemRef(op.getLevel()));
    return success();
  }
};

/// Sparse codegen rule for accessing the coordinates arrays.
class SparseToCoordinatesConverter
    : public OpConversionPattern<ToCoordinatesOp> {
public:
  using OpAdaptor = typename ToCoordinatesOp::Adaptor;
  using OpConversionPattern<ToCoordinatesOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToCoordinatesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the requested coordinates access with corresponding field.
    // The cast_op is inserted by type converter to intermix 1:N type
    // conversion.
    Location loc = op.getLoc();
    auto desc = getDescriptorFromTensorTuple(adaptor.getTensor());
    Value field = desc.getCrdMemRefOrView(rewriter, loc, op.getLevel());

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

/// Sparse codegen rule for accessing the linear coordinates buffer.
class SparseToCoordinatesBufferConverter
    : public OpConversionPattern<ToCoordinatesBufferOp> {
public:
  using OpAdaptor = typename ToCoordinatesBufferOp::Adaptor;
  using OpConversionPattern<ToCoordinatesBufferOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToCoordinatesBufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the requested coordinates access with corresponding field.
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
    // Replace the requested values access with corresponding field.
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
    // The output tensor can not be a slice and those cases should have been
    // rejected by ConvertOp::verify() already.
    assert(!encDst.isSlice() && "Cannot convert to a sparse tensor slices.");
    // Different encoding (except for different bitwidth) should be handled by
    // rewriting.
    // We need further rewrites if the input tensor is a slice too.
    if (encDst.withoutBitWidths() != encSrc.withoutBitWidths() ||
        encSrc.isSlice()) {
      return failure();
    }

    Type retElemTp = op.getResult().getType().getElementType();
    Type srcElemTp = op.getSource().getType().getElementType();
    // Fold the trivial cases.
    if (retElemTp == srcElemTp && encDst == encSrc) {
      rewriter.replaceOp(op, adaptor.getSource());
      return success();
    }
    //
    // Do element-wise type conversion without using InsertOp.
    //
    // for each memref in srcTensor:
    //   dst = memref.alloc
    //   if srcMemRefType != dstMemRefType:
    //     for every dst[i] = cast(src[i])
    //   else:
    //     dst = memref.copy(src)
    Location loc = op.getLoc();
    auto srcDesc = getDescriptorFromTensorTuple(adaptor.getSource());
    SmallVector<Value> fields;
    foreachFieldAndTypeInSparseTensor(
        SparseTensorType(op.getResult().getType().cast<RankedTensorType>()),
        [&rewriter, &fields, srcDesc,
         loc](Type fTp, FieldIndex fIdx, SparseTensorFieldKind fKind, Level lvl,
              DimLevelType /*dlt*/) -> bool {
          // Simply reuses the storage specifier as it is an SSA value.
          if (fKind == SparseTensorFieldKind::StorageSpec) {
            fields.push_back(srcDesc.getSpecifier());
          } else {
            // Allocates new memrefs
            Value srcMem = srcDesc.getMemRefField(fIdx);
            // TODO: We can instead use the actual memSize in specifier, that
            // would require a subViewOp to avoid overflow when copying
            // values.
            Value sz = linalg::createOrFoldDimOp(rewriter, loc, srcMem, 0);
            auto dstMem = rewriter.create<memref::AllocOp>(
                loc, fTp.cast<MemRefType>(), sz);
            if (fTp != srcMem.getType()) {
              // Converts elements type.
              scf::buildLoopNest(
                  rewriter, loc, constantIndex(rewriter, loc, 0), sz,
                  constantIndex(rewriter, loc, 1),
                  [srcMem, &dstMem](OpBuilder &builder, Location loc,
                                    ValueRange ivs) {
                    Value v = builder.create<memref::LoadOp>(loc, srcMem, ivs);
                    Value casted = genCast(builder, loc, v,
                                           dstMem.getType().getElementType());
                    builder.create<memref::StoreOp>(loc, casted, dstMem, ivs);
                  });
            } else {
              // TODO: We can even reuse the same memref for the new tensor,
              // but that requires a `ref-counting` based memory management
              // for shared memrefs between multiple sparse tensors.
              rewriter.create<memref::CopyOp>(loc, srcMem, dstMem);
            }
            fields.push_back(dstMem);
          }
          return true;
        });

    rewriter.replaceOp(
        op, genTuple(rewriter, loc, op.getResult().getType(), fields));
    return success();
  }
};

class SparseExtractSliceConverter
    : public OpConversionPattern<tensor::ExtractSliceOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();
    auto srcEnc = getSparseTensorEncoding(op.getSourceType());
    auto dstEnc = getSparseTensorEncoding(op.getResult().getType());
    if (!srcEnc && !dstEnc)
      return failure();

    // TODO: We should check these in ExtractSliceOp::verify.
    assert(srcEnc && dstEnc && dstEnc.isSlice());
    assert(srcEnc.getDimLevelType() == dstEnc.getDimLevelType());
    assert(srcEnc.getDimOrdering() == dstEnc.getDimOrdering());
    assert(srcEnc.getHigherOrdering() == dstEnc.getHigherOrdering());
    assert(srcEnc.getPosWidth() == dstEnc.getPosWidth());
    assert(srcEnc.getCrdWidth() == dstEnc.getCrdWidth());

    SmallVector<Value> fields;
    auto desc = getMutDescriptorFromTensorTuple(adaptor.getSource(), fields);

    auto newSpec = rewriter.create<StorageSpecifierInitOp>(
        loc, StorageSpecifierType::get(ctx, dstEnc), desc.getSpecifier());
    desc.setSpecifier(newSpec);

    // Fills in slice information.
    for (auto [idx, offset, size, stride] : llvm::enumerate(
             op.getMixedOffsets(), op.getMixedSizes(), op.getMixedStrides())) {
      Dimension dim = idx;

      Value offsetV = getValueOrCreateConstantIndexOp(rewriter, loc, offset);
      Value sizeV = getValueOrCreateConstantIndexOp(rewriter, loc, size);
      Value strideV = getValueOrCreateConstantIndexOp(rewriter, loc, stride);
      // TODO: We could probably only set dynamic value here. But it would
      // requires us to fill the hole when casting a static slice to dynamic
      // slice.
      desc.setSpecifierField(rewriter, loc, StorageSpecifierKind::DimOffset,
                             dim, offsetV);

      // FIXME: we need to distinguish level sizes and dimension size for slices
      // here. Maybe we should store slice level sizes in a different array
      // instead of reusing it.
      assert(srcEnc.hasIdDimOrdering());
      desc.setSpecifierField(rewriter, loc, StorageSpecifierKind::LvlSize, dim,
                             sizeV);
      desc.setSpecifierField(rewriter, loc, StorageSpecifierKind::DimStride,
                             dim, strideV);
    }

    // NOTE: we can not generate tuples directly from descriptor here, as the
    // descriptor is holding the original type, yet we want the slice type
    // here (they shared every memref but with an updated specifier).
    rewriter.replaceOp(op, genTuple(rewriter, loc, op.getResult().getType(),
                                    desc.getFields()));
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

    const auto stt = getSparseTensorType(op.getResult());
    assert(isUniqueCOOType(stt));

    SmallVector<Value> fields;
    Location loc = op.getLoc();

    foreachFieldAndTypeInSparseTensor(
        stt,
        [&rewriter, &fields, &op, stt,
         loc](Type fType, FieldIndex fIdx, SparseTensorFieldKind fKind,
              Level /*lvl*/, DimLevelType /*dlt*/) -> bool {
          assert(fields.size() == fIdx);
          Value field;
          switch (fKind) {
          case SparseTensorFieldKind::StorageSpec:
            field = SparseTensorSpecifier::getInitValue(rewriter, loc, stt);
            break;
          case SparseTensorFieldKind::PosMemRef: {
            // TACO-style COO starts with a PosBuffer
            // By creating a constant value for it, we avoid the complexity of
            // memory management.
            const auto posTp = stt.getPosType();
            auto tensorType = RankedTensorType::get({2}, posTp);
            auto memrefType = MemRefType::get(tensorType.getShape(),
                                              tensorType.getElementType());
            auto cstPtr = rewriter.create<arith::ConstantOp>(
                loc, tensorType,
                DenseElementsAttr::get(
                    tensorType,
                    ArrayRef<Attribute>{
                        IntegerAttr::get(posTp, 0),
                        IntegerAttr::get(
                            posTp, op.getValues().getType().getShape()[0])}));
            field = rewriter.create<bufferization::ToMemrefOp>(loc, memrefType,
                                                               cstPtr);
            break;
          }
          case SparseTensorFieldKind::CrdMemRef: {
            auto tensorType = op.getCoordinates().getType();
            auto memrefType = MemRefType::get(tensorType.getShape(),
                                              tensorType.getElementType());
            auto crdMemRef = rewriter.create<bufferization::ToMemrefOp>(
                op->getLoc(), memrefType, op.getCoordinates());
            ReassociationIndices reassociation;
            for (int i = 0, e = tensorType.getRank(); i < e; i++)
              reassociation.push_back(i);

            // Flattened the indices buffer to rank 1.
            field = rewriter.create<memref::CollapseShapeOp>(
                loc, crdMemRef, ArrayRef<ReassociationIndices>(reassociation));
            break;
          }
          case SparseTensorFieldKind::ValMemRef: {
            auto tensorType = op.getValues().getType();
            auto memrefType = MemRefType::get(tensorType.getShape(),
                                              tensorType.getElementType());
            field = rewriter.create<bufferization::ToMemrefOp>(
                op->getLoc(), memrefType, op.getValues());
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

    MutSparseTensorDescriptor desc(stt, fields);
    auto noe = linalg::createOrFoldDimOp(rewriter, loc, op.getValues(), 0);
    for (Level lvl = 0, lvlRank = stt.getLvlRank(); lvl < lvlRank; lvl++) {
      // FIXME: dim/lvl confusion!
      const auto sh = stt.getDimShape()[lvl];
      assert(!ShapedType::isDynamic(sh));
      desc.setLvlSize(rewriter, loc, lvl, constantIndex(rewriter, loc, sh));
      if (lvl == 0)
        desc.setPosMemSize(rewriter, loc, lvl, constantIndex(rewriter, loc, 2));

      desc.setCrdMemSize(rewriter, loc, lvl, noe);
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
    const auto srcTp = getSparseTensorType(op.getTensor());
    const Level lvlRank = srcTp.getLvlRank();

    assert(isUniqueCOOType(srcTp) && desc.getFields().size() == 4);

    Value flatBuf = lvlRank == 1 ? desc.getCrdMemRefOrView(rewriter, loc, 0)
                                 : desc.getAOSMemRef();
    Value valuesBuf = desc.getValMemRef();

    // If frontend requests a static buffer, we reallocate the
    // values/coordinates to ensure that we meet their need.
    const auto valuesTp = getRankedTensorType(op.getValues());
    if (valuesTp.hasStaticShape()) {
      valuesBuf =
          reallocOrSubView(rewriter, loc, valuesTp.getShape()[0], valuesBuf);
    }

    const auto coordinatesTp = getRankedTensorType(op.getCoordinates());
    if (coordinatesTp.hasStaticShape()) {
      auto len = coordinatesTp.getShape()[0] * coordinatesTp.getShape()[1];
      flatBuf = reallocOrSubView(rewriter, loc, len, flatBuf);
    }

    Value coordinatesBuf = rewriter.create<memref::ExpandShapeOp>(
        loc,
        MemRefType::get(coordinatesTp.getShape(),
                        coordinatesTp.getElementType()),
        flatBuf, ArrayRef{ReassociationIndices{0, 1}});

    // Converts MemRefs back to Tensors.
    Value values = rewriter.create<bufferization::ToTensorOp>(loc, valuesBuf);
    Value coordinates =
        rewriter.create<bufferization::ToTensorOp>(loc, coordinatesBuf);
    Value nse = genCast(rewriter, loc, desc.getValMemSize(rewriter, loc),
                        op.getNse().getType());

    rewriter.replaceOp(op, {values, coordinates, nse});
    return success();
  }
};

struct SparseNewOpConverter : public OpConversionPattern<NewOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(NewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    const auto dstTp = getSparseTensorType(op.getResult());
    // Creating COO with NewOp is handled by direct IR codegen. All other cases
    // are handled by rewriting.
    if (!dstTp.hasEncoding() || getCOOStart(dstTp.getEncoding()) != 0)
      return failure();

    // Implement the NewOp(filename) as follows:
    //   %reader = @getSparseTensorReader(%filename)
    //   %nse = @getSparseTensorNSE(%reader)
    //   %coo = bufferization.alloc_tensor an ordered COO with
    //          dst dim ordering, size_hint = %nse
    //   %coordinates = sparse_tensor.coordinates_buffer(%coo)
    //   %values = sparse_tensor.values(%coo)
    //   %isSorted = @sparseTensorReaderReadToBuffers(%coordinates, %values)
    //   if (! %isSorted) sparse_tensor.sort_coo(%nse, %coordinates, %values)
    //   update storage specifier
    //   @delSparseTensorReader(%reader)

    // Create a sparse tensor reader.
    const Value fileName = op.getSource();
    const Type opaqueTp = getOpaquePointerType(rewriter);
    // FIXME: use `createCheckedSparseTensorReader` instead, because
    // `createSparseTensorReader` is unsafe.
    Value reader = createFuncCall(rewriter, loc, "createSparseTensorReader",
                                  {opaqueTp}, {fileName}, EmitCInterface::Off)
                       .getResult(0);

    const Type indexTp = rewriter.getIndexType();
    const Dimension dimRank = dstTp.getDimRank();
    const Level lvlRank = dstTp.getLvlRank();

    // If the result tensor has dynamic dimensions, get the dynamic sizes from
    // the sparse tensor reader.
    SmallVector<Value> dynSizes;
    if (dstTp.hasDynamicDimShape()) {
      // FIXME: call `getSparseTensorReaderDimSizes` instead, because
      // `copySparseTensorReaderDimSizes` copies the memref over,
      // instead of just accessing the reader's memory directly.
      Value dimSizes = genAlloca(rewriter, loc, dimRank, indexTp);
      createFuncCall(rewriter, loc, "copySparseTensorReaderDimSizes", {},
                     {reader, dimSizes}, EmitCInterface::On);
      for (const auto &d : llvm::enumerate(dstTp.getDimShape()))
        if (ShapedType::isDynamic(d.value()))
          dynSizes.push_back(rewriter.create<memref::LoadOp>(
              loc, dimSizes, constantIndex(rewriter, loc, d.index())));
    }

    Value nse = createFuncCall(rewriter, loc, "getSparseTensorReaderNSE",
                               {indexTp}, {reader}, EmitCInterface::Off)
                    .getResult(0);
    // Construct allocation for each field.
    SmallVector<Value> fields;
    createAllocFields(rewriter, loc, dstTp, dynSizes, /*enableInit=*/false,
                      fields, nse);
    MutSparseTensorDescriptor desc(dstTp, fields);

    // Construct the `dim2lvl` buffer for handing off to the runtime library.
    // FIXME: This code is (mostly) copied from the SparseTensorConversion.cpp
    // handling of `NewOp`, and only handles permutations.  Fixing this
    // requires waiting for wrengr to finish redoing the CL that handles
    // all dim<->lvl stuff more robustly.
    SmallVector<Value> dim2lvlValues(dimRank);
    if (!dstTp.isIdentity()) {
      const auto dimOrder = dstTp.getDimToLvlMap();
      assert(dimOrder.isPermutation() && "Got non-permutation");
      for (Level l = 0; l < lvlRank; l++) {
        const Dimension d = dimOrder.getDimPosition(l);
        dim2lvlValues[d] = constantIndex(rewriter, loc, l);
      }
    } else {
      // The `SparseTensorType` ctor already ensures `dimRank == lvlRank`
      // when `isIdentity`; so no need to re-assert it here.
      for (Dimension d = 0; d < dimRank; d++)
        dim2lvlValues[d] = constantIndex(rewriter, loc, d);
    }
    Value dim2lvl = allocaBuffer(rewriter, loc, dim2lvlValues);

    // Read the COO tensor data.
    Value xs = desc.getAOSMemRef();
    Value ys = desc.getValMemRef();

    const Type boolTp = rewriter.getIntegerType(1);
    const Type elemTp = dstTp.getElementType();
    const Type crdTp = dstTp.getCrdType();
    // FIXME: This function name is weird; should rename to
    // "sparseTensorReaderReadToBuffers".
    SmallString<32> readToBuffersFuncName{"getSparseTensorReaderRead",
                                          overheadTypeFunctionSuffix(crdTp),
                                          primaryTypeFunctionSuffix(elemTp)};
    Value isSorted =
        createFuncCall(rewriter, loc, readToBuffersFuncName, {boolTp},
                       {reader, dim2lvl, xs, ys}, EmitCInterface::On)
            .getResult(0);

    // If the destination tensor is a sorted COO, we need to sort the COO tensor
    // data if the input elements aren't sorted yet.
    if (dstTp.isOrderedLvl(lvlRank - 1)) {
      Value kFalse = constantI1(rewriter, loc, false);
      Value notSorted = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, isSorted, kFalse);
      scf::IfOp ifOp =
          rewriter.create<scf::IfOp>(loc, notSorted, /*else*/ false);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      rewriter.create<SortCooOp>(
          loc, nse, xs, ValueRange{ys}, rewriter.getIndexAttr(lvlRank),
          rewriter.getIndexAttr(0), SparseTensorSortKind::HybridQuickSort);
      rewriter.setInsertionPointAfter(ifOp);
    }

    // Set PosMemRef0[1] = nse.
    const Value c1 = constantIndex(rewriter, loc, 1);
    const Value posMemref0 = desc.getPosMemRef(0);
    const Type posTp = dstTp.getPosType();
    const Value posNse = genCast(rewriter, loc, nse, posTp);
    rewriter.create<memref::StoreOp>(loc, posNse, posMemref0, c1);

    // Update storage specifier.
    Value coordinatesSize = rewriter.create<arith::MulIOp>(
        loc, nse, constantIndex(rewriter, loc, lvlRank));
    desc.setSpecifierField(rewriter, loc, StorageSpecifierKind::CrdMemSize, 0,
                           coordinatesSize);
    desc.setSpecifierField(rewriter, loc, StorageSpecifierKind::ValMemSize,
                           std::nullopt, nse);

    // Release the sparse tensor reader.
    createFuncCall(rewriter, loc, "delSparseTensorReader", {}, {reader},
                   EmitCInterface::Off);

    // Replace operation with resulting memrefs.
    rewriter.replaceOp(op, genTuple(rewriter, loc, dstTp, fields));
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
    bool createSparseDeallocs, bool enableBufferInitialization) {
  patterns.add<SparsePackOpConverter, SparseUnpackOpConverter,
               SparseReturnConverter, SparseCallConverter, SparseDimOpConverter,
               SparseCastConverter, SparseExtractSliceConverter,
               SparseTensorLoadConverter, SparseExpandConverter,
               SparseCompressConverter, SparseInsertConverter,
               SparseSliceGetterOpConverter<ToSliceOffsetOp,
                                            StorageSpecifierKind::DimOffset>,
               SparseSliceGetterOpConverter<ToSliceStrideOp,
                                            StorageSpecifierKind::DimStride>,
               SparseToPositionsConverter, SparseToCoordinatesConverter,
               SparseToCoordinatesBufferConverter, SparseToValuesConverter,
               SparseConvertConverter, SparseNewOpConverter,
               SparseNumberOfEntriesConverter>(typeConverter,
                                               patterns.getContext());
  patterns.add<SparseTensorDeallocConverter>(
      typeConverter, patterns.getContext(), createSparseDeallocs);
  patterns.add<SparseTensorAllocConverter>(typeConverter, patterns.getContext(),
                                           enableBufferInitialization);
}
