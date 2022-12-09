//===- ConvertFromLLVMIR.cpp - MLIR to LLVM IR conversion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between LLVM IR and the MLIR LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "DebugImporter.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Target/LLVMIR/TypeFromLLVM.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::LLVM;
using mlir::LLVM::detail::DebugImporter;

#include "mlir/Dialect/LLVMIR/LLVMConversionEnumsFromLLVM.inc"

/// Returns true if the LLVM IR intrinsic is convertible to an MLIR LLVM dialect
/// intrinsic, or false if no counterpart exists.
static bool isConvertibleIntrinsic(llvm::Intrinsic::ID id) {
  static const DenseSet<unsigned> convertibleIntrinsics = {
#include "mlir/Dialect/LLVMIR/LLVMConvertibleLLVMIRIntrinsics.inc"
  };
  return convertibleIntrinsics.contains(id);
}

// Utility to print an LLVM value as a string for passing to emitError().
// FIXME: Diagnostic should be able to natively handle types that have
// operator << (raw_ostream&) defined.
static std::string diag(llvm::Value &value) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << value;
  return os.str();
}

/// Creates an attribute containing ABI and preferred alignment numbers parsed
/// a string. The string may be either "abi:preferred" or just "abi". In the
/// latter case, the prefrred alignment is considered equal to ABI alignment.
static DenseIntElementsAttr parseDataLayoutAlignment(MLIRContext &ctx,
                                                     StringRef spec) {
  auto i32 = IntegerType::get(&ctx, 32);

  StringRef abiString, preferredString;
  std::tie(abiString, preferredString) = spec.split(':');
  int abi, preferred;
  if (abiString.getAsInteger(/*Radix=*/10, abi))
    return nullptr;

  if (preferredString.empty())
    preferred = abi;
  else if (preferredString.getAsInteger(/*Radix=*/10, preferred))
    return nullptr;

  return DenseIntElementsAttr::get(VectorType::get({2}, i32), {abi, preferred});
}

/// Returns a supported MLIR floating point type of the given bit width or null
/// if the bit width is not supported.
static FloatType getDLFloatType(MLIRContext &ctx, int32_t bitwidth) {
  switch (bitwidth) {
  case 16:
    return FloatType::getF16(&ctx);
  case 32:
    return FloatType::getF32(&ctx);
  case 64:
    return FloatType::getF64(&ctx);
  case 80:
    return FloatType::getF80(&ctx);
  case 128:
    return FloatType::getF128(&ctx);
  default:
    return nullptr;
  }
}

static ICmpPredicate getICmpPredicate(llvm::CmpInst::Predicate pred) {
  switch (pred) {
  default:
    llvm_unreachable("incorrect comparison predicate");
  case llvm::CmpInst::Predicate::ICMP_EQ:
    return LLVM::ICmpPredicate::eq;
  case llvm::CmpInst::Predicate::ICMP_NE:
    return LLVM::ICmpPredicate::ne;
  case llvm::CmpInst::Predicate::ICMP_SLT:
    return LLVM::ICmpPredicate::slt;
  case llvm::CmpInst::Predicate::ICMP_SLE:
    return LLVM::ICmpPredicate::sle;
  case llvm::CmpInst::Predicate::ICMP_SGT:
    return LLVM::ICmpPredicate::sgt;
  case llvm::CmpInst::Predicate::ICMP_SGE:
    return LLVM::ICmpPredicate::sge;
  case llvm::CmpInst::Predicate::ICMP_ULT:
    return LLVM::ICmpPredicate::ult;
  case llvm::CmpInst::Predicate::ICMP_ULE:
    return LLVM::ICmpPredicate::ule;
  case llvm::CmpInst::Predicate::ICMP_UGT:
    return LLVM::ICmpPredicate::ugt;
  case llvm::CmpInst::Predicate::ICMP_UGE:
    return LLVM::ICmpPredicate::uge;
  }
  llvm_unreachable("incorrect integer comparison predicate");
}

static FCmpPredicate getFCmpPredicate(llvm::CmpInst::Predicate pred) {
  switch (pred) {
  default:
    llvm_unreachable("incorrect comparison predicate");
  case llvm::CmpInst::Predicate::FCMP_FALSE:
    return LLVM::FCmpPredicate::_false;
  case llvm::CmpInst::Predicate::FCMP_TRUE:
    return LLVM::FCmpPredicate::_true;
  case llvm::CmpInst::Predicate::FCMP_OEQ:
    return LLVM::FCmpPredicate::oeq;
  case llvm::CmpInst::Predicate::FCMP_ONE:
    return LLVM::FCmpPredicate::one;
  case llvm::CmpInst::Predicate::FCMP_OLT:
    return LLVM::FCmpPredicate::olt;
  case llvm::CmpInst::Predicate::FCMP_OLE:
    return LLVM::FCmpPredicate::ole;
  case llvm::CmpInst::Predicate::FCMP_OGT:
    return LLVM::FCmpPredicate::ogt;
  case llvm::CmpInst::Predicate::FCMP_OGE:
    return LLVM::FCmpPredicate::oge;
  case llvm::CmpInst::Predicate::FCMP_ORD:
    return LLVM::FCmpPredicate::ord;
  case llvm::CmpInst::Predicate::FCMP_ULT:
    return LLVM::FCmpPredicate::ult;
  case llvm::CmpInst::Predicate::FCMP_ULE:
    return LLVM::FCmpPredicate::ule;
  case llvm::CmpInst::Predicate::FCMP_UGT:
    return LLVM::FCmpPredicate::ugt;
  case llvm::CmpInst::Predicate::FCMP_UGE:
    return LLVM::FCmpPredicate::uge;
  case llvm::CmpInst::Predicate::FCMP_UNO:
    return LLVM::FCmpPredicate::uno;
  case llvm::CmpInst::Predicate::FCMP_UEQ:
    return LLVM::FCmpPredicate::ueq;
  case llvm::CmpInst::Predicate::FCMP_UNE:
    return LLVM::FCmpPredicate::une;
  }
  llvm_unreachable("incorrect floating point comparison predicate");
}

static AtomicOrdering getLLVMAtomicOrdering(llvm::AtomicOrdering ordering) {
  switch (ordering) {
  case llvm::AtomicOrdering::NotAtomic:
    return LLVM::AtomicOrdering::not_atomic;
  case llvm::AtomicOrdering::Unordered:
    return LLVM::AtomicOrdering::unordered;
  case llvm::AtomicOrdering::Monotonic:
    return LLVM::AtomicOrdering::monotonic;
  case llvm::AtomicOrdering::Acquire:
    return LLVM::AtomicOrdering::acquire;
  case llvm::AtomicOrdering::Release:
    return LLVM::AtomicOrdering::release;
  case llvm::AtomicOrdering::AcquireRelease:
    return LLVM::AtomicOrdering::acq_rel;
  case llvm::AtomicOrdering::SequentiallyConsistent:
    return LLVM::AtomicOrdering::seq_cst;
  }
  llvm_unreachable("incorrect atomic ordering");
}

static AtomicBinOp getLLVMAtomicBinOp(llvm::AtomicRMWInst::BinOp binOp) {
  switch (binOp) {
  case llvm::AtomicRMWInst::Xchg:
    return LLVM::AtomicBinOp::xchg;
  case llvm::AtomicRMWInst::Add:
    return LLVM::AtomicBinOp::add;
  case llvm::AtomicRMWInst::Sub:
    return LLVM::AtomicBinOp::sub;
  case llvm::AtomicRMWInst::And:
    return LLVM::AtomicBinOp::_and;
  case llvm::AtomicRMWInst::Nand:
    return LLVM::AtomicBinOp::nand;
  case llvm::AtomicRMWInst::Or:
    return LLVM::AtomicBinOp::_or;
  case llvm::AtomicRMWInst::Xor:
    return LLVM::AtomicBinOp::_xor;
  case llvm::AtomicRMWInst::Max:
    return LLVM::AtomicBinOp::max;
  case llvm::AtomicRMWInst::Min:
    return LLVM::AtomicBinOp::min;
  case llvm::AtomicRMWInst::UMax:
    return LLVM::AtomicBinOp::umax;
  case llvm::AtomicRMWInst::UMin:
    return LLVM::AtomicBinOp::umin;
  case llvm::AtomicRMWInst::FAdd:
    return LLVM::AtomicBinOp::fadd;
  case llvm::AtomicRMWInst::FSub:
    return LLVM::AtomicBinOp::fsub;
  default:
    llvm_unreachable("unsupported atomic binary operation");
  }
}

/// Converts the sync scope identifier of `fenceInst` to the string
/// representation necessary to build the LLVM dialect fence operation.
static StringRef getLLVMSyncScope(llvm::FenceInst *fenceInst) {
  llvm::LLVMContext &llvmContext = fenceInst->getContext();
  SmallVector<StringRef> syncScopeNames;
  llvmContext.getSyncScopeNames(syncScopeNames);
  for (StringRef name : syncScopeNames)
    if (fenceInst->getSyncScopeID() == llvmContext.getOrInsertSyncScopeID(name))
      return name;
  llvm_unreachable("incorrect sync scope identifier");
}

/// Converts an array of unsigned indices to a signed integer position array.
static SmallVector<int64_t> getPositionFromIndices(ArrayRef<unsigned> indices) {
  SmallVector<int64_t> position;
  llvm::append_range(position, indices);
  return position;
}

DataLayoutSpecInterface
mlir::translateDataLayout(const llvm::DataLayout &dataLayout,
                          MLIRContext *context) {
  assert(context && "expected MLIR context");
  std::string layoutstr = dataLayout.getStringRepresentation();

  // Remaining unhandled default layout defaults
  // e (little endian if not set)
  // p[n]:64:64:64 (non zero address spaces have 64-bit properties)
  std::string append =
      "p:64:64:64-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f16:16:16-f64:"
      "64:64-f128:128:128-v64:64:64-v128:128:128-a:0:64";
  if (layoutstr.empty())
    layoutstr = append;
  else
    layoutstr = layoutstr + "-" + append;

  StringRef layout(layoutstr);

  SmallVector<DataLayoutEntryInterface> entries;
  StringSet<> seen;
  while (!layout.empty()) {
    // Split at '-'.
    std::pair<StringRef, StringRef> split = layout.split('-');
    StringRef current;
    std::tie(current, layout) = split;

    // Split at ':'.
    StringRef kind, spec;
    std::tie(kind, spec) = current.split(':');
    if (seen.contains(kind))
      continue;
    seen.insert(kind);

    char symbol = kind.front();
    StringRef parameter = kind.substr(1);

    if (symbol == 'i' || symbol == 'f') {
      unsigned bitwidth;
      if (parameter.getAsInteger(/*Radix=*/10, bitwidth))
        return nullptr;
      DenseIntElementsAttr params = parseDataLayoutAlignment(*context, spec);
      if (!params)
        return nullptr;
      auto entry = DataLayoutEntryAttr::get(
          symbol == 'i' ? static_cast<Type>(IntegerType::get(context, bitwidth))
                        : getDLFloatType(*context, bitwidth),
          params);
      entries.emplace_back(entry);
    } else if (symbol == 'e' || symbol == 'E') {
      auto value = StringAttr::get(
          context, symbol == 'e' ? DLTIDialect::kDataLayoutEndiannessLittle
                                 : DLTIDialect::kDataLayoutEndiannessBig);
      auto entry = DataLayoutEntryAttr::get(
          StringAttr::get(context, DLTIDialect::kDataLayoutEndiannessKey),
          value);
      entries.emplace_back(entry);
    }
  }

  return DataLayoutSpecAttr::get(context, entries);
}

/// Get a topologically sorted list of blocks for the given function.
static SetVector<llvm::BasicBlock *>
getTopologicallySortedBlocks(llvm::Function *func) {
  SetVector<llvm::BasicBlock *> blocks;
  for (llvm::BasicBlock &bb : *func) {
    if (blocks.count(&bb) == 0) {
      llvm::ReversePostOrderTraversal<llvm::BasicBlock *> traversal(&bb);
      blocks.insert(traversal.begin(), traversal.end());
    }
  }
  assert(blocks.size() == func->getBasicBlockList().size() &&
         "some blocks are not sorted");

  return blocks;
}

// Handles importing globals and functions from an LLVM module.
namespace {
class Importer {
public:
  Importer(MLIRContext *context, ModuleOp module)
      : builder(context), context(context), module(module),
        typeTranslator(*context), debugImporter(context) {
    builder.setInsertionPointToStart(module.getBody());
  }

  /// Stores the mapping between an LLVM value and its MLIR counterpart.
  void mapValue(llvm::Value *llvm, Value mlir) { mapValue(llvm) = mlir; }

  /// Provides write-once access to store the MLIR value corresponding to the
  /// given LLVM value.
  Value &mapValue(llvm::Value *value) {
    Value &mlir = valueMapping[value];
    assert(mlir == nullptr &&
           "attempting to map a value that is already mapped");
    return mlir;
  }

  /// Returns the MLIR value mapped to the given LLVM value.
  Value lookupValue(llvm::Value *value) { return valueMapping.lookup(value); }

  /// Stores the mapping between an LLVM block and its MLIR counterpart.
  void mapBlock(llvm::BasicBlock *llvm, Block *mlir) {
    auto result = blockMapping.try_emplace(llvm, mlir);
    (void)result;
    assert(result.second && "attempting to map a block that is already mapped");
  }

  /// Returns the MLIR block mapped to the given LLVM block.
  Block *lookupBlock(llvm::BasicBlock *block) const {
    return blockMapping.lookup(block);
  }

  /// Converts an LLVM value to an MLIR value, or returns failure if the
  /// conversion fails. Uses the `convertConstant` method to translate constant
  /// LLVM values.
  FailureOr<Value> convertValue(llvm::Value *value);

  /// Converts a range of LLVM values to a range of MLIR values using the
  /// `convertValue` method, or returns failure if the conversion fails.
  FailureOr<SmallVector<Value>> convertValues(ArrayRef<llvm::Value *> values);

  /// Converts `value` to an integer attribute. Asserts if the matching fails.
  IntegerAttr matchIntegerAttr(llvm::Value *value);

  /// Converts `value` to a local variable attribute. Asserts if the matching
  /// fails.
  DILocalVariableAttr matchLocalVariableAttr(llvm::Value *value);

  /// Translates the debug location.
  Location translateLoc(llvm::DILocation *loc) {
    return debugImporter.translateLoc(loc);
  }

  /// Converts the type from LLVM to MLIR LLVM dialect.
  Type convertType(llvm::Type *type) {
    return typeTranslator.translateType(type);
  }

  /// Converts an LLVM intrinsic to an MLIR LLVM dialect operation if an MLIR
  /// counterpart exists. Otherwise, returns failure.
  LogicalResult convertIntrinsic(OpBuilder &odsBuilder, llvm::CallInst *inst,
                                 llvm::Intrinsic::ID intrinsicID);

  /// Converts an LLVM instruction to an MLIR LLVM dialect operation if an MLIR
  /// counterpart exists. Otherwise, returns failure.
  LogicalResult convertOperation(OpBuilder &odsBuilder,
                                 llvm::Instruction *inst);

  /// Imports `func` into the current module.
  LogicalResult processFunction(llvm::Function *func);

  /// Converts function attributes of LLVM Function \p func
  /// into LLVM dialect attributes of LLVMFuncOp \p funcOp.
  void processFunctionAttributes(llvm::Function *func, LLVMFuncOp funcOp);

  /// Imports `globalVar` as a GlobalOp, creating it if it doesn't exist.
  GlobalOp processGlobal(llvm::GlobalVariable *globalVar);

private:
  /// Clears the block and value mapping before processing a new region.
  void clearBlockAndValueMapping() {
    valueMapping.clear();
    blockMapping.clear();
  }
  /// Sets the constant insertion point to the start of the given block.
  void setConstantInsertionPointToStart(Block *block) {
    constantInsertionBlock = block;
    constantInsertionOp = nullptr;
  }

  /// Returns personality of `func` as a FlatSymbolRefAttr.
  FlatSymbolRefAttr getPersonalityAsAttr(llvm::Function *func);
  /// Imports `bb` into `block`, which must be initially empty.
  LogicalResult processBasicBlock(llvm::BasicBlock *bb, Block *block);
  /// Imports `inst` and populates valueMapping[inst] with the result of the
  /// imported operation.
  LogicalResult processInstruction(llvm::Instruction *inst);
  /// Converts the `branch` arguments in the order of the phi's found in
  /// `target` and appends them to the `blockArguments` to attach to the
  /// generated branch operation. The `blockArguments` thus have the same order
  /// as the phi's in `target`.
  LogicalResult convertBranchArgs(llvm::Instruction *branch,
                                  llvm::BasicBlock *target,
                                  SmallVectorImpl<Value> &blockArguments);
  /// Appends the converted result type and operands of `callInst` to the
  /// `types` and `operands` arrays. For indirect calls, the method additionally
  /// inserts the called function at the beginning of the `operands` array.
  LogicalResult convertCallTypeAndOperands(llvm::CallBase *callInst,
                                           SmallVectorImpl<Type> &types,
                                           SmallVectorImpl<Value> &operands);
  /// Returns the builtin type equivalent to be used in attributes for the given
  /// LLVM IR dialect type.
  Type getStdTypeForAttr(Type type);
  /// Returns `value` as an attribute to attach to a GlobalOp.
  Attribute getConstantAsAttr(llvm::Constant *value);
  /// Returns the topologically sorted set of transitive dependencies needed to
  /// convert the given constant.
  SetVector<llvm::Constant *> getConstantsToConvert(llvm::Constant *constant);
  /// Converts an LLVM constant to an MLIR value, or returns failure if the
  /// conversion fails. The MLIR value may be produced by a ConstantOp,
  /// AddressOfOp, NullOp, or a side-effect free operation (for ConstantExprs or
  /// ConstantGEPs).
  FailureOr<Value> convertConstant(llvm::Constant *constant);
  /// Converts an LLVM constant and its transitive constant dependencies to MLIR
  /// operations by converting them in topological order using the
  /// `convertConstant` method, or returns failure if the conversion of any of
  /// them fails. All operations are inserted at the start of the current
  /// function entry block.
  FailureOr<Value> convertConstantExpr(llvm::Constant *constant);

  /// Builder pointing at where the next instruction should be generated.
  OpBuilder builder;
  /// Block to insert the next constant into.
  Block *constantInsertionBlock = nullptr;
  /// Operation to insert the next constant after.
  Operation *constantInsertionOp = nullptr;
  /// Operation to insert the next global after.
  Operation *globalInsertionOp = nullptr;
  /// The current context.
  MLIRContext *context;
  /// The current module being created.
  ModuleOp module;

  /// Function-local mapping between original and imported block.
  DenseMap<llvm::BasicBlock *, Block *> blockMapping;
  /// Function-local mapping between original and imported values.
  DenseMap<llvm::Value *, Value> valueMapping;
  /// Uniquing map of GlobalVariables.
  DenseMap<llvm::GlobalVariable *, GlobalOp> globals;
  /// The stateful type translator (contains named structs).
  LLVM::TypeFromLLVMIRTranslator typeTranslator;
  /// Stateful debug information importer.
  DebugImporter debugImporter;
};
} // namespace

// We only need integers, floats, doubles, and vectors and tensors thereof for
// attributes. Scalar and vector types are converted to the standard
// equivalents. Array types are converted to ranked tensors; nested array types
// are converted to multi-dimensional tensors or vectors, depending on the
// innermost type being a scalar or a vector.
Type Importer::getStdTypeForAttr(Type type) {
  if (!type)
    return nullptr;

  if (type.isa<IntegerType, FloatType>())
    return type;

  // LLVM vectors can only contain scalars.
  if (LLVM::isCompatibleVectorType(type)) {
    llvm::ElementCount numElements = LLVM::getVectorNumElements(type);
    if (numElements.isScalable()) {
      emitError(UnknownLoc::get(context)) << "scalable vectors not supported";
      return nullptr;
    }
    Type elementType = getStdTypeForAttr(LLVM::getVectorElementType(type));
    if (!elementType)
      return nullptr;
    return VectorType::get(numElements.getKnownMinValue(), elementType);
  }

  // LLVM arrays can contain other arrays or vectors.
  if (auto arrayType = type.dyn_cast<LLVMArrayType>()) {
    // Recover the nested array shape.
    SmallVector<int64_t, 4> shape;
    shape.push_back(arrayType.getNumElements());
    while (arrayType.getElementType().isa<LLVMArrayType>()) {
      arrayType = arrayType.getElementType().cast<LLVMArrayType>();
      shape.push_back(arrayType.getNumElements());
    }

    // If the innermost type is a vector, use the multi-dimensional vector as
    // attribute type.
    if (LLVM::isCompatibleVectorType(arrayType.getElementType())) {
      llvm::ElementCount numElements =
          LLVM::getVectorNumElements(arrayType.getElementType());
      if (numElements.isScalable()) {
        emitError(UnknownLoc::get(context)) << "scalable vectors not supported";
        return nullptr;
      }
      shape.push_back(numElements.getKnownMinValue());

      Type elementType = getStdTypeForAttr(
          LLVM::getVectorElementType(arrayType.getElementType()));
      if (!elementType)
        return nullptr;
      return VectorType::get(shape, elementType);
    }

    // Otherwise use a tensor.
    Type elementType = getStdTypeForAttr(arrayType.getElementType());
    if (!elementType)
      return nullptr;
    return RankedTensorType::get(shape, elementType);
  }

  return nullptr;
}

// Get the given constant as an attribute. Not all constants can be represented
// as attributes.
Attribute Importer::getConstantAsAttr(llvm::Constant *value) {
  if (auto *ci = dyn_cast<llvm::ConstantInt>(value))
    return builder.getIntegerAttr(
        IntegerType::get(context, ci->getType()->getBitWidth()),
        ci->getValue());
  if (auto *c = dyn_cast<llvm::ConstantDataArray>(value))
    if (c->isString())
      return builder.getStringAttr(c->getAsString());
  if (auto *c = dyn_cast<llvm::ConstantFP>(value)) {
    llvm::Type *type = c->getType();
    FloatType floatTy;
    if (type->isBFloatTy())
      floatTy = FloatType::getBF16(context);
    else
      floatTy = getDLFloatType(*context, type->getScalarSizeInBits());
    assert(floatTy && "unsupported floating point type");
    return builder.getFloatAttr(floatTy, c->getValueAPF());
  }
  if (auto *f = dyn_cast<llvm::Function>(value))
    return SymbolRefAttr::get(builder.getContext(), f->getName());

  // Convert constant data to a dense elements attribute.
  if (auto *cd = dyn_cast<llvm::ConstantDataSequential>(value)) {
    Type type = convertType(cd->getElementType());
    auto attrType = getStdTypeForAttr(convertType(cd->getType()))
                        .dyn_cast_or_null<ShapedType>();
    if (!attrType)
      return nullptr;

    if (type.isa<IntegerType>()) {
      SmallVector<APInt, 8> values;
      values.reserve(cd->getNumElements());
      for (unsigned i = 0, e = cd->getNumElements(); i < e; ++i)
        values.push_back(cd->getElementAsAPInt(i));
      return DenseElementsAttr::get(attrType, values);
    }

    if (type.isa<Float32Type, Float64Type>()) {
      SmallVector<APFloat, 8> values;
      values.reserve(cd->getNumElements());
      for (unsigned i = 0, e = cd->getNumElements(); i < e; ++i)
        values.push_back(cd->getElementAsAPFloat(i));
      return DenseElementsAttr::get(attrType, values);
    }

    return nullptr;
  }

  // Unpack constant aggregates to create dense elements attribute whenever
  // possible. Return nullptr (failure) otherwise.
  if (isa<llvm::ConstantAggregate>(value)) {
    auto outerType = getStdTypeForAttr(convertType(value->getType()))
                         .dyn_cast_or_null<ShapedType>();
    if (!outerType)
      return nullptr;

    SmallVector<Attribute, 8> values;
    SmallVector<int64_t, 8> shape;

    for (unsigned i = 0, e = value->getNumOperands(); i < e; ++i) {
      auto nested = getConstantAsAttr(value->getAggregateElement(i))
                        .dyn_cast_or_null<DenseElementsAttr>();
      if (!nested)
        return nullptr;

      values.append(nested.value_begin<Attribute>(),
                    nested.value_end<Attribute>());
    }

    return DenseElementsAttr::get(outerType, values);
  }

  return nullptr;
}

GlobalOp Importer::processGlobal(llvm::GlobalVariable *globalVar) {
  if (globals.count(globalVar))
    return globals[globalVar];

  // Insert the global after the last one or at the start of the module.
  OpBuilder::InsertionGuard guard(builder);
  if (!globalInsertionOp) {
    builder.setInsertionPointToStart(module.getBody());
  } else {
    builder.setInsertionPointAfter(globalInsertionOp);
  }

  Attribute valueAttr;
  if (globalVar->hasInitializer())
    valueAttr = getConstantAsAttr(globalVar->getInitializer());
  Type type = convertType(globalVar->getValueType());

  uint64_t alignment = 0;
  llvm::MaybeAlign maybeAlign = globalVar->getAlign();
  if (maybeAlign.has_value()) {
    llvm::Align align = maybeAlign.value();
    alignment = align.value();
  }

  GlobalOp globalOp = builder.create<GlobalOp>(
      UnknownLoc::get(context), type, globalVar->isConstant(),
      convertLinkageFromLLVM(globalVar->getLinkage()), globalVar->getName(),
      valueAttr, alignment, /*addr_space=*/globalVar->getAddressSpace(),
      /*dso_local=*/globalVar->isDSOLocal(),
      /*thread_local=*/globalVar->isThreadLocal());
  globalInsertionOp = globalOp;

  if (globalVar->hasInitializer() && !valueAttr) {
    clearBlockAndValueMapping();
    Block *block = builder.createBlock(&globalOp.getInitializerRegion());
    setConstantInsertionPointToStart(block);
    FailureOr<Value> initializer =
        convertConstantExpr(globalVar->getInitializer());
    if (failed(initializer))
      return {};
    builder.create<ReturnOp>(globalOp.getLoc(), initializer.value());
  }
  if (globalVar->hasAtLeastLocalUnnamedAddr()) {
    globalOp.setUnnamedAddr(
        convertUnnamedAddrFromLLVM(globalVar->getUnnamedAddr()));
  }
  if (globalVar->hasSection())
    globalOp.setSection(globalVar->getSection());

  return globals[globalVar] = globalOp;
}

SetVector<llvm::Constant *>
Importer::getConstantsToConvert(llvm::Constant *constant) {
  // Traverse the constant dependencies in post order.
  SmallVector<llvm::Constant *> workList;
  SmallVector<llvm::Constant *> orderedList;
  workList.push_back(constant);
  while (!workList.empty()) {
    llvm::Constant *current = workList.pop_back_val();
    // Skip constants that have been converted before and store all other ones.
    if (valueMapping.count(constant))
      continue;
    orderedList.push_back(current);
    // Add the current constant's dependencies to the work list. Only add
    // constant dependencies and skip any other values such as basic block
    // addresses.
    for (llvm::Value *operand : current->operands())
      if (auto *constDependency = dyn_cast<llvm::Constant>(operand))
        workList.push_back(constDependency);
    // Use the `getElementValue` method to add the dependencies of zero
    // initialized aggregate constants since they do not take any operands.
    if (auto *constAgg = dyn_cast<llvm::ConstantAggregateZero>(current)) {
      unsigned numElements = constAgg->getElementCount().getFixedValue();
      for (unsigned i = 0, e = numElements; i != e; ++i)
        workList.push_back(constAgg->getElementValue(i));
    }
  }

  // Add the constants in reverse post order to the result set to ensure all
  // dependencies are satisfied. Avoid storing duplicates since LLVM constants
  // are uniqued and only one `valueMapping` entry per constant is possible.
  SetVector<llvm::Constant *> orderedSet;
  for (llvm::Constant *orderedConst : llvm::reverse(orderedList))
    orderedSet.insert(orderedConst);
  return orderedSet;
}

FailureOr<Value> Importer::convertConstant(llvm::Constant *constant) {
  // Constants have no location attached.
  Location loc = UnknownLoc::get(context);

  // Convert constants that can be represented as attributes.
  if (Attribute attr = getConstantAsAttr(constant)) {
    Type type = convertType(constant->getType());
    if (auto symbolRef = attr.dyn_cast<FlatSymbolRefAttr>()) {
      return builder.create<AddressOfOp>(loc, type, symbolRef.getValue())
          .getResult();
    }
    return builder.create<ConstantOp>(loc, type, attr).getResult();
  }

  // Convert null pointer constants.
  if (auto *nullPtr = dyn_cast<llvm::ConstantPointerNull>(constant)) {
    Type type = convertType(nullPtr->getType());
    return builder.create<NullOp>(loc, type).getResult();
  }

  // Convert undef.
  if (auto *undefVal = dyn_cast<llvm::UndefValue>(constant)) {
    Type type = convertType(undefVal->getType());
    return builder.create<UndefOp>(loc, type).getResult();
  }

  // Convert global variable accesses.
  if (auto *globalVar = dyn_cast<llvm::GlobalVariable>(constant)) {
    return builder.create<AddressOfOp>(loc, processGlobal(globalVar))
        .getResult();
  }

  // Convert constant expressions.
  if (auto *constExpr = dyn_cast<llvm::ConstantExpr>(constant)) {
    // Convert the constant expression to a temporary LLVM instruction and
    // translate it using the `processInstruction` method. Delete the
    // instruction after the translation and remove it from `valueMapping`,
    // since later calls to `getAsInstruction` may return the same address
    // resulting in a conflicting `valueMapping` entry.
    llvm::Instruction *inst = constExpr->getAsInstruction();
    auto guard = llvm::make_scope_exit([&]() {
      valueMapping.erase(inst);
      inst->deleteValue();
    });
    // Note: `processInstruction` does not call `convertConstant` recursively
    // since all constant dependencies have been converted before.
    assert(llvm::all_of(inst->operands(), [&](llvm::Value *value) {
      return valueMapping.count(value);
    }));
    if (failed(processInstruction(inst)))
      return failure();
    return lookupValue(inst);
  }

  // Convert aggregate constants.
  if (isa<llvm::ConstantAggregate>(constant) ||
      isa<llvm::ConstantAggregateZero>(constant)) {
    // Lookup the aggregate elements that have been converted before.
    SmallVector<Value> elementValues;
    if (auto *constAgg = dyn_cast<llvm::ConstantAggregate>(constant)) {
      elementValues.reserve(constAgg->getNumOperands());
      for (llvm::Value *operand : constAgg->operands())
        elementValues.push_back(lookupValue(operand));
    }
    if (auto *constAgg = dyn_cast<llvm::ConstantAggregateZero>(constant)) {
      unsigned numElements = constAgg->getElementCount().getFixedValue();
      elementValues.reserve(numElements);
      for (unsigned i = 0, e = numElements; i != e; ++i)
        elementValues.push_back(lookupValue(constAgg->getElementValue(i)));
    }
    assert(llvm::count(elementValues, nullptr) == 0 &&
           "expected all elements have been converted before");

    // Generate an UndefOp as root value and insert the aggregate elements.
    Type rootType = convertType(constant->getType());
    bool isArrayOrStruct = rootType.isa<LLVMArrayType, LLVMStructType>();
    assert((isArrayOrStruct || LLVM::isCompatibleVectorType(rootType)) &&
           "unrecognized aggregate type");
    Value root = builder.create<UndefOp>(loc, rootType);
    for (const auto &it : llvm::enumerate(elementValues)) {
      if (isArrayOrStruct) {
        root = builder.create<InsertValueOp>(loc, root, it.value(), it.index());
      } else {
        Attribute indexAttr = builder.getI32IntegerAttr(it.index());
        Value indexValue =
            builder.create<ConstantOp>(loc, builder.getI32Type(), indexAttr);
        root = builder.create<InsertElementOp>(loc, rootType, root, it.value(),
                                               indexValue);
      }
    }
    return root;
  }

  return emitError(loc) << "unhandled constant " << diag(*constant);
}

FailureOr<Value> Importer::convertConstantExpr(llvm::Constant *constant) {
  assert(constantInsertionBlock &&
         "expected the constant insertion block to be non-null");

  // Insert the constant after the last one or at the start or the entry block.
  OpBuilder::InsertionGuard guard(builder);
  if (!constantInsertionOp) {
    builder.setInsertionPointToStart(constantInsertionBlock);
  } else {
    builder.setInsertionPointAfter(constantInsertionOp);
  }

  // Convert all constants of the expression and add them to `valueMapping`.
  SetVector<llvm::Constant *> constantsToConvert =
      getConstantsToConvert(constant);
  for (llvm::Constant *constantToConvert : constantsToConvert) {
    FailureOr<Value> converted = convertConstant(constantToConvert);
    if (failed(converted))
      return failure();
    mapValue(constantToConvert, converted.value());
  }

  // Update the constant insertion point and return the converted constant.
  Value result = lookupValue(constant);
  constantInsertionOp = result.getDefiningOp();
  return result;
}

FailureOr<Value> Importer::convertValue(llvm::Value *value) {
  // A value may be wrapped as metadata, for example, when passed to a debug
  // intrinsic. Unwrap these values before the conversion.
  if (auto *nodeAsVal = dyn_cast<llvm::MetadataAsValue>(value))
    if (auto *node = dyn_cast<llvm::ValueAsMetadata>(nodeAsVal->getMetadata()))
      value = node->getValue();

  // Return the mapped value if it has been converted before.
  if (valueMapping.count(value))
    return lookupValue(value);

  // Convert constants such as immediate values that have no mapping yet.
  if (auto *constant = dyn_cast<llvm::Constant>(value))
    return convertConstantExpr(constant);

  Location loc = UnknownLoc::get(context);
  if (auto *inst = dyn_cast<llvm::Instruction>(value))
    loc = translateLoc(inst->getDebugLoc());
  return emitError(loc) << "unhandled value " << diag(*value);
}

FailureOr<SmallVector<Value>>
Importer::convertValues(ArrayRef<llvm::Value *> values) {
  SmallVector<Value> remapped;
  remapped.reserve(values.size());
  for (llvm::Value *value : values) {
    FailureOr<Value> converted = convertValue(value);
    if (failed(converted))
      return failure();
    remapped.push_back(converted.value());
  }
  return remapped;
}

IntegerAttr Importer::matchIntegerAttr(llvm::Value *value) {
  IntegerAttr integerAttr;
  FailureOr<Value> converted = convertValue(value);
  bool success = succeeded(converted) &&
                 matchPattern(converted.value(), m_Constant(&integerAttr));
  assert(success && "expected a constant value");
  (void)success;
  return integerAttr;
}

DILocalVariableAttr Importer::matchLocalVariableAttr(llvm::Value *value) {
  auto *nodeAsVal = cast<llvm::MetadataAsValue>(value);
  auto *node = cast<llvm::DILocalVariable>(nodeAsVal->getMetadata());
  return debugImporter.translate(node);
}

LogicalResult
Importer::convertBranchArgs(llvm::Instruction *branch, llvm::BasicBlock *target,
                            SmallVectorImpl<Value> &blockArguments) {
  for (auto inst = target->begin(); isa<llvm::PHINode>(inst); ++inst) {
    auto *phiInst = cast<llvm::PHINode>(&*inst);
    llvm::Value *value = phiInst->getIncomingValueForBlock(branch->getParent());
    FailureOr<Value> converted = convertValue(value);
    if (failed(converted))
      return failure();
    blockArguments.push_back(converted.value());
  }
  return success();
}

LogicalResult
Importer::convertCallTypeAndOperands(llvm::CallBase *callInst,
                                     SmallVectorImpl<Type> &types,
                                     SmallVectorImpl<Value> &operands) {
  if (!callInst->getType()->isVoidTy())
    types.push_back(convertType(callInst->getType()));

  if (!callInst->getCalledFunction()) {
    FailureOr<Value> called = convertValue(callInst->getCalledOperand());
    if (failed(called))
      return failure();
    operands.push_back(called.value());
  }
  SmallVector<llvm::Value *> args(callInst->args());
  FailureOr<SmallVector<Value>> arguments = convertValues(args);
  if (failed(arguments))
    return failure();
  llvm::append_range(operands, arguments.value());
  return success();
}

LogicalResult Importer::convertIntrinsic(OpBuilder &odsBuilder,
                                         llvm::CallInst *inst,
                                         llvm::Intrinsic::ID intrinsicID) {
  Location loc = translateLoc(inst->getDebugLoc());

  // Check if the intrinsic is convertible to an MLIR dialect counterpart and
  // copy the arguments to an an LLVM operands array reference for conversion.
  if (isConvertibleIntrinsic(intrinsicID)) {
    SmallVector<llvm::Value *> args(inst->args());
    ArrayRef<llvm::Value *> llvmOperands(args);
#include "mlir/Dialect/LLVMIR/LLVMIntrinsicFromLLVMIRConversions.inc"
  }

  return emitError(loc) << "unhandled intrinsic " << diag(*inst);
}

LogicalResult Importer::convertOperation(OpBuilder &odsBuilder,
                                         llvm::Instruction *inst) {
  // Copy the operands to an LLVM operands array reference for conversion.
  SmallVector<llvm::Value *> operands(inst->operands());
  ArrayRef<llvm::Value *> llvmOperands(operands);

  // Convert all instructions that provide an MLIR builder.
#include "mlir/Dialect/LLVMIR/LLVMOpFromLLVMIRConversions.inc"

  // Convert all remaining instructions that do not provide an MLIR builder.
  Location loc = translateLoc(inst->getDebugLoc());
  if (inst->getOpcode() == llvm::Instruction::Br) {
    auto *brInst = cast<llvm::BranchInst>(inst);

    SmallVector<Block *> succBlocks;
    SmallVector<SmallVector<Value>> succBlockArgs;
    for (auto i : llvm::seq<unsigned>(0, brInst->getNumSuccessors())) {
      llvm::BasicBlock *succ = brInst->getSuccessor(i);
      SmallVector<Value> blockArgs;
      if (failed(convertBranchArgs(brInst, succ, blockArgs)))
        return failure();
      succBlocks.push_back(lookupBlock(succ));
      succBlockArgs.push_back(blockArgs);
    }

    if (brInst->isConditional()) {
      FailureOr<Value> condition = convertValue(brInst->getCondition());
      if (failed(condition))
        return failure();
      builder.create<LLVM::CondBrOp>(loc, condition.value(), succBlocks.front(),
                                     succBlockArgs.front(), succBlocks.back(),
                                     succBlockArgs.back());
    } else {
      builder.create<LLVM::BrOp>(loc, succBlockArgs.front(),
                                 succBlocks.front());
    }
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::Switch) {
    auto *swInst = cast<llvm::SwitchInst>(inst);
    // Process the condition value.
    FailureOr<Value> condition = convertValue(swInst->getCondition());
    if (failed(condition))
      return failure();
    SmallVector<Value> defaultBlockArgs;
    // Process the default case.
    llvm::BasicBlock *defaultBB = swInst->getDefaultDest();
    if (failed(convertBranchArgs(swInst, defaultBB, defaultBlockArgs)))
      return failure();

    // Process the cases.
    unsigned numCases = swInst->getNumCases();
    SmallVector<SmallVector<Value>> caseOperands(numCases);
    SmallVector<ValueRange> caseOperandRefs(numCases);
    SmallVector<int32_t> caseValues(numCases);
    SmallVector<Block *> caseBlocks(numCases);
    for (const auto &it : llvm::enumerate(swInst->cases())) {
      const llvm::SwitchInst::CaseHandle &caseHandle = it.value();
      llvm::BasicBlock *succBB = caseHandle.getCaseSuccessor();
      if (failed(convertBranchArgs(swInst, succBB, caseOperands[it.index()])))
        return failure();
      caseOperandRefs[it.index()] = caseOperands[it.index()];
      caseValues[it.index()] = caseHandle.getCaseValue()->getSExtValue();
      caseBlocks[it.index()] = lookupBlock(succBB);
    }

    builder.create<SwitchOp>(loc, condition.value(), lookupBlock(defaultBB),
                             defaultBlockArgs, caseValues, caseBlocks,
                             caseOperandRefs);
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::PHI) {
    Type type = convertType(inst->getType());
    mapValue(inst, builder.getInsertionBlock()->addArgument(
                       type, translateLoc(inst->getDebugLoc())));
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::Call) {
    auto *callInst = cast<llvm::CallInst>(inst);

    SmallVector<Type> types;
    SmallVector<Value> operands;
    if (failed(convertCallTypeAndOperands(callInst, types, operands)))
      return failure();

    CallOp callOp;
    if (llvm::Function *callee = callInst->getCalledFunction()) {
      callOp = builder.create<CallOp>(
          loc, types, SymbolRefAttr::get(context, callee->getName()), operands);
    } else {
      callOp = builder.create<CallOp>(loc, types, operands);
    }
    if (!callInst->getType()->isVoidTy())
      mapValue(inst, callOp.getResult());
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::LandingPad) {
    auto *lpInst = cast<llvm::LandingPadInst>(inst);

    SmallVector<Value> operands;
    operands.reserve(lpInst->getNumClauses());
    for (auto i : llvm::seq<unsigned>(0, lpInst->getNumClauses())) {
      FailureOr<Value> operand = convertConstantExpr(lpInst->getClause(i));
      if (failed(operand))
        return failure();
      operands.push_back(operand.value());
    }

    Type type = convertType(lpInst->getType());
    Value res =
        builder.create<LandingpadOp>(loc, type, lpInst->isCleanup(), operands);
    mapValue(inst, res);
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::Invoke) {
    auto *invokeInst = cast<llvm::InvokeInst>(inst);

    SmallVector<Type> types;
    SmallVector<Value> operands;
    if (failed(convertCallTypeAndOperands(invokeInst, types, operands)))
      return failure();

    SmallVector<Value> normalArgs, unwindArgs;
    (void)convertBranchArgs(invokeInst, invokeInst->getNormalDest(),
                            normalArgs);
    (void)convertBranchArgs(invokeInst, invokeInst->getUnwindDest(),
                            unwindArgs);

    InvokeOp invokeOp;
    if (llvm::Function *callee = invokeInst->getCalledFunction()) {
      invokeOp = builder.create<InvokeOp>(
          loc, types,
          SymbolRefAttr::get(builder.getContext(), callee->getName()), operands,
          lookupBlock(invokeInst->getNormalDest()), normalArgs,
          lookupBlock(invokeInst->getUnwindDest()), unwindArgs);
    } else {
      invokeOp = builder.create<InvokeOp>(
          loc, types, operands, lookupBlock(invokeInst->getNormalDest()),
          normalArgs, lookupBlock(invokeInst->getUnwindDest()), unwindArgs);
    }
    if (!invokeInst->getType()->isVoidTy())
      mapValue(inst, invokeOp.getResults().front());
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::GetElementPtr) {
    // FIXME: Support inbounds GEPs.
    auto *gepInst = cast<llvm::GetElementPtrInst>(inst);
    Type sourceElementType = convertType(gepInst->getSourceElementType());
    FailureOr<Value> basePtr = convertValue(gepInst->getOperand(0));
    if (failed(basePtr))
      return failure();

    // Treat every indices as dynamic since GEPOp::build will refine those
    // indices into static attributes later. One small downside of this
    // approach is that many unused `llvm.mlir.constant` would be emitted
    // at first place.
    SmallVector<GEPArg> indices;
    for (llvm::Value *operand : llvm::drop_begin(gepInst->operand_values())) {
      FailureOr<Value> index = convertValue(operand);
      if (failed(index))
        return failure();
      indices.push_back(index.value());
    }

    Type type = convertType(inst->getType());
    Value res = builder.create<GEPOp>(loc, type, sourceElementType,
                                      basePtr.value(), indices);
    mapValue(inst, res);
    return success();
  }

  return emitError(loc) << "unhandled instruction " << diag(*inst);
}

LogicalResult Importer::processInstruction(llvm::Instruction *inst) {
  // FIXME: Support uses of SubtargetData.
  // FIXME: Add support for inbounds GEPs.
  // FIXME: Add support for fast-math flags and call / operand attributes.
  // FIXME: Add support for the indirectbr, cleanupret, catchret, catchswitch,
  // callbr, vaarg, landingpad, catchpad, cleanuppad instructions.

  // Convert LLVM intrinsics calls to MLIR intrinsics.
  if (auto *callInst = dyn_cast<llvm::CallInst>(inst)) {
    llvm::Function *callee = callInst->getCalledFunction();
    if (callee && callee->isIntrinsic())
      return convertIntrinsic(builder, callInst, callInst->getIntrinsicID());
  }

  // Convert all remaining LLVM instructions to MLIR operations.
  return convertOperation(builder, inst);
}

FlatSymbolRefAttr Importer::getPersonalityAsAttr(llvm::Function *f) {
  if (!f->hasPersonalityFn())
    return nullptr;

  llvm::Constant *pf = f->getPersonalityFn();

  // If it directly has a name, we can use it.
  if (pf->hasName())
    return SymbolRefAttr::get(builder.getContext(), pf->getName());

  // If it doesn't have a name, currently, only function pointers that are
  // bitcast to i8* are parsed.
  if (auto *ce = dyn_cast<llvm::ConstantExpr>(pf)) {
    if (ce->getOpcode() == llvm::Instruction::BitCast &&
        ce->getType() == llvm::Type::getInt8PtrTy(f->getContext())) {
      if (auto *func = dyn_cast<llvm::Function>(ce->getOperand(0)))
        return SymbolRefAttr::get(builder.getContext(), func->getName());
    }
  }
  return FlatSymbolRefAttr();
}

void Importer::processFunctionAttributes(llvm::Function *func,
                                         LLVMFuncOp funcOp) {
  auto addNamedUnitAttr = [&](StringRef name) {
    return funcOp->setAttr(name, UnitAttr::get(context));
  };
  if (func->doesNotAccessMemory())
    addNamedUnitAttr(LLVMDialect::getReadnoneAttrName());
}

LogicalResult Importer::processFunction(llvm::Function *func) {
  clearBlockAndValueMapping();

  auto functionType =
      convertType(func->getFunctionType()).dyn_cast<LLVMFunctionType>();
  if (func->isIntrinsic() && isConvertibleIntrinsic(func->getIntrinsicID()))
    return success();

  bool dsoLocal = func->hasLocalLinkage();
  CConv cconv = convertCConvFromLLVM(func->getCallingConv());

  // Insert the function at the end of the module.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(module.getBody(), module.getBody()->end());

  LLVMFuncOp funcOp = builder.create<LLVMFuncOp>(
      UnknownLoc::get(context), func->getName(), functionType,
      convertLinkageFromLLVM(func->getLinkage()), dsoLocal, cconv);

  // Set the function debug information if available.
  debugImporter.translate(func, funcOp);

  for (const auto &it : llvm::enumerate(functionType.getParams())) {
    llvm::SmallVector<NamedAttribute, 1> argAttrs;
    if (auto *type = func->getParamByValType(it.index())) {
      Type mlirType = convertType(type);
      argAttrs.push_back(
          NamedAttribute(builder.getStringAttr(LLVMDialect::getByValAttrName()),
                         TypeAttr::get(mlirType)));
    }
    if (auto *type = func->getParamByRefType(it.index())) {
      Type mlirType = convertType(type);
      argAttrs.push_back(
          NamedAttribute(builder.getStringAttr(LLVMDialect::getByRefAttrName()),
                         TypeAttr::get(mlirType)));
    }
    if (auto *type = func->getParamStructRetType(it.index())) {
      Type mlirType = convertType(type);
      argAttrs.push_back(NamedAttribute(
          builder.getStringAttr(LLVMDialect::getStructRetAttrName()),
          TypeAttr::get(mlirType)));
    }
    if (auto *type = func->getParamInAllocaType(it.index())) {
      Type mlirType = convertType(type);
      argAttrs.push_back(NamedAttribute(
          builder.getStringAttr(LLVMDialect::getInAllocaAttrName()),
          TypeAttr::get(mlirType)));
    }

    funcOp.setArgAttrs(it.index(), argAttrs);
  }

  if (FlatSymbolRefAttr personality = getPersonalityAsAttr(func))
    funcOp.setPersonalityAttr(personality);
  else if (func->hasPersonalityFn())
    emitWarning(UnknownLoc::get(context),
                "could not deduce personality, skipping it");

  if (func->hasGC())
    funcOp.setGarbageCollector(StringRef(func->getGC()));

  // Handle Function attributes.
  processFunctionAttributes(func, funcOp);

  if (func->isDeclaration())
    return success();

  // Eagerly create all blocks.
  for (llvm::BasicBlock &bb : *func) {
    Block *block =
        builder.createBlock(&funcOp.getBody(), funcOp.getBody().end());
    mapBlock(&bb, block);
  }

  // Add function arguments to the entry block.
  for (const auto &it : llvm::enumerate(func->args())) {
    BlockArgument blockArg = funcOp.getFunctionBody().addArgument(
        functionType.getParamType(it.index()), funcOp.getLoc());
    mapValue(&it.value(), blockArg);
  }

  // Process the blocks in topological order. The ordered traversal ensures
  // operands defined in a dominating block have a valid mapping to an MLIR
  // value once a block is translated.
  SetVector<llvm::BasicBlock *> blocks = getTopologicallySortedBlocks(func);
  setConstantInsertionPointToStart(lookupBlock(blocks.front()));
  for (llvm::BasicBlock *bb : blocks) {
    if (failed(processBasicBlock(bb, lookupBlock(bb))))
      return failure();
  }

  return success();
}

LogicalResult Importer::processBasicBlock(llvm::BasicBlock *bb, Block *block) {
  builder.setInsertionPointToStart(block);
  for (llvm::Instruction &inst : *bb) {
    if (failed(processInstruction(&inst)))
      return failure();
  }
  return success();
}

OwningOpRef<ModuleOp>
mlir::translateLLVMIRToModule(std::unique_ptr<llvm::Module> llvmModule,
                              MLIRContext *context) {
  context->loadDialect<LLVMDialect>();
  context->loadDialect<DLTIDialect>();
  OwningOpRef<ModuleOp> module(ModuleOp::create(FileLineColLoc::get(
      StringAttr::get(context, llvmModule->getSourceFileName()), /*line=*/0,
      /*column=*/0)));

  DataLayoutSpecInterface dlSpec =
      translateDataLayout(llvmModule->getDataLayout(), context);
  if (!dlSpec) {
    emitError(UnknownLoc::get(context), "can't translate data layout");
    return {};
  }

  module.get()->setAttr(DLTIDialect::kDataLayoutAttrName, dlSpec);

  Importer deserializer(context, module.get());
  for (llvm::GlobalVariable &gv : llvmModule->globals()) {
    if (!deserializer.processGlobal(&gv))
      return {};
  }
  for (llvm::Function &f : llvmModule->functions()) {
    if (failed(deserializer.processFunction(&f)))
      return {};
  }

  return module;
}

// Deserializes the LLVM bitcode stored in `input` into an MLIR module in the
// LLVM dialect.
static OwningOpRef<Operation *>
translateLLVMIRToModule(llvm::SourceMgr &sourceMgr, MLIRContext *context) {
  llvm::SMDiagnostic err;
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = llvm::parseIR(
      *sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()), err, llvmContext);
  if (!llvmModule) {
    std::string errStr;
    llvm::raw_string_ostream errStream(errStr);
    err.print(/*ProgName=*/"", errStream);
    emitError(UnknownLoc::get(context)) << errStream.str();
    return {};
  }
  return translateLLVMIRToModule(std::move(llvmModule), context);
}

namespace mlir {
void registerFromLLVMIRTranslation() {
  TranslateToMLIRRegistration fromLLVM(
      "import-llvm", "from llvm to mlir",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        return ::translateLLVMIRToModule(sourceMgr, context);
      });
}
} // namespace mlir
