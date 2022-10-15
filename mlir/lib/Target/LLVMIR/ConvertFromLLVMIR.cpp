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

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Type.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::LLVM;

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
static std::string diag(llvm::Value &v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << v;
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

static ICmpPredicate getICmpPredicate(llvm::CmpInst::Predicate p) {
  switch (p) {
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

static FCmpPredicate getFCmpPredicate(llvm::CmpInst::Predicate p) {
  switch (p) {
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

// Handles importing globals and functions from an LLVM module.
namespace {
class Importer {
public:
  Importer(MLIRContext *context, ModuleOp module)
      : b(context), context(context), module(module), typeTranslator(*context) {
    b.setInsertionPointToStart(module.getBody());
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

  /// Returns the remapped version of `value` or a placeholder that will be
  /// remapped later if the defining instruction has not yet been visited.
  Value processValue(llvm::Value *value);

  /// Calls `processValue` for a range of `values` and returns their remapped
  /// values or placeholders if the defining instructions have not yet been
  /// visited.
  SmallVector<Value> processValues(ArrayRef<llvm::Value *> values);

  /// Converts `value` to an integer attribute. Asserts if the conversion fails.
  IntegerAttr matchIntegerAttr(Value value);

  /// Translate the debug location to a FileLineColLoc, if `loc` is non-null.
  /// Otherwise, return UnknownLoc.
  Location translateLoc(llvm::DILocation *loc);

  /// Converts the type from LLVM to MLIR LLVM dialect.
  Type convertType(llvm::Type *type);

  /// Converts an LLVM intrinsic to an MLIR LLVM dialect operation if an MLIR
  /// counterpart exists. Otherwise, returns failure.
  LogicalResult convertIntrinsic(OpBuilder &odsBuilder, llvm::CallInst *inst);

  /// Converts an LLVM instruction to an MLIR LLVM dialect operation if the
  /// operation defines an MLIR Builder. Otherwise, returns failure.
  LogicalResult convertOperation(OpBuilder &odsBuilder,
                                 llvm::Instruction *inst);

  /// Imports `f` into the current module.
  LogicalResult processFunction(llvm::Function *f);

  /// Converts function attributes of LLVM Function \p f
  /// into LLVM dialect attributes of LLVMFuncOp \p funcOp.
  void processFunctionAttributes(llvm::Function *f, LLVMFuncOp funcOp);

  /// Imports GV as a GlobalOp, creating it if it doesn't exist.
  GlobalOp processGlobal(llvm::GlobalVariable *gv);

private:
  /// Returns personality of `f` as a FlatSymbolRefAttr.
  FlatSymbolRefAttr getPersonalityAsAttr(llvm::Function *f);
  /// Imports `bb` into `block`, which must be initially empty.
  LogicalResult processBasicBlock(llvm::BasicBlock *bb, Block *block);
  /// Imports `inst` and populates valueMapping[inst] with the result of the
  /// imported operation.
  LogicalResult processInstruction(llvm::Instruction *inst);
  /// `br` branches to `target`. Append the block arguments to attach to the
  /// generated branch op to `blockArguments`. These should be in the same order
  /// as the PHIs in `target`.
  LogicalResult processBranchArgs(llvm::Instruction *br,
                                  llvm::BasicBlock *target,
                                  SmallVectorImpl<Value> &blockArguments);
  /// Returns the builtin type equivalent to be used in attributes for the given
  /// LLVM IR dialect type.
  Type getStdTypeForAttr(Type type);
  /// Return `value` as an attribute to attach to a GlobalOp.
  Attribute getConstantAsAttr(llvm::Constant *value);
  /// Return `c` as an MLIR Value. This could either be a ConstantOp, or
  /// an expanded sequence of ops in the current function's entry block (for
  /// ConstantExprs or ConstantGEPs).
  Value processConstant(llvm::Constant *c);

  /// The current builder, pointing at where the next Instruction should be
  /// generated.
  OpBuilder b;
  /// The current context.
  MLIRContext *context;
  /// The current module being created.
  ModuleOp module;
  /// The entry block of the current function being processed.
  Block *currentEntryBlock = nullptr;

  /// Globals are inserted before the first function, if any.
  Block::iterator getGlobalInsertPt() {
    auto it = module.getBody()->begin();
    auto endIt = module.getBody()->end();
    while (it != endIt && !isa<LLVMFuncOp>(it))
      ++it;
    return it;
  }

  /// Functions are always inserted before the module terminator.
  Block::iterator getFuncInsertPt() {
    return std::prev(module.getBody()->end());
  }

  /// Remapped blocks, for the current function.
  DenseMap<llvm::BasicBlock *, Block *> blocks;
  /// Mappings between original and imported values. These are function-local.
  DenseMap<llvm::Value *, Value> valueMapping;
  /// Instructions that had not been defined when first encountered as a use.
  /// Maps to the dummy Operation that was created in processValue().
  DenseMap<llvm::Value *, Operation *> unknownInstMap;
  /// Uniquing map of GlobalVariables.
  DenseMap<llvm::GlobalVariable *, GlobalOp> globals;
  /// The stateful type translator (contains named structs).
  LLVM::TypeFromLLVMIRTranslator typeTranslator;
};
} // namespace

Location Importer::translateLoc(llvm::DILocation *loc) {
  if (!loc)
    return UnknownLoc::get(context);

  return FileLineColLoc::get(context, loc->getFilename(), loc->getLine(),
                             loc->getColumn());
}

Type Importer::convertType(llvm::Type *type) {
  return typeTranslator.translateType(type);
}

LogicalResult Importer::convertIntrinsic(OpBuilder &odsBuilder,
                                         llvm::CallInst *inst) {
  // Check if the callee is an intrinsic.
  llvm::Function *callee = inst->getCalledFunction();
  if (!callee || !callee->isIntrinsic())
    return failure();

  // Check if the intrinsic is convertible to an MLIR dialect counterpart.
  llvm::Intrinsic::ID intrinsicID = callee->getIntrinsicID();
  if (!isConvertibleIntrinsic(intrinsicID))
    return failure();

  // Copy the call arguments to an operands array used by the conversion.
  SmallVector<llvm::Value *> llvmOperands(inst->args());
#include "mlir/Dialect/LLVMIR/LLVMIntrinsicFromLLVMIRConversions.inc"

  return failure();
}

LogicalResult Importer::convertOperation(OpBuilder &odsBuilder,
                                         llvm::Instruction *inst) {
  // Copy the instruction operands used for the conversion.
  SmallVector<llvm::Value *> llvmOperands(inst->operands());
#include "mlir/Dialect/LLVMIR/LLVMOpFromLLVMIRConversions.inc"

  return failure();
}

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
    auto numElements = LLVM::getVectorNumElements(type);
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
      auto numElements = LLVM::getVectorNumElements(arrayType.getElementType());
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
    return b.getIntegerAttr(
        IntegerType::get(context, ci->getType()->getBitWidth()),
        ci->getValue());
  if (auto *c = dyn_cast<llvm::ConstantDataArray>(value))
    if (c->isString())
      return b.getStringAttr(c->getAsString());
  if (auto *c = dyn_cast<llvm::ConstantFP>(value)) {
    auto *type = c->getType();
    FloatType floatTy;
    if (type->isBFloatTy())
      floatTy = FloatType::getBF16(context);
    else
      floatTy = getDLFloatType(*context, type->getScalarSizeInBits());
    assert(floatTy && "unsupported floating point type");
    return b.getFloatAttr(floatTy, c->getValueAPF());
  }
  if (auto *f = dyn_cast<llvm::Function>(value))
    return SymbolRefAttr::get(b.getContext(), f->getName());

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

GlobalOp Importer::processGlobal(llvm::GlobalVariable *gv) {
  auto it = globals.find(gv);
  if (it != globals.end())
    return it->second;

  OpBuilder b(module.getBody(), getGlobalInsertPt());
  Attribute valueAttr;
  if (gv->hasInitializer())
    valueAttr = getConstantAsAttr(gv->getInitializer());
  Type type = convertType(gv->getValueType());

  uint64_t alignment = 0;
  llvm::MaybeAlign maybeAlign = gv->getAlign();
  if (maybeAlign.has_value()) {
    llvm::Align align = maybeAlign.value();
    alignment = align.value();
  }

  GlobalOp op = b.create<GlobalOp>(
      UnknownLoc::get(context), type, gv->isConstant(),
      convertLinkageFromLLVM(gv->getLinkage()), gv->getName(), valueAttr,
      alignment, /*addr_space=*/gv->getAddressSpace(),
      /*dso_local=*/gv->isDSOLocal(), /*thread_local=*/gv->isThreadLocal());

  if (gv->hasInitializer() && !valueAttr) {
    Region &r = op.getInitializerRegion();
    currentEntryBlock = b.createBlock(&r);
    b.setInsertionPoint(currentEntryBlock, currentEntryBlock->begin());
    Value v = processConstant(gv->getInitializer());
    if (!v)
      return nullptr;
    b.create<ReturnOp>(op.getLoc(), ArrayRef<Value>({v}));
  }
  if (gv->hasAtLeastLocalUnnamedAddr())
    op.setUnnamedAddr(convertUnnamedAddrFromLLVM(gv->getUnnamedAddr()));
  if (gv->hasSection())
    op.setSection(gv->getSection());

  return globals[gv] = op;
}

Value Importer::processConstant(llvm::Constant *c) {
  OpBuilder bEntry(currentEntryBlock, currentEntryBlock->begin());
  if (Attribute attr = getConstantAsAttr(c)) {
    // These constants can be represented as attributes.
    OpBuilder b(currentEntryBlock, currentEntryBlock->begin());
    Type type = convertType(c->getType());
    if (auto symbolRef = attr.dyn_cast<FlatSymbolRefAttr>())
      return bEntry.create<AddressOfOp>(UnknownLoc::get(context), type,
                                        symbolRef.getValue());
    return bEntry.create<ConstantOp>(UnknownLoc::get(context), type, attr);
  }
  if (auto *cn = dyn_cast<llvm::ConstantPointerNull>(c)) {
    Type type = convertType(cn->getType());
    return bEntry.create<NullOp>(UnknownLoc::get(context), type);
  }
  if (auto *gv = dyn_cast<llvm::GlobalVariable>(c))
    return bEntry.create<AddressOfOp>(UnknownLoc::get(context),
                                      processGlobal(gv));

  if (auto *ce = dyn_cast<llvm::ConstantExpr>(c)) {
    llvm::Instruction *i = ce->getAsInstruction();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(currentEntryBlock, currentEntryBlock->begin());
    if (failed(processInstruction(i)))
      return nullptr;
    assert(valueMapping.count(i));

    // If we don't remove entry of `i` here, it's totally possible that the
    // next time llvm::ConstantExpr::getAsInstruction is called again, which
    // always allocates a new Instruction, memory address of the newly
    // created Instruction might be the same as `i`. Making processInstruction
    // falsely believe that the new Instruction has been processed before
    // and raised an assertion error.
    Value value = valueMapping[i];
    valueMapping.erase(i);
    // Remove this zombie LLVM instruction now, leaving us only with the MLIR
    // op.
    i->deleteValue();
    return value;
  }
  if (auto *ue = dyn_cast<llvm::UndefValue>(c)) {
    Type type = convertType(ue->getType());
    return bEntry.create<UndefOp>(UnknownLoc::get(context), type);
  }

  if (isa<llvm::ConstantAggregate>(c) || isa<llvm::ConstantAggregateZero>(c)) {
    unsigned numElements = c->getNumOperands();
    std::function<llvm::Constant *(unsigned)> getElement =
        [&](unsigned index) -> llvm::Constant * {
      return c->getAggregateElement(index);
    };
    // llvm::ConstantAggregateZero doesn't take any operand
    // so its getNumOperands is always zero.
    if (auto *caz = dyn_cast<llvm::ConstantAggregateZero>(c)) {
      numElements = caz->getElementCount().getFixedValue();
      // We want to capture the pointer rather than reference
      // to the pointer since the latter will become dangling upon
      // exiting the scope.
      getElement = [=](unsigned index) -> llvm::Constant * {
        return caz->getElementValue(index);
      };
    }

    // Generate a llvm.undef as the root value first.
    Type rootType = convertType(c->getType());
    bool useInsertValue = rootType.isa<LLVMArrayType, LLVMStructType>();
    assert((useInsertValue || LLVM::isCompatibleVectorType(rootType)) &&
           "unrecognized aggregate type");
    Value root = bEntry.create<UndefOp>(UnknownLoc::get(context), rootType);
    for (unsigned i = 0; i < numElements; ++i) {
      llvm::Constant *element = getElement(i);
      Value elementValue = processConstant(element);
      if (!elementValue)
        return nullptr;
      if (useInsertValue) {
        root = bEntry.create<InsertValueOp>(UnknownLoc::get(context), root,
                                            elementValue, i);
      } else {
        Attribute indexAttr = bEntry.getI32IntegerAttr(static_cast<int32_t>(i));
        Value indexValue = bEntry.create<ConstantOp>(
            UnknownLoc::get(context), bEntry.getI32Type(), indexAttr);
        if (!indexValue)
          return nullptr;
        root = bEntry.create<InsertElementOp>(
            UnknownLoc::get(context), rootType, root, elementValue, indexValue);
      }
    }
    return root;
  }

  emitError(UnknownLoc::get(context)) << "unhandled constant: " << diag(*c);
  return nullptr;
}

Value Importer::processValue(llvm::Value *value) {
  auto it = valueMapping.find(value);
  if (it != valueMapping.end())
    return it->second;

  // We don't expect to see instructions in dominator order. If we haven't seen
  // this instruction yet, create an unknown op and remap it later.
  if (isa<llvm::Instruction>(value)) {
    Type type = convertType(value->getType());
    unknownInstMap[value] =
        b.create(UnknownLoc::get(context), b.getStringAttr("llvm.unknown"),
                 /*operands=*/{}, type);
    return unknownInstMap[value]->getResult(0);
  }

  if (auto *c = dyn_cast<llvm::Constant>(value))
    return processConstant(c);

  llvm::errs() << diag(*value) << "\n";
  llvm_unreachable("unhandled value");
}

SmallVector<Value> Importer::processValues(ArrayRef<llvm::Value *> values) {
  SmallVector<Value> remapped;
  remapped.reserve(values.size());
  for (llvm::Value *value : values)
    remapped.push_back(processValue(value));
  return remapped;
}

IntegerAttr Importer::matchIntegerAttr(Value value) {
  IntegerAttr integerAttr;
  bool success = matchPattern(value, m_Constant(&integerAttr));
  assert(success && "expected a constant value");
  (void)success;
  return integerAttr;
}

// `br` branches to `target`. Return the branch arguments to `br`, in the
// same order of the PHIs in `target`.
LogicalResult
Importer::processBranchArgs(llvm::Instruction *br, llvm::BasicBlock *target,
                            SmallVectorImpl<Value> &blockArguments) {
  for (auto inst = target->begin(); isa<llvm::PHINode>(inst); ++inst) {
    auto *pn = cast<llvm::PHINode>(&*inst);
    Value value = processValue(pn->getIncomingValueForBlock(br->getParent()));
    blockArguments.push_back(value);
  }
  return success();
}

LogicalResult Importer::processInstruction(llvm::Instruction *inst) {
  // FIXME: Support uses of SubtargetData.
  // FIXME: Add support for inbounds GEPs.
  // FIXME: Add support for fast-math flags and call / operand attributes.
  // FIXME: Add support for the indirectbr, cleanupret, catchret, catchswitch,
  // callbr, vaarg, landingpad, catchpad, cleanuppad instructions.

  // Convert all intrinsics that provide an MLIR builder.
  if (auto *callInst = dyn_cast<llvm::CallInst>(inst))
    if (succeeded(convertIntrinsic(b, callInst)))
      return success();

  // Convert all operations that provide an MLIR builder.
  if (succeeded(convertOperation(b, inst)))
    return success();

  // Convert all special instructions that do not provide an MLIR builder.
  Location loc = translateLoc(inst->getDebugLoc());
  if (inst->getOpcode() == llvm::Instruction::Br) {
    auto *brInst = cast<llvm::BranchInst>(inst);
    OperationState state(loc,
                         brInst->isConditional() ? "llvm.cond_br" : "llvm.br");
    if (brInst->isConditional()) {
      Value condition = processValue(brInst->getCondition());
      state.addOperands(condition);
    }

    std::array<int32_t, 3> operandSegmentSizes = {1, 0, 0};
    for (int i : llvm::seq<int>(0, brInst->getNumSuccessors())) {
      auto *succ = brInst->getSuccessor(i);
      SmallVector<Value, 4> blockArguments;
      if (failed(processBranchArgs(brInst, succ, blockArguments)))
        return failure();
      state.addSuccessors(blocks[succ]);
      state.addOperands(blockArguments);
      operandSegmentSizes[i + 1] = blockArguments.size();
    }

    if (brInst->isConditional()) {
      state.addAttribute(LLVM::CondBrOp::getOperandSegmentSizeAttr(),
                         b.getDenseI32ArrayAttr(operandSegmentSizes));
    }

    b.create(state);
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::Switch) {
    auto *swInst = cast<llvm::SwitchInst>(inst);
    // Process the condition value.
    Value condition = processValue(swInst->getCondition());
    SmallVector<Value> defaultBlockArgs;
    // Process the default case.
    llvm::BasicBlock *defaultBB = swInst->getDefaultDest();
    if (failed(processBranchArgs(swInst, defaultBB, defaultBlockArgs)))
      return failure();

    // Process the cases.
    unsigned numCases = swInst->getNumCases();
    SmallVector<SmallVector<Value>> caseOperands(numCases);
    SmallVector<ValueRange> caseOperandRefs(numCases);
    SmallVector<int32_t> caseValues(numCases);
    SmallVector<Block *> caseBlocks(numCases);
    for (const auto &en : llvm::enumerate(swInst->cases())) {
      const llvm::SwitchInst::CaseHandle &caseHandle = en.value();
      unsigned i = en.index();
      llvm::BasicBlock *succBB = caseHandle.getCaseSuccessor();
      if (failed(processBranchArgs(swInst, succBB, caseOperands[i])))
        return failure();
      caseOperandRefs[i] = caseOperands[i];
      caseValues[i] = caseHandle.getCaseValue()->getSExtValue();
      caseBlocks[i] = blocks[succBB];
    }

    b.create<SwitchOp>(loc, condition, blocks[defaultBB], defaultBlockArgs,
                       caseValues, caseBlocks, caseOperandRefs);
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::PHI) {
    Type type = convertType(inst->getType());
    mapValue(inst, b.getInsertionBlock()->addArgument(
                       type, translateLoc(inst->getDebugLoc())));
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::Call) {
    llvm::CallInst *ci = cast<llvm::CallInst>(inst);
    SmallVector<llvm::Value *> args(ci->args());
    SmallVector<Value> ops = processValues(args);
    SmallVector<Type, 2> tys;
    if (!ci->getType()->isVoidTy()) {
      Type type = convertType(inst->getType());
      tys.push_back(type);
    }
    Operation *op;
    if (llvm::Function *callee = ci->getCalledFunction()) {
      op = b.create<CallOp>(
          loc, tys, SymbolRefAttr::get(b.getContext(), callee->getName()), ops);
    } else {
      Value calledValue = processValue(ci->getCalledOperand());
      ops.insert(ops.begin(), calledValue);
      op = b.create<CallOp>(loc, tys, ops);
    }
    if (!ci->getType()->isVoidTy())
      mapValue(inst, op->getResult(0));
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::LandingPad) {
    llvm::LandingPadInst *lpi = cast<llvm::LandingPadInst>(inst);
    SmallVector<Value, 4> ops;

    for (unsigned i = 0, ie = lpi->getNumClauses(); i < ie; i++)
      ops.push_back(processConstant(lpi->getClause(i)));

    Type ty = convertType(lpi->getType());
    Value res = b.create<LandingpadOp>(loc, ty, lpi->isCleanup(), ops);
    mapValue(inst, res);
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::Invoke) {
    llvm::InvokeInst *ii = cast<llvm::InvokeInst>(inst);

    SmallVector<Type, 2> tys;
    if (!ii->getType()->isVoidTy())
      tys.push_back(convertType(inst->getType()));

    SmallVector<llvm::Value *> args(ii->args());
    SmallVector<Value> ops = processValues(args);

    SmallVector<Value, 4> normalArgs, unwindArgs;
    (void)processBranchArgs(ii, ii->getNormalDest(), normalArgs);
    (void)processBranchArgs(ii, ii->getUnwindDest(), unwindArgs);

    Operation *op;
    if (llvm::Function *callee = ii->getCalledFunction()) {
      op = b.create<InvokeOp>(
          loc, tys, SymbolRefAttr::get(b.getContext(), callee->getName()), ops,
          blocks[ii->getNormalDest()], normalArgs, blocks[ii->getUnwindDest()],
          unwindArgs);
    } else {
      ops.insert(ops.begin(), processValue(ii->getCalledOperand()));
      op = b.create<InvokeOp>(loc, tys, ops, blocks[ii->getNormalDest()],
                              normalArgs, blocks[ii->getUnwindDest()],
                              unwindArgs);
    }

    if (!ii->getType()->isVoidTy())
      mapValue(inst, op->getResult(0));
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::GetElementPtr) {
    // FIXME: Support inbounds GEPs.
    llvm::GetElementPtrInst *gep = cast<llvm::GetElementPtrInst>(inst);
    Value basePtr = processValue(gep->getOperand(0));
    Type sourceElementType = convertType(gep->getSourceElementType());

    // Treat every indices as dynamic since GEPOp::build will refine those
    // indices into static attributes later. One small downside of this
    // approach is that many unused `llvm.mlir.constant` would be emitted
    // at first place.
    SmallVector<GEPArg> indices;
    for (llvm::Value *operand : llvm::drop_begin(gep->operand_values())) {
      Value val = processValue(operand);
      indices.push_back(val);
    }

    Type type = convertType(inst->getType());
    Value res = b.create<GEPOp>(loc, type, sourceElementType, basePtr, indices);
    mapValue(inst, res);
    return success();
  }

  return emitError(loc) << "unknown instruction: " << diag(*inst);
}

FlatSymbolRefAttr Importer::getPersonalityAsAttr(llvm::Function *f) {
  if (!f->hasPersonalityFn())
    return nullptr;

  llvm::Constant *pf = f->getPersonalityFn();

  // If it directly has a name, we can use it.
  if (pf->hasName())
    return SymbolRefAttr::get(b.getContext(), pf->getName());

  // If it doesn't have a name, currently, only function pointers that are
  // bitcast to i8* are parsed.
  if (auto *ce = dyn_cast<llvm::ConstantExpr>(pf)) {
    if (ce->getOpcode() == llvm::Instruction::BitCast &&
        ce->getType() == llvm::Type::getInt8PtrTy(f->getContext())) {
      if (auto *func = dyn_cast<llvm::Function>(ce->getOperand(0)))
        return SymbolRefAttr::get(b.getContext(), func->getName());
    }
  }
  return FlatSymbolRefAttr();
}

void Importer::processFunctionAttributes(llvm::Function *func,
                                         LLVMFuncOp funcOp) {
  auto addNamedUnitAttr = [&](StringRef name) {
    return funcOp->setAttr(name, UnitAttr::get(context));
  };
  if (func->hasFnAttribute(llvm::Attribute::ReadNone))
    addNamedUnitAttr(LLVMDialect::getReadnoneAttrName());
}

LogicalResult Importer::processFunction(llvm::Function *f) {
  blocks.clear();
  valueMapping.clear();
  unknownInstMap.clear();

  auto functionType =
      convertType(f->getFunctionType()).dyn_cast<LLVMFunctionType>();
  if (f->isIntrinsic() && isConvertibleIntrinsic(f->getIntrinsicID()))
    return success();

  bool dsoLocal = f->hasLocalLinkage();
  CConv cconv = convertCConvFromLLVM(f->getCallingConv());

  b.setInsertionPoint(module.getBody(), getFuncInsertPt());
  LLVMFuncOp fop = b.create<LLVMFuncOp>(
      UnknownLoc::get(context), f->getName(), functionType,
      convertLinkageFromLLVM(f->getLinkage()), dsoLocal, cconv);

  for (const auto &arg : llvm::enumerate(functionType.getParams())) {
    llvm::SmallVector<NamedAttribute, 1> argAttrs;
    if (auto *type = f->getParamByValType(arg.index())) {
      auto mlirType = convertType(type);
      argAttrs.push_back(
          NamedAttribute(b.getStringAttr(LLVMDialect::getByValAttrName()),
                         TypeAttr::get(mlirType)));
    }
    if (auto *type = f->getParamByRefType(arg.index())) {
      auto mlirType = convertType(type);
      argAttrs.push_back(
          NamedAttribute(b.getStringAttr(LLVMDialect::getByRefAttrName()),
                         TypeAttr::get(mlirType)));
    }
    if (auto *type = f->getParamStructRetType(arg.index())) {
      auto mlirType = convertType(type);
      argAttrs.push_back(
          NamedAttribute(b.getStringAttr(LLVMDialect::getStructRetAttrName()),
                         TypeAttr::get(mlirType)));
    }
    if (auto *type = f->getParamInAllocaType(arg.index())) {
      auto mlirType = convertType(type);
      argAttrs.push_back(
          NamedAttribute(b.getStringAttr(LLVMDialect::getInAllocaAttrName()),
                         TypeAttr::get(mlirType)));
    }

    fop.setArgAttrs(arg.index(), argAttrs);
  }

  if (FlatSymbolRefAttr personality = getPersonalityAsAttr(f))
    fop.setPersonalityAttr(personality);
  else if (f->hasPersonalityFn())
    emitWarning(UnknownLoc::get(context),
                "could not deduce personality, skipping it");

  if (f->hasGC())
    fop.setGarbageCollector(StringRef(f->getGC()));

  // Handle Function attributes.
  processFunctionAttributes(f, fop);

  if (f->isDeclaration())
    return success();

  // Eagerly create all blocks.
  SmallVector<Block *, 4> blockList;
  for (llvm::BasicBlock &bb : *f) {
    blockList.push_back(b.createBlock(&fop.getBody(), fop.getBody().end()));
    blocks[&bb] = blockList.back();
  }
  currentEntryBlock = blockList[0];

  // Add function arguments to the entry block.
  for (const auto &kv : llvm::enumerate(f->args())) {
    mapValue(&kv.value(),
             blockList[0]->addArgument(functionType.getParamType(kv.index()),
                                       fop.getLoc()));
  }

  for (auto bbs : llvm::zip(*f, blockList)) {
    if (failed(processBasicBlock(&std::get<0>(bbs), std::get<1>(bbs))))
      return failure();
  }

  // Now that all instructions are guaranteed to have been visited, ensure
  // any unknown uses we encountered are remapped.
  for (auto &llvmAndUnknown : unknownInstMap) {
    assert(valueMapping.count(llvmAndUnknown.first));
    Value newValue = valueMapping[llvmAndUnknown.first];
    Value oldValue = llvmAndUnknown.second->getResult(0);
    oldValue.replaceAllUsesWith(newValue);
    llvmAndUnknown.second->erase();
  }
  return success();
}

LogicalResult Importer::processBasicBlock(llvm::BasicBlock *bb, Block *block) {
  b.setInsertionPointToStart(block);
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
  OwningOpRef<ModuleOp> module(ModuleOp::create(
      FileLineColLoc::get(context, "", /*line=*/0, /*column=*/0)));

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
OwningOpRef<ModuleOp> translateLLVMIRToModule(llvm::SourceMgr &sourceMgr,
                                              MLIRContext *context) {
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
