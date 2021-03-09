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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Translation.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::LLVM;

#include "mlir/Dialect/LLVMIR/LLVMConversionEnumsFromLLVM.inc"

// Utility to print an LLVM value as a string for passing to emitError().
// FIXME: Diagnostic should be able to natively handle types that have
// operator << (raw_ostream&) defined.
static std::string diag(llvm::Value &v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << v;
  return os.str();
}

namespace mlir {
namespace LLVM {
namespace detail {
/// Support for translating LLVM IR types to MLIR LLVM dialect types.
class TypeFromLLVMIRTranslatorImpl {
public:
  /// Constructs a class creating types in the given MLIR context.
  TypeFromLLVMIRTranslatorImpl(MLIRContext &context) : context(context) {}

  /// Translates the given type.
  Type translateType(llvm::Type *type) {
    if (knownTranslations.count(type))
      return knownTranslations.lookup(type);

    Type translated =
        llvm::TypeSwitch<llvm::Type *, Type>(type)
            .Case<llvm::ArrayType, llvm::FunctionType, llvm::IntegerType,
                  llvm::PointerType, llvm::StructType, llvm::FixedVectorType,
                  llvm::ScalableVectorType>(
                [this](auto *type) { return this->translate(type); })
            .Default([this](llvm::Type *type) {
              return translatePrimitiveType(type);
            });
    knownTranslations.try_emplace(type, translated);
    return translated;
  }

private:
  /// Translates the given primitive, i.e. non-parametric in MLIR nomenclature,
  /// type.
  Type translatePrimitiveType(llvm::Type *type) {
    if (type->isVoidTy())
      return LLVM::LLVMVoidType::get(&context);
    if (type->isHalfTy())
      return Float16Type::get(&context);
    if (type->isBFloatTy())
      return BFloat16Type::get(&context);
    if (type->isFloatTy())
      return Float32Type::get(&context);
    if (type->isDoubleTy())
      return Float64Type::get(&context);
    if (type->isFP128Ty())
      return Float128Type::get(&context);
    if (type->isX86_FP80Ty())
      return Float80Type::get(&context);
    if (type->isPPC_FP128Ty())
      return LLVM::LLVMPPCFP128Type::get(&context);
    if (type->isX86_MMXTy())
      return LLVM::LLVMX86MMXType::get(&context);
    if (type->isLabelTy())
      return LLVM::LLVMLabelType::get(&context);
    if (type->isMetadataTy())
      return LLVM::LLVMMetadataType::get(&context);
    llvm_unreachable("not a primitive type");
  }

  /// Translates the given array type.
  Type translate(llvm::ArrayType *type) {
    return LLVM::LLVMArrayType::get(translateType(type->getElementType()),
                                    type->getNumElements());
  }

  /// Translates the given function type.
  Type translate(llvm::FunctionType *type) {
    SmallVector<Type, 8> paramTypes;
    translateTypes(type->params(), paramTypes);
    return LLVM::LLVMFunctionType::get(translateType(type->getReturnType()),
                                       paramTypes, type->isVarArg());
  }

  /// Translates the given integer type.
  Type translate(llvm::IntegerType *type) {
    return IntegerType::get(&context, type->getBitWidth());
  }

  /// Translates the given pointer type.
  Type translate(llvm::PointerType *type) {
    return LLVM::LLVMPointerType::get(translateType(type->getElementType()),
                                      type->getAddressSpace());
  }

  /// Translates the given structure type.
  Type translate(llvm::StructType *type) {
    SmallVector<Type, 8> subtypes;
    if (type->isLiteral()) {
      translateTypes(type->subtypes(), subtypes);
      return LLVM::LLVMStructType::getLiteral(&context, subtypes,
                                              type->isPacked());
    }

    if (type->isOpaque())
      return LLVM::LLVMStructType::getOpaque(type->getName(), &context);

    LLVM::LLVMStructType translated =
        LLVM::LLVMStructType::getIdentified(&context, type->getName());
    knownTranslations.try_emplace(type, translated);
    translateTypes(type->subtypes(), subtypes);
    LogicalResult bodySet = translated.setBody(subtypes, type->isPacked());
    assert(succeeded(bodySet) &&
           "could not set the body of an identified struct");
    (void)bodySet;
    return translated;
  }

  /// Translates the given fixed-vector type.
  Type translate(llvm::FixedVectorType *type) {
    return LLVM::getFixedVectorType(translateType(type->getElementType()),
                                    type->getNumElements());
  }

  /// Translates the given scalable-vector type.
  Type translate(llvm::ScalableVectorType *type) {
    return LLVM::LLVMScalableVectorType::get(
        translateType(type->getElementType()), type->getMinNumElements());
  }

  /// Translates a list of types.
  void translateTypes(ArrayRef<llvm::Type *> types,
                      SmallVectorImpl<Type> &result) {
    result.reserve(result.size() + types.size());
    for (llvm::Type *type : types)
      result.push_back(translateType(type));
  }

  /// Map of known translations. Serves as a cache and as recursion stopper for
  /// translating recursive structs.
  llvm::DenseMap<llvm::Type *, Type> knownTranslations;

  /// The context in which MLIR types are created.
  MLIRContext &context;
};
} // end namespace detail

/// Utility class to translate LLVM IR types to the MLIR LLVM dialect. Stores
/// the translation state, in particular any identified structure types that are
/// reused across translations.
class TypeFromLLVMIRTranslator {
public:
  TypeFromLLVMIRTranslator(MLIRContext &context);
  ~TypeFromLLVMIRTranslator();

  /// Translates the given LLVM IR type to the MLIR LLVM dialect.
  Type translateType(llvm::Type *type);

private:
  /// Private implementation.
  std::unique_ptr<detail::TypeFromLLVMIRTranslatorImpl> impl;
};

} // end namespace LLVM
} // end namespace mlir

LLVM::TypeFromLLVMIRTranslator::TypeFromLLVMIRTranslator(MLIRContext &context)
    : impl(new detail::TypeFromLLVMIRTranslatorImpl(context)) {}

LLVM::TypeFromLLVMIRTranslator::~TypeFromLLVMIRTranslator() {}

Type LLVM::TypeFromLLVMIRTranslator::translateType(llvm::Type *type) {
  return impl->translateType(type);
}

// Handles importing globals and functions from an LLVM module.
namespace {
class Importer {
public:
  Importer(MLIRContext *context, ModuleOp module)
      : b(context), context(context), module(module),
        unknownLoc(FileLineColLoc::get(context, "imported-bitcode", 0, 0)),
        typeTranslator(*context) {
    b.setInsertionPointToStart(module.getBody());
  }

  /// Imports `f` into the current module.
  LogicalResult processFunction(llvm::Function *f);

  /// Imports GV as a GlobalOp, creating it if it doesn't exist.
  GlobalOp processGlobal(llvm::GlobalVariable *GV);

private:
  /// Returns personality of `f` as a FlatSymbolRefAttr.
  FlatSymbolRefAttr getPersonalityAsAttr(llvm::Function *f);
  /// Imports `bb` into `block`, which must be initially empty.
  LogicalResult processBasicBlock(llvm::BasicBlock *bb, Block *block);
  /// Imports `inst` and populates instMap[inst] with the imported Value.
  LogicalResult processInstruction(llvm::Instruction *inst);
  /// Creates an LLVM-compatible MLIR type for `type`.
  Type processType(llvm::Type *type);
  /// `value` is an SSA-use. Return the remapped version of `value` or a
  /// placeholder that will be remapped later if this is an instruction that
  /// has not yet been visited.
  Value processValue(llvm::Value *value);
  /// Create the most accurate Location possible using a llvm::DebugLoc and
  /// possibly an llvm::Instruction to narrow the Location if debug information
  /// is unavailable.
  Location processDebugLoc(const llvm::DebugLoc &loc,
                           llvm::Instruction *inst = nullptr);
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
  Block *currentEntryBlock;

  /// Globals are inserted before the first function, if any.
  Block::iterator getGlobalInsertPt() {
    auto i = module.getBody()->begin();
    while (!isa<LLVMFuncOp, ModuleTerminatorOp>(i))
      ++i;
    return i;
  }

  /// Functions are always inserted before the module terminator.
  Block::iterator getFuncInsertPt() {
    return std::prev(module.getBody()->end());
  }

  /// Remapped blocks, for the current function.
  DenseMap<llvm::BasicBlock *, Block *> blocks;
  /// Remapped values. These are function-local.
  DenseMap<llvm::Value *, Value> instMap;
  /// Instructions that had not been defined when first encountered as a use.
  /// Maps to the dummy Operation that was created in processValue().
  DenseMap<llvm::Value *, Operation *> unknownInstMap;
  /// Uniquing map of GlobalVariables.
  DenseMap<llvm::GlobalVariable *, GlobalOp> globals;
  /// Cached FileLineColLoc::get("imported-bitcode", 0, 0).
  Location unknownLoc;
  /// The stateful type translator (contains named structs).
  LLVM::TypeFromLLVMIRTranslator typeTranslator;
};
} // namespace

Location Importer::processDebugLoc(const llvm::DebugLoc &loc,
                                   llvm::Instruction *inst) {
  if (!loc && inst) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "llvm-imported-inst-%";
    inst->printAsOperand(os, /*PrintType=*/false);
    return FileLineColLoc::get(context, os.str(), 0, 0);
  } else if (!loc) {
    return unknownLoc;
  }
  // FIXME: Obtain the filename from DILocationInfo.
  return FileLineColLoc::get(context, "imported-bitcode", loc.getLine(),
                             loc.getCol());
}

Type Importer::processType(llvm::Type *type) {
  if (Type result = typeTranslator.translateType(type))
    return result;

  // FIXME: Diagnostic should be able to natively handle types that have
  // operator<<(raw_ostream&) defined.
  std::string s;
  llvm::raw_string_ostream os(s);
  os << *type;
  emitError(unknownLoc) << "unhandled type: " << os.str();
  return nullptr;
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
      emitError(unknownLoc) << "scalable vectors not supported";
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
        emitError(unknownLoc) << "scalable vectors not supported";
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
    if (c->getType()->isDoubleTy())
      return b.getFloatAttr(FloatType::getF64(context), c->getValueAPF());
    if (c->getType()->isFloatingPointTy())
      return b.getFloatAttr(FloatType::getF32(context), c->getValueAPF());
  }
  if (auto *f = dyn_cast<llvm::Function>(value))
    return b.getSymbolRefAttr(f->getName());

  // Convert constant data to a dense elements attribute.
  if (auto *cd = dyn_cast<llvm::ConstantDataSequential>(value)) {
    Type type = processType(cd->getElementType());
    if (!type)
      return nullptr;

    auto attrType = getStdTypeForAttr(processType(cd->getType()))
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
    auto outerType = getStdTypeForAttr(processType(value->getType()))
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

      values.append(nested.attr_value_begin(), nested.attr_value_end());
    }

    return DenseElementsAttr::get(outerType, values);
  }

  return nullptr;
}

GlobalOp Importer::processGlobal(llvm::GlobalVariable *GV) {
  auto it = globals.find(GV);
  if (it != globals.end())
    return it->second;

  OpBuilder b(module.getBody(), getGlobalInsertPt());
  Attribute valueAttr;
  if (GV->hasInitializer())
    valueAttr = getConstantAsAttr(GV->getInitializer());
  Type type = processType(GV->getValueType());
  if (!type)
    return nullptr;
  GlobalOp op = b.create<GlobalOp>(
      UnknownLoc::get(context), type, GV->isConstant(),
      convertLinkageFromLLVM(GV->getLinkage()), GV->getName(), valueAttr);
  if (GV->hasInitializer() && !valueAttr) {
    Region &r = op.getInitializerRegion();
    currentEntryBlock = b.createBlock(&r);
    b.setInsertionPoint(currentEntryBlock, currentEntryBlock->begin());
    Value v = processConstant(GV->getInitializer());
    if (!v)
      return nullptr;
    b.create<ReturnOp>(op.getLoc(), ArrayRef<Value>({v}));
  }
  return globals[GV] = op;
}

Value Importer::processConstant(llvm::Constant *c) {
  OpBuilder bEntry(currentEntryBlock, currentEntryBlock->begin());
  if (Attribute attr = getConstantAsAttr(c)) {
    // These constants can be represented as attributes.
    OpBuilder b(currentEntryBlock, currentEntryBlock->begin());
    Type type = processType(c->getType());
    if (!type)
      return nullptr;
    if (auto symbolRef = attr.dyn_cast<FlatSymbolRefAttr>())
      return instMap[c] = bEntry.create<AddressOfOp>(unknownLoc, type,
                                                     symbolRef.getValue());
    return instMap[c] = bEntry.create<ConstantOp>(unknownLoc, type, attr);
  }
  if (auto *cn = dyn_cast<llvm::ConstantPointerNull>(c)) {
    Type type = processType(cn->getType());
    if (!type)
      return nullptr;
    return instMap[c] = bEntry.create<NullOp>(unknownLoc, type);
  }
  if (auto *GV = dyn_cast<llvm::GlobalVariable>(c))
    return bEntry.create<AddressOfOp>(UnknownLoc::get(context),
                                      processGlobal(GV));

  if (auto *ce = dyn_cast<llvm::ConstantExpr>(c)) {
    llvm::Instruction *i = ce->getAsInstruction();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(currentEntryBlock, currentEntryBlock->begin());
    if (failed(processInstruction(i)))
      return nullptr;
    assert(instMap.count(i));

    // Remove this zombie LLVM instruction now, leaving us only with the MLIR
    // op.
    i->deleteValue();
    return instMap[c] = instMap[i];
  }
  if (auto *ue = dyn_cast<llvm::UndefValue>(c)) {
    Type type = processType(ue->getType());
    if (!type)
      return nullptr;
    return instMap[c] = bEntry.create<UndefOp>(UnknownLoc::get(context), type);
  }
  emitError(unknownLoc) << "unhandled constant: " << diag(*c);
  return nullptr;
}

Value Importer::processValue(llvm::Value *value) {
  auto it = instMap.find(value);
  if (it != instMap.end())
    return it->second;

  // We don't expect to see instructions in dominator order. If we haven't seen
  // this instruction yet, create an unknown op and remap it later.
  if (isa<llvm::Instruction>(value)) {
    OperationState state(UnknownLoc::get(context), "llvm.unknown");
    Type type = processType(value->getType());
    if (!type)
      return nullptr;
    state.addTypes(type);
    unknownInstMap[value] = b.createOperation(state);
    return unknownInstMap[value]->getResult(0);
  }

  if (auto *c = dyn_cast<llvm::Constant>(value))
    return processConstant(c);

  emitError(unknownLoc) << "unhandled value: " << diag(*value);
  return nullptr;
}

/// Return the MLIR OperationName for the given LLVM opcode.
static StringRef lookupOperationNameFromOpcode(unsigned opcode) {
// Maps from LLVM opcode to MLIR OperationName. This is deliberately ordered
// as in llvm/IR/Instructions.def to aid comprehension and spot missing
// instructions.
#define INST(llvm_n, mlir_n)                                                   \
  { llvm::Instruction::llvm_n, LLVM::mlir_n##Op::getOperationName() }
  static const DenseMap<unsigned, StringRef> opcMap = {
      // Ret is handled specially.
      // Br is handled specially.
      // FIXME: switch
      // FIXME: indirectbr
      // FIXME: invoke
      INST(Resume, Resume),
      // FIXME: unreachable
      // FIXME: cleanupret
      // FIXME: catchret
      // FIXME: catchswitch
      // FIXME: callbr
      // FIXME: fneg
      INST(Add, Add), INST(FAdd, FAdd), INST(Sub, Sub), INST(FSub, FSub),
      INST(Mul, Mul), INST(FMul, FMul), INST(UDiv, UDiv), INST(SDiv, SDiv),
      INST(FDiv, FDiv), INST(URem, URem), INST(SRem, SRem), INST(FRem, FRem),
      INST(Shl, Shl), INST(LShr, LShr), INST(AShr, AShr), INST(And, And),
      INST(Or, Or), INST(Xor, XOr), INST(Alloca, Alloca), INST(Load, Load),
      INST(Store, Store),
      // Getelementptr is handled specially.
      INST(Ret, Return), INST(Fence, Fence),
      // FIXME: atomiccmpxchg
      // FIXME: atomicrmw
      INST(Trunc, Trunc), INST(ZExt, ZExt), INST(SExt, SExt),
      INST(FPToUI, FPToUI), INST(FPToSI, FPToSI), INST(UIToFP, UIToFP),
      INST(SIToFP, SIToFP), INST(FPTrunc, FPTrunc), INST(FPExt, FPExt),
      INST(PtrToInt, PtrToInt), INST(IntToPtr, IntToPtr),
      INST(BitCast, Bitcast), INST(AddrSpaceCast, AddrSpaceCast),
      // FIXME: cleanuppad
      // FIXME: catchpad
      // ICmp is handled specially.
      // FIXME: fcmp
      // PHI is handled specially.
      INST(Freeze, Freeze), INST(Call, Call),
      // FIXME: select
      // FIXME: vaarg
      // FIXME: extractelement
      // FIXME: insertelement
      // FIXME: shufflevector
      // FIXME: extractvalue
      // FIXME: insertvalue
      // FIXME: landingpad
  };
#undef INST

  return opcMap.lookup(opcode);
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
  llvm_unreachable("incorrect comparison predicate");
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

// `br` branches to `target`. Return the branch arguments to `br`, in the
// same order of the PHIs in `target`.
LogicalResult
Importer::processBranchArgs(llvm::Instruction *br, llvm::BasicBlock *target,
                            SmallVectorImpl<Value> &blockArguments) {
  for (auto inst = target->begin(); isa<llvm::PHINode>(inst); ++inst) {
    auto *PN = cast<llvm::PHINode>(&*inst);
    Value value = processValue(PN->getIncomingValueForBlock(br->getParent()));
    if (!value)
      return failure();
    blockArguments.push_back(value);
  }
  return success();
}

LogicalResult Importer::processInstruction(llvm::Instruction *inst) {
  // FIXME: Support uses of SubtargetData. Currently inbounds GEPs, fast-math
  // flags and call / operand attributes are not supported.
  Location loc = processDebugLoc(inst->getDebugLoc(), inst);
  Value &v = instMap[inst];
  assert(!v && "processInstruction must be called only once per instruction!");
  switch (inst->getOpcode()) {
  default:
    return emitError(loc) << "unknown instruction: " << diag(*inst);
  case llvm::Instruction::Add:
  case llvm::Instruction::FAdd:
  case llvm::Instruction::Sub:
  case llvm::Instruction::FSub:
  case llvm::Instruction::Mul:
  case llvm::Instruction::FMul:
  case llvm::Instruction::UDiv:
  case llvm::Instruction::SDiv:
  case llvm::Instruction::FDiv:
  case llvm::Instruction::URem:
  case llvm::Instruction::SRem:
  case llvm::Instruction::FRem:
  case llvm::Instruction::Shl:
  case llvm::Instruction::LShr:
  case llvm::Instruction::AShr:
  case llvm::Instruction::And:
  case llvm::Instruction::Or:
  case llvm::Instruction::Xor:
  case llvm::Instruction::Alloca:
  case llvm::Instruction::Load:
  case llvm::Instruction::Store:
  case llvm::Instruction::Ret:
  case llvm::Instruction::Resume:
  case llvm::Instruction::Trunc:
  case llvm::Instruction::ZExt:
  case llvm::Instruction::SExt:
  case llvm::Instruction::FPToUI:
  case llvm::Instruction::FPToSI:
  case llvm::Instruction::UIToFP:
  case llvm::Instruction::SIToFP:
  case llvm::Instruction::FPTrunc:
  case llvm::Instruction::FPExt:
  case llvm::Instruction::PtrToInt:
  case llvm::Instruction::IntToPtr:
  case llvm::Instruction::AddrSpaceCast:
  case llvm::Instruction::Freeze:
  case llvm::Instruction::BitCast: {
    OperationState state(loc, lookupOperationNameFromOpcode(inst->getOpcode()));
    SmallVector<Value, 4> ops;
    ops.reserve(inst->getNumOperands());
    for (auto *op : inst->operand_values()) {
      Value value = processValue(op);
      if (!value)
        return failure();
      ops.push_back(value);
    }
    state.addOperands(ops);
    if (!inst->getType()->isVoidTy()) {
      Type type = processType(inst->getType());
      if (!type)
        return failure();
      state.addTypes(type);
    }
    Operation *op = b.createOperation(state);
    if (!inst->getType()->isVoidTy())
      v = op->getResult(0);
    return success();
  }
  case llvm::Instruction::ICmp: {
    Value lhs = processValue(inst->getOperand(0));
    Value rhs = processValue(inst->getOperand(1));
    if (!lhs || !rhs)
      return failure();
    v = b.create<ICmpOp>(
        loc, getICmpPredicate(cast<llvm::ICmpInst>(inst)->getPredicate()), lhs,
        rhs);
    return success();
  }
  case llvm::Instruction::Br: {
    auto *brInst = cast<llvm::BranchInst>(inst);
    OperationState state(loc,
                         brInst->isConditional() ? "llvm.cond_br" : "llvm.br");
    if (brInst->isConditional()) {
      Value condition = processValue(brInst->getCondition());
      if (!condition)
        return failure();
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
                         b.getI32VectorAttr(operandSegmentSizes));
    }

    b.createOperation(state);
    return success();
  }
  case llvm::Instruction::PHI: {
    Type type = processType(inst->getType());
    if (!type)
      return failure();
    v = b.getInsertionBlock()->addArgument(type);
    return success();
  }
  case llvm::Instruction::Call: {
    llvm::CallInst *ci = cast<llvm::CallInst>(inst);
    SmallVector<Value, 4> ops;
    ops.reserve(inst->getNumOperands());
    for (auto &op : ci->arg_operands()) {
      Value arg = processValue(op.get());
      if (!arg)
        return failure();
      ops.push_back(arg);
    }

    SmallVector<Type, 2> tys;
    if (!ci->getType()->isVoidTy()) {
      Type type = processType(inst->getType());
      if (!type)
        return failure();
      tys.push_back(type);
    }
    Operation *op;
    if (llvm::Function *callee = ci->getCalledFunction()) {
      op = b.create<CallOp>(loc, tys, b.getSymbolRefAttr(callee->getName()),
                            ops);
    } else {
      Value calledValue = processValue(ci->getCalledOperand());
      if (!calledValue)
        return failure();
      ops.insert(ops.begin(), calledValue);
      op = b.create<CallOp>(loc, tys, ops);
    }
    if (!ci->getType()->isVoidTy())
      v = op->getResult(0);
    return success();
  }
  case llvm::Instruction::LandingPad: {
    llvm::LandingPadInst *lpi = cast<llvm::LandingPadInst>(inst);
    SmallVector<Value, 4> ops;

    for (unsigned i = 0, ie = lpi->getNumClauses(); i < ie; i++)
      ops.push_back(processConstant(lpi->getClause(i)));

    Type ty = processType(lpi->getType());
    if (!ty)
      return failure();

    v = b.create<LandingpadOp>(loc, ty, lpi->isCleanup(), ops);
    return success();
  }
  case llvm::Instruction::Invoke: {
    llvm::InvokeInst *ii = cast<llvm::InvokeInst>(inst);

    SmallVector<Type, 2> tys;
    if (!ii->getType()->isVoidTy())
      tys.push_back(processType(inst->getType()));

    SmallVector<Value, 4> ops;
    ops.reserve(inst->getNumOperands() + 1);
    for (auto &op : ii->arg_operands())
      ops.push_back(processValue(op.get()));

    SmallVector<Value, 4> normalArgs, unwindArgs;
    (void)processBranchArgs(ii, ii->getNormalDest(), normalArgs);
    (void)processBranchArgs(ii, ii->getUnwindDest(), unwindArgs);

    Operation *op;
    if (llvm::Function *callee = ii->getCalledFunction()) {
      op = b.create<InvokeOp>(loc, tys, b.getSymbolRefAttr(callee->getName()),
                              ops, blocks[ii->getNormalDest()], normalArgs,
                              blocks[ii->getUnwindDest()], unwindArgs);
    } else {
      ops.insert(ops.begin(), processValue(ii->getCalledOperand()));
      op = b.create<InvokeOp>(loc, tys, ops, blocks[ii->getNormalDest()],
                              normalArgs, blocks[ii->getUnwindDest()],
                              unwindArgs);
    }

    if (!ii->getType()->isVoidTy())
      v = op->getResult(0);
    return success();
  }
  case llvm::Instruction::Fence: {
    StringRef syncscope;
    SmallVector<StringRef, 4> ssNs;
    llvm::LLVMContext &llvmContext = inst->getContext();
    llvm::FenceInst *fence = cast<llvm::FenceInst>(inst);
    llvmContext.getSyncScopeNames(ssNs);
    int fenceSyncScopeID = fence->getSyncScopeID();
    for (unsigned i = 0, e = ssNs.size(); i != e; i++) {
      if (fenceSyncScopeID == llvmContext.getOrInsertSyncScopeID(ssNs[i])) {
        syncscope = ssNs[i];
        break;
      }
    }
    b.create<FenceOp>(loc, getLLVMAtomicOrdering(fence->getOrdering()),
                      syncscope);
    return success();
  }
  case llvm::Instruction::GetElementPtr: {
    // FIXME: Support inbounds GEPs.
    llvm::GetElementPtrInst *gep = cast<llvm::GetElementPtrInst>(inst);
    SmallVector<Value, 4> ops;
    for (auto *op : gep->operand_values()) {
      Value value = processValue(op);
      if (!value)
        return failure();
      ops.push_back(value);
    }
    Type type = processType(inst->getType());
    if (!type)
      return failure();
    v = b.create<GEPOp>(loc, type, ops);
    return success();
  }
  }
}

FlatSymbolRefAttr Importer::getPersonalityAsAttr(llvm::Function *f) {
  if (!f->hasPersonalityFn())
    return nullptr;

  llvm::Constant *pf = f->getPersonalityFn();

  // If it directly has a name, we can use it.
  if (pf->hasName())
    return b.getSymbolRefAttr(pf->getName());

  // If it doesn't have a name, currently, only function pointers that are
  // bitcast to i8* are parsed.
  if (auto ce = dyn_cast<llvm::ConstantExpr>(pf)) {
    if (ce->getOpcode() == llvm::Instruction::BitCast &&
        ce->getType() == llvm::Type::getInt8PtrTy(f->getContext())) {
      if (auto func = dyn_cast<llvm::Function>(ce->getOperand(0)))
        return b.getSymbolRefAttr(func->getName());
    }
  }
  return FlatSymbolRefAttr();
}

LogicalResult Importer::processFunction(llvm::Function *f) {
  blocks.clear();
  instMap.clear();
  unknownInstMap.clear();

  auto functionType =
      processType(f->getFunctionType()).dyn_cast<LLVMFunctionType>();
  if (!functionType)
    return failure();

  b.setInsertionPoint(module.getBody(), getFuncInsertPt());
  LLVMFuncOp fop =
      b.create<LLVMFuncOp>(UnknownLoc::get(context), f->getName(), functionType,
                           convertLinkageFromLLVM(f->getLinkage()));

  if (FlatSymbolRefAttr personality = getPersonalityAsAttr(f))
    fop->setAttr(b.getIdentifier("personality"), personality);
  else if (f->hasPersonalityFn())
    emitWarning(UnknownLoc::get(context),
                "could not deduce personality, skipping it");

  if (f->isDeclaration())
    return success();

  // Eagerly create all blocks.
  SmallVector<Block *, 4> blockList;
  for (llvm::BasicBlock &bb : *f) {
    blockList.push_back(b.createBlock(&fop.body(), fop.body().end()));
    blocks[&bb] = blockList.back();
  }
  currentEntryBlock = blockList[0];

  // Add function arguments to the entry block.
  for (auto kv : llvm::enumerate(f->args()))
    instMap[&kv.value()] =
        blockList[0]->addArgument(functionType.getParamType(kv.index()));

  for (auto bbs : llvm::zip(*f, blockList)) {
    if (failed(processBasicBlock(&std::get<0>(bbs), std::get<1>(bbs))))
      return failure();
  }

  // Now that all instructions are guaranteed to have been visited, ensure
  // any unknown uses we encountered are remapped.
  for (auto &llvmAndUnknown : unknownInstMap) {
    assert(instMap.count(llvmAndUnknown.first));
    Value newValue = instMap[llvmAndUnknown.first];
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

OwningModuleRef
mlir::translateLLVMIRToModule(std::unique_ptr<llvm::Module> llvmModule,
                              MLIRContext *context) {
  context->loadDialect<LLVMDialect>();
  OwningModuleRef module(ModuleOp::create(
      FileLineColLoc::get(context, "", /*line=*/0, /*column=*/0)));

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
OwningModuleRef translateLLVMIRToModule(llvm::SourceMgr &sourceMgr,
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
      "import-llvm", [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        return ::translateLLVMIRToModule(sourceMgr, context);
      });
}
} // namespace mlir
