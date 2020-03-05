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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Translation.h"

#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
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

// Handles importing globals and functions from an LLVM module.
namespace {
class Importer {
public:
  Importer(MLIRContext *context, ModuleOp module)
      : b(context), context(context), module(module),
        unknownLoc(FileLineColLoc::get("imported-bitcode", 0, 0, context)) {
    b.setInsertionPointToStart(module.getBody());
    dialect = context->getRegisteredDialect<LLVMDialect>();
  }

  /// Imports `f` into the current module.
  LogicalResult processFunction(llvm::Function *f);

  /// Imports GV as a GlobalOp, creating it if it doesn't exist.
  GlobalOp processGlobal(llvm::GlobalVariable *GV);

private:
  /// Imports `bb` into `block`, which must be initially empty.
  LogicalResult processBasicBlock(llvm::BasicBlock *bb, Block *block);
  /// Imports `inst` and populates instMap[inst] with the imported Value.
  LogicalResult processInstruction(llvm::Instruction *inst);
  /// Creates an LLVMType for `type`.
  LLVMType processType(llvm::Type *type);
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
  /// Returns the standard type equivalent to be used in attributes for the
  /// given LLVM IR dialect type.
  Type getStdTypeForAttr(LLVMType type);
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
    while (!isa<LLVMFuncOp>(i) && !isa<ModuleTerminatorOp>(i))
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
  /// Cached dialect.
  LLVMDialect *dialect;
};
} // namespace

Location Importer::processDebugLoc(const llvm::DebugLoc &loc,
                                   llvm::Instruction *inst) {
  if (!loc && inst) {
    std::string s;
    llvm::raw_string_ostream os(s);
    os << "llvm-imported-inst-%";
    inst->printAsOperand(os, /*PrintType=*/false);
    return FileLineColLoc::get(os.str(), 0, 0, context);
  } else if (!loc) {
    return unknownLoc;
  }
  // FIXME: Obtain the filename from DILocationInfo.
  return FileLineColLoc::get("imported-bitcode", loc.getLine(), loc.getCol(),
                             context);
}

LLVMType Importer::processType(llvm::Type *type) {
  switch (type->getTypeID()) {
  case llvm::Type::FloatTyID:
    return LLVMType::getFloatTy(dialect);
  case llvm::Type::DoubleTyID:
    return LLVMType::getDoubleTy(dialect);
  case llvm::Type::IntegerTyID:
    return LLVMType::getIntNTy(dialect, type->getIntegerBitWidth());
  case llvm::Type::PointerTyID: {
    LLVMType elementType = processType(type->getPointerElementType());
    if (!elementType)
      return nullptr;
    return elementType.getPointerTo(type->getPointerAddressSpace());
  }
  case llvm::Type::ArrayTyID: {
    LLVMType elementType = processType(type->getArrayElementType());
    if (!elementType)
      return nullptr;
    return LLVMType::getArrayTy(elementType, type->getArrayNumElements());
  }
  case llvm::Type::VectorTyID: {
    if (type->getVectorIsScalable()) {
      emitError(unknownLoc) << "scalable vector types not supported";
      return nullptr;
    }
    LLVMType elementType = processType(type->getVectorElementType());
    if (!elementType)
      return nullptr;
    return LLVMType::getVectorTy(elementType, type->getVectorNumElements());
  }
  case llvm::Type::VoidTyID:
    return LLVMType::getVoidTy(dialect);
  case llvm::Type::FP128TyID:
    return LLVMType::getFP128Ty(dialect);
  case llvm::Type::X86_FP80TyID:
    return LLVMType::getX86_FP80Ty(dialect);
  case llvm::Type::StructTyID: {
    SmallVector<LLVMType, 4> elementTypes;
    elementTypes.reserve(type->getStructNumElements());
    for (unsigned i = 0, e = type->getStructNumElements(); i != e; ++i) {
      LLVMType ty = processType(type->getStructElementType(i));
      if (!ty)
        return nullptr;
      elementTypes.push_back(ty);
    }
    return LLVMType::getStructTy(dialect, elementTypes,
                                 cast<llvm::StructType>(type)->isPacked());
  }
  case llvm::Type::FunctionTyID: {
    llvm::FunctionType *fty = cast<llvm::FunctionType>(type);
    SmallVector<LLVMType, 4> paramTypes;
    for (unsigned i = 0, e = fty->getNumParams(); i != e; ++i) {
      LLVMType ty = processType(fty->getParamType(i));
      if (!ty)
        return nullptr;
      paramTypes.push_back(ty);
    }
    LLVMType result = processType(fty->getReturnType());
    if (!result)
      return nullptr;

    return LLVMType::getFunctionTy(result, paramTypes, fty->isVarArg());
  }
  default: {
    // FIXME: Diagnostic should be able to natively handle types that have
    // operator<<(raw_ostream&) defined.
    std::string s;
    llvm::raw_string_ostream os(s);
    os << *type;
    emitError(unknownLoc) << "unhandled type: " << os.str();
    return nullptr;
  }
  }
}

// We only need integers, floats, doubles, and vectors and tensors thereof for
// attributes. Scalar and vector types are converted to the standard
// equivalents. Array types are converted to ranked tensors; nested array types
// are converted to multi-dimensional tensors or vectors, depending on the
// innermost type being a scalar or a vector.
Type Importer::getStdTypeForAttr(LLVMType type) {
  if (!type)
    return nullptr;

  if (type.isIntegerTy())
    return b.getIntegerType(type.getUnderlyingType()->getIntegerBitWidth());

  if (type.getUnderlyingType()->isFloatTy())
    return b.getF32Type();

  if (type.getUnderlyingType()->isDoubleTy())
    return b.getF64Type();

  // LLVM vectors can only contain scalars.
  if (type.isVectorTy()) {
    auto numElements = type.getUnderlyingType()->getVectorElementCount();
    if (numElements.Scalable) {
      emitError(unknownLoc) << "scalable vectors not supported";
      return nullptr;
    }
    Type elementType = getStdTypeForAttr(type.getVectorElementType());
    if (!elementType)
      return nullptr;
    return VectorType::get(numElements.Min, elementType);
  }

  // LLVM arrays can contain other arrays or vectors.
  if (type.isArrayTy()) {
    // Recover the nested array shape.
    SmallVector<int64_t, 4> shape;
    shape.push_back(type.getArrayNumElements());
    while (type.getArrayElementType().isArrayTy()) {
      type = type.getArrayElementType();
      shape.push_back(type.getArrayNumElements());
    }

    // If the innermost type is a vector, use the multi-dimensional vector as
    // attribute type.
    if (type.getArrayElementType().isVectorTy()) {
      LLVMType vectorType = type.getArrayElementType();
      auto numElements =
          vectorType.getUnderlyingType()->getVectorElementCount();
      if (numElements.Scalable) {
        emitError(unknownLoc) << "scalable vectors not supported";
        return nullptr;
      }
      shape.push_back(numElements.Min);

      Type elementType = getStdTypeForAttr(vectorType.getVectorElementType());
      if (!elementType)
        return nullptr;
      return VectorType::get(shape, elementType);
    }

    // Otherwise use a tensor.
    Type elementType = getStdTypeForAttr(type.getArrayElementType());
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
        IntegerType::get(ci->getType()->getBitWidth(), context),
        ci->getValue());
  if (auto *c = dyn_cast<llvm::ConstantDataArray>(value))
    if (c->isString())
      return b.getStringAttr(c->getAsString());
  if (auto *c = dyn_cast<llvm::ConstantFP>(value)) {
    if (c->getType()->isDoubleTy())
      return b.getFloatAttr(FloatType::getF64(context), c->getValueAPF());
    else if (c->getType()->isFloatingPointTy())
      return b.getFloatAttr(FloatType::getF32(context), c->getValueAPF());
  }
  if (auto *f = dyn_cast<llvm::Function>(value))
    return b.getSymbolRefAttr(f->getName());

  // Convert constant data to a dense elements attribute.
  if (auto *cd = dyn_cast<llvm::ConstantDataSequential>(value)) {
    LLVMType type = processType(cd->getElementType());
    if (!type)
      return nullptr;

    auto attrType = getStdTypeForAttr(processType(cd->getType()))
                        .dyn_cast_or_null<ShapedType>();
    if (!attrType)
      return nullptr;

    if (type.isIntegerTy()) {
      SmallVector<APInt, 8> values;
      values.reserve(cd->getNumElements());
      for (unsigned i = 0, e = cd->getNumElements(); i < e; ++i)
        values.push_back(cd->getElementAsAPInt(i));
      return DenseElementsAttr::get(attrType, values);
    }

    if (type.isFloatTy() || type.isDoubleTy()) {
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
  LLVMType type = processType(GV->getValueType());
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
    LLVMType type = processType(c->getType());
    if (!type)
      return nullptr;
    return instMap[c] = bEntry.create<ConstantOp>(unknownLoc, type, attr);
  }
  if (auto *cn = dyn_cast<llvm::ConstantPointerNull>(c)) {
    LLVMType type = processType(cn->getType());
    if (!type)
      return nullptr;
    return instMap[c] = bEntry.create<NullOp>(unknownLoc, type);
  }
  if (auto *GV = dyn_cast<llvm::GlobalVariable>(c))
    return bEntry.create<AddressOfOp>(UnknownLoc::get(context),
                                      processGlobal(GV),
                                      ArrayRef<NamedAttribute>());

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
    LLVMType type = processType(ue->getType());
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
    OperationState state(UnknownLoc::get(context), "unknown");
    LLVMType type = processType(value->getType());
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
    // FIXME: resume
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
    INST(Ret, Return),
    // FIXME: fence
    // FIXME: atomiccmpxchg
    // FIXME: atomicrmw
    INST(Trunc, Trunc), INST(ZExt, ZExt), INST(SExt, SExt),
    INST(FPToUI, FPToUI), INST(FPToSI, FPToSI), INST(UIToFP, UIToFP),
    INST(SIToFP, SIToFP), INST(FPTrunc, FPTrunc), INST(FPExt, FPExt),
    INST(PtrToInt, PtrToInt), INST(IntToPtr, IntToPtr), INST(BitCast, Bitcast),
    INST(AddrSpaceCast, AddrSpaceCast),
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
    OperationState state(loc, opcMap.lookup(inst->getOpcode()));
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
      LLVMType type = processType(inst->getType());
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
    LLVMType type = processType(inst->getType());
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
      LLVMType type = processType(inst->getType());
      if (!type)
        return failure();
      tys.push_back(type);
    }
    Operation *op;
    if (llvm::Function *callee = ci->getCalledFunction()) {
      op = b.create<CallOp>(loc, tys, b.getSymbolRefAttr(callee->getName()),
                            ops);
    } else {
      Value calledValue = processValue(ci->getCalledValue());
      if (!calledValue)
        return failure();
      ops.insert(ops.begin(), calledValue);
      op = b.create<CallOp>(loc, tys, ops, ArrayRef<NamedAttribute>());
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

    b.create<LandingpadOp>(loc, processType(lpi->getType()), lpi->isCleanup(),
                           ops);
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
    processBranchArgs(ii, ii->getNormalDest(), normalArgs);
    processBranchArgs(ii, ii->getUnwindDest(), unwindArgs);

    Operation *op;
    if (llvm::Function *callee = ii->getCalledFunction()) {
      op = b.create<InvokeOp>(loc, tys, b.getSymbolRefAttr(callee->getName()),
                              ops, blocks[ii->getNormalDest()], normalArgs,
                              blocks[ii->getUnwindDest()], unwindArgs);
    } else {
      ops.insert(ops.begin(), processValue(ii->getCalledValue()));
      op = b.create<InvokeOp>(loc, tys, ops, blocks[ii->getNormalDest()],
                              normalArgs, blocks[ii->getUnwindDest()],
                              unwindArgs);
    }

    if (!ii->getType()->isVoidTy())
      v = op->getResult(0);
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
    v = b.create<GEPOp>(loc, type, ops, ArrayRef<NamedAttribute>());
    return success();
  }
  }
}

LogicalResult Importer::processFunction(llvm::Function *f) {
  blocks.clear();
  instMap.clear();
  unknownInstMap.clear();

  LLVMType functionType = processType(f->getFunctionType());
  if (!functionType)
    return failure();

  b.setInsertionPoint(module.getBody(), getFuncInsertPt());
  LLVMFuncOp fop = b.create<LLVMFuncOp>(UnknownLoc::get(context), f->getName(),
                                        functionType);
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
    instMap[&kv.value()] = blockList[0]->addArgument(
        functionType.getFunctionParamType(kv.index()));

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
  OwningModuleRef module(ModuleOp::create(
      FileLineColLoc::get("", /*line=*/0, /*column=*/0, context)));

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
  LLVMDialect *dialect = context->getRegisteredDialect<LLVMDialect>();
  assert(dialect && "Could not find LLVMDialect?");

  llvm::SMDiagnostic err;
  std::unique_ptr<llvm::Module> llvmModule =
      llvm::parseIR(*sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID()), err,
                    dialect->getLLVMContext(),
                    /*UpgradeDebugInfo=*/true,
                    /*DataLayoutString=*/"");
  if (!llvmModule) {
    std::string errStr;
    llvm::raw_string_ostream errStream(errStr);
    err.print(/*ProgName=*/"", errStream);
    emitError(UnknownLoc::get(context)) << errStream.str();
    return {};
  }
  return translateLLVMIRToModule(std::move(llvmModule), context);
}

static TranslateToMLIRRegistration
    fromLLVM("import-llvm",
             [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
               return translateLLVMIRToModule(sourceMgr, context);
             });
