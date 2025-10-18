//===- OpenACC.cpp - OpenACC MLIR Operations ------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include <variant>

using namespace mlir;
using namespace acc;

#include "mlir/Dialect/OpenACC/OpenACCOpsDialect.cpp.inc"
#include "mlir/Dialect/OpenACC/OpenACCOpsEnums.cpp.inc"
#include "mlir/Dialect/OpenACC/OpenACCOpsInterfaces.cpp.inc"
#include "mlir/Dialect/OpenACC/OpenACCTypeInterfaces.cpp.inc"
#include "mlir/Dialect/OpenACCMPCommon/Interfaces/OpenACCMPOpsInterfaces.cpp.inc"

namespace {

static bool isScalarLikeType(Type type) {
  return type.isIntOrIndexOrFloat() || isa<ComplexType>(type);
}

/// Helper function to attach the `VarName` attribute to an operation
/// if a variable name is provided.
static void attachVarNameAttr(Operation *op, OpBuilder &builder,
                              StringRef varName) {
  if (!varName.empty()) {
    auto varNameAttr = acc::VarNameAttr::get(builder.getContext(), varName);
    op->setAttr(acc::getVarNameAttrName(), varNameAttr);
  }
}

struct MemRefPointerLikeModel
    : public PointerLikeType::ExternalModel<MemRefPointerLikeModel,
                                            MemRefType> {
  Type getElementType(Type pointer) const {
    return cast<MemRefType>(pointer).getElementType();
  }

  mlir::acc::VariableTypeCategory
  getPointeeTypeCategory(Type pointer, TypedValue<PointerLikeType> varPtr,
                         Type varType) const {
    if (auto mappableTy = dyn_cast<MappableType>(varType)) {
      return mappableTy.getTypeCategory(varPtr);
    }
    auto memrefTy = cast<MemRefType>(pointer);
    if (!memrefTy.hasRank()) {
      // This memref is unranked - aka it could have any rank, including a
      // rank of 0 which could mean scalar. For now, return uncategorized.
      return mlir::acc::VariableTypeCategory::uncategorized;
    }

    if (memrefTy.getRank() == 0) {
      if (isScalarLikeType(memrefTy.getElementType())) {
        return mlir::acc::VariableTypeCategory::scalar;
      }
      // Zero-rank non-scalar - need further analysis to determine the type
      // category. For now, return uncategorized.
      return mlir::acc::VariableTypeCategory::uncategorized;
    }

    // It has a rank - must be an array.
    assert(memrefTy.getRank() > 0 && "rank expected to be positive");
    return mlir::acc::VariableTypeCategory::array;
  }

  mlir::Value genAllocate(Type pointer, OpBuilder &builder, Location loc,
                          StringRef varName, Type varType, Value originalVar,
                          bool &needsFree) const {
    auto memrefTy = cast<MemRefType>(pointer);

    // Check if this is a static memref (all dimensions are known) - if yes
    // then we can generate an alloca operation.
    if (memrefTy.hasStaticShape()) {
      needsFree = false; // alloca doesn't need deallocation
      auto allocaOp = memref::AllocaOp::create(builder, loc, memrefTy);
      attachVarNameAttr(allocaOp, builder, varName);
      return allocaOp.getResult();
    }

    // For dynamic memrefs, extract sizes from the original variable if
    // provided. Otherwise they cannot be handled.
    if (originalVar && originalVar.getType() == memrefTy &&
        memrefTy.hasRank()) {
      SmallVector<Value> dynamicSizes;
      for (int64_t i = 0; i < memrefTy.getRank(); ++i) {
        if (memrefTy.isDynamicDim(i)) {
          // Extract the size of dimension i from the original variable
          auto indexValue = arith::ConstantIndexOp::create(builder, loc, i);
          auto dimSize =
              memref::DimOp::create(builder, loc, originalVar, indexValue);
          dynamicSizes.push_back(dimSize);
        }
        // Note: We only add dynamic sizes to the dynamicSizes array
        // Static dimensions are handled automatically by AllocOp
      }
      needsFree = true; // alloc needs deallocation
      auto allocOp =
          memref::AllocOp::create(builder, loc, memrefTy, dynamicSizes);
      attachVarNameAttr(allocOp, builder, varName);
      return allocOp.getResult();
    }

    // TODO: Unranked not yet supported.
    return {};
  }

  bool genFree(Type pointer, OpBuilder &builder, Location loc,
               TypedValue<PointerLikeType> varToFree, Value allocRes,
               Type varType) const {
    if (auto memrefValue = dyn_cast<TypedValue<MemRefType>>(varToFree)) {
      // Use allocRes if provided to determine the allocation type
      Value valueToInspect = allocRes ? allocRes : memrefValue;

      // Walk through casts to find the original allocation
      Value currentValue = valueToInspect;
      Operation *originalAlloc = nullptr;

      // Follow the chain of operations to find the original allocation
      // even if a casted result is provided.
      while (currentValue) {
        if (auto *definingOp = currentValue.getDefiningOp()) {
          // Check if this is an allocation operation
          if (isa<memref::AllocOp, memref::AllocaOp>(definingOp)) {
            originalAlloc = definingOp;
            break;
          }

          // Check if this is a cast operation we can look through
          if (auto castOp = dyn_cast<memref::CastOp>(definingOp)) {
            currentValue = castOp.getSource();
            continue;
          }

          // Check for other cast-like operations
          if (auto reinterpretCastOp =
                  dyn_cast<memref::ReinterpretCastOp>(definingOp)) {
            currentValue = reinterpretCastOp.getSource();
            continue;
          }

          // If we can't look through this operation, stop
          break;
        }
        // This is a block argument or similar - can't trace further.
        break;
      }

      if (originalAlloc) {
        if (isa<memref::AllocaOp>(originalAlloc)) {
          // This is an alloca - no dealloc needed, but return true (success)
          return true;
        }
        if (isa<memref::AllocOp>(originalAlloc)) {
          // This is an alloc - generate dealloc on varToFree
          memref::DeallocOp::create(builder, loc, memrefValue);
          return true;
        }
      }
    }

    return false;
  }

  bool genCopy(Type pointer, OpBuilder &builder, Location loc,
               TypedValue<PointerLikeType> destination,
               TypedValue<PointerLikeType> source, Type varType) const {
    // Generate a copy operation between two memrefs
    auto destMemref = dyn_cast_if_present<TypedValue<MemRefType>>(destination);
    auto srcMemref = dyn_cast_if_present<TypedValue<MemRefType>>(source);

    // As per memref documentation, source and destination must have same
    // element type and shape in order to be compatible. We do not want to fail
    // with an IR verification error - thus check that before generating the
    // copy operation.
    if (destMemref && srcMemref &&
        destMemref.getType().getElementType() ==
            srcMemref.getType().getElementType() &&
        destMemref.getType().getShape() == srcMemref.getType().getShape()) {
      memref::CopyOp::create(builder, loc, srcMemref, destMemref);
      return true;
    }

    return false;
  }
};

struct LLVMPointerPointerLikeModel
    : public PointerLikeType::ExternalModel<LLVMPointerPointerLikeModel,
                                            LLVM::LLVMPointerType> {
  Type getElementType(Type pointer) const { return Type(); }
};

/// Helper function for any of the times we need to modify an ArrayAttr based on
/// a device type list.  Returns a new ArrayAttr with all of the
/// existingDeviceTypes, plus the effective new ones(or an added none if hte new
/// list is empty).
mlir::ArrayAttr addDeviceTypeAffectedOperandHelper(
    MLIRContext *context, mlir::ArrayAttr existingDeviceTypes,
    llvm::ArrayRef<acc::DeviceType> newDeviceTypes) {
  llvm::SmallVector<mlir::Attribute> deviceTypes;
  if (existingDeviceTypes)
    llvm::copy(existingDeviceTypes, std::back_inserter(deviceTypes));

  if (newDeviceTypes.empty())
    deviceTypes.push_back(
        acc::DeviceTypeAttr::get(context, acc::DeviceType::None));

  for (DeviceType dt : newDeviceTypes)
    deviceTypes.push_back(acc::DeviceTypeAttr::get(context, dt));

  return mlir::ArrayAttr::get(context, deviceTypes);
}

/// Helper function for any of the times we need to add operands that are
/// affected by a device type list. Returns a new ArrayAttr with all of the
/// existingDeviceTypes, plus the effective new ones (or an added none, if the
/// new list is empty). Additionally, adds the arguments to the argCollection
/// the correct number of times. This will also update a 'segments' array, even
/// if it won't be used.
mlir::ArrayAttr addDeviceTypeAffectedOperandHelper(
    MLIRContext *context, mlir::ArrayAttr existingDeviceTypes,
    llvm::ArrayRef<acc::DeviceType> newDeviceTypes, mlir::ValueRange arguments,
    mlir::MutableOperandRange argCollection,
    llvm::SmallVector<int32_t> &segments) {
  llvm::SmallVector<mlir::Attribute> deviceTypes;
  if (existingDeviceTypes)
    llvm::copy(existingDeviceTypes, std::back_inserter(deviceTypes));

  if (newDeviceTypes.empty()) {
    argCollection.append(arguments);
    segments.push_back(arguments.size());
    deviceTypes.push_back(
        acc::DeviceTypeAttr::get(context, acc::DeviceType::None));
  }

  for (DeviceType dt : newDeviceTypes) {
    argCollection.append(arguments);
    segments.push_back(arguments.size());
    deviceTypes.push_back(acc::DeviceTypeAttr::get(context, dt));
  }

  return mlir::ArrayAttr::get(context, deviceTypes);
}

/// Overload for when the 'segments' aren't needed.
mlir::ArrayAttr addDeviceTypeAffectedOperandHelper(
    MLIRContext *context, mlir::ArrayAttr existingDeviceTypes,
    llvm::ArrayRef<acc::DeviceType> newDeviceTypes, mlir::ValueRange arguments,
    mlir::MutableOperandRange argCollection) {
  llvm::SmallVector<int32_t> segments;
  return addDeviceTypeAffectedOperandHelper(context, existingDeviceTypes,
                                            newDeviceTypes, arguments,
                                            argCollection, segments);
}
} // namespace

//===----------------------------------------------------------------------===//
// OpenACC operations
//===----------------------------------------------------------------------===//

void OpenACCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/OpenACC/OpenACCOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/OpenACC/OpenACCOpsAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/OpenACC/OpenACCOpsTypes.cpp.inc"
      >();

  // By attaching interfaces here, we make the OpenACC dialect dependent on
  // the other dialects. This is probably better than having dialects like LLVM
  // and memref be dependent on OpenACC.
  MemRefType::attachInterface<MemRefPointerLikeModel>(*getContext());
  LLVM::LLVMPointerType::attachInterface<LLVMPointerPointerLikeModel>(
      *getContext());
}

//===----------------------------------------------------------------------===//
// device_type support helpers
//===----------------------------------------------------------------------===//

static bool hasDeviceTypeValues(std::optional<mlir::ArrayAttr> arrayAttr) {
  return arrayAttr && *arrayAttr && arrayAttr->size() > 0;
}

static bool hasDeviceType(std::optional<mlir::ArrayAttr> arrayAttr,
                          mlir::acc::DeviceType deviceType) {
  if (!hasDeviceTypeValues(arrayAttr))
    return false;

  for (auto attr : *arrayAttr) {
    auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(attr);
    if (deviceTypeAttr.getValue() == deviceType)
      return true;
  }

  return false;
}

static void printDeviceTypes(mlir::OpAsmPrinter &p,
                             std::optional<mlir::ArrayAttr> deviceTypes) {
  if (!hasDeviceTypeValues(deviceTypes))
    return;

  p << "[";
  llvm::interleaveComma(*deviceTypes, p,
                        [&](mlir::Attribute attr) { p << attr; });
  p << "]";
}

static std::optional<unsigned> findSegment(ArrayAttr segments,
                                           mlir::acc::DeviceType deviceType) {
  unsigned segmentIdx = 0;
  for (auto attr : segments) {
    auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(attr);
    if (deviceTypeAttr.getValue() == deviceType)
      return std::make_optional(segmentIdx);
    ++segmentIdx;
  }
  return std::nullopt;
}

static mlir::Operation::operand_range
getValuesFromSegments(std::optional<mlir::ArrayAttr> arrayAttr,
                      mlir::Operation::operand_range range,
                      std::optional<llvm::ArrayRef<int32_t>> segments,
                      mlir::acc::DeviceType deviceType) {
  if (!arrayAttr)
    return range.take_front(0);
  if (auto pos = findSegment(*arrayAttr, deviceType)) {
    int32_t nbOperandsBefore = 0;
    for (unsigned i = 0; i < *pos; ++i)
      nbOperandsBefore += (*segments)[i];
    return range.drop_front(nbOperandsBefore).take_front((*segments)[*pos]);
  }
  return range.take_front(0);
}

static mlir::Value
getWaitDevnumValue(std::optional<mlir::ArrayAttr> deviceTypeAttr,
                   mlir::Operation::operand_range operands,
                   std::optional<llvm::ArrayRef<int32_t>> segments,
                   std::optional<mlir::ArrayAttr> hasWaitDevnum,
                   mlir::acc::DeviceType deviceType) {
  if (!hasDeviceTypeValues(deviceTypeAttr))
    return {};
  if (auto pos = findSegment(*deviceTypeAttr, deviceType))
    if (hasWaitDevnum->getValue()[*pos])
      return getValuesFromSegments(deviceTypeAttr, operands, segments,
                                   deviceType)
          .front();
  return {};
}

static mlir::Operation::operand_range
getWaitValuesWithoutDevnum(std::optional<mlir::ArrayAttr> deviceTypeAttr,
                           mlir::Operation::operand_range operands,
                           std::optional<llvm::ArrayRef<int32_t>> segments,
                           std::optional<mlir::ArrayAttr> hasWaitDevnum,
                           mlir::acc::DeviceType deviceType) {
  auto range =
      getValuesFromSegments(deviceTypeAttr, operands, segments, deviceType);
  if (range.empty())
    return range;
  if (auto pos = findSegment(*deviceTypeAttr, deviceType)) {
    if (hasWaitDevnum && *hasWaitDevnum) {
      auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>((*hasWaitDevnum)[*pos]);
      if (boolAttr.getValue())
        return range.drop_front(1); // first value is devnum
    }
  }
  return range;
}

template <typename Op>
static LogicalResult checkWaitAndAsyncConflict(Op op) {
  for (uint32_t dtypeInt = 0; dtypeInt != acc::getMaxEnumValForDeviceType();
       ++dtypeInt) {
    auto dtype = static_cast<acc::DeviceType>(dtypeInt);

    // The asyncOnly attribute represent the async clause without value.
    // Therefore the attribute and operand cannot appear at the same time.
    if (hasDeviceType(op.getAsyncOperandsDeviceType(), dtype) &&
        op.hasAsyncOnly(dtype))
      return op.emitError(
          "asyncOnly attribute cannot appear with asyncOperand");

    // The wait attribute represent the wait clause without values. Therefore
    // the attribute and operands cannot appear at the same time.
    if (hasDeviceType(op.getWaitOperandsDeviceType(), dtype) &&
        op.hasWaitOnly(dtype))
      return op.emitError("wait attribute cannot appear with waitOperands");
  }
  return success();
}

template <typename Op>
static LogicalResult checkVarAndVarType(Op op) {
  if (!op.getVar())
    return op.emitError("must have var operand");

  // A variable must have a type that is either pointer-like or mappable.
  if (!mlir::isa<mlir::acc::PointerLikeType>(op.getVar().getType()) &&
      !mlir::isa<mlir::acc::MappableType>(op.getVar().getType()))
    return op.emitError("var must be mappable or pointer-like");

  // When it is a pointer-like type, the varType must capture the target type.
  if (mlir::isa<mlir::acc::PointerLikeType>(op.getVar().getType()) &&
      op.getVarType() == op.getVar().getType())
    return op.emitError("varType must capture the element type of var");

  return success();
}

template <typename Op>
static LogicalResult checkVarAndAccVar(Op op) {
  if (op.getVar().getType() != op.getAccVar().getType())
    return op.emitError("input and output types must match");

  return success();
}

template <typename Op>
static LogicalResult checkNoModifier(Op op) {
  if (op.getModifiers() != acc::DataClauseModifier::none)
    return op.emitError("no data clause modifiers are allowed");
  return success();
}

template <typename Op>
static LogicalResult
checkValidModifier(Op op, acc::DataClauseModifier validModifiers) {
  if (acc::bitEnumContainsAny(op.getModifiers(), ~validModifiers))
    return op.emitError(
        "invalid data clause modifiers: " +
        acc::stringifyDataClauseModifier(op.getModifiers() & ~validModifiers));

  return success();
}

static ParseResult parseVar(mlir::OpAsmParser &parser,
                            OpAsmParser::UnresolvedOperand &var) {
  // Either `var` or `varPtr` keyword is required.
  if (failed(parser.parseOptionalKeyword("varPtr"))) {
    if (failed(parser.parseKeyword("var")))
      return failure();
  }
  if (failed(parser.parseLParen()))
    return failure();
  if (failed(parser.parseOperand(var)))
    return failure();

  return success();
}

static void printVar(mlir::OpAsmPrinter &p, mlir::Operation *op,
                     mlir::Value var) {
  if (mlir::isa<mlir::acc::PointerLikeType>(var.getType()))
    p << "varPtr(";
  else
    p << "var(";
  p.printOperand(var);
}

static ParseResult parseAccVar(mlir::OpAsmParser &parser,
                               OpAsmParser::UnresolvedOperand &var,
                               mlir::Type &accVarType) {
  // Either `accVar` or `accPtr` keyword is required.
  if (failed(parser.parseOptionalKeyword("accPtr"))) {
    if (failed(parser.parseKeyword("accVar")))
      return failure();
  }
  if (failed(parser.parseLParen()))
    return failure();
  if (failed(parser.parseOperand(var)))
    return failure();
  if (failed(parser.parseColon()))
    return failure();
  if (failed(parser.parseType(accVarType)))
    return failure();
  if (failed(parser.parseRParen()))
    return failure();

  return success();
}

static void printAccVar(mlir::OpAsmPrinter &p, mlir::Operation *op,
                        mlir::Value accVar, mlir::Type accVarType) {
  if (mlir::isa<mlir::acc::PointerLikeType>(accVar.getType()))
    p << "accPtr(";
  else
    p << "accVar(";
  p.printOperand(accVar);
  p << " : ";
  p.printType(accVarType);
  p << ")";
}

static ParseResult parseVarPtrType(mlir::OpAsmParser &parser,
                                   mlir::Type &varPtrType,
                                   mlir::TypeAttr &varTypeAttr) {
  if (failed(parser.parseType(varPtrType)))
    return failure();
  if (failed(parser.parseRParen()))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("varType"))) {
    if (failed(parser.parseLParen()))
      return failure();
    mlir::Type varType;
    if (failed(parser.parseType(varType)))
      return failure();
    varTypeAttr = mlir::TypeAttr::get(varType);
    if (failed(parser.parseRParen()))
      return failure();
  } else {
    // Set `varType` from the element type of the type of `varPtr`.
    if (mlir::isa<mlir::acc::PointerLikeType>(varPtrType))
      varTypeAttr = mlir::TypeAttr::get(
          mlir::cast<mlir::acc::PointerLikeType>(varPtrType).getElementType());
    else
      varTypeAttr = mlir::TypeAttr::get(varPtrType);
  }

  return success();
}

static void printVarPtrType(mlir::OpAsmPrinter &p, mlir::Operation *op,
                            mlir::Type varPtrType, mlir::TypeAttr varTypeAttr) {
  p.printType(varPtrType);
  p << ")";

  // Print the `varType` only if it differs from the element type of
  // `varPtr`'s type.
  mlir::Type varType = varTypeAttr.getValue();
  mlir::Type typeToCheckAgainst =
      mlir::isa<mlir::acc::PointerLikeType>(varPtrType)
          ? mlir::cast<mlir::acc::PointerLikeType>(varPtrType).getElementType()
          : varPtrType;
  if (typeToCheckAgainst != varType) {
    p << " varType(";
    p.printType(varType);
    p << ")";
  }
}

//===----------------------------------------------------------------------===//
// DataBoundsOp
//===----------------------------------------------------------------------===//
LogicalResult acc::DataBoundsOp::verify() {
  auto extent = getExtent();
  auto upperbound = getUpperbound();
  if (!extent && !upperbound)
    return emitError("expected extent or upperbound.");
  return success();
}

//===----------------------------------------------------------------------===//
// PrivateOp
//===----------------------------------------------------------------------===//
LogicalResult acc::PrivateOp::verify() {
  if (getDataClause() != acc::DataClause::acc_private)
    return emitError(
        "data clause associated with private operation must match its intent");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkNoModifier(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// FirstprivateOp
//===----------------------------------------------------------------------===//
LogicalResult acc::FirstprivateOp::verify() {
  if (getDataClause() != acc::DataClause::acc_firstprivate)
    return emitError("data clause associated with firstprivate operation must "
                     "match its intent");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkNoModifier(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// ReductionOp
//===----------------------------------------------------------------------===//
LogicalResult acc::ReductionOp::verify() {
  if (getDataClause() != acc::DataClause::acc_reduction)
    return emitError("data clause associated with reduction operation must "
                     "match its intent");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkNoModifier(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// DevicePtrOp
//===----------------------------------------------------------------------===//
LogicalResult acc::DevicePtrOp::verify() {
  if (getDataClause() != acc::DataClause::acc_deviceptr)
    return emitError("data clause associated with deviceptr operation must "
                     "match its intent");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkVarAndAccVar(*this)))
    return failure();
  if (failed(checkNoModifier(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// PresentOp
//===----------------------------------------------------------------------===//
LogicalResult acc::PresentOp::verify() {
  if (getDataClause() != acc::DataClause::acc_present)
    return emitError(
        "data clause associated with present operation must match its intent");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkVarAndAccVar(*this)))
    return failure();
  if (failed(checkNoModifier(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// CopyinOp
//===----------------------------------------------------------------------===//
LogicalResult acc::CopyinOp::verify() {
  // Test for all clauses this operation can be decomposed from:
  if (!getImplicit() && getDataClause() != acc::DataClause::acc_copyin &&
      getDataClause() != acc::DataClause::acc_copyin_readonly &&
      getDataClause() != acc::DataClause::acc_copy &&
      getDataClause() != acc::DataClause::acc_reduction)
    return emitError(
        "data clause associated with copyin operation must match its intent"
        " or specify original clause this operation was decomposed from");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkVarAndAccVar(*this)))
    return failure();
  if (failed(checkValidModifier(*this, acc::DataClauseModifier::readonly |
                                           acc::DataClauseModifier::always |
                                           acc::DataClauseModifier::capture)))
    return failure();
  return success();
}

bool acc::CopyinOp::isCopyinReadonly() {
  return getDataClause() == acc::DataClause::acc_copyin_readonly ||
         acc::bitEnumContainsAny(getModifiers(),
                                 acc::DataClauseModifier::readonly);
}

//===----------------------------------------------------------------------===//
// CreateOp
//===----------------------------------------------------------------------===//
LogicalResult acc::CreateOp::verify() {
  // Test for all clauses this operation can be decomposed from:
  if (getDataClause() != acc::DataClause::acc_create &&
      getDataClause() != acc::DataClause::acc_create_zero &&
      getDataClause() != acc::DataClause::acc_copyout &&
      getDataClause() != acc::DataClause::acc_copyout_zero)
    return emitError(
        "data clause associated with create operation must match its intent"
        " or specify original clause this operation was decomposed from");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkVarAndAccVar(*this)))
    return failure();
  // this op is the entry part of copyout, so it also needs to allow all
  // modifiers allowed on copyout.
  if (failed(checkValidModifier(*this, acc::DataClauseModifier::zero |
                                           acc::DataClauseModifier::always |
                                           acc::DataClauseModifier::capture)))
    return failure();
  return success();
}

bool acc::CreateOp::isCreateZero() {
  // The zero modifier is encoded in the data clause.
  return getDataClause() == acc::DataClause::acc_create_zero ||
         getDataClause() == acc::DataClause::acc_copyout_zero ||
         acc::bitEnumContainsAny(getModifiers(), acc::DataClauseModifier::zero);
}

//===----------------------------------------------------------------------===//
// NoCreateOp
//===----------------------------------------------------------------------===//
LogicalResult acc::NoCreateOp::verify() {
  if (getDataClause() != acc::DataClause::acc_no_create)
    return emitError("data clause associated with no_create operation must "
                     "match its intent");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkVarAndAccVar(*this)))
    return failure();
  if (failed(checkNoModifier(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// AttachOp
//===----------------------------------------------------------------------===//
LogicalResult acc::AttachOp::verify() {
  if (getDataClause() != acc::DataClause::acc_attach)
    return emitError(
        "data clause associated with attach operation must match its intent");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkVarAndAccVar(*this)))
    return failure();
  if (failed(checkNoModifier(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// DeclareDeviceResidentOp
//===----------------------------------------------------------------------===//

LogicalResult acc::DeclareDeviceResidentOp::verify() {
  if (getDataClause() != acc::DataClause::acc_declare_device_resident)
    return emitError("data clause associated with device_resident operation "
                     "must match its intent");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkVarAndAccVar(*this)))
    return failure();
  if (failed(checkNoModifier(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// DeclareLinkOp
//===----------------------------------------------------------------------===//

LogicalResult acc::DeclareLinkOp::verify() {
  if (getDataClause() != acc::DataClause::acc_declare_link)
    return emitError(
        "data clause associated with link operation must match its intent");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkVarAndAccVar(*this)))
    return failure();
  if (failed(checkNoModifier(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// CopyoutOp
//===----------------------------------------------------------------------===//
LogicalResult acc::CopyoutOp::verify() {
  // Test for all clauses this operation can be decomposed from:
  if (getDataClause() != acc::DataClause::acc_copyout &&
      getDataClause() != acc::DataClause::acc_copyout_zero &&
      getDataClause() != acc::DataClause::acc_copy &&
      getDataClause() != acc::DataClause::acc_reduction)
    return emitError(
        "data clause associated with copyout operation must match its intent"
        " or specify original clause this operation was decomposed from");
  if (!getVar() || !getAccVar())
    return emitError("must have both host and device pointers");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkVarAndAccVar(*this)))
    return failure();
  if (failed(checkValidModifier(*this, acc::DataClauseModifier::zero |
                                           acc::DataClauseModifier::always |
                                           acc::DataClauseModifier::capture)))
    return failure();
  return success();
}

bool acc::CopyoutOp::isCopyoutZero() {
  return getDataClause() == acc::DataClause::acc_copyout_zero ||
         acc::bitEnumContainsAny(getModifiers(), acc::DataClauseModifier::zero);
}

//===----------------------------------------------------------------------===//
// DeleteOp
//===----------------------------------------------------------------------===//
LogicalResult acc::DeleteOp::verify() {
  // Test for all clauses this operation can be decomposed from:
  if (getDataClause() != acc::DataClause::acc_delete &&
      getDataClause() != acc::DataClause::acc_create &&
      getDataClause() != acc::DataClause::acc_create_zero &&
      getDataClause() != acc::DataClause::acc_copyin &&
      getDataClause() != acc::DataClause::acc_copyin_readonly &&
      getDataClause() != acc::DataClause::acc_present &&
      getDataClause() != acc::DataClause::acc_no_create &&
      getDataClause() != acc::DataClause::acc_declare_device_resident &&
      getDataClause() != acc::DataClause::acc_declare_link)
    return emitError(
        "data clause associated with delete operation must match its intent"
        " or specify original clause this operation was decomposed from");
  if (!getAccVar())
    return emitError("must have device pointer");
  // This op is the exit part of copyin and create - thus allow all modifiers
  // allowed on either case.
  if (failed(checkValidModifier(*this, acc::DataClauseModifier::zero |
                                           acc::DataClauseModifier::readonly |
                                           acc::DataClauseModifier::always |
                                           acc::DataClauseModifier::capture)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// DetachOp
//===----------------------------------------------------------------------===//
LogicalResult acc::DetachOp::verify() {
  // Test for all clauses this operation can be decomposed from:
  if (getDataClause() != acc::DataClause::acc_detach &&
      getDataClause() != acc::DataClause::acc_attach)
    return emitError(
        "data clause associated with detach operation must match its intent"
        " or specify original clause this operation was decomposed from");
  if (!getAccVar())
    return emitError("must have device pointer");
  if (failed(checkNoModifier(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// HostOp
//===----------------------------------------------------------------------===//
LogicalResult acc::UpdateHostOp::verify() {
  // Test for all clauses this operation can be decomposed from:
  if (getDataClause() != acc::DataClause::acc_update_host &&
      getDataClause() != acc::DataClause::acc_update_self)
    return emitError(
        "data clause associated with host operation must match its intent"
        " or specify original clause this operation was decomposed from");
  if (!getVar() || !getAccVar())
    return emitError("must have both host and device pointers");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkVarAndAccVar(*this)))
    return failure();
  if (failed(checkNoModifier(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// DeviceOp
//===----------------------------------------------------------------------===//
LogicalResult acc::UpdateDeviceOp::verify() {
  // Test for all clauses this operation can be decomposed from:
  if (getDataClause() != acc::DataClause::acc_update_device)
    return emitError(
        "data clause associated with device operation must match its intent"
        " or specify original clause this operation was decomposed from");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkVarAndAccVar(*this)))
    return failure();
  if (failed(checkNoModifier(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// UseDeviceOp
//===----------------------------------------------------------------------===//
LogicalResult acc::UseDeviceOp::verify() {
  // Test for all clauses this operation can be decomposed from:
  if (getDataClause() != acc::DataClause::acc_use_device)
    return emitError(
        "data clause associated with use_device operation must match its intent"
        " or specify original clause this operation was decomposed from");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkVarAndAccVar(*this)))
    return failure();
  if (failed(checkNoModifier(*this)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// CacheOp
//===----------------------------------------------------------------------===//
LogicalResult acc::CacheOp::verify() {
  // Test for all clauses this operation can be decomposed from:
  if (getDataClause() != acc::DataClause::acc_cache &&
      getDataClause() != acc::DataClause::acc_cache_readonly)
    return emitError(
        "data clause associated with cache operation must match its intent"
        " or specify original clause this operation was decomposed from");
  if (failed(checkVarAndVarType(*this)))
    return failure();
  if (failed(checkVarAndAccVar(*this)))
    return failure();
  if (failed(checkValidModifier(*this, acc::DataClauseModifier::readonly)))
    return failure();
  return success();
}

bool acc::CacheOp::isCacheReadonly() {
  return getDataClause() == acc::DataClause::acc_cache_readonly ||
         acc::bitEnumContainsAny(getModifiers(),
                                 acc::DataClauseModifier::readonly);
}

template <typename StructureOp>
static ParseResult parseRegions(OpAsmParser &parser, OperationState &state,
                                unsigned nRegions = 1) {

  SmallVector<Region *, 2> regions;
  for (unsigned i = 0; i < nRegions; ++i)
    regions.push_back(state.addRegion());

  for (Region *region : regions)
    if (parser.parseRegion(*region, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();

  return success();
}

static bool isComputeOperation(Operation *op) {
  return isa<ACC_COMPUTE_CONSTRUCT_AND_LOOP_OPS>(op);
}

namespace {
/// Pattern to remove operation without region that have constant false `ifCond`
/// and remove the condition from the operation if the `ifCond` is a true
/// constant.
template <typename OpTy>
struct RemoveConstantIfCondition : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Early return if there is no condition.
    Value ifCond = op.getIfCond();
    if (!ifCond)
      return failure();

    IntegerAttr constAttr;
    if (!matchPattern(ifCond, m_Constant(&constAttr)))
      return failure();
    if (constAttr.getInt())
      rewriter.modifyOpInPlace(op, [&]() { op.getIfCondMutable().erase(0); });
    else
      rewriter.eraseOp(op);

    return success();
  }
};

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(region.hasOneBlock() && "expected single-block region");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

/// Pattern to remove operation with region that have constant false `ifCond`
/// and remove the condition from the operation if the `ifCond` is constant
/// true.
template <typename OpTy>
struct RemoveConstantIfConditionWithRegion : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Early return if there is no condition.
    Value ifCond = op.getIfCond();
    if (!ifCond)
      return failure();

    IntegerAttr constAttr;
    if (!matchPattern(ifCond, m_Constant(&constAttr)))
      return failure();
    if (constAttr.getInt())
      rewriter.modifyOpInPlace(op, [&]() { op.getIfCondMutable().erase(0); });
    else
      replaceOpWithRegion(rewriter, op, op.getRegion());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Recipe Region Helpers
//===----------------------------------------------------------------------===//

/// Create and populate an init region for privatization recipes.
/// Returns success if the region is populated, failure otherwise.
/// Sets needsFree to indicate if the allocated memory requires deallocation.
static LogicalResult createInitRegion(OpBuilder &builder, Location loc,
                                      Region &initRegion, Type varType,
                                      StringRef varName, ValueRange bounds,
                                      bool &needsFree) {
  // Create init block with arguments: original value + bounds
  SmallVector<Type> argTypes{varType};
  SmallVector<Location> argLocs{loc};
  for (Value bound : bounds) {
    argTypes.push_back(bound.getType());
    argLocs.push_back(loc);
  }

  Block *initBlock = builder.createBlock(&initRegion);
  initBlock->addArguments(argTypes, argLocs);
  builder.setInsertionPointToStart(initBlock);

  Value privatizedValue;

  // Get the block argument that represents the original variable
  Value blockArgVar = initBlock->getArgument(0);

  // Generate init region body based on variable type
  if (isa<MappableType>(varType)) {
    auto mappableTy = cast<MappableType>(varType);
    auto typedVar = cast<TypedValue<MappableType>>(blockArgVar);
    privatizedValue = mappableTy.generatePrivateInit(
        builder, loc, typedVar, varName, bounds, {}, needsFree);
    if (!privatizedValue)
      return failure();
  } else {
    assert(isa<PointerLikeType>(varType) && "Expected PointerLikeType");
    auto pointerLikeTy = cast<PointerLikeType>(varType);
    // Use PointerLikeType's allocation API with the block argument
    privatizedValue = pointerLikeTy.genAllocate(builder, loc, varName, varType,
                                                blockArgVar, needsFree);
    if (!privatizedValue)
      return failure();
  }

  // Add yield operation to init block
  acc::YieldOp::create(builder, loc, privatizedValue);

  return success();
}

/// Create and populate a copy region for firstprivate recipes.
/// Returns success if the region is populated, failure otherwise.
/// TODO: Handle MappableType - it does not yet have a copy API.
static LogicalResult createCopyRegion(OpBuilder &builder, Location loc,
                                      Region &copyRegion, Type varType,
                                      ValueRange bounds) {
  // Create copy block with arguments: original value + privatized value +
  // bounds
  SmallVector<Type> copyArgTypes{varType, varType};
  SmallVector<Location> copyArgLocs{loc, loc};
  for (Value bound : bounds) {
    copyArgTypes.push_back(bound.getType());
    copyArgLocs.push_back(loc);
  }

  Block *copyBlock = builder.createBlock(&copyRegion);
  copyBlock->addArguments(copyArgTypes, copyArgLocs);
  builder.setInsertionPointToStart(copyBlock);

  bool isMappable = isa<MappableType>(varType);
  bool isPointerLike = isa<PointerLikeType>(varType);
  // TODO: Handle MappableType - it does not yet have a copy API.
  // Otherwise, for now just fallback to pointer-like behavior.
  if (isMappable && !isPointerLike)
    return failure();

  // Generate copy region body based on variable type
  if (isPointerLike) {
    auto pointerLikeTy = cast<PointerLikeType>(varType);
    Value originalArg = copyBlock->getArgument(0);
    Value privatizedArg = copyBlock->getArgument(1);

    // Generate copy operation using PointerLikeType interface
    if (!pointerLikeTy.genCopy(
            builder, loc, cast<TypedValue<PointerLikeType>>(privatizedArg),
            cast<TypedValue<PointerLikeType>>(originalArg), varType))
      return failure();
  }

  // Add terminator to copy block
  acc::TerminatorOp::create(builder, loc);

  return success();
}

/// Create and populate a destroy region for privatization recipes.
/// Returns success if the region is populated, failure otherwise.
static LogicalResult createDestroyRegion(OpBuilder &builder, Location loc,
                                         Region &destroyRegion, Type varType,
                                         Value allocRes, ValueRange bounds) {
  // Create destroy block with arguments: original value + privatized value +
  // bounds
  SmallVector<Type> destroyArgTypes{varType, varType};
  SmallVector<Location> destroyArgLocs{loc, loc};
  for (Value bound : bounds) {
    destroyArgTypes.push_back(bound.getType());
    destroyArgLocs.push_back(loc);
  }

  Block *destroyBlock = builder.createBlock(&destroyRegion);
  destroyBlock->addArguments(destroyArgTypes, destroyArgLocs);
  builder.setInsertionPointToStart(destroyBlock);

  auto varToFree =
      cast<TypedValue<PointerLikeType>>(destroyBlock->getArgument(1));
  if (isa<MappableType>(varType)) {
    auto mappableTy = cast<MappableType>(varType);
    if (!mappableTy.generatePrivateDestroy(builder, loc, varToFree))
      return failure();
  } else {
    assert(isa<PointerLikeType>(varType) && "Expected PointerLikeType");
    auto pointerLikeTy = cast<PointerLikeType>(varType);
    if (!pointerLikeTy.genFree(builder, loc, varToFree, allocRes, varType))
      return failure();
  }

  acc::TerminatorOp::create(builder, loc);
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// PrivateRecipeOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyInitLikeSingleArgRegion(
    Operation *op, Region &region, StringRef regionType, StringRef regionName,
    Type type, bool verifyYield, bool optional = false) {
  if (optional && region.empty())
    return success();

  if (region.empty())
    return op->emitOpError() << "expects non-empty " << regionName << " region";
  Block &firstBlock = region.front();
  if (firstBlock.getNumArguments() < 1 ||
      firstBlock.getArgument(0).getType() != type)
    return op->emitOpError() << "expects " << regionName
                             << " region first "
                                "argument of the "
                             << regionType << " type";

  if (verifyYield) {
    for (YieldOp yieldOp : region.getOps<acc::YieldOp>()) {
      if (yieldOp.getOperands().size() != 1 ||
          yieldOp.getOperands().getTypes()[0] != type)
        return op->emitOpError() << "expects " << regionName
                                 << " region to "
                                    "yield a value of the "
                                 << regionType << " type";
    }
  }
  return success();
}

LogicalResult acc::PrivateRecipeOp::verifyRegions() {
  if (failed(verifyInitLikeSingleArgRegion(*this, getInitRegion(),
                                           "privatization", "init", getType(),
                                           /*verifyYield=*/false)))
    return failure();
  if (failed(verifyInitLikeSingleArgRegion(
          *this, getDestroyRegion(), "privatization", "destroy", getType(),
          /*verifyYield=*/false, /*optional=*/true)))
    return failure();
  return success();
}

std::optional<PrivateRecipeOp>
PrivateRecipeOp::createAndPopulate(OpBuilder &builder, Location loc,
                                   StringRef recipeName, Type varType,
                                   StringRef varName, ValueRange bounds) {
  // First, validate that we can handle this variable type
  bool isMappable = isa<MappableType>(varType);
  bool isPointerLike = isa<PointerLikeType>(varType);

  // Unsupported type
  if (!isMappable && !isPointerLike)
    return std::nullopt;

  OpBuilder::InsertionGuard guard(builder);

  // Create the recipe operation first so regions have proper parent context
  auto recipe = PrivateRecipeOp::create(builder, loc, recipeName, varType);

  // Populate the init region
  bool needsFree = false;
  if (failed(createInitRegion(builder, loc, recipe.getInitRegion(), varType,
                              varName, bounds, needsFree))) {
    recipe.erase();
    return std::nullopt;
  }

  // Only create destroy region if the allocation needs deallocation
  if (needsFree) {
    // Extract the allocated value from the init block's yield operation
    auto yieldOp =
        cast<acc::YieldOp>(recipe.getInitRegion().front().getTerminator());
    Value allocRes = yieldOp.getOperand(0);

    if (failed(createDestroyRegion(builder, loc, recipe.getDestroyRegion(),
                                   varType, allocRes, bounds))) {
      recipe.erase();
      return std::nullopt;
    }
  }

  return recipe;
}

//===----------------------------------------------------------------------===//
// FirstprivateRecipeOp
//===----------------------------------------------------------------------===//

LogicalResult acc::FirstprivateRecipeOp::verifyRegions() {
  if (failed(verifyInitLikeSingleArgRegion(*this, getInitRegion(),
                                           "privatization", "init", getType(),
                                           /*verifyYield=*/false)))
    return failure();

  if (getCopyRegion().empty())
    return emitOpError() << "expects non-empty copy region";

  Block &firstBlock = getCopyRegion().front();
  if (firstBlock.getNumArguments() < 2 ||
      firstBlock.getArgument(0).getType() != getType())
    return emitOpError() << "expects copy region with two arguments of the "
                            "privatization type";

  if (getDestroyRegion().empty())
    return success();

  if (failed(verifyInitLikeSingleArgRegion(*this, getDestroyRegion(),
                                           "privatization", "destroy",
                                           getType(), /*verifyYield=*/false)))
    return failure();

  return success();
}

std::optional<FirstprivateRecipeOp>
FirstprivateRecipeOp::createAndPopulate(OpBuilder &builder, Location loc,
                                        StringRef recipeName, Type varType,
                                        StringRef varName, ValueRange bounds) {
  // First, validate that we can handle this variable type
  bool isMappable = isa<MappableType>(varType);
  bool isPointerLike = isa<PointerLikeType>(varType);

  // Unsupported type
  if (!isMappable && !isPointerLike)
    return std::nullopt;

  OpBuilder::InsertionGuard guard(builder);

  // Create the recipe operation first so regions have proper parent context
  auto recipe = FirstprivateRecipeOp::create(builder, loc, recipeName, varType);

  // Populate the init region
  bool needsFree = false;
  if (failed(createInitRegion(builder, loc, recipe.getInitRegion(), varType,
                              varName, bounds, needsFree))) {
    recipe.erase();
    return std::nullopt;
  }

  // Populate the copy region
  if (failed(createCopyRegion(builder, loc, recipe.getCopyRegion(), varType,
                              bounds))) {
    recipe.erase();
    return std::nullopt;
  }

  // Only create destroy region if the allocation needs deallocation
  if (needsFree) {
    // Extract the allocated value from the init block's yield operation
    auto yieldOp =
        cast<acc::YieldOp>(recipe.getInitRegion().front().getTerminator());
    Value allocRes = yieldOp.getOperand(0);

    if (failed(createDestroyRegion(builder, loc, recipe.getDestroyRegion(),
                                   varType, allocRes, bounds))) {
      recipe.erase();
      return std::nullopt;
    }
  }

  return recipe;
}

//===----------------------------------------------------------------------===//
// ReductionRecipeOp
//===----------------------------------------------------------------------===//

LogicalResult acc::ReductionRecipeOp::verifyRegions() {
  if (failed(verifyInitLikeSingleArgRegion(*this, getInitRegion(), "reduction",
                                           "init", getType(),
                                           /*verifyYield=*/false)))
    return failure();

  if (getCombinerRegion().empty())
    return emitOpError() << "expects non-empty combiner region";

  Block &reductionBlock = getCombinerRegion().front();
  if (reductionBlock.getNumArguments() < 2 ||
      reductionBlock.getArgument(0).getType() != getType() ||
      reductionBlock.getArgument(1).getType() != getType())
    return emitOpError() << "expects combiner region with the first two "
                         << "arguments of the reduction type";

  for (YieldOp yieldOp : getCombinerRegion().getOps<YieldOp>()) {
    if (yieldOp.getOperands().size() != 1 ||
        yieldOp.getOperands().getTypes()[0] != getType())
      return emitOpError() << "expects combiner region to yield a value "
                              "of the reduction type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Custom parser and printer verifier for private clause
//===----------------------------------------------------------------------===//

static ParseResult parseSymOperandList(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<Type> &types, mlir::ArrayAttr &symbols) {
  llvm::SmallVector<SymbolRefAttr> attributes;
  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parser.parseAttribute(attributes.emplace_back()) ||
            parser.parseArrow() ||
            parser.parseOperand(operands.emplace_back()) ||
            parser.parseColonType(types.emplace_back()))
          return failure();
        return success();
      })))
    return failure();
  llvm::SmallVector<mlir::Attribute> arrayAttr(attributes.begin(),
                                               attributes.end());
  symbols = ArrayAttr::get(parser.getContext(), arrayAttr);
  return success();
}

static void printSymOperandList(mlir::OpAsmPrinter &p, mlir::Operation *op,
                                mlir::OperandRange operands,
                                mlir::TypeRange types,
                                std::optional<mlir::ArrayAttr> attributes) {
  llvm::interleaveComma(llvm::zip(*attributes, operands), p, [&](auto it) {
    p << std::get<0>(it) << " -> " << std::get<1>(it) << " : "
      << std::get<1>(it).getType();
  });
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

/// Check dataOperands for acc.parallel, acc.serial and acc.kernels.
template <typename Op>
static LogicalResult checkDataOperands(Op op,
                                       const mlir::ValueRange &operands) {
  for (mlir::Value operand : operands)
    if (!mlir::isa<acc::AttachOp, acc::CopyinOp, acc::CopyoutOp, acc::CreateOp,
                   acc::DeleteOp, acc::DetachOp, acc::DevicePtrOp,
                   acc::GetDevicePtrOp, acc::NoCreateOp, acc::PresentOp>(
            operand.getDefiningOp()))
      return op.emitError(
          "expect data entry/exit operation or acc.getdeviceptr "
          "as defining op");
  return success();
}

template <typename Op>
static LogicalResult
checkSymOperandList(Operation *op, std::optional<mlir::ArrayAttr> attributes,
                    mlir::OperandRange operands, llvm::StringRef operandName,
                    llvm::StringRef symbolName, bool checkOperandType = true) {
  if (!operands.empty()) {
    if (!attributes || attributes->size() != operands.size())
      return op->emitOpError()
             << "expected as many " << symbolName << " symbol reference as "
             << operandName << " operands";
  } else {
    if (attributes)
      return op->emitOpError()
             << "unexpected " << symbolName << " symbol reference";
    return success();
  }

  llvm::DenseSet<Value> set;
  for (auto args : llvm::zip(operands, *attributes)) {
    mlir::Value operand = std::get<0>(args);

    if (!set.insert(operand).second)
      return op->emitOpError()
             << operandName << " operand appears more than once";

    mlir::Type varType = operand.getType();
    auto symbolRef = llvm::cast<SymbolRefAttr>(std::get<1>(args));
    auto decl = SymbolTable::lookupNearestSymbolFrom<Op>(op, symbolRef);
    if (!decl)
      return op->emitOpError()
             << "expected symbol reference " << symbolRef << " to point to a "
             << operandName << " declaration";

    if (checkOperandType && decl.getType() && decl.getType() != varType)
      return op->emitOpError() << "expected " << operandName << " (" << varType
                               << ") to be the same type as " << operandName
                               << " declaration (" << decl.getType() << ")";
  }

  return success();
}

unsigned ParallelOp::getNumDataOperands() {
  return getReductionOperands().size() + getPrivateOperands().size() +
         getFirstprivateOperands().size() + getDataClauseOperands().size();
}

Value ParallelOp::getDataOperand(unsigned i) {
  unsigned numOptional = getAsyncOperands().size();
  numOptional += getNumGangs().size();
  numOptional += getNumWorkers().size();
  numOptional += getVectorLength().size();
  numOptional += getIfCond() ? 1 : 0;
  numOptional += getSelfCond() ? 1 : 0;
  return getOperand(getWaitOperands().size() + numOptional + i);
}

template <typename Op>
static LogicalResult verifyDeviceTypeCountMatch(Op op, OperandRange operands,
                                                ArrayAttr deviceTypes,
                                                llvm::StringRef keyword) {
  if (!operands.empty() && deviceTypes.getValue().size() != operands.size())
    return op.emitOpError() << keyword << " operands count must match "
                            << keyword << " device_type count";
  return success();
}

template <typename Op>
static LogicalResult verifyDeviceTypeAndSegmentCountMatch(
    Op op, OperandRange operands, DenseI32ArrayAttr segments,
    ArrayAttr deviceTypes, llvm::StringRef keyword, int32_t maxInSegment = 0) {
  std::size_t numOperandsInSegments = 0;
  std::size_t nbOfSegments = 0;

  if (segments) {
    for (auto segCount : segments.asArrayRef()) {
      if (maxInSegment != 0 && segCount > maxInSegment)
        return op.emitOpError() << keyword << " expects a maximum of "
                                << maxInSegment << " values per segment";
      numOperandsInSegments += segCount;
      ++nbOfSegments;
    }
  }

  if ((numOperandsInSegments != operands.size()) ||
      (!deviceTypes && !operands.empty()))
    return op.emitOpError()
           << keyword << " operand count does not match count in segments";
  if (deviceTypes && deviceTypes.getValue().size() != nbOfSegments)
    return op.emitOpError()
           << keyword << " segment count does not match device_type count";
  return success();
}

LogicalResult acc::ParallelOp::verify() {
  if (failed(checkSymOperandList<mlir::acc::PrivateRecipeOp>(
          *this, getPrivatizationRecipes(), getPrivateOperands(), "private",
          "privatizations", /*checkOperandType=*/false)))
    return failure();
  if (failed(checkSymOperandList<mlir::acc::FirstprivateRecipeOp>(
          *this, getFirstprivatizationRecipes(), getFirstprivateOperands(),
          "firstprivate", "firstprivatizations", /*checkOperandType=*/false)))
    return failure();
  if (failed(checkSymOperandList<mlir::acc::ReductionRecipeOp>(
          *this, getReductionRecipes(), getReductionOperands(), "reduction",
          "reductions", false)))
    return failure();

  if (failed(verifyDeviceTypeAndSegmentCountMatch(
          *this, getNumGangs(), getNumGangsSegmentsAttr(),
          getNumGangsDeviceTypeAttr(), "num_gangs", 3)))
    return failure();

  if (failed(verifyDeviceTypeAndSegmentCountMatch(
          *this, getWaitOperands(), getWaitOperandsSegmentsAttr(),
          getWaitOperandsDeviceTypeAttr(), "wait")))
    return failure();

  if (failed(verifyDeviceTypeCountMatch(*this, getNumWorkers(),
                                        getNumWorkersDeviceTypeAttr(),
                                        "num_workers")))
    return failure();

  if (failed(verifyDeviceTypeCountMatch(*this, getVectorLength(),
                                        getVectorLengthDeviceTypeAttr(),
                                        "vector_length")))
    return failure();

  if (failed(verifyDeviceTypeCountMatch(*this, getAsyncOperands(),
                                        getAsyncOperandsDeviceTypeAttr(),
                                        "async")))
    return failure();

  if (failed(checkWaitAndAsyncConflict<acc::ParallelOp>(*this)))
    return failure();

  return checkDataOperands<acc::ParallelOp>(*this, getDataClauseOperands());
}

static mlir::Value
getValueInDeviceTypeSegment(std::optional<mlir::ArrayAttr> arrayAttr,
                            mlir::Operation::operand_range range,
                            mlir::acc::DeviceType deviceType) {
  if (!arrayAttr)
    return {};
  if (auto pos = findSegment(*arrayAttr, deviceType))
    return range[*pos];
  return {};
}

bool acc::ParallelOp::hasAsyncOnly() {
  return hasAsyncOnly(mlir::acc::DeviceType::None);
}

bool acc::ParallelOp::hasAsyncOnly(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getAsyncOnly(), deviceType);
}

mlir::Value acc::ParallelOp::getAsyncValue() {
  return getAsyncValue(mlir::acc::DeviceType::None);
}

mlir::Value acc::ParallelOp::getAsyncValue(mlir::acc::DeviceType deviceType) {
  return getValueInDeviceTypeSegment(getAsyncOperandsDeviceType(),
                                     getAsyncOperands(), deviceType);
}

mlir::Value acc::ParallelOp::getNumWorkersValue() {
  return getNumWorkersValue(mlir::acc::DeviceType::None);
}

mlir::Value
acc::ParallelOp::getNumWorkersValue(mlir::acc::DeviceType deviceType) {
  return getValueInDeviceTypeSegment(getNumWorkersDeviceType(), getNumWorkers(),
                                     deviceType);
}

mlir::Value acc::ParallelOp::getVectorLengthValue() {
  return getVectorLengthValue(mlir::acc::DeviceType::None);
}

mlir::Value
acc::ParallelOp::getVectorLengthValue(mlir::acc::DeviceType deviceType) {
  return getValueInDeviceTypeSegment(getVectorLengthDeviceType(),
                                     getVectorLength(), deviceType);
}

mlir::Operation::operand_range ParallelOp::getNumGangsValues() {
  return getNumGangsValues(mlir::acc::DeviceType::None);
}

mlir::Operation::operand_range
ParallelOp::getNumGangsValues(mlir::acc::DeviceType deviceType) {
  return getValuesFromSegments(getNumGangsDeviceType(), getNumGangs(),
                               getNumGangsSegments(), deviceType);
}

bool acc::ParallelOp::hasWaitOnly() {
  return hasWaitOnly(mlir::acc::DeviceType::None);
}

bool acc::ParallelOp::hasWaitOnly(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getWaitOnly(), deviceType);
}

mlir::Operation::operand_range ParallelOp::getWaitValues() {
  return getWaitValues(mlir::acc::DeviceType::None);
}

mlir::Operation::operand_range
ParallelOp::getWaitValues(mlir::acc::DeviceType deviceType) {
  return getWaitValuesWithoutDevnum(
      getWaitOperandsDeviceType(), getWaitOperands(), getWaitOperandsSegments(),
      getHasWaitDevnum(), deviceType);
}

mlir::Value ParallelOp::getWaitDevnum() {
  return getWaitDevnum(mlir::acc::DeviceType::None);
}

mlir::Value ParallelOp::getWaitDevnum(mlir::acc::DeviceType deviceType) {
  return getWaitDevnumValue(getWaitOperandsDeviceType(), getWaitOperands(),
                            getWaitOperandsSegments(), getHasWaitDevnum(),
                            deviceType);
}

void ParallelOp::build(mlir::OpBuilder &odsBuilder,
                       mlir::OperationState &odsState,
                       mlir::ValueRange numGangs, mlir::ValueRange numWorkers,
                       mlir::ValueRange vectorLength,
                       mlir::ValueRange asyncOperands,
                       mlir::ValueRange waitOperands, mlir::Value ifCond,
                       mlir::Value selfCond, mlir::ValueRange reductionOperands,
                       mlir::ValueRange gangPrivateOperands,
                       mlir::ValueRange gangFirstPrivateOperands,
                       mlir::ValueRange dataClauseOperands) {

  ParallelOp::build(
      odsBuilder, odsState, asyncOperands, /*asyncOperandsDeviceType=*/nullptr,
      /*asyncOnly=*/nullptr, waitOperands, /*waitOperandsSegments=*/nullptr,
      /*waitOperandsDeviceType=*/nullptr, /*hasWaitDevnum=*/nullptr,
      /*waitOnly=*/nullptr, numGangs, /*numGangsSegments=*/nullptr,
      /*numGangsDeviceType=*/nullptr, numWorkers,
      /*numWorkersDeviceType=*/nullptr, vectorLength,
      /*vectorLengthDeviceType=*/nullptr, ifCond, selfCond,
      /*selfAttr=*/nullptr, reductionOperands, /*reductionRecipes=*/nullptr,
      gangPrivateOperands, /*privatizations=*/nullptr, gangFirstPrivateOperands,
      /*firstprivatizations=*/nullptr, dataClauseOperands,
      /*defaultAttr=*/nullptr, /*combined=*/nullptr);
}

void acc::ParallelOp::addNumWorkersOperand(
    MLIRContext *context, mlir::Value newValue,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setNumWorkersDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getNumWorkersDeviceTypeAttr(), effectiveDeviceTypes, newValue,
      getNumWorkersMutable()));
}
void acc::ParallelOp::addVectorLengthOperand(
    MLIRContext *context, mlir::Value newValue,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setVectorLengthDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getVectorLengthDeviceTypeAttr(), effectiveDeviceTypes, newValue,
      getVectorLengthMutable()));
}

void acc::ParallelOp::addAsyncOnly(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setAsyncOnlyAttr(addDeviceTypeAffectedOperandHelper(
      context, getAsyncOnlyAttr(), effectiveDeviceTypes));
}

void acc::ParallelOp::addAsyncOperand(
    MLIRContext *context, mlir::Value newValue,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setAsyncOperandsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getAsyncOperandsDeviceTypeAttr(), effectiveDeviceTypes, newValue,
      getAsyncOperandsMutable()));
}

void acc::ParallelOp::addNumGangsOperands(
    MLIRContext *context, mlir::ValueRange newValues,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  llvm::SmallVector<int32_t> segments;
  if (getNumGangsSegments())
    llvm::copy(*getNumGangsSegments(), std::back_inserter(segments));

  setNumGangsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getNumGangsDeviceTypeAttr(), effectiveDeviceTypes, newValues,
      getNumGangsMutable(), segments));

  setNumGangsSegments(segments);
}
void acc::ParallelOp::addWaitOnly(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setWaitOnlyAttr(addDeviceTypeAffectedOperandHelper(context, getWaitOnlyAttr(),
                                                     effectiveDeviceTypes));
}
void acc::ParallelOp::addWaitOperands(
    MLIRContext *context, bool hasDevnum, mlir::ValueRange newValues,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {

  llvm::SmallVector<int32_t> segments;
  if (getWaitOperandsSegments())
    llvm::copy(*getWaitOperandsSegments(), std::back_inserter(segments));

  setWaitOperandsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getWaitOperandsDeviceTypeAttr(), effectiveDeviceTypes, newValues,
      getWaitOperandsMutable(), segments));
  setWaitOperandsSegments(segments);

  llvm::SmallVector<mlir::Attribute> hasDevnums;
  if (getHasWaitDevnumAttr())
    llvm::copy(getHasWaitDevnumAttr(), std::back_inserter(hasDevnums));
  hasDevnums.insert(
      hasDevnums.end(),
      std::max(effectiveDeviceTypes.size(), static_cast<size_t>(1)),
      mlir::BoolAttr::get(context, hasDevnum));
  setHasWaitDevnumAttr(mlir::ArrayAttr::get(context, hasDevnums));
}

void acc::ParallelOp::addPrivatization(MLIRContext *context,
                                       mlir::acc::PrivateOp op,
                                       mlir::acc::PrivateRecipeOp recipe) {
  getPrivateOperandsMutable().append(op.getResult());

  llvm::SmallVector<mlir::Attribute> recipes;

  if (getPrivatizationRecipesAttr())
    llvm::copy(getPrivatizationRecipesAttr(), std::back_inserter(recipes));

  recipes.push_back(
      mlir::SymbolRefAttr::get(context, recipe.getSymName().str()));
  setPrivatizationRecipesAttr(mlir::ArrayAttr::get(context, recipes));
}

void acc::ParallelOp::addFirstPrivatization(
    MLIRContext *context, mlir::acc::FirstprivateOp op,
    mlir::acc::FirstprivateRecipeOp recipe) {
  getFirstprivateOperandsMutable().append(op.getResult());

  llvm::SmallVector<mlir::Attribute> recipes;

  if (getFirstprivatizationRecipesAttr())
    llvm::copy(getFirstprivatizationRecipesAttr(), std::back_inserter(recipes));

  recipes.push_back(
      mlir::SymbolRefAttr::get(context, recipe.getSymName().str()));
  setFirstprivatizationRecipesAttr(mlir::ArrayAttr::get(context, recipes));
}

void acc::ParallelOp::addReduction(MLIRContext *context,
                                   mlir::acc::ReductionOp op,
                                   mlir::acc::ReductionRecipeOp recipe) {
  getReductionOperandsMutable().append(op.getResult());

  llvm::SmallVector<mlir::Attribute> recipes;

  if (getReductionRecipesAttr())
    llvm::copy(getReductionRecipesAttr(), std::back_inserter(recipes));

  recipes.push_back(
      mlir::SymbolRefAttr::get(context, recipe.getSymName().str()));
  setReductionRecipesAttr(mlir::ArrayAttr::get(context, recipes));
}

static ParseResult parseNumGangs(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<Type> &types, mlir::ArrayAttr &deviceTypes,
    mlir::DenseI32ArrayAttr &segments) {
  llvm::SmallVector<DeviceTypeAttr> attributes;
  llvm::SmallVector<int32_t> seg;

  do {
    if (failed(parser.parseLBrace()))
      return failure();

    int32_t crtOperandsSize = operands.size();
    if (failed(parser.parseCommaSeparatedList(
            mlir::AsmParser::Delimiter::None, [&]() {
              if (parser.parseOperand(operands.emplace_back()) ||
                  parser.parseColonType(types.emplace_back()))
                return failure();
              return success();
            })))
      return failure();
    seg.push_back(operands.size() - crtOperandsSize);

    if (failed(parser.parseRBrace()))
      return failure();

    if (succeeded(parser.parseOptionalLSquare())) {
      if (parser.parseAttribute(attributes.emplace_back()) ||
          parser.parseRSquare())
        return failure();
    } else {
      attributes.push_back(mlir::acc::DeviceTypeAttr::get(
          parser.getContext(), mlir::acc::DeviceType::None));
    }
  } while (succeeded(parser.parseOptionalComma()));

  llvm::SmallVector<mlir::Attribute> arrayAttr(attributes.begin(),
                                               attributes.end());
  deviceTypes = ArrayAttr::get(parser.getContext(), arrayAttr);
  segments = DenseI32ArrayAttr::get(parser.getContext(), seg);

  return success();
}

static void printSingleDeviceType(mlir::OpAsmPrinter &p, mlir::Attribute attr) {
  auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(attr);
  if (deviceTypeAttr.getValue() != mlir::acc::DeviceType::None)
    p << " [" << attr << "]";
}

static void printNumGangs(mlir::OpAsmPrinter &p, mlir::Operation *op,
                          mlir::OperandRange operands, mlir::TypeRange types,
                          std::optional<mlir::ArrayAttr> deviceTypes,
                          std::optional<mlir::DenseI32ArrayAttr> segments) {
  unsigned opIdx = 0;
  llvm::interleaveComma(llvm::enumerate(*deviceTypes), p, [&](auto it) {
    p << "{";
    llvm::interleaveComma(
        llvm::seq<int32_t>(0, (*segments)[it.index()]), p, [&](auto it) {
          p << operands[opIdx] << " : " << operands[opIdx].getType();
          ++opIdx;
        });
    p << "}";
    printSingleDeviceType(p, it.value());
  });
}

static ParseResult parseDeviceTypeOperandsWithSegment(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<Type> &types, mlir::ArrayAttr &deviceTypes,
    mlir::DenseI32ArrayAttr &segments) {
  llvm::SmallVector<DeviceTypeAttr> attributes;
  llvm::SmallVector<int32_t> seg;

  do {
    if (failed(parser.parseLBrace()))
      return failure();

    int32_t crtOperandsSize = operands.size();

    if (failed(parser.parseCommaSeparatedList(
            mlir::AsmParser::Delimiter::None, [&]() {
              if (parser.parseOperand(operands.emplace_back()) ||
                  parser.parseColonType(types.emplace_back()))
                return failure();
              return success();
            })))
      return failure();

    seg.push_back(operands.size() - crtOperandsSize);

    if (failed(parser.parseRBrace()))
      return failure();

    if (succeeded(parser.parseOptionalLSquare())) {
      if (parser.parseAttribute(attributes.emplace_back()) ||
          parser.parseRSquare())
        return failure();
    } else {
      attributes.push_back(mlir::acc::DeviceTypeAttr::get(
          parser.getContext(), mlir::acc::DeviceType::None));
    }
  } while (succeeded(parser.parseOptionalComma()));

  llvm::SmallVector<mlir::Attribute> arrayAttr(attributes.begin(),
                                               attributes.end());
  deviceTypes = ArrayAttr::get(parser.getContext(), arrayAttr);
  segments = DenseI32ArrayAttr::get(parser.getContext(), seg);

  return success();
}

static void printDeviceTypeOperandsWithSegment(
    mlir::OpAsmPrinter &p, mlir::Operation *op, mlir::OperandRange operands,
    mlir::TypeRange types, std::optional<mlir::ArrayAttr> deviceTypes,
    std::optional<mlir::DenseI32ArrayAttr> segments) {
  unsigned opIdx = 0;
  llvm::interleaveComma(llvm::enumerate(*deviceTypes), p, [&](auto it) {
    p << "{";
    llvm::interleaveComma(
        llvm::seq<int32_t>(0, (*segments)[it.index()]), p, [&](auto it) {
          p << operands[opIdx] << " : " << operands[opIdx].getType();
          ++opIdx;
        });
    p << "}";
    printSingleDeviceType(p, it.value());
  });
}

static ParseResult parseWaitClause(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<Type> &types, mlir::ArrayAttr &deviceTypes,
    mlir::DenseI32ArrayAttr &segments, mlir::ArrayAttr &hasDevNum,
    mlir::ArrayAttr &keywordOnly) {
  llvm::SmallVector<mlir::Attribute> deviceTypeAttrs, keywordAttrs, devnum;
  llvm::SmallVector<int32_t> seg;

  bool needCommaBeforeOperands = false;

  // Keyword only
  if (failed(parser.parseOptionalLParen())) {
    keywordAttrs.push_back(mlir::acc::DeviceTypeAttr::get(
        parser.getContext(), mlir::acc::DeviceType::None));
    keywordOnly = ArrayAttr::get(parser.getContext(), keywordAttrs);
    return success();
  }

  // Parse keyword only attributes
  if (succeeded(parser.parseOptionalLSquare())) {
    if (failed(parser.parseCommaSeparatedList([&]() {
          if (parser.parseAttribute(keywordAttrs.emplace_back()))
            return failure();
          return success();
        })))
      return failure();
    if (parser.parseRSquare())
      return failure();
    needCommaBeforeOperands = true;
  }

  if (needCommaBeforeOperands && failed(parser.parseComma()))
    return failure();

  do {
    if (failed(parser.parseLBrace()))
      return failure();

    int32_t crtOperandsSize = operands.size();

    if (succeeded(parser.parseOptionalKeyword("devnum"))) {
      if (failed(parser.parseColon()))
        return failure();
      devnum.push_back(BoolAttr::get(parser.getContext(), true));
    } else {
      devnum.push_back(BoolAttr::get(parser.getContext(), false));
    }

    if (failed(parser.parseCommaSeparatedList(
            mlir::AsmParser::Delimiter::None, [&]() {
              if (parser.parseOperand(operands.emplace_back()) ||
                  parser.parseColonType(types.emplace_back()))
                return failure();
              return success();
            })))
      return failure();

    seg.push_back(operands.size() - crtOperandsSize);

    if (failed(parser.parseRBrace()))
      return failure();

    if (succeeded(parser.parseOptionalLSquare())) {
      if (parser.parseAttribute(deviceTypeAttrs.emplace_back()) ||
          parser.parseRSquare())
        return failure();
    } else {
      deviceTypeAttrs.push_back(mlir::acc::DeviceTypeAttr::get(
          parser.getContext(), mlir::acc::DeviceType::None));
    }
  } while (succeeded(parser.parseOptionalComma()));

  if (failed(parser.parseRParen()))
    return failure();

  deviceTypes = ArrayAttr::get(parser.getContext(), deviceTypeAttrs);
  keywordOnly = ArrayAttr::get(parser.getContext(), keywordAttrs);
  segments = DenseI32ArrayAttr::get(parser.getContext(), seg);
  hasDevNum = ArrayAttr::get(parser.getContext(), devnum);

  return success();
}

static bool hasOnlyDeviceTypeNone(std::optional<mlir::ArrayAttr> attrs) {
  if (!hasDeviceTypeValues(attrs))
    return false;
  if (attrs->size() != 1)
    return false;
  if (auto deviceTypeAttr =
          mlir::dyn_cast<mlir::acc::DeviceTypeAttr>((*attrs)[0]))
    return deviceTypeAttr.getValue() == mlir::acc::DeviceType::None;
  return false;
}

static void printWaitClause(mlir::OpAsmPrinter &p, mlir::Operation *op,
                            mlir::OperandRange operands, mlir::TypeRange types,
                            std::optional<mlir::ArrayAttr> deviceTypes,
                            std::optional<mlir::DenseI32ArrayAttr> segments,
                            std::optional<mlir::ArrayAttr> hasDevNum,
                            std::optional<mlir::ArrayAttr> keywordOnly) {

  if (operands.begin() == operands.end() && hasOnlyDeviceTypeNone(keywordOnly))
    return;

  p << "(";

  printDeviceTypes(p, keywordOnly);
  if (hasDeviceTypeValues(keywordOnly) && hasDeviceTypeValues(deviceTypes))
    p << ", ";

  if (hasDeviceTypeValues(deviceTypes)) {
    unsigned opIdx = 0;
    llvm::interleaveComma(llvm::enumerate(*deviceTypes), p, [&](auto it) {
      p << "{";
      auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>((*hasDevNum)[it.index()]);
      if (boolAttr && boolAttr.getValue())
        p << "devnum: ";
      llvm::interleaveComma(
          llvm::seq<int32_t>(0, (*segments)[it.index()]), p, [&](auto it) {
            p << operands[opIdx] << " : " << operands[opIdx].getType();
            ++opIdx;
          });
      p << "}";
      printSingleDeviceType(p, it.value());
    });
  }

  p << ")";
}

static ParseResult parseDeviceTypeOperands(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<Type> &types, mlir::ArrayAttr &deviceTypes) {
  llvm::SmallVector<DeviceTypeAttr> attributes;
  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parser.parseOperand(operands.emplace_back()) ||
            parser.parseColonType(types.emplace_back()))
          return failure();
        if (succeeded(parser.parseOptionalLSquare())) {
          if (parser.parseAttribute(attributes.emplace_back()) ||
              parser.parseRSquare())
            return failure();
        } else {
          attributes.push_back(mlir::acc::DeviceTypeAttr::get(
              parser.getContext(), mlir::acc::DeviceType::None));
        }
        return success();
      })))
    return failure();
  llvm::SmallVector<mlir::Attribute> arrayAttr(attributes.begin(),
                                               attributes.end());
  deviceTypes = ArrayAttr::get(parser.getContext(), arrayAttr);
  return success();
}

static void
printDeviceTypeOperands(mlir::OpAsmPrinter &p, mlir::Operation *op,
                        mlir::OperandRange operands, mlir::TypeRange types,
                        std::optional<mlir::ArrayAttr> deviceTypes) {
  if (!hasDeviceTypeValues(deviceTypes))
    return;
  llvm::interleaveComma(llvm::zip(*deviceTypes, operands), p, [&](auto it) {
    p << std::get<1>(it) << " : " << std::get<1>(it).getType();
    printSingleDeviceType(p, std::get<0>(it));
  });
}

static ParseResult parseDeviceTypeOperandsWithKeywordOnly(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<Type> &types, mlir::ArrayAttr &deviceTypes,
    mlir::ArrayAttr &keywordOnlyDeviceType) {

  llvm::SmallVector<mlir::Attribute> keywordOnlyDeviceTypeAttributes;
  bool needCommaBeforeOperands = false;

  if (failed(parser.parseOptionalLParen())) {
    // Keyword only
    keywordOnlyDeviceTypeAttributes.push_back(mlir::acc::DeviceTypeAttr::get(
        parser.getContext(), mlir::acc::DeviceType::None));
    keywordOnlyDeviceType =
        ArrayAttr::get(parser.getContext(), keywordOnlyDeviceTypeAttributes);
    return success();
  }

  // Parse keyword only attributes
  if (succeeded(parser.parseOptionalLSquare())) {
    // Parse keyword only attributes
    if (failed(parser.parseCommaSeparatedList([&]() {
          if (parser.parseAttribute(
                  keywordOnlyDeviceTypeAttributes.emplace_back()))
            return failure();
          return success();
        })))
      return failure();
    if (parser.parseRSquare())
      return failure();
    needCommaBeforeOperands = true;
  }

  if (needCommaBeforeOperands && failed(parser.parseComma()))
    return failure();

  llvm::SmallVector<DeviceTypeAttr> attributes;
  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parser.parseOperand(operands.emplace_back()) ||
            parser.parseColonType(types.emplace_back()))
          return failure();
        if (succeeded(parser.parseOptionalLSquare())) {
          if (parser.parseAttribute(attributes.emplace_back()) ||
              parser.parseRSquare())
            return failure();
        } else {
          attributes.push_back(mlir::acc::DeviceTypeAttr::get(
              parser.getContext(), mlir::acc::DeviceType::None));
        }
        return success();
      })))
    return failure();

  if (failed(parser.parseRParen()))
    return failure();

  llvm::SmallVector<mlir::Attribute> arrayAttr(attributes.begin(),
                                               attributes.end());
  deviceTypes = ArrayAttr::get(parser.getContext(), arrayAttr);
  return success();
}

static void printDeviceTypeOperandsWithKeywordOnly(
    mlir::OpAsmPrinter &p, mlir::Operation *op, mlir::OperandRange operands,
    mlir::TypeRange types, std::optional<mlir::ArrayAttr> deviceTypes,
    std::optional<mlir::ArrayAttr> keywordOnlyDeviceTypes) {

  if (operands.begin() == operands.end() &&
      hasOnlyDeviceTypeNone(keywordOnlyDeviceTypes)) {
    return;
  }

  p << "(";
  printDeviceTypes(p, keywordOnlyDeviceTypes);
  if (hasDeviceTypeValues(keywordOnlyDeviceTypes) &&
      hasDeviceTypeValues(deviceTypes))
    p << ", ";
  printDeviceTypeOperands(p, op, operands, types, deviceTypes);
  p << ")";
}

static ParseResult parseOperandWithKeywordOnly(
    mlir::OpAsmParser &parser,
    std::optional<OpAsmParser::UnresolvedOperand> &operand,
    mlir::Type &operandType, mlir::UnitAttr &attr) {
  // Keyword only
  if (failed(parser.parseOptionalLParen())) {
    attr = mlir::UnitAttr::get(parser.getContext());
    return success();
  }

  OpAsmParser::UnresolvedOperand op;
  if (failed(parser.parseOperand(op)))
    return failure();
  operand = op;
  if (failed(parser.parseColon()))
    return failure();
  if (failed(parser.parseType(operandType)))
    return failure();
  if (failed(parser.parseRParen()))
    return failure();

  return success();
}

static void printOperandWithKeywordOnly(mlir::OpAsmPrinter &p,
                                        mlir::Operation *op,
                                        std::optional<mlir::Value> operand,
                                        mlir::Type operandType,
                                        mlir::UnitAttr attr) {
  if (attr)
    return;

  p << "(";
  p.printOperand(*operand);
  p << " : ";
  p.printType(operandType);
  p << ")";
}

static ParseResult parseOperandsWithKeywordOnly(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<Type> &types, mlir::UnitAttr &attr) {
  // Keyword only
  if (failed(parser.parseOptionalLParen())) {
    attr = mlir::UnitAttr::get(parser.getContext());
    return success();
  }

  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parser.parseOperand(operands.emplace_back()))
          return failure();
        return success();
      })))
    return failure();
  if (failed(parser.parseColon()))
    return failure();
  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parser.parseType(types.emplace_back()))
          return failure();
        return success();
      })))
    return failure();
  if (failed(parser.parseRParen()))
    return failure();

  return success();
}

static void printOperandsWithKeywordOnly(mlir::OpAsmPrinter &p,
                                         mlir::Operation *op,
                                         mlir::OperandRange operands,
                                         mlir::TypeRange types,
                                         mlir::UnitAttr attr) {
  if (attr)
    return;

  p << "(";
  llvm::interleaveComma(operands, p, [&](auto it) { p << it; });
  p << " : ";
  llvm::interleaveComma(types, p, [&](auto it) { p << it; });
  p << ")";
}

static ParseResult
parseCombinedConstructsLoop(mlir::OpAsmParser &parser,
                            mlir::acc::CombinedConstructsTypeAttr &attr) {
  if (succeeded(parser.parseOptionalKeyword("kernels"))) {
    attr = mlir::acc::CombinedConstructsTypeAttr::get(
        parser.getContext(), mlir::acc::CombinedConstructsType::KernelsLoop);
  } else if (succeeded(parser.parseOptionalKeyword("parallel"))) {
    attr = mlir::acc::CombinedConstructsTypeAttr::get(
        parser.getContext(), mlir::acc::CombinedConstructsType::ParallelLoop);
  } else if (succeeded(parser.parseOptionalKeyword("serial"))) {
    attr = mlir::acc::CombinedConstructsTypeAttr::get(
        parser.getContext(), mlir::acc::CombinedConstructsType::SerialLoop);
  } else {
    parser.emitError(parser.getCurrentLocation(),
                     "expected compute construct name");
    return failure();
  }
  return success();
}

static void
printCombinedConstructsLoop(mlir::OpAsmPrinter &p, mlir::Operation *op,
                            mlir::acc::CombinedConstructsTypeAttr attr) {
  if (attr) {
    switch (attr.getValue()) {
    case mlir::acc::CombinedConstructsType::KernelsLoop:
      p << "kernels";
      break;
    case mlir::acc::CombinedConstructsType::ParallelLoop:
      p << "parallel";
      break;
    case mlir::acc::CombinedConstructsType::SerialLoop:
      p << "serial";
      break;
    };
  }
}

//===----------------------------------------------------------------------===//
// SerialOp
//===----------------------------------------------------------------------===//

unsigned SerialOp::getNumDataOperands() {
  return getReductionOperands().size() + getPrivateOperands().size() +
         getFirstprivateOperands().size() + getDataClauseOperands().size();
}

Value SerialOp::getDataOperand(unsigned i) {
  unsigned numOptional = getAsyncOperands().size();
  numOptional += getIfCond() ? 1 : 0;
  numOptional += getSelfCond() ? 1 : 0;
  return getOperand(getWaitOperands().size() + numOptional + i);
}

bool acc::SerialOp::hasAsyncOnly() {
  return hasAsyncOnly(mlir::acc::DeviceType::None);
}

bool acc::SerialOp::hasAsyncOnly(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getAsyncOnly(), deviceType);
}

mlir::Value acc::SerialOp::getAsyncValue() {
  return getAsyncValue(mlir::acc::DeviceType::None);
}

mlir::Value acc::SerialOp::getAsyncValue(mlir::acc::DeviceType deviceType) {
  return getValueInDeviceTypeSegment(getAsyncOperandsDeviceType(),
                                     getAsyncOperands(), deviceType);
}

bool acc::SerialOp::hasWaitOnly() {
  return hasWaitOnly(mlir::acc::DeviceType::None);
}

bool acc::SerialOp::hasWaitOnly(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getWaitOnly(), deviceType);
}

mlir::Operation::operand_range SerialOp::getWaitValues() {
  return getWaitValues(mlir::acc::DeviceType::None);
}

mlir::Operation::operand_range
SerialOp::getWaitValues(mlir::acc::DeviceType deviceType) {
  return getWaitValuesWithoutDevnum(
      getWaitOperandsDeviceType(), getWaitOperands(), getWaitOperandsSegments(),
      getHasWaitDevnum(), deviceType);
}

mlir::Value SerialOp::getWaitDevnum() {
  return getWaitDevnum(mlir::acc::DeviceType::None);
}

mlir::Value SerialOp::getWaitDevnum(mlir::acc::DeviceType deviceType) {
  return getWaitDevnumValue(getWaitOperandsDeviceType(), getWaitOperands(),
                            getWaitOperandsSegments(), getHasWaitDevnum(),
                            deviceType);
}

LogicalResult acc::SerialOp::verify() {
  if (failed(checkSymOperandList<mlir::acc::PrivateRecipeOp>(
          *this, getPrivatizationRecipes(), getPrivateOperands(), "private",
          "privatizations", /*checkOperandType=*/false)))
    return failure();
  if (failed(checkSymOperandList<mlir::acc::FirstprivateRecipeOp>(
          *this, getFirstprivatizationRecipes(), getFirstprivateOperands(),
          "firstprivate", "firstprivatizations", /*checkOperandType=*/false)))
    return failure();
  if (failed(checkSymOperandList<mlir::acc::ReductionRecipeOp>(
          *this, getReductionRecipes(), getReductionOperands(), "reduction",
          "reductions", false)))
    return failure();

  if (failed(verifyDeviceTypeAndSegmentCountMatch(
          *this, getWaitOperands(), getWaitOperandsSegmentsAttr(),
          getWaitOperandsDeviceTypeAttr(), "wait")))
    return failure();

  if (failed(verifyDeviceTypeCountMatch(*this, getAsyncOperands(),
                                        getAsyncOperandsDeviceTypeAttr(),
                                        "async")))
    return failure();

  if (failed(checkWaitAndAsyncConflict<acc::SerialOp>(*this)))
    return failure();

  return checkDataOperands<acc::SerialOp>(*this, getDataClauseOperands());
}

void acc::SerialOp::addAsyncOnly(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setAsyncOnlyAttr(addDeviceTypeAffectedOperandHelper(
      context, getAsyncOnlyAttr(), effectiveDeviceTypes));
}

void acc::SerialOp::addAsyncOperand(
    MLIRContext *context, mlir::Value newValue,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setAsyncOperandsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getAsyncOperandsDeviceTypeAttr(), effectiveDeviceTypes, newValue,
      getAsyncOperandsMutable()));
}

void acc::SerialOp::addWaitOnly(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setWaitOnlyAttr(addDeviceTypeAffectedOperandHelper(context, getWaitOnlyAttr(),
                                                     effectiveDeviceTypes));
}
void acc::SerialOp::addWaitOperands(
    MLIRContext *context, bool hasDevnum, mlir::ValueRange newValues,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {

  llvm::SmallVector<int32_t> segments;
  if (getWaitOperandsSegments())
    llvm::copy(*getWaitOperandsSegments(), std::back_inserter(segments));

  setWaitOperandsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getWaitOperandsDeviceTypeAttr(), effectiveDeviceTypes, newValues,
      getWaitOperandsMutable(), segments));
  setWaitOperandsSegments(segments);

  llvm::SmallVector<mlir::Attribute> hasDevnums;
  if (getHasWaitDevnumAttr())
    llvm::copy(getHasWaitDevnumAttr(), std::back_inserter(hasDevnums));
  hasDevnums.insert(
      hasDevnums.end(),
      std::max(effectiveDeviceTypes.size(), static_cast<size_t>(1)),
      mlir::BoolAttr::get(context, hasDevnum));
  setHasWaitDevnumAttr(mlir::ArrayAttr::get(context, hasDevnums));
}

void acc::SerialOp::addPrivatization(MLIRContext *context,
                                     mlir::acc::PrivateOp op,
                                     mlir::acc::PrivateRecipeOp recipe) {
  getPrivateOperandsMutable().append(op.getResult());

  llvm::SmallVector<mlir::Attribute> recipes;

  if (getPrivatizationRecipesAttr())
    llvm::copy(getPrivatizationRecipesAttr(), std::back_inserter(recipes));

  recipes.push_back(
      mlir::SymbolRefAttr::get(context, recipe.getSymName().str()));
  setPrivatizationRecipesAttr(mlir::ArrayAttr::get(context, recipes));
}

void acc::SerialOp::addFirstPrivatization(
    MLIRContext *context, mlir::acc::FirstprivateOp op,
    mlir::acc::FirstprivateRecipeOp recipe) {
  getFirstprivateOperandsMutable().append(op.getResult());

  llvm::SmallVector<mlir::Attribute> recipes;

  if (getFirstprivatizationRecipesAttr())
    llvm::copy(getFirstprivatizationRecipesAttr(), std::back_inserter(recipes));

  recipes.push_back(
      mlir::SymbolRefAttr::get(context, recipe.getSymName().str()));
  setFirstprivatizationRecipesAttr(mlir::ArrayAttr::get(context, recipes));
}

void acc::SerialOp::addReduction(MLIRContext *context,
                                 mlir::acc::ReductionOp op,
                                 mlir::acc::ReductionRecipeOp recipe) {
  getReductionOperandsMutable().append(op.getResult());

  llvm::SmallVector<mlir::Attribute> recipes;

  if (getReductionRecipesAttr())
    llvm::copy(getReductionRecipesAttr(), std::back_inserter(recipes));

  recipes.push_back(
      mlir::SymbolRefAttr::get(context, recipe.getSymName().str()));
  setReductionRecipesAttr(mlir::ArrayAttr::get(context, recipes));
}

//===----------------------------------------------------------------------===//
// KernelsOp
//===----------------------------------------------------------------------===//

unsigned KernelsOp::getNumDataOperands() {
  return getDataClauseOperands().size();
}

Value KernelsOp::getDataOperand(unsigned i) {
  unsigned numOptional = getAsyncOperands().size();
  numOptional += getWaitOperands().size();
  numOptional += getNumGangs().size();
  numOptional += getNumWorkers().size();
  numOptional += getVectorLength().size();
  numOptional += getIfCond() ? 1 : 0;
  numOptional += getSelfCond() ? 1 : 0;
  return getOperand(numOptional + i);
}

bool acc::KernelsOp::hasAsyncOnly() {
  return hasAsyncOnly(mlir::acc::DeviceType::None);
}

bool acc::KernelsOp::hasAsyncOnly(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getAsyncOnly(), deviceType);
}

mlir::Value acc::KernelsOp::getAsyncValue() {
  return getAsyncValue(mlir::acc::DeviceType::None);
}

mlir::Value acc::KernelsOp::getAsyncValue(mlir::acc::DeviceType deviceType) {
  return getValueInDeviceTypeSegment(getAsyncOperandsDeviceType(),
                                     getAsyncOperands(), deviceType);
}

mlir::Value acc::KernelsOp::getNumWorkersValue() {
  return getNumWorkersValue(mlir::acc::DeviceType::None);
}

mlir::Value
acc::KernelsOp::getNumWorkersValue(mlir::acc::DeviceType deviceType) {
  return getValueInDeviceTypeSegment(getNumWorkersDeviceType(), getNumWorkers(),
                                     deviceType);
}

mlir::Value acc::KernelsOp::getVectorLengthValue() {
  return getVectorLengthValue(mlir::acc::DeviceType::None);
}

mlir::Value
acc::KernelsOp::getVectorLengthValue(mlir::acc::DeviceType deviceType) {
  return getValueInDeviceTypeSegment(getVectorLengthDeviceType(),
                                     getVectorLength(), deviceType);
}

mlir::Operation::operand_range KernelsOp::getNumGangsValues() {
  return getNumGangsValues(mlir::acc::DeviceType::None);
}

mlir::Operation::operand_range
KernelsOp::getNumGangsValues(mlir::acc::DeviceType deviceType) {
  return getValuesFromSegments(getNumGangsDeviceType(), getNumGangs(),
                               getNumGangsSegments(), deviceType);
}

bool acc::KernelsOp::hasWaitOnly() {
  return hasWaitOnly(mlir::acc::DeviceType::None);
}

bool acc::KernelsOp::hasWaitOnly(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getWaitOnly(), deviceType);
}

mlir::Operation::operand_range KernelsOp::getWaitValues() {
  return getWaitValues(mlir::acc::DeviceType::None);
}

mlir::Operation::operand_range
KernelsOp::getWaitValues(mlir::acc::DeviceType deviceType) {
  return getWaitValuesWithoutDevnum(
      getWaitOperandsDeviceType(), getWaitOperands(), getWaitOperandsSegments(),
      getHasWaitDevnum(), deviceType);
}

mlir::Value KernelsOp::getWaitDevnum() {
  return getWaitDevnum(mlir::acc::DeviceType::None);
}

mlir::Value KernelsOp::getWaitDevnum(mlir::acc::DeviceType deviceType) {
  return getWaitDevnumValue(getWaitOperandsDeviceType(), getWaitOperands(),
                            getWaitOperandsSegments(), getHasWaitDevnum(),
                            deviceType);
}

LogicalResult acc::KernelsOp::verify() {
  if (failed(verifyDeviceTypeAndSegmentCountMatch(
          *this, getNumGangs(), getNumGangsSegmentsAttr(),
          getNumGangsDeviceTypeAttr(), "num_gangs", 3)))
    return failure();

  if (failed(verifyDeviceTypeAndSegmentCountMatch(
          *this, getWaitOperands(), getWaitOperandsSegmentsAttr(),
          getWaitOperandsDeviceTypeAttr(), "wait")))
    return failure();

  if (failed(verifyDeviceTypeCountMatch(*this, getNumWorkers(),
                                        getNumWorkersDeviceTypeAttr(),
                                        "num_workers")))
    return failure();

  if (failed(verifyDeviceTypeCountMatch(*this, getVectorLength(),
                                        getVectorLengthDeviceTypeAttr(),
                                        "vector_length")))
    return failure();

  if (failed(verifyDeviceTypeCountMatch(*this, getAsyncOperands(),
                                        getAsyncOperandsDeviceTypeAttr(),
                                        "async")))
    return failure();

  if (failed(checkWaitAndAsyncConflict<acc::KernelsOp>(*this)))
    return failure();

  return checkDataOperands<acc::KernelsOp>(*this, getDataClauseOperands());
}

void acc::KernelsOp::addNumWorkersOperand(
    MLIRContext *context, mlir::Value newValue,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setNumWorkersDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getNumWorkersDeviceTypeAttr(), effectiveDeviceTypes, newValue,
      getNumWorkersMutable()));
}

void acc::KernelsOp::addVectorLengthOperand(
    MLIRContext *context, mlir::Value newValue,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setVectorLengthDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getVectorLengthDeviceTypeAttr(), effectiveDeviceTypes, newValue,
      getVectorLengthMutable()));
}
void acc::KernelsOp::addAsyncOnly(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setAsyncOnlyAttr(addDeviceTypeAffectedOperandHelper(
      context, getAsyncOnlyAttr(), effectiveDeviceTypes));
}

void acc::KernelsOp::addAsyncOperand(
    MLIRContext *context, mlir::Value newValue,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setAsyncOperandsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getAsyncOperandsDeviceTypeAttr(), effectiveDeviceTypes, newValue,
      getAsyncOperandsMutable()));
}

void acc::KernelsOp::addNumGangsOperands(
    MLIRContext *context, mlir::ValueRange newValues,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  llvm::SmallVector<int32_t> segments;
  if (getNumGangsSegmentsAttr())
    llvm::copy(*getNumGangsSegments(), std::back_inserter(segments));

  setNumGangsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getNumGangsDeviceTypeAttr(), effectiveDeviceTypes, newValues,
      getNumGangsMutable(), segments));

  setNumGangsSegments(segments);
}

void acc::KernelsOp::addWaitOnly(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setWaitOnlyAttr(addDeviceTypeAffectedOperandHelper(context, getWaitOnlyAttr(),
                                                     effectiveDeviceTypes));
}
void acc::KernelsOp::addWaitOperands(
    MLIRContext *context, bool hasDevnum, mlir::ValueRange newValues,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {

  llvm::SmallVector<int32_t> segments;
  if (getWaitOperandsSegments())
    llvm::copy(*getWaitOperandsSegments(), std::back_inserter(segments));

  setWaitOperandsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getWaitOperandsDeviceTypeAttr(), effectiveDeviceTypes, newValues,
      getWaitOperandsMutable(), segments));
  setWaitOperandsSegments(segments);

  llvm::SmallVector<mlir::Attribute> hasDevnums;
  if (getHasWaitDevnumAttr())
    llvm::copy(getHasWaitDevnumAttr(), std::back_inserter(hasDevnums));
  hasDevnums.insert(
      hasDevnums.end(),
      std::max(effectiveDeviceTypes.size(), static_cast<size_t>(1)),
      mlir::BoolAttr::get(context, hasDevnum));
  setHasWaitDevnumAttr(mlir::ArrayAttr::get(context, hasDevnums));
}

//===----------------------------------------------------------------------===//
// HostDataOp
//===----------------------------------------------------------------------===//

LogicalResult acc::HostDataOp::verify() {
  if (getDataClauseOperands().empty())
    return emitError("at least one operand must appear on the host_data "
                     "operation");

  for (mlir::Value operand : getDataClauseOperands())
    if (!mlir::isa<acc::UseDeviceOp>(operand.getDefiningOp()))
      return emitError("expect data entry operation as defining op");
  return success();
}

void acc::HostDataOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<RemoveConstantIfConditionWithRegion<HostDataOp>>(context);
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

static ParseResult parseGangValue(
    OpAsmParser &parser, llvm::StringRef keyword,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<Type> &types,
    llvm::SmallVector<GangArgTypeAttr> &attributes, GangArgTypeAttr gangArgType,
    bool &needCommaBetweenValues, bool &newValue) {
  if (succeeded(parser.parseOptionalKeyword(keyword))) {
    if (parser.parseEqual())
      return failure();
    if (parser.parseOperand(operands.emplace_back()) ||
        parser.parseColonType(types.emplace_back()))
      return failure();
    attributes.push_back(gangArgType);
    needCommaBetweenValues = true;
    newValue = true;
  }
  return success();
}

static ParseResult parseGangClause(
    OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &gangOperands,
    llvm::SmallVectorImpl<Type> &gangOperandsType, mlir::ArrayAttr &gangArgType,
    mlir::ArrayAttr &deviceType, mlir::DenseI32ArrayAttr &segments,
    mlir::ArrayAttr &gangOnlyDeviceType) {
  llvm::SmallVector<GangArgTypeAttr> gangArgTypeAttributes;
  llvm::SmallVector<mlir::Attribute> deviceTypeAttributes;
  llvm::SmallVector<mlir::Attribute> gangOnlyDeviceTypeAttributes;
  llvm::SmallVector<int32_t> seg;
  bool needCommaBetweenValues = false;
  bool needCommaBeforeOperands = false;

  if (failed(parser.parseOptionalLParen())) {
    // Gang only keyword
    gangOnlyDeviceTypeAttributes.push_back(mlir::acc::DeviceTypeAttr::get(
        parser.getContext(), mlir::acc::DeviceType::None));
    gangOnlyDeviceType =
        ArrayAttr::get(parser.getContext(), gangOnlyDeviceTypeAttributes);
    return success();
  }

  // Parse gang only attributes
  if (succeeded(parser.parseOptionalLSquare())) {
    // Parse gang only attributes
    if (failed(parser.parseCommaSeparatedList([&]() {
          if (parser.parseAttribute(
                  gangOnlyDeviceTypeAttributes.emplace_back()))
            return failure();
          return success();
        })))
      return failure();
    if (parser.parseRSquare())
      return failure();
    needCommaBeforeOperands = true;
  }

  auto argNum = mlir::acc::GangArgTypeAttr::get(parser.getContext(),
                                                mlir::acc::GangArgType::Num);
  auto argDim = mlir::acc::GangArgTypeAttr::get(parser.getContext(),
                                                mlir::acc::GangArgType::Dim);
  auto argStatic = mlir::acc::GangArgTypeAttr::get(
      parser.getContext(), mlir::acc::GangArgType::Static);

  do {
    if (needCommaBeforeOperands) {
      needCommaBeforeOperands = false;
      continue;
    }

    if (failed(parser.parseLBrace()))
      return failure();

    int32_t crtOperandsSize = gangOperands.size();
    while (true) {
      bool newValue = false;
      bool needValue = false;
      if (needCommaBetweenValues) {
        if (succeeded(parser.parseOptionalComma()))
          needValue = true; // expect a new value after comma.
        else
          break;
      }

      if (failed(parseGangValue(parser, LoopOp::getGangNumKeyword(),
                                gangOperands, gangOperandsType,
                                gangArgTypeAttributes, argNum,
                                needCommaBetweenValues, newValue)))
        return failure();
      if (failed(parseGangValue(parser, LoopOp::getGangDimKeyword(),
                                gangOperands, gangOperandsType,
                                gangArgTypeAttributes, argDim,
                                needCommaBetweenValues, newValue)))
        return failure();
      if (failed(parseGangValue(parser, LoopOp::getGangStaticKeyword(),
                                gangOperands, gangOperandsType,
                                gangArgTypeAttributes, argStatic,
                                needCommaBetweenValues, newValue)))
        return failure();

      if (!newValue && needValue) {
        parser.emitError(parser.getCurrentLocation(),
                         "new value expected after comma");
        return failure();
      }

      if (!newValue)
        break;
    }

    if (gangOperands.empty())
      return parser.emitError(
          parser.getCurrentLocation(),
          "expect at least one of num, dim or static values");

    if (failed(parser.parseRBrace()))
      return failure();

    if (succeeded(parser.parseOptionalLSquare())) {
      if (parser.parseAttribute(deviceTypeAttributes.emplace_back()) ||
          parser.parseRSquare())
        return failure();
    } else {
      deviceTypeAttributes.push_back(mlir::acc::DeviceTypeAttr::get(
          parser.getContext(), mlir::acc::DeviceType::None));
    }

    seg.push_back(gangOperands.size() - crtOperandsSize);

  } while (succeeded(parser.parseOptionalComma()));

  if (failed(parser.parseRParen()))
    return failure();

  llvm::SmallVector<mlir::Attribute> arrayAttr(gangArgTypeAttributes.begin(),
                                               gangArgTypeAttributes.end());
  gangArgType = ArrayAttr::get(parser.getContext(), arrayAttr);
  deviceType = ArrayAttr::get(parser.getContext(), deviceTypeAttributes);

  llvm::SmallVector<mlir::Attribute> gangOnlyAttr(
      gangOnlyDeviceTypeAttributes.begin(), gangOnlyDeviceTypeAttributes.end());
  gangOnlyDeviceType = ArrayAttr::get(parser.getContext(), gangOnlyAttr);

  segments = DenseI32ArrayAttr::get(parser.getContext(), seg);
  return success();
}

void printGangClause(OpAsmPrinter &p, Operation *op,
                     mlir::OperandRange operands, mlir::TypeRange types,
                     std::optional<mlir::ArrayAttr> gangArgTypes,
                     std::optional<mlir::ArrayAttr> deviceTypes,
                     std::optional<mlir::DenseI32ArrayAttr> segments,
                     std::optional<mlir::ArrayAttr> gangOnlyDeviceTypes) {

  if (operands.begin() == operands.end() &&
      hasOnlyDeviceTypeNone(gangOnlyDeviceTypes)) {
    return;
  }

  p << "(";

  printDeviceTypes(p, gangOnlyDeviceTypes);

  if (hasDeviceTypeValues(gangOnlyDeviceTypes) &&
      hasDeviceTypeValues(deviceTypes))
    p << ", ";

  if (hasDeviceTypeValues(deviceTypes)) {
    unsigned opIdx = 0;
    llvm::interleaveComma(llvm::enumerate(*deviceTypes), p, [&](auto it) {
      p << "{";
      llvm::interleaveComma(
          llvm::seq<int32_t>(0, (*segments)[it.index()]), p, [&](auto it) {
            auto gangArgTypeAttr = mlir::dyn_cast<mlir::acc::GangArgTypeAttr>(
                (*gangArgTypes)[opIdx]);
            if (gangArgTypeAttr.getValue() == mlir::acc::GangArgType::Num)
              p << LoopOp::getGangNumKeyword();
            else if (gangArgTypeAttr.getValue() == mlir::acc::GangArgType::Dim)
              p << LoopOp::getGangDimKeyword();
            else if (gangArgTypeAttr.getValue() ==
                     mlir::acc::GangArgType::Static)
              p << LoopOp::getGangStaticKeyword();
            p << "=" << operands[opIdx] << " : " << operands[opIdx].getType();
            ++opIdx;
          });
      p << "}";
      printSingleDeviceType(p, it.value());
    });
  }
  p << ")";
}

bool hasDuplicateDeviceTypes(
    std::optional<mlir::ArrayAttr> segments,
    llvm::SmallSet<mlir::acc::DeviceType, 3> &deviceTypes) {
  if (!segments)
    return false;
  for (auto attr : *segments) {
    auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(attr);
    if (!deviceTypes.insert(deviceTypeAttr.getValue()).second)
      return true;
  }
  return false;
}

/// Check for duplicates in the DeviceType array attribute.
LogicalResult checkDeviceTypes(mlir::ArrayAttr deviceTypes) {
  llvm::SmallSet<mlir::acc::DeviceType, 3> crtDeviceTypes;
  if (!deviceTypes)
    return success();
  for (auto attr : deviceTypes) {
    auto deviceTypeAttr =
        mlir::dyn_cast_or_null<mlir::acc::DeviceTypeAttr>(attr);
    if (!deviceTypeAttr)
      return failure();
    if (!crtDeviceTypes.insert(deviceTypeAttr.getValue()).second)
      return failure();
  }
  return success();
}

LogicalResult acc::LoopOp::verify() {
  if (getUpperbound().size() != getStep().size())
    return emitError() << "number of upperbounds expected to be the same as "
                          "number of steps";

  if (getUpperbound().size() != getLowerbound().size())
    return emitError() << "number of upperbounds expected to be the same as "
                          "number of lowerbounds";

  if (!getUpperbound().empty() && getInclusiveUpperbound() &&
      (getUpperbound().size() != getInclusiveUpperbound()->size()))
    return emitError() << "inclusiveUpperbound size is expected to be the same"
                       << " as upperbound size";

  // Check collapse
  if (getCollapseAttr() && !getCollapseDeviceTypeAttr())
    return emitOpError() << "collapse device_type attr must be define when"
                         << " collapse attr is present";

  if (getCollapseAttr() && getCollapseDeviceTypeAttr() &&
      getCollapseAttr().getValue().size() !=
          getCollapseDeviceTypeAttr().getValue().size())
    return emitOpError() << "collapse attribute count must match collapse"
                         << " device_type count";
  if (failed(checkDeviceTypes(getCollapseDeviceTypeAttr())))
    return emitOpError()
           << "duplicate device_type found in collapseDeviceType attribute";

  // Check gang
  if (!getGangOperands().empty()) {
    if (!getGangOperandsArgType())
      return emitOpError() << "gangOperandsArgType attribute must be defined"
                           << " when gang operands are present";

    if (getGangOperands().size() !=
        getGangOperandsArgTypeAttr().getValue().size())
      return emitOpError() << "gangOperandsArgType attribute count must match"
                           << " gangOperands count";
  }
  if (getGangAttr() && failed(checkDeviceTypes(getGangAttr())))
    return emitOpError() << "duplicate device_type found in gang attribute";

  if (failed(verifyDeviceTypeAndSegmentCountMatch(
          *this, getGangOperands(), getGangOperandsSegmentsAttr(),
          getGangOperandsDeviceTypeAttr(), "gang")))
    return failure();

  // Check worker
  if (failed(checkDeviceTypes(getWorkerAttr())))
    return emitOpError() << "duplicate device_type found in worker attribute";
  if (failed(checkDeviceTypes(getWorkerNumOperandsDeviceTypeAttr())))
    return emitOpError() << "duplicate device_type found in "
                            "workerNumOperandsDeviceType attribute";
  if (failed(verifyDeviceTypeCountMatch(*this, getWorkerNumOperands(),
                                        getWorkerNumOperandsDeviceTypeAttr(),
                                        "worker")))
    return failure();

  // Check vector
  if (failed(checkDeviceTypes(getVectorAttr())))
    return emitOpError() << "duplicate device_type found in vector attribute";
  if (failed(checkDeviceTypes(getVectorOperandsDeviceTypeAttr())))
    return emitOpError() << "duplicate device_type found in "
                            "vectorOperandsDeviceType attribute";
  if (failed(verifyDeviceTypeCountMatch(*this, getVectorOperands(),
                                        getVectorOperandsDeviceTypeAttr(),
                                        "vector")))
    return failure();

  if (failed(verifyDeviceTypeAndSegmentCountMatch(
          *this, getTileOperands(), getTileOperandsSegmentsAttr(),
          getTileOperandsDeviceTypeAttr(), "tile")))
    return failure();

  // auto, independent and seq attribute are mutually exclusive.
  llvm::SmallSet<mlir::acc::DeviceType, 3> deviceTypes;
  if (hasDuplicateDeviceTypes(getAuto_(), deviceTypes) ||
      hasDuplicateDeviceTypes(getIndependent(), deviceTypes) ||
      hasDuplicateDeviceTypes(getSeq(), deviceTypes)) {
    return emitError() << "only one of auto, independent, seq can be present "
                          "at the same time";
  }

  // Check that at least one of auto, independent, or seq is present
  // for the device-independent default clauses.
  auto hasDeviceNone = [](mlir::acc::DeviceTypeAttr attr) -> bool {
    return attr.getValue() == mlir::acc::DeviceType::None;
  };
  bool hasDefaultSeq =
      getSeqAttr()
          ? llvm::any_of(getSeqAttr().getAsRange<mlir::acc::DeviceTypeAttr>(),
                         hasDeviceNone)
          : false;
  bool hasDefaultIndependent =
      getIndependentAttr()
          ? llvm::any_of(
                getIndependentAttr().getAsRange<mlir::acc::DeviceTypeAttr>(),
                hasDeviceNone)
          : false;
  bool hasDefaultAuto =
      getAuto_Attr()
          ? llvm::any_of(getAuto_Attr().getAsRange<mlir::acc::DeviceTypeAttr>(),
                         hasDeviceNone)
          : false;
  if (!hasDefaultSeq && !hasDefaultIndependent && !hasDefaultAuto) {
    return emitError()
           << "at least one of auto, independent, seq must be present";
  }

  // Gang, worker and vector are incompatible with seq.
  if (getSeqAttr()) {
    for (auto attr : getSeqAttr()) {
      auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(attr);
      if (hasVector(deviceTypeAttr.getValue()) ||
          getVectorValue(deviceTypeAttr.getValue()) ||
          hasWorker(deviceTypeAttr.getValue()) ||
          getWorkerValue(deviceTypeAttr.getValue()) ||
          hasGang(deviceTypeAttr.getValue()) ||
          getGangValue(mlir::acc::GangArgType::Num,
                       deviceTypeAttr.getValue()) ||
          getGangValue(mlir::acc::GangArgType::Dim,
                       deviceTypeAttr.getValue()) ||
          getGangValue(mlir::acc::GangArgType::Static,
                       deviceTypeAttr.getValue()))
        return emitError() << "gang, worker or vector cannot appear with seq";
    }
  }

  if (failed(checkSymOperandList<mlir::acc::PrivateRecipeOp>(
          *this, getPrivatizationRecipes(), getPrivateOperands(), "private",
          "privatizations", false)))
    return failure();

  if (failed(checkSymOperandList<mlir::acc::FirstprivateRecipeOp>(
          *this, getFirstprivatizationRecipes(), getFirstprivateOperands(),
          "firstprivate", "firstprivatizations", /*checkOperandType=*/false)))
    return failure();

  if (failed(checkSymOperandList<mlir::acc::ReductionRecipeOp>(
          *this, getReductionRecipes(), getReductionOperands(), "reduction",
          "reductions", false)))
    return failure();

  if (getCombined().has_value() &&
      (getCombined().value() != acc::CombinedConstructsType::ParallelLoop &&
       getCombined().value() != acc::CombinedConstructsType::KernelsLoop &&
       getCombined().value() != acc::CombinedConstructsType::SerialLoop)) {
    return emitError("unexpected combined constructs attribute");
  }

  // Check non-empty body().
  if (getRegion().empty())
    return emitError("expected non-empty body.");

  // When it is container-like - it is expected to hold a loop-like operation.
  if (isContainerLike()) {
    // Obtain the maximum collapse count - we use this to check that there
    // are enough loops contained.
    uint64_t collapseCount = getCollapseValue().value_or(1);
    if (getCollapseAttr()) {
      for (auto collapseEntry : getCollapseAttr()) {
        auto intAttr = mlir::dyn_cast<IntegerAttr>(collapseEntry);
        if (intAttr.getValue().getZExtValue() > collapseCount)
          collapseCount = intAttr.getValue().getZExtValue();
      }
    }

    // We want to check that we find enough loop-like operations inside.
    // PreOrder walk allows us to walk in a breadth-first manner at each nesting
    // level.
    mlir::Operation *expectedParent = this->getOperation();
    bool foundSibling = false;
    getRegion().walk<WalkOrder::PreOrder>([&](mlir::Operation *op) {
      if (mlir::isa<mlir::LoopLikeOpInterface>(op)) {
        // This effectively checks that we are not looking at a sibling loop.
        if (op->getParentOfType<mlir::LoopLikeOpInterface>() !=
            expectedParent) {
          foundSibling = true;
          return mlir::WalkResult::interrupt();
        }

        collapseCount--;
        expectedParent = op;
      }
      // We found enough contained loops.
      if (collapseCount == 0)
        return mlir::WalkResult::interrupt();
      return mlir::WalkResult::advance();
    });

    if (foundSibling)
      return emitError("found sibling loops inside container-like acc.loop");
    if (collapseCount != 0)
      return emitError("failed to find enough loop-like operations inside "
                       "container-like acc.loop");
  }

  return success();
}

unsigned LoopOp::getNumDataOperands() {
  return getReductionOperands().size() + getPrivateOperands().size() +
         getFirstprivateOperands().size();
}

Value LoopOp::getDataOperand(unsigned i) {
  unsigned numOptional =
      getLowerbound().size() + getUpperbound().size() + getStep().size();
  numOptional += getGangOperands().size();
  numOptional += getVectorOperands().size();
  numOptional += getWorkerNumOperands().size();
  numOptional += getTileOperands().size();
  numOptional += getCacheOperands().size();
  return getOperand(numOptional + i);
}

bool LoopOp::hasAuto() { return hasAuto(mlir::acc::DeviceType::None); }

bool LoopOp::hasAuto(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getAuto_(), deviceType);
}

bool LoopOp::hasIndependent() {
  return hasIndependent(mlir::acc::DeviceType::None);
}

bool LoopOp::hasIndependent(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getIndependent(), deviceType);
}

bool LoopOp::hasSeq() { return hasSeq(mlir::acc::DeviceType::None); }

bool LoopOp::hasSeq(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getSeq(), deviceType);
}

mlir::Value LoopOp::getVectorValue() {
  return getVectorValue(mlir::acc::DeviceType::None);
}

mlir::Value LoopOp::getVectorValue(mlir::acc::DeviceType deviceType) {
  return getValueInDeviceTypeSegment(getVectorOperandsDeviceType(),
                                     getVectorOperands(), deviceType);
}

bool LoopOp::hasVector() { return hasVector(mlir::acc::DeviceType::None); }

bool LoopOp::hasVector(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getVector(), deviceType);
}

mlir::Value LoopOp::getWorkerValue() {
  return getWorkerValue(mlir::acc::DeviceType::None);
}

mlir::Value LoopOp::getWorkerValue(mlir::acc::DeviceType deviceType) {
  return getValueInDeviceTypeSegment(getWorkerNumOperandsDeviceType(),
                                     getWorkerNumOperands(), deviceType);
}

bool LoopOp::hasWorker() { return hasWorker(mlir::acc::DeviceType::None); }

bool LoopOp::hasWorker(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getWorker(), deviceType);
}

mlir::Operation::operand_range LoopOp::getTileValues() {
  return getTileValues(mlir::acc::DeviceType::None);
}

mlir::Operation::operand_range
LoopOp::getTileValues(mlir::acc::DeviceType deviceType) {
  return getValuesFromSegments(getTileOperandsDeviceType(), getTileOperands(),
                               getTileOperandsSegments(), deviceType);
}

std::optional<int64_t> LoopOp::getCollapseValue() {
  return getCollapseValue(mlir::acc::DeviceType::None);
}

std::optional<int64_t>
LoopOp::getCollapseValue(mlir::acc::DeviceType deviceType) {
  if (!getCollapseAttr())
    return std::nullopt;
  if (auto pos = findSegment(getCollapseDeviceTypeAttr(), deviceType)) {
    auto intAttr =
        mlir::dyn_cast<IntegerAttr>(getCollapseAttr().getValue()[*pos]);
    return intAttr.getValue().getZExtValue();
  }
  return std::nullopt;
}

mlir::Value LoopOp::getGangValue(mlir::acc::GangArgType gangArgType) {
  return getGangValue(gangArgType, mlir::acc::DeviceType::None);
}

mlir::Value LoopOp::getGangValue(mlir::acc::GangArgType gangArgType,
                                 mlir::acc::DeviceType deviceType) {
  if (getGangOperands().empty())
    return {};
  if (auto pos = findSegment(*getGangOperandsDeviceType(), deviceType)) {
    int32_t nbOperandsBefore = 0;
    for (unsigned i = 0; i < *pos; ++i)
      nbOperandsBefore += (*getGangOperandsSegments())[i];
    mlir::Operation::operand_range values =
        getGangOperands()
            .drop_front(nbOperandsBefore)
            .take_front((*getGangOperandsSegments())[*pos]);

    int32_t argTypeIdx = nbOperandsBefore;
    for (auto value : values) {
      auto gangArgTypeAttr = mlir::dyn_cast<mlir::acc::GangArgTypeAttr>(
          (*getGangOperandsArgType())[argTypeIdx]);
      if (gangArgTypeAttr.getValue() == gangArgType)
        return value;
      ++argTypeIdx;
    }
  }
  return {};
}

bool LoopOp::hasGang() { return hasGang(mlir::acc::DeviceType::None); }

bool LoopOp::hasGang(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getGang(), deviceType);
}

llvm::SmallVector<mlir::Region *> acc::LoopOp::getLoopRegions() {
  return {&getRegion()};
}

/// loop-control ::= `control` `(` ssa-id-and-type-list `)` `=`
/// `(` ssa-id-and-type-list `)` `to` `(` ssa-id-and-type-list `)` `step`
/// `(` ssa-id-and-type-list `)`
/// region
ParseResult
parseLoopControl(OpAsmParser &parser, Region &region,
                 SmallVectorImpl<OpAsmParser::UnresolvedOperand> &lowerbound,
                 SmallVectorImpl<Type> &lowerboundType,
                 SmallVectorImpl<OpAsmParser::UnresolvedOperand> &upperbound,
                 SmallVectorImpl<Type> &upperboundType,
                 SmallVectorImpl<OpAsmParser::UnresolvedOperand> &step,
                 SmallVectorImpl<Type> &stepType) {

  SmallVector<OpAsmParser::Argument> inductionVars;
  if (succeeded(
          parser.parseOptionalKeyword(acc::LoopOp::getControlKeyword()))) {
    if (parser.parseLParen() ||
        parser.parseArgumentList(inductionVars, OpAsmParser::Delimiter::None,
                                 /*allowType=*/true) ||
        parser.parseRParen() || parser.parseEqual() || parser.parseLParen() ||
        parser.parseOperandList(lowerbound, inductionVars.size(),
                                OpAsmParser::Delimiter::None) ||
        parser.parseColonTypeList(lowerboundType) || parser.parseRParen() ||
        parser.parseKeyword("to") || parser.parseLParen() ||
        parser.parseOperandList(upperbound, inductionVars.size(),
                                OpAsmParser::Delimiter::None) ||
        parser.parseColonTypeList(upperboundType) || parser.parseRParen() ||
        parser.parseKeyword("step") || parser.parseLParen() ||
        parser.parseOperandList(step, inductionVars.size(),
                                OpAsmParser::Delimiter::None) ||
        parser.parseColonTypeList(stepType) || parser.parseRParen())
      return failure();
  }
  return parser.parseRegion(region, inductionVars);
}

void printLoopControl(OpAsmPrinter &p, Operation *op, Region &region,
                      ValueRange lowerbound, TypeRange lowerboundType,
                      ValueRange upperbound, TypeRange upperboundType,
                      ValueRange steps, TypeRange stepType) {
  ValueRange regionArgs = region.front().getArguments();
  if (!regionArgs.empty()) {
    p << acc::LoopOp::getControlKeyword() << "(";
    llvm::interleaveComma(regionArgs, p,
                          [&p](Value v) { p << v << " : " << v.getType(); });
    p << ") = (" << lowerbound << " : " << lowerboundType << ") to ("
      << upperbound << " : " << upperboundType << ") " << " step (" << steps
      << " : " << stepType << ") ";
  }
  p.printRegion(region, /*printEntryBlockArgs=*/false);
}

void acc::LoopOp::addSeq(MLIRContext *context,
                         llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setSeqAttr(addDeviceTypeAffectedOperandHelper(context, getSeqAttr(),
                                                effectiveDeviceTypes));
}

void acc::LoopOp::addIndependent(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setIndependentAttr(addDeviceTypeAffectedOperandHelper(
      context, getIndependentAttr(), effectiveDeviceTypes));
}

void acc::LoopOp::addAuto(MLIRContext *context,
                          llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setAuto_Attr(addDeviceTypeAffectedOperandHelper(context, getAuto_Attr(),
                                                  effectiveDeviceTypes));
}

void acc::LoopOp::setCollapseForDeviceTypes(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes,
    llvm::APInt value) {
  llvm::SmallVector<mlir::Attribute> newValues;
  llvm::SmallVector<mlir::Attribute> newDeviceTypes;

  assert((getCollapseAttr() == nullptr) ==
         (getCollapseDeviceTypeAttr() == nullptr));
  assert(value.getBitWidth() == 64);

  if (getCollapseAttr()) {
    for (const auto &existing :
         llvm::zip_equal(getCollapseAttr(), getCollapseDeviceTypeAttr())) {
      newValues.push_back(std::get<0>(existing));
      newDeviceTypes.push_back(std::get<1>(existing));
    }
  }

  if (effectiveDeviceTypes.empty()) {
    // If the effective device-types list is empty, this is before there are any
    // being applied by device_type, so this should be added as a 'none'.
    newValues.push_back(
        mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), value));
    newDeviceTypes.push_back(
        acc::DeviceTypeAttr::get(context, DeviceType::None));
  } else {
    for (DeviceType dt : effectiveDeviceTypes) {
      newValues.push_back(
          mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), value));
      newDeviceTypes.push_back(acc::DeviceTypeAttr::get(context, dt));
    }
  }

  setCollapseAttr(ArrayAttr::get(context, newValues));
  setCollapseDeviceTypeAttr(ArrayAttr::get(context, newDeviceTypes));
}

void acc::LoopOp::setTileForDeviceTypes(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes,
    ValueRange values) {
  llvm::SmallVector<int32_t> segments;
  if (getTileOperandsSegments())
    llvm::copy(*getTileOperandsSegments(), std::back_inserter(segments));

  setTileOperandsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getTileOperandsDeviceTypeAttr(), effectiveDeviceTypes, values,
      getTileOperandsMutable(), segments));

  setTileOperandsSegments(segments);
}

void acc::LoopOp::addVectorOperand(
    MLIRContext *context, mlir::Value newValue,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setVectorOperandsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getVectorOperandsDeviceTypeAttr(), effectiveDeviceTypes,
      newValue, getVectorOperandsMutable()));
}

void acc::LoopOp::addEmptyVector(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setVectorAttr(addDeviceTypeAffectedOperandHelper(context, getVectorAttr(),
                                                   effectiveDeviceTypes));
}

void acc::LoopOp::addWorkerNumOperand(
    MLIRContext *context, mlir::Value newValue,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setWorkerNumOperandsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getWorkerNumOperandsDeviceTypeAttr(), effectiveDeviceTypes,
      newValue, getWorkerNumOperandsMutable()));
}

void acc::LoopOp::addEmptyWorker(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setWorkerAttr(addDeviceTypeAffectedOperandHelper(context, getWorkerAttr(),
                                                   effectiveDeviceTypes));
}

void acc::LoopOp::addEmptyGang(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setGangAttr(addDeviceTypeAffectedOperandHelper(context, getGangAttr(),
                                                 effectiveDeviceTypes));
}

bool acc::LoopOp::hasParallelismFlag(DeviceType dt) {
  auto hasDevice = [=](DeviceTypeAttr attr) -> bool {
    return attr.getValue() == dt;
  };
  auto testFromArr = [=](ArrayAttr arr) -> bool {
    return llvm::any_of(arr.getAsRange<DeviceTypeAttr>(), hasDevice);
  };

  if (ArrayAttr arr = getSeqAttr(); arr && testFromArr(arr))
    return true;
  if (ArrayAttr arr = getIndependentAttr(); arr && testFromArr(arr))
    return true;
  if (ArrayAttr arr = getAuto_Attr(); arr && testFromArr(arr))
    return true;

  return false;
}

bool acc::LoopOp::hasDefaultGangWorkerVector() {
  return hasVector() || getVectorValue() || hasWorker() || getWorkerValue() ||
         hasGang() || getGangValue(GangArgType::Num) ||
         getGangValue(GangArgType::Dim) || getGangValue(GangArgType::Static);
}

acc::LoopParMode
acc::LoopOp::getDefaultOrDeviceTypeParallelism(DeviceType deviceType) {
  if (hasSeq(deviceType))
    return LoopParMode::loop_seq;
  if (hasAuto(deviceType))
    return LoopParMode::loop_auto;
  if (hasIndependent(deviceType))
    return LoopParMode::loop_independent;
  if (hasSeq())
    return LoopParMode::loop_seq;
  if (hasAuto())
    return LoopParMode::loop_auto;
  assert(hasIndependent() &&
         "loop must have default auto, seq, or independent");
  return LoopParMode::loop_independent;
}

void acc::LoopOp::addGangOperands(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes,
    llvm::ArrayRef<GangArgType> argTypes, mlir::ValueRange values) {
  llvm::SmallVector<int32_t> segments;
  if (std::optional<ArrayRef<int32_t>> existingSegments =
          getGangOperandsSegments())
    llvm::copy(*existingSegments, std::back_inserter(segments));

  unsigned beforeCount = segments.size();

  setGangOperandsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getGangOperandsDeviceTypeAttr(), effectiveDeviceTypes, values,
      getGangOperandsMutable(), segments));

  setGangOperandsSegments(segments);

  // This is a bit of extra work to make sure we update the 'types' correctly by
  // adding to the types collection the correct number of times. We could
  // potentially add something similar to the
  // addDeviceTypeAffectedOperandHelper, but it seems that would be pretty
  // excessive for a one-off case.
  unsigned numAdded = segments.size() - beforeCount;

  if (numAdded > 0) {
    llvm::SmallVector<mlir::Attribute> gangTypes;
    if (getGangOperandsArgTypeAttr())
      llvm::copy(getGangOperandsArgTypeAttr(), std::back_inserter(gangTypes));

    for (auto i : llvm::index_range(0u, numAdded)) {
      llvm::transform(argTypes, std::back_inserter(gangTypes),
                      [=](mlir::acc::GangArgType gangTy) {
                        return mlir::acc::GangArgTypeAttr::get(context, gangTy);
                      });
      (void)i;
    }

    setGangOperandsArgTypeAttr(mlir::ArrayAttr::get(context, gangTypes));
  }
}

void acc::LoopOp::addPrivatization(MLIRContext *context,
                                   mlir::acc::PrivateOp op,
                                   mlir::acc::PrivateRecipeOp recipe) {
  getPrivateOperandsMutable().append(op.getResult());

  llvm::SmallVector<mlir::Attribute> recipes;

  if (getPrivatizationRecipesAttr())
    llvm::copy(getPrivatizationRecipesAttr(), std::back_inserter(recipes));

  recipes.push_back(
      mlir::SymbolRefAttr::get(context, recipe.getSymName().str()));
  setPrivatizationRecipesAttr(mlir::ArrayAttr::get(context, recipes));
}

void acc::LoopOp::addFirstPrivatization(
    MLIRContext *context, mlir::acc::FirstprivateOp op,
    mlir::acc::FirstprivateRecipeOp recipe) {
  getFirstprivateOperandsMutable().append(op.getResult());

  llvm::SmallVector<mlir::Attribute> recipes;

  if (getFirstprivatizationRecipesAttr())
    llvm::copy(getFirstprivatizationRecipesAttr(), std::back_inserter(recipes));

  recipes.push_back(
      mlir::SymbolRefAttr::get(context, recipe.getSymName().str()));
  setFirstprivatizationRecipesAttr(mlir::ArrayAttr::get(context, recipes));
}

void acc::LoopOp::addReduction(MLIRContext *context, mlir::acc::ReductionOp op,
                               mlir::acc::ReductionRecipeOp recipe) {
  getReductionOperandsMutable().append(op.getResult());

  llvm::SmallVector<mlir::Attribute> recipes;

  if (getReductionRecipesAttr())
    llvm::copy(getReductionRecipesAttr(), std::back_inserter(recipes));

  recipes.push_back(
      mlir::SymbolRefAttr::get(context, recipe.getSymName().str()));
  setReductionRecipesAttr(mlir::ArrayAttr::get(context, recipes));
}

//===----------------------------------------------------------------------===//
// DataOp
//===----------------------------------------------------------------------===//

LogicalResult acc::DataOp::verify() {
  // 2.6.5. Data Construct restriction
  // At least one copy, copyin, copyout, create, no_create, present, deviceptr,
  // attach, or default clause must appear on a data construct.
  if (getOperands().empty() && !getDefaultAttr())
    return emitError("at least one operand or the default attribute "
                     "must appear on the data operation");

  for (mlir::Value operand : getDataClauseOperands())
    if (isa<BlockArgument>(operand) ||
        !mlir::isa<acc::AttachOp, acc::CopyinOp, acc::CopyoutOp, acc::CreateOp,
                   acc::DeleteOp, acc::DetachOp, acc::DevicePtrOp,
                   acc::GetDevicePtrOp, acc::NoCreateOp, acc::PresentOp>(
            operand.getDefiningOp()))
      return emitError("expect data entry/exit operation or acc.getdeviceptr "
                       "as defining op");

  if (failed(checkWaitAndAsyncConflict<acc::DataOp>(*this)))
    return failure();

  return success();
}

unsigned DataOp::getNumDataOperands() { return getDataClauseOperands().size(); }

Value DataOp::getDataOperand(unsigned i) {
  unsigned numOptional = getIfCond() ? 1 : 0;
  numOptional += getAsyncOperands().size() ? 1 : 0;
  numOptional += getWaitOperands().size();
  return getOperand(numOptional + i);
}

bool acc::DataOp::hasAsyncOnly() {
  return hasAsyncOnly(mlir::acc::DeviceType::None);
}

bool acc::DataOp::hasAsyncOnly(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getAsyncOnly(), deviceType);
}

mlir::Value DataOp::getAsyncValue() {
  return getAsyncValue(mlir::acc::DeviceType::None);
}

mlir::Value DataOp::getAsyncValue(mlir::acc::DeviceType deviceType) {
  return getValueInDeviceTypeSegment(getAsyncOperandsDeviceType(),
                                     getAsyncOperands(), deviceType);
}

bool DataOp::hasWaitOnly() { return hasWaitOnly(mlir::acc::DeviceType::None); }

bool DataOp::hasWaitOnly(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getWaitOnly(), deviceType);
}

mlir::Operation::operand_range DataOp::getWaitValues() {
  return getWaitValues(mlir::acc::DeviceType::None);
}

mlir::Operation::operand_range
DataOp::getWaitValues(mlir::acc::DeviceType deviceType) {
  return getWaitValuesWithoutDevnum(
      getWaitOperandsDeviceType(), getWaitOperands(), getWaitOperandsSegments(),
      getHasWaitDevnum(), deviceType);
}

mlir::Value DataOp::getWaitDevnum() {
  return getWaitDevnum(mlir::acc::DeviceType::None);
}

mlir::Value DataOp::getWaitDevnum(mlir::acc::DeviceType deviceType) {
  return getWaitDevnumValue(getWaitOperandsDeviceType(), getWaitOperands(),
                            getWaitOperandsSegments(), getHasWaitDevnum(),
                            deviceType);
}

void acc::DataOp::addAsyncOnly(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setAsyncOnlyAttr(addDeviceTypeAffectedOperandHelper(
      context, getAsyncOnlyAttr(), effectiveDeviceTypes));
}

void acc::DataOp::addAsyncOperand(
    MLIRContext *context, mlir::Value newValue,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setAsyncOperandsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getAsyncOperandsDeviceTypeAttr(), effectiveDeviceTypes, newValue,
      getAsyncOperandsMutable()));
}

void acc::DataOp::addWaitOnly(MLIRContext *context,
                              llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setWaitOnlyAttr(addDeviceTypeAffectedOperandHelper(context, getWaitOnlyAttr(),
                                                     effectiveDeviceTypes));
}

void acc::DataOp::addWaitOperands(
    MLIRContext *context, bool hasDevnum, mlir::ValueRange newValues,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {

  llvm::SmallVector<int32_t> segments;
  if (getWaitOperandsSegments())
    llvm::copy(*getWaitOperandsSegments(), std::back_inserter(segments));

  setWaitOperandsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getWaitOperandsDeviceTypeAttr(), effectiveDeviceTypes, newValues,
      getWaitOperandsMutable(), segments));
  setWaitOperandsSegments(segments);

  llvm::SmallVector<mlir::Attribute> hasDevnums;
  if (getHasWaitDevnumAttr())
    llvm::copy(getHasWaitDevnumAttr(), std::back_inserter(hasDevnums));
  hasDevnums.insert(
      hasDevnums.end(),
      std::max(effectiveDeviceTypes.size(), static_cast<size_t>(1)),
      mlir::BoolAttr::get(context, hasDevnum));
  setHasWaitDevnumAttr(mlir::ArrayAttr::get(context, hasDevnums));
}

//===----------------------------------------------------------------------===//
// ExitDataOp
//===----------------------------------------------------------------------===//

LogicalResult acc::ExitDataOp::verify() {
  // 2.6.6. Data Exit Directive restriction
  // At least one copyout, delete, or detach clause must appear on an exit data
  // directive.
  if (getDataClauseOperands().empty())
    return emitError("at least one operand must be present in dataOperands on "
                     "the exit data operation");

  // The async attribute represent the async clause without value. Therefore the
  // attribute and operand cannot appear at the same time.
  if (getAsyncOperand() && getAsync())
    return emitError("async attribute cannot appear with asyncOperand");

  // The wait attribute represent the wait clause without values. Therefore the
  // attribute and operands cannot appear at the same time.
  if (!getWaitOperands().empty() && getWait())
    return emitError("wait attribute cannot appear with waitOperands");

  if (getWaitDevnum() && getWaitOperands().empty())
    return emitError("wait_devnum cannot appear without waitOperands");

  return success();
}

unsigned ExitDataOp::getNumDataOperands() {
  return getDataClauseOperands().size();
}

Value ExitDataOp::getDataOperand(unsigned i) {
  unsigned numOptional = getIfCond() ? 1 : 0;
  numOptional += getAsyncOperand() ? 1 : 0;
  numOptional += getWaitDevnum() ? 1 : 0;
  return getOperand(getWaitOperands().size() + numOptional + i);
}

void ExitDataOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add<RemoveConstantIfCondition<ExitDataOp>>(context);
}

void ExitDataOp::addAsyncOnly(MLIRContext *context,
                              llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  assert(effectiveDeviceTypes.empty());
  assert(!getAsyncAttr());
  assert(!getAsyncOperand());

  setAsyncAttr(mlir::UnitAttr::get(context));
}

void ExitDataOp::addAsyncOperand(
    MLIRContext *context, mlir::Value newValue,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  assert(effectiveDeviceTypes.empty());
  assert(!getAsyncAttr());
  assert(!getAsyncOperand());

  getAsyncOperandMutable().append(newValue);
}

void ExitDataOp::addWaitOnly(MLIRContext *context,
                             llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  assert(effectiveDeviceTypes.empty());
  assert(!getWaitAttr());
  assert(getWaitOperands().empty());
  assert(!getWaitDevnum());

  setWaitAttr(mlir::UnitAttr::get(context));
}

void ExitDataOp::addWaitOperands(
    MLIRContext *context, bool hasDevnum, mlir::ValueRange newValues,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  assert(effectiveDeviceTypes.empty());
  assert(!getWaitAttr());
  assert(getWaitOperands().empty());
  assert(!getWaitDevnum());

  // if hasDevnum, the first value is the devnum. The 'rest' go into the
  // operands list.
  if (hasDevnum) {
    getWaitDevnumMutable().append(newValues.front());
    newValues = newValues.drop_front();
  }

  getWaitOperandsMutable().append(newValues);
}

//===----------------------------------------------------------------------===//
// EnterDataOp
//===----------------------------------------------------------------------===//

LogicalResult acc::EnterDataOp::verify() {
  // 2.6.6. Data Enter Directive restriction
  // At least one copyin, create, or attach clause must appear on an enter data
  // directive.
  if (getDataClauseOperands().empty())
    return emitError("at least one operand must be present in dataOperands on "
                     "the enter data operation");

  // The async attribute represent the async clause without value. Therefore the
  // attribute and operand cannot appear at the same time.
  if (getAsyncOperand() && getAsync())
    return emitError("async attribute cannot appear with asyncOperand");

  // The wait attribute represent the wait clause without values. Therefore the
  // attribute and operands cannot appear at the same time.
  if (!getWaitOperands().empty() && getWait())
    return emitError("wait attribute cannot appear with waitOperands");

  if (getWaitDevnum() && getWaitOperands().empty())
    return emitError("wait_devnum cannot appear without waitOperands");

  for (mlir::Value operand : getDataClauseOperands())
    if (!mlir::isa<acc::AttachOp, acc::CreateOp, acc::CopyinOp>(
            operand.getDefiningOp()))
      return emitError("expect data entry operation as defining op");

  return success();
}

unsigned EnterDataOp::getNumDataOperands() {
  return getDataClauseOperands().size();
}

Value EnterDataOp::getDataOperand(unsigned i) {
  unsigned numOptional = getIfCond() ? 1 : 0;
  numOptional += getAsyncOperand() ? 1 : 0;
  numOptional += getWaitDevnum() ? 1 : 0;
  return getOperand(getWaitOperands().size() + numOptional + i);
}

void EnterDataOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<RemoveConstantIfCondition<EnterDataOp>>(context);
}

void EnterDataOp::addAsyncOnly(
    MLIRContext *context, llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  assert(effectiveDeviceTypes.empty());
  assert(!getAsyncAttr());
  assert(!getAsyncOperand());

  setAsyncAttr(mlir::UnitAttr::get(context));
}

void EnterDataOp::addAsyncOperand(
    MLIRContext *context, mlir::Value newValue,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  assert(effectiveDeviceTypes.empty());
  assert(!getAsyncAttr());
  assert(!getAsyncOperand());

  getAsyncOperandMutable().append(newValue);
}

void EnterDataOp::addWaitOnly(MLIRContext *context,
                              llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  assert(effectiveDeviceTypes.empty());
  assert(!getWaitAttr());
  assert(getWaitOperands().empty());
  assert(!getWaitDevnum());

  setWaitAttr(mlir::UnitAttr::get(context));
}

void EnterDataOp::addWaitOperands(
    MLIRContext *context, bool hasDevnum, mlir::ValueRange newValues,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  assert(effectiveDeviceTypes.empty());
  assert(!getWaitAttr());
  assert(getWaitOperands().empty());
  assert(!getWaitDevnum());

  // if hasDevnum, the first value is the devnum. The 'rest' go into the
  // operands list.
  if (hasDevnum) {
    getWaitDevnumMutable().append(newValues.front());
    newValues = newValues.drop_front();
  }

  getWaitOperandsMutable().append(newValues);
}

//===----------------------------------------------------------------------===//
// AtomicReadOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicReadOp::verify() { return verifyCommon(); }

//===----------------------------------------------------------------------===//
// AtomicWriteOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicWriteOp::verify() { return verifyCommon(); }

//===----------------------------------------------------------------------===//
// AtomicUpdateOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicUpdateOp::canonicalize(AtomicUpdateOp op,
                                           PatternRewriter &rewriter) {
  if (op.isNoOp()) {
    rewriter.eraseOp(op);
    return success();
  }

  if (Value writeVal = op.getWriteOpVal()) {
    rewriter.replaceOpWithNewOp<AtomicWriteOp>(op, op.getX(), writeVal,
                                               op.getIfCond());
    return success();
  }

  return failure();
}

LogicalResult AtomicUpdateOp::verify() { return verifyCommon(); }

LogicalResult AtomicUpdateOp::verifyRegions() { return verifyRegionsCommon(); }

//===----------------------------------------------------------------------===//
// AtomicCaptureOp
//===----------------------------------------------------------------------===//

AtomicReadOp AtomicCaptureOp::getAtomicReadOp() {
  if (auto op = dyn_cast<AtomicReadOp>(getFirstOp()))
    return op;
  return dyn_cast<AtomicReadOp>(getSecondOp());
}

AtomicWriteOp AtomicCaptureOp::getAtomicWriteOp() {
  if (auto op = dyn_cast<AtomicWriteOp>(getFirstOp()))
    return op;
  return dyn_cast<AtomicWriteOp>(getSecondOp());
}

AtomicUpdateOp AtomicCaptureOp::getAtomicUpdateOp() {
  if (auto op = dyn_cast<AtomicUpdateOp>(getFirstOp()))
    return op;
  return dyn_cast<AtomicUpdateOp>(getSecondOp());
}

LogicalResult AtomicCaptureOp::verifyRegions() { return verifyRegionsCommon(); }

//===----------------------------------------------------------------------===//
// DeclareEnterOp
//===----------------------------------------------------------------------===//

template <typename Op>
static LogicalResult
checkDeclareOperands(Op &op, const mlir::ValueRange &operands,
                     bool requireAtLeastOneOperand = true) {
  if (operands.empty() && requireAtLeastOneOperand)
    return emitError(
        op->getLoc(),
        "at least one operand must appear on the declare operation");

  for (mlir::Value operand : operands) {
    if (isa<BlockArgument>(operand) ||
        !mlir::isa<acc::CopyinOp, acc::CopyoutOp, acc::CreateOp,
                   acc::DevicePtrOp, acc::GetDevicePtrOp, acc::PresentOp,
                   acc::DeclareDeviceResidentOp, acc::DeclareLinkOp>(
            operand.getDefiningOp()))
      return op.emitError(
          "expect valid declare data entry operation or acc.getdeviceptr "
          "as defining op");

    mlir::Value var{getVar(operand.getDefiningOp())};
    assert(var && "declare operands can only be data entry operations which "
                  "must have var");
    (void)var;
    std::optional<mlir::acc::DataClause> dataClauseOptional{
        getDataClause(operand.getDefiningOp())};
    assert(dataClauseOptional.has_value() &&
           "declare operands can only be data entry operations which must have "
           "dataClause");
    (void)dataClauseOptional;
  }

  return success();
}

LogicalResult acc::DeclareEnterOp::verify() {
  return checkDeclareOperands(*this, this->getDataClauseOperands());
}

//===----------------------------------------------------------------------===//
// DeclareExitOp
//===----------------------------------------------------------------------===//

LogicalResult acc::DeclareExitOp::verify() {
  if (getToken())
    return checkDeclareOperands(*this, this->getDataClauseOperands(),
                                /*requireAtLeastOneOperand=*/false);
  return checkDeclareOperands(*this, this->getDataClauseOperands());
}

//===----------------------------------------------------------------------===//
// DeclareOp
//===----------------------------------------------------------------------===//

LogicalResult acc::DeclareOp::verify() {
  return checkDeclareOperands(*this, this->getDataClauseOperands());
}

//===----------------------------------------------------------------------===//
// RoutineOp
//===----------------------------------------------------------------------===//

static unsigned getParallelismForDeviceType(acc::RoutineOp op,
                                            acc::DeviceType dtype) {
  unsigned parallelism = 0;
  parallelism += (op.hasGang(dtype) || op.getGangDimValue(dtype)) ? 1 : 0;
  parallelism += op.hasWorker(dtype) ? 1 : 0;
  parallelism += op.hasVector(dtype) ? 1 : 0;
  parallelism += op.hasSeq(dtype) ? 1 : 0;
  return parallelism;
}

LogicalResult acc::RoutineOp::verify() {
  unsigned baseParallelism =
      getParallelismForDeviceType(*this, acc::DeviceType::None);

  if (baseParallelism > 1)
    return emitError() << "only one of `gang`, `worker`, `vector`, `seq` can "
                          "be present at the same time";

  for (uint32_t dtypeInt = 0; dtypeInt != acc::getMaxEnumValForDeviceType();
       ++dtypeInt) {
    auto dtype = static_cast<acc::DeviceType>(dtypeInt);
    if (dtype == acc::DeviceType::None)
      continue;
    unsigned parallelism = getParallelismForDeviceType(*this, dtype);

    if (parallelism > 1 || (baseParallelism == 1 && parallelism == 1))
      return emitError() << "only one of `gang`, `worker`, `vector`, `seq` can "
                            "be present at the same time";
  }

  return success();
}

static ParseResult parseBindName(OpAsmParser &parser,
                                 mlir::ArrayAttr &bindIdName,
                                 mlir::ArrayAttr &bindStrName,
                                 mlir::ArrayAttr &deviceIdTypes,
                                 mlir::ArrayAttr &deviceStrTypes) {
  llvm::SmallVector<mlir::Attribute> bindIdNameAttrs;
  llvm::SmallVector<mlir::Attribute> bindStrNameAttrs;
  llvm::SmallVector<mlir::Attribute> deviceIdTypeAttrs;
  llvm::SmallVector<mlir::Attribute> deviceStrTypeAttrs;

  if (failed(parser.parseCommaSeparatedList([&]() {
        mlir::Attribute newAttr;
        bool isSymbolRefAttr;
        auto parseResult = parser.parseAttribute(newAttr);
        if (auto symbolRefAttr = dyn_cast<mlir::SymbolRefAttr>(newAttr)) {
          bindIdNameAttrs.push_back(symbolRefAttr);
          isSymbolRefAttr = true;
        } else if (auto stringAttr = dyn_cast<mlir::StringAttr>(newAttr)) {
          bindStrNameAttrs.push_back(stringAttr);
          isSymbolRefAttr = false;
        }
        if (parseResult)
          return failure();
        if (failed(parser.parseOptionalLSquare())) {
          if (isSymbolRefAttr) {
            deviceIdTypeAttrs.push_back(mlir::acc::DeviceTypeAttr::get(
                parser.getContext(), mlir::acc::DeviceType::None));
          } else {
            deviceStrTypeAttrs.push_back(mlir::acc::DeviceTypeAttr::get(
                parser.getContext(), mlir::acc::DeviceType::None));
          }
        } else {
          if (isSymbolRefAttr) {
            if (parser.parseAttribute(deviceIdTypeAttrs.emplace_back()) ||
                parser.parseRSquare())
              return failure();
          } else {
            if (parser.parseAttribute(deviceStrTypeAttrs.emplace_back()) ||
                parser.parseRSquare())
              return failure();
          }
        }
        return success();
      })))
    return failure();

  bindIdName = ArrayAttr::get(parser.getContext(), bindIdNameAttrs);
  bindStrName = ArrayAttr::get(parser.getContext(), bindStrNameAttrs);
  deviceIdTypes = ArrayAttr::get(parser.getContext(), deviceIdTypeAttrs);
  deviceStrTypes = ArrayAttr::get(parser.getContext(), deviceStrTypeAttrs);

  return success();
}

static void printBindName(mlir::OpAsmPrinter &p, mlir::Operation *op,
                          std::optional<mlir::ArrayAttr> bindIdName,
                          std::optional<mlir::ArrayAttr> bindStrName,
                          std::optional<mlir::ArrayAttr> deviceIdTypes,
                          std::optional<mlir::ArrayAttr> deviceStrTypes) {
  // Create combined vectors for all bind names and device types
  llvm::SmallVector<mlir::Attribute> allBindNames;
  llvm::SmallVector<mlir::Attribute> allDeviceTypes;

  // Append bindIdName and deviceIdTypes
  if (hasDeviceTypeValues(deviceIdTypes)) {
    allBindNames.append(bindIdName->begin(), bindIdName->end());
    allDeviceTypes.append(deviceIdTypes->begin(), deviceIdTypes->end());
  }

  // Append bindStrName and deviceStrTypes
  if (hasDeviceTypeValues(deviceStrTypes)) {
    allBindNames.append(bindStrName->begin(), bindStrName->end());
    allDeviceTypes.append(deviceStrTypes->begin(), deviceStrTypes->end());
  }

  // Print the combined sequence
  if (!allBindNames.empty())
    llvm::interleaveComma(llvm::zip(allBindNames, allDeviceTypes), p,
                          [&](const auto &pair) {
                            p << std::get<0>(pair);
                            printSingleDeviceType(p, std::get<1>(pair));
                          });
}

static ParseResult parseRoutineGangClause(OpAsmParser &parser,
                                          mlir::ArrayAttr &gang,
                                          mlir::ArrayAttr &gangDim,
                                          mlir::ArrayAttr &gangDimDeviceTypes) {

  llvm::SmallVector<mlir::Attribute> gangAttrs, gangDimAttrs,
      gangDimDeviceTypeAttrs;
  bool needCommaBeforeOperands = false;

  // Gang keyword only
  if (failed(parser.parseOptionalLParen())) {
    gangAttrs.push_back(mlir::acc::DeviceTypeAttr::get(
        parser.getContext(), mlir::acc::DeviceType::None));
    gang = ArrayAttr::get(parser.getContext(), gangAttrs);
    return success();
  }

  // Parse keyword only attributes
  if (succeeded(parser.parseOptionalLSquare())) {
    if (failed(parser.parseCommaSeparatedList([&]() {
          if (parser.parseAttribute(gangAttrs.emplace_back()))
            return failure();
          return success();
        })))
      return failure();
    if (parser.parseRSquare())
      return failure();
    needCommaBeforeOperands = true;
  }

  if (needCommaBeforeOperands && failed(parser.parseComma()))
    return failure();

  if (failed(parser.parseCommaSeparatedList([&]() {
        if (parser.parseKeyword(acc::RoutineOp::getGangDimKeyword()) ||
            parser.parseColon() ||
            parser.parseAttribute(gangDimAttrs.emplace_back()))
          return failure();
        if (succeeded(parser.parseOptionalLSquare())) {
          if (parser.parseAttribute(gangDimDeviceTypeAttrs.emplace_back()) ||
              parser.parseRSquare())
            return failure();
        } else {
          gangDimDeviceTypeAttrs.push_back(mlir::acc::DeviceTypeAttr::get(
              parser.getContext(), mlir::acc::DeviceType::None));
        }
        return success();
      })))
    return failure();

  if (failed(parser.parseRParen()))
    return failure();

  gang = ArrayAttr::get(parser.getContext(), gangAttrs);
  gangDim = ArrayAttr::get(parser.getContext(), gangDimAttrs);
  gangDimDeviceTypes =
      ArrayAttr::get(parser.getContext(), gangDimDeviceTypeAttrs);

  return success();
}

void printRoutineGangClause(OpAsmPrinter &p, Operation *op,
                            std::optional<mlir::ArrayAttr> gang,
                            std::optional<mlir::ArrayAttr> gangDim,
                            std::optional<mlir::ArrayAttr> gangDimDeviceTypes) {

  if (!hasDeviceTypeValues(gangDimDeviceTypes) && hasDeviceTypeValues(gang) &&
      gang->size() == 1) {
    auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>((*gang)[0]);
    if (deviceTypeAttr.getValue() == mlir::acc::DeviceType::None)
      return;
  }

  p << "(";

  printDeviceTypes(p, gang);

  if (hasDeviceTypeValues(gang) && hasDeviceTypeValues(gangDimDeviceTypes))
    p << ", ";

  if (hasDeviceTypeValues(gangDimDeviceTypes))
    llvm::interleaveComma(llvm::zip(*gangDim, *gangDimDeviceTypes), p,
                          [&](const auto &pair) {
                            p << acc::RoutineOp::getGangDimKeyword() << ": ";
                            p << std::get<0>(pair);
                            printSingleDeviceType(p, std::get<1>(pair));
                          });

  p << ")";
}

static ParseResult parseDeviceTypeArrayAttr(OpAsmParser &parser,
                                            mlir::ArrayAttr &deviceTypes) {
  llvm::SmallVector<mlir::Attribute> attributes;
  // Keyword only
  if (failed(parser.parseOptionalLParen())) {
    attributes.push_back(mlir::acc::DeviceTypeAttr::get(
        parser.getContext(), mlir::acc::DeviceType::None));
    deviceTypes = ArrayAttr::get(parser.getContext(), attributes);
    return success();
  }

  // Parse device type attributes
  if (succeeded(parser.parseOptionalLSquare())) {
    if (failed(parser.parseCommaSeparatedList([&]() {
          if (parser.parseAttribute(attributes.emplace_back()))
            return failure();
          return success();
        })))
      return failure();
    if (parser.parseRSquare() || parser.parseRParen())
      return failure();
  }
  deviceTypes = ArrayAttr::get(parser.getContext(), attributes);
  return success();
}

static void
printDeviceTypeArrayAttr(mlir::OpAsmPrinter &p, mlir::Operation *op,
                         std::optional<mlir::ArrayAttr> deviceTypes) {

  if (hasDeviceTypeValues(deviceTypes) && deviceTypes->size() == 1) {
    auto deviceTypeAttr =
        mlir::dyn_cast<mlir::acc::DeviceTypeAttr>((*deviceTypes)[0]);
    if (deviceTypeAttr.getValue() == mlir::acc::DeviceType::None)
      return;
  }

  if (!hasDeviceTypeValues(deviceTypes))
    return;

  p << "([";
  llvm::interleaveComma(*deviceTypes, p, [&](mlir::Attribute attr) {
    auto dTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(attr);
    p << dTypeAttr;
  });
  p << "])";
}

bool RoutineOp::hasWorker() { return hasWorker(mlir::acc::DeviceType::None); }

bool RoutineOp::hasWorker(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getWorker(), deviceType);
}

bool RoutineOp::hasVector() { return hasVector(mlir::acc::DeviceType::None); }

bool RoutineOp::hasVector(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getVector(), deviceType);
}

bool RoutineOp::hasSeq() { return hasSeq(mlir::acc::DeviceType::None); }

bool RoutineOp::hasSeq(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getSeq(), deviceType);
}

std::optional<std::variant<mlir::SymbolRefAttr, mlir::StringAttr>>
RoutineOp::getBindNameValue() {
  return getBindNameValue(mlir::acc::DeviceType::None);
}

std::optional<std::variant<mlir::SymbolRefAttr, mlir::StringAttr>>
RoutineOp::getBindNameValue(mlir::acc::DeviceType deviceType) {
  if (!hasDeviceTypeValues(getBindIdNameDeviceType()) &&
      !hasDeviceTypeValues(getBindStrNameDeviceType())) {
    return std::nullopt;
  }

  if (auto pos = findSegment(*getBindIdNameDeviceType(), deviceType)) {
    auto attr = (*getBindIdName())[*pos];
    auto symbolRefAttr = dyn_cast<mlir::SymbolRefAttr>(attr);
    assert(symbolRefAttr && "expected SymbolRef");
    return symbolRefAttr;
  }

  if (auto pos = findSegment(*getBindStrNameDeviceType(), deviceType)) {
    auto attr = (*getBindStrName())[*pos];
    auto stringAttr = dyn_cast<mlir::StringAttr>(attr);
    assert(stringAttr && "expected String");
    return stringAttr;
  }

  return std::nullopt;
}

bool RoutineOp::hasGang() { return hasGang(mlir::acc::DeviceType::None); }

bool RoutineOp::hasGang(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getGang(), deviceType);
}

std::optional<int64_t> RoutineOp::getGangDimValue() {
  return getGangDimValue(mlir::acc::DeviceType::None);
}

std::optional<int64_t>
RoutineOp::getGangDimValue(mlir::acc::DeviceType deviceType) {
  if (!hasDeviceTypeValues(getGangDimDeviceType()))
    return std::nullopt;
  if (auto pos = findSegment(*getGangDimDeviceType(), deviceType)) {
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>((*getGangDim())[*pos]);
    return intAttr.getInt();
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// InitOp
//===----------------------------------------------------------------------===//

LogicalResult acc::InitOp::verify() {
  Operation *currOp = *this;
  while ((currOp = currOp->getParentOp()))
    if (isComputeOperation(currOp))
      return emitOpError("cannot be nested in a compute operation");
  return success();
}

void acc::InitOp::addDeviceType(MLIRContext *context,
                                mlir::acc::DeviceType deviceType) {
  llvm::SmallVector<mlir::Attribute> deviceTypes;
  if (getDeviceTypesAttr())
    llvm::copy(getDeviceTypesAttr(), std::back_inserter(deviceTypes));

  deviceTypes.push_back(acc::DeviceTypeAttr::get(context, deviceType));
  setDeviceTypesAttr(mlir::ArrayAttr::get(context, deviceTypes));
}

//===----------------------------------------------------------------------===//
// ShutdownOp
//===----------------------------------------------------------------------===//

LogicalResult acc::ShutdownOp::verify() {
  Operation *currOp = *this;
  while ((currOp = currOp->getParentOp()))
    if (isComputeOperation(currOp))
      return emitOpError("cannot be nested in a compute operation");
  return success();
}

void acc::ShutdownOp::addDeviceType(MLIRContext *context,
                                    mlir::acc::DeviceType deviceType) {
  llvm::SmallVector<mlir::Attribute> deviceTypes;
  if (getDeviceTypesAttr())
    llvm::copy(getDeviceTypesAttr(), std::back_inserter(deviceTypes));

  deviceTypes.push_back(acc::DeviceTypeAttr::get(context, deviceType));
  setDeviceTypesAttr(mlir::ArrayAttr::get(context, deviceTypes));
}

//===----------------------------------------------------------------------===//
// SetOp
//===----------------------------------------------------------------------===//

LogicalResult acc::SetOp::verify() {
  Operation *currOp = *this;
  while ((currOp = currOp->getParentOp()))
    if (isComputeOperation(currOp))
      return emitOpError("cannot be nested in a compute operation");
  if (!getDeviceTypeAttr() && !getDefaultAsync() && !getDeviceNum())
    return emitOpError("at least one default_async, device_num, or device_type "
                       "operand must appear");
  return success();
}

//===----------------------------------------------------------------------===//
// UpdateOp
//===----------------------------------------------------------------------===//

LogicalResult acc::UpdateOp::verify() {
  // At least one of host or device should have a value.
  if (getDataClauseOperands().empty())
    return emitError("at least one value must be present in dataOperands");

  if (failed(verifyDeviceTypeCountMatch(*this, getAsyncOperands(),
                                        getAsyncOperandsDeviceTypeAttr(),
                                        "async")))
    return failure();

  if (failed(verifyDeviceTypeAndSegmentCountMatch(
          *this, getWaitOperands(), getWaitOperandsSegmentsAttr(),
          getWaitOperandsDeviceTypeAttr(), "wait")))
    return failure();

  if (failed(checkWaitAndAsyncConflict<acc::UpdateOp>(*this)))
    return failure();

  for (mlir::Value operand : getDataClauseOperands())
    if (!mlir::isa<acc::UpdateDeviceOp, acc::UpdateHostOp, acc::GetDevicePtrOp>(
            operand.getDefiningOp()))
      return emitError("expect data entry/exit operation or acc.getdeviceptr "
                       "as defining op");

  return success();
}

unsigned UpdateOp::getNumDataOperands() {
  return getDataClauseOperands().size();
}

Value UpdateOp::getDataOperand(unsigned i) {
  unsigned numOptional = getAsyncOperands().size();
  numOptional += getIfCond() ? 1 : 0;
  return getOperand(getWaitOperands().size() + numOptional + i);
}

void UpdateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<RemoveConstantIfCondition<UpdateOp>>(context);
}

bool UpdateOp::hasAsyncOnly() {
  return hasAsyncOnly(mlir::acc::DeviceType::None);
}

bool UpdateOp::hasAsyncOnly(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getAsyncOnly(), deviceType);
}

mlir::Value UpdateOp::getAsyncValue() {
  return getAsyncValue(mlir::acc::DeviceType::None);
}

mlir::Value UpdateOp::getAsyncValue(mlir::acc::DeviceType deviceType) {
  if (!hasDeviceTypeValues(getAsyncOperandsDeviceType()))
    return {};

  if (auto pos = findSegment(*getAsyncOperandsDeviceType(), deviceType))
    return getAsyncOperands()[*pos];

  return {};
}

bool UpdateOp::hasWaitOnly() {
  return hasWaitOnly(mlir::acc::DeviceType::None);
}

bool UpdateOp::hasWaitOnly(mlir::acc::DeviceType deviceType) {
  return hasDeviceType(getWaitOnly(), deviceType);
}

mlir::Operation::operand_range UpdateOp::getWaitValues() {
  return getWaitValues(mlir::acc::DeviceType::None);
}

mlir::Operation::operand_range
UpdateOp::getWaitValues(mlir::acc::DeviceType deviceType) {
  return getWaitValuesWithoutDevnum(
      getWaitOperandsDeviceType(), getWaitOperands(), getWaitOperandsSegments(),
      getHasWaitDevnum(), deviceType);
}

mlir::Value UpdateOp::getWaitDevnum() {
  return getWaitDevnum(mlir::acc::DeviceType::None);
}

mlir::Value UpdateOp::getWaitDevnum(mlir::acc::DeviceType deviceType) {
  return getWaitDevnumValue(getWaitOperandsDeviceType(), getWaitOperands(),
                            getWaitOperandsSegments(), getHasWaitDevnum(),
                            deviceType);
}

void UpdateOp::addAsyncOnly(MLIRContext *context,
                            llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setAsyncOnlyAttr(addDeviceTypeAffectedOperandHelper(
      context, getAsyncOnlyAttr(), effectiveDeviceTypes));
}

void UpdateOp::addAsyncOperand(
    MLIRContext *context, mlir::Value newValue,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setAsyncOperandsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getAsyncOperandsDeviceTypeAttr(), effectiveDeviceTypes, newValue,
      getAsyncOperandsMutable()));
}

void UpdateOp::addWaitOnly(MLIRContext *context,
                           llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {
  setWaitOnlyAttr(addDeviceTypeAffectedOperandHelper(context, getWaitOnlyAttr(),
                                                     effectiveDeviceTypes));
}

void UpdateOp::addWaitOperands(
    MLIRContext *context, bool hasDevnum, mlir::ValueRange newValues,
    llvm::ArrayRef<DeviceType> effectiveDeviceTypes) {

  llvm::SmallVector<int32_t> segments;
  if (getWaitOperandsSegments())
    llvm::copy(*getWaitOperandsSegments(), std::back_inserter(segments));

  setWaitOperandsDeviceTypeAttr(addDeviceTypeAffectedOperandHelper(
      context, getWaitOperandsDeviceTypeAttr(), effectiveDeviceTypes, newValues,
      getWaitOperandsMutable(), segments));
  setWaitOperandsSegments(segments);

  llvm::SmallVector<mlir::Attribute> hasDevnums;
  if (getHasWaitDevnumAttr())
    llvm::copy(getHasWaitDevnumAttr(), std::back_inserter(hasDevnums));
  hasDevnums.insert(
      hasDevnums.end(),
      std::max(effectiveDeviceTypes.size(), static_cast<size_t>(1)),
      mlir::BoolAttr::get(context, hasDevnum));
  setHasWaitDevnumAttr(mlir::ArrayAttr::get(context, hasDevnums));
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

LogicalResult acc::WaitOp::verify() {
  // The async attribute represent the async clause without value. Therefore the
  // attribute and operand cannot appear at the same time.
  if (getAsyncOperand() && getAsync())
    return emitError("async attribute cannot appear with asyncOperand");

  if (getWaitDevnum() && getWaitOperands().empty())
    return emitError("wait_devnum cannot appear without waitOperands");

  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/OpenACC/OpenACCOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// acc dialect utilities
//===----------------------------------------------------------------------===//

mlir::TypedValue<mlir::acc::PointerLikeType>
mlir::acc::getVarPtr(mlir::Operation *accDataClauseOp) {
  auto varPtr{llvm::TypeSwitch<mlir::Operation *,
                               mlir::TypedValue<mlir::acc::PointerLikeType>>(
                  accDataClauseOp)
                  .Case<ACC_DATA_ENTRY_OPS>(
                      [&](auto entry) { return entry.getVarPtr(); })
                  .Case<mlir::acc::CopyoutOp, mlir::acc::UpdateHostOp>(
                      [&](auto exit) { return exit.getVarPtr(); })
                  .Default([&](mlir::Operation *) {
                    return mlir::TypedValue<mlir::acc::PointerLikeType>();
                  })};
  return varPtr;
}

mlir::Value mlir::acc::getVar(mlir::Operation *accDataClauseOp) {
  auto varPtr{
      llvm::TypeSwitch<mlir::Operation *, mlir::Value>(accDataClauseOp)
          .Case<ACC_DATA_ENTRY_OPS>([&](auto entry) { return entry.getVar(); })
          .Default([&](mlir::Operation *) { return mlir::Value(); })};
  return varPtr;
}

mlir::Type mlir::acc::getVarType(mlir::Operation *accDataClauseOp) {
  auto varType{llvm::TypeSwitch<mlir::Operation *, mlir::Type>(accDataClauseOp)
                   .Case<ACC_DATA_ENTRY_OPS>(
                       [&](auto entry) { return entry.getVarType(); })
                   .Case<mlir::acc::CopyoutOp, mlir::acc::UpdateHostOp>(
                       [&](auto exit) { return exit.getVarType(); })
                   .Default([&](mlir::Operation *) { return mlir::Type(); })};
  return varType;
}

mlir::TypedValue<mlir::acc::PointerLikeType>
mlir::acc::getAccPtr(mlir::Operation *accDataClauseOp) {
  auto accPtr{llvm::TypeSwitch<mlir::Operation *,
                               mlir::TypedValue<mlir::acc::PointerLikeType>>(
                  accDataClauseOp)
                  .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>(
                      [&](auto dataClause) { return dataClause.getAccPtr(); })
                  .Default([&](mlir::Operation *) {
                    return mlir::TypedValue<mlir::acc::PointerLikeType>();
                  })};
  return accPtr;
}

mlir::Value mlir::acc::getAccVar(mlir::Operation *accDataClauseOp) {
  auto accPtr{llvm::TypeSwitch<mlir::Operation *, mlir::Value>(accDataClauseOp)
                  .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>(
                      [&](auto dataClause) { return dataClause.getAccVar(); })
                  .Default([&](mlir::Operation *) { return mlir::Value(); })};
  return accPtr;
}

mlir::Value mlir::acc::getVarPtrPtr(mlir::Operation *accDataClauseOp) {
  auto varPtrPtr{
      llvm::TypeSwitch<mlir::Operation *, mlir::Value>(accDataClauseOp)
          .Case<ACC_DATA_ENTRY_OPS>(
              [&](auto dataClause) { return dataClause.getVarPtrPtr(); })
          .Default([&](mlir::Operation *) { return mlir::Value(); })};
  return varPtrPtr;
}

mlir::SmallVector<mlir::Value>
mlir::acc::getBounds(mlir::Operation *accDataClauseOp) {
  mlir::SmallVector<mlir::Value> bounds{
      llvm::TypeSwitch<mlir::Operation *, mlir::SmallVector<mlir::Value>>(
          accDataClauseOp)
          .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>([&](auto dataClause) {
            return mlir::SmallVector<mlir::Value>(
                dataClause.getBounds().begin(), dataClause.getBounds().end());
          })
          .Default([&](mlir::Operation *) {
            return mlir::SmallVector<mlir::Value, 0>();
          })};
  return bounds;
}

mlir::SmallVector<mlir::Value>
mlir::acc::getAsyncOperands(mlir::Operation *accDataClauseOp) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::SmallVector<mlir::Value>>(
             accDataClauseOp)
      .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>([&](auto dataClause) {
        return mlir::SmallVector<mlir::Value>(
            dataClause.getAsyncOperands().begin(),
            dataClause.getAsyncOperands().end());
      })
      .Default([&](mlir::Operation *) {
        return mlir::SmallVector<mlir::Value, 0>();
      });
}

mlir::ArrayAttr
mlir::acc::getAsyncOperandsDeviceType(mlir::Operation *accDataClauseOp) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::ArrayAttr>(accDataClauseOp)
      .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>([&](auto dataClause) {
        return dataClause.getAsyncOperandsDeviceTypeAttr();
      })
      .Default([&](mlir::Operation *) { return mlir::ArrayAttr{}; });
}

mlir::ArrayAttr mlir::acc::getAsyncOnly(mlir::Operation *accDataClauseOp) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::ArrayAttr>(accDataClauseOp)
      .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>(
          [&](auto dataClause) { return dataClause.getAsyncOnlyAttr(); })
      .Default([&](mlir::Operation *) { return mlir::ArrayAttr{}; });
}

std::optional<llvm::StringRef> mlir::acc::getVarName(mlir::Operation *accOp) {
  auto name{
      llvm::TypeSwitch<mlir::Operation *, std::optional<llvm::StringRef>>(accOp)
          .Case<ACC_DATA_ENTRY_OPS>([&](auto entry) { return entry.getName(); })
          .Default([&](mlir::Operation *) -> std::optional<llvm::StringRef> {
            return {};
          })};
  return name;
}

std::optional<mlir::acc::DataClause>
mlir::acc::getDataClause(mlir::Operation *accDataEntryOp) {
  auto dataClause{
      llvm::TypeSwitch<mlir::Operation *, std::optional<mlir::acc::DataClause>>(
          accDataEntryOp)
          .Case<ACC_DATA_ENTRY_OPS>(
              [&](auto entry) { return entry.getDataClause(); })
          .Default([&](mlir::Operation *) { return std::nullopt; })};
  return dataClause;
}

bool mlir::acc::getImplicitFlag(mlir::Operation *accDataEntryOp) {
  auto implicit{llvm::TypeSwitch<mlir::Operation *, bool>(accDataEntryOp)
                    .Case<ACC_DATA_ENTRY_OPS>(
                        [&](auto entry) { return entry.getImplicit(); })
                    .Default([&](mlir::Operation *) { return false; })};
  return implicit;
}

mlir::ValueRange mlir::acc::getDataOperands(mlir::Operation *accOp) {
  auto dataOperands{
      llvm::TypeSwitch<mlir::Operation *, mlir::ValueRange>(accOp)
          .Case<ACC_COMPUTE_AND_DATA_CONSTRUCT_OPS>(
              [&](auto entry) { return entry.getDataClauseOperands(); })
          .Default([&](mlir::Operation *) { return mlir::ValueRange(); })};
  return dataOperands;
}

mlir::MutableOperandRange
mlir::acc::getMutableDataOperands(mlir::Operation *accOp) {
  auto dataOperands{
      llvm::TypeSwitch<mlir::Operation *, mlir::MutableOperandRange>(accOp)
          .Case<ACC_COMPUTE_AND_DATA_CONSTRUCT_OPS>(
              [&](auto entry) { return entry.getDataClauseOperandsMutable(); })
          .Default([&](mlir::Operation *) { return nullptr; })};
  return dataOperands;
}

mlir::Operation *mlir::acc::getEnclosingComputeOp(mlir::Region &region) {
  mlir::Operation *parentOp = region.getParentOp();
  while (parentOp) {
    if (mlir::isa<ACC_COMPUTE_CONSTRUCT_OPS>(parentOp)) {
      return parentOp;
    }
    parentOp = parentOp->getParentOp();
  }
  return nullptr;
}
