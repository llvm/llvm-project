//===- OpenACC.cpp - OpenACC MLIR Operations ------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace acc;

#include "mlir/Dialect/OpenACC/OpenACCOpsDialect.cpp.inc"
#include "mlir/Dialect/OpenACC/OpenACCOpsEnums.cpp.inc"
#include "mlir/Dialect/OpenACC/OpenACCOpsInterfaces.cpp.inc"
#include "mlir/Dialect/OpenACC/OpenACCTypeInterfaces.cpp.inc"

namespace {
struct MemRefPointerLikeModel
    : public PointerLikeType::ExternalModel<MemRefPointerLikeModel,
                                            MemRefType> {
  Type getElementType(Type pointer) const {
    return llvm::cast<MemRefType>(pointer).getElementType();
  }
};

struct LLVMPointerPointerLikeModel
    : public PointerLikeType::ExternalModel<LLVMPointerPointerLikeModel,
                                            LLVM::LLVMPointerType> {
  Type getElementType(Type pointer) const { return Type(); }
};
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
  return success();
}

//===----------------------------------------------------------------------===//
// FirstprivateOp
//===----------------------------------------------------------------------===//
LogicalResult acc::FirstprivateOp::verify() {
  if (getDataClause() != acc::DataClause::acc_firstprivate)
    return emitError("data clause associated with firstprivate operation must "
                     "match its intent");
  return success();
}

//===----------------------------------------------------------------------===//
// ReductionOp
//===----------------------------------------------------------------------===//
LogicalResult acc::ReductionOp::verify() {
  if (getDataClause() != acc::DataClause::acc_reduction)
    return emitError("data clause associated with reduction operation must "
                     "match its intent");
  return success();
}

//===----------------------------------------------------------------------===//
// DevicePtrOp
//===----------------------------------------------------------------------===//
LogicalResult acc::DevicePtrOp::verify() {
  if (getDataClause() != acc::DataClause::acc_deviceptr)
    return emitError("data clause associated with deviceptr operation must "
                     "match its intent");
  return success();
}

//===----------------------------------------------------------------------===//
// PresentOp
//===----------------------------------------------------------------------===//
LogicalResult acc::PresentOp::verify() {
  if (getDataClause() != acc::DataClause::acc_present)
    return emitError(
        "data clause associated with present operation must match its intent");
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
  return success();
}

bool acc::CopyinOp::isCopyinReadonly() {
  return getDataClause() == acc::DataClause::acc_copyin_readonly;
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
  return success();
}

bool acc::CreateOp::isCreateZero() {
  // The zero modifier is encoded in the data clause.
  return getDataClause() == acc::DataClause::acc_create_zero ||
         getDataClause() == acc::DataClause::acc_copyout_zero;
}

//===----------------------------------------------------------------------===//
// NoCreateOp
//===----------------------------------------------------------------------===//
LogicalResult acc::NoCreateOp::verify() {
  if (getDataClause() != acc::DataClause::acc_no_create)
    return emitError("data clause associated with no_create operation must "
                     "match its intent");
  return success();
}

//===----------------------------------------------------------------------===//
// AttachOp
//===----------------------------------------------------------------------===//
LogicalResult acc::AttachOp::verify() {
  if (getDataClause() != acc::DataClause::acc_attach)
    return emitError(
        "data clause associated with attach operation must match its intent");
  return success();
}

//===----------------------------------------------------------------------===//
// DeclareDeviceResidentOp
//===----------------------------------------------------------------------===//

LogicalResult acc::DeclareDeviceResidentOp::verify() {
  if (getDataClause() != acc::DataClause::acc_declare_device_resident)
    return emitError("data clause associated with device_resident operation "
                     "must match its intent");
  return success();
}

//===----------------------------------------------------------------------===//
// DeclareLinkOp
//===----------------------------------------------------------------------===//

LogicalResult acc::DeclareLinkOp::verify() {
  if (getDataClause() != acc::DataClause::acc_declare_link)
    return emitError(
        "data clause associated with link operation must match its intent");
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
  if (!getVarPtr() || !getAccPtr())
    return emitError("must have both host and device pointers");
  return success();
}

bool acc::CopyoutOp::isCopyoutZero() {
  return getDataClause() == acc::DataClause::acc_copyout_zero;
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
      getDataClause() != acc::DataClause::acc_declare_device_resident &&
      getDataClause() != acc::DataClause::acc_declare_link)
    return emitError(
        "data clause associated with delete operation must match its intent"
        " or specify original clause this operation was decomposed from");
  if (!getAccPtr())
    return emitError("must have device pointer");
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
  if (!getAccPtr())
    return emitError("must have device pointer");
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
  if (!getVarPtr() || !getAccPtr())
    return emitError("must have both host and device pointers");
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
  return success();
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
  return isa<acc::ParallelOp>(op) || isa<acc::LoopOp>(op);
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
      rewriter.updateRootInPlace(op, [&]() { op.getIfCondMutable().erase(0); });
    else
      rewriter.eraseOp(op);

    return success();
  }
};

/// Replaces the given op with the contents of the given single-block region,
/// using the operands of the block terminator to replace operation results.
static void replaceOpWithRegion(PatternRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected single-region block");
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
      rewriter.updateRootInPlace(op, [&]() { op.getIfCondMutable().erase(0); });
    else
      replaceOpWithRegion(rewriter, op, op.getRegion());

    return success();
  }
};

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
  for (unsigned i = 0, e = attributes->size(); i < e; ++i) {
    if (i != 0)
      p << ", ";
    p << (*attributes)[i] << " -> " << operands[i] << " : "
      << operands[i].getType();
  }
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
  return getReductionOperands().size() + getGangPrivateOperands().size() +
         getGangFirstPrivateOperands().size() + getDataClauseOperands().size();
}

Value ParallelOp::getDataOperand(unsigned i) {
  unsigned numOptional = getAsync().size();
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

  if (!segments)
    return success();

  for (auto segCount : segments.asArrayRef()) {
    if (maxInSegment != 0 && segCount > maxInSegment)
      return op.emitOpError() << keyword << " expects a maximum of "
                              << maxInSegment << " values per segment";
    numOperandsInSegments += segCount;
  }
  if (numOperandsInSegments != operands.size())
    return op.emitOpError()
           << keyword << " operand count does not match count in segments";
  if (deviceTypes.getValue().size() != (size_t)segments.size())
    return op.emitOpError()
           << keyword << " segment count does not match device_type count";
  return success();
}

LogicalResult acc::ParallelOp::verify() {
  if (failed(checkSymOperandList<mlir::acc::PrivateRecipeOp>(
          *this, getPrivatizations(), getGangPrivateOperands(), "private",
          "privatizations", /*checkOperandType=*/false)))
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

  if (failed(verifyDeviceTypeCountMatch(*this, getAsync(),
                                        getAsyncDeviceTypeAttr(), "async")))
    return failure();

  return checkDataOperands<acc::ParallelOp>(*this, getDataClauseOperands());
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
  if (auto arrayAttr = getAsyncOnly()) {
    if (findSegment(*arrayAttr, deviceType))
      return true;
  }
  return false;
}

mlir::Value acc::ParallelOp::getAsyncValue() {
  return getAsyncValue(mlir::acc::DeviceType::None);
}

mlir::Value acc::ParallelOp::getAsyncValue(mlir::acc::DeviceType deviceType) {
  return getValueInDeviceTypeSegment(getAsyncDeviceType(), getAsync(),
                                     deviceType);
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

mlir::Operation::operand_range
ParallelOp::getNumGangsValues(mlir::acc::DeviceType deviceType) {
  return getValuesFromSegments(getNumGangsDeviceType(), getNumGangs(),
                               getNumGangsSegments(), deviceType);
}

bool acc::ParallelOp::hasWaitOnly() {
  return hasWaitOnly(mlir::acc::DeviceType::None);
}

bool acc::ParallelOp::hasWaitOnly(mlir::acc::DeviceType deviceType) {
  if (auto arrayAttr = getWaitOnly()) {
    if (findSegment(*arrayAttr, deviceType))
      return true;
  }
  return false;
}

mlir::Operation::operand_range ParallelOp::getWaitValues() {
  return getWaitValues(mlir::acc::DeviceType::None);
}

mlir::Operation::operand_range
ParallelOp::getWaitValues(mlir::acc::DeviceType deviceType) {
  return getValuesFromSegments(getWaitOperandsDeviceType(), getWaitOperands(),
                               getWaitOperandsSegments(), deviceType);
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

    if (failed(parser.parseCommaSeparatedList(
            mlir::AsmParser::Delimiter::None, [&]() {
              if (parser.parseOperand(operands.emplace_back()) ||
                  parser.parseColonType(types.emplace_back()))
                return failure();
              return success();
            })))
      return failure();

    seg.push_back(operands.size());

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

static void printNumGangs(mlir::OpAsmPrinter &p, mlir::Operation *op,
                          mlir::OperandRange operands, mlir::TypeRange types,
                          std::optional<mlir::ArrayAttr> deviceTypes,
                          std::optional<mlir::DenseI32ArrayAttr> segments) {
  unsigned opIdx = 0;
  for (unsigned i = 0; i < deviceTypes->size(); ++i) {
    if (i != 0)
      p << ", ";
    p << "{";
    for (int32_t j = 0; j < (*segments)[i]; ++j) {
      if (j != 0)
        p << ", ";
      p << operands[opIdx] << " : " << operands[opIdx].getType();
      ++opIdx;
    }
    p << "}";
    auto deviceTypeAttr =
        mlir::dyn_cast<mlir::acc::DeviceTypeAttr>((*deviceTypes)[i]);
    if (deviceTypeAttr.getValue() != mlir::acc::DeviceType::None)
      p << " [" << (*deviceTypes)[i] << "]";
  }
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

static void printSingleDeviceType(mlir::OpAsmPrinter &p, mlir::Attribute attr) {
  auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(attr);
  if (deviceTypeAttr.getValue() != mlir::acc::DeviceType::None)
    p << " [" << attr << "]";
}

static void printDeviceTypeOperandsWithSegment(
    mlir::OpAsmPrinter &p, mlir::Operation *op, mlir::OperandRange operands,
    mlir::TypeRange types, std::optional<mlir::ArrayAttr> deviceTypes,
    std::optional<mlir::DenseI32ArrayAttr> segments) {
  unsigned opIdx = 0;
  for (unsigned i = 0; i < deviceTypes->size(); ++i) {
    if (i != 0)
      p << ", ";
    p << "{";
    for (int32_t j = 0; j < (*segments)[i]; ++j) {
      if (j != 0)
        p << ", ";
      p << operands[opIdx] << " : " << operands[opIdx].getType();
      ++opIdx;
    }
    p << "}";
    printSingleDeviceType(p, (*deviceTypes)[i]);
  }
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
  for (unsigned i = 0, e = deviceTypes->size(); i < e; ++i) {
    if (i != 0)
      p << ", ";
    p << operands[i] << " : " << operands[i].getType();
    printSingleDeviceType(p, (*deviceTypes)[i]);
  }
}

static ParseResult parseDeviceTypeOperandsWithKeywordOnly(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<Type> &types, mlir::ArrayAttr &deviceTypes,
    mlir::ArrayAttr &keywordOnlyDeviceType) {

  llvm::SmallVector<mlir::Attribute> keywordOnlyDeviceTypeAttributes;
  bool needCommaBeforeOperands = false;

  // Keyword only
  if (failed(parser.parseOptionalLParen())) {
    keywordOnlyDeviceTypeAttributes.push_back(mlir::acc::DeviceTypeAttr::get(
        parser.getContext(), mlir::acc::DeviceType::None));
    keywordOnlyDeviceType =
        ArrayAttr::get(parser.getContext(), keywordOnlyDeviceTypeAttributes);
    return success();
  }

  // Parse keyword only attributes
  if (succeeded(parser.parseOptionalLSquare())) {
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

bool hasDeviceTypeValues(std::optional<mlir::ArrayAttr> arrayAttr) {
  if (arrayAttr && *arrayAttr && arrayAttr->size() > 0)
    return true;
  return false;
}

static void printDeviceTypes(mlir::OpAsmPrinter &p,
                             std::optional<mlir::ArrayAttr> deviceTypes) {
  if (!hasDeviceTypeValues(deviceTypes))
    return;
  p << "[";
  for (unsigned i = 0; i < deviceTypes.value().size(); ++i) {
    if (i != 0)
      p << ", ";
    auto deviceTypeAttr =
        mlir::dyn_cast<mlir::acc::DeviceTypeAttr>((*deviceTypes)[i]);
    p << deviceTypeAttr;
  }
  p << "]";
}

static void printDeviceTypeOperandsWithKeywordOnly(
    mlir::OpAsmPrinter &p, mlir::Operation *op, mlir::OperandRange operands,
    mlir::TypeRange types, std::optional<mlir::ArrayAttr> deviceTypes,
    std::optional<mlir::ArrayAttr> keywordOnlyDeviceTypes) {

  if (operands.begin() == operands.end() && keywordOnlyDeviceTypes &&
      keywordOnlyDeviceTypes->size() == 1) {
    auto deviceTypeAttr =
        mlir::dyn_cast<mlir::acc::DeviceTypeAttr>((*keywordOnlyDeviceTypes)[0]);
    if (deviceTypeAttr.getValue() == mlir::acc::DeviceType::None)
      return;
  }

  p << "(";

  printDeviceTypes(p, keywordOnlyDeviceTypes);

  if (hasDeviceTypeValues(keywordOnlyDeviceTypes) &&
      hasDeviceTypeValues(deviceTypes))
    p << ", ";

  for (unsigned i = 0, e = deviceTypes->size(); i < e; ++i) {
    if (i != 0)
      p << ", ";
    p << operands[i] << " : " << operands[i].getType();
    printSingleDeviceType(p, (*deviceTypes)[i]);
  }
  p << ")";
}

//===----------------------------------------------------------------------===//
// SerialOp
//===----------------------------------------------------------------------===//

unsigned SerialOp::getNumDataOperands() {
  return getReductionOperands().size() + getGangPrivateOperands().size() +
         getGangFirstPrivateOperands().size() + getDataClauseOperands().size();
}

Value SerialOp::getDataOperand(unsigned i) {
  unsigned numOptional = getAsync().size();
  numOptional += getIfCond() ? 1 : 0;
  numOptional += getSelfCond() ? 1 : 0;
  return getOperand(getWaitOperands().size() + numOptional + i);
}

bool acc::SerialOp::hasAsyncOnly() {
  return hasAsyncOnly(mlir::acc::DeviceType::None);
}

bool acc::SerialOp::hasAsyncOnly(mlir::acc::DeviceType deviceType) {
  if (auto arrayAttr = getAsyncOnly()) {
    if (findSegment(*arrayAttr, deviceType))
      return true;
  }
  return false;
}

mlir::Value acc::SerialOp::getAsyncValue() {
  return getAsyncValue(mlir::acc::DeviceType::None);
}

mlir::Value acc::SerialOp::getAsyncValue(mlir::acc::DeviceType deviceType) {
  return getValueInDeviceTypeSegment(getAsyncDeviceType(), getAsync(),
                                     deviceType);
}

bool acc::SerialOp::hasWaitOnly() {
  return hasWaitOnly(mlir::acc::DeviceType::None);
}

bool acc::SerialOp::hasWaitOnly(mlir::acc::DeviceType deviceType) {
  if (auto arrayAttr = getWaitOnly()) {
    if (findSegment(*arrayAttr, deviceType))
      return true;
  }
  return false;
}

mlir::Operation::operand_range SerialOp::getWaitValues() {
  return getWaitValues(mlir::acc::DeviceType::None);
}

mlir::Operation::operand_range
SerialOp::getWaitValues(mlir::acc::DeviceType deviceType) {
  return getValuesFromSegments(getWaitOperandsDeviceType(), getWaitOperands(),
                               getWaitOperandsSegments(), deviceType);
}

LogicalResult acc::SerialOp::verify() {
  if (failed(checkSymOperandList<mlir::acc::PrivateRecipeOp>(
          *this, getPrivatizations(), getGangPrivateOperands(), "private",
          "privatizations", /*checkOperandType=*/false)))
    return failure();
  if (failed(checkSymOperandList<mlir::acc::ReductionRecipeOp>(
          *this, getReductionRecipes(), getReductionOperands(), "reduction",
          "reductions", false)))
    return failure();

  if (failed(verifyDeviceTypeAndSegmentCountMatch(
          *this, getWaitOperands(), getWaitOperandsSegmentsAttr(),
          getWaitOperandsDeviceTypeAttr(), "wait")))
    return failure();

  if (failed(verifyDeviceTypeCountMatch(*this, getAsync(),
                                        getAsyncDeviceTypeAttr(), "async")))
    return failure();

  return checkDataOperands<acc::SerialOp>(*this, getDataClauseOperands());
}

//===----------------------------------------------------------------------===//
// KernelsOp
//===----------------------------------------------------------------------===//

unsigned KernelsOp::getNumDataOperands() {
  return getDataClauseOperands().size();
}

Value KernelsOp::getDataOperand(unsigned i) {
  unsigned numOptional = getAsync().size();
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
  if (auto arrayAttr = getAsyncOnly()) {
    if (findSegment(*arrayAttr, deviceType))
      return true;
  }
  return false;
}

mlir::Value acc::KernelsOp::getAsyncValue() {
  return getAsyncValue(mlir::acc::DeviceType::None);
}

mlir::Value acc::KernelsOp::getAsyncValue(mlir::acc::DeviceType deviceType) {
  return getValueInDeviceTypeSegment(getAsyncDeviceType(), getAsync(),
                                     deviceType);
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
  if (auto arrayAttr = getWaitOnly()) {
    if (findSegment(*arrayAttr, deviceType))
      return true;
  }
  return false;
}

mlir::Operation::operand_range KernelsOp::getWaitValues() {
  return getWaitValues(mlir::acc::DeviceType::None);
}

mlir::Operation::operand_range
KernelsOp::getWaitValues(mlir::acc::DeviceType deviceType) {
  return getValuesFromSegments(getWaitOperandsDeviceType(), getWaitOperands(),
                               getWaitOperandsSegments(), deviceType);
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

  if (failed(verifyDeviceTypeCountMatch(*this, getAsync(),
                                        getAsyncDeviceTypeAttr(), "async")))
    return failure();

  return checkDataOperands<acc::KernelsOp>(*this, getDataClauseOperands());
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

  // Gang only keyword
  if (failed(parser.parseOptionalLParen())) {
    gangOnlyDeviceTypeAttributes.push_back(mlir::acc::DeviceTypeAttr::get(
        parser.getContext(), mlir::acc::DeviceType::None));
    gangOnlyDeviceType =
        ArrayAttr::get(parser.getContext(), gangOnlyDeviceTypeAttributes);
    return success();
  }

  // Parse gang only attributes
  if (succeeded(parser.parseOptionalLSquare())) {
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

  if (operands.begin() == operands.end() && gangOnlyDeviceTypes &&
      gangOnlyDeviceTypes->size() == 1) {
    auto deviceTypeAttr =
        mlir::dyn_cast<mlir::acc::DeviceTypeAttr>((*gangOnlyDeviceTypes)[0]);
    if (deviceTypeAttr.getValue() == mlir::acc::DeviceType::None)
      return;
  }

  p << "(";
  if (hasDeviceTypeValues(gangOnlyDeviceTypes)) {
    p << "[";
    for (unsigned i = 0; i < gangOnlyDeviceTypes.value().size(); ++i) {
      if (i != 0)
        p << ", ";
      auto deviceTypeAttr =
          mlir::dyn_cast<mlir::acc::DeviceTypeAttr>((*gangOnlyDeviceTypes)[i]);
      p << deviceTypeAttr;
    }
    p << "]";
  }

  if (hasDeviceTypeValues(gangOnlyDeviceTypes) &&
      hasDeviceTypeValues(deviceTypes))
    p << ", ";

  if (deviceTypes) {
    unsigned opIdx = 0;
    for (unsigned i = 0; i < deviceTypes->size(); ++i) {
      if (i != 0)
        p << ", ";
      p << "{";
      for (int32_t j = 0; j < (*segments)[i]; ++j) {
        if (j != 0)
          p << ", ";
        auto gangArgTypeAttr =
            mlir::dyn_cast<mlir::acc::GangArgTypeAttr>((*gangArgTypes)[opIdx]);
        if (gangArgTypeAttr.getValue() == mlir::acc::GangArgType::Num)
          p << LoopOp::getGangNumKeyword();
        else if (gangArgTypeAttr.getValue() == mlir::acc::GangArgType::Dim)
          p << LoopOp::getGangDimKeyword();
        else if (gangArgTypeAttr.getValue() == mlir::acc::GangArgType::Static)
          p << LoopOp::getGangStaticKeyword();
        p << "=" << operands[opIdx] << " : " << operands[opIdx].getType();
        ++opIdx;
      }

      p << "}";
      auto deviceTypeAttr =
          mlir::dyn_cast<mlir::acc::DeviceTypeAttr>((*deviceTypes)[i]);
      if (deviceTypeAttr.getValue() != mlir::acc::DeviceType::None)
        p << " [" << (*deviceTypes)[i] << "]";
    }
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
    if (deviceTypes.contains(deviceTypeAttr.getValue()))
      return true;
    deviceTypes.insert(deviceTypeAttr.getValue());
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
    if (crtDeviceTypes.contains(deviceTypeAttr.getValue()))
      return failure();
    crtDeviceTypes.insert(deviceTypeAttr.getValue());
  }
  return success();
}

LogicalResult acc::LoopOp::verify() {
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
    return emitError() << "only one of \"" << acc::LoopOp::getAutoAttrStrName()
                       << "\", " << getIndependentAttrName() << ", "
                       << getSeqAttrName()
                       << " can be present at the same time";
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
        return emitError()
               << "gang, worker or vector cannot appear with the seq attr";
    }
  }

  if (failed(checkSymOperandList<mlir::acc::PrivateRecipeOp>(
          *this, getPrivatizations(), getPrivateOperands(), "private",
          "privatizations", false)))
    return failure();

  if (failed(checkSymOperandList<mlir::acc::ReductionRecipeOp>(
          *this, getReductionRecipes(), getReductionOperands(), "reduction",
          "reductions", false)))
    return failure();

  // Check non-empty body().
  if (getRegion().empty())
    return emitError("expected non-empty body.");

  return success();
}

unsigned LoopOp::getNumDataOperands() {
  return getReductionOperands().size() + getPrivateOperands().size();
}

Value LoopOp::getDataOperand(unsigned i) {
  unsigned numOptional = getGangOperands().size();
  numOptional += getVectorOperands().size();
  numOptional += getWorkerNumOperands().size();
  numOptional += getTileOperands().size();
  numOptional += getCacheOperands().size();
  return getOperand(numOptional + i);
}

bool LoopOp::hasAuto() { return hasAuto(mlir::acc::DeviceType::None); }

bool LoopOp::hasAuto(mlir::acc::DeviceType deviceType) {
  if (auto arrayAttr = getAuto_()) {
    if (findSegment(*arrayAttr, deviceType))
      return true;
  }
  return false;
}

bool LoopOp::hasIndependent() {
  return hasIndependent(mlir::acc::DeviceType::None);
}

bool LoopOp::hasIndependent(mlir::acc::DeviceType deviceType) {
  if (auto arrayAttr = getIndependent()) {
    if (findSegment(*arrayAttr, deviceType))
      return true;
  }
  return false;
}

bool LoopOp::hasSeq() { return hasSeq(mlir::acc::DeviceType::None); }

bool LoopOp::hasSeq(mlir::acc::DeviceType deviceType) {
  if (auto arrayAttr = getSeq()) {
    if (findSegment(*arrayAttr, deviceType))
      return true;
  }
  return false;
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
  if (auto arrayAttr = getVector()) {
    if (findSegment(*arrayAttr, deviceType))
      return true;
  }
  return false;
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
  if (auto arrayAttr = getWorker()) {
    if (findSegment(*arrayAttr, deviceType))
      return true;
  }
  return false;
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
  if (auto arrayAttr = getGang()) {
    if (findSegment(*arrayAttr, deviceType))
      return true;
  }
  return false;
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
    if (!mlir::isa<acc::AttachOp, acc::CopyinOp, acc::CopyoutOp, acc::CreateOp,
                   acc::DeleteOp, acc::DetachOp, acc::DevicePtrOp,
                   acc::GetDevicePtrOp, acc::NoCreateOp, acc::PresentOp>(
            operand.getDefiningOp()))
      return emitError("expect data entry/exit operation or acc.getdeviceptr "
                       "as defining op");

  return success();
}

unsigned DataOp::getNumDataOperands() { return getDataClauseOperands().size(); }

Value DataOp::getDataOperand(unsigned i) {
  unsigned numOptional = getIfCond() ? 1 : 0;
  numOptional += getAsync().size() ? 1 : 0;
  numOptional += getWaitOperands().size();
  return getOperand(numOptional + i);
}

bool acc::DataOp::hasAsyncOnly() {
  return hasAsyncOnly(mlir::acc::DeviceType::None);
}

bool acc::DataOp::hasAsyncOnly(mlir::acc::DeviceType deviceType) {
  if (auto arrayAttr = getAsyncOnly()) {
    if (findSegment(*arrayAttr, deviceType))
      return true;
  }
  return false;
}

mlir::Value DataOp::getAsyncValue() {
  return getAsyncValue(mlir::acc::DeviceType::None);
}

mlir::Value DataOp::getAsyncValue(mlir::acc::DeviceType deviceType) {
  return getValueInDeviceTypeSegment(getAsyncDeviceType(), getAsync(),
                                     deviceType);
}

bool DataOp::hasWaitOnly() { return hasWaitOnly(mlir::acc::DeviceType::None); }

bool DataOp::hasWaitOnly(mlir::acc::DeviceType deviceType) {
  if (auto arrayAttr = getWaitOnly()) {
    if (findSegment(*arrayAttr, deviceType))
      return true;
  }
  return false;
}

mlir::Operation::operand_range DataOp::getWaitValues() {
  return getWaitValues(mlir::acc::DeviceType::None);
}

mlir::Operation::operand_range
DataOp::getWaitValues(mlir::acc::DeviceType deviceType) {
  return getValuesFromSegments(getWaitOperandsDeviceType(), getWaitOperands(),
                               getWaitOperandsSegments(), deviceType);
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
    rewriter.replaceOpWithNewOp<AtomicWriteOp>(op, op.getX(), writeVal);
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
    if (!mlir::isa<acc::CopyinOp, acc::CopyoutOp, acc::CreateOp,
                   acc::DevicePtrOp, acc::GetDevicePtrOp, acc::PresentOp,
                   acc::DeclareDeviceResidentOp, acc::DeclareLinkOp>(
            operand.getDefiningOp()))
      return op.emitError(
          "expect valid declare data entry operation or acc.getdeviceptr "
          "as defining op");

    mlir::Value varPtr{getVarPtr(operand.getDefiningOp())};
    assert(varPtr && "declare operands can only be data entry operations which "
                     "must have varPtr");
    std::optional<mlir::acc::DataClause> dataClauseOptional{
        getDataClause(operand.getDefiningOp())};
    assert(dataClauseOptional.has_value() &&
           "declare operands can only be data entry operations which must have "
           "dataClause");

    // If varPtr has no defining op - there is nothing to check further.
    if (!varPtr.getDefiningOp())
      continue;

    // Check that the varPtr has a declare attribute.
    auto declareAttribute{
        varPtr.getDefiningOp()->getAttr(mlir::acc::getDeclareAttrName())};
    if (!declareAttribute)
      return op.emitError(
          "expect declare attribute on variable in declare operation");

    auto declAttr = mlir::cast<mlir::acc::DeclareAttr>(declareAttribute);
    if (declAttr.getDataClause().getValue() != dataClauseOptional.value())
      return op.emitError(
          "expect matching declare attribute on variable in declare operation");

    // If the variable is marked with implicit attribute, the matching declare
    // data action must also be marked implicit. The reverse is not checked
    // since implicit data action may be inserted to do actions like updating
    // device copy, in which case the variable is not necessarily implicitly
    // declare'd.
    if (declAttr.getImplicit() &&
        declAttr.getImplicit() != acc::getImplicitFlag(operand.getDefiningOp()))
      return op.emitError(
          "implicitness must match between declare op and flag on variable");
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

LogicalResult acc::RoutineOp::verify() {
  int parallelism = 0;
  parallelism += getGang() ? 1 : 0;
  parallelism += getWorker() ? 1 : 0;
  parallelism += getVector() ? 1 : 0;
  parallelism += getSeq() ? 1 : 0;

  if (parallelism > 1)
    return emitError() << "only one of `gang`, `worker`, `vector`, `seq` can "
                          "be present at the same time";

  return success();
}

static ParseResult parseRoutineGangClause(OpAsmParser &parser, UnitAttr &gang,
                                          IntegerAttr &gangDim) {
  // Since gang clause exists, ensure that unit attribute is set.
  gang = UnitAttr::get(parser.getBuilder().getContext());

  // Next, look for dim on gang. Don't initialize `gangDim` yet since
  // we leave it without attribute if there is no `dim` specifier.
  if (succeeded(parser.parseOptionalLParen())) {
    // Look for syntax that looks like `dim = 1 : i32`.
    // Thus first look for `dim =`
    if (failed(parser.parseKeyword(RoutineOp::getGangDimKeyword())) ||
        failed(parser.parseEqual()))
      return failure();

    int64_t dimValue;
    Type valueType;
    // Now look for `1 : i32`
    if (failed(parser.parseInteger(dimValue)) ||
        failed(parser.parseColonType(valueType)))
      return failure();

    gangDim = IntegerAttr::get(valueType, dimValue);

    if (failed(parser.parseRParen()))
      return failure();
  }

  return success();
}

void printRoutineGangClause(OpAsmPrinter &p, Operation *op, UnitAttr gang,
                            IntegerAttr gangDim) {
  if (gangDim)
    p << "(" << RoutineOp::getGangDimKeyword() << " = " << gangDim.getValue()
      << " : " << gangDim.getType() << ")";
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
  unsigned numOptional = getAsyncOperand() ? 1 : 0;
  numOptional += getWaitDevnum() ? 1 : 0;
  numOptional += getIfCond() ? 1 : 0;
  return getOperand(getWaitOperands().size() + numOptional + i);
}

void UpdateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<RemoveConstantIfCondition<UpdateOp>>(context);
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

mlir::Value mlir::acc::getVarPtr(mlir::Operation *accDataClauseOp) {
  auto varPtr{llvm::TypeSwitch<mlir::Operation *, mlir::Value>(accDataClauseOp)
                  .Case<ACC_DATA_ENTRY_OPS>(
                      [&](auto entry) { return entry.getVarPtr(); })
                  .Case<mlir::acc::CopyoutOp, mlir::acc::UpdateHostOp>(
                      [&](auto exit) { return exit.getVarPtr(); })
                  .Default([&](mlir::Operation *) { return mlir::Value(); })};
  return varPtr;
}

mlir::Value mlir::acc::getAccPtr(mlir::Operation *accDataClauseOp) {
  auto accPtr{llvm::TypeSwitch<mlir::Operation *, mlir::Value>(accDataClauseOp)
                  .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>(
                      [&](auto dataClause) { return dataClause.getAccPtr(); })
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
