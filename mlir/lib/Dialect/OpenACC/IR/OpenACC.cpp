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
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace acc;

#include "mlir/Dialect/OpenACC/OpenACCOpsDialect.cpp.inc"
#include "mlir/Dialect/OpenACC/OpenACCOpsEnums.cpp.inc"
#include "mlir/Dialect/OpenACC/OpenACCOpsInterfaces.cpp.inc"
#include "mlir/Dialect/OpenACC/OpenACCTypeInterfaces.cpp.inc"

namespace {
/// Model for pointer-like types that already provide a `getElementType` method.
template <typename T>
struct PointerLikeModel
    : public PointerLikeType::ExternalModel<PointerLikeModel<T>, T> {
  Type getElementType(Type pointer) const {
    return llvm::cast<T>(pointer).getElementType();
  }
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
  LLVM::LLVMPointerType::attachInterface<
      PointerLikeModel<LLVM::LLVMPointerType>>(*getContext());
  MemRefType::attachInterface<PointerLikeModel<MemRefType>>(*getContext());
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
      getDataClause() != acc::DataClause::acc_copy)
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
      getDataClause() != acc::DataClause::acc_copy)
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
  if (!getVarPtr() && !getAccPtr())
    return emitError("must have either host or device pointer");
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
  if (!getVarPtr() && !getAccPtr())
    return emitError("must have either host or device pointer");
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
                                           /*verifyYield=*/true)))
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
  unsigned numOptional = getAsync() ? 1 : 0;
  numOptional += getNumGangs().size();
  numOptional += getNumWorkers() ? 1 : 0;
  numOptional += getVectorLength() ? 1 : 0;
  numOptional += getIfCond() ? 1 : 0;
  numOptional += getSelfCond() ? 1 : 0;
  return getOperand(getWaitOperands().size() + numOptional + i);
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
  if (getNumGangs().size() > 3)
    return emitOpError() << "num_gangs expects a maximum of 3 values";
  return checkDataOperands<acc::ParallelOp>(*this, getDataClauseOperands());
}

//===----------------------------------------------------------------------===//
// SerialOp
//===----------------------------------------------------------------------===//

unsigned SerialOp::getNumDataOperands() {
  return getReductionOperands().size() + getGangPrivateOperands().size() +
         getGangFirstPrivateOperands().size() + getDataClauseOperands().size();
}

Value SerialOp::getDataOperand(unsigned i) {
  unsigned numOptional = getAsync() ? 1 : 0;
  numOptional += getIfCond() ? 1 : 0;
  numOptional += getSelfCond() ? 1 : 0;
  return getOperand(getWaitOperands().size() + numOptional + i);
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
  return checkDataOperands<acc::SerialOp>(*this, getDataClauseOperands());
}

//===----------------------------------------------------------------------===//
// KernelsOp
//===----------------------------------------------------------------------===//

unsigned KernelsOp::getNumDataOperands() {
  return getDataClauseOperands().size();
}

Value KernelsOp::getDataOperand(unsigned i) {
  unsigned numOptional = getAsync() ? 1 : 0;
  numOptional += getWaitOperands().size();
  numOptional += getNumGangs().size();
  numOptional += getNumWorkers() ? 1 : 0;
  numOptional += getVectorLength() ? 1 : 0;
  numOptional += getIfCond() ? 1 : 0;
  numOptional += getSelfCond() ? 1 : 0;
  return getOperand(numOptional + i);
}

LogicalResult acc::KernelsOp::verify() {
  if (getNumGangs().size() > 3)
    return emitOpError() << "num_gangs expects a maximum of 3 values";
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

static ParseResult
parseGangValue(OpAsmParser &parser, llvm::StringRef keyword,
               std::optional<OpAsmParser::UnresolvedOperand> &value,
               Type &valueType, bool &needComa, bool &newValue) {
  if (succeeded(parser.parseOptionalKeyword(keyword))) {
    if (parser.parseEqual())
      return failure();
    value = OpAsmParser::UnresolvedOperand{};
    if (parser.parseOperand(*value) || parser.parseColonType(valueType))
      return failure();
    needComa = true;
    newValue = true;
  }
  return success();
}

static ParseResult parseGangClause(
    OpAsmParser &parser, std::optional<OpAsmParser::UnresolvedOperand> &gangNum,
    Type &gangNumType, std::optional<OpAsmParser::UnresolvedOperand> &gangDim,
    Type &gangDimType,
    std::optional<OpAsmParser::UnresolvedOperand> &gangStatic,
    Type &gangStaticType, UnitAttr &hasGang) {
  hasGang = UnitAttr::get(parser.getBuilder().getContext());
  gangNum = std::nullopt;
  gangDim = std::nullopt;
  gangStatic = std::nullopt;
  bool needComa = false;

  // optional gang operands
  if (succeeded(parser.parseOptionalLParen())) {
    while (true) {
      bool newValue = false;
      bool needValue = false;
      if (needComa) {
        if (succeeded(parser.parseOptionalComma()))
          needValue = true; // expect a new value after comma.
        else
          break;
      }

      if (failed(parseGangValue(parser, LoopOp::getGangNumKeyword(), gangNum,
                                gangNumType, needComa, newValue)))
        return failure();
      if (failed(parseGangValue(parser, LoopOp::getGangDimKeyword(), gangDim,
                                gangDimType, needComa, newValue)))
        return failure();
      if (failed(parseGangValue(parser, LoopOp::getGangStaticKeyword(),
                                gangStatic, gangStaticType, needComa,
                                newValue)))
        return failure();

      if (!newValue && needValue) {
        parser.emitError(parser.getCurrentLocation(),
                         "new value expected after comma");
        return failure();
      }

      if (!newValue)
        break;
    }

    if (!gangNum && !gangDim && !gangStatic) {
      parser.emitError(parser.getCurrentLocation(),
                       "expect at least one of num, dim or static values");
      return failure();
    }

    if (failed(parser.parseRParen()))
      return failure();
  }
  return success();
}

void printGangClause(OpAsmPrinter &p, Operation *op, Value gangNum,
                     Type gangNumType, Value gangDim, Type gangDimType,
                     Value gangStatic, Type gangStaticType, UnitAttr hasGang) {
  if (gangNum || gangStatic || gangDim) {
    p << "(";
    if (gangNum) {
      p << LoopOp::getGangNumKeyword() << "=" << gangNum << " : "
        << gangNumType;
      if (gangStatic || gangDim)
        p << ", ";
    }
    if (gangDim) {
      p << LoopOp::getGangDimKeyword() << "=" << gangDim << " : "
        << gangDimType;
      if (gangStatic)
        p << ", ";
    }
    if (gangStatic)
      p << LoopOp::getGangStaticKeyword() << "=" << gangStatic << " : "
        << gangStaticType;
    p << ")";
  }
}

static ParseResult
parseWorkerClause(OpAsmParser &parser,
                  std::optional<OpAsmParser::UnresolvedOperand> &workerNum,
                  Type &workerNumType, UnitAttr &hasWorker) {
  hasWorker = UnitAttr::get(parser.getBuilder().getContext());
  if (succeeded(parser.parseOptionalLParen())) {
    workerNum = OpAsmParser::UnresolvedOperand{};
    if (parser.parseOperand(*workerNum) ||
        parser.parseColonType(workerNumType) || parser.parseRParen())
      return failure();
  }
  return success();
}

void printWorkerClause(OpAsmPrinter &p, Operation *op, Value workerNum,
                       Type workerNumType, UnitAttr hasWorker) {
  if (workerNum)
    p << "(" << workerNum << " : " << workerNumType << ")";
}

static ParseResult
parseVectorClause(OpAsmParser &parser,
                  std::optional<OpAsmParser::UnresolvedOperand> &vectorLength,
                  Type &vectorLengthType, UnitAttr &hasVector) {
  hasVector = UnitAttr::get(parser.getBuilder().getContext());
  if (succeeded(parser.parseOptionalLParen())) {
    vectorLength = OpAsmParser::UnresolvedOperand{};
    if (parser.parseOperand(*vectorLength) ||
        parser.parseColonType(vectorLengthType) || parser.parseRParen())
      return failure();
  }
  return success();
}

void printVectorClause(OpAsmPrinter &p, Operation *op, Value vectorLength,
                       Type vectorLengthType, UnitAttr hasVector) {
  if (vectorLength)
    p << "(" << vectorLength << " : " << vectorLengthType << ")";
}

LogicalResult acc::LoopOp::verify() {
  // auto, independent and seq attribute are mutually exclusive.
  if ((getAuto_() && (getIndependent() || getSeq())) ||
      (getIndependent() && getSeq())) {
    return emitError() << "only one of \"" << acc::LoopOp::getAutoAttrStrName()
                       << "\", " << getIndependentAttrName() << ", "
                       << getSeqAttrName()
                       << " can be present at the same time";
  }

  // Gang, worker and vector are incompatible with seq.
  if (getSeq() && (getHasGang() || getHasWorker() || getHasVector()))
    return emitError("gang, worker or vector cannot appear with the seq attr");

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
  numOptional += getAsync() ? 1 : 0;
  numOptional += getWaitOperands().size();
  return getOperand(numOptional + i);
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

LogicalResult AtomicReadOp::verify() {
  return verifyCommon();
}

//===----------------------------------------------------------------------===//
// AtomicWriteOp
//===----------------------------------------------------------------------===//

LogicalResult AtomicWriteOp::verify() {
  return verifyCommon();
}

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

LogicalResult AtomicUpdateOp::verify() {
  return verifyCommon();
}

LogicalResult AtomicUpdateOp::verifyRegions() {
  return verifyRegionsCommon();
}

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

LogicalResult AtomicCaptureOp::verifyRegions() {
  return verifyRegionsCommon();
}

//===----------------------------------------------------------------------===//
// DeclareEnterOp
//===----------------------------------------------------------------------===//

template <typename Op>
static LogicalResult checkDeclareOperands(Op &op,
                                          const mlir::ValueRange &operands) {
  if (operands.empty())
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
  if (!getDeviceType() && !getDefaultAsync() && !getDeviceNum())
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
  return getOperand(getWaitOperands().size() + getDeviceTypeOperands().size() +
                    numOptional + i);
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

mlir::Value mlir::acc::getVarPtr(mlir::Operation *accDataEntryOp) {
  auto varPtr{llvm::TypeSwitch<mlir::Operation *, mlir::Value>(accDataEntryOp)
                  .Case<ACC_DATA_ENTRY_OPS>(
                      [&](auto entry) { return entry.getVarPtr(); })
                  .Default([&](mlir::Operation *) { return mlir::Value(); })};
  return varPtr;
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
