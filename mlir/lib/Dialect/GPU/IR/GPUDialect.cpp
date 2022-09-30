//===- GPUDialect.cpp - MLIR Dialect for GPU Kernels implementation -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the GPU kernel-related dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::gpu;

#include "mlir/Dialect/GPU/IR/GPUOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MMAMatrixType
//===----------------------------------------------------------------------===//

MMAMatrixType MMAMatrixType::get(ArrayRef<int64_t> shape, Type elementType,
                                 StringRef operand) {
  return Base::get(elementType.getContext(), shape, elementType, operand);
}

MMAMatrixType
MMAMatrixType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<int64_t> shape, Type elementType,
                          StringRef operand) {
  return Base::getChecked(emitError, elementType.getContext(), shape,
                          elementType, operand);
}

unsigned MMAMatrixType::getNumDims() const { return getImpl()->numDims; }

ArrayRef<int64_t> MMAMatrixType::getShape() const {
  return getImpl()->getShape();
}

Type MMAMatrixType::getElementType() const { return getImpl()->elementType; }

StringRef MMAMatrixType::getOperand() const { return getImpl()->getOperand(); }

bool MMAMatrixType::isValidElementType(Type elementType) {
  return elementType.isF16() || elementType.isF32();
}

LogicalResult
MMAMatrixType::verify(function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<int64_t> shape, Type elementType,
                      StringRef operand) {
  if (!operand.equals("AOp") && !operand.equals("BOp") &&
      !operand.equals("COp"))
    return emitError() << "operand expected to be one of AOp, BOp or COp";

  if (shape.size() != 2)
    return emitError() << "MMAMatrixType must have exactly two dimensions";

  if (!MMAMatrixType::isValidElementType(elementType))
    return emitError() << "MMAMatrixType elements must be F16 or F32";

  return success();
}

//===----------------------------------------------------------------------===//
// GPUDialect
//===----------------------------------------------------------------------===//

/// GPU memory space identifiers.
enum GPUMemorySpace {
  /// Generic memory space identifier.
  kGenericMemorySpace = 0,

  /// Global memory space identifier.
  kGlobalMemorySpace = 1,

  /// Shared memory space identifier.
  kSharedMemorySpace = 3
};

bool GPUDialect::isKernel(Operation *op) {
  UnitAttr isKernelAttr = op->getAttrOfType<UnitAttr>(getKernelFuncAttrName());
  return static_cast<bool>(isKernelAttr);
}

namespace {
/// This class defines the interface for handling inlining with gpu
/// operations.
struct GPUInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All gpu dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // namespace

void GPUDialect::initialize() {
  addTypes<AsyncTokenType>();
  addTypes<MMAMatrixType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/GPU/IR/GPUOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/GPU/IR/GPUOpsAttributes.cpp.inc"
      >();
  addInterfaces<GPUInlinerInterface>();
}

Type GPUDialect::parseType(DialectAsmParser &parser) const {
  // Parse the main keyword for the type.
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();
  MLIRContext *context = getContext();

  // Handle 'async token' types.
  if (keyword == "async.token")
    return AsyncTokenType::get(context);

  if (keyword == "mma_matrix") {
    SMLoc beginLoc = parser.getNameLoc();

    // Parse '<'.
    if (parser.parseLess())
      return nullptr;

    // Parse the size and elementType.
    SmallVector<int64_t> shape;
    Type elementType;
    if (parser.parseDimensionList(shape, /*allowDynamic=*/false) ||
        parser.parseType(elementType))
      return nullptr;

    // Parse ','
    if (parser.parseComma())
      return nullptr;

    // Parse operand.
    std::string operand;
    if (failed(parser.parseOptionalString(&operand)))
      return nullptr;

    // Parse '>'.
    if (parser.parseGreater())
      return nullptr;

    return MMAMatrixType::getChecked(mlir::detail::getDefaultDiagnosticEmitFn(
                                         parser.getEncodedSourceLoc(beginLoc)),
                                     shape, elementType, operand);
  }

  parser.emitError(parser.getNameLoc(), "unknown gpu type: " + keyword);
  return Type();
}

void GPUDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<AsyncTokenType>([&](Type) { os << "async.token"; })
      .Case<MMAMatrixType>([&](MMAMatrixType fragTy) {
        os << "mma_matrix<";
        auto shape = fragTy.getShape();
        for (auto dim = shape.begin(), e = shape.end() - 1; dim != e; ++dim)
          os << *dim << 'x';
        os << shape.back() << 'x' << fragTy.getElementType();
        os << ", \"" << fragTy.getOperand() << "\"" << '>';
      })
      .Default([](Type) { llvm_unreachable("unexpected 'gpu' type kind"); });
}

LogicalResult GPUDialect::verifyOperationAttribute(Operation *op,
                                                   NamedAttribute attr) {
  if (!attr.getValue().isa<UnitAttr>() ||
      attr.getName() != getContainerModuleAttrName())
    return success();

  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("expected '")
           << getContainerModuleAttrName() << "' attribute to be attached to '"
           << ModuleOp::getOperationName() << '\'';

  auto walkResult = module.walk([&module](LaunchFuncOp launchOp) -> WalkResult {
    // Ignore launches that are nested more or less deep than functions in the
    // module we are currently checking.
    if (!launchOp->getParentOp() ||
        launchOp->getParentOp()->getParentOp() != module)
      return success();

    // Ignore launch ops with missing attributes here. The errors will be
    // reported by the verifiers of those ops.
    if (!launchOp->getAttrOfType<SymbolRefAttr>(
            LaunchFuncOp::getKernelAttrName(launchOp->getName())))
      return success();

    // Check that `launch_func` refers to a well-formed GPU kernel module.
    StringAttr kernelModuleName = launchOp.getKernelModuleName();
    auto kernelModule = module.lookupSymbol<GPUModuleOp>(kernelModuleName);
    if (!kernelModule)
      return launchOp.emitOpError()
             << "kernel module '" << kernelModuleName.getValue()
             << "' is undefined";

    // Check that `launch_func` refers to a well-formed kernel function.
    Operation *kernelFunc = module.lookupSymbol(launchOp.getKernelAttr());
    if (!kernelFunc)
      return launchOp.emitOpError("kernel function '")
             << launchOp.getKernel() << "' is undefined";
    auto kernelConvertedFunction = dyn_cast<FunctionOpInterface>(kernelFunc);
    if (!kernelConvertedFunction) {
      InFlightDiagnostic diag = launchOp.emitOpError()
                                << "referenced kernel '" << launchOp.getKernel()
                                << "' is not a function";
      diag.attachNote(kernelFunc->getLoc()) << "see the kernel definition here";
      return diag;
    }

    if (!kernelFunc->getAttrOfType<mlir::UnitAttr>(
            GPUDialect::getKernelFuncAttrName()))
      return launchOp.emitOpError("kernel function is missing the '")
             << GPUDialect::getKernelFuncAttrName() << "' attribute";

    // TODO: If the kernel isn't a GPU function (which happens during separate
    // compilation), do not check type correspondence as it would require the
    // verifier to be aware of the type conversion.
    auto kernelGPUFunction = dyn_cast<gpu::GPUFuncOp>(kernelFunc);
    if (!kernelGPUFunction)
      return success();

    unsigned actualNumArguments = launchOp.getNumKernelOperands();
    unsigned expectedNumArguments = kernelGPUFunction.getNumArguments();
    if (expectedNumArguments != actualNumArguments)
      return launchOp.emitOpError("got ")
             << actualNumArguments << " kernel operands but expected "
             << expectedNumArguments;

    auto functionType = kernelGPUFunction.getFunctionType();
    for (unsigned i = 0; i < expectedNumArguments; ++i) {
      if (launchOp.getKernelOperand(i).getType() != functionType.getInput(i)) {
        return launchOp.emitOpError("type of function argument ")
               << i << " does not match";
      }
    }

    return success();
  });

  return walkResult.wasInterrupted() ? failure() : success();
}

/// Parses an optional list of async operands with an optional leading keyword.
/// (`async`)? (`[` ssa-id-list `]`)?
///
/// This method is used by the tablegen assembly format for async ops as well.
static ParseResult parseAsyncDependencies(
    OpAsmParser &parser, Type &asyncTokenType,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &asyncDependencies) {
  auto loc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("async"))) {
    if (parser.getNumResults() == 0)
      return parser.emitError(loc, "needs to be named when marked 'async'");
    asyncTokenType = parser.getBuilder().getType<AsyncTokenType>();
  }
  return parser.parseOperandList(asyncDependencies,
                                 OpAsmParser::Delimiter::OptionalSquare);
}

/// Prints optional async dependencies with its leading keyword.
///   (`async`)? (`[` ssa-id-list `]`)?
// Used by the tablegen assembly format for several async ops.
static void printAsyncDependencies(OpAsmPrinter &printer, Operation *op,
                                   Type asyncTokenType,
                                   OperandRange asyncDependencies) {
  if (asyncTokenType)
    printer << "async";
  if (asyncDependencies.empty())
    return;
  if (asyncTokenType)
    printer << ' ';
  printer << '[';
  llvm::interleaveComma(asyncDependencies, printer);
  printer << ']';
}

//===----------------------------------------------------------------------===//
// AllReduceOp
//===----------------------------------------------------------------------===//

LogicalResult gpu::AllReduceOp::verifyRegions() {
  if (getBody().empty() != getOp().has_value())
    return emitError("expected either an op attribute or a non-empty body");
  if (!getBody().empty()) {
    if (getBody().getNumArguments() != 2)
      return emitError("expected two region arguments");
    for (auto argument : getBody().getArguments()) {
      if (argument.getType() != getType())
        return emitError("incorrect region argument type");
    }
    unsigned yieldCount = 0;
    for (Block &block : getBody()) {
      if (auto yield = dyn_cast<gpu::YieldOp>(block.getTerminator())) {
        if (yield.getNumOperands() != 1)
          return emitError("expected one gpu.yield operand");
        if (yield.getOperand(0).getType() != getType())
          return emitError("incorrect gpu.yield type");
        ++yieldCount;
      }
    }
    if (yieldCount == 0)
      return emitError("expected gpu.yield op in region");
  } else {
    gpu::AllReduceOperation opName = *getOp();
    if ((opName == gpu::AllReduceOperation::AND ||
         opName == gpu::AllReduceOperation::OR ||
         opName == gpu::AllReduceOperation::XOR) &&
        !getType().isa<IntegerType>()) {
      return emitError()
             << '`' << gpu::stringifyAllReduceOperation(opName)
             << "` accumulator is only compatible with Integer type";
    }
  }
  return success();
}

// TODO: Support optional custom attributes (without dialect prefix).
static ParseResult parseAllReduceOperation(AsmParser &parser,
                                           AllReduceOperationAttr &attr) {
  StringRef enumStr;
  if (!parser.parseOptionalKeyword(&enumStr)) {
    Optional<AllReduceOperation> op = gpu::symbolizeAllReduceOperation(enumStr);
    if (!op)
      return parser.emitError(parser.getCurrentLocation(), "invalid op kind");
    attr = AllReduceOperationAttr::get(parser.getContext(), *op);
  }
  return success();
}

static void printAllReduceOperation(AsmPrinter &printer, Operation *op,
                                    AllReduceOperationAttr attr) {
  if (attr)
    attr.print(printer);
}

//===----------------------------------------------------------------------===//
// AsyncOpInterface
//===----------------------------------------------------------------------===//

void gpu::addAsyncDependency(Operation *op, Value token) {
  op->insertOperands(0, {token});
  if (!op->template hasTrait<OpTrait::AttrSizedOperandSegments>())
    return;
  auto attrName =
      OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr();
  auto sizeAttr = op->template getAttrOfType<DenseI32ArrayAttr>(attrName);

  // Async dependencies is the only variadic operand.
  if (!sizeAttr)
    return;

  SmallVector<int32_t, 8> sizes(sizeAttr.asArrayRef());
  ++sizes.front();
  op->setAttr(attrName, Builder(op->getContext()).getDenseI32ArrayAttr(sizes));
}

//===----------------------------------------------------------------------===//
// LaunchOp
//===----------------------------------------------------------------------===//

void LaunchOp::build(OpBuilder &builder, OperationState &result,
                     Value gridSizeX, Value gridSizeY, Value gridSizeZ,
                     Value getBlockSizeX, Value getBlockSizeY,
                     Value getBlockSizeZ, Value dynamicSharedMemorySize,
                     Type asyncTokenType, ValueRange asyncDependencies) {
  result.addOperands(asyncDependencies);
  if (asyncTokenType)
    result.types.push_back(builder.getType<AsyncTokenType>());

  // Add grid and block sizes as op operands, followed by the data operands.
  result.addOperands({gridSizeX, gridSizeY, gridSizeZ, getBlockSizeX,
                      getBlockSizeY, getBlockSizeZ});
  if (dynamicSharedMemorySize)
    result.addOperands(dynamicSharedMemorySize);

  // Create a kernel body region with kNumConfigRegionAttributes + N arguments,
  // where the first kNumConfigRegionAttributes arguments have `index` type and
  // the rest have the same types as the data operands.
  Region *kernelRegion = result.addRegion();
  Block *body = new Block();
  for (unsigned i = 0; i < kNumConfigRegionAttributes; ++i)
    body->addArgument(builder.getIndexType(), result.location);
  kernelRegion->push_back(body);
  SmallVector<int32_t, 8> segmentSizes(8, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes.back() = dynamicSharedMemorySize ? 1 : 0;
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(segmentSizes));
}

KernelDim3 LaunchOp::getBlockIds() {
  assert(!getBody().empty() && "LaunchOp body must not be empty.");
  auto args = getBody().getArguments();
  return KernelDim3{args[0], args[1], args[2]};
}

KernelDim3 LaunchOp::getThreadIds() {
  assert(!getBody().empty() && "LaunchOp body must not be empty.");
  auto args = getBody().getArguments();
  return KernelDim3{args[3], args[4], args[5]};
}

KernelDim3 LaunchOp::getGridSize() {
  assert(!getBody().empty() && "LaunchOp body must not be empty.");
  auto args = getBody().getArguments();
  return KernelDim3{args[6], args[7], args[8]};
}

KernelDim3 LaunchOp::getBlockSize() {
  assert(!getBody().empty() && "LaunchOp body must not be empty.");
  auto args = getBody().getArguments();
  return KernelDim3{args[9], args[10], args[11]};
}

KernelDim3 LaunchOp::getGridSizeOperandValues() {
  auto operands = getOperands().drop_front(getAsyncDependencies().size());
  return KernelDim3{operands[0], operands[1], operands[2]};
}

KernelDim3 LaunchOp::getBlockSizeOperandValues() {
  auto operands = getOperands().drop_front(getAsyncDependencies().size());
  return KernelDim3{operands[3], operands[4], operands[5]};
}

LogicalResult LaunchOp::verifyRegions() {
  // Kernel launch takes kNumConfigOperands leading operands for grid/block
  // sizes and transforms them into kNumConfigRegionAttributes region arguments
  // for block/thread identifiers and grid/block sizes.
  if (!getBody().empty()) {
    if (getBody().getNumArguments() !=
        LaunchOp::kNumConfigOperands + getNumOperands() -
            (getDynamicSharedMemorySize() ? 1 : 0) -
            getAsyncDependencies().size())
      return emitOpError("unexpected number of region arguments");
  }

  // Block terminators without successors are expected to exit the kernel region
  // and must be `gpu.terminator`.
  for (Block &block : getBody()) {
    if (block.empty())
      continue;
    if (block.back().getNumSuccessors() != 0)
      continue;
    if (!isa<gpu::TerminatorOp>(&block.back())) {
      return block.back()
          .emitError()
          .append("expected '", gpu::TerminatorOp::getOperationName(),
                  "' or a terminator with successors")
          .attachNote(getLoc())
          .append("in '", LaunchOp::getOperationName(), "' body region");
    }
  }

  if (getNumResults() == 0 && getAsyncToken())
    return emitOpError("needs to be named when async keyword is specified");

  return success();
}

// Pretty-print the kernel grid/block size assignment as
//   (%iter-x, %iter-y, %iter-z) in
//   (%size-x = %ssa-use, %size-y = %ssa-use, %size-z = %ssa-use)
// where %size-* and %iter-* will correspond to the body region arguments.
static void printSizeAssignment(OpAsmPrinter &p, KernelDim3 size,
                                KernelDim3 operands, KernelDim3 ids) {
  p << '(' << ids.x << ", " << ids.y << ", " << ids.z << ") in (";
  p << size.x << " = " << operands.x << ", ";
  p << size.y << " = " << operands.y << ", ";
  p << size.z << " = " << operands.z << ')';
}

void LaunchOp::print(OpAsmPrinter &p) {
  if (getAsyncToken()) {
    p << " async";
    if (!getAsyncDependencies().empty())
      p << " [" << getAsyncDependencies() << ']';
  }
  // Print the launch configuration.
  p << ' ' << getBlocksKeyword();
  printSizeAssignment(p, getGridSize(), getGridSizeOperandValues(),
                      getBlockIds());
  p << ' ' << getThreadsKeyword();
  printSizeAssignment(p, getBlockSize(), getBlockSizeOperandValues(),
                      getThreadIds());
  if (getDynamicSharedMemorySize())
    p << ' ' << getDynamicSharedMemorySizeKeyword() << ' '
      << getDynamicSharedMemorySize();

  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{
                              LaunchOp::getOperandSegmentSizeAttr()});
}

// Parse the size assignment blocks for blocks and threads.  These have the form
//   (%region_arg, %region_arg, %region_arg) in
//   (%region_arg = %operand, %region_arg = %operand, %region_arg = %operand)
// where %region_arg are percent-identifiers for the region arguments to be
// introduced further (SSA defs), and %operand are percent-identifiers for the
// SSA value uses.
static ParseResult
parseSizeAssignment(OpAsmParser &parser,
                    MutableArrayRef<OpAsmParser::UnresolvedOperand> sizes,
                    MutableArrayRef<OpAsmParser::UnresolvedOperand> regionSizes,
                    MutableArrayRef<OpAsmParser::UnresolvedOperand> indices) {
  assert(indices.size() == 3 && "space for three indices expected");
  SmallVector<OpAsmParser::UnresolvedOperand, 3> args;
  if (parser.parseOperandList(args, OpAsmParser::Delimiter::Paren,
                              /*allowResultNumber=*/false) ||
      parser.parseKeyword("in") || parser.parseLParen())
    return failure();
  std::move(args.begin(), args.end(), indices.begin());

  for (int i = 0; i < 3; ++i) {
    if (i != 0 && parser.parseComma())
      return failure();
    if (parser.parseOperand(regionSizes[i], /*allowResultNumber=*/false) ||
        parser.parseEqual() || parser.parseOperand(sizes[i]))
      return failure();
  }

  return parser.parseRParen();
}

/// Parses a Launch operation.
/// operation ::= `gpu.launch` (`async` `[` ssa-id-list `]`)?
//        `blocks` `(` ssa-id-list `)` `in` ssa-reassignment
///       `threads` `(` ssa-id-list `)` `in` ssa-reassignment
///       region attr-dict?
/// ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
ParseResult LaunchOp::parse(OpAsmParser &parser, OperationState &result) {
  // Sizes of the grid and block.
  SmallVector<OpAsmParser::UnresolvedOperand, LaunchOp::kNumConfigOperands>
      sizes(LaunchOp::kNumConfigOperands);
  MutableArrayRef<OpAsmParser::UnresolvedOperand> sizesRef(sizes);

  // Actual (data) operands passed to the kernel.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> dataOperands;

  // Region arguments to be created.
  SmallVector<OpAsmParser::UnresolvedOperand, 16> regionArgs(
      LaunchOp::kNumConfigRegionAttributes);
  MutableArrayRef<OpAsmParser::UnresolvedOperand> regionArgsRef(regionArgs);

  // Parse optional async dependencies.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> asyncDependencies;
  Type asyncTokenType;
  if (failed(
          parseAsyncDependencies(parser, asyncTokenType, asyncDependencies)) ||
      parser.resolveOperands(asyncDependencies, asyncTokenType,
                             result.operands))
    return failure();
  if (parser.getNumResults() > 0)
    result.types.push_back(asyncTokenType);

  // Parse the size assignment segments: the first segment assigns grid sizes
  // and defines values for block identifiers; the second segment assigns block
  // sizes and defines values for thread identifiers.  In the region argument
  // list, identifiers precede sizes, and block-related values precede
  // thread-related values.
  if (parser.parseKeyword(LaunchOp::getBlocksKeyword().data()) ||
      parseSizeAssignment(parser, sizesRef.take_front(3),
                          regionArgsRef.slice(6, 3),
                          regionArgsRef.slice(0, 3)) ||
      parser.parseKeyword(LaunchOp::getThreadsKeyword().data()) ||
      parseSizeAssignment(parser, sizesRef.drop_front(3),
                          regionArgsRef.slice(9, 3),
                          regionArgsRef.slice(3, 3)) ||
      parser.resolveOperands(sizes, parser.getBuilder().getIndexType(),
                             result.operands))
    return failure();

  OpAsmParser::UnresolvedOperand dynamicSharedMemorySize;
  bool hasDynamicSharedMemorySize = false;
  if (!parser.parseOptionalKeyword(
          LaunchOp::getDynamicSharedMemorySizeKeyword())) {
    hasDynamicSharedMemorySize = true;
    if (parser.parseOperand(dynamicSharedMemorySize) ||
        parser.resolveOperand(dynamicSharedMemorySize,
                              parser.getBuilder().getI32Type(),
                              result.operands))
      return failure();
  }

  // Introduce the body region and parse it. The region has
  // kNumConfigRegionAttributes arguments that correspond to
  // block/thread identifiers and grid/block sizes, all of the `index` type.
  Type index = parser.getBuilder().getIndexType();
  SmallVector<Type, LaunchOp::kNumConfigRegionAttributes> dataTypes(
      LaunchOp::kNumConfigRegionAttributes, index);

  SmallVector<OpAsmParser::Argument> regionArguments;
  for (auto ssaValueAndType : llvm::zip(regionArgs, dataTypes)) {
    OpAsmParser::Argument arg;
    arg.ssaName = std::get<0>(ssaValueAndType);
    arg.type = std::get<1>(ssaValueAndType);
    regionArguments.push_back(arg);
  }

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArguments) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<int32_t, 8> segmentSizes(8, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes.back() = hasDynamicSharedMemorySize ? 1 : 0;
  result.addAttribute(LaunchOp::getOperandSegmentSizeAttr(),
                      parser.getBuilder().getDenseI32ArrayAttr(segmentSizes));
  return success();
}

/// Simplify the gpu.launch when the range of a thread or block ID is
/// trivially known to be one.
struct FoldLaunchArguments : public OpRewritePattern<LaunchOp> {
  using OpRewritePattern<LaunchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(LaunchOp op,
                                PatternRewriter &rewriter) const override {
    // If the range implies a single value for `id`, replace `id`'s uses by
    // zero.
    Value zero;
    bool simplified = false;
    auto constPropIdUses = [&](Value id, Value size) {
      // Check if size is trivially one.
      if (!matchPattern(size, m_One()))
        return;
      if (!simplified) {
        // Create a zero value the first time.
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&op.getBody().front());
        zero =
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), /*value=*/0);
      }
      id.replaceAllUsesWith(zero);
      simplified = true;
    };
    constPropIdUses(op.getBlockIds().x, op.getGridSizeX());
    constPropIdUses(op.getBlockIds().y, op.getGridSizeY());
    constPropIdUses(op.getBlockIds().z, op.getGridSizeZ());
    constPropIdUses(op.getThreadIds().x, op.getBlockSizeX());
    constPropIdUses(op.getThreadIds().y, op.getBlockSizeY());
    constPropIdUses(op.getThreadIds().z, op.getBlockSizeZ());

    return success(simplified);
  }
};

void LaunchOp::getCanonicalizationPatterns(RewritePatternSet &rewrites,
                                           MLIRContext *context) {
  rewrites.add<FoldLaunchArguments>(context);
}

//===----------------------------------------------------------------------===//
// LaunchFuncOp
//===----------------------------------------------------------------------===//

void LaunchFuncOp::build(OpBuilder &builder, OperationState &result,
                         GPUFuncOp kernelFunc, KernelDim3 gridSize,
                         KernelDim3 getBlockSize, Value dynamicSharedMemorySize,
                         ValueRange kernelOperands, Type asyncTokenType,
                         ValueRange asyncDependencies) {
  result.addOperands(asyncDependencies);
  if (asyncTokenType)
    result.types.push_back(builder.getType<AsyncTokenType>());

  // Add grid and block sizes as op operands, followed by the data operands.
  result.addOperands({gridSize.x, gridSize.y, gridSize.z, getBlockSize.x,
                      getBlockSize.y, getBlockSize.z});
  if (dynamicSharedMemorySize)
    result.addOperands(dynamicSharedMemorySize);
  result.addOperands(kernelOperands);
  auto kernelModule = kernelFunc->getParentOfType<GPUModuleOp>();
  auto kernelSymbol =
      SymbolRefAttr::get(kernelModule.getNameAttr(),
                         {SymbolRefAttr::get(kernelFunc.getNameAttr())});
  result.addAttribute(getKernelAttrName(result.name), kernelSymbol);
  SmallVector<int32_t, 9> segmentSizes(9, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes[segmentSizes.size() - 2] = dynamicSharedMemorySize ? 1 : 0;
  segmentSizes.back() = static_cast<int32_t>(kernelOperands.size());
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(segmentSizes));
}

StringAttr LaunchFuncOp::getKernelModuleName() {
  return getKernel().getRootReference();
}

StringAttr LaunchFuncOp::getKernelName() {
  return getKernel().getLeafReference();
}

unsigned LaunchFuncOp::getNumKernelOperands() {
  return getKernelOperands().size();
}

Value LaunchFuncOp::getKernelOperand(unsigned i) {
  return getKernelOperands()[i];
}

KernelDim3 LaunchFuncOp::getGridSizeOperandValues() {
  auto operands = getOperands().drop_front(getAsyncDependencies().size());
  return KernelDim3{operands[0], operands[1], operands[2]};
}

KernelDim3 LaunchFuncOp::getBlockSizeOperandValues() {
  auto operands = getOperands().drop_front(getAsyncDependencies().size());
  return KernelDim3{operands[3], operands[4], operands[5]};
}

LogicalResult LaunchFuncOp::verify() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError("expected to belong to a module");

  if (!module->getAttrOfType<UnitAttr>(
          GPUDialect::getContainerModuleAttrName()))
    return emitOpError("expected the closest surrounding module to have the '" +
                       GPUDialect::getContainerModuleAttrName() +
                       "' attribute");

  return success();
}

static ParseResult parseLaunchFuncOperands(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &argNames,
    SmallVectorImpl<Type> &argTypes) {
  if (parser.parseOptionalKeyword("args"))
    return success();

  SmallVector<OpAsmParser::Argument> args;
  if (parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true))
    return failure();
  for (auto &arg : args) {
    argNames.push_back(arg.ssaName);
    argTypes.push_back(arg.type);
  }
  return success();
}

static void printLaunchFuncOperands(OpAsmPrinter &printer, Operation *,
                                    OperandRange operands, TypeRange types) {
  if (operands.empty())
    return;
  printer << "args(";
  llvm::interleaveComma(llvm::zip(operands, types), printer,
                        [&](const auto &pair) {
                          printer.printOperand(std::get<0>(pair));
                          printer << " : ";
                          printer.printType(std::get<1>(pair));
                        });
  printer << ")";
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//

void ShuffleOp::build(OpBuilder &builder, OperationState &result, Value value,
                      int32_t offset, int32_t width, ShuffleMode mode) {
  build(builder, result, value,
        builder.create<arith::ConstantOp>(result.location,
                                          builder.getI32IntegerAttr(offset)),
        builder.create<arith::ConstantOp>(result.location,
                                          builder.getI32IntegerAttr(width)),
        mode);
}

//===----------------------------------------------------------------------===//
// GPUFuncOp
//===----------------------------------------------------------------------===//

/// Adds a new block argument that corresponds to buffers located in
/// workgroup memory.
BlockArgument GPUFuncOp::addWorkgroupAttribution(Type type, Location loc) {
  auto attrName = getNumWorkgroupAttributionsAttrName();
  auto attr = (*this)->getAttrOfType<IntegerAttr>(attrName);
  (*this)->setAttr(attrName,
                   IntegerAttr::get(attr.getType(), attr.getValue() + 1));
  return getBody().insertArgument(
      getFunctionType().getNumInputs() + attr.getInt(), type, loc);
}

/// Adds a new block argument that corresponds to buffers located in
/// private memory.
BlockArgument GPUFuncOp::addPrivateAttribution(Type type, Location loc) {
  // Buffers on the private memory always come after buffers on the workgroup
  // memory.
  return getBody().addArgument(type, loc);
}

void GPUFuncOp::build(OpBuilder &builder, OperationState &result,
                      StringRef name, FunctionType type,
                      TypeRange workgroupAttributions,
                      TypeRange privateAttributions,
                      ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  result.addAttribute(getNumWorkgroupAttributionsAttrName(),
                      builder.getI64IntegerAttr(workgroupAttributions.size()));
  result.addAttributes(attrs);
  Region *body = result.addRegion();
  Block *entryBlock = new Block;

  // TODO: Allow passing in proper locations here.
  for (Type argTy : type.getInputs())
    entryBlock->addArgument(argTy, result.location);
  for (Type argTy : workgroupAttributions)
    entryBlock->addArgument(argTy, result.location);
  for (Type argTy : privateAttributions)
    entryBlock->addArgument(argTy, result.location);

  body->getBlocks().push_back(entryBlock);
}

/// Parses a GPU function memory attribution.
///
/// memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
///                        (`private` `(` ssa-id-and-type-list `)`)?
///
/// Note that this function parses only one of the two similar parts, with the
/// keyword provided as argument.
static ParseResult
parseAttributions(OpAsmParser &parser, StringRef keyword,
                  SmallVectorImpl<OpAsmParser::Argument> &args) {
  // If we could not parse the keyword, just assume empty list and succeed.
  if (failed(parser.parseOptionalKeyword(keyword)))
    return success();

  return parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                                  /*allowType=*/true);
}

/// Parses a GPU function.
///
/// <operation> ::= `gpu.func` symbol-ref-id `(` argument-list `)`
///                 (`->` function-result-list)? memory-attribution `kernel`?
///                 function-attributes? region
ParseResult GPUFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  bool isVariadic;

  // Parse the function name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  auto signatureLocation = parser.getCurrentLocation();
  if (failed(function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, entryArgs, isVariadic, resultTypes,
          resultAttrs)))
    return failure();

  if (!entryArgs.empty() && entryArgs[0].ssaName.name.empty())
    return parser.emitError(signatureLocation)
           << "gpu.func requires named arguments";

  // Construct the function type. More types will be added to the region, but
  // not to the function type.
  Builder &builder = parser.getBuilder();

  SmallVector<Type> argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(GPUFuncOp::getTypeAttrName(), TypeAttr::get(type));

  function_interface_impl::addArgAndResultAttrs(builder, result, entryArgs,
                                                resultAttrs);

  // Parse workgroup memory attributions.
  if (failed(parseAttributions(parser, GPUFuncOp::getWorkgroupKeyword(),
                               entryArgs)))
    return failure();

  // Store the number of operands we just parsed as the number of workgroup
  // memory attributions.
  unsigned numWorkgroupAttrs = entryArgs.size() - type.getNumInputs();
  result.addAttribute(GPUFuncOp::getNumWorkgroupAttributionsAttrName(),
                      builder.getI64IntegerAttr(numWorkgroupAttrs));

  // Parse private memory attributions.
  if (failed(
          parseAttributions(parser, GPUFuncOp::getPrivateKeyword(), entryArgs)))
    return failure();

  // Parse the kernel attribute if present.
  if (succeeded(parser.parseOptionalKeyword(GPUFuncOp::getKernelKeyword())))
    result.addAttribute(GPUDialect::getKernelFuncAttrName(),
                        builder.getUnitAttr());

  // Parse attributes.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();

  // Parse the region. If no argument names were provided, take all names
  // (including those of attributions) from the entry block.
  auto *body = result.addRegion();
  return parser.parseRegion(*body, entryArgs);
}

static void printAttributions(OpAsmPrinter &p, StringRef keyword,
                              ArrayRef<BlockArgument> values) {
  if (values.empty())
    return;

  p << ' ' << keyword << '(';
  llvm::interleaveComma(
      values, p, [&p](BlockArgument v) { p << v << " : " << v.getType(); });
  p << ')';
}

void GPUFuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getName());

  FunctionType type = getFunctionType();
  function_interface_impl::printFunctionSignature(p, *this, type.getInputs(),
                                                  /*isVariadic=*/false,
                                                  type.getResults());

  printAttributions(p, getWorkgroupKeyword(), getWorkgroupAttributions());
  printAttributions(p, getPrivateKeyword(), getPrivateAttributions());
  if (isKernel())
    p << ' ' << getKernelKeyword();

  function_interface_impl::printFunctionAttributes(
      p, *this, type.getNumInputs(), type.getNumResults(),
      {getNumWorkgroupAttributionsAttrName(),
       GPUDialect::getKernelFuncAttrName()});
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult GPUFuncOp::verifyType() {
  Type type = getFunctionTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");

  if (isKernel() && getFunctionType().getNumResults() != 0)
    return emitOpError() << "expected void return type for kernel function";

  return success();
}

static LogicalResult verifyAttributions(Operation *op,
                                        ArrayRef<BlockArgument> attributions,
                                        unsigned memorySpace) {
  for (Value v : attributions) {
    auto type = v.getType().dyn_cast<MemRefType>();
    if (!type)
      return op->emitOpError() << "expected memref type in attribution";

    if (type.getMemorySpaceAsInt() != memorySpace) {
      return op->emitOpError()
             << "expected memory space " << memorySpace << " in attribution";
    }
  }
  return success();
}

/// Verifies the body of the function.
LogicalResult GPUFuncOp::verifyBody() {
  if (empty())
    return emitOpError() << "expected body with at least one block";
  unsigned numFuncArguments = getNumArguments();
  unsigned numWorkgroupAttributions = getNumWorkgroupAttributions();
  unsigned numBlockArguments = front().getNumArguments();
  if (numBlockArguments < numFuncArguments + numWorkgroupAttributions)
    return emitOpError() << "expected at least "
                         << numFuncArguments + numWorkgroupAttributions
                         << " arguments to body region";

  ArrayRef<Type> funcArgTypes = getFunctionType().getInputs();
  for (unsigned i = 0; i < numFuncArguments; ++i) {
    Type blockArgType = front().getArgument(i).getType();
    if (funcArgTypes[i] != blockArgType)
      return emitOpError() << "expected body region argument #" << i
                           << " to be of type " << funcArgTypes[i] << ", got "
                           << blockArgType;
  }

  if (failed(verifyAttributions(getOperation(), getWorkgroupAttributions(),
                                GPUDialect::getWorkgroupAddressSpace())) ||
      failed(verifyAttributions(getOperation(), getPrivateAttributions(),
                                GPUDialect::getPrivateAddressSpace())))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult gpu::ReturnOp::verify() {
  GPUFuncOp function = (*this)->getParentOfType<GPUFuncOp>();

  FunctionType funType = function.getFunctionType();

  if (funType.getNumResults() != operands().size())
    return emitOpError()
        .append("expected ", funType.getNumResults(), " result operands")
        .attachNote(function.getLoc())
        .append("return type declared here");

  for (const auto &pair : llvm::enumerate(
           llvm::zip(function.getFunctionType().getResults(), operands()))) {
    auto [type, operand] = pair.value();
    if (type != operand.getType())
      return emitOpError() << "unexpected type `" << operand.getType()
                           << "' for operand #" << pair.index();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GPUModuleOp
//===----------------------------------------------------------------------===//

void GPUModuleOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name) {
  ensureTerminator(*result.addRegion(), builder, result.location);
  result.attributes.push_back(builder.getNamedAttr(
      ::mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
}

ParseResult GPUModuleOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      // If module attributes are present, parse them.
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the module body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, {}))
    return failure();

  // Ensure that this module has a valid terminator.
  GPUModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

void GPUModuleOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getName());
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     {mlir::SymbolTable::getSymbolAttrName()});
  p << ' ';
  p.printRegion(getRegion(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

//===----------------------------------------------------------------------===//
// GPUMemcpyOp
//===----------------------------------------------------------------------===//

LogicalResult MemcpyOp::verify() {
  auto srcType = getSrc().getType();
  auto dstType = getDst().getType();

  if (getElementTypeOrSelf(srcType) != getElementTypeOrSelf(dstType))
    return emitOpError("arguments have incompatible element type");

  if (failed(verifyCompatibleShape(srcType, dstType)))
    return emitOpError("arguments have incompatible shape");

  return success();
}

namespace {

/// Erases a common case of copy ops where a destination value is used only by
/// the copy op, alloc and dealloc ops.
struct EraseTrivialCopyOp : public OpRewritePattern<MemcpyOp> {
  using OpRewritePattern<MemcpyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemcpyOp op,
                                PatternRewriter &rewriter) const override {
    Value dest = op.getDst();
    Operation *destDefOp = dest.getDefiningOp();
    // `dest` must be defined by an op having Allocate memory effect in order to
    // perform the folding.
    if (!destDefOp ||
        !hasSingleEffect<MemoryEffects::Allocate>(destDefOp, dest))
      return failure();
    // We can erase `op` iff `dest` has no other use apart from its
    // use by `op` and dealloc ops.
    if (llvm::any_of(dest.getUsers(), [op, dest](Operation *user) {
          return user != op &&
                 !hasSingleEffect<MemoryEffects::Free>(user, dest);
        }))
      return failure();
    // We can perform the folding if and only if op has a single async
    // dependency and produces an async token as result, or if it does not have
    // any async dependency and does not produce any async token result.
    if (op.getAsyncDependencies().size() > 1 ||
        ((op.getAsyncDependencies().empty() && op.getAsyncToken()) ||
         (!op.getAsyncDependencies().empty() && !op.getAsyncToken())))
      return failure();
    rewriter.replaceOp(op, op.getAsyncDependencies());
    return success();
  }
};

} // end anonymous namespace

void MemcpyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<EraseTrivialCopyOp>(context);
}

//===----------------------------------------------------------------------===//
// GPU_SubgroupMmaLoadMatrixOp
//===----------------------------------------------------------------------===//

/// Return true if the last dimension of the MemRefType has unit stride. Also
/// return true for memrefs with no strides.
static bool isLastMemrefDimUnitStride(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(getStridesAndOffset(type, strides, offset))) {
    return false;
  }
  return strides.back() == 1;
}

LogicalResult SubgroupMmaLoadMatrixOp::verify() {
  auto srcType = getSrcMemref().getType();
  auto resType = getRes().getType();
  auto resMatrixType = resType.cast<gpu::MMAMatrixType>();
  auto operand = resMatrixType.getOperand();
  auto srcMemrefType = srcType.cast<MemRefType>();
  auto srcMemSpace = srcMemrefType.getMemorySpaceAsInt();

  if (!isLastMemrefDimUnitStride(srcMemrefType))
    return emitError(
        "expected source memref most minor dim must have unit stride");

  if (srcMemSpace != kGenericMemorySpace && srcMemSpace != kSharedMemorySpace &&
      srcMemSpace != kGlobalMemorySpace)
    return emitError(
        "source memorySpace kGenericMemorySpace, kSharedMemorySpace or "
        "kGlobalMemorySpace only allowed");

  if (!operand.equals("AOp") && !operand.equals("BOp") &&
      !operand.equals("COp"))
    return emitError("only AOp, BOp and COp can be loaded");

  return success();
}

//===----------------------------------------------------------------------===//
// GPU_SubgroupMmaStoreMatrixOp
//===----------------------------------------------------------------------===//

LogicalResult SubgroupMmaStoreMatrixOp::verify() {
  auto srcType = getSrc().getType();
  auto dstType = getDstMemref().getType();
  auto srcMatrixType = srcType.cast<gpu::MMAMatrixType>();
  auto dstMemrefType = dstType.cast<MemRefType>();
  auto dstMemSpace = dstMemrefType.getMemorySpaceAsInt();

  if (!isLastMemrefDimUnitStride(dstMemrefType))
    return emitError(
        "expected destination memref most minor dim must have unit stride");

  if (dstMemSpace != kGenericMemorySpace && dstMemSpace != kSharedMemorySpace &&
      dstMemSpace != kGlobalMemorySpace)
    return emitError("destination memorySpace of kGenericMemorySpace, "
                     "kGlobalMemorySpace or kSharedMemorySpace only allowed");

  if (!srcMatrixType.getOperand().equals("COp"))
    return emitError(
        "expected the operand matrix being stored to have 'COp' operand type");

  return success();
}

//===----------------------------------------------------------------------===//
// GPU_SubgroupMmaComputeOp
//===----------------------------------------------------------------------===//

LogicalResult SubgroupMmaComputeOp::verify() {
  enum OperandMap { A, B, C };
  SmallVector<MMAMatrixType, 3> opTypes;
  opTypes.push_back(getOpA().getType().cast<MMAMatrixType>());
  opTypes.push_back(getOpB().getType().cast<MMAMatrixType>());
  opTypes.push_back(getOpC().getType().cast<MMAMatrixType>());

  if (!opTypes[A].getOperand().equals("AOp") ||
      !opTypes[B].getOperand().equals("BOp") ||
      !opTypes[C].getOperand().equals("COp"))
    return emitError("operands must be in the order AOp, BOp, COp");

  ArrayRef<int64_t> aShape, bShape, cShape;
  aShape = opTypes[A].getShape();
  bShape = opTypes[B].getShape();
  cShape = opTypes[C].getShape();

  if (aShape[1] != bShape[0] || aShape[0] != cShape[0] ||
      bShape[1] != cShape[1])
    return emitError("operand shapes do not satisfy matmul constraints");

  return success();
}

/// This is a common class used for patterns of the form
/// "someop(memrefcast) -> someop".  It folds the source of any memref.cast
/// into the root operation directly.
static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto cast = operand.get().getDefiningOp<mlir::memref::CastOp>();
    if (cast) {
      operand.set(cast.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

LogicalResult MemcpyOp::fold(ArrayRef<Attribute> operands,
                             SmallVectorImpl<::mlir::OpFoldResult> &results) {
  return foldMemRefCast(*this);
}

LogicalResult MemsetOp::fold(ArrayRef<Attribute> operands,
                             SmallVectorImpl<::mlir::OpFoldResult> &results) {
  return foldMemRefCast(*this);
}

//===----------------------------------------------------------------------===//
// GPU_WaitOp
//===----------------------------------------------------------------------===//

namespace {

/// Remove gpu.wait op use of gpu.wait op def without async dependencies.
/// %t = gpu.wait async []       // No async dependencies.
/// ...  gpu.wait ... [%t, ...]  // %t can be removed.
struct EraseRedundantGpuWaitOpPairs : public OpRewritePattern<WaitOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WaitOp op,
                                PatternRewriter &rewriter) const final {
    auto predicate = [](Value value) {
      auto waitOp = value.getDefiningOp<WaitOp>();
      return waitOp && waitOp->getNumOperands() == 0;
    };
    if (llvm::none_of(op.getAsyncDependencies(), predicate))
      return failure();
    SmallVector<Value> validOperands;
    for (Value operand : op->getOperands()) {
      if (predicate(operand))
        continue;
      validOperands.push_back(operand);
    }
    op->setOperands(validOperands);
    return success();
  }
};

/// Simplify trivial gpu.wait ops for the following patterns.
/// 1. %t = gpu.wait async ... ops, where %t has no uses (regardless of async
/// dependencies).
/// 2. %t1 = gpu.wait async [%t0], in this case, we can replace uses of %t1 with
/// %t0.
/// 3. gpu.wait [] ops, i.e gpu.wait ops that neither have any async
/// dependencies nor return any token.
struct SimplifyGpuWaitOp : public OpRewritePattern<WaitOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WaitOp op,
                                PatternRewriter &rewriter) const final {
    // Erase gpu.wait ops that neither have any async dependencies nor return
    // any async token.
    if (op.getAsyncDependencies().empty() && !op.getAsyncToken()) {
      rewriter.eraseOp(op);
      return success();
    }
    // Replace uses of %t1 = gpu.wait async [%t0] ops with %t0 and erase the op.
    if (llvm::hasSingleElement(op.getAsyncDependencies()) &&
        op.getAsyncToken()) {
      rewriter.replaceOp(op, op.getAsyncDependencies());
      return success();
    }
    // Erase %t = gpu.wait async ... ops, where %t has no uses.
    if (op.getAsyncToken() && op.getAsyncToken().use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

} // end anonymous namespace

void WaitOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results.add<EraseRedundantGpuWaitOpPairs, SimplifyGpuWaitOp>(context);
}

//===----------------------------------------------------------------------===//
// GPU_AllocOp
//===----------------------------------------------------------------------===//

LogicalResult AllocOp::verify() {
  auto memRefType = getMemref().getType().cast<MemRefType>();

  if (static_cast<int64_t>(getDynamicSizes().size()) !=
      memRefType.getNumDynamicDims())
    return emitOpError("dimension operand count does not equal memref "
                       "dynamic dimension count");

  unsigned numSymbols = 0;
  if (!memRefType.getLayout().isIdentity())
    numSymbols = memRefType.getLayout().getAffineMap().getNumSymbols();
  if (getSymbolOperands().size() != numSymbols) {
    return emitOpError(
        "symbol operand count does not equal memref symbol count");
  }

  return success();
}

namespace {

/// Folding of memref.dim(gpu.alloc(%size), %idx) -> %size similar to
/// `memref::AllocOp`.
struct SimplifyDimOfAllocOp : public OpRewritePattern<memref::DimOp> {
  using OpRewritePattern<memref::DimOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::DimOp dimOp,
                                PatternRewriter &rewriter) const override {
    auto index = dimOp.getIndex().getDefiningOp<arith::ConstantIndexOp>();
    if (!index)
      return failure();

    auto memrefType = dimOp.getSource().getType().dyn_cast<MemRefType>();
    if (!memrefType || !memrefType.isDynamicDim(index.value()))
      return failure();

    auto alloc = dimOp.getSource().getDefiningOp<AllocOp>();
    if (!alloc)
      return failure();

    Value substituteOp = *(alloc.getDynamicSizes().begin() +
                           memrefType.getDynamicDimIndex(index.value()));
    rewriter.replaceOp(dimOp, substituteOp);
    return success();
  }
};

} // namespace

void AllocOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                          MLIRContext *context) {
  results.add<SimplifyDimOfAllocOp>(context);
}

#include "mlir/Dialect/GPU/IR/GPUOpInterfaces.cpp.inc"
#include "mlir/Dialect/GPU/IR/GPUOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/GPU/IR/GPUOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/IR/GPUOps.cpp.inc"
