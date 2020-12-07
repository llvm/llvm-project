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

#include "mlir/Dialect/GPU/GPUDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::gpu;

//===----------------------------------------------------------------------===//
// GPUDialect
//===----------------------------------------------------------------------===//

bool GPUDialect::isKernel(Operation *op) {
  UnitAttr isKernelAttr = op->getAttrOfType<UnitAttr>(getKernelFuncAttrName());
  return static_cast<bool>(isKernelAttr);
}

void GPUDialect::initialize() {
  addTypes<AsyncTokenType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/GPU/GPUOps.cpp.inc"
      >();
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

  parser.emitError(parser.getNameLoc(), "unknown gpu type: " + keyword);
  return Type();
}

void GPUDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<AsyncTokenType>([&](Type) { os << "async.token"; })
      .Default([](Type) { llvm_unreachable("unexpected 'gpu' type kind"); });
}

LogicalResult GPUDialect::verifyOperationAttribute(Operation *op,
                                                   NamedAttribute attr) {
  if (!attr.second.isa<UnitAttr>() ||
      attr.first != getContainerModuleAttrName())
    return success();

  auto module = dyn_cast<ModuleOp>(op);
  if (!module)
    return op->emitError("expected '")
           << getContainerModuleAttrName() << "' attribute to be attached to '"
           << ModuleOp::getOperationName() << '\'';

  auto walkResult = module.walk([&module](LaunchFuncOp launchOp) -> WalkResult {
    // Ignore launches that are nested more or less deep than functions in the
    // module we are currently checking.
    if (!launchOp.getParentOp() ||
        launchOp.getParentOp()->getParentOp() != module)
      return success();

    // Ignore launch ops with missing attributes here. The errors will be
    // reported by the verifiers of those ops.
    if (!launchOp.getAttrOfType<SymbolRefAttr>(
            LaunchFuncOp::getKernelAttrName()))
      return success();

    // Check that `launch_func` refers to a well-formed GPU kernel module.
    StringRef kernelModuleName = launchOp.getKernelModuleName();
    auto kernelModule = module.lookupSymbol<GPUModuleOp>(kernelModuleName);
    if (!kernelModule)
      return launchOp.emitOpError()
             << "kernel module '" << kernelModuleName << "' is undefined";

    // Check that `launch_func` refers to a well-formed kernel function.
    Operation *kernelFunc = module.lookupSymbol(launchOp.kernel());
    auto kernelGPUFunction = dyn_cast_or_null<gpu::GPUFuncOp>(kernelFunc);
    auto kernelLLVMFunction = dyn_cast_or_null<LLVM::LLVMFuncOp>(kernelFunc);
    if (!kernelGPUFunction && !kernelLLVMFunction)
      return launchOp.emitOpError("kernel function '")
             << launchOp.kernel() << "' is undefined";
    if (!kernelFunc->getAttrOfType<mlir::UnitAttr>(
            GPUDialect::getKernelFuncAttrName()))
      return launchOp.emitOpError("kernel function is missing the '")
             << GPUDialect::getKernelFuncAttrName() << "' attribute";

    // TODO: if the kernel function has been converted to
    // the LLVM dialect but the caller hasn't (which happens during the
    // separate compilation), do not check type correspondence as it would
    // require the verifier to be aware of the LLVM type conversion.
    if (kernelLLVMFunction)
      return success();

    unsigned actualNumArguments = launchOp.getNumKernelOperands();
    unsigned expectedNumArguments = kernelGPUFunction.getNumArguments();
    if (expectedNumArguments != actualNumArguments)
      return launchOp.emitOpError("got ")
             << actualNumArguments << " kernel operands but expected "
             << expectedNumArguments;

    auto functionType = kernelGPUFunction.getType();
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

template <typename T> static LogicalResult verifyIndexOp(T op) {
  auto dimension = op.dimension();
  if (dimension != "x" && dimension != "y" && dimension != "z")
    return op.emitError("dimension \"") << dimension << "\" is invalid";
  return success();
}

static LogicalResult verifyAllReduce(gpu::AllReduceOp allReduce) {
  if (allReduce.body().empty() != allReduce.op().hasValue())
    return allReduce.emitError(
        "expected either an op attribute or a non-empty body");
  if (!allReduce.body().empty()) {
    if (allReduce.body().getNumArguments() != 2)
      return allReduce.emitError("expected two region arguments");
    for (auto argument : allReduce.body().getArguments()) {
      if (argument.getType() != allReduce.getType())
        return allReduce.emitError("incorrect region argument type");
    }
    unsigned yieldCount = 0;
    for (Block &block : allReduce.body()) {
      if (auto yield = dyn_cast<gpu::YieldOp>(block.getTerminator())) {
        if (yield.getNumOperands() != 1)
          return allReduce.emitError("expected one gpu.yield operand");
        if (yield.getOperand(0).getType() != allReduce.getType())
          return allReduce.emitError("incorrect gpu.yield type");
        ++yieldCount;
      }
    }
    if (yieldCount == 0)
      return allReduce.emitError("expected gpu.yield op in region");
  } else {
    StringRef opName = *allReduce.op();
    if ((opName == "and" || opName == "or" || opName == "xor") &&
        !allReduce.getType().isa<IntegerType>()) {
      return allReduce.emitError()
             << '`' << opName << '`'
             << " accumulator is only compatible with Integer type";
    }
  }
  return success();
}

static LogicalResult verifyShuffleOp(gpu::ShuffleOp shuffleOp) {
  auto type = shuffleOp.value().getType();
  if (shuffleOp.result().getType() != type) {
    return shuffleOp.emitOpError()
           << "requires the same type for value operand and result";
  }
  if (!type.isSignlessIntOrFloat() || type.getIntOrFloatBitWidth() != 32) {
    return shuffleOp.emitOpError()
           << "requires value operand type to be f32 or i32";
  }
  return success();
}

static void printShuffleOp(OpAsmPrinter &p, ShuffleOp op) {
  p << ShuffleOp::getOperationName() << ' ' << op.getOperands() << ' '
    << op.mode() << " : " << op.value().getType();
}

static ParseResult parseShuffleOp(OpAsmParser &parser, OperationState &state) {
  SmallVector<OpAsmParser::OperandType, 3> operandInfo;
  if (parser.parseOperandList(operandInfo, 3))
    return failure();

  StringRef mode;
  if (parser.parseKeyword(&mode))
    return failure();
  state.addAttribute("mode", parser.getBuilder().getStringAttr(mode));

  Type valueType;
  Type int32Type = parser.getBuilder().getIntegerType(32);
  Type int1Type = parser.getBuilder().getI1Type();
  if (parser.parseColonType(valueType) ||
      parser.resolveOperands(operandInfo, {valueType, int32Type, int32Type},
                             parser.getCurrentLocation(), state.operands) ||
      parser.addTypesToList({valueType, int1Type}, state.types))
    return failure();
  return success();
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
  auto sizeAttr = op->template getAttrOfType<DenseIntElementsAttr>(attrName);
  if (!sizeAttr)
    return; // Async dependencies is the only variadic operand.
  SmallVector<int32_t, 8> sizes;
  for (auto size : sizeAttr.getIntValues())
    sizes.push_back(size.getSExtValue());
  ++sizes.front();
  op->setAttr(attrName, Builder(op->getContext()).getI32VectorAttr(sizes));
}

//===----------------------------------------------------------------------===//
// LaunchOp
//===----------------------------------------------------------------------===//

void LaunchOp::build(OpBuilder &builder, OperationState &result,
                     Value gridSizeX, Value gridSizeY, Value gridSizeZ,
                     Value blockSizeX, Value blockSizeY, Value blockSizeZ) {
  // Add grid and block sizes as op operands, followed by the data operands.
  result.addOperands(
      {gridSizeX, gridSizeY, gridSizeZ, blockSizeX, blockSizeY, blockSizeZ});

  // Create a kernel body region with kNumConfigRegionAttributes + N arguments,
  // where the first kNumConfigRegionAttributes arguments have `index` type and
  // the rest have the same types as the data operands.
  Region *kernelRegion = result.addRegion();
  Block *body = new Block();
  body->addArguments(
      std::vector<Type>(kNumConfigRegionAttributes, builder.getIndexType()));
  kernelRegion->push_back(body);
}

KernelDim3 LaunchOp::getBlockIds() {
  assert(!body().empty() && "LaunchOp body must not be empty.");
  auto args = body().getArguments();
  return KernelDim3{args[0], args[1], args[2]};
}

KernelDim3 LaunchOp::getThreadIds() {
  assert(!body().empty() && "LaunchOp body must not be empty.");
  auto args = body().getArguments();
  return KernelDim3{args[3], args[4], args[5]};
}

KernelDim3 LaunchOp::getGridSize() {
  assert(!body().empty() && "LaunchOp body must not be empty.");
  auto args = body().getArguments();
  return KernelDim3{args[6], args[7], args[8]};
}

KernelDim3 LaunchOp::getBlockSize() {
  assert(!body().empty() && "LaunchOp body must not be empty.");
  auto args = body().getArguments();
  return KernelDim3{args[9], args[10], args[11]};
}

KernelDim3 LaunchOp::getGridSizeOperandValues() {
  return KernelDim3{getOperand(0), getOperand(1), getOperand(2)};
}

KernelDim3 LaunchOp::getBlockSizeOperandValues() {
  return KernelDim3{getOperand(3), getOperand(4), getOperand(5)};
}

static LogicalResult verify(LaunchOp op) {
  // Kernel launch takes kNumConfigOperands leading operands for grid/block
  // sizes and transforms them into kNumConfigRegionAttributes region arguments
  // for block/thread identifiers and grid/block sizes.
  if (!op.body().empty()) {
    if (op.body().getNumArguments() !=
        LaunchOp::kNumConfigOperands + op.getNumOperands())
      return op.emitOpError("unexpected number of region arguments");
  }

  // Block terminators without successors are expected to exit the kernel region
  // and must be `gpu.terminator`.
  for (Block &block : op.body()) {
    if (block.empty())
      continue;
    if (block.back().getNumSuccessors() != 0)
      continue;
    if (!isa<gpu::TerminatorOp>(&block.back())) {
      return block.back()
          .emitError()
          .append("expected '", gpu::TerminatorOp::getOperationName(),
                  "' or a terminator with successors")
          .attachNote(op.getLoc())
          .append("in '", LaunchOp::getOperationName(), "' body region");
    }
  }

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

static void printLaunchOp(OpAsmPrinter &p, LaunchOp op) {
  // Print the launch configuration.
  p << LaunchOp::getOperationName() << ' ' << op.getBlocksKeyword();
  printSizeAssignment(p, op.getGridSize(), op.getGridSizeOperandValues(),
                      op.getBlockIds());
  p << ' ' << op.getThreadsKeyword();
  printSizeAssignment(p, op.getBlockSize(), op.getBlockSizeOperandValues(),
                      op.getThreadIds());

  p.printRegion(op.body(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict(op.getAttrs());
}

// Parse the size assignment blocks for blocks and threads.  These have the form
//   (%region_arg, %region_arg, %region_arg) in
//   (%region_arg = %operand, %region_arg = %operand, %region_arg = %operand)
// where %region_arg are percent-identifiers for the region arguments to be
// introduced further (SSA defs), and %operand are percent-identifiers for the
// SSA value uses.
static ParseResult
parseSizeAssignment(OpAsmParser &parser,
                    MutableArrayRef<OpAsmParser::OperandType> sizes,
                    MutableArrayRef<OpAsmParser::OperandType> regionSizes,
                    MutableArrayRef<OpAsmParser::OperandType> indices) {
  assert(indices.size() == 3 && "space for three indices expected");
  SmallVector<OpAsmParser::OperandType, 3> args;
  if (parser.parseRegionArgumentList(args, /*requiredOperandCount=*/3,
                                     OpAsmParser::Delimiter::Paren) ||
      parser.parseKeyword("in") || parser.parseLParen())
    return failure();
  std::move(args.begin(), args.end(), indices.begin());

  for (int i = 0; i < 3; ++i) {
    if (i != 0 && parser.parseComma())
      return failure();
    if (parser.parseRegionArgument(regionSizes[i]) || parser.parseEqual() ||
        parser.parseOperand(sizes[i]))
      return failure();
  }

  return parser.parseRParen();
}

// Parses a Launch operation.
// operation ::= `gpu.launch` `blocks` `(` ssa-id-list `)` `in` ssa-reassignment
//                           `threads` `(` ssa-id-list `)` `in` ssa-reassignment
//                            region attr-dict?
// ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
static ParseResult parseLaunchOp(OpAsmParser &parser, OperationState &result) {
  // Sizes of the grid and block.
  SmallVector<OpAsmParser::OperandType, LaunchOp::kNumConfigOperands> sizes(
      LaunchOp::kNumConfigOperands);
  MutableArrayRef<OpAsmParser::OperandType> sizesRef(sizes);

  // Actual (data) operands passed to the kernel.
  SmallVector<OpAsmParser::OperandType, 4> dataOperands;

  // Region arguments to be created.
  SmallVector<OpAsmParser::OperandType, 16> regionArgs(
      LaunchOp::kNumConfigRegionAttributes);
  MutableArrayRef<OpAsmParser::OperandType> regionArgsRef(regionArgs);

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

  // Introduce the body region and parse it. The region has
  // kNumConfigRegionAttributes arguments that correspond to
  // block/thread identifiers and grid/block sizes, all of the `index` type.
  Type index = parser.getBuilder().getIndexType();
  SmallVector<Type, LaunchOp::kNumConfigRegionAttributes> dataTypes(
      LaunchOp::kNumConfigRegionAttributes, index);
  Region *body = result.addRegion();
  return failure(parser.parseRegion(*body, regionArgs, dataTypes) ||
                 parser.parseOptionalAttrDict(result.attributes));
}

//===----------------------------------------------------------------------===//
// LaunchFuncOp
//===----------------------------------------------------------------------===//

void LaunchFuncOp::build(OpBuilder &builder, OperationState &result,
                         GPUFuncOp kernelFunc, KernelDim3 gridSize,
                         KernelDim3 blockSize, ValueRange kernelOperands) {
  // Add grid and block sizes as op operands, followed by the data operands.
  result.addOperands({gridSize.x, gridSize.y, gridSize.z, blockSize.x,
                      blockSize.y, blockSize.z});
  result.addOperands(kernelOperands);
  auto kernelModule = kernelFunc.getParentOfType<GPUModuleOp>();
  auto kernelSymbol = builder.getSymbolRefAttr(
      kernelModule.getName(), {builder.getSymbolRefAttr(kernelFunc.getName())});
  result.addAttribute(getKernelAttrName(), kernelSymbol);
  SmallVector<int32_t, 8> segmentSizes(8, 1);
  segmentSizes.front() = 0; // Initially no async dependencies.
  segmentSizes.back() = static_cast<int32_t>(kernelOperands.size());
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr(segmentSizes));
}

unsigned LaunchFuncOp::getNumKernelOperands() {
  return getNumOperands() - asyncDependencies().size() - kNumConfigOperands;
}

StringRef LaunchFuncOp::getKernelModuleName() {
  return kernel().getRootReference();
}

StringRef LaunchFuncOp::getKernelName() { return kernel().getLeafReference(); }

Value LaunchFuncOp::getKernelOperand(unsigned i) {
  return getOperand(asyncDependencies().size() + kNumConfigOperands + i);
}

KernelDim3 LaunchFuncOp::getGridSizeOperandValues() {
  auto operands = getOperands().drop_front(asyncDependencies().size());
  return KernelDim3{operands[0], operands[1], operands[2]};
}

KernelDim3 LaunchFuncOp::getBlockSizeOperandValues() {
  auto operands = getOperands().drop_front(asyncDependencies().size());
  return KernelDim3{operands[3], operands[4], operands[5]};
}

static LogicalResult verify(LaunchFuncOp op) {
  auto module = op.getParentOfType<ModuleOp>();
  if (!module)
    return op.emitOpError("expected to belong to a module");

  if (!module.getAttrOfType<UnitAttr>(GPUDialect::getContainerModuleAttrName()))
    return op.emitOpError(
        "expected the closest surrounding module to have the '" +
        GPUDialect::getContainerModuleAttrName() + "' attribute");

  auto kernelAttr = op.getAttrOfType<SymbolRefAttr>(op.getKernelAttrName());
  if (!kernelAttr)
    return op.emitOpError("symbol reference attribute '" +
                          op.getKernelAttrName() + "' must be specified");

  return success();
}

static ParseResult
parseLaunchFuncOperands(OpAsmParser &parser,
                        SmallVectorImpl<OpAsmParser::OperandType> &argNames,
                        SmallVectorImpl<Type> &argTypes) {
  if (parser.parseOptionalKeyword("args"))
    return success();
  SmallVector<NamedAttrList, 4> argAttrs;
  bool isVariadic = false;
  return impl::parseFunctionArgumentList(parser, /*allowAttributes=*/false,
                                         /*allowVariadic=*/false, argNames,
                                         argTypes, argAttrs, isVariadic);
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
// GPUFuncOp
//===----------------------------------------------------------------------===//

/// Adds a new block argument that corresponds to buffers located in
/// workgroup memory.
BlockArgument GPUFuncOp::addWorkgroupAttribution(Type type) {
  auto attrName = getNumWorkgroupAttributionsAttrName();
  auto attr = getAttrOfType<IntegerAttr>(attrName);
  setAttr(attrName, IntegerAttr::get(attr.getType(), attr.getValue() + 1));
  return getBody().insertArgument(getType().getNumInputs() + attr.getInt(),
                                  type);
}

/// Adds a new block argument that corresponds to buffers located in
/// private memory.
BlockArgument GPUFuncOp::addPrivateAttribution(Type type) {
  // Buffers on the private memory always come after buffers on the workgroup
  // memory.
  return getBody().addArgument(type);
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
  entryBlock->addArguments(type.getInputs());
  entryBlock->addArguments(workgroupAttributions);
  entryBlock->addArguments(privateAttributions);

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
                  SmallVectorImpl<OpAsmParser::OperandType> &args,
                  SmallVectorImpl<Type> &argTypes) {
  // If we could not parse the keyword, just assume empty list and succeed.
  if (failed(parser.parseOptionalKeyword(keyword)))
    return success();

  if (failed(parser.parseLParen()))
    return failure();

  // Early exit for an empty list.
  if (succeeded(parser.parseOptionalRParen()))
    return success();

  do {
    OpAsmParser::OperandType arg;
    Type type;

    if (parser.parseRegionArgument(arg) || parser.parseColonType(type))
      return failure();

    args.push_back(arg);
    argTypes.push_back(type);
  } while (succeeded(parser.parseOptionalComma()));

  return parser.parseRParen();
}

/// Parses a GPU function.
///
/// <operation> ::= `gpu.func` symbol-ref-id `(` argument-list `)`
///                 (`->` function-result-list)? memory-attribution `kernel`?
///                 function-attributes? region
static ParseResult parseGPUFuncOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 8> entryArgs;
  SmallVector<NamedAttrList, 1> argAttrs;
  SmallVector<NamedAttrList, 1> resultAttrs;
  SmallVector<Type, 8> argTypes;
  SmallVector<Type, 4> resultTypes;
  bool isVariadic;

  // Parse the function name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, ::mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  auto signatureLocation = parser.getCurrentLocation();
  if (failed(impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, entryArgs, argTypes, argAttrs,
          isVariadic, resultTypes, resultAttrs)))
    return failure();

  if (entryArgs.empty() && !argTypes.empty())
    return parser.emitError(signatureLocation)
           << "gpu.func requires named arguments";

  // Construct the function type. More types will be added to the region, but
  // not to the function type.
  Builder &builder = parser.getBuilder();
  auto type = builder.getFunctionType(argTypes, resultTypes);
  result.addAttribute(GPUFuncOp::getTypeAttrName(), TypeAttr::get(type));

  // Parse workgroup memory attributions.
  if (failed(parseAttributions(parser, GPUFuncOp::getWorkgroupKeyword(),
                               entryArgs, argTypes)))
    return failure();

  // Store the number of operands we just parsed as the number of workgroup
  // memory attributions.
  unsigned numWorkgroupAttrs = argTypes.size() - type.getNumInputs();
  result.addAttribute(GPUFuncOp::getNumWorkgroupAttributionsAttrName(),
                      builder.getI64IntegerAttr(numWorkgroupAttrs));

  // Parse private memory attributions.
  if (failed(parseAttributions(parser, GPUFuncOp::getPrivateKeyword(),
                               entryArgs, argTypes)))
    return failure();

  // Parse the kernel attribute if present.
  if (succeeded(parser.parseOptionalKeyword(GPUFuncOp::getKernelKeyword())))
    result.addAttribute(GPUDialect::getKernelFuncAttrName(),
                        builder.getUnitAttr());

  // Parse attributes.
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();
  mlir::impl::addArgAndResultAttrs(builder, result, argAttrs, resultAttrs);

  // Parse the region. If no argument names were provided, take all names
  // (including those of attributions) from the entry block.
  auto *body = result.addRegion();
  return parser.parseRegion(*body, entryArgs, argTypes);
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

/// Prints a GPU Func op.
static void printGPUFuncOp(OpAsmPrinter &p, GPUFuncOp op) {
  p << GPUFuncOp::getOperationName() << ' ';
  p.printSymbolName(op.getName());

  FunctionType type = op.getType();
  impl::printFunctionSignature(p, op.getOperation(), type.getInputs(),
                               /*isVariadic=*/false, type.getResults());

  printAttributions(p, op.getWorkgroupKeyword(), op.getWorkgroupAttributions());
  printAttributions(p, op.getPrivateKeyword(), op.getPrivateAttributions());
  if (op.isKernel())
    p << ' ' << op.getKernelKeyword();

  impl::printFunctionAttributes(p, op.getOperation(), type.getNumInputs(),
                                type.getNumResults(),
                                {op.getNumWorkgroupAttributionsAttrName(),
                                 GPUDialect::getKernelFuncAttrName()});
  p.printRegion(op.getBody(), /*printEntryBlockArgs=*/false);
}

void GPUFuncOp::setType(FunctionType newType) {
  auto oldType = getType();
  assert(newType.getNumResults() == oldType.getNumResults() &&
         "unimplemented: changes to the number of results");

  SmallVector<char, 16> nameBuf;
  for (int i = newType.getNumInputs(), e = oldType.getNumInputs(); i < e; i++)
    removeAttr(getArgAttrName(i, nameBuf));

  setAttr(getTypeAttrName(), TypeAttr::get(newType));
}

/// Hook for FunctionLike verifier.
LogicalResult GPUFuncOp::verifyType() {
  Type type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");

  if (isKernel() && getType().getNumResults() != 0)
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

    if (type.getMemorySpace() != memorySpace) {
      return op->emitOpError()
             << "expected memory space " << memorySpace << " in attribution";
    }
  }
  return success();
}

/// Verifies the body of the function.
LogicalResult GPUFuncOp::verifyBody() {
  unsigned numFuncArguments = getNumArguments();
  unsigned numWorkgroupAttributions = getNumWorkgroupAttributions();
  unsigned numBlockArguments = front().getNumArguments();
  if (numBlockArguments < numFuncArguments + numWorkgroupAttributions)
    return emitOpError() << "expected at least "
                         << numFuncArguments + numWorkgroupAttributions
                         << " arguments to body region";

  ArrayRef<Type> funcArgTypes = getType().getInputs();
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

static ParseResult parseReturnOp(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::OperandType, 4> operands;
  llvm::SmallVector<Type, 4> types;
  if (parser.parseOperandList(operands) ||
      parser.parseOptionalColonTypeList(types) ||
      parser.resolveOperands(operands, types, parser.getCurrentLocation(),
                             result.operands))
    return failure();

  return success();
}

static LogicalResult verify(gpu::ReturnOp returnOp) {
  GPUFuncOp function = returnOp.getParentOfType<GPUFuncOp>();

  FunctionType funType = function.getType();

  if (funType.getNumResults() != returnOp.operands().size())
    return returnOp.emitOpError()
        .append("expected ", funType.getNumResults(), " result operands")
        .attachNote(function.getLoc())
        .append("return type declared here");

  for (auto pair : llvm::enumerate(
           llvm::zip(function.getType().getResults(), returnOp.operands()))) {
    Type type;
    Value operand;
    std::tie(type, operand) = pair.value();
    if (type != operand.getType())
      return returnOp.emitOpError() << "unexpected type `" << operand.getType()
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

static ParseResult parseGPUModuleOp(OpAsmParser &parser,
                                    OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // If module attributes are present, parse them.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the module body.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, None, None))
    return failure();

  // Ensure that this module has a valid terminator.
  GPUModuleOp::ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

static void print(OpAsmPrinter &p, GPUModuleOp op) {
  p << op.getOperationName() << ' ';
  p.printSymbolName(op.getName());
  p.printOptionalAttrDictWithKeyword(op.getAttrs(),
                                     {SymbolTable::getSymbolAttrName()});
  p.printRegion(op->getRegion(0), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
}

static ParseResult parseAsyncDependencies(
    OpAsmParser &parser, Type &asyncTokenType,
    SmallVectorImpl<OpAsmParser::OperandType> &asyncDependencies) {
  auto loc = parser.getCurrentLocation();
  if (succeeded(parser.parseOptionalKeyword("async"))) {
    if (parser.getNumResults() == 0)
      return parser.emitError(loc, "needs to be named when marked 'async'");
    asyncTokenType = parser.getBuilder().getType<AsyncTokenType>();
  }
  return parser.parseOperandList(asyncDependencies,
                                 OpAsmParser::Delimiter::OptionalSquare);
}

static void printAsyncDependencies(OpAsmPrinter &printer, Operation *op,
                                   Type asyncTokenType,
                                   OperandRange asyncDependencies) {
  if (asyncTokenType)
    printer << "async ";
  if (asyncDependencies.empty())
    return;
  printer << "[";
  llvm::interleaveComma(asyncDependencies, printer);
  printer << "]";
}

#include "mlir/Dialect/GPU/GPUOpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/GPUOps.cpp.inc"
