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
#include "mlir/Dialect/Bufferization/IR/BufferDeallocationOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/StringSaver.h"
#include <cassert>
#include <numeric>

using namespace mlir;
using namespace mlir::gpu;

#include "mlir/Dialect/GPU/IR/GPUOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// GPU Device Mapping Attributes
//===----------------------------------------------------------------------===//

int64_t GPUBlockMappingAttr::getMappingId() const {
  return static_cast<int64_t>(getBlock());
}

bool GPUBlockMappingAttr::isLinearMapping() const {
  return getMappingId() >= static_cast<int64_t>(MappingId::LinearDim0);
}

int64_t GPUBlockMappingAttr::getRelativeIndex() const {
  return isLinearMapping()
             ? getMappingId() - static_cast<int64_t>(MappingId::LinearDim0)
             : getMappingId();
}

int64_t GPUWarpgroupMappingAttr::getMappingId() const {
  return static_cast<int64_t>(getWarpgroup());
}

bool GPUWarpgroupMappingAttr::isLinearMapping() const {
  return getMappingId() >= static_cast<int64_t>(MappingId::LinearDim0);
}

int64_t GPUWarpgroupMappingAttr::getRelativeIndex() const {
  return isLinearMapping()
             ? getMappingId() - static_cast<int64_t>(MappingId::LinearDim0)
             : getMappingId();
}

int64_t GPUWarpMappingAttr::getMappingId() const {
  return static_cast<int64_t>(getWarp());
}

bool GPUWarpMappingAttr::isLinearMapping() const {
  return getMappingId() >= static_cast<int64_t>(MappingId::LinearDim0);
}

int64_t GPUWarpMappingAttr::getRelativeIndex() const {
  return isLinearMapping()
             ? getMappingId() - static_cast<int64_t>(MappingId::LinearDim0)
             : getMappingId();
}

int64_t GPUThreadMappingAttr::getMappingId() const {
  return static_cast<int64_t>(getThread());
}

bool GPUThreadMappingAttr::isLinearMapping() const {
  return getMappingId() >= static_cast<int64_t>(MappingId::LinearDim0);
}

int64_t GPUThreadMappingAttr::getRelativeIndex() const {
  return isLinearMapping()
             ? getMappingId() - static_cast<int64_t>(MappingId::LinearDim0)
             : getMappingId();
}

int64_t GPUMemorySpaceMappingAttr::getMappingId() const {
  return static_cast<int64_t>(getAddressSpace());
}

bool GPUMemorySpaceMappingAttr::isLinearMapping() const {
  llvm_unreachable("GPUMemorySpaceMappingAttr does not support linear mapping");
}

int64_t GPUMemorySpaceMappingAttr::getRelativeIndex() const {
  llvm_unreachable("GPUMemorySpaceMappingAttr does not support relative index");
}

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
  return elementType.isF16() || elementType.isF32() ||
         elementType.isUnsignedInteger(8) || elementType.isSignedInteger(8) ||
         elementType.isInteger(32);
}

LogicalResult
MMAMatrixType::verifyInvariants(function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<int64_t> shape, Type elementType,
                                StringRef operand) {
  if (operand != "AOp" && operand != "BOp" && operand != "COp")
    return emitError() << "operand expected to be one of AOp, BOp or COp";

  if (shape.size() != 2)
    return emitError() << "MMAMatrixType must have exactly two dimensions";

  if (!MMAMatrixType::isValidElementType(elementType))
    return emitError()
           << "MMAMatrixType elements must be SI8, UI8, I32, F16, or F32";

  return success();
}

//===----------------------------------------------------------------------===//
// GPUDialect
//===----------------------------------------------------------------------===//

bool GPUDialect::isWorkgroupMemoryAddressSpace(Attribute memorySpace) {
  if (!memorySpace)
    return false;
  if (auto gpuAttr = llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    return gpuAttr.getValue() == getWorkgroupAddressSpace();
  return false;
}

bool GPUDialect::hasWorkgroupMemoryAddressSpace(MemRefType type) {
  Attribute memorySpace = type.getMemorySpace();
  return isWorkgroupMemoryAddressSpace(memorySpace);
}

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
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void GPUDialect::initialize() {
  addTypes<AsyncTokenType>();
  addTypes<MMAMatrixType>();
  addTypes<SparseDnTensorHandleType>();
  addTypes<SparseSpMatHandleType>();
  addTypes<SparseSpGEMMOpHandleType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/GPU/IR/GPUOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/GPU/IR/GPUOpsAttributes.cpp.inc"
      >();
  addInterfaces<GPUInlinerInterface>();
  declarePromisedInterface<bufferization::BufferDeallocationOpInterface,
                           TerminatorOp>();
}

static std::string getSparseHandleKeyword(SparseHandleKind kind) {
  switch (kind) {
  case SparseHandleKind::DnTensor:
    return "sparse.dntensor_handle";
  case SparseHandleKind::SpMat:
    return "sparse.spmat_handle";
  case SparseHandleKind::SpGEMMOp:
    return "sparse.spgemmop_handle";
  }
  llvm_unreachable("unknown sparse handle kind");
  return "";
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

  if (keyword == getSparseHandleKeyword(SparseHandleKind::DnTensor))
    return SparseDnTensorHandleType::get(context);
  if (keyword == getSparseHandleKeyword(SparseHandleKind::SpMat))
    return SparseSpMatHandleType::get(context);
  if (keyword == getSparseHandleKeyword(SparseHandleKind::SpGEMMOp))
    return SparseSpGEMMOpHandleType::get(context);

  parser.emitError(parser.getNameLoc(), "unknown gpu type: " + keyword);
  return Type();
}
// TODO: print refined type here. Notice that should be corresponding to the
// parser
void GPUDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<AsyncTokenType>([&](Type) { os << "async.token"; })
      .Case<SparseDnTensorHandleType>([&](Type) {
        os << getSparseHandleKeyword(SparseHandleKind::DnTensor);
      })
      .Case<SparseSpMatHandleType>(
          [&](Type) { os << getSparseHandleKeyword(SparseHandleKind::SpMat); })
      .Case<SparseSpGEMMOpHandleType>([&](Type) {
        os << getSparseHandleKeyword(SparseHandleKind::SpGEMMOp);
      })
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

static LogicalResult verifyKnownLaunchSizeAttr(Operation *op,
                                               NamedAttribute attr) {
  auto array = dyn_cast<DenseI32ArrayAttr>(attr.getValue());
  if (!array)
    return op->emitOpError(Twine(attr.getName()) +
                           " must be a dense i32 array");
  if (array.size() != 3)
    return op->emitOpError(Twine(attr.getName()) +
                           " must contain exactly 3 elements");
  return success();
}

LogicalResult GPUDialect::verifyOperationAttribute(Operation *op,
                                                   NamedAttribute attr) {
  if (attr.getName() == getKnownBlockSizeAttrHelper().getName())
    return verifyKnownLaunchSizeAttr(op, attr);
  if (attr.getName() == getKnownGridSizeAttrHelper().getName())
    return verifyKnownLaunchSizeAttr(op, attr);
  if (!llvm::isa<UnitAttr>(attr.getValue()) ||
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

    // Check that `launch_func` refers to a well-formed GPU kernel container.
    StringAttr kernelContainerName = launchOp.getKernelModuleName();
    Operation *kernelContainer = module.lookupSymbol(kernelContainerName);
    if (!kernelContainer)
      return launchOp.emitOpError()
             << "kernel container '" << kernelContainerName.getValue()
             << "' is undefined";

    // If the container is a GPU binary op return success.
    if (isa<BinaryOp>(kernelContainer))
      return success();

    auto kernelModule = dyn_cast<GPUModuleOp>(kernelContainer);
    if (!kernelModule)
      return launchOp.emitOpError()
             << "kernel module '" << kernelContainerName.getValue()
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

// GPU Memory attributions functions shared by LaunchOp and GPUFuncOp.
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

/// Prints a GPU function memory attribution.
static void printAttributions(OpAsmPrinter &p, StringRef keyword,
                              ArrayRef<BlockArgument> values) {
  if (values.empty())
    return;

  p << ' ' << keyword << '(';
  llvm::interleaveComma(
      values, p, [&p](BlockArgument v) { p << v << " : " << v.getType(); });
  p << ')';
}

/// Verifies a GPU function memory attribution.
static LogicalResult verifyAttributions(Operation *op,
                                        ArrayRef<BlockArgument> attributions,
                                        gpu::AddressSpace memorySpace) {
  for (Value v : attributions) {
    auto type = llvm::dyn_cast<MemRefType>(v.getType());
    if (!type)
      return op->emitOpError() << "expected memref type in attribution";

    // We can only verify the address space if it hasn't already been lowered
    // from the AddressSpaceAttr to a target-specific numeric value.
    auto addressSpace =
        llvm::dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace());
    if (!addressSpace)
      continue;
    if (addressSpace.getValue() != memorySpace)
      return op->emitOpError()
             << "expected memory space " << stringifyAddressSpace(memorySpace)
             << " in attribution";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AllReduceOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyReduceOpAndType(gpu::AllReduceOperation opName,
                                           Type resType) {
  using Kind = gpu::AllReduceOperation;
  if (llvm::is_contained(
          {Kind::MINNUMF, Kind::MAXNUMF, Kind::MINIMUMF, Kind::MAXIMUMF},
          opName)) {
    if (!isa<FloatType>(resType))
      return failure();
  }

  if (llvm::is_contained({Kind::MINSI, Kind::MINUI, Kind::MAXSI, Kind::MAXUI,
                          Kind::AND, Kind::OR, Kind::XOR},
                         opName)) {
    if (!isa<IntegerType>(resType))
      return failure();
  }

  return success();
}

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
    if (failed(verifyReduceOpAndType(opName, getType()))) {
      return emitError() << '`' << gpu::stringifyAllReduceOperation(opName)
                         << "` reduction operation is not compatible with type "
                         << getType();
    }
  }

  return success();
}

static bool canMakeGroupOpUniform(Operation *op) {
  auto launchOp = dyn_cast<gpu::LaunchOp>(op->getParentOp());
  if (!launchOp)
    return false;

  Region &body = launchOp.getBody();
  assert(!body.empty() && "Invalid region");

  // Only convert ops in gpu::launch entry block for now.
  return op->getBlock() == &body.front();
}

OpFoldResult gpu::AllReduceOp::fold(FoldAdaptor /*adaptor*/) {
  if (!getUniform() && canMakeGroupOpUniform(*this)) {
    setUniform(true);
    return getResult();
  }

  return nullptr;
}

// TODO: Support optional custom attributes (without dialect prefix).
static ParseResult parseAllReduceOperation(AsmParser &parser,
                                           AllReduceOperationAttr &attr) {
  StringRef enumStr;
  if (!parser.parseOptionalKeyword(&enumStr)) {
    std::optional<AllReduceOperation> op =
        gpu::symbolizeAllReduceOperation(enumStr);
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
// SubgroupReduceOp
//===----------------------------------------------------------------------===//

LogicalResult gpu::SubgroupReduceOp::verify() {
  Type elemType = getType();
  if (auto vecTy = dyn_cast<VectorType>(elemType)) {
    if (vecTy.isScalable())
      return emitOpError() << "is not compatible with scalable vector types";

    elemType = vecTy.getElementType();
  }

  gpu::AllReduceOperation opName = getOp();
  if (failed(verifyReduceOpAndType(opName, elemType))) {
    return emitError() << '`' << gpu::stringifyAllReduceOperation(opName)
                       << "` reduction operation is not compatible with type "
                       << getType();
  }

  auto clusterSize = getClusterSize();
  if (clusterSize) {
    uint32_t size = *clusterSize;
    if (!llvm::isPowerOf2_32(size)) {
      return emitOpError() << "cluster size " << size
                           << " is not a power of two";
    }
  }

  uint32_t stride = getClusterStride();
  if (stride != 1 && !clusterSize) {
    return emitOpError() << "cluster stride can only be specified if cluster "
                            "size is specified";
  }
  if (!llvm::isPowerOf2_32(stride)) {
    return emitOpError() << "cluster stride " << stride
                         << " is not a power of two";
  }

  return success();
}

OpFoldResult gpu::SubgroupReduceOp::fold(FoldAdaptor /*adaptor*/) {
  if (getClusterSize() == 1)
    return getValue();

  if (!getUniform() && canMakeGroupOpUniform(*this)) {
    setUniform(true);
    return getResult();
  }

  return nullptr;
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
                     Type asyncTokenType, ValueRange asyncDependencies,
                     TypeRange workgroupAttributions,
                     TypeRange privateAttributions, Value clusterSizeX,
                     Value clusterSizeY, Value clusterSizeZ) {
  OpBuilder::InsertionGuard g(builder);

  // Add a WorkGroup attribution attribute. This attribute is required to
  // identify private attributions in the list of block argguments.
  result.addAttribute(getNumWorkgroupAttributionsAttrName(),
                      builder.getI64IntegerAttr(workgroupAttributions.size()));

  // Add Op operands.
  result.addOperands(asyncDependencies);
  if (asyncTokenType)
    result.types.push_back(builder.getType<AsyncTokenType>());

  // Add grid and block sizes as op operands, followed by the data operands.
  result.addOperands({gridSizeX, gridSizeY, gridSizeZ, getBlockSizeX,
                      getBlockSizeY, getBlockSizeZ});
  if (clusterSizeX)
    result.addOperands(clusterSizeX);
  if (clusterSizeY)
    result.addOperands(clusterSizeY);
  if (clusterSizeZ)
    result.addOperands(clusterSizeZ);
  if (dynamicSharedMemorySize)
    result.addOperands(dynamicSharedMemorySize);

  // Create a kernel body region with kNumConfigRegionAttributes + N memory
  // attributions, where the first kNumConfigRegionAttributes arguments have
  // `index` type and the rest have the same types as the data operands.
  Region *kernelRegion = result.addRegion();
  Block *body = builder.createBlock(kernelRegion);
  // TODO: Allow passing in proper locations here.
  for (unsigned i = 0; i < kNumConfigRegionAttributes; ++i)
    body->addArgument(builder.getIndexType(), result.location);
  // Add WorkGroup & Private attributions to the region arguments.
  for (Type argTy : workgroupAttributions)
    body->addArgument(argTy, result.location);
  for (Type argTy : privateAttributions)
    body->addArgument(argTy, result.location);
  // Fill OperandSegmentSize Attribute.
  SmallVector<int32_t, 11> segmentSizes(11, 1);
  segmentSizes.front() = asyncDependencies.size();
  segmentSizes.back() = dynamicSharedMemorySize ? 1 : 0;
  segmentSizes[7] = clusterSizeX ? 1 : 0;
  segmentSizes[8] = clusterSizeY ? 1 : 0;
  segmentSizes[9] = clusterSizeZ ? 1 : 0;
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

std::optional<KernelDim3> LaunchOp::getClusterIds() {
  assert(!getBody().empty() && "LaunchOp body must not be empty.");
  if (!hasClusterSize())
    return std::nullopt;
  auto args = getBody().getArguments();
  return KernelDim3{args[12], args[13], args[14]};
}

std::optional<KernelDim3> LaunchOp::getClusterSize() {
  assert(!getBody().empty() && "LaunchOp body must not be empty.");
  if (!hasClusterSize())
    return std::nullopt;
  auto args = getBody().getArguments();
  return KernelDim3{args[15], args[16], args[17]};
}

KernelDim3 LaunchOp::getGridSizeOperandValues() {
  auto operands = getOperands().drop_front(getAsyncDependencies().size());
  return KernelDim3{operands[0], operands[1], operands[2]};
}

KernelDim3 LaunchOp::getBlockSizeOperandValues() {
  auto operands = getOperands().drop_front(getAsyncDependencies().size());
  return KernelDim3{operands[3], operands[4], operands[5]};
}

std::optional<KernelDim3> LaunchOp::getClusterSizeOperandValues() {
  auto operands = getOperands().drop_front(getAsyncDependencies().size());
  if (!hasClusterSize())
    return std::nullopt;
  return KernelDim3{operands[6], operands[7], operands[8]};
}

LogicalResult LaunchOp::verify() {
  if (!(hasClusterSize()) &&
      (getClusterSizeX() || getClusterSizeY() || getClusterSizeZ()))
    return emitOpError() << "cluster size must be all present";
  return success();
}

LogicalResult LaunchOp::verifyRegions() {
  // Kernel launch takes kNumConfigOperands leading operands for grid/block
  // sizes and transforms them into kNumConfigRegionAttributes region arguments
  // for block/thread identifiers and grid/block sizes.
  if (!getBody().empty()) {
    if (getBody().getNumArguments() <
        kNumConfigRegionAttributes + getNumWorkgroupAttributions())
      return emitOpError("unexpected number of region arguments");
  }

  // Verify Attributions Address Spaces.
  if (failed(verifyAttributions(getOperation(), getWorkgroupAttributions(),
                                GPUDialect::getWorkgroupAddressSpace())) ||
      failed(verifyAttributions(getOperation(), getPrivateAttributions(),
                                GPUDialect::getPrivateAddressSpace())))
    return failure();

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
  if (hasClusterSize()) {
    p << ' ' << getClustersKeyword();
    printSizeAssignment(p, getClusterSize().value(),
                        getClusterSizeOperandValues().value(),
                        getClusterIds().value());
  }
  p << ' ' << getBlocksKeyword();
  printSizeAssignment(p, getGridSize(), getGridSizeOperandValues(),
                      getBlockIds());
  p << ' ' << getThreadsKeyword();
  printSizeAssignment(p, getBlockSize(), getBlockSizeOperandValues(),
                      getThreadIds());
  if (getDynamicSharedMemorySize())
    p << ' ' << getDynamicSharedMemorySizeKeyword() << ' '
      << getDynamicSharedMemorySize();

  printAttributions(p, getWorkgroupKeyword(), getWorkgroupAttributions());
  printAttributions(p, getPrivateKeyword(), getPrivateAttributions());

  p << ' ';

  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{
                              LaunchOp::getOperandSegmentSizeAttr(),
                              getNumWorkgroupAttributionsAttrName()});
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
///       `clusters` `(` ssa-id-list `)` `in` ssa-reassignment (Optional)
///       `blocks` `(` ssa-id-list `)` `in` ssa-reassignment
///       `threads` `(` ssa-id-list `)` `in` ssa-reassignment
///       memory-attribution
///       region attr-dict?
/// ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
ParseResult LaunchOp::parse(OpAsmParser &parser, OperationState &result) {
  // Sizes of the grid and block.
  SmallVector<OpAsmParser::UnresolvedOperand, LaunchOp::kNumConfigOperands>
      sizes(LaunchOp::kNumConfigOperands);

  // Actual (data) operands passed to the kernel.
  SmallVector<OpAsmParser::UnresolvedOperand, 4> dataOperands;

  // Region arguments to be created.
  SmallVector<OpAsmParser::UnresolvedOperand, 16> regionArgs(
      LaunchOp::kNumConfigRegionAttributes);

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

  bool hasCluster = false;
  if (succeeded(
          parser.parseOptionalKeyword(LaunchOp::getClustersKeyword().data()))) {
    hasCluster = true;
    sizes.resize(9);
    regionArgs.resize(18);
  }
  MutableArrayRef<OpAsmParser::UnresolvedOperand> sizesRef(sizes);
  MutableArrayRef<OpAsmParser::UnresolvedOperand> regionArgsRef(regionArgs);

  // Last three segment assigns the cluster size. In the region argument
  // list, this is last 6 arguments.
  if (hasCluster) {
    if (parseSizeAssignment(parser, sizesRef.drop_front(6),
                            regionArgsRef.slice(15, 3),
                            regionArgsRef.slice(12, 3)))
      return failure();
  }
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

  // Create the region arguments, it has kNumConfigRegionAttributes arguments
  // that correspond to block/thread identifiers and grid/block sizes, all
  // having `index` type, a variadic number of WorkGroup Attributions and
  // a variadic number of Private Attributions. The number of WorkGroup
  // Attributions is stored in the attr with name:
  // LaunchOp::getNumWorkgroupAttributionsAttrName().
  Type index = parser.getBuilder().getIndexType();
  SmallVector<Type, LaunchOp::kNumConfigRegionAttributes> dataTypes(
      LaunchOp::kNumConfigRegionAttributes + 6, index);

  SmallVector<OpAsmParser::Argument> regionArguments;
  for (auto ssaValueAndType : llvm::zip(regionArgs, dataTypes)) {
    OpAsmParser::Argument arg;
    arg.ssaName = std::get<0>(ssaValueAndType);
    arg.type = std::get<1>(ssaValueAndType);
    regionArguments.push_back(arg);
  }

  Builder &builder = parser.getBuilder();
  // Parse workgroup memory attributions.
  if (failed(parseAttributions(parser, LaunchOp::getWorkgroupKeyword(),
                               regionArguments)))
    return failure();

  // Store the number of operands we just parsed as the number of workgroup
  // memory attributions.
  unsigned numWorkgroupAttrs = regionArguments.size() -
                               LaunchOp::kNumConfigRegionAttributes -
                               (hasCluster ? 6 : 0);
  result.addAttribute(LaunchOp::getNumWorkgroupAttributionsAttrName(),
                      builder.getI64IntegerAttr(numWorkgroupAttrs));

  // Parse private memory attributions.
  if (failed(parseAttributions(parser, LaunchOp::getPrivateKeyword(),
                               regionArguments)))
    return failure();

  // Introduce the body region and parse it. The region has
  // kNumConfigRegionAttributes arguments that correspond to
  // block/thread identifiers and grid/block sizes, all having `index` type.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArguments) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();

  SmallVector<int32_t, 11> segmentSizes(11, 1);
  segmentSizes.front() = asyncDependencies.size();

  if (!hasCluster) {
    segmentSizes[7] = 0;
    segmentSizes[8] = 0;
    segmentSizes[9] = 0;
  }
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
      if (id.getUses().empty())
        return;
      if (!simplified) {
        // Create a zero value the first time.
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&op.getBody().front());
        zero =
            rewriter.create<arith::ConstantIndexOp>(op.getLoc(), /*value=*/0);
      }
      rewriter.replaceAllUsesWith(id, zero);
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

/// Adds a new block argument that corresponds to buffers located in
/// workgroup memory.
BlockArgument LaunchOp::addWorkgroupAttribution(Type type, Location loc) {
  auto attrName = getNumWorkgroupAttributionsAttrName();
  auto attr = (*this)->getAttrOfType<IntegerAttr>(attrName);
  (*this)->setAttr(attrName,
                   IntegerAttr::get(attr.getType(), attr.getValue() + 1));
  return getBody().insertArgument(
      LaunchOp::getNumConfigRegionAttributes() + attr.getInt(), type, loc);
}

/// Adds a new block argument that corresponds to buffers located in
/// private memory.
BlockArgument LaunchOp::addPrivateAttribution(Type type, Location loc) {
  // Buffers on the private memory always come after buffers on the workgroup
  // memory.
  return getBody().addArgument(type, loc);
}

//===----------------------------------------------------------------------===//
// LaunchFuncOp
//===----------------------------------------------------------------------===//

void LaunchFuncOp::build(OpBuilder &builder, OperationState &result,
                         SymbolRefAttr kernelSymbol, KernelDim3 gridSize,
                         KernelDim3 getBlockSize, Value dynamicSharedMemorySize,
                         ValueRange kernelOperands, Type asyncTokenType,
                         ValueRange asyncDependencies,
                         std::optional<KernelDim3> clusterSize) {
  assert(kernelSymbol.getNestedReferences().size() == 1 &&
         "expected a symbol reference with a single nested reference");
  result.addOperands(asyncDependencies);
  if (asyncTokenType)
    result.types.push_back(builder.getType<AsyncTokenType>());

  // Add grid and block sizes as op operands, followed by the data operands.
  result.addOperands({gridSize.x, gridSize.y, gridSize.z, getBlockSize.x,
                      getBlockSize.y, getBlockSize.z});
  if (clusterSize.has_value())
    result.addOperands({clusterSize->x, clusterSize->y, clusterSize->z});
  if (dynamicSharedMemorySize)
    result.addOperands(dynamicSharedMemorySize);
  result.addOperands(kernelOperands);

  Properties &prop = result.getOrAddProperties<Properties>();
  prop.kernel = kernelSymbol;
  size_t segmentSizesLen = std::size(prop.operandSegmentSizes);
  // Initialize the segment sizes to 1.
  for (auto &sz : prop.operandSegmentSizes)
    sz = 1;
  prop.operandSegmentSizes[0] = asyncDependencies.size();
  if (!clusterSize.has_value()) {
    prop.operandSegmentSizes[segmentSizesLen - 4] = 0;
    prop.operandSegmentSizes[segmentSizesLen - 5] = 0;
    prop.operandSegmentSizes[segmentSizesLen - 6] = 0;
  }
  prop.operandSegmentSizes[segmentSizesLen - 3] =
      dynamicSharedMemorySize ? 1 : 0;
  prop.operandSegmentSizes[segmentSizesLen - 2] =
      static_cast<int32_t>(kernelOperands.size());
  prop.operandSegmentSizes[segmentSizesLen - 1] = 0;
}

void LaunchFuncOp::build(OpBuilder &builder, OperationState &result,
                         GPUFuncOp kernelFunc, KernelDim3 gridSize,
                         KernelDim3 getBlockSize, Value dynamicSharedMemorySize,
                         ValueRange kernelOperands, Type asyncTokenType,
                         ValueRange asyncDependencies,
                         std::optional<KernelDim3> clusterSize) {
  auto kernelModule = kernelFunc->getParentOfType<GPUModuleOp>();
  auto kernelSymbol =
      SymbolRefAttr::get(kernelModule.getNameAttr(),
                         {SymbolRefAttr::get(kernelFunc.getNameAttr())});
  build(builder, result, kernelSymbol, gridSize, getBlockSize,
        dynamicSharedMemorySize, kernelOperands, asyncTokenType,
        asyncDependencies, clusterSize);
}

void LaunchFuncOp::build(OpBuilder &builder, OperationState &result,
                         SymbolRefAttr kernel, KernelDim3 gridSize,
                         KernelDim3 getBlockSize, Value dynamicSharedMemorySize,
                         ValueRange kernelOperands, Value asyncObject,
                         std::optional<KernelDim3> clusterSize) {
  // Add grid and block sizes as op operands, followed by the data operands.
  result.addOperands({gridSize.x, gridSize.y, gridSize.z, getBlockSize.x,
                      getBlockSize.y, getBlockSize.z});
  if (clusterSize.has_value())
    result.addOperands({clusterSize->x, clusterSize->y, clusterSize->z});
  if (dynamicSharedMemorySize)
    result.addOperands(dynamicSharedMemorySize);
  result.addOperands(kernelOperands);
  if (asyncObject)
    result.addOperands(asyncObject);
  Properties &prop = result.getOrAddProperties<Properties>();
  prop.kernel = kernel;
  size_t segmentSizesLen = std::size(prop.operandSegmentSizes);
  // Initialize the segment sizes to 1.
  for (auto &sz : prop.operandSegmentSizes)
    sz = 1;
  prop.operandSegmentSizes[0] = 0;
  if (!clusterSize.has_value()) {
    prop.operandSegmentSizes[segmentSizesLen - 4] = 0;
    prop.operandSegmentSizes[segmentSizesLen - 5] = 0;
    prop.operandSegmentSizes[segmentSizesLen - 6] = 0;
  }
  prop.operandSegmentSizes[segmentSizesLen - 3] =
      dynamicSharedMemorySize ? 1 : 0;
  prop.operandSegmentSizes[segmentSizesLen - 2] =
      static_cast<int32_t>(kernelOperands.size());
  prop.operandSegmentSizes[segmentSizesLen - 1] = asyncObject ? 1 : 0;
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

KernelDim3 LaunchFuncOp::getClusterSizeOperandValues() {
  assert(hasClusterSize() &&
         "cluster size is not set, check hasClusterSize() first");
  auto operands = getOperands().drop_front(getAsyncDependencies().size());
  return KernelDim3{operands[6], operands[7], operands[8]};
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

  if (hasClusterSize()) {
    if (getClusterSizeY().getType() != getClusterSizeX().getType() ||
        getClusterSizeZ().getType() != getClusterSizeX().getType())
      return emitOpError()
             << "expects types of the cluster dimensions must be the same";
  }

  return success();
}

static ParseResult
parseLaunchDimType(OpAsmParser &parser, Type &dimTy,
                   std::optional<OpAsmParser::UnresolvedOperand> clusterValue,
                   Type &clusterXTy, Type &clusterYTy, Type &clusterZTy) {
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseType(dimTy))
      return failure();
  } else {
    dimTy = IndexType::get(parser.getContext());
  }
  if (clusterValue.has_value()) {
    clusterXTy = clusterYTy = clusterZTy = dimTy;
  }
  return success();
}

static void printLaunchDimType(OpAsmPrinter &printer, Operation *op, Type dimTy,
                               Value clusterValue, Type clusterXTy,
                               Type clusterYTy, Type clusterZTy) {
  if (!dimTy.isIndex())
    printer << ": " << dimTy;
}

static ParseResult parseLaunchFuncOperands(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &argNames,
    SmallVectorImpl<Type> &argTypes) {
  if (parser.parseOptionalKeyword("args"))
    return success();

  auto parseElement = [&]() -> ParseResult {
    return failure(parser.parseOperand(argNames.emplace_back()) ||
                   parser.parseColonType(argTypes.emplace_back()));
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseElement, " in argument list");
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
// BarrierOp
//===----------------------------------------------------------------------===//

namespace {

/// Remove gpu.barrier after gpu.barrier, the threads are already synchronized!
LogicalResult eraseRedundantGpuBarrierOps(BarrierOp op,
                                          PatternRewriter &rewriter) {
  if (isa_and_nonnull<BarrierOp>(op->getNextNode())) {
    rewriter.eraseOp(op);
    return success();
  }
  return failure();
}

} // end anonymous namespace

void BarrierOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add(eraseRedundantGpuBarrierOps);
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
  OpBuilder::InsertionGuard g(builder);

  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));
  result.addAttribute(getNumWorkgroupAttributionsAttrName(),
                      builder.getI64IntegerAttr(workgroupAttributions.size()));
  result.addAttributes(attrs);
  Region *body = result.addRegion();
  Block *entryBlock = builder.createBlock(body);

  // TODO: Allow passing in proper locations here.
  for (Type argTy : type.getInputs())
    entryBlock->addArgument(argTy, result.location);
  for (Type argTy : workgroupAttributions)
    entryBlock->addArgument(argTy, result.location);
  for (Type argTy : privateAttributions)
    entryBlock->addArgument(argTy, result.location);
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
                  SmallVectorImpl<OpAsmParser::Argument> &args,
                  Attribute &attributionAttrs) {
  // If we could not parse the keyword, just assume empty list and succeed.
  if (failed(parser.parseOptionalKeyword(keyword)))
    return success();

  size_t existingArgs = args.size();
  ParseResult result =
      parser.parseArgumentList(args, OpAsmParser::Delimiter::Paren,
                               /*allowType=*/true, /*allowAttrs=*/true);
  if (failed(result))
    return result;

  bool hadAttrs = llvm::any_of(ArrayRef(args).drop_front(existingArgs),
                               [](const OpAsmParser::Argument &arg) -> bool {
                                 return arg.attrs && !arg.attrs.empty();
                               });
  if (!hadAttrs) {
    attributionAttrs = nullptr;
    return result;
  }

  Builder &builder = parser.getBuilder();
  SmallVector<Attribute> attributionAttrsVec;
  for (const auto &argument : ArrayRef(args).drop_front(existingArgs)) {
    if (!argument.attrs)
      attributionAttrsVec.push_back(builder.getDictionaryAttr({}));
    else
      attributionAttrsVec.push_back(argument.attrs);
  }
  attributionAttrs = builder.getArrayAttr(attributionAttrsVec);
  return result;
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
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(type));

  function_interface_impl::addArgAndResultAttrs(
      builder, result, entryArgs, resultAttrs, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));

  Attribute workgroupAttributionAttrs;
  // Parse workgroup memory attributions.
  if (failed(parseAttributions(parser, GPUFuncOp::getWorkgroupKeyword(),
                               entryArgs, workgroupAttributionAttrs)))
    return failure();

  // Store the number of operands we just parsed as the number of workgroup
  // memory attributions.
  unsigned numWorkgroupAttrs = entryArgs.size() - type.getNumInputs();
  result.addAttribute(GPUFuncOp::getNumWorkgroupAttributionsAttrName(),
                      builder.getI64IntegerAttr(numWorkgroupAttrs));
  if (workgroupAttributionAttrs)
    result.addAttribute(GPUFuncOp::getWorkgroupAttribAttrsAttrName(result.name),
                        workgroupAttributionAttrs);

  Attribute privateAttributionAttrs;
  // Parse private memory attributions.
  if (failed(parseAttributions(parser, GPUFuncOp::getPrivateKeyword(),
                               entryArgs, privateAttributionAttrs)))
    return failure();
  if (privateAttributionAttrs)
    result.addAttribute(GPUFuncOp::getPrivateAttribAttrsAttrName(result.name),
                        privateAttributionAttrs);

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
                              ArrayRef<BlockArgument> values,
                              ArrayAttr attributes) {
  if (values.empty())
    return;

  p << ' ' << keyword << '(';
  llvm::interleaveComma(
      llvm::enumerate(values), p, [&p, attributes](auto pair) {
        BlockArgument v = pair.value();
        p << v << " : " << v.getType();

        size_t attributionIndex = pair.index();
        DictionaryAttr attrs;
        if (attributes && attributionIndex < attributes.size())
          attrs = llvm::cast<DictionaryAttr>(attributes[attributionIndex]);
        if (attrs)
          p.printOptionalAttrDict(attrs.getValue());
      });
  p << ')';
}

void GPUFuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getName());

  FunctionType type = getFunctionType();
  function_interface_impl::printFunctionSignature(p, *this, type.getInputs(),
                                                  /*isVariadic=*/false,
                                                  type.getResults());

  printAttributions(p, getWorkgroupKeyword(), getWorkgroupAttributions(),
                    getWorkgroupAttribAttrs().value_or(nullptr));
  printAttributions(p, getPrivateKeyword(), getPrivateAttributions(),
                    getPrivateAttribAttrs().value_or(nullptr));
  if (isKernel())
    p << ' ' << getKernelKeyword();

  function_interface_impl::printFunctionAttributes(
      p, *this,
      {getNumWorkgroupAttributionsAttrName(),
       GPUDialect::getKernelFuncAttrName(), getFunctionTypeAttrName(),
       getArgAttrsAttrName(), getResAttrsAttrName(),
       getWorkgroupAttribAttrsAttrName(), getPrivateAttribAttrsAttrName()});
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

static DictionaryAttr getAttributionAttrs(GPUFuncOp op, unsigned index,
                                          StringAttr attrName) {
  auto allAttrs = llvm::dyn_cast_or_null<ArrayAttr>(op->getAttr(attrName));
  if (!allAttrs || index >= allAttrs.size())
    return DictionaryAttr();
  return llvm::cast<DictionaryAttr>(allAttrs[index]);
}

DictionaryAttr GPUFuncOp::getworkgroupAttributionAttrs(unsigned index) {
  return getAttributionAttrs(*this, index, getWorkgroupAttribAttrsAttrName());
}

DictionaryAttr GPUFuncOp::getPrivateAttributionAttrs(unsigned index) {
  return getAttributionAttrs(*this, index, getPrivateAttribAttrsAttrName());
}

static void setAttributionAttrs(GPUFuncOp op, unsigned index,
                                DictionaryAttr value, StringAttr attrName) {
  MLIRContext *ctx = op.getContext();
  auto allAttrs = llvm::dyn_cast_or_null<ArrayAttr>(op->getAttr(attrName));
  SmallVector<Attribute> elements;
  if (allAttrs)
    elements.append(allAttrs.begin(), allAttrs.end());
  while (elements.size() <= index)
    elements.push_back(DictionaryAttr::get(ctx));
  if (!value)
    elements[index] = DictionaryAttr::get(ctx);
  else
    elements[index] = value;
  ArrayAttr newValue = ArrayAttr::get(ctx, elements);
  op->setAttr(attrName, newValue);
}

void GPUFuncOp::setworkgroupAttributionAttrs(unsigned index,
                                             DictionaryAttr value) {
  setAttributionAttrs(*this, index, value, getWorkgroupAttribAttrsAttrName());
}

void GPUFuncOp::setPrivateAttributionAttrs(unsigned int index,
                                           DictionaryAttr value) {
  setAttributionAttrs(*this, index, value, getPrivateAttribAttrsAttrName());
}

static Attribute getAttributionAttr(GPUFuncOp op, unsigned index,
                                    StringAttr name, StringAttr attrsName) {
  DictionaryAttr dict = getAttributionAttrs(op, index, attrsName);
  if (!dict)
    return Attribute();
  return dict.get(name);
}

Attribute GPUFuncOp::getWorkgroupAttributionAttr(unsigned index,
                                                 StringAttr name) {
  assert(index < getNumWorkgroupAttributions() &&
         "index must map to a workgroup attribution");
  return getAttributionAttr(*this, index, name,
                            getWorkgroupAttribAttrsAttrName());
}

Attribute GPUFuncOp::getPrivateAttributionAttr(unsigned index,
                                               StringAttr name) {
  assert(index < getNumPrivateAttributions() &&
         "index must map to a private attribution");
  return getAttributionAttr(*this, index, name,
                            getPrivateAttribAttrsAttrName());
}

static void setAttributionAttr(GPUFuncOp op, unsigned index, StringAttr name,
                               Attribute value, StringAttr attrsName) {
  MLIRContext *ctx = op.getContext();
  SmallVector<NamedAttribute> elems;
  DictionaryAttr oldDict = getAttributionAttrs(op, index, attrsName);
  if (oldDict)
    elems.append(oldDict.getValue().begin(), oldDict.getValue().end());

  bool found = false;
  bool mustSort = true;
  for (unsigned i = 0, e = elems.size(); i < e; ++i) {
    if (elems[i].getName() == name) {
      found = true;
      if (!value) {
        std::swap(elems[i], elems[elems.size() - 1]);
        elems.pop_back();
      } else {
        mustSort = false;
        elems[i] = NamedAttribute(elems[i].getName(), value);
      }
      break;
    }
  }
  if (!found) {
    if (!value)
      return;
    elems.emplace_back(name, value);
  }
  if (mustSort) {
    DictionaryAttr::sortInPlace(elems);
  }
  auto newDict = DictionaryAttr::getWithSorted(ctx, elems);
  setAttributionAttrs(op, index, newDict, attrsName);
}

void GPUFuncOp::setWorkgroupAttributionAttr(unsigned index, StringAttr name,
                                            Attribute value) {
  assert(index < getNumWorkgroupAttributions() &&
         "index must map to a workgroup attribution");
  setAttributionAttr(*this, index, name, value,
                     getWorkgroupAttribAttrsAttrName());
}

void GPUFuncOp::setPrivateAttributionAttr(unsigned index, StringAttr name,
                                          Attribute value) {
  assert(index < getNumPrivateAttributions() &&
         "index must map to a private attribution");
  setAttributionAttr(*this, index, name, value,
                     getPrivateAttribAttrsAttrName());
}

LogicalResult GPUFuncOp::verifyType() {
  if (isKernel() && getFunctionType().getNumResults() != 0)
    return emitOpError() << "expected void return type for kernel function";

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

  if (funType.getNumResults() != getOperands().size())
    return emitOpError()
        .append("expected ", funType.getNumResults(), " result operands")
        .attachNote(function.getLoc())
        .append("return type declared here");

  for (const auto &pair : llvm::enumerate(
           llvm::zip(function.getFunctionType().getResults(), getOperands()))) {
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
                        StringRef name, ArrayAttr targets,
                        Attribute offloadingHandler) {
  result.addRegion()->emplaceBlock();
  Properties &props = result.getOrAddProperties<Properties>();
  if (targets)
    props.targets = targets;
  props.setSymName(builder.getStringAttr(name));
  props.offloadingHandler = offloadingHandler;
}

void GPUModuleOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name, ArrayRef<Attribute> targets,
                        Attribute offloadingHandler) {
  build(builder, result, name,
        targets.empty() ? ArrayAttr() : builder.getArrayAttr(targets),
        offloadingHandler);
}

bool GPUModuleOp::hasTarget(Attribute target) {
  if (ArrayAttr targets = getTargetsAttr())
    return llvm::count(targets.getValue(), target);
  return false;
}

void GPUModuleOp::setTargets(ArrayRef<TargetAttrInterface> targets) {
  ArrayAttr &targetsAttr = getProperties().targets;
  SmallVector<Attribute> targetsVector(targets);
  targetsAttr = ArrayAttr::get(getContext(), targetsVector);
}

//===----------------------------------------------------------------------===//
// GPUBinaryOp
//===----------------------------------------------------------------------===//
void BinaryOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     Attribute offloadingHandler, ArrayAttr objects) {
  auto &properties = result.getOrAddProperties<Properties>();
  result.attributes.push_back(builder.getNamedAttr(
      SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
  properties.objects = objects;
  if (offloadingHandler)
    properties.offloadingHandler = offloadingHandler;
  else
    properties.offloadingHandler = builder.getAttr<SelectObjectAttr>(nullptr);
}

void BinaryOp::build(OpBuilder &builder, OperationState &result, StringRef name,
                     Attribute offloadingHandler, ArrayRef<Attribute> objects) {
  build(builder, result, name, offloadingHandler,
        objects.empty() ? ArrayAttr() : builder.getArrayAttr(objects));
}

static ParseResult parseOffloadingHandler(OpAsmParser &parser,
                                          Attribute &offloadingHandler) {
  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseAttribute(offloadingHandler))
      return failure();
    if (parser.parseGreater())
      return failure();
  }
  if (!offloadingHandler)
    offloadingHandler = parser.getBuilder().getAttr<SelectObjectAttr>(nullptr);
  return success();
}

static void printOffloadingHandler(OpAsmPrinter &printer, Operation *op,
                                   Attribute offloadingHandler) {
  if (offloadingHandler != SelectObjectAttr::get(op->getContext(), nullptr))
    printer << '<' << offloadingHandler << '>';
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

LogicalResult SubgroupMmaLoadMatrixOp::verify() {
  auto srcType = getSrcMemref().getType();
  auto resType = getRes().getType();
  auto resMatrixType = llvm::cast<gpu::MMAMatrixType>(resType);
  auto operand = resMatrixType.getOperand();
  auto srcMemrefType = llvm::cast<MemRefType>(srcType);

  if (!isLastMemrefDimUnitStride(srcMemrefType))
    return emitError(
        "expected source memref most minor dim must have unit stride");

  if (operand != "AOp" && operand != "BOp" && operand != "COp")
    return emitError("only AOp, BOp and COp can be loaded");

  return success();
}

//===----------------------------------------------------------------------===//
// GPU_SubgroupMmaStoreMatrixOp
//===----------------------------------------------------------------------===//

LogicalResult SubgroupMmaStoreMatrixOp::verify() {
  auto srcType = getSrc().getType();
  auto dstType = getDstMemref().getType();
  auto srcMatrixType = llvm::cast<gpu::MMAMatrixType>(srcType);
  auto dstMemrefType = llvm::cast<MemRefType>(dstType);

  if (!isLastMemrefDimUnitStride(dstMemrefType))
    return emitError(
        "expected destination memref most minor dim must have unit stride");

  if (srcMatrixType.getOperand() != "COp")
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
  opTypes.push_back(llvm::cast<MMAMatrixType>(getOpA().getType()));
  opTypes.push_back(llvm::cast<MMAMatrixType>(getOpB().getType()));
  opTypes.push_back(llvm::cast<MMAMatrixType>(getOpC().getType()));

  if (opTypes[A].getOperand() != "AOp" || opTypes[B].getOperand() != "BOp" ||
      opTypes[C].getOperand() != "COp")
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

LogicalResult MemcpyOp::fold(FoldAdaptor adaptor,
                             SmallVectorImpl<::mlir::OpFoldResult> &results) {
  return memref::foldMemRefCast(*this);
}

LogicalResult MemsetOp::fold(FoldAdaptor adaptor,
                             SmallVectorImpl<::mlir::OpFoldResult> &results) {
  return memref::foldMemRefCast(*this);
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
    rewriter.modifyOpInPlace(op, [&]() { op->setOperands(validOperands); });
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
  auto memRefType = llvm::cast<MemRefType>(getMemref().getType());

  if (getDynamicSizes().size() != memRefType.getNumDynamicDims())
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
    std::optional<int64_t> index = dimOp.getConstantIndex();
    if (!index)
      return failure();

    auto memrefType = llvm::dyn_cast<MemRefType>(dimOp.getSource().getType());
    if (!memrefType || index.value() >= memrefType.getRank() ||
        !memrefType.isDynamicDim(index.value()))
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

//===----------------------------------------------------------------------===//
// GPU object attribute
//===----------------------------------------------------------------------===//

LogicalResult ObjectAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 Attribute target, CompilationTarget format,
                                 StringAttr object, DictionaryAttr properties,
                                 KernelTableAttr kernels) {
  if (!target)
    return emitError() << "the target attribute cannot be null";
  if (target.hasPromiseOrImplementsInterface<TargetAttrInterface>())
    return success();
  return emitError() << "the target attribute must implement or promise the "
                        "`gpu::TargetAttrInterface`";
}

namespace {
ParseResult parseObject(AsmParser &odsParser, CompilationTarget &format,
                        StringAttr &object) {
  std::optional<CompilationTarget> formatResult;
  StringRef enumKeyword;
  auto loc = odsParser.getCurrentLocation();
  if (failed(odsParser.parseOptionalKeyword(&enumKeyword)))
    formatResult = CompilationTarget::Fatbin;
  if (!formatResult &&
      (formatResult =
           gpu::symbolizeEnum<gpu::CompilationTarget>(enumKeyword)) &&
      odsParser.parseEqual())
    return odsParser.emitError(loc, "expected an equal sign");
  if (!formatResult)
    return odsParser.emitError(loc, "expected keyword for GPU object format");
  FailureOr<StringAttr> objectResult =
      FieldParser<StringAttr>::parse(odsParser);
  if (failed(objectResult))
    return odsParser.emitError(odsParser.getCurrentLocation(),
                               "failed to parse GPU_ObjectAttr parameter "
                               "'object' which is to be a `StringAttr`");
  format = *formatResult;
  object = *objectResult;
  return success();
}

void printObject(AsmPrinter &odsParser, CompilationTarget format,
                 StringAttr object) {
  if (format != CompilationTarget::Fatbin)
    odsParser << stringifyEnum(format) << " = ";
  odsParser << object;
}
} // namespace

//===----------------------------------------------------------------------===//
// GPU select object attribute
//===----------------------------------------------------------------------===//

LogicalResult
gpu::SelectObjectAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                              Attribute target) {
  // Check `target`, it can be null, an integer attr or a GPU Target attribute.
  if (target) {
    if (auto intAttr = mlir::dyn_cast<IntegerAttr>(target)) {
      if (intAttr.getInt() < 0) {
        return emitError() << "the object index must be positive";
      }
    } else if (!target.hasPromiseOrImplementsInterface<TargetAttrInterface>()) {
      return emitError()
             << "the target attribute must be a GPU Target attribute";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicSharedMemoryOp
//===----------------------------------------------------------------------===//

LogicalResult gpu::DynamicSharedMemoryOp::verify() {
  if (!getOperation()->getParentWithTrait<OpTrait::SymbolTable>())
    return emitOpError() << "must be inside an op with symbol table";

  MemRefType memrefType = getResultMemref().getType();
  // Check address space
  if (!GPUDialect::hasWorkgroupMemoryAddressSpace(memrefType)) {
    return emitOpError() << "address space must be "
                         << gpu::AddressSpaceAttr::getMnemonic() << "<"
                         << stringifyEnum(gpu::AddressSpace::Workgroup) << ">";
  }
  if (memrefType.hasStaticShape()) {
    return emitOpError() << "result memref type must be memref<?xi8, "
                            "#gpu.address_space<workgroup>>";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GPU WarpExecuteOnLane0Op
//===----------------------------------------------------------------------===//

void WarpExecuteOnLane0Op::print(OpAsmPrinter &p) {
  p << "(" << getLaneid() << ")";

  SmallVector<StringRef> coreAttr = {getWarpSizeAttrName()};
  auto warpSizeAttr = getOperation()->getAttr(getWarpSizeAttrName());
  p << "[" << llvm::cast<IntegerAttr>(warpSizeAttr).getInt() << "]";

  if (!getArgs().empty())
    p << " args(" << getArgs() << " : " << getArgs().getTypes() << ")";
  if (!getResults().empty())
    p << " -> (" << getResults().getTypes() << ')';
  p << " ";
  p.printRegion(getRegion(),
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/!getResults().empty());
  p.printOptionalAttrDict(getOperation()->getAttrs(), coreAttr);
}

ParseResult WarpExecuteOnLane0Op::parse(OpAsmParser &parser,
                                        OperationState &result) {
  // Create the region.
  result.regions.reserve(1);
  Region *warpRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand laneId;

  // Parse predicate operand.
  if (parser.parseLParen() ||
      parser.parseOperand(laneId, /*allowResultNumber=*/false) ||
      parser.parseRParen())
    return failure();

  int64_t warpSize;
  if (parser.parseLSquare() || parser.parseInteger(warpSize) ||
      parser.parseRSquare())
    return failure();
  result.addAttribute(getWarpSizeAttrName(OperationName(getOperationName(),
                                                        builder.getContext())),
                      builder.getI64IntegerAttr(warpSize));

  if (parser.resolveOperand(laneId, builder.getIndexType(), result.operands))
    return failure();

  llvm::SMLoc inputsOperandsLoc;
  SmallVector<OpAsmParser::UnresolvedOperand> inputsOperands;
  SmallVector<Type> inputTypes;
  if (succeeded(parser.parseOptionalKeyword("args"))) {
    if (parser.parseLParen())
      return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }
  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands))
    return failure();

  // Parse optional results type list.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();
  // Parse the region.
  if (parser.parseRegion(*warpRegion, /*arguments=*/{},
                         /*argTypes=*/{}))
    return failure();
  WarpExecuteOnLane0Op::ensureTerminator(*warpRegion, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void WarpExecuteOnLane0Op::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  // The warp region is always executed
  regions.push_back(RegionSuccessor(&getWarpRegion()));
}

void WarpExecuteOnLane0Op::build(OpBuilder &builder, OperationState &result,
                                 TypeRange resultTypes, Value laneId,
                                 int64_t warpSize) {
  build(builder, result, resultTypes, laneId, warpSize,
        /*operands=*/std::nullopt, /*argTypes=*/std::nullopt);
}

void WarpExecuteOnLane0Op::build(OpBuilder &builder, OperationState &result,
                                 TypeRange resultTypes, Value laneId,
                                 int64_t warpSize, ValueRange args,
                                 TypeRange blockArgTypes) {
  result.addOperands(laneId);
  result.addAttribute(getAttributeNames()[0],
                      builder.getI64IntegerAttr(warpSize));
  result.addTypes(resultTypes);
  result.addOperands(args);
  assert(args.size() == blockArgTypes.size());
  OpBuilder::InsertionGuard guard(builder);
  Region *warpRegion = result.addRegion();
  Block *block = builder.createBlock(warpRegion);
  for (auto [type, arg] : llvm::zip_equal(blockArgTypes, args))
    block->addArgument(type, arg.getLoc());
}

/// Helper check if the distributed vector type is consistent with the expanded
/// type and distributed size.
static LogicalResult verifyDistributedType(Type expanded, Type distributed,
                                           int64_t warpSize, Operation *op) {
  // If the types matches there is no distribution.
  if (expanded == distributed)
    return success();
  auto expandedVecType = llvm::dyn_cast<VectorType>(expanded);
  auto distributedVecType = llvm::dyn_cast<VectorType>(distributed);
  if (!expandedVecType || !distributedVecType)
    return op->emitOpError("expected vector type for distributed operands.");
  if (expandedVecType.getRank() != distributedVecType.getRank() ||
      expandedVecType.getElementType() != distributedVecType.getElementType())
    return op->emitOpError(
        "expected distributed vectors to have same rank and element type.");

  SmallVector<int64_t> scales(expandedVecType.getRank(), 1);
  for (int64_t i = 0, e = expandedVecType.getRank(); i < e; i++) {
    int64_t eDim = expandedVecType.getDimSize(i);
    int64_t dDim = distributedVecType.getDimSize(i);
    if (eDim == dDim)
      continue;
    if (eDim % dDim != 0)
      return op->emitOpError()
             << "expected expanded vector dimension #" << i << " (" << eDim
             << ") to be a multipler of the distributed vector dimension ("
             << dDim << ")";
    scales[i] = eDim / dDim;
  }
  if (std::accumulate(scales.begin(), scales.end(), 1,
                      std::multiplies<int64_t>()) != warpSize)
    return op->emitOpError()
           << "incompatible distribution dimensions from " << expandedVecType
           << " to " << distributedVecType << " with warp size = " << warpSize;

  return success();
}

LogicalResult WarpExecuteOnLane0Op::verify() {
  if (getArgs().size() != getWarpRegion().getNumArguments())
    return emitOpError(
        "expected same number op arguments and block arguments.");
  auto yield =
      cast<YieldOp>(getWarpRegion().getBlocks().begin()->getTerminator());
  if (yield.getNumOperands() != getNumResults())
    return emitOpError(
        "expected same number of yield operands and return values.");
  int64_t warpSize = getWarpSize();
  for (auto [regionArg, arg] :
       llvm::zip_equal(getWarpRegion().getArguments(), getArgs())) {
    if (failed(verifyDistributedType(regionArg.getType(), arg.getType(),
                                     warpSize, getOperation())))
      return failure();
  }
  for (auto [yieldOperand, result] :
       llvm::zip_equal(yield.getOperands(), getResults())) {
    if (failed(verifyDistributedType(yieldOperand.getType(), result.getType(),
                                     warpSize, getOperation())))
      return failure();
  }
  return success();
}
bool WarpExecuteOnLane0Op::areTypesCompatible(Type lhs, Type rhs) {
  return succeeded(
      verifyDistributedType(lhs, rhs, getWarpSize(), getOperation()));
}

//===----------------------------------------------------------------------===//
// GPU KernelMetadataAttr
//===----------------------------------------------------------------------===//

KernelMetadataAttr KernelMetadataAttr::get(FunctionOpInterface kernel,
                                           DictionaryAttr metadata) {
  assert(kernel && "invalid kernel");
  return get(kernel.getNameAttr(), kernel.getFunctionType(),
             kernel.getAllArgAttrs(), metadata);
}

KernelMetadataAttr
KernelMetadataAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                               FunctionOpInterface kernel,
                               DictionaryAttr metadata) {
  assert(kernel && "invalid kernel");
  return getChecked(emitError, kernel.getNameAttr(), kernel.getFunctionType(),
                    kernel.getAllArgAttrs(), metadata);
}

KernelMetadataAttr
KernelMetadataAttr::appendMetadata(ArrayRef<NamedAttribute> attrs) const {
  if (attrs.empty())
    return *this;
  NamedAttrList attrList;
  if (DictionaryAttr dict = getMetadata())
    attrList.append(dict);
  attrList.append(attrs);
  return KernelMetadataAttr::get(getName(), getFunctionType(), getArgAttrs(),
                                 attrList.getDictionary(getContext()));
}

LogicalResult
KernelMetadataAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           StringAttr name, Type functionType,
                           ArrayAttr argAttrs, DictionaryAttr metadata) {
  if (name.empty())
    return emitError() << "the kernel name can't be empty";
  if (argAttrs) {
    if (llvm::any_of(argAttrs, [](Attribute attr) {
          return !llvm::isa<DictionaryAttr>(attr);
        }))
      return emitError()
             << "all attributes in the array must be a dictionary attribute";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GPU KernelTableAttr
//===----------------------------------------------------------------------===//

KernelTableAttr KernelTableAttr::get(MLIRContext *context,
                                     ArrayRef<KernelMetadataAttr> kernels,
                                     bool isSorted) {
  // Note that `is_sorted` is always only invoked once even with assertions ON.
  assert((!isSorted || llvm::is_sorted(kernels)) &&
         "expected a sorted kernel array");
  // Immediately return the attribute if the array is sorted.
  if (isSorted || llvm::is_sorted(kernels))
    return Base::get(context, kernels);
  // Sort the array.
  SmallVector<KernelMetadataAttr> kernelsTmp(kernels);
  llvm::array_pod_sort(kernelsTmp.begin(), kernelsTmp.end());
  return Base::get(context, kernelsTmp);
}

KernelTableAttr KernelTableAttr::getChecked(
    function_ref<InFlightDiagnostic()> emitError, MLIRContext *context,
    ArrayRef<KernelMetadataAttr> kernels, bool isSorted) {
  // Note that `is_sorted` is always only invoked once even with assertions ON.
  assert((!isSorted || llvm::is_sorted(kernels)) &&
         "expected a sorted kernel array");
  // Immediately return the attribute if the array is sorted.
  if (isSorted || llvm::is_sorted(kernels))
    return Base::getChecked(emitError, context, kernels);
  // Sort the array.
  SmallVector<KernelMetadataAttr> kernelsTmp(kernels);
  llvm::array_pod_sort(kernelsTmp.begin(), kernelsTmp.end());
  return Base::getChecked(emitError, context, kernelsTmp);
}

LogicalResult
KernelTableAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        ArrayRef<KernelMetadataAttr> kernels) {
  if (kernels.size() < 2)
    return success();
  // Check that the kernels are uniquely named.
  if (std::adjacent_find(kernels.begin(), kernels.end(),
                         [](KernelMetadataAttr l, KernelMetadataAttr r) {
                           return l.getName() == r.getName();
                         }) != kernels.end()) {
    return emitError() << "expected all kernels to be uniquely named";
  }
  return success();
}

KernelMetadataAttr KernelTableAttr::lookup(StringRef key) const {
  auto [iterator, found] = impl::findAttrSorted(begin(), end(), key);
  return found ? *iterator : KernelMetadataAttr();
}

KernelMetadataAttr KernelTableAttr::lookup(StringAttr key) const {
  auto [iterator, found] = impl::findAttrSorted(begin(), end(), key);
  return found ? *iterator : KernelMetadataAttr();
}

//===----------------------------------------------------------------------===//
// GPU target options
//===----------------------------------------------------------------------===//

TargetOptions::TargetOptions(
    StringRef toolkitPath, ArrayRef<Attribute> librariesToLink,
    StringRef cmdOptions, StringRef elfSection,
    CompilationTarget compilationTarget,
    function_ref<SymbolTable *()> getSymbolTableCallback,
    function_ref<void(llvm::Module &)> initialLlvmIRCallback,
    function_ref<void(llvm::Module &)> linkedLlvmIRCallback,
    function_ref<void(llvm::Module &)> optimizedLlvmIRCallback,
    function_ref<void(StringRef)> isaCallback)
    : TargetOptions(TypeID::get<TargetOptions>(), toolkitPath, librariesToLink,
                    cmdOptions, elfSection, compilationTarget,
                    getSymbolTableCallback, initialLlvmIRCallback,
                    linkedLlvmIRCallback, optimizedLlvmIRCallback,
                    isaCallback) {}

TargetOptions::TargetOptions(
    TypeID typeID, StringRef toolkitPath, ArrayRef<Attribute> librariesToLink,
    StringRef cmdOptions, StringRef elfSection,
    CompilationTarget compilationTarget,
    function_ref<SymbolTable *()> getSymbolTableCallback,
    function_ref<void(llvm::Module &)> initialLlvmIRCallback,
    function_ref<void(llvm::Module &)> linkedLlvmIRCallback,
    function_ref<void(llvm::Module &)> optimizedLlvmIRCallback,
    function_ref<void(StringRef)> isaCallback)
    : toolkitPath(toolkitPath.str()), librariesToLink(librariesToLink),
      cmdOptions(cmdOptions.str()), elfSection(elfSection.str()),
      compilationTarget(compilationTarget),
      getSymbolTableCallback(getSymbolTableCallback),
      initialLlvmIRCallback(initialLlvmIRCallback),
      linkedLlvmIRCallback(linkedLlvmIRCallback),
      optimizedLlvmIRCallback(optimizedLlvmIRCallback),
      isaCallback(isaCallback), typeID(typeID) {}

TypeID TargetOptions::getTypeID() const { return typeID; }

StringRef TargetOptions::getToolkitPath() const { return toolkitPath; }

ArrayRef<Attribute> TargetOptions::getLibrariesToLink() const {
  return librariesToLink;
}

StringRef TargetOptions::getCmdOptions() const { return cmdOptions; }

StringRef TargetOptions::getELFSection() const { return elfSection; }

SymbolTable *TargetOptions::getSymbolTable() const {
  return getSymbolTableCallback ? getSymbolTableCallback() : nullptr;
}

function_ref<void(llvm::Module &)>
TargetOptions::getInitialLlvmIRCallback() const {
  return initialLlvmIRCallback;
}

function_ref<void(llvm::Module &)>
TargetOptions::getLinkedLlvmIRCallback() const {
  return linkedLlvmIRCallback;
}

function_ref<void(llvm::Module &)>
TargetOptions::getOptimizedLlvmIRCallback() const {
  return optimizedLlvmIRCallback;
}

function_ref<void(StringRef)> TargetOptions::getISACallback() const {
  return isaCallback;
}

CompilationTarget TargetOptions::getCompilationTarget() const {
  return compilationTarget;
}

CompilationTarget TargetOptions::getDefaultCompilationTarget() {
  return CompilationTarget::Fatbin;
}

std::pair<llvm::BumpPtrAllocator, SmallVector<const char *>>
TargetOptions::tokenizeCmdOptions() const {
  std::pair<llvm::BumpPtrAllocator, SmallVector<const char *>> options;
  llvm::StringSaver stringSaver(options.first);
  StringRef opts = cmdOptions;
  // For a correct tokenization of the command line options `opts` must be
  // unquoted, otherwise the tokenization function returns a single string: the
  // unquoted `cmdOptions` -which is not the desired behavior.
  // Remove any quotes if they are at the beginning and end of the string:
  if (!opts.empty() && opts.front() == '"' && opts.back() == '"')
    opts.consume_front("\""), opts.consume_back("\"");
  if (!opts.empty() && opts.front() == '\'' && opts.back() == '\'')
    opts.consume_front("'"), opts.consume_back("'");
#ifdef _WIN32
  llvm::cl::TokenizeWindowsCommandLine(opts, stringSaver, options.second,
                                       /*MarkEOLs=*/false);
#else
  llvm::cl::TokenizeGNUCommandLine(opts, stringSaver, options.second,
                                   /*MarkEOLs=*/false);
#endif // _WIN32
  return options;
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::TargetOptions)

#include "mlir/Dialect/GPU/IR/GPUOpInterfaces.cpp.inc"
#include "mlir/Dialect/GPU/IR/GPUOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/GPU/IR/GPUOpsAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/IR/GPUOps.cpp.inc"

#include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.cpp.inc"
