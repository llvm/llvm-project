//===- GPUBase.cpp - GPU types and attributes -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements GPU types, attributes, and enums.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUBase.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/StringSaver.h"

using namespace mlir;
using namespace mlir::gpu;

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

int64_t GPULaneMappingAttr::getMappingId() const {
  return static_cast<int64_t>(getLane());
}

bool GPULaneMappingAttr::isLinearMapping() const {
  return getMappingId() >= static_cast<int64_t>(MappingId::LinearDim0);
}

int64_t GPULaneMappingAttr::getRelativeIndex() const {
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
  return elementType.isF16() || elementType.isF32() || elementType.isF64() ||
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
           << "MMAMatrixType elements must be SI8, UI8, I32, F16, F32, or F64";

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

bool GPUDialect::isConstantMemoryAddressSpace(Attribute memorySpace) {
  if (!memorySpace)
    return false;
  if (auto gpuAttr = llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    return gpuAttr.getValue() == getConstantAddressSpace();
  return false;
}

bool GPUDialect::hasConstantMemoryAddressSpace(MemRefType type) {
  Attribute memorySpace = type.getMemorySpace();
  return isConstantMemoryAddressSpace(memorySpace);
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
TargetOptions::tokenizeCmdOptions(const std::string &cmdOptions) {
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

std::pair<llvm::BumpPtrAllocator, SmallVector<const char *>>
TargetOptions::tokenizeCmdOptions() const {
  return tokenizeCmdOptions(cmdOptions);
}

std::pair<llvm::BumpPtrAllocator, SmallVector<const char *>>
TargetOptions::tokenizeAndRemoveSuffixCmdOptions(llvm::StringRef startsWith) {
  size_t startPos = cmdOptions.find(startsWith);
  if (startPos == std::string::npos)
    return {llvm::BumpPtrAllocator(), SmallVector<const char *>()};

  auto tokenized =
      tokenizeCmdOptions(cmdOptions.substr(startPos + startsWith.size()));
  cmdOptions.resize(startPos);
  return tokenized;
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::TargetOptions)

#include "mlir/Dialect/GPU/IR/GPUOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/GPU/IR/GPUOpsAttributes.cpp.inc"

namespace {
template <typename... Attrs>
void registerAttributeStorageImpl(MLIRContext *context) {
  (mlir::detail::AttributeUniquer::registerAttribute<Attrs>(context), ...);
}
} // namespace

void mlir::gpu::detail::registerAttributeStorage(MLIRContext *context) {
  registerAttributeStorageImpl<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/GPU/IR/GPUOpsAttributes.cpp.inc"
      >(context);
}

#include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.cpp.inc"
