//===- ROCDLDialect.cpp - ROCDL IR Ops and Dialect registration -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the types and operation details for the ROCDL IR dialect in
// MLIR, and the LLVM IR dialect.  It also registers the dialect.
//
// The ROCDL dialect only contains GPU specific additions on top of the general
// LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace ROCDL;

#include "mlir/Dialect/LLVMIR/ROCDLOpsDialect.cpp.inc"
#include "mlir/Dialect/LLVMIR/ROCDLOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// ROCDLDialect initialization, type parsing, and registration.
//===----------------------------------------------------------------------===//

namespace {
struct ROCDLInlinerInterface final : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

// TODO: This should be the llvm.rocdl dialect once this is supported.
void ROCDLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/LLVMIR/ROCDLOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/ROCDLOpsAttributes.cpp.inc"
      >();

  // Support unknown operations because not all ROCDL operations are registered.
  allowUnknownOperations();
  addInterfaces<ROCDLInlinerInterface>();
  declarePromisedInterface<gpu::TargetAttrInterface, ROCDLTargetAttr>();
}

LogicalResult ROCDLDialect::verifyOperationAttribute(Operation *op,
                                                     NamedAttribute attr) {
  // Kernel function attribute should be attached to functions.
  if (kernelAttrName.getName() == attr.getName()) {
    if (!isa<LLVM::LLVMFuncOp>(op)) {
      return op->emitError() << "'" << kernelAttrName.getName()
                             << "' attribute attached to unexpected op";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ROCDL op custom parsers/printers.
//===----------------------------------------------------------------------===//

template <typename EnumAttrT, typename EnumT>
static ParseResult parseCachePolicyEnum(OpAsmParser &parser,
                                        Attribute &cachePolicy) {
  if (parser.parseLess())
    return failure();
  FailureOr<EnumT> parsed = FieldParser<EnumT>::parse(parser);
  if (failed(parsed))
    return failure();
  if (parser.parseGreater())
    return failure();
  cachePolicy = EnumAttrT::get(parser.getContext(), *parsed);
  return success();
}

static ParseResult parseCachePolicy(OpAsmParser &parser,
                                    Attribute &cachePolicy) {
  uint32_t rawValue;
  OptionalParseResult rawValueParseResult =
      parser.parseOptionalInteger(rawValue);
  if (rawValueParseResult.has_value()) {
    if (failed(*rawValueParseResult))
      return failure();
    cachePolicy =
        IntegerAttr::get(IntegerType::get(parser.getContext(), 32), rawValue);
    return success();
  }

  StringRef policyFamily;
  auto loc = parser.getCurrentLocation();
  if (failed(parser.parseOptionalKeyword(
          &policyFamily, {"pre_gfx12", "gfx942", "gfx12", "gfx12_atomic"}))) {
    return parser.emitError(loc)
           << "expected cache policy family 'pre_gfx12', 'gfx942', 'gfx12', "
              "'gfx12_atomic', or a 32-bit integer";
  }

  if (policyFamily == "pre_gfx12")
    return parseCachePolicyEnum<PreGfx12CachePolicyAttr, PreGfx12CachePolicy>(
        parser, cachePolicy);
  if (policyFamily == "gfx942")
    return parseCachePolicyEnum<Gfx942CachePolicyAttr, Gfx942CachePolicy>(
        parser, cachePolicy);
  if (policyFamily == "gfx12")
    return parseCachePolicyEnum<Gfx12CachePolicyAttr, Gfx12CachePolicy>(
        parser, cachePolicy);
  return parseCachePolicyEnum<Gfx12AtomicCachePolicyAttr,
                              Gfx12AtomicCachePolicy>(parser, cachePolicy);
}

template <typename EnumAttrT>
static void printCachePolicyEnum(OpAsmPrinter &printer, EnumAttrT cachePolicy,
                                 StringRef family) {
  printer << family << "<" << cachePolicy.getValue() << ">";
}

static void printCachePolicy(OpAsmPrinter &printer, Operation *,
                             Attribute cachePolicy) {
  llvm::TypeSwitch<Attribute>(cachePolicy)
      .Case<IntegerAttr>([&](IntegerAttr rawPolicy) {
        printer << rawPolicy.getValue().getZExtValue();
      })
      .Case<PreGfx12CachePolicyAttr>([&](PreGfx12CachePolicyAttr policy) {
        printCachePolicyEnum(printer, policy, "pre_gfx12");
      })
      .Case<Gfx942CachePolicyAttr>([&](Gfx942CachePolicyAttr policy) {
        printCachePolicyEnum(printer, policy, "gfx942");
      })
      .Case<Gfx12CachePolicyAttr>([&](Gfx12CachePolicyAttr policy) {
        printCachePolicyEnum(printer, policy, "gfx12");
      })
      .Case<Gfx12AtomicCachePolicyAttr>([&](Gfx12AtomicCachePolicyAttr policy) {
        printCachePolicyEnum(printer, policy, "gfx12_atomic");
      })
      .DefaultUnreachable("unknown ROCDL cache policy attribute");
}

//===----------------------------------------------------------------------===//
// ROCDL target attribute.
//===----------------------------------------------------------------------===//
LogicalResult
ROCDLTargetAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                        int optLevel, StringRef triple, StringRef chip,
                        StringRef features, StringRef abiVersion,
                        DictionaryAttr flags, ArrayAttr files) {
  if (optLevel < 0 || optLevel > 3) {
    emitError() << "The optimization level must be a number between 0 and 3.";
    return failure();
  }
  if (triple.empty()) {
    emitError() << "The target triple cannot be empty.";
    return failure();
  }
  if (chip.empty()) {
    emitError() << "The target chip cannot be empty.";
    return failure();
  }
  if (abiVersion != "400" && abiVersion != "500" && abiVersion != "600") {
    emitError() << "Invalid ABI version, it must be `400`, `500` or '600'.";
    return failure();
  }
  if (files && !llvm::all_of(files, [](::mlir::Attribute attr) {
        return mlir::isa_and_nonnull<StringAttr>(attr);
      })) {
    emitError() << "All the elements in the `link` array must be strings.";
    return failure();
  }
  return success();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/ROCDLOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/ROCDLOpsAttributes.cpp.inc"
