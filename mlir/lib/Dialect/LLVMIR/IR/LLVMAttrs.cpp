//===- LLVMAttrs.cpp - LLVM Attributes registration -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the attribute details for the LLVM IR dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include <optional>

using namespace mlir;
using namespace mlir::LLVM;

/// Parses DWARF expression arguments with respect to the DWARF operation
/// opcode. Some DWARF expression operations have a specific number of operands
/// and may appear in a textual form.
static LogicalResult parseExpressionArg(AsmParser &parser, uint64_t opcode,
                                        SmallVector<uint64_t> &args);

/// Prints DWARF expression arguments with respect to the specific DWARF
/// operation. Some operands are printed in their textual form.
static void printExpressionArg(AsmPrinter &printer, uint64_t opcode,
                               ArrayRef<uint64_t> args);

#include "mlir/Dialect/LLVMIR/LLVMAttrInterfaces.cpp.inc"
#include "mlir/Dialect/LLVMIR/LLVMOpsEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// LLVMDialect registration
//===----------------------------------------------------------------------===//

void LLVMDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AddressSpaceAttr
//===----------------------------------------------------------------------===//

static bool isLoadableType(Type type) {
  return /*LLVM_PrimitiveType*/ (
             LLVM::isCompatibleOuterType(type) &&
             !isa<LLVM::LLVMVoidType, LLVM::LLVMFunctionType>(type)) &&
         /*LLVM_OpaqueStruct*/
         !(isa<LLVM::LLVMStructType>(type) &&
           cast<LLVM::LLVMStructType>(type).isOpaque()) &&
         /*LLVM_AnyTargetExt*/
         !(isa<LLVM::LLVMTargetExtType>(type) &&
           !cast<LLVM::LLVMTargetExtType>(type).supportsMemOps());
}

/// Returns true if the given type is supported by atomic operations. All
/// integer and float types with limited bit width are supported. Additionally,
/// depending on the operation pointers may be supported as well.
static bool isTypeCompatibleWithAtomicOp(Type type) {
  if (llvm::isa<LLVMPointerType>(type))
    return true;

  std::optional<unsigned> bitWidth;
  if (auto floatType = llvm::dyn_cast<FloatType>(type)) {
    if (!isCompatibleFloatingPointType(type))
      return false;
    bitWidth = floatType.getWidth();
  }
  if (auto integerType = llvm::dyn_cast<IntegerType>(type))
    bitWidth = integerType.getWidth();
  // The type is neither an integer, float, or pointer type.
  if (!bitWidth)
    return false;
  return *bitWidth == 8 || *bitWidth == 16 || *bitWidth == 32 ||
         *bitWidth == 64;
}

Dialect *AddressSpaceAttr::getMemorySpaceDialect() const {
  return &getDialect();
}

Attribute AddressSpaceAttr::getDefaultMemorySpace() const {
  return AddressSpaceAttr::get(getContext(), 0);
}

unsigned AddressSpaceAttr::getAddressSpace() const { return getAs(); }

LogicalResult AddressSpaceAttr::isValidLoad(Type type,
                                            mlir::ptr::AtomicOrdering ordering,
                                            IntegerAttr alignment,
                                            Operation *diagnosticOp) const {
  if (!isLoadableType(type))
    return diagnosticOp ? diagnosticOp->emitError(
                              "type must be LLVM type with size, but got ")
                              << type
                        : failure();
  if (ordering != ptr::AtomicOrdering::not_atomic &&
      !isTypeCompatibleWithAtomicOp(type))
    return diagnosticOp ? diagnosticOp->emitError("unsupported type ")
                              << type << " for atomic access"
                        : failure();
  return success();
}

LogicalResult AddressSpaceAttr::isValidStore(Type type,
                                             mlir::ptr::AtomicOrdering ordering,
                                             IntegerAttr alignment,
                                             Operation *diagnosticOp) const {
  if (!isLoadableType(type))
    return diagnosticOp ? diagnosticOp->emitError(
                              "type must be LLVM type with size, but got ")
                              << type
                        : failure();
  if (ordering != ptr::AtomicOrdering::not_atomic &&
      !isTypeCompatibleWithAtomicOp(type))
    return diagnosticOp ? diagnosticOp->emitError("unsupported type ")
                              << type << " for atomic access"
                        : failure();
  return success();
}

LogicalResult AddressSpaceAttr::isValidAtomicOp(
    mlir::ptr::AtomicBinOp binOp, Type type, mlir::ptr::AtomicOrdering ordering,
    IntegerAttr alignment, Operation *diagnosticOp) const {
  if (binOp == ptr::AtomicBinOp::fadd || binOp == ptr::AtomicBinOp::fsub ||
      binOp == ptr::AtomicBinOp::fmin || binOp == ptr::AtomicBinOp::fmax) {
    if (!mlir::LLVM::isCompatibleFloatingPointType(type))
      return diagnosticOp ? diagnosticOp->emitError(
                                "expected LLVM IR floating point type")
                          : failure();
  } else if (binOp == ptr::AtomicBinOp::xchg) {
    if (!isTypeCompatibleWithAtomicOp(type))
      return diagnosticOp ? diagnosticOp->emitError(
                                "unexpected LLVM IR type for 'xchg' bin_op")
                          : failure();
  } else {
    auto intType = llvm::dyn_cast<IntegerType>(type);
    unsigned intBitWidth = intType ? intType.getWidth() : 0;
    if (intBitWidth != 8 && intBitWidth != 16 && intBitWidth != 32 &&
        intBitWidth != 64)
      return diagnosticOp
                 ? diagnosticOp->emitError("expected LLVM IR integer type")
                 : failure();
  }
  return success();
}

LogicalResult AddressSpaceAttr::isValidAtomicXchg(
    Type type, mlir::ptr::AtomicOrdering successOrdering,
    mlir::ptr::AtomicOrdering failureOrdering, IntegerAttr alignment,
    Operation *diagnosticOp) const {
  if (!isLoadableType(type))
    return diagnosticOp ? diagnosticOp->emitError(
                              "type must be LLVM type with size, but got ")
                              << type
                        : failure();
  if (!isTypeCompatibleWithAtomicOp(type))
    return diagnosticOp ? diagnosticOp->emitError("unexpected LLVM IR type")
                        : failure();
  return success();
}

template <typename Ty>
static bool isScalarOrVectorOf(Type ty) {
  return isa<Ty>(ty) || (LLVM::isCompatibleVectorType(ty) &&
                         isa<Ty>(LLVM::getVectorElementType(ty)));
}

LogicalResult
AddressSpaceAttr::isValidAddrSpaceCast(Type tgt, Type src,
                                       Operation *diagnosticOp) const {
  if (!isScalarOrVectorOf<LLVMPointerType>(tgt))
    return diagnosticOp ? diagnosticOp->emitError("invalid ptr-like operand")
                        : failure();
  if (!isScalarOrVectorOf<LLVMPointerType>(src))
    return diagnosticOp ? diagnosticOp->emitError("invalid ptr-like operand")
                        : failure();
  return success();
}

LogicalResult
AddressSpaceAttr::isValidPtrIntCast(Type intLikeTy, Type ptrLikeTy,
                                    Operation *diagnosticOp) const {
  if (!isScalarOrVectorOf<IntegerType>(intLikeTy))
    return diagnosticOp ? diagnosticOp->emitError("invalid int-like type")
                        : failure();
  if (!isScalarOrVectorOf<LLVMPointerType>(ptrLikeTy))
    return diagnosticOp ? diagnosticOp->emitError("invalid ptr-like type")
                        : failure();
  return success();
}

//===----------------------------------------------------------------------===//
// DINodeAttr
//===----------------------------------------------------------------------===//

bool DINodeAttr::classof(Attribute attr) {
  return llvm::isa<DIBasicTypeAttr, DICompileUnitAttr, DICompositeTypeAttr,
                   DIDerivedTypeAttr, DIFileAttr, DIGlobalVariableAttr,
                   DILabelAttr, DILexicalBlockAttr, DILexicalBlockFileAttr,
                   DILocalVariableAttr, DIModuleAttr, DINamespaceAttr,
                   DINullTypeAttr, DISubprogramAttr, DISubrangeAttr,
                   DISubroutineTypeAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DIScopeAttr
//===----------------------------------------------------------------------===//

bool DIScopeAttr::classof(Attribute attr) {
  return llvm::isa<DICompileUnitAttr, DICompositeTypeAttr, DIFileAttr,
                   DILocalScopeAttr, DIModuleAttr, DINamespaceAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DILocalScopeAttr
//===----------------------------------------------------------------------===//

bool DILocalScopeAttr::classof(Attribute attr) {
  return llvm::isa<DILexicalBlockAttr, DILexicalBlockFileAttr,
                   DISubprogramAttr>(attr);
}

//===----------------------------------------------------------------------===//
// DITypeAttr
//===----------------------------------------------------------------------===//

bool DITypeAttr::classof(Attribute attr) {
  return llvm::isa<DINullTypeAttr, DIBasicTypeAttr, DICompositeTypeAttr,
                   DIDerivedTypeAttr, DISubroutineTypeAttr>(attr);
}

//===----------------------------------------------------------------------===//
// MemoryEffectsAttr
//===----------------------------------------------------------------------===//

MemoryEffectsAttr MemoryEffectsAttr::get(MLIRContext *context,
                                         ArrayRef<ModRefInfo> memInfoArgs) {
  if (memInfoArgs.empty())
    return MemoryEffectsAttr::get(context, ModRefInfo::ModRef,
                                  ModRefInfo::ModRef, ModRefInfo::ModRef);
  if (memInfoArgs.size() == 3)
    return MemoryEffectsAttr::get(context, memInfoArgs[0], memInfoArgs[1],
                                  memInfoArgs[2]);
  return {};
}

bool MemoryEffectsAttr::isReadWrite() {
  if (this->getArgMem() != ModRefInfo::ModRef)
    return false;
  if (this->getInaccessibleMem() != ModRefInfo::ModRef)
    return false;
  if (this->getOther() != ModRefInfo::ModRef)
    return false;
  return true;
}

//===----------------------------------------------------------------------===//
// DIExpression
//===----------------------------------------------------------------------===//

DIExpressionAttr DIExpressionAttr::get(MLIRContext *context) {
  return get(context, ArrayRef<DIExpressionElemAttr>({}));
}

LogicalResult parseExpressionArg(AsmParser &parser, uint64_t opcode,
                                 SmallVector<uint64_t> &args) {
  auto operandParser = [&]() -> LogicalResult {
    uint64_t operand = 0;
    if (!args.empty() && opcode == llvm::dwarf::DW_OP_LLVM_convert) {
      // Attempt to parse a keyword.
      StringRef keyword;
      if (succeeded(parser.parseOptionalKeyword(&keyword))) {
        operand = llvm::dwarf::getAttributeEncoding(keyword);
        if (operand == 0) {
          // The keyword is invalid.
          return parser.emitError(parser.getCurrentLocation())
                 << "encountered unknown attribute encoding \"" << keyword
                 << "\"";
        }
      }
    }

    // operand should be non-zero if a keyword was parsed. Otherwise, the
    // operand MUST be an integer.
    if (operand == 0) {
      // Parse the next operand as an integer.
      if (parser.parseInteger(operand)) {
        return parser.emitError(parser.getCurrentLocation())
               << "expected integer operand";
      }
    }

    args.push_back(operand);
    return success();
  };

  // Parse operands as a comma-separated list.
  return parser.parseCommaSeparatedList(operandParser);
}

void printExpressionArg(AsmPrinter &printer, uint64_t opcode,
                        ArrayRef<uint64_t> args) {
  size_t i = 0;
  llvm::interleaveComma(args, printer, [&](uint64_t operand) {
    if (i > 0 && opcode == llvm::dwarf::DW_OP_LLVM_convert) {
      if (const StringRef keyword =
              llvm::dwarf::AttributeEncodingString(operand);
          !keyword.empty()) {
        printer << keyword;
        return;
      }
    }
    // All operands are expected to be printed as integers.
    printer << operand;
    i++;
  });
}

//===----------------------------------------------------------------------===//
// DICompositeTypeAttr
//===----------------------------------------------------------------------===//

DIRecursiveTypeAttrInterface
DICompositeTypeAttr::withRecId(DistinctAttr recId) {
  return DICompositeTypeAttr::get(getContext(), getTag(), recId, getName(),
                                  getFile(), getLine(), getScope(),
                                  getBaseType(), getFlags(), getSizeInBits(),
                                  getAlignInBits(), getElements());
}

DIRecursiveTypeAttrInterface
DICompositeTypeAttr::getRecSelf(DistinctAttr recId) {
  return DICompositeTypeAttr::get(recId.getContext(), 0, recId, {}, {}, 0, {},
                                  {}, DIFlags(), 0, 0, {});
}

//===----------------------------------------------------------------------===//
// TargetFeaturesAttr
//===----------------------------------------------------------------------===//

TargetFeaturesAttr TargetFeaturesAttr::get(MLIRContext *context,
                                           llvm::ArrayRef<StringRef> features) {
  return Base::get(context,
                   llvm::map_to_vector(features, [&](StringRef feature) {
                     return StringAttr::get(context, feature);
                   }));
}

TargetFeaturesAttr TargetFeaturesAttr::get(MLIRContext *context,
                                           StringRef targetFeatures) {
  SmallVector<StringRef> features;
  targetFeatures.split(features, ',', /*MaxSplit=*/-1,
                       /*KeepEmpty=*/false);
  return get(context, features);
}

LogicalResult
TargetFeaturesAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                           llvm::ArrayRef<StringAttr> features) {
  for (StringAttr featureAttr : features) {
    if (!featureAttr || featureAttr.empty())
      return emitError() << "target features can not be null or empty";
    auto feature = featureAttr.strref();
    if (feature[0] != '+' && feature[0] != '-')
      return emitError() << "target features must start with '+' or '-'";
    if (feature.contains(','))
      return emitError() << "target features can not contain ','";
  }
  return success();
}

bool TargetFeaturesAttr::contains(StringAttr feature) const {
  if (nullOrEmpty())
    return false;
  // Note: Using StringAttr does pointer comparisons.
  return llvm::is_contained(getFeatures(), feature);
}

bool TargetFeaturesAttr::contains(StringRef feature) const {
  if (nullOrEmpty())
    return false;
  return llvm::is_contained(getFeatures(), feature);
}

std::string TargetFeaturesAttr::getFeaturesString() const {
  std::string featuresString;
  llvm::raw_string_ostream ss(featuresString);
  llvm::interleave(
      getFeatures(), ss, [&](auto &feature) { ss << feature.strref(); }, ",");
  return ss.str();
}

TargetFeaturesAttr TargetFeaturesAttr::featuresAt(Operation *op) {
  auto parentFunction = op->getParentOfType<FunctionOpInterface>();
  if (!parentFunction)
    return {};
  return parentFunction.getOperation()->getAttrOfType<TargetFeaturesAttr>(
      getAttributeName());
}
