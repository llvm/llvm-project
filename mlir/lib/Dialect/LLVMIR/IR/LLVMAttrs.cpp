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
// TBAANodeAttr
//===----------------------------------------------------------------------===//

bool TBAANodeAttr::classof(Attribute attr) {
  return llvm::isa<TBAATypeDescriptorAttr, TBAARootAttr>(attr);
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
