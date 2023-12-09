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

#include "AttrDetail.h"

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
// DIExpressionAttr
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

//===----------------------------------------------------------------------===//
// DICompositeTypeAttr
//===----------------------------------------------------------------------===//

DICompositeTypeAttr
DICompositeTypeAttr::get(MLIRContext *context, unsigned tag, StringAttr name,
                         DIFileAttr file, uint32_t line, DIScopeAttr scope,
                         DITypeAttr baseType, DIFlags flags,
                         uint64_t sizeInBits, uint64_t alignInBits,
                         ::llvm::ArrayRef<DINodeAttr> elements) {
  return Base::get(context, tag, name, file, line, scope, baseType, flags,
                   sizeInBits, alignInBits, elements, StringAttr());
}

DICompositeTypeAttr DICompositeTypeAttr::get(
    MLIRContext *context, StringAttr identifier, unsigned tag, StringAttr name,
    DIFileAttr file, uint32_t line, DIScopeAttr scope, DITypeAttr baseType,
    DIFlags flags, uint64_t sizeInBits, uint64_t alignInBits,
    ::llvm::ArrayRef<DINodeAttr> elements) {
  return Base::get(context, tag, name, file, line, scope, baseType, flags,
                   sizeInBits, alignInBits, elements, identifier);
}

unsigned DICompositeTypeAttr::getTag() const { return getImpl()->getTag(); }

StringAttr DICompositeTypeAttr::getName() const { return getImpl()->getName(); }

DIFileAttr DICompositeTypeAttr::getFile() const { return getImpl()->getFile(); }

uint32_t DICompositeTypeAttr::getLine() const { return getImpl()->getLine(); }

DIScopeAttr DICompositeTypeAttr::getScope() const {
  return getImpl()->getScope();
}

DITypeAttr DICompositeTypeAttr::getBaseType() const {
  return getImpl()->getBaseType();
}

DIFlags DICompositeTypeAttr::getFlags() const { return getImpl()->getFlags(); }

uint64_t DICompositeTypeAttr::getSizeInBits() const {
  return getImpl()->getSizeInBits();
}

uint64_t DICompositeTypeAttr::getAlignInBits() const {
  return getImpl()->getAlignInBits();
}

::llvm::ArrayRef<DINodeAttr> DICompositeTypeAttr::getElements() const {
  return getImpl()->getElements();
}

StringAttr DICompositeTypeAttr::getIdentifier() const {
  return getImpl()->getIdentifier();
}

Attribute DICompositeTypeAttr::parse(AsmParser &parser, Type type) {
  FailureOr<AsmParser::CyclicParseReset> cyclicParse;
  FailureOr<unsigned> tag;
  FailureOr<StringAttr> name;
  FailureOr<DIFileAttr> file;
  FailureOr<uint32_t> line;
  FailureOr<DIScopeAttr> scope;
  FailureOr<DITypeAttr> baseType;
  FailureOr<DIFlags> flags;
  FailureOr<uint64_t> sizeInBits;
  FailureOr<uint64_t> alignInBits;
  SmallVector<DINodeAttr> elements;
  StringAttr identifier;
  const Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());

  auto paramParser = [&]() -> LogicalResult {
    StringRef paramKey;
    if (parser.parseKeyword(&paramKey)) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected parameter name.");
    }

    if (parser.parseEqual()) {
      return parser.emitError(parser.getCurrentLocation(),
                              "expected `=` following parameter name.");
    }

    if (failed(tag) && paramKey == "tag") {
      tag = [&]() -> FailureOr<unsigned> {
        StringRef nameKeyword;
        if (parser.parseKeyword(&nameKeyword))
          return failure();
        if (const unsigned value = llvm::dwarf::getTag(nameKeyword))
          return value;
        return parser.emitError(parser.getCurrentLocation())
               << "invalid debug info debug info tag name: " << nameKeyword;
      }();
    } else if (failed(name) && paramKey == "name") {
      name = FieldParser<StringAttr>::parse(parser);
      if (failed(name)) {
        return parser.emitError(parser.getCurrentLocation(),
                                "failed to parse parameter 'name'");
      }
    } else if (failed(file) && paramKey == "file") {
      file = FieldParser<DIFileAttr>::parse(parser);
      if (failed(file)) {
        return parser.emitError(parser.getCurrentLocation(),
                                "failed to parse parameter 'file'");
      }
    } else if (failed(line) && paramKey == "line") {
      line = FieldParser<uint32_t>::parse(parser);
      if (failed(line)) {
        return parser.emitError(parser.getCurrentLocation(),
                                "failed to parse parameter 'line'");
      }
    } else if (failed(scope) && paramKey == "scope") {
      scope = FieldParser<DIScopeAttr>::parse(parser);
      if (failed(scope)) {
        return parser.emitError(parser.getCurrentLocation(),
                                "failed to parse parameter 'scope'");
      }
    } else if (failed(baseType) && paramKey == "baseType") {
      baseType = FieldParser<DITypeAttr>::parse(parser);
      if (failed(baseType)) {
        return parser.emitError(parser.getCurrentLocation(),
                                "failed to parse parameter 'baseType'");
      }
    } else if (failed(flags) && paramKey == "flags") {
      flags = FieldParser<DIFlags>::parse(parser);
      if (failed(flags)) {
        return parser.emitError(parser.getCurrentLocation(),
                                "failed to parse parameter 'flags'");
      }
    } else if (failed(sizeInBits) && paramKey == "sizeInBits") {
      sizeInBits = FieldParser<uint32_t>::parse(parser);
      if (failed(sizeInBits)) {
        return parser.emitError(parser.getCurrentLocation(),
                                "failed to parse parameter 'sizeInBits'");
      }
    } else if (failed(alignInBits) && paramKey == "alignInBits") {
      alignInBits = FieldParser<uint32_t>::parse(parser);
      if (failed(alignInBits)) {
        return parser.emitError(parser.getCurrentLocation(),
                                "failed to parse parameter 'alignInBits'");
      }
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unknown parameter '")
             << paramKey << "'";
    }
    return success();
  };

  // Begin parsing.
  if (parser.parseLess()) {
    parser.emitError(parser.getCurrentLocation(), "expected `<`");
    return {};
  }

  // First, attempt to parse the identifier attribute.
  const OptionalParseResult idResult =
      parser.parseOptionalAttribute(identifier);
  if (idResult.has_value() && succeeded(*idResult)) {
    if (succeeded(parser.parseOptionalGreater())) {
      DICompositeTypeAttr result =
          getIdentified(parser.getContext(), identifier);
      // Cyclic parsing should not initiate with only the identifier. Only
      // nested instances should terminate early.
      if (succeeded(parser.tryStartCyclicParse(result))) {
        parser.emitError(parser.getCurrentLocation(),
                         "Expected identified attribute to contain at least "
                         "one other parameter");
        return {};
      }
      return result;
    }

    if (parser.parseComma()) {
      parser.emitError(parser.getCurrentLocation(), "Expected `,`");
    }
  }

  // Parse immutable parameters.
  if (parser.parseCommaSeparatedList(paramParser)) {
    return {};
  }

  if (identifier) {
    // Create the identified attribute.
    DICompositeTypeAttr result =
        get(parser.getContext(), identifier, tag.value_or(0),
            name.value_or(StringAttr()), file.value_or(DIFileAttr()),
            line.value_or(0), scope.value_or(DIScopeAttr()),
            baseType.value_or(DITypeAttr()), flags.value_or(DIFlags::Zero),
            sizeInBits.value_or(0), alignInBits.value_or(0));

    // Initiate cyclic parsing.
    if (cyclicParse = parser.tryStartCyclicParse(result); failed(cyclicParse)) {
      return {};
    }
  }

  // Parse the elements now.
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseCommaSeparatedList([&]() -> LogicalResult {
          Attribute attr;
          if (parser.parseAttribute(attr)) {
            return parser.emitError(parser.getCurrentLocation(),
                                    "expected attribute");
          }
          elements.push_back(mlir::cast<DINodeAttr>(attr));
          return success();
        })) {
      return {};
    }

    if (parser.parseRParen()) {
      parser.emitError(parser.getCurrentLocation(), "expected `)");
      return {};
    }
  }

  // Expect the attribute to terminate.
  if (parser.parseGreater()) {
    parser.emitError(parser.getCurrentLocation(), "expected `>`");
    return {};
  }

  if (!identifier)
    return get(loc.getContext(), tag.value_or(0), name.value_or(StringAttr()),
               file.value_or(DIFileAttr()), line.value_or(0),
               scope.value_or(DIScopeAttr()), baseType.value_or(DITypeAttr()),
               flags.value_or(DIFlags::Zero), sizeInBits.value_or(0),
               alignInBits.value_or(0), elements);

  // Replace the elements if the attribute is identified.
  DICompositeTypeAttr result = getIdentified(parser.getContext(), identifier);
  result.replaceElements(elements);
  return result;
}

void DICompositeTypeAttr::print(AsmPrinter &printer) const {
  FailureOr<AsmPrinter::CyclicPrintReset> cyclicPrint;
  SmallVector<std::function<void()>> valuePrinters;
  printer << "<";
  if (getImpl()->isIdentified()) {
    cyclicPrint = printer.tryStartCyclicPrint(*this);
    if (failed(cyclicPrint)) {
      printer << getIdentifier() << ">";
      return;
    }
    valuePrinters.push_back([&]() { printer << getIdentifier(); });
  }

  if (getTag() > 0) {
    valuePrinters.push_back(
        [&]() {
          printer << "tag = " << llvm::dwarf::TagString(getTag());
        });
  }

  if (getName()) {
    valuePrinters.push_back([&]() {



      printer << "name = ";
      printer.printStrippedAttrOrType(getName());
    });
  }

  if (getFile()) {
    valuePrinters.push_back([&]() {
      printer << "file = ";
      printer.printStrippedAttrOrType(getFile());
    });
  }

  if (getLine() > 0) {
    valuePrinters.push_back([&]() {
      printer << "line = ";
      printer.printStrippedAttrOrType(getLine());
    });
  }

  if (getScope()) {
    valuePrinters.push_back([&]() {
      printer << "scope = ";
      printer.printStrippedAttrOrType(getScope());
    });
  }

  if (getBaseType()) {
    valuePrinters.push_back([&]() {
      printer << "baseType = ";
      printer.printStrippedAttrOrType(getBaseType());
    });
  }

  if (getFlags() != DIFlags::Zero) {
    valuePrinters.push_back([&]() {
      printer << "flags = ";
      printer.printStrippedAttrOrType(getFlags());
    });
  }

  if (getSizeInBits() > 0) {
    valuePrinters.push_back([&]() {
      printer << "sizeInBits = ";
      printer.printStrippedAttrOrType(getSizeInBits());
    });
  }

  if (getAlignInBits() > 0) {
    valuePrinters.push_back([&]() {
      printer << "alignInBits = ";
      printer.printStrippedAttrOrType(getAlignInBits());
    });
  }
  interleaveComma(valuePrinters, printer,
                  [&](const std::function<void()> &fn) { fn(); });

  if (!getElements().empty()) {
    printer << " (";
    printer.printStrippedAttrOrType(getElements());
    printer << ")";
  }
  printer << ">";
}

DICompositeTypeAttr DICompositeTypeAttr::getIdentified(MLIRContext *context,
                                                       StringAttr identifier) {
  return Base::get(context, 0, StringAttr(), DIFileAttr(), 0, DIScopeAttr(),
                   DITypeAttr(), DIFlags::Zero, 0, 0, ArrayRef<DINodeAttr>(),
                   identifier);
}

void DICompositeTypeAttr::replaceElements(
    const ArrayRef<DINodeAttr> &elements) {
  (void)Base::mutate(elements);
}
