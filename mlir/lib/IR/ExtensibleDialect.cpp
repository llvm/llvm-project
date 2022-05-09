//===- ExtensibleDialect.cpp - Extensible dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StorageUniquerSupport.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Dynamic types and attributes shared functions
//===----------------------------------------------------------------------===//

/// Default parser for dynamic attribute or type parameters.
/// Parse in the format '(<>)?' or '<attr (,attr)*>'.
static LogicalResult
typeOrAttrParser(AsmParser &parser, SmallVectorImpl<Attribute> &parsedParams) {
  // No parameters
  if (parser.parseOptionalLess() || !parser.parseOptionalGreater())
    return success();

  Attribute attr;
  if (parser.parseAttribute(attr))
    return failure();
  parsedParams.push_back(attr);

  while (parser.parseOptionalGreater()) {
    Attribute attr;
    if (parser.parseComma() || parser.parseAttribute(attr))
      return failure();
    parsedParams.push_back(attr);
  }

  return success();
}

/// Default printer for dynamic attribute or type parameters.
/// Print in the format '(<>)?' or '<attr (,attr)*>'.
static void typeOrAttrPrinter(AsmPrinter &printer, ArrayRef<Attribute> params) {
  if (params.empty())
    return;

  printer << "<";
  interleaveComma(params, printer.getStream());
  printer << ">";
}

//===----------------------------------------------------------------------===//
// Dynamic type
//===----------------------------------------------------------------------===//

std::unique_ptr<DynamicTypeDefinition>
DynamicTypeDefinition::get(StringRef name, ExtensibleDialect *dialect,
                           VerifierFn &&verifier) {
  return DynamicTypeDefinition::get(name, dialect, std::move(verifier),
                                    typeOrAttrParser, typeOrAttrPrinter);
}

std::unique_ptr<DynamicTypeDefinition>
DynamicTypeDefinition::get(StringRef name, ExtensibleDialect *dialect,
                           VerifierFn &&verifier, ParserFn &&parser,
                           PrinterFn &&printer) {
  return std::unique_ptr<DynamicTypeDefinition>(
      new DynamicTypeDefinition(name, dialect, std::move(verifier),
                                std::move(parser), std::move(printer)));
}

DynamicTypeDefinition::DynamicTypeDefinition(StringRef nameRef,
                                             ExtensibleDialect *dialect,
                                             VerifierFn &&verifier,
                                             ParserFn &&parser,
                                             PrinterFn &&printer)
    : name(nameRef), dialect(dialect), verifier(std::move(verifier)),
      parser(std::move(parser)), printer(std::move(printer)),
      ctx(dialect->getContext()) {}

DynamicTypeDefinition::DynamicTypeDefinition(ExtensibleDialect *dialect,
                                             StringRef nameRef)
    : name(nameRef), dialect(dialect), ctx(dialect->getContext()) {}

void DynamicTypeDefinition::registerInTypeUniquer() {
  detail::TypeUniquer::registerType<DynamicType>(&getContext(), getTypeID());
}

namespace mlir {
namespace detail {
/// Storage of DynamicType.
/// Contains a pointer to the type definition and type parameters.
struct DynamicTypeStorage : public TypeStorage {

  using KeyTy = std::pair<DynamicTypeDefinition *, ArrayRef<Attribute>>;

  explicit DynamicTypeStorage(DynamicTypeDefinition *typeDef,
                              ArrayRef<Attribute> params)
      : typeDef(typeDef), params(params) {}

  bool operator==(const KeyTy &key) const {
    return typeDef == key.first && params == key.second;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static DynamicTypeStorage *construct(TypeStorageAllocator &alloc,
                                       const KeyTy &key) {
    return new (alloc.allocate<DynamicTypeStorage>())
        DynamicTypeStorage(key.first, alloc.copyInto(key.second));
  }

  /// Definition of the type.
  DynamicTypeDefinition *typeDef;

  /// The type parameters.
  ArrayRef<Attribute> params;
};
} // namespace detail
} // namespace mlir

DynamicType DynamicType::get(DynamicTypeDefinition *typeDef,
                             ArrayRef<Attribute> params) {
  auto &ctx = typeDef->getContext();
  auto emitError = detail::getDefaultDiagnosticEmitFn(&ctx);
  assert(succeeded(typeDef->verify(emitError, params)));
  return detail::TypeUniquer::getWithTypeID<DynamicType>(
      &ctx, typeDef->getTypeID(), typeDef, params);
}

DynamicType
DynamicType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                        DynamicTypeDefinition *typeDef,
                        ArrayRef<Attribute> params) {
  if (failed(typeDef->verify(emitError, params)))
    return {};
  auto &ctx = typeDef->getContext();
  return detail::TypeUniquer::getWithTypeID<DynamicType>(
      &ctx, typeDef->getTypeID(), typeDef, params);
}

DynamicTypeDefinition *DynamicType::getTypeDef() { return getImpl()->typeDef; }

ArrayRef<Attribute> DynamicType::getParams() { return getImpl()->params; }

bool DynamicType::classof(Type type) {
  return type.hasTrait<TypeTrait::IsDynamicType>();
}

ParseResult DynamicType::parse(AsmParser &parser,
                               DynamicTypeDefinition *typeDef,
                               DynamicType &parsedType) {
  SmallVector<Attribute> params;
  if (failed(typeDef->parser(parser, params)))
    return failure();
  parsedType = parser.getChecked<DynamicType>(typeDef, params);
  if (!parsedType)
    return failure();
  return success();
}

void DynamicType::print(AsmPrinter &printer) {
  printer << getTypeDef()->getName();
  getTypeDef()->printer(printer, getParams());
}

//===----------------------------------------------------------------------===//
// Dynamic attribute
//===----------------------------------------------------------------------===//

std::unique_ptr<DynamicAttrDefinition>
DynamicAttrDefinition::get(StringRef name, ExtensibleDialect *dialect,
                           VerifierFn &&verifier) {
  return DynamicAttrDefinition::get(name, dialect, std::move(verifier),
                                    typeOrAttrParser, typeOrAttrPrinter);
}

std::unique_ptr<DynamicAttrDefinition>
DynamicAttrDefinition::get(StringRef name, ExtensibleDialect *dialect,
                           VerifierFn &&verifier, ParserFn &&parser,
                           PrinterFn &&printer) {
  return std::unique_ptr<DynamicAttrDefinition>(
      new DynamicAttrDefinition(name, dialect, std::move(verifier),
                                std::move(parser), std::move(printer)));
}

DynamicAttrDefinition::DynamicAttrDefinition(StringRef nameRef,
                                             ExtensibleDialect *dialect,
                                             VerifierFn &&verifier,
                                             ParserFn &&parser,
                                             PrinterFn &&printer)
    : name(nameRef), dialect(dialect), verifier(std::move(verifier)),
      parser(std::move(parser)), printer(std::move(printer)),
      ctx(dialect->getContext()) {}

DynamicAttrDefinition::DynamicAttrDefinition(ExtensibleDialect *dialect,
                                             StringRef nameRef)
    : name(nameRef), dialect(dialect), ctx(dialect->getContext()) {}

void DynamicAttrDefinition::registerInAttrUniquer() {
  detail::AttributeUniquer::registerAttribute<DynamicAttr>(&getContext(),
                                                           getTypeID());
}

namespace mlir {
namespace detail {
/// Storage of DynamicAttr.
/// Contains a pointer to the attribute definition and attribute parameters.
struct DynamicAttrStorage : public AttributeStorage {
  using KeyTy = std::pair<DynamicAttrDefinition *, ArrayRef<Attribute>>;

  explicit DynamicAttrStorage(DynamicAttrDefinition *attrDef,
                              ArrayRef<Attribute> params)
      : attrDef(attrDef), params(params) {}

  bool operator==(const KeyTy &key) const {
    return attrDef == key.first && params == key.second;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  static DynamicAttrStorage *construct(AttributeStorageAllocator &alloc,
                                       const KeyTy &key) {
    return new (alloc.allocate<DynamicAttrStorage>())
        DynamicAttrStorage(key.first, alloc.copyInto(key.second));
  }

  /// Definition of the type.
  DynamicAttrDefinition *attrDef;

  /// The type parameters.
  ArrayRef<Attribute> params;
};
} // namespace detail
} // namespace mlir

DynamicAttr DynamicAttr::get(DynamicAttrDefinition *attrDef,
                             ArrayRef<Attribute> params) {
  auto &ctx = attrDef->getContext();
  return detail::AttributeUniquer::getWithTypeID<DynamicAttr>(
      &ctx, attrDef->getTypeID(), attrDef, params);
}

DynamicAttr
DynamicAttr::getChecked(function_ref<InFlightDiagnostic()> emitError,
                        DynamicAttrDefinition *attrDef,
                        ArrayRef<Attribute> params) {
  if (failed(attrDef->verify(emitError, params)))
    return {};
  return get(attrDef, params);
}

DynamicAttrDefinition *DynamicAttr::getAttrDef() { return getImpl()->attrDef; }

ArrayRef<Attribute> DynamicAttr::getParams() { return getImpl()->params; }

bool DynamicAttr::classof(Attribute attr) {
  return attr.hasTrait<AttributeTrait::IsDynamicAttr>();
}

ParseResult DynamicAttr::parse(AsmParser &parser,
                               DynamicAttrDefinition *attrDef,
                               DynamicAttr &parsedAttr) {
  SmallVector<Attribute> params;
  if (failed(attrDef->parser(parser, params)))
    return failure();
  parsedAttr = parser.getChecked<DynamicAttr>(attrDef, params);
  if (!parsedAttr)
    return failure();
  return success();
}

void DynamicAttr::print(AsmPrinter &printer) {
  printer << getAttrDef()->getName();
  getAttrDef()->printer(printer, getParams());
}

//===----------------------------------------------------------------------===//
// Dynamic operation
//===----------------------------------------------------------------------===//

DynamicOpDefinition::DynamicOpDefinition(
    StringRef name, ExtensibleDialect *dialect,
    OperationName::VerifyInvariantsFn &&verifyFn,
    OperationName::VerifyRegionInvariantsFn &&verifyRegionFn,
    OperationName::ParseAssemblyFn &&parseFn,
    OperationName::PrintAssemblyFn &&printFn,
    OperationName::FoldHookFn &&foldHookFn,
    OperationName::GetCanonicalizationPatternsFn
        &&getCanonicalizationPatternsFn)
    : typeID(dialect->allocateTypeID()),
      name((dialect->getNamespace() + "." + name).str()), dialect(dialect),
      verifyFn(std::move(verifyFn)), verifyRegionFn(std::move(verifyRegionFn)),
      parseFn(std::move(parseFn)), printFn(std::move(printFn)),
      foldHookFn(std::move(foldHookFn)),
      getCanonicalizationPatternsFn(std::move(getCanonicalizationPatternsFn)) {}

std::unique_ptr<DynamicOpDefinition> DynamicOpDefinition::get(
    StringRef name, ExtensibleDialect *dialect,
    OperationName::VerifyInvariantsFn &&verifyFn,
    OperationName::VerifyRegionInvariantsFn &&verifyRegionFn) {
  auto parseFn = [](OpAsmParser &parser, OperationState &result) {
    return parser.emitError(
        parser.getCurrentLocation(),
        "dynamic operation do not define any parser function");
  };

  auto printFn = [](Operation *op, OpAsmPrinter &printer, StringRef) {
    printer.printGenericOp(op);
  };

  return DynamicOpDefinition::get(name, dialect, std::move(verifyFn),
                                  std::move(verifyRegionFn), std::move(parseFn),
                                  std::move(printFn));
}

std::unique_ptr<DynamicOpDefinition> DynamicOpDefinition::get(
    StringRef name, ExtensibleDialect *dialect,
    OperationName::VerifyInvariantsFn &&verifyFn,
    OperationName::VerifyRegionInvariantsFn &&verifyRegionFn,
    OperationName::ParseAssemblyFn &&parseFn,
    OperationName::PrintAssemblyFn &&printFn) {
  auto foldHookFn = [](Operation *op, ArrayRef<Attribute> operands,
                       SmallVectorImpl<OpFoldResult> &results) {
    return failure();
  };

  auto getCanonicalizationPatternsFn = [](RewritePatternSet &, MLIRContext *) {
  };

  return DynamicOpDefinition::get(name, dialect, std::move(verifyFn),
                                  std::move(verifyRegionFn), std::move(parseFn),
                                  std::move(printFn), std::move(foldHookFn),
                                  std::move(getCanonicalizationPatternsFn));
}

std::unique_ptr<DynamicOpDefinition>
DynamicOpDefinition::get(StringRef name, ExtensibleDialect *dialect,
                         OperationName::VerifyInvariantsFn &&verifyFn,
                         OperationName::VerifyInvariantsFn &&verifyRegionFn,
                         OperationName::ParseAssemblyFn &&parseFn,
                         OperationName::PrintAssemblyFn &&printFn,
                         OperationName::FoldHookFn &&foldHookFn,
                         OperationName::GetCanonicalizationPatternsFn
                             &&getCanonicalizationPatternsFn) {
  return std::unique_ptr<DynamicOpDefinition>(new DynamicOpDefinition(
      name, dialect, std::move(verifyFn), std::move(verifyRegionFn),
      std::move(parseFn), std::move(printFn), std::move(foldHookFn),
      std::move(getCanonicalizationPatternsFn)));
}

//===----------------------------------------------------------------------===//
// Extensible dialect
//===----------------------------------------------------------------------===//

namespace {
/// Interface that can only be implemented by extensible dialects.
/// The interface is used to check if a dialect is extensible or not.
class IsExtensibleDialect : public DialectInterface::Base<IsExtensibleDialect> {
public:
  IsExtensibleDialect(Dialect *dialect) : Base(dialect) {}

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IsExtensibleDialect)
};
} // namespace

ExtensibleDialect::ExtensibleDialect(StringRef name, MLIRContext *ctx,
                                     TypeID typeID)
    : Dialect(name, ctx, typeID) {
  addInterfaces<IsExtensibleDialect>();
}

void ExtensibleDialect::registerDynamicType(
    std::unique_ptr<DynamicTypeDefinition> &&type) {
  DynamicTypeDefinition *typePtr = type.get();
  TypeID typeID = type->getTypeID();
  StringRef name = type->getName();
  ExtensibleDialect *dialect = type->getDialect();

  assert(dialect == this &&
         "trying to register a dynamic type in the wrong dialect");

  // If a type with the same name is already defined, fail.
  auto registered = dynTypes.try_emplace(typeID, std::move(type)).second;
  (void)registered;
  assert(registered && "type TypeID was not unique");

  registered = nameToDynTypes.insert({name, typePtr}).second;
  (void)registered;
  assert(registered &&
         "Trying to create a new dynamic type with an existing name");

  auto abstractType =
      AbstractType::get(*dialect, DynamicAttr::getInterfaceMap(),
                        DynamicType::getHasTraitFn(), typeID);

  /// Add the type to the dialect and the type uniquer.
  addType(typeID, std::move(abstractType));
  typePtr->registerInTypeUniquer();
}

void ExtensibleDialect::registerDynamicAttr(
    std::unique_ptr<DynamicAttrDefinition> &&attr) {
  auto *attrPtr = attr.get();
  auto typeID = attr->getTypeID();
  auto name = attr->getName();
  auto *dialect = attr->getDialect();

  assert(dialect == this &&
         "trying to register a dynamic attribute in the wrong dialect");

  // If an attribute with the same name is already defined, fail.
  auto registered = dynAttrs.try_emplace(typeID, std::move(attr)).second;
  (void)registered;
  assert(registered && "attribute TypeID was not unique");

  registered = nameToDynAttrs.insert({name, attrPtr}).second;
  (void)registered;
  assert(registered &&
         "Trying to create a new dynamic attribute with an existing name");

  auto abstractAttr =
      AbstractAttribute::get(*dialect, DynamicAttr::getInterfaceMap(),
                             DynamicAttr::getHasTraitFn(), typeID);

  /// Add the type to the dialect and the type uniquer.
  addAttribute(typeID, std::move(abstractAttr));
  attrPtr->registerInAttrUniquer();
}

void ExtensibleDialect::registerDynamicOp(
    std::unique_ptr<DynamicOpDefinition> &&op) {
  assert(op->dialect == this &&
         "trying to register a dynamic op in the wrong dialect");
  auto hasTraitFn = [](TypeID traitId) { return false; };

  RegisteredOperationName::insert(
      op->name, *op->dialect, op->typeID, std::move(op->parseFn),
      std::move(op->printFn), std::move(op->verifyFn),
      std::move(op->verifyRegionFn), std::move(op->foldHookFn),
      std::move(op->getCanonicalizationPatternsFn),
      detail::InterfaceMap::get<>(), std::move(hasTraitFn), {});
}

bool ExtensibleDialect::classof(const Dialect *dialect) {
  return const_cast<Dialect *>(dialect)
      ->getRegisteredInterface<IsExtensibleDialect>();
}

OptionalParseResult ExtensibleDialect::parseOptionalDynamicType(
    StringRef typeName, AsmParser &parser, Type &resultType) const {
  DynamicTypeDefinition *typeDef = lookupTypeDefinition(typeName);
  if (!typeDef)
    return llvm::None;

  DynamicType dynType;
  if (DynamicType::parse(parser, typeDef, dynType))
    return failure();
  resultType = dynType;
  return success();
}

LogicalResult ExtensibleDialect::printIfDynamicType(Type type,
                                                    AsmPrinter &printer) {
  if (auto dynType = type.dyn_cast<DynamicType>()) {
    dynType.print(printer);
    return success();
  }
  return failure();
}

OptionalParseResult ExtensibleDialect::parseOptionalDynamicAttr(
    StringRef attrName, AsmParser &parser, Attribute &resultAttr) const {
  DynamicAttrDefinition *attrDef = lookupAttrDefinition(attrName);
  if (!attrDef)
    return llvm::None;

  DynamicAttr dynAttr;
  if (DynamicAttr::parse(parser, attrDef, dynAttr))
    return failure();
  resultAttr = dynAttr;
  return success();
}

LogicalResult ExtensibleDialect::printIfDynamicAttr(Attribute attribute,
                                                    AsmPrinter &printer) {
  if (auto dynAttr = attribute.dyn_cast<DynamicAttr>()) {
    dynAttr.print(printer);
    return success();
  }
  return failure();
}
