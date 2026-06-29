//===- ExtensibleDialect - C API for MLIR Extensible Dialect --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/ExtensibleDialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"

using namespace mlir;

DEFINE_C_API_PTR_METHODS(MlirDynamicDialect, DynamicDialect)
DEFINE_C_API_PTR_METHODS(MlirDynamicOpDefinition, DynamicOpDefinition)
DEFINE_C_API_PTR_METHODS(MlirDynamicOpTrait, DynamicOpTrait)
DEFINE_C_API_PTR_METHODS(MlirDynamicTypeDefinition, DynamicTypeDefinition)
DEFINE_C_API_PTR_METHODS(MlirDynamicAttrDefinition, DynamicAttrDefinition)

//===----------------------------------------------------------------------===//
// Dynamic dialect creation
//===----------------------------------------------------------------------===//

MlirDynamicDialect mlirDynamicDialectCreate(MlirContext ctx,
                                            MlirStringRef name) {
  DynamicDialect *dialect = unwrap(ctx)->getOrLoadDynamicDialect(
      unwrap(name), [](DynamicDialect *) {});
  return wrap(dialect);
}

MlirDialect mlirDynamicDialectAsDialect(MlirDynamicDialect dialect) {
  return wrap(static_cast<Dialect *>(unwrap(dialect)));
}

//===----------------------------------------------------------------------===//
// Dynamic op definition creation
//===----------------------------------------------------------------------===//

MlirDynamicOpDefinition
mlirDynamicOpDefinitionCreate(MlirDynamicDialect dialect, MlirStringRef name,
                              MlirDynamicOpDefinitionCallbacks callbacks,
                              void *userData) {
  auto cppVerifyFn = [callbacks, userData](Operation *op) -> LogicalResult {
    if (!callbacks.verify)
      return success();
    return unwrap(callbacks.verify(wrap(op), userData));
  };
  auto cppVerifyRegionFn = [callbacks,
                            userData](Operation *op) -> LogicalResult {
    if (!callbacks.verifyRegion)
      return success();
    return unwrap(callbacks.verifyRegion(wrap(op), userData));
  };
  std::unique_ptr<DynamicOpDefinition> opDef = DynamicOpDefinition::get(
      unwrap(name), unwrap(dialect), std::move(cppVerifyFn),
      std::move(cppVerifyRegionFn));
  return wrap(opDef.release());
}

void mlirDynamicOpDefinitionDestroy(MlirDynamicOpDefinition opDef) {
  delete unwrap(opDef);
}

void mlirDynamicDialectRegisterOp(MlirDynamicDialect dialect,
                                  MlirDynamicOpDefinition opDef) {
  unwrap(dialect)->registerDynamicOp(
      std::unique_ptr<DynamicOpDefinition>(unwrap(opDef)));
}

//===----------------------------------------------------------------------===//
// Dynamic type definition creation
//===----------------------------------------------------------------------===//

MlirDynamicTypeDefinition
mlirDynamicTypeDefinitionCreate(MlirDynamicDialect dialect, MlirStringRef name,
                                MlirDynamicTypeDefinitionCallbacks callbacks,
                                void *userData) {
  auto cppVerifyFn = [callbacks,
                      userData](function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<Attribute> params) -> LogicalResult {
    if (!callbacks.verify)
      return success();
    SmallVector<MlirAttribute> cParams;
    cParams.reserve(params.size());
    for (Attribute p : params)
      cParams.push_back(wrap(p));
    return unwrap(callbacks.verify(static_cast<intptr_t>(cParams.size()),
                                   cParams.data(), userData));
  };
  std::unique_ptr<DynamicTypeDefinition> typeDef = DynamicTypeDefinition::get(
      unwrap(name), unwrap(dialect), std::move(cppVerifyFn));
  return wrap(typeDef.release());
}

void mlirDynamicTypeDefinitionDestroy(MlirDynamicTypeDefinition typeDef) {
  delete unwrap(typeDef);
}

void mlirDynamicDialectRegisterType(MlirDynamicDialect dialect,
                                    MlirDynamicTypeDefinition typeDef) {
  unwrap(dialect)->registerDynamicType(
      std::unique_ptr<DynamicTypeDefinition>(unwrap(typeDef)));
}

//===----------------------------------------------------------------------===//
// Dynamic attribute definition creation
//===----------------------------------------------------------------------===//

MlirDynamicAttrDefinition
mlirDynamicAttrDefinitionCreate(MlirDynamicDialect dialect, MlirStringRef name,
                                MlirDynamicAttrDefinitionCallbacks callbacks,
                                void *userData) {
  auto cppVerifyFn = [callbacks,
                      userData](function_ref<InFlightDiagnostic()> emitError,
                                ArrayRef<Attribute> params) -> LogicalResult {
    if (!callbacks.verify)
      return success();
    SmallVector<MlirAttribute> cParams;
    cParams.reserve(params.size());
    for (Attribute p : params)
      cParams.push_back(wrap(p));
    return unwrap(callbacks.verify(static_cast<intptr_t>(cParams.size()),
                                   cParams.data(), userData));
  };
  std::unique_ptr<DynamicAttrDefinition> attrDef = DynamicAttrDefinition::get(
      unwrap(name), unwrap(dialect), std::move(cppVerifyFn));
  return wrap(attrDef.release());
}

void mlirDynamicAttrDefinitionDestroy(MlirDynamicAttrDefinition attrDef) {
  delete unwrap(attrDef);
}

void mlirDynamicDialectRegisterAttr(MlirDynamicDialect dialect,
                                    MlirDynamicAttrDefinition attrDef) {
  unwrap(dialect)->registerDynamicAttr(
      std::unique_ptr<DynamicAttrDefinition>(unwrap(attrDef)));
}

//===----------------------------------------------------------------------===//
// Dynamic op trait APIs
//===----------------------------------------------------------------------===//

bool mlirDynamicOpTraitAttach(MlirDynamicOpTrait dynamicOpTrait,
                              MlirStringRef opName, MlirContext context) {
  std::optional<RegisteredOperationName> opNameFound =
      RegisteredOperationName::lookup(unwrap(opName), unwrap(context));
  assert(opNameFound && "operation name must be registered in the context");

  // The original getImpl() is protected, so we create a small helper struct
  // here.
  struct RegisteredOperationNameWithImpl : RegisteredOperationName {
    Impl *getImpl() { return RegisteredOperationName::getImpl(); }
  };
  OperationName::Impl *impl =
      static_cast<RegisteredOperationNameWithImpl &>(*opNameFound).getImpl();

  std::unique_ptr<DynamicOpTrait> trait(unwrap(dynamicOpTrait));
  // TODO: we should enable llvm-style RTTI for `OperationName::Impl` and check
  // whether the `impl` is a `DynamicOpDefinition` here.
  return static_cast<DynamicOpDefinition *>(impl)->addTrait(std::move(trait));
}

MlirDynamicOpTrait mlirDynamicOpTraitIsTerminatorCreate() {
  return wrap(new DynamicOpTraits::IsTerminator());
}

MlirTypeID mlirDynamicOpTraitIsTerminatorGetTypeID() {
  return wrap(DynamicOpTraits::IsTerminator::getStaticTypeID());
}

MlirDynamicOpTrait mlirDynamicOpTraitNoTerminatorCreate() {
  return wrap(new DynamicOpTraits::NoTerminator());
}

MlirTypeID mlirDynamicOpTraitNoTerminatorGetTypeID() {
  return wrap(DynamicOpTraits::NoTerminator::getStaticTypeID());
}

void mlirDynamicOpTraitDestroy(MlirDynamicOpTrait dynamicOpTrait) {
  delete unwrap(dynamicOpTrait);
}

namespace mlir {

class ExternalDynamicOpTrait : public DynamicOpTrait {
public:
  ExternalDynamicOpTrait(TypeID typeID, MlirDynamicOpTraitCallbacks callbacks,
                         void *userData)
      : typeID(typeID), callbacks(callbacks), userData(userData) {
    if (callbacks.construct)
      callbacks.construct(userData);
  }
  ~ExternalDynamicOpTrait() {
    if (callbacks.destruct)
      callbacks.destruct(userData);
  }

  LogicalResult verifyTrait(Operation *op) const override {
    return unwrap(callbacks.verifyTrait(wrap(op), userData));
  };
  LogicalResult verifyRegionTrait(Operation *op) const override {
    return unwrap(callbacks.verifyRegionTrait(wrap(op), userData));
  };

  TypeID getTypeID() const override { return typeID; };

private:
  TypeID typeID;
  MlirDynamicOpTraitCallbacks callbacks;
  void *userData;
};

} // namespace mlir

MlirDynamicOpTrait mlirDynamicOpTraitCreate(
    MlirTypeID typeID, MlirDynamicOpTraitCallbacks callbacks, void *userData) {
  return wrap(
      new mlir::ExternalDynamicOpTrait(unwrap(typeID), callbacks, userData));
}

bool mlirDialectIsAExtensibleDialect(MlirDialect dialect) {
  return llvm::isa<mlir::ExtensibleDialect>(unwrap(dialect));
}

MlirDynamicTypeDefinition
mlirExtensibleDialectLookupTypeDefinition(MlirDialect dialect,
                                          MlirStringRef typeName) {
  return wrap(llvm::cast<mlir::ExtensibleDialect>(unwrap(dialect))
                  ->lookupTypeDefinition(unwrap(typeName)));
}

bool mlirTypeIsADynamicType(MlirType type) {
  return llvm::isa<mlir::DynamicType>(unwrap(type));
}

MlirTypeID mlirDynamicTypeGetTypeID() {
  return wrap(mlir::DynamicType::getTypeID());
}

MlirType mlirDynamicTypeGet(MlirDynamicTypeDefinition typeDef,
                            MlirAttribute *attrs, intptr_t numAttrs) {
  llvm::SmallVector<mlir::Attribute> attributes;
  attributes.reserve(numAttrs);
  for (intptr_t i = 0; i < numAttrs; ++i)
    attributes.push_back(unwrap(attrs[i]));

  return wrap(mlir::DynamicType::get(unwrap(typeDef), attributes));
}

intptr_t mlirDynamicTypeGetNumParams(MlirType type) {
  return llvm::cast<mlir::DynamicType>(unwrap(type)).getParams().size();
}

MlirAttribute mlirDynamicTypeGetParam(MlirType type, intptr_t index) {
  return wrap(llvm::cast<mlir::DynamicType>(unwrap(type)).getParams()[index]);
}

MlirDynamicTypeDefinition mlirDynamicTypeGetTypeDef(MlirType type) {
  return wrap(llvm::cast<mlir::DynamicType>(unwrap(type)).getTypeDef());
}

MlirTypeID
mlirDynamicTypeDefinitionGetTypeID(MlirDynamicTypeDefinition typeDef) {
  return wrap(unwrap(typeDef)->getTypeID());
}

MlirStringRef
mlirDynamicTypeDefinitionGetName(MlirDynamicTypeDefinition typeDef) {
  return wrap(unwrap(typeDef)->getName());
}

MlirDialect
mlirDynamicTypeDefinitionGetDialect(MlirDynamicTypeDefinition typeDef) {
  return wrap(unwrap(typeDef)->getDialect());
}

MlirDynamicAttrDefinition
mlirExtensibleDialectLookupAttrDefinition(MlirDialect dialect,
                                          MlirStringRef attrName) {
  return wrap(llvm::cast<mlir::ExtensibleDialect>(unwrap(dialect))
                  ->lookupAttrDefinition(unwrap(attrName)));
}

bool mlirAttributeIsADynamicAttr(MlirAttribute attr) {
  return llvm::isa<mlir::DynamicAttr>(unwrap(attr));
}

MlirTypeID mlirDynamicAttrGetTypeID(void) {
  return wrap(mlir::DynamicAttr::getTypeID());
}

MlirAttribute mlirDynamicAttrGet(MlirDynamicAttrDefinition attrDef,
                                 MlirAttribute *attrs, intptr_t numAttrs) {
  llvm::SmallVector<mlir::Attribute> attributes;
  attributes.reserve(numAttrs);
  for (intptr_t i = 0; i < numAttrs; ++i)
    attributes.push_back(unwrap(attrs[i]));

  return wrap(mlir::DynamicAttr::get(unwrap(attrDef), attributes));
}

intptr_t mlirDynamicAttrGetNumParams(MlirAttribute attr) {
  return llvm::cast<mlir::DynamicAttr>(unwrap(attr)).getParams().size();
}

MlirAttribute mlirDynamicAttrGetParam(MlirAttribute attr, intptr_t index) {
  return wrap(llvm::cast<mlir::DynamicAttr>(unwrap(attr)).getParams()[index]);
}

MlirDynamicAttrDefinition mlirDynamicAttrGetAttrDef(MlirAttribute attr) {
  return wrap(llvm::cast<mlir::DynamicAttr>(unwrap(attr)).getAttrDef());
}

MlirTypeID
mlirDynamicAttrDefinitionGetTypeID(MlirDynamicAttrDefinition attrDef) {
  return wrap(unwrap(attrDef)->getTypeID());
}

MlirStringRef
mlirDynamicAttrDefinitionGetName(MlirDynamicAttrDefinition attrDef) {
  return wrap(unwrap(attrDef)->getName());
}

MlirDialect
mlirDynamicAttrDefinitionGetDialect(MlirDynamicAttrDefinition attrDef) {
  return wrap(unwrap(attrDef)->getDialect());
}
