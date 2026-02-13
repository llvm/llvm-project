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
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OperationSupport.h"

using namespace mlir;

DEFINE_C_API_PTR_METHODS(MlirDynamicOpTrait, DynamicOpTrait)

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

MlirDynamicOpTrait mlirDynamicOpTraitNoTerminatorCreate() {
  return wrap(new DynamicOpTraits::NoTerminator());
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
