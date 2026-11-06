//===- OpenACC.cpp - C Interface for OpenACC dialect ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/OpenACC.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/OpenACC/Extensions/AllExtensions.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"

using namespace mlir;
using namespace mlir::acc;

namespace {
// Keep the C API's complete registration behavior. Frontends that only need
// OpenACC IR can link MLIROpenACCDialect directly instead.
void registerOpenACC(DialectRegistry &registry) {
  registry.insert<OpenACCDialect, gpu::GPUDialect>();
  acc::registerAllExtensions(registry);
  registry.addExtension(+[](MLIRContext *context, OpenACCDialect *) {
    context->getOrLoadDialect<gpu::GPUDialect>();
  });
}

void insertOpenACC(MlirDialectRegistry registry) {
  registerOpenACC(*unwrap(registry));
}

MlirDialect loadOpenACC(MlirContext context) {
  DialectRegistry registry;
  registerOpenACC(registry);
  unwrap(context)->appendDialectRegistry(registry);
  return wrap(unwrap(context)->getOrLoadDialect<OpenACCDialect>());
}

MlirStringRef getNamespace() {
  return wrap(OpenACCDialect::getDialectNamespace());
}
} // namespace

MlirDialectHandle mlirGetDialectHandle__acc__() {
  static MlirDialectRegistrationHooks hooks = {insertOpenACC, loadOpenACC,
                                               getNamespace};
  return MlirDialectHandle{&hooks};
}
