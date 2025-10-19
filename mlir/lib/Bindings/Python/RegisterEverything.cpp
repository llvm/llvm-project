//===- RegisterEverything.cpp - API to register all dialects/passes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/RegisterAllExtensions.h"
#include "mlir-c/RegisterAllExternalModels.h"
#include "mlir-c/RegisterAllLLVMTranslations.h"
#include "mlir-c/RegisterAllPasses.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

NB_MODULE(_mlirRegisterEverything, m) {
  m.doc() =
      "MLIR All Upstream Extensions, Translations and Passes Registration";

  m.def("register_external_models", [](MlirDialectRegistry registry) {
    mlirRegisterAllExternalModels(registry);
  });
  m.def("register_extensions", [](MlirDialectRegistry registry) {
    mlirRegisterAllExtensions(registry);
  });
  m.def("register_llvm_translations",
        [](MlirContext context) { mlirRegisterAllLLVMTranslations(context); });

  // Register all passes on load.
  mlirRegisterAllPasses();
}
