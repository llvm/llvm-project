//===- RegisterEverything.cpp - API to register all dialects/passes -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/RegisterEverything.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

PYBIND11_MODULE(_mlirRegisterEverything, m) {
  m.doc() = "MLIR All Upstream Dialects, Translations and Passes Registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);
  });
  m.def("register_llvm_translations",
        [](MlirContext context) { mlirRegisterAllLLVMTranslations(context); });

  // Register all passes on load.
  mlirRegisterAllPasses();
}
