//===- MainModule.cpp - Main pybind module --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <tuple>

#include <pybind11/pybind11.h>

#include "mlir/IR/MLIRContext.h"

using namespace mlir;

PYBIND11_MODULE(_mlir, m) {
  m.doc() = "MLIR Python Native Extension";

  m.def("get_test_value", []() {
    // This is just calling a method on the MLIRContext as a smoketest
    // for linkage.
    MLIRContext context;
    return std::make_tuple(std::string("From the native module"),
                           context.isMultithreadingEnabled());
  });
}
