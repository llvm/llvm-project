//===-- passes.td - PassGen test definition file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/Pass/PassBase.td"
include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def TestPass : Pass<"test"> {
  let summary = "Test pass";
}

def TestPassWithOptions : Pass<"test"> {
  let summary = "Test pass with options";

  let options = [
    Option<"testOption", "testOption", "int", "0", "Test option">,
    ListOption<"testListOption", "test-list-option", "int64_t",
               "Test list option">
  ];
}

def TestPassWithCustomConstructor : Pass<"test"> {
  let summary = "Test pass with custom constructor";

  let constructor = "::createTestPassWithCustomConstructor()";
}
