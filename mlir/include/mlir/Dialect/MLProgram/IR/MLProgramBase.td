//===- MLProgramBase.td - Base defs for ml_program dialect --*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLPROGRAM_BASE
#define MLPROGRAM_BASE

include "mlir/IR/OpBase.td"

def MLProgram_Dialect : Dialect {
  let name = "ml_program";
  let cppNamespace = "::mlir::ml_program";
  let description = [{
    The MLProgram dialect contains structural operations and types for
    defining a compiled Machine-Learning program, as created from common
    ML frameworks, such as TensorFlow, PyTorch, JAX, etc. It does not itself
    define computation ops common to such frameworks but establishes a common
    programming model for establishing modules, functions, globals and
    memory model components appropriate for such an abstract level of detail.

    This dialect is under active development, and while stability is an
    eventual goal, it is not guaranteed at this juncture. Given the early state,
    it is recommended to inquire further prior to using this dialect.
  }];

  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

#endif // MLPROGRAM_BASE
