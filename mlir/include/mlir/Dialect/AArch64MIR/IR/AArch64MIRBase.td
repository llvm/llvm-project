//===- AArch64MIRBase.td - AArch64 MIR dialect base --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The `aarch64_mir` dialect models AArch64 selected (post-instruction-select,
// pre-register-allocation) MachineIR. It is closely coupled to the `mir`
// dialect: its ops reuse the mir LLT type system (MIR_AnyLLT) and live in the
// same mir.func container, intermixing with generic (mir.g_*) ops in a
// partially-selected function.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AARCH64MIR_IR_AARCH64MIRBASE_TD
#define MLIR_DIALECT_AARCH64MIR_IR_AARCH64MIRBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/Dialect/MIR/IR/MIRTypes.td"

def AArch64MIR_Dialect : Dialect {
  let name = "aarch64_mir";
  let cppNamespace = "::mlir::aarch64_mir";
  let summary = "AArch64 selected, pre-register-allocation MachineIR";
  let description = [{
    Operations for AArch64 selected MachineIR, generated from AArch64's
    instruction TableGen by the `-gen-mir-target-ops` backend. The many
    per-encoding / per-operand-kind record variants of a mnemonic are collapsed
    into a single op; the concrete target opcode is recovered at export time
    from the operand types plus an optional `variant` discriminator.
  }];

  let dependentDialects = ["::mlir::mir::MIRDialect"];
}

// Base class for generated AArch64 target ops. Uses the same uniform assembly
// format as the generic MIR ops.
class AArch64MIR_TargetOp<string mnemonic, list<Trait> traits = []>
    : Op<AArch64MIR_Dialect, mnemonic, traits> {
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

#endif // MLIR_DIALECT_AARCH64MIR_IR_AARCH64MIRBASE_TD
