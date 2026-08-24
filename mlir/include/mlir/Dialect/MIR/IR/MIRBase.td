//===- MIRBase.td - MIR dialect base definitions -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MIR dialect, a faithful, SSA-form MLIR representation
// of pre-register-allocation LLVM MachineIR (the GlobalISel generic opcode
// layer). Shared types and attributes defined here are reused by target
// dialects such as `aarch64_mir`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIR_IR_MIRBASE_TD
#define MLIR_DIALECT_MIR_IR_MIRBASE_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// MIR dialect definition
//===----------------------------------------------------------------------===//

def MIR_Dialect : Dialect {
  let name = "mir";
  let cppNamespace = "::mlir::mir";
  let summary = "Pre-register-allocation LLVM MachineIR, in SSA form";
  let description = [{
    The `mir` dialect models the target-independent (GlobalISel generic opcode)
    layer of LLVM MachineIR at a point in the pipeline before register
    allocation, where virtual registers are still in SSA form. Generic virtual
    registers map onto MLIR SSA values whose type is the register's Low Level
    Type (LLT): a scalar `sN`, a pointer `pN`, or a vector.

    Physical registers only appear pre-RA at ABI boundaries (argument, return,
    and call copies). They are modeled as attributes on dedicated side-effecting
    copy operations rather than as SSA values, keeping the value graph pure SSA.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    /// Register the dialect's types.
    void registerTypes();
    /// Register the dialect's attributes.
    void registerAttributes();
  }];
}

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class MIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<MIR_Dialect, mnemonic, traits>;

// Base class for generated generic (G_*) opcodes. Uses a uniform assembly
// format that works for any operand/result/attribute shape: operands, then the
// attribute dictionary (which carries immediates and MachineIR metadata), then
// the functional type. This avoids per-op custom formats while the generated
// op set is filled in from GenericOpcodes.td.
class MIR_GenericOp<string mnemonic, list<Trait> traits = []>
    : MIR_Op<mnemonic, traits> {
  let assemblyFormat =
      "operands attr-dict `:` functional-type(operands, results)";
}

class MIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MIR_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class MIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<MIR_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // MLIR_DIALECT_MIR_IR_MIRBASE_TD
