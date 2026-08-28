//===- MIROps.td - MIR dialect operations ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Operations for the MIR dialect. This file currently hand-defines a
// representative subset of the generic (G_*) opcodes plus the structural and
// ABI-boundary operations. The full set of ~279 generic opcodes will be
// generated from `llvm/include/llvm/Target/GenericOpcodes.td` by the
// `-gen-mir-generic-ops` mlir-tblgen backend and included here.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIR_IR_MIROPS_TD
#define MLIR_DIALECT_MIR_IR_MIROPS_TD

include "mlir/Dialect/MIR/IR/MIRBase.td"
include "mlir/Dialect/MIR/IR/MIRTypes.td"
include "mlir/Dialect/MIR/IR/MIRAttrs.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// FuncOp : a MachineFunction container
//===----------------------------------------------------------------------===//

def MIR_FuncOp : MIR_Op<"func", [IsolatedFromAbove, NoTerminator]> {
  let summary = "A MachineFunction";
  let description = [{
    Container for a machine function body. Its single region is an SSACFG region
    whose blocks correspond to MachineBasicBlocks; block successors are expressed
    through terminator operations.
  }];
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";
}

//===----------------------------------------------------------------------===//
// Generic (G_*) opcodes
//
// The full set of generic opcodes is generated from
// llvm/include/llvm/Target/GenericOpcodes.td by the `-gen-mir-generic-ops`
// mlir-tblgen backend and checked in as MIRGenericOps.td.
//===----------------------------------------------------------------------===//

include "mlir/Dialect/MIR/IR/MIRGenericOps.td"

//===----------------------------------------------------------------------===//
// COPY and ABI-boundary physical-register copies
//
// COPY is a shared TargetOpcode (not a GenericInstruction), and the physical
// register copies are specific to this dialect's SSA modeling, so these are
// defined by hand rather than generated.
//===----------------------------------------------------------------------===//

def MIR_CopyOp : MIR_Op<"copy", [Pure, SameOperandsAndResultType]> {
  let summary = "COPY between virtual registers";
  let arguments = (ins MIR_AnyLLT:$src);
  let results = (outs MIR_AnyLLT:$dst);
  let assemblyFormat = "$src attr-dict `:` type($dst)";
}

def MIR_CopyFromPhysOp : MIR_Op<"copy_from_phys"> {
  let summary = "Define a virtual register from a physical register (ABI in)";
  let description = [{
    Materializes a virtual register from an incoming physical register at an ABI
    boundary (function argument or call result). The physical register is an
    attribute, not an operand, so the value graph stays SSA.
  }];
  let arguments = (ins MIR_PhysRegAttr:$reg);
  let results = (outs MIR_AnyLLT:$dst);
  let assemblyFormat = "qualified($reg) attr-dict `->` type($dst)";
}

def MIR_CopyToPhysOp : MIR_Op<"copy_to_phys"> {
  let summary = "Copy a virtual register to a physical register (ABI out)";
  let arguments = (ins MIR_AnyLLT:$src, MIR_PhysRegAttr:$reg);
  let assemblyFormat = "$src `:` type($src) `to` qualified($reg) attr-dict";
}

#endif // MLIR_DIALECT_MIR_IR_MIROPS_TD
