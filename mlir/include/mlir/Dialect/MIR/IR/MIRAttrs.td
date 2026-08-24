//===- MIRAttrs.td - MIR dialect attributes ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Attributes carrying MachineIR metadata that is not naturally an SSA operand:
// register banks, physical registers, memory operands, and MI flags.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIR_IR_MIRATTRS_TD
#define MLIR_DIALECT_MIR_IR_MIRATTRS_TD

include "mlir/Dialect/MIR/IR/MIRBase.td"

//===----------------------------------------------------------------------===//
// RegBankAttr : the register bank assigned by RegBankSelect
//===----------------------------------------------------------------------===//

def MIR_RegBankAttr : MIR_Attr<"RegBank", "regbank"> {
  let summary = "Register bank of a generic virtual register";
  let description = [{
    The register bank a generic virtual register was assigned to by
    RegBankSelect, e.g. `#mir.regbank<"GPR">`. Carried as an optional attribute
    because it is only present after regbankselect.
  }];
  let parameters = (ins StringRefParameter<"bank name">:$name);
  let assemblyFormat = "`<` $name `>`";
}

//===----------------------------------------------------------------------===//
// PhysRegAttr : a named physical register
//===----------------------------------------------------------------------===//

def MIR_PhysRegAttr : MIR_Attr<"PhysReg", "physreg"> {
  let summary = "A named physical register";
  let description = [{
    A physical register named by the target, e.g. `#mir.physreg<"x0">`. Used
    only on the ABI-boundary copy operations; physical registers are never
    modeled as SSA values.
  }];
  let parameters = (ins StringRefParameter<"physical register name">:$name);
  let assemblyFormat = "`<` $name `>`";
}

//===----------------------------------------------------------------------===//
// MMOAttr : a MachineMemOperand
//===----------------------------------------------------------------------===//

def MIR_MMOAttr : MIR_Attr<"MMO", "mmo"> {
  let summary = "A MachineMemOperand";
  let description = [{
    Describes a memory access attached to a memory-touching operation: the
    access size in bytes, alignment in bytes, and whether it is volatile, e.g.
    `#mir.mmo<size = 4, align = 4>`.
  }];
  let parameters = (ins
    "uint64_t":$size,
    "uint64_t":$alignment,
    DefaultValuedParameter<"bool", "false">:$isVolatile);
  let assemblyFormat =
      "`<` `size` `=` $size `,` `align` `=` $alignment"
      "(`,` `volatile` $isVolatile^)? `>`";
}

#endif // MLIR_DIALECT_MIR_IR_MIRATTRS_TD
