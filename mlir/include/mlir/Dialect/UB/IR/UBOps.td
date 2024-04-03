//===- UBOps.td - UB operations definitions ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UB_IR_UBOPS_TD
#define MLIR_DIALECT_UB_IR_UBOPS_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

include "UBOpsInterfaces.td"

def UB_Dialect : Dialect {
  let name = "ub";
  let cppNamespace = "::mlir::ub";

  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
}

// Base class for UB dialect attributes.
class UB_Attr<string name, string attrMnemonic, list<Trait> traits = []> :
    AttrDef<UB_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

// Base class for UB dialect ops.
class UB_Op<string mnemonic, list<Trait> traits = []> :
    Op<UB_Dialect, mnemonic, traits>;

def PoisonAttr : UB_Attr<"Poison", "poison", [PoisonAttrInterface]> {
}

//===----------------------------------------------------------------------===//
// PoisonOp
//===----------------------------------------------------------------------===//

def PoisonOp : UB_Op<"poison", [ConstantLike, Pure]> {
  let summary = "Poisoned constant operation.";
  let description = [{
    The `poison` operation materializes a compile-time poisoned constant value
    to indicate deferred undefined behavior.
    `value` attribute is needed to indicate an optional additional poison
    semantics (e.g. partially poisoned vectors), default value indicates results
    is fully poisoned.

    Examples:

    ```
    // Short form
    %0 = ub.poison : i32
    // Long form
    %1 = ub.poison <#custom_poison_elements_attr> : vector<4xi64>
    ```
  }];

  let arguments = (ins DefaultValuedAttr<PoisonAttrInterface, "{}">:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict (`<` $value^ `>`)? `:` type($result)";

  let hasFolder = 1;
}

#endif // MLIR_DIALECT_UB_IR_UBOPS_TD
