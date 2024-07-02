//===-- PtrAttrDefs.td - Ptr Attributes definition file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PTR_ATTRDEFS
#define PTR_ATTRDEFS

include "mlir/Dialect/Ptr/IR/PtrDialect.td"
include "mlir/IR/AttrTypeBase.td"

// All of the attributes will extend this class.
class Ptr_Attr<string name, string attrMnemonic,
                list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<Ptr_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// SpecAttr
//===----------------------------------------------------------------------===//

def Ptr_SpecAttr : Ptr_Attr<"Spec", "spec"> {
  let summary = "ptr data layout spec";
  let description = [{
    Defines the data layout spec for a pointer type. This attribute has 4
    fields:
     - [Required] size: size of the pointer in bits.
     - [Required] abi: ABI-required alignment for the pointer in bits.
     - [Required] preferred: preferred alignment for the pointer in bits.
     - [Optional] index: bitwidth that should be used when performing index
     computations for the type. Setting the field to `kOptionalSpecValue`, means
     the field is optional.
    
    Furthermore, the attribute will verify that all present values are divisible
    by 8 (number of bits in a byte), and that `preferred` > `abi`.

    Example:
    ```mlir
    // Spec for a 64 bit ptr, with a required alignment of 64 bits, but with
    // a preferred alignment of 128 bits and an index bitwidth of 64 bits.
    #ptr.spec<size = 64, abi = 64, preferred = 128, index = 64>
    ```
  }];
  let parameters = (ins
    "uint32_t":$size,
    "uint32_t":$abi,
    "uint32_t":$preferred,
    DefaultValuedParameter<"uint32_t", "kOptionalSpecValue">:$index
  );
  let skipDefaultBuilders = 1;
  let builders = [
    AttrBuilder<(ins "uint32_t":$size, "uint32_t":$abi, "uint32_t":$preferred,
                     CArg<"uint32_t", "kOptionalSpecValue">:$index), [{
      return $_get($_ctxt, size, abi, preferred, index);
    }]>
  ];
  let assemblyFormat = "`<` struct(params) `>`";
  let extraClassDeclaration = [{
    /// Constant for specifying a spec entry is optional.
    static constexpr uint32_t kOptionalSpecValue = std::numeric_limits<uint32_t>::max();
  }];
  let genVerifyDecl = 1;
}

#endif // PTR_ATTRDEFS
