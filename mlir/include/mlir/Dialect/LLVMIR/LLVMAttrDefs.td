//===-- LLVMAttrDefs.td - LLVM Attributes definition file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_ATTRDEFS
#define LLVMIR_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

// All of the attributes will extend this class.
class LLVM_Attr<string name> : AttrDef<LLVM_Dialect, name>;

// Attribute definition for the LLVM Linkage enum.
def LinkageAttr : LLVM_Attr<"Linkage"> {
  let mnemonic = "linkage";
  let parameters = (ins
    "linkage::Linkage":$linkage
  );
  let hasCustomAssemblyFormat = 1;
}

// Attribute definition for the LLVM Linkage enum.
def CConvAttr : LLVM_Attr<"CConv"> {
  let mnemonic = "cconv";
  let parameters = (ins
    "CConv":$CallingConv
  );
  let hasCustomAssemblyFormat = 1;
}

def LoopOptionsAttr : LLVM_Attr<"LoopOptions"> {
  let mnemonic = "loopopts";

  let description = [{
    This attributes encapsulates "loop options". It is means to decorate
    branches that are "latches" (loop backedges) and maps to the `!llvm.loop`
    metadatas: https://llvm.org/docs/LangRef.html#llvm-loop
    It store the options as a pair <enum,int64_t> in a sorted array and expose
    APIs to retrieve the value for each option with a stronger type (bool for
    example).
  }];

  // List of type parameters.
  let parameters = (ins
    ArrayRefParameter<"std::pair<LoopOptionCase, int64_t>", "">:$options
  );

  let extraClassDeclaration = [{
    using OptionValuePair = std::pair<LoopOptionCase, int64_t>;
    using OptionsArray = ArrayRef<std::pair<LoopOptionCase, int64_t>>;
    Optional<bool> disableUnroll();
    Optional<bool> disableLICM();
    Optional<int64_t> interleaveCount();
  }];

  let builders = [
    /// Build the LoopOptions Attribute from a sorted array of individual options.
    AttrBuilder<(ins "ArrayRef<std::pair<LoopOptionCase, int64_t>>":$sortedOptions)>,
    AttrBuilder<(ins "LoopOptionsAttrBuilder &":$optionBuilders)>
  ];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

#endif // LLVMIR_ATTRDEFS
