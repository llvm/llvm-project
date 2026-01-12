//===- AlignmentAttrInterface.td - Alignment attribute interface -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an interface for operations that expose an optional
// alignment attribute.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_ALIGNMENTATTRINTERFACE_TD
#define MLIR_INTERFACES_ALIGNMENTATTRINTERFACE_TD

include "mlir/IR/OpBase.td"

def AlignmentAttrOpInterface : OpInterface<"AlignmentAttrOpInterface"> {
  let description = [{
    An interface for operations that carry an optional alignment attribute and
    want to expose it as an `llvm::MaybeAlign` helper.
  }];

  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the alignment encoded on the operation as an `llvm::MaybeAlign`.
        Operations providing a differently named accessor can override the
        default implementation.
      }],
      "::llvm::MaybeAlign",
      "getMaybeAlign",
      (ins),
      [{
        // Defensive: trait implementations are expected to validate power-of-two
        // alignments, but we still guard against accidental misuse.
        auto alignmentOpt = $_op.getAlignment();
        if (!alignmentOpt || *alignmentOpt <= 0)
          return ::llvm::MaybeAlign();
        uint64_t value = static_cast<uint64_t>(*alignmentOpt);
        if (!::llvm::isPowerOf2_64(value))
          return ::llvm::MaybeAlign();
        return ::llvm::MaybeAlign(value);
      }]
    >
  ];

  let extraTraitClassDeclaration = [{
    ::llvm::MaybeAlign getMaybeAlign() {
      // Defensive: trait implementations are expected to validate power-of-two
      // alignments, but we still guard against accidental misuse.
      auto alignmentOpt = (*static_cast<ConcreteOp *>(this)).getAlignment();
      if (!alignmentOpt || *alignmentOpt <= 0)
        return ::llvm::MaybeAlign();
      uint64_t value = static_cast<uint64_t>(*alignmentOpt);
      if (!::llvm::isPowerOf2_64(value))
        return ::llvm::MaybeAlign();
      return ::llvm::MaybeAlign(value);
    }
  }];
}

#endif // MLIR_INTERFACES_ALIGNMENTATTRINTERFACE_TD
