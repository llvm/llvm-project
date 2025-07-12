//===-- LLVMAttrDefs.td - LLVM Attributes definition file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_TARGETFEATURESATTRDEFS
#define LLVMIR_TARGETFEATURESATTRDEFS

include "mlir/Dialect/LLVMIR/LLVMDialect.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"

//===----------------------------------------------------------------------===//
// TargetFeaturesAttr
//===----------------------------------------------------------------------===//

def LLVM_TargetFeaturesAttr : LLVM_Attr<"TargetFeatures", "target_features",
    [DLTIQueryInterface]> {
  let summary = "LLVM target features attribute";

  let description = [{
    Represents the LLVM target features as a list that can be checked within
    passes/rewrites.

    Example:
    ```mlir
    #llvm.target_features<["+sme", "+sve", "+sme-f64f64"]>
    ```

    Then within a pass or rewrite the features active at an op can be queried:

    ```c++
    auto targetFeatures = LLVM::TargetFeaturesAttr::featuresAt(op);

    if (!targetFeatures.contains("+sme-f64f64"))
      return failure();
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"StringAttr">:$features);

  let builders = [
    TypeBuilder<(ins "::llvm::StringRef":$features)>,
    TypeBuilder<(ins "::llvm::ArrayRef<::llvm::StringRef>":$features)>
  ];

  let extraClassDeclaration = [{
    /// Checks if a feature is contained within the features list.
    /// Note: Using a StringAttr allows doing pointer-comparisons.
    bool contains(::mlir::StringAttr feature) const;
    bool contains(::llvm::StringRef feature) const;

    bool nullOrEmpty() const {
      // Checks if this attribute is null, or the features are empty.
      return !bool(*this) || getFeatures().empty();
    }

    /// Returns the list of features as an LLVM-compatible string.
    std::string getFeaturesString() const;

    /// Finds the target features on the parent FunctionOpInterface.
    /// Note: This assumes the attribute name matches the return value of
    /// `getAttributeName()`.
    static TargetFeaturesAttr featuresAt(Operation* op);

    /// Canonical name for this attribute within MLIR.
    static constexpr StringLiteral getAttributeName() {
      return StringLiteral("target_features");
    }

    /// Returns the attribute associated with the key.
    FailureOr<Attribute> query(DataLayoutEntryKey key);
  }];

  let assemblyFormat = "`<` `[` (`]`) : ($features^ `]`)? `>`";
  let genVerifyDecl = 1;
}

#endif // LLVMIR_TARGETFEATURESATTRDEFS
