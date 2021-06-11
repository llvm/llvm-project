//===-- SubElementInterfaces.td - Sub-Element Interfaces ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces that can be used to interface with
// sub-elements, e.g. held attributes and types, of a composite attribute or
// type.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_SUBELEMENTINTERFACES_TD_
#define MLIR_IR_SUBELEMENTINTERFACES_TD_

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SubElementInterfaceBase
//===----------------------------------------------------------------------===//

class SubElementInterfaceBase<string interfaceName, string derivedValue> {
  string cppNamespace = "::mlir";

  list<InterfaceMethod> methods = [
    InterfaceMethod<
      /*desc=*/[{
        Walk all of the immediately nested sub-attributes and sub-types. This
        method does not recurse into sub elements.
      }], "void", "walkImmediateSubElements",
      (ins "llvm::function_ref<void(mlir::Attribute)>":$walkAttrsFn,
           "llvm::function_ref<void(mlir::Type)>":$walkTypesFn)
    >,
  ];

  code extraClassDeclaration = [{
    /// Walk all of the held sub-attributes.
    void walkSubAttrs(llvm::function_ref<void(mlir::Attribute)> walkFn) {
      walkSubElements(walkFn, /*walkTypesFn=*/[](mlir::Type) {});
    }

    /// Walk all of the held sub-types.
    void walkSubTypes(llvm::function_ref<void(mlir::Type)> walkFn) {
      walkSubElements(/*walkAttrsFn=*/[](mlir::Attribute) {}, walkFn);
    }

    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn);
  }];

  code extraTraitClassDeclaration = [{
    /// Walk all of the held sub-attributes.
    void walkSubAttrs(llvm::function_ref<void(mlir::Attribute)> walkFn) {
      walkSubElements(walkFn, /*walkTypesFn=*/[](mlir::Type) {});
    }

    /// Walk all of the held sub-types.
    void walkSubTypes(llvm::function_ref<void(mlir::Type)> walkFn) {
      walkSubElements(/*walkAttrsFn=*/[](mlir::Attribute) {}, walkFn);
    }

    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn) {
      }] # interfaceName # " interface(" # derivedValue # [{);
      interface.walkSubElements(walkAttrsFn, walkTypesFn);
    }
  }];
}

//===----------------------------------------------------------------------===//
// SubElementAttrInterface
//===----------------------------------------------------------------------===//

def SubElementAttrInterface
    : AttrInterface<"SubElementAttrInterface">,
      SubElementInterfaceBase<"SubElementAttrInterface", "$_attr"> {
  let description = [{
    An interface used to query and manipulate sub-elements, such as sub-types
    and sub-attributes of a composite attribute.
  }];
}

//===----------------------------------------------------------------------===//
// SubElementTypeInterface
//===----------------------------------------------------------------------===//

def SubElementTypeInterface
    : TypeInterface<"SubElementTypeInterface">,
      SubElementInterfaceBase<"SubElementTypeInterface", "$_type"> {
  let description = [{
    An interface used to query and manipulate sub-elements, such as sub-types
    and sub-attributes of a composite type.
  }];
}

#endif // MLIR_IR_SUBELEMENTINTERFACES_TD_
