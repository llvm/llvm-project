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

class SubElementInterfaceBase<string interfaceName, string attrOrType,
                              string derivedValue> {
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
    InterfaceMethod<
      /*desc=*/[{
        Replace the immediately nested sub-attributes and sub-types with those provided.
        The order of the provided elements is derived from the order of the elements
        returned by the callbacks of `walkImmediateSubElements`. The element at index 0
        would replace the very first attribute given by `walkImmediateSubElements`.
        On success, the new instance with the values replaced is returned. If replacement
        fails, nullptr is returned.
      }], attrOrType, "replaceImmediateSubElements", (ins
        "::llvm::ArrayRef<::mlir::Attribute>":$replAttrs,
        "::llvm::ArrayRef<::mlir::Type>":$replTypes
      )>,
  ];

  code extraClassDeclaration = [{
    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn);

    /// Recursively replace all of the nested sub-attributes and sub-types using the
    /// provided map functions. Returns nullptr in the case of failure.
    }] # attrOrType # [{ replaceSubElements(
      mlir::SubElementReplFn<mlir::Attribute> replaceAttrFn,
      mlir::SubElementReplFn<mlir::Type> replaceTypeFn
    ) {
      return replaceSubElements(
        [&](Attribute attr) { return std::make_pair(replaceAttrFn(attr), WalkResult::advance()); },
        [&](Type type) { return std::make_pair(replaceTypeFn(type), WalkResult::advance()); }
      );
    }
    /// Recursively replace all of the nested sub-attributes and sub-types using the
    /// provided map functions. This variant allows for the map function to return an
    /// additional walk result. Returns nullptr in the case of failure.
    }] # attrOrType # [{ replaceSubElements(
      llvm::function_ref<
        std::pair<mlir::Attribute, mlir::WalkResult>(mlir::Attribute)> replaceAttrFn,
      llvm::function_ref<
        std::pair<mlir::Type, mlir::WalkResult>(mlir::Type)> replaceTypeFn
    );
  }];
  code extraTraitClassDeclaration = [{
    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn) {
      }] # interfaceName # " interface(" # derivedValue # [{);
      interface.walkSubElements(walkAttrsFn, walkTypesFn);
    }

    /// Recursively replace all of the nested sub-attributes and sub-types using the
    /// provided map functions. Returns nullptr in the case of failure.
    }] # attrOrType # [{ replaceSubElements(
      mlir::SubElementReplFn<mlir::Attribute> replaceAttrFn,
      mlir::SubElementReplFn<mlir::Type> replaceTypeFn) {
      }] # interfaceName # " interface(" # derivedValue # [{);
      return interface.replaceSubElements(replaceAttrFn, replaceTypeFn);
    }
    }] # attrOrType # [{ replaceSubElements(
      mlir::SubElementResultReplFn<mlir::Attribute> replaceAttrFn,
      mlir::SubElementResultReplFn<mlir::Type> replaceTypeFn) {
      }] # interfaceName # " interface(" # derivedValue # [{);
      return interface.replaceSubElements(replaceAttrFn, replaceTypeFn);
    }
  }];
  code extraSharedClassDeclaration = [{
    /// Walk all of the held sub-attributes.
    void walkSubAttrs(llvm::function_ref<void(mlir::Attribute)> walkFn) {
      walkSubElements(walkFn, /*walkTypesFn=*/[](mlir::Type) {});
    }
    /// Walk all of the held sub-types.
    void walkSubTypes(llvm::function_ref<void(mlir::Type)> walkFn) {
      walkSubElements(/*walkAttrsFn=*/[](mlir::Attribute) {}, walkFn);
    }
    
    /// Recursively replace all of the nested sub-attributes using the provided
    /// map function. Returns nullptr in the case of failure.
    }] # attrOrType # [{ replaceSubElements(
      mlir::SubElementReplFn<mlir::Attribute> replaceAttrFn) {
      return replaceSubElements(
        replaceAttrFn, [](mlir::Type type) { return type; });
    }
    }] # attrOrType # [{ replaceSubElements(
      mlir::SubElementResultReplFn<mlir::Attribute> replaceAttrFn) {
      return replaceSubElements(
        replaceAttrFn,
        [](mlir::Type type) { return std::make_pair(type, WalkResult::advance()); }
      );
    }
    /// Recursively replace all of the nested sub-types using the provided map
    /// function. Returns nullptr in the case of failure.
    }] # attrOrType # [{ replaceSubElements(
      mlir::SubElementReplFn<mlir::Type> replaceTypeFn) {
      return replaceSubElements(
        [](mlir::Attribute attr) { return attr; }, replaceTypeFn);
    }
    }] # attrOrType # [{ replaceSubElements(
      mlir::SubElementResultReplFn<mlir::Type> replaceTypeFn) {
      return replaceSubElements(
        [](mlir::Attribute attr) { return std::make_pair(attr, WalkResult::advance()); },
        replaceTypeFn
      );
    }
  }];
}

//===----------------------------------------------------------------------===//
// SubElementAttrInterface
//===----------------------------------------------------------------------===//

def SubElementAttrInterface
    : AttrInterface<"SubElementAttrInterface">,
      SubElementInterfaceBase<"SubElementAttrInterface", "::mlir::Attribute",
                              "$_attr"> {
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
      SubElementInterfaceBase<"SubElementTypeInterface", "::mlir::Type",
                              "$_type"> {
  let description = [{
    An interface used to query and manipulate sub-elements, such as sub-types
    and sub-attributes of a composite type.
  }];
}

#endif // MLIR_IR_SUBELEMENTINTERFACES_TD_
