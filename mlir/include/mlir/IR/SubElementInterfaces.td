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
           "llvm::function_ref<void(mlir::Type)>":$walkTypesFn),
      /*methodBody=*/[{}], /*defaultImplementation=*/[{
        ::mlir::detail::walkImmediateSubElementsImpl(
          }] # derivedValue # [{, walkAttrsFn, walkTypesFn);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Replace the immediately nested sub-attributes and sub-types with those provided.
        The order of the provided elements is derived from the order of the elements
        returned by the callbacks of `walkImmediateSubElements`. The element at index 0
        would replace the very first attribute given by `walkImmediateSubElements`.
        On success, the new instance with the values replaced is returned. If replacement
        fails, nullptr is returned.

        Note that replacing the sub-elements of mutable types or attributes is
        not currently supported by the interface. If an implementing type or
        attribute is mutable, it should return `nullptr` if it has no mechanism
        for replacing sub elements.
      }], attrOrType, "replaceImmediateSubElements",
      (ins "::llvm::ArrayRef<::mlir::Attribute>":$replAttrs,
           "::llvm::ArrayRef<::mlir::Type>":$replTypes),
      /*methodBody=*/[{}], /*defaultImplementation=*/[{
        return ::mlir::detail::replaceImmediateSubElementsImpl(
           }] # derivedValue # [{, replAttrs, replTypes);
      }]>,
  ];

  code extraClassDeclaration = [{
    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn);

    /// Recursively replace all of the nested sub-attributes and sub-types using the
    /// provided map functions. Returns nullptr in the case of failure. See
    /// `AttrTypeReplacer` for information on the support replacement function types.
    template <typename... ReplacementFns>
    }] # attrOrType # [{ replaceSubElements(ReplacementFns &&... replacementFns) {
      AttrTypeReplacer replacer;
      (replacer.addReplacement(std::forward<ReplacementFns>(replacementFns)), ...);
      return replacer.replace(*this);
    }
  }];
  code extraTraitClassDeclaration = [{
    /// Walk all of the held sub-attributes and sub-types.
    void walkSubElements(llvm::function_ref<void(mlir::Attribute)> walkAttrsFn,
                         llvm::function_ref<void(mlir::Type)> walkTypesFn) {
      }] # interfaceName # " interface(" # derivedValue # [{);
      interface.walkSubElements(walkAttrsFn, walkTypesFn);
    }

    /// Recursively replace all of the nested sub-attributes and sub-types using the
    /// provided map functions. Returns nullptr in the case of failure. See
    /// `AttrTypeReplacer` for information on the support replacement function types.
    template <typename... ReplacementFns>
    }] # attrOrType # [{ replaceSubElements(ReplacementFns &&... replacementFns) {
      AttrTypeReplacer replacer;
      (replacer.addReplacement(std::forward<ReplacementFns>(replacementFns)), ...);
      return replacer.replace(}] # derivedValue # [{);
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

    To support the introspection of custom parameters that hold sub-elements,
    a specialization of the `AttrTypeSubElementHandler` class must be provided.
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

    To support the introspection of custom parameters that hold sub-elements,
    a specialization of the `AttrTypeSubElementHandler` class must be provided.
  }];
}

#endif // MLIR_IR_SUBELEMENTINTERFACES_TD_
