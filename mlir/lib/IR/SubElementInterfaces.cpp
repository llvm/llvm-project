//===- SubElementInterfaces.cpp - Attr and Type SubElement Interfaces -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/SubElementInterfaces.h"

using namespace mlir;

template <typename InterfaceT>
static void walkSubElementsImpl(InterfaceT interface,
                                function_ref<void(Attribute)> walkAttrsFn,
                                function_ref<void(Type)> walkTypesFn) {
  interface.walkImmediateSubElements(
      [&](Attribute attr) {
        // Guard against potentially null inputs. This removes the need for the
        // derived attribute/type to do it.
        if (!attr)
          return;

        // Walk any sub elements first.
        if (auto interface = attr.dyn_cast<SubElementAttrInterface>())
          walkSubElementsImpl(interface, walkAttrsFn, walkTypesFn);

        // Walk this attribute.
        walkAttrsFn(attr);
      },
      [&](Type type) {
        // Guard against potentially null inputs. This removes the need for the
        // derived attribute/type to do it.
        if (!type)
          return;

        // Walk any sub elements first.
        if (auto interface = type.dyn_cast<SubElementTypeInterface>())
          walkSubElementsImpl(interface, walkAttrsFn, walkTypesFn);

        // Walk this type.
        walkTypesFn(type);
      });
}

void SubElementAttrInterface::walkSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) {
  assert(walkAttrsFn && walkTypesFn && "expected valid walk functions");
  walkSubElementsImpl(*this, walkAttrsFn, walkTypesFn);
}

void SubElementTypeInterface::walkSubElements(
    function_ref<void(Attribute)> walkAttrsFn,
    function_ref<void(Type)> walkTypesFn) {
  assert(walkAttrsFn && walkTypesFn && "expected valid walk functions");
  walkSubElementsImpl(*this, walkAttrsFn, walkTypesFn);
}

//===----------------------------------------------------------------------===//
// SubElementInterface Tablegen definitions
//===----------------------------------------------------------------------===//

#include "mlir/IR/SubElementAttrInterfaces.cpp.inc"
#include "mlir/IR/SubElementTypeInterfaces.cpp.inc"
