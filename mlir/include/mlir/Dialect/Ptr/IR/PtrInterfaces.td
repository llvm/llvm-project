//===-- PtrInterfaces.td - Ptr dialect interfaces ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines ptr dialect interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef PTR_INTERFACES
#define PTR_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Access group interface.
//===----------------------------------------------------------------------===//

def AccessGroupOpInterface : OpInterface<"AccessGroupOpInterface"> {
  let description = [{
    An interface for memory operations that can carry access groups metadata.
    It provides setters and getters for the operation's access groups attribute.
    The default implementations of the interface methods expect the operation
    to have an attribute of type ArrayAttr named access_groups.
  }];

  let cppNamespace = "::mlir::ptr";
  let verify = [{ return detail::verifyAccessGroupOpInterface($_op); }];

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the access groups attribute or nullptr",
      /*returnType=*/  "ArrayAttr",
      /*methodName=*/  "getAccessGroupsOrNull",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getAccessGroupsAttr();
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Sets the access groups attribute",
      /*returnType=*/  "void",
      /*methodName=*/  "setAccessGroups",
      /*args=*/        (ins "const ArrayAttr":$attr),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        op.setAccessGroupsAttr(attr);
      }]
      >
  ];
}

//===----------------------------------------------------------------------===//
// Alias analysis interface.
//===----------------------------------------------------------------------===//

def AliasAnalysisOpInterface : OpInterface<"AliasAnalysisOpInterface"> {
  let description = [{
    An interface for memory operations that can carry alias analysis metadata.
    It provides setters and getters for the operation's alias analysis
    attributes. The default implementations of the interface methods expect
    the operation to have attributes of type ArrayAttr named alias_scopes,
    noalias_scopes, and tbaa.
  }];

  let cppNamespace = "::mlir::ptr";
  let verify = [{ return detail::verifyAliasAnalysisOpInterface($_op); }];

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the alias scopes attribute or nullptr",
      /*returnType=*/  "ArrayAttr",
      /*methodName=*/  "getAliasScopesOrNull",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getAliasScopesAttr();
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Sets the alias scopes attribute",
      /*returnType=*/  "void",
      /*methodName=*/  "setAliasScopes",
      /*args=*/        (ins "const ArrayAttr":$attr),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        op.setAliasScopesAttr(attr);
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Returns the noalias scopes attribute or nullptr",
      /*returnType=*/  "ArrayAttr",
      /*methodName=*/  "getNoAliasScopesOrNull",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getNoaliasScopesAttr();
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Sets the noalias scopes attribute",
      /*returnType=*/  "void",
      /*methodName=*/  "setNoAliasScopes",
      /*args=*/        (ins "const ArrayAttr":$attr),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        op.setNoaliasScopesAttr(attr);
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Returns the tbaa attribute or nullptr",
      /*returnType=*/  "ArrayAttr",
      /*methodName=*/  "getTBAATagsOrNull",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getTbaaAttr();
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Sets the tbaa attribute",
      /*returnType=*/  "void",
      /*methodName=*/  "setTBAATags",
      /*args=*/        (ins "const ArrayAttr":$attr),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        op.setTbaaAttr(attr);
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Returns a list of all pointer operands accessed by the "
                       "operation",
      /*returnType=*/  "::llvm::SmallVector<::mlir::Value>",
      /*methodName=*/  "getAccessedOperands",
      /*args=*/        (ins)
      >
  ];
}

#endif // PTR_MEMORYSPACEINTERFACES
