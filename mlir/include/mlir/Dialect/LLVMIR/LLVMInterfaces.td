//===-- LLVMInterfaces.td - LLVM Interfaces ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op and type interfaces for the LLVM dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVMIR_INTERFACES
#define LLVMIR_INTERFACES

include "mlir/IR/OpBase.td"

def FastmathFlagsInterface : OpInterface<"FastmathFlagsInterface"> {
  let description = [{
    Access to op fastmath flags.
  }];

  let cppNamespace = "::mlir::LLVM";

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns a FastmathFlagsAttr attribute for the operation",
      /*returnType=*/  "FastmathFlagsAttr",
      /*methodName=*/  "getFastmathAttr",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getFastmathFlagsAttr();
      }]
      >,
    StaticInterfaceMethod<
      /*desc=*/        [{Returns the name of the FastmathFlagsAttr attribute
                         for the operation}],
      /*returnType=*/  "StringRef",
      /*methodName=*/  "getFastmathAttrName",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        return "fastmathFlags";
      }]
      >
  ];
}

def AccessGroupOpInterface : OpInterface<"AccessGroupOpInterface"> {
  let description = [{
    An interface for memory operations that can carry access groups metadata.
    It provides setters and getters for the operation's access groups attribute.
    The default implementations of the interface methods expect the operation
    to have an array attribute named access_groups.
  }];

  let cppNamespace = "::mlir::LLVM";
  let verify = [{ return detail::verifyAccessGroupOpInterface($_op); }];

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the access groups attribute or nullptr",
      /*returnType=*/  "ArrayAttr",
      /*methodName=*/  "getAccessGroupsOrNull",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
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
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        op.setAccessGroupsAttr(attr);
      }]
      >
  ];
}

def AliasAnalysisOpInterface : OpInterface<"AliasAnalysisOpInterface"> {
  let description = [{
    An interface for memory operations that can carry alias analysis metadata.
    It provides setters and getters for the operation's alias analysis
    attributes. The default implementations of the interface methods expect
    the operation to have attributes of type ArrayAttr named alias_scopes,
    noalias_scopes, and tbaa.
  }];

  let cppNamespace = "::mlir::LLVM";
  let verify = [{ return detail::verifyAliasAnalysisOpInterface($_op); }];

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the alias scopes attribute or nullptr",
      /*returnType=*/  "ArrayAttr",
      /*methodName=*/  "getAliasScopesOrNull",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
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
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
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
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
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
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
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
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
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
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        op.setTbaaAttr(attr);
      }]
      >
  ];
}

//===----------------------------------------------------------------------===//
// LLVM dialect type interfaces.
//===----------------------------------------------------------------------===//

// An interface for LLVM pointer element types.
def LLVM_PointerElementTypeInterface
    : TypeInterface<"PointerElementTypeInterface"> {
  let cppNamespace = "::mlir::LLVM";

  let description = [{
    An interface for types that are allowed as elements of LLVM pointer type.
    Such types must have a size.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Returns the size of the type in bytes.",
      /*retTy=*/"unsigned",
      /*methodName=*/"getSizeInBytes",
      /*args=*/(ins "const DataLayout &":$dataLayout),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return dataLayout.getTypeSize($_type);
      }]
    >
  ];
}

#endif // LLVMIR_INTERFACES
