//===-- LLVMInterfaces.td - LLVM Interfaces ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for the LLVM dialect in MLIR.
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
        auto op = cast<ConcreteOp>(this->getOperation());
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

def IntegerOverflowFlagsInterface : OpInterface<"IntegerOverflowFlagsInterface"> {
  let description = [{
    This interface defines an LLVM operation with integer overflow flags and
    provides a uniform API for accessing them.
  }];

  let cppNamespace = "::mlir::LLVM";

  let methods = [
    InterfaceMethod<[{
      Get the integer overflow flags for the operation.
    }], "IntegerOverflowFlags", "getOverflowFlags", (ins), [{}], [{
      return $_op.getProperties().overflowFlags;
    }]>,
    InterfaceMethod<[{
      Set the integer overflow flags for the operation.
    }], "void", "setOverflowFlags", (ins "IntegerOverflowFlags":$flags), [{}], [{
      $_op.getProperties().overflowFlags = flags;
    }]>,
    InterfaceMethod<[{
      Returns whether the operation has the No Unsigned Wrap keyword.
    }], "bool", "hasNoUnsignedWrap", (ins), [{}], [{
      return bitEnumContainsAll($_op.getOverflowFlags(),
                                IntegerOverflowFlags::nuw);
    }]>,
    InterfaceMethod<[{
      Returns whether the operation has the No Signed Wrap keyword.
    }], "bool", "hasNoSignedWrap", (ins), [{}], [{
      return bitEnumContainsAll($_op.getOverflowFlags(),
                                IntegerOverflowFlags::nsw);
    }]>,
    StaticInterfaceMethod<[{
      Get the attribute name of the overflow flags property.
    }], "StringRef", "getOverflowFlagsAttrName", (ins), [{}], [{
      return "overflowFlags";
    }]>,
  ];
}

def BranchWeightOpInterface : OpInterface<"BranchWeightOpInterface"> {
  let description = [{
    An interface for operations that can carry branch weights metadata. It
    provides setters and getters for the operation's branch weights attribute.
    The default implementation of the interface methods expect the operation to
    have an attribute of type DenseI32ArrayAttr named branch_weights.
  }];

  let cppNamespace = "::mlir::LLVM";

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the branch weights attribute or nullptr",
      /*returnType=*/  "DenseI32ArrayAttr",
      /*methodName=*/  "getBranchWeightsOrNull",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getBranchWeightsAttr();
      }]
      >,
    InterfaceMethod<
      /*desc=*/        "Sets the branch weights attribute",
      /*returnType=*/  "void",
      /*methodName=*/  "setBranchWeights",
      /*args=*/        (ins "DenseI32ArrayAttr":$attr),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        op.setBranchWeightsAttr(attr);
      }]
      >
  ];
}

def AccessGroupOpInterface : OpInterface<"AccessGroupOpInterface"> {
  let description = [{
    An interface for memory operations that can carry access groups metadata.
    It provides setters and getters for the operation's access groups attribute.
    The default implementations of the interface methods expect the operation
    to have an attribute of type ArrayAttr named access_groups.
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

def GetResultPtrElementType : OpInterface<"GetResultPtrElementType"> {
  let description = [{
    An interface for operations that yield an LLVMPointer. Allows the
    operation to provide the type of the element an LLVMPointer points to,
    if known. This is only a hint as to how to interpret a given pointer,
    translating how the current operation understands it.
  }];

  let cppNamespace = "::mlir::LLVM";

  let methods = [
    InterfaceMethod<
      /*desc=*/        [{Returns the the element type hint of the result
                         LLVMPointer, if known. Returns nullptr if the
                         requested result is not an LLVMPointer or if the
                         element type is unknown.}],
      /*returnType=*/  "Type",
      /*methodName=*/  "getResultPtrElementType",
      /*args=*/        (ins)
      >
  ];
}

def FPExceptionBehaviorOpInterface : OpInterface<"FPExceptionBehaviorOpInterface"> {
  let description = [{
    An interface for operations receiving an exception behavior attribute
    controlling FP exception behavior.
  }];

  let cppNamespace = "::mlir::LLVM";

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns a FPExceptionBehavior attribute for the operation",
      /*returnType=*/  "FPExceptionBehaviorAttr",
      /*methodName=*/  "getFPExceptionBehaviorAttr",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getFpExceptionBehaviorAttr();
      }]
    >,
    StaticInterfaceMethod<
      /*desc=*/        [{Returns the name of the FPExceptionBehaviorAttr
                        attribute for the operation}],
      /*returnType=*/  "StringRef",
      /*methodName=*/  "getFPExceptionBehaviorAttrName",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        return "fpExceptionBehavior";
      }]
    >
  ];
}

def RoundingModeOpInterface : OpInterface<"RoundingModeOpInterface"> {
  let description = [{
    An interface for operations receiving a rounding mode attribute
    controlling FP rounding mode.
  }];

  let cppNamespace = "::mlir::LLVM";

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns a RoundingMode attribute for the operation",
      /*returnType=*/  "RoundingModeAttr",
      /*methodName=*/  "getRoundingModeAttr",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getRoundingmodeAttr();
      }]
    >,
    StaticInterfaceMethod<
      /*desc=*/        [{Returns the name of the RoundingModeAttr attribute
                         for the operation}],
      /*returnType=*/  "StringRef",
      /*methodName=*/  "getRoundingModeAttrName",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        return "roundingmode";
      }]
    >,
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

//===----------------------------------------------------------------------===//
// LLVM dialect attr interfaces.
//===----------------------------------------------------------------------===//

def LLVM_DIRecursiveTypeAttrInterface
  : AttrInterface<"DIRecursiveTypeAttrInterface"> {
  let description = [{
    This attribute represents a DITypeAttr that is recursive. Only DITypeAttrs
    that translate to LLVM DITypes that support mutation should implement this
    interface.

    There are two modes for conforming attributes:

    1. "rec-decl":
      - This attr is a recursive declaration identified by a recId.

    2. "rec-self":
      - This attr is considered a recursive self reference.
      - This attr itself is a placeholder type that should be conceptually
        replaced with the closest parent attr of the same type with the same
        recId.

    For example, to represent a linked list struct:

      #rec_self = di_composite_type<recId = 0>
      #ptr = di_derived_type<baseType: #rec_self, ...>
      #field = di_derived_type<name = "next", baseType: #ptr, ...>
      #rec = di_composite_type<recId = 0, name = "Node", elements: #field, ...>
      #var = di_local_variable<type = #rec, ...>

    Note that a rec-self without an outer rec-decl with the same recId is
    conceptually the same as an "unbound" variable. The context needs to provide
    meaning to the rec-self.
  }];
  let cppNamespace = "::mlir::LLVM";
  let methods = [
    InterfaceMethod<[{
      Get whether this attr describes a recursive self reference.
    }], "bool", "isRecSelf", (ins)>,
    InterfaceMethod<[{
      Get the recursive ID used for matching "rec-decl" with "rec-self".
      If this attr instance is not recursive, return a null attribute.
    }], "DistinctAttr", "getRecId", (ins)>,
    InterfaceMethod<[{
      Get a copy of this type attr but with the recursive ID set to `recId`.
    }], "DIRecursiveTypeAttrInterface", "withRecId",
    (ins "DistinctAttr":$recId)>,
    StaticInterfaceMethod<[{
      Build a rec-self instance using the provided `recId`.
    }], "DIRecursiveTypeAttrInterface", "getRecSelf",
    (ins "DistinctAttr":$recId)>
  ];
}

#endif // LLVMIR_INTERFACES
