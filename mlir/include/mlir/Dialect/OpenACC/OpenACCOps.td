//===- OpenACCOps.td - OpenACC operation definitions -------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================
//
// Defines MLIR OpenACC operations.
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_OPS
#define OPENACC_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/OpenACC/OpenACCBase.td"
include "mlir/Dialect/OpenACC/OpenACCOpsTypes.td"
include "mlir/Dialect/OpenACC/OpenACCTypeInterfaces.td"

// AccCommon requires definition of OpenACC_Dialect.
include "mlir/Dialect/OpenACC/AccCommon.td"

// Base class for OpenACC dialect ops.
class OpenACC_Op<string mnemonic, list<Trait> traits = []> :
  Op<OpenACC_Dialect, mnemonic, traits>;

// Reduction operation enumeration.
def OpenACC_ReductionOperatorAdd     : I32EnumAttrCase<"AccAdd", 0, "add">;
def OpenACC_ReductionOperatorMul     : I32EnumAttrCase<"AccMul", 1, "mul">;
def OpenACC_ReductionOperatorMax     : I32EnumAttrCase<"AccMax", 2, "max">;
def OpenACC_ReductionOperatorMin     : I32EnumAttrCase<"AccMin", 3, "min">;
def OpenACC_ReductionOperatorAnd     : I32EnumAttrCase<"AccIand", 4, "iand">;
def OpenACC_ReductionOperatorOr      : I32EnumAttrCase<"AccIor", 5, "ior">;
def OpenACC_ReductionOperatorXor     : I32EnumAttrCase<"AccXor", 6, "xor">;
def OpenACC_ReductionOperatorLogEqv  : I32EnumAttrCase<"AccEqv", 7, "eqv">;
def OpenACC_ReductionOperatorLogNeqv : I32EnumAttrCase<"AccNeqv", 8, "neqv">;
def OpenACC_ReductionOperatorLogAnd  : I32EnumAttrCase<"AccLand", 9, "land">;
def OpenACC_ReductionOperatorLogOr   : I32EnumAttrCase<"AccLor", 10, "lor">;

def OpenACC_ReductionOperator : I32EnumAttr<"ReductionOperator",
    "built-in reduction operations supported by OpenACC",
    [OpenACC_ReductionOperatorAdd, OpenACC_ReductionOperatorMul,
     OpenACC_ReductionOperatorMax, OpenACC_ReductionOperatorMin,
     OpenACC_ReductionOperatorAnd, OpenACC_ReductionOperatorOr,
     OpenACC_ReductionOperatorXor, OpenACC_ReductionOperatorLogEqv,
     OpenACC_ReductionOperatorLogNeqv, OpenACC_ReductionOperatorLogAnd,
     OpenACC_ReductionOperatorLogOr
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::acc";
}
def OpenACC_ReductionOperatorAttr : EnumAttr<OpenACC_Dialect,
                                             OpenACC_ReductionOperator,
                                             "reduction_operator"> {
  let assemblyFormat = [{ ```<` $value `>` }];
}

// Type used in operation below.
def IntOrIndex : AnyTypeOf<[AnyInteger, Index]>;

// Simple alias to pointer-like interface to reduce verbosity.
def OpenACC_PointerLikeType : TypeAlias<OpenACC_PointerLikeTypeInterface,
	"pointer-like type">;

// Define the OpenACC data clauses. There are a few cases where a modifier
// is used, like create(zero), copyin(readonly), and copyout(zero). Since in
// some cases we decompose the original acc data clauses into multiple acc
// dialect operations, we need to keep track of original clause. Thus even
// for the clause with modifier, we create separate operation to make this
// possible.
def OpenACC_CopyinClause          : I64EnumAttrCase<"acc_copyin", 1>;
def OpenACC_CopyinReadonlyClause  : I64EnumAttrCase<"acc_copyin_readonly", 2>;
def OpenACC_CopyClause            : I64EnumAttrCase<"acc_copy", 3>;
def OpenACC_CopyoutClause         : I64EnumAttrCase<"acc_copyout", 4>;
def OpenACC_CopyoutZeroClause     : I64EnumAttrCase<"acc_copyout_zero", 5>;
def OpenACC_PresentClause         : I64EnumAttrCase<"acc_present", 6>;
def OpenACC_CreateClause          : I64EnumAttrCase<"acc_create", 7>;
def OpenACC_CreateZeroClause      : I64EnumAttrCase<"acc_create_zero", 8>;
def OpenACC_DeleteClause          : I64EnumAttrCase<"acc_delete", 9>;
def OpenACC_AttachClause          : I64EnumAttrCase<"acc_attach", 10>;
def OpenACC_DetachClause          : I64EnumAttrCase<"acc_detach", 11>;
def OpenACC_NoCreateClause        : I64EnumAttrCase<"acc_no_create", 12>;
def OpenACC_PrivateClause         : I64EnumAttrCase<"acc_private", 13>;
def OpenACC_FirstPrivateClause    : I64EnumAttrCase<"acc_firstprivate", 14>;
def OpenACC_IsDevicePtrClause     : I64EnumAttrCase<"acc_deviceptr", 15>;
def OpenACC_GetDevicePtrClause    : I64EnumAttrCase<"acc_getdeviceptr", 16>;
def OpenACC_UpdateHost            : I64EnumAttrCase<"acc_update_host", 17>;
def OpenACC_UpdateSelf            : I64EnumAttrCase<"acc_update_self", 18>;
def OpenACC_UpdateDevice          : I64EnumAttrCase<"acc_update_device", 19>;
def OpenACC_UseDevice             : I64EnumAttrCase<"acc_use_device", 20>;
def OpenACC_Reduction             : I64EnumAttrCase<"acc_reduction", 21>;
def OpenACC_DeclareDeviceResident : I64EnumAttrCase<"acc_declare_device_resident", 22>;
def OpenACC_DeclareLink           : I64EnumAttrCase<"acc_declare_link", 23>;

def OpenACC_DataClauseEnum : I64EnumAttr<"DataClause",
    "data clauses supported by OpenACC",
    [OpenACC_CopyinClause, OpenACC_CopyinReadonlyClause, OpenACC_CopyClause,
     OpenACC_CopyoutClause, OpenACC_CopyoutZeroClause, OpenACC_PresentClause,
     OpenACC_CreateClause, OpenACC_CreateZeroClause, OpenACC_DeleteClause,
     OpenACC_AttachClause, OpenACC_DetachClause, OpenACC_NoCreateClause,
     OpenACC_PrivateClause, OpenACC_FirstPrivateClause,
     OpenACC_IsDevicePtrClause, OpenACC_GetDevicePtrClause, OpenACC_UpdateHost,
     OpenACC_UpdateSelf, OpenACC_UpdateDevice, OpenACC_UseDevice,
     OpenACC_Reduction, OpenACC_DeclareDeviceResident, OpenACC_DeclareLink,
    ]> {
  let cppNamespace = "::mlir::acc";
  let genSpecializedAttr = 0;
}

def OpenACC_DataClauseAttr : EnumAttr<OpenACC_Dialect, OpenACC_DataClauseEnum,
                                      "data_clause">;

class OpenACC_Attr<string name, string attrMnemonic,
                   list<Trait> traits = [],
                   string baseCppClass = "::mlir::Attribute">
    : AttrDef<OpenACC_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

// Attribute to describe the declare data clause used on variable.
// Intended to be used at the variable creation site (on the global op or the
// corresponding allocation operation). This is used in conjunction with the
// declare operations (`acc.declare_enter` and `acc.declare_exit`) since those
// describe how the data action is performed. The attribute itself makes it
// easier to find out whether the variable is in a declare clause and what kind
// of clause it is.
def DeclareAttr : OpenACC_Attr<"Declare", "declare"> {
  let parameters = (ins "DataClauseAttr":$dataClause);
  let assemblyFormat = "`<` struct(params) `>`";
}

// Used for data specification in data clauses (2.7.1).
// Either (or both) extent and upperbound must be specified.
def OpenACC_DataBoundsOp : OpenACC_Op<"bounds",
    [AttrSizedOperandSegments, NoMemoryEffect]> {
  let summary = "Represents normalized bounds information for acc data clause.";

  let description = [{
    This operation is used to record bounds used in acc data clause in a
    normalized fashion (zero-based). This works well with the `PointerLikeType`
    requirement in data clauses - since a `lowerbound` of 0 means looking
    at data at the zero offset from pointer.

    The operation must have an `upperbound` or `extent` (or both are allowed -
    but not checked for consistency). When the source language's arrays are
    not zero-based, the `startIdx` must specify the zero-position index.

    Examples below show copying a slice of 10-element array except first element.
    Note that the examples use extent in data clause for C++ and upperbound
    for Fortran (as per 2.7.1). To simplify examples, the constants are used
    directly in the acc.bounds operands - this is not the syntax of operation.

    C++:
    ```
    int array[10];
    #pragma acc copy(array[1:9])
    ```
    =>
    ```mlir
    acc.bounds lb(1) ub(9) extent(9) startIdx(0)
    ```

    Fortran:
    ```
    integer :: array(1:10)
    !$acc copy(array(2:10))
    ```
    =>
    ```mlir
    acc.bounds lb(1) ub(9) extent(9) startIdx(1)
    ```
  }];

  let arguments = (ins Optional<IntOrIndex>:$lowerbound,
                       Optional<IntOrIndex>:$upperbound,
                       Optional<IntOrIndex>:$extent,
                       Optional<IntOrIndex>:$stride,
                       DefaultValuedAttr<BoolAttr, "false">:$strideInBytes,
                       Optional<IntOrIndex>:$startIdx);
  let results = (outs OpenACC_DataBoundsType:$result);

  let assemblyFormat = [{
    oilist(
        `lowerbound` `(` $lowerbound `:` type($lowerbound) `)`
      | `upperbound` `(` $upperbound `:` type($upperbound) `)`
      | `extent` `(` $extent `:` type($extent) `)`
      | `stride` `(` $stride `:` type($stride) `)`
      | `startIdx` `(` $startIdx `:` type($startIdx) `)`
    ) attr-dict
  }];

  let hasVerifier = 1;
}

// Data entry operation does not refer to OpenACC spec terminology, but to
// terminology used in this dialect. It refers to data operations that will
// appear before data or compute region. It will be used as the base of acc
// dialect operations for the following OpenACC data clauses: copyin, create,
// present, attach, deviceptr.
//
// The bounds are represented in rank order. Rank 0 (inner-most dimension) is
// the first.
class OpenACC_DataEntryOp<string mnemonic, string clause, string extraDescription,
                          list<Trait> traits = []> :
    OpenACC_Op<mnemonic, !listconcat(traits,
        [AttrSizedOperandSegments])> {
  let arguments = (ins OpenACC_PointerLikeTypeInterface:$varPtr,
                       Optional<OpenACC_PointerLikeTypeInterface>:$varPtrPtr,
                       Variadic<OpenACC_DataBoundsType>:$bounds, /* rank-0 to rank-{n-1} */
                       DefaultValuedAttr<OpenACC_DataClauseAttr,clause>:$dataClause,
                       DefaultValuedAttr<BoolAttr, "true">:$structured,
                       DefaultValuedAttr<BoolAttr, "false">:$implicit,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs OpenACC_PointerLikeTypeInterface:$accPtr);

  let description = !strconcat(extraDescription, [{
    Description of arguments:
    - `varPtr`: The address of variable to copy.
    - `varPtrPtr`: Specifies the address of varPtr - only used when the variable
    copied is a field in a struct. This is important for OpenACC due to implicit
    attach semantics on data clauses (2.6.4).
    - `bounds`: Used when copying just slice of array or array's bounds are not
    encoded in type. They are in rank order where rank 0 is inner-most dimension.
    - `dataClause`: Keeps track of the data clause the user used. This is because
    the acc operations are decomposed. So a 'copy' clause is decomposed to both 
    `acc.copyin` and `acc.copyout` operations, but both have dataClause that
    specifies `acc_copy` in this field.
    - `structured`: Flag to note whether this is associated with structured region
    (parallel, kernels, data) or unstructured (enter data, exit data). This is
    important due to spec specifically calling out structured and dynamic reference
    counters (2.6.7).
    - `implicit`: Whether this is an implicitly generated operation, such as copies
    done to satisfy "Variables with Implicitly Determined Data Attributes" in 2.6.2.
    - `name`: Holds the name of variable as specified in user clause (including bounds).
  }]);

  let assemblyFormat = [{
    `varPtr` `(` $varPtr `:` type($varPtr) `)`
    oilist(
        `varPtrPtr` `(` $varPtrPtr `:` type($varPtrPtr) `)`
      | `bounds` `(` $bounds `)`
    ) `->` type($accPtr) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.5.13 private clause
//===----------------------------------------------------------------------===//
def OpenACC_PrivateOp : OpenACC_DataEntryOp<"private",
    "mlir::acc::DataClause::acc_private", ""> {
  let summary = "Represents private semantics for acc private clause.";
}

//===----------------------------------------------------------------------===//
// 2.5.14 firstprivate clause
//===----------------------------------------------------------------------===//
def OpenACC_FirstprivateOp : OpenACC_DataEntryOp<"firstprivate",
    "mlir::acc::DataClause::acc_firstprivate", ""> {
  let summary = "Represents firstprivate semantic for the acc firstprivate "
                "clause.";
}

//===----------------------------------------------------------------------===//
// 2.5.15 reduction clause
//===----------------------------------------------------------------------===//
def OpenACC_ReductionOp : OpenACC_DataEntryOp<"reduction",
    "mlir::acc::DataClause::acc_reduction", ""> {
  let summary = "Represents reduction semantics for acc reduction clause.";
}

//===----------------------------------------------------------------------===//
// 2.7.4 deviceptr clause
//===----------------------------------------------------------------------===//
def OpenACC_DevicePtrOp : OpenACC_DataEntryOp<"deviceptr",
    "mlir::acc::DataClause::acc_deviceptr", ""> {
  let summary = "Specifies that the variable pointer is a device pointer.";
}

//===----------------------------------------------------------------------===//
// 2.7.5 present clause
//===----------------------------------------------------------------------===//
def OpenACC_PresentOp : OpenACC_DataEntryOp<"present",
    "mlir::acc::DataClause::acc_present", ""> {
  let summary = "Specifies that the variable is already present on device.";
}

//===----------------------------------------------------------------------===//
// 2.7.7 copyin clause
//===----------------------------------------------------------------------===//
def OpenACC_CopyinOp : OpenACC_DataEntryOp<"copyin",
    "mlir::acc::DataClause::acc_copyin", ""> {
  let summary = "Represents copyin semantics for acc data clauses like acc "
                "copyin and acc copy.";

  let extraClassDeclaration = [{
    /// Check if this is a copyin with readonly modifier.
    bool isCopyinReadonly();
  }];
}

//===----------------------------------------------------------------------===//
// 2.7.9 create clause
//===----------------------------------------------------------------------===//
def OpenACC_CreateOp : OpenACC_DataEntryOp<"create",
    "mlir::acc::DataClause::acc_create", ""> {
  let summary = "Represents create semantics for acc data clauses like acc "
                "create and acc copyout.";

  let extraClassDeclaration = [{
    /// Check if this is a create with zero modifier.
    bool isCreateZero();
  }];
}

//===----------------------------------------------------------------------===//
// 2.7.10 no_create clause
//===----------------------------------------------------------------------===//
def OpenACC_NoCreateOp : OpenACC_DataEntryOp<"nocreate",
    "mlir::acc::DataClause::acc_no_create", ""> {
  let summary = "Represents acc no_create semantics.";
}

//===----------------------------------------------------------------------===//
// 2.7.12 attach clause
//===----------------------------------------------------------------------===//
def OpenACC_AttachOp : OpenACC_DataEntryOp<"attach",
    "mlir::acc::DataClause::acc_attach", ""> {
  let summary = "Represents acc attach semantics which updates a pointer in "
                "device memory with the corresponding device address of the "
                "pointee.";
}

//===----------------------------------------------------------------------===//
// 3.2.23 acc_deviceptr
//===----------------------------------------------------------------------===//
// This is needed to get device address without the additional semantics in
// acc present. Effectively, it can be used to get "accPtr" for any variable.
// It is also useful for providing the device address for unstructured construct
// exit_data since unlike structured constructs, there is no matching data entry
// operation.
def OpenACC_GetDevicePtrOp : OpenACC_DataEntryOp<"getdeviceptr",
    "mlir::acc::DataClause::acc_getdeviceptr", [{
      This operation is used to get the `accPtr` for a variable. This is often
      used in conjunction with data exit operations when the data entry
      operation is not visible. This operation can have a `dataClause` argument
      that is any of the valid `mlir::acc::DataClause` entries.
      \
    }]> {
  let summary = "Gets device address if variable exists on device.";
  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// 2.14.4 device clause
//===----------------------------------------------------------------------===//
def OpenACC_UpdateDeviceOp : OpenACC_DataEntryOp<"update_device",
    "mlir::acc::DataClause::acc_update_device", ""> {
  let summary = "Represents acc update device semantics.";
}

//===----------------------------------------------------------------------===//
// 2.8 use_device clause
//===----------------------------------------------------------------------===//
def OpenACC_UseDeviceOp : OpenACC_DataEntryOp<"use_device",
    "mlir::acc::DataClause::acc_use_device", ""> {
  let summary = "Represents acc use_device semantics.";
}

//===----------------------------------------------------------------------===//
// 2.13.1 device_resident clause
//===----------------------------------------------------------------------===//
def OpenACC_DeclareDeviceResidentOp : OpenACC_DataEntryOp<"declare_device_resident",
    "mlir::acc::DataClause::acc_declare_device_resident", ""> {
  let summary = "Represents acc declare device_resident semantics.";
}

//===----------------------------------------------------------------------===//
// 2.13.3 link clause
//===----------------------------------------------------------------------===//
def OpenACC_DeclareLinkOp : OpenACC_DataEntryOp<"declare_link",
    "mlir::acc::DataClause::acc_declare_link", ""> {
  let summary = "Represents acc declare link semantics.";
}

// Data exit operation does not refer to OpenACC spec terminology, but to
// terminology used in this dialect. It refers to data operations that will appear
// after data or compute region. It will be used as the base of acc dialect
// operations for the following OpenACC data clauses: copyout, detach, delete.
class OpenACC_DataExitOp<string mnemonic, string clause, list<Trait> traits = []> :
    OpenACC_Op<mnemonic, !listconcat(traits,
        [AttrSizedOperandSegments])> {
  let arguments = (ins OpenACC_PointerLikeTypeInterface:$accPtr,
                       Optional<OpenACC_PointerLikeTypeInterface>:$varPtr,
                       Variadic<OpenACC_DataBoundsType>:$bounds,
                       DefaultValuedAttr<OpenACC_DataClauseAttr,clause>:$dataClause,
                       DefaultValuedAttr<BoolAttr, "true">:$structured,
                       DefaultValuedAttr<BoolAttr, "false">:$implicit,
                       OptionalAttr<StrAttr>:$name);

  let description = [{
    - `varPtr`: The address of variable to copy back to. This only applies to
    `acc.copyout`
    - `accPtr`: The acc address of variable. This is the link from the data-entry
    operation used.
    - `bounds`: Used when copying just slice of array or array's bounds are not
    encoded in type. They are in rank order where rank 0 is inner-most dimension.
    - `dataClause`: Keeps track of the data clause the user used. This is because
    the acc operations are decomposed. So a 'copy' clause is decomposed to both 
    `acc.copyin` and `acc.copyout` operations, but both have dataClause that
    specifies `acc_copy` in this field.
    - `structured`: Flag to note whether this is associated with structured region
    (parallel, kernels, data) or unstructured (enter data, exit data). This is
    important due to spec specifically calling out structured and dynamic reference
    counters (2.6.7).
    - `implicit`: Whether this is an implicitly generated operation, such as copies
    done to satisfy "Variables with Implicitly Determined Data Attributes" in 2.6.2.
    - `name`: Holds the name of variable as specified in user clause (including bounds).
  }];

  let assemblyFormat = [{
    `accPtr` `(` $accPtr `:` type($accPtr) `)`
    oilist(
        `bounds` `(` $bounds `)`
      | `to` `varPtr` `(` $varPtr `:` type($varPtr) `)`
    ) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.7.8 copyout clause
//===----------------------------------------------------------------------===//
def OpenACC_CopyoutOp : OpenACC_DataExitOp<"copyout",
    "mlir::acc::DataClause::acc_copyout"> {
  let summary = "Represents acc copyout semantics - reverse of copyin.";

  let extraClassDeclaration = [{
    /// Check if this is a copyout with zero modifier.
    bool isCopyoutZero();
  }];
}

//===----------------------------------------------------------------------===//
// 2.7.11 delete clause
//===----------------------------------------------------------------------===//
def OpenACC_DeleteOp : OpenACC_DataExitOp<"delete",
    "mlir::acc::DataClause::acc_delete"> {
  let summary = "Represents acc delete semantics - reverse of create.";
}

//===----------------------------------------------------------------------===//
// 2.7.13 detach clause
//===----------------------------------------------------------------------===//
def OpenACC_DetachOp : OpenACC_DataExitOp<"detach",
    "mlir::acc::DataClause::acc_detach"> {
  let summary = "Represents acc detach semantics - reverse of attach.";
}

//===----------------------------------------------------------------------===//
// 2.14.4 host clause
//===----------------------------------------------------------------------===//
def OpenACC_UpdateHostOp : OpenACC_DataExitOp<"update_host",
    "mlir::acc::DataClause::acc_update_host"> {
  let summary = "Represents acc update host semantics.";
  let extraClassDeclaration = [{
    /// Check if this is an acc update self.
    bool isSelf() {
      return getDataClause() == acc::DataClause::acc_update_self;
    }
  }];
}

//===----------------------------------------------------------------------===//
// 2.5.13 private clause
//===----------------------------------------------------------------------===//

def OpenACC_PrivateRecipeOp : OpenACC_Op<"private.recipe",
    [IsolatedFromAbove, Symbol]> {
  let summary = "privatization recipe";

  let description = [{
    Declares an OpenACC privatization recipe. The operation requires one
    mandatory and one optional region.

      1. The initializer region specifies how to allocate and initialize a new
         private value. For example in Fortran, a derived-type might have a
         default initialization. The region has an argument that contains the
         value that need to be privatized. This is useful if the type is not
         known at compile time and the private value is needed to create its
         copy.
      2. The destroy region specifies how to destruct the value when it reaches
         its end of life. It takes the privatized value as argument.

    A single privatization recipe can be used for multiple operand if they have
    the same type and do not require a specific default initialization.

    Example:

    ```mlir
    acc.private.recipe @privatization_f32 : f32 init {
    ^bb0(%0: f32):
      // init region contains a sequence of operations to create and
      // initialize the copy if needed. It yields the create copy.
    } destroy {
    ^bb0(%0: f32)
      // destroy region contains a sequences of operations to destruct the
      // created copy.
    }

    // The privatization symbol is then used in the corresponding operation.
    acc.parallel private(@privatization_f32 -> %a : f32) {
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$type);

  let regions = (region AnyRegion:$initRegion,
                        AnyRegion:$destroyRegion);

  let assemblyFormat = [{
    $sym_name `:` $type attr-dict-with-keyword `init` $initRegion
    (`destroy` $destroyRegion^)?
  }];

  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.5.14 firstprivate clause
//===----------------------------------------------------------------------===//

def OpenACC_FirstprivateRecipeOp : OpenACC_Op<"firstprivate.recipe",
    [IsolatedFromAbove, Symbol]> {
  let summary = "privatization recipe";

  let description = [{
    Declares an OpenACC privatization recipe with copy of the initial value.
    The operation requires two mandatory regions and one optional.

      1. The initializer region specifies how to allocate and initialize a new
         private value. For example in Fortran, a derived-type might have a
         default initialization. The region has an argument that contains the
         value that need to be privatized. This is useful if the type is not
         known at compile time and the private value is needed to create its
         copy.
      2. The copy region specifies how to copy the initial value to the newly
         created private value. It takes the initial value and the privatized
         value as arguments.
      3. The destroy region specifies how to destruct the value when it reaches
         its end of life. It takes the privatized value as argument. It is
         optional.

    A single privatization recipe can be used for multiple operand if they have
    the same type and do not require a specific default initialization.

    Example:

    ```mlir
    acc.firstprivate.recipe @privatization_f32 : f32 init {
    ^bb0(%0: f32):
      // init region contains a sequence of operations to create and
      // initialize the copy if needed. It yields the create copy.
    } copy {
    ^bb0(%0: f32, %1: !llvm.ptr<f32>):
      // copy region contains a sequence of operations to copy the initial value
      // of the firstprivate value to the newly created value.
    } destroy {
    ^bb0(%0: f32)
      // destroy region contains a sequences of operations to destruct the
      // created copy.
    }

    // The privatization symbol is then used in the corresponding operation.
    acc.parallel firstprivate(@privatization_f32 -> %a : f32) {
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$type);

  let regions = (region AnyRegion:$initRegion, AnyRegion:$copyRegion,
                        AnyRegion:$destroyRegion);

  let assemblyFormat = [{
    $sym_name `:` $type attr-dict-with-keyword `init` $initRegion
    `copy` $copyRegion
    (`destroy` $destroyRegion^)?
  }];

  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.5.15 reduction clause
//===----------------------------------------------------------------------===//

def OpenACC_ReductionRecipeOp : OpenACC_Op<"reduction.recipe",
    [IsolatedFromAbove, Symbol]> {
  let summary = "reduction recipe";

  let description = [{
    Declares an OpenACC reduction recipe. The operation requires two
    mandatory regions.

      1. The initializer region specifies how to initialize the local reduction
         value. The region has a first argument that contains the value of the
         reduction accumulator at the start of the reduction. It is expected to
         `acc.yield` the new value. Extra arguments can be added to deal with
         dynamic arrays.
      2. The reduction region contains a sequences of operations to combine two
         values of the reduction type into one. It has at least two arguments
         and it is expected to `acc.yield` the combined value. Extra arguments
         can be added to deal with dynamic arrays.

    Example:

    ```mlir
    acc.reduction.recipe @reduction_add_i64 : i64 reduction_operator<add> init {
    ^bb0(%0: i64):
      // init region contains a sequence of operations to initialize the local
      // reduction value as specified in 2.5.15
      %c0 = arith.constant 0 : i64
      acc.yield %c0 : i64
    } combiner {
    ^bb0(%0: i64, %1: i64)
      // combiner region contains a sequence of operations to combine
      // two values into one.
      %2 = arith.addi %0, %1 : i64
      acc.yield %2 : i64
    }

    // The reduction symbol is then used in the corresponding operation.
    acc.parallel reduction(@reduction_add_i64 -> %a : i64) {
    }
    ```

    The following table lists the valid operators and the initialization values
    according to OpenACC 3.3:

    |------------------------------------------------|
    |        C/C++          |        Fortran         |
    |-----------------------|------------------------|
    | operator | init value | operator | init value  |
    |     +    |      0     |     +    |      0      |
    |     *    |      1     |     *    |      1      |
    |    max   |    least   |    max   |    least    |
    |    min   |   largest  |    min   |   largest   |
    |     &    |     ~0     |   iand   | all bits on |
    |     |    |      0     |    ior   |      0      |
    |     ^    |      0     |   ieor   |      0      |
    |    &&    |      1     |   .and.  |    .true.   |
    |    ||    |      0     |    .or.  |   .false.   |
    |          |            |   .eqv.  |    .true.   |
    |          |            |  .neqv.  |   .false.   |
    -------------------------------------------------|
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$type,
                       OpenACC_ReductionOperatorAttr:$reductionOperator);

  let regions = (region AnyRegion:$initRegion,
                        AnyRegion:$combinerRegion);

  let assemblyFormat = [{
    $sym_name `:` $type attr-dict-with-keyword
    `reduction_operator` $reductionOperator
    `init` $initRegion `combiner` $combinerRegion
  }];

  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.5.1 parallel Construct
//===----------------------------------------------------------------------===//

def OpenACC_ParallelOp : OpenACC_Op<"parallel",
    [AttrSizedOperandSegments, RecursiveMemoryEffects]> {
  let summary = "parallel construct";
  let description = [{
    The "acc.parallel" operation represents a parallel construct block. It has
    one region to be executed in parallel on the current device.

    Example:

    ```mlir
    acc.parallel num_gangs(%c10) num_workers(%c10)
        private(%c : memref<10xf32>) {
      // parallel region
    }
    ```
  }];

  let arguments = (ins Optional<IntOrIndex>:$async,
                       UnitAttr:$asyncAttr,
                       Variadic<IntOrIndex>:$waitOperands,
                       UnitAttr:$waitAttr,
                       Variadic<IntOrIndex>:$numGangs,
                       Optional<IntOrIndex>:$numWorkers,
                       Optional<IntOrIndex>:$vectorLength,
                       Optional<I1>:$ifCond,
                       Optional<I1>:$selfCond,
                       UnitAttr:$selfAttr,
                       Variadic<AnyType>:$reductionOperands,
                       OptionalAttr<SymbolRefArrayAttr>:$reductionRecipes,
                       Variadic<OpenACC_PointerLikeTypeInterface>:$gangPrivateOperands,
                       OptionalAttr<SymbolRefArrayAttr>:$privatizations,
                       Variadic<OpenACC_PointerLikeTypeInterface>:$gangFirstPrivateOperands,
                       OptionalAttr<SymbolRefArrayAttr>:$firstprivatizations,
                       Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands,
                       OptionalAttr<DefaultValueAttr>:$defaultAttr);

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);
  }];

  let assemblyFormat = [{
    oilist(
        `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
      | `async` `(` $async `:` type($async) `)`
      | `firstprivate` `(` custom<SymOperandList>($gangFirstPrivateOperands,
            type($gangFirstPrivateOperands), $firstprivatizations)
        `)`
      | `num_gangs` `(` $numGangs `:` type($numGangs) `)`
      | `num_workers` `(` $numWorkers `:` type($numWorkers) `)`
      | `private` `(` custom<SymOperandList>(
            $gangPrivateOperands, type($gangPrivateOperands), $privatizations)
        `)`
      | `vector_length` `(` $vectorLength `:` type($vectorLength) `)`
      | `wait` `(` $waitOperands `:` type($waitOperands) `)`
      | `self` `(` $selfCond `)`
      | `if` `(` $ifCond `)`
      | `reduction` `(` custom<SymOperandList>(
            $reductionOperands, type($reductionOperands), $reductionRecipes)
        `)`
    )
    $region attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.5.2 serial Construct
//===----------------------------------------------------------------------===//

def OpenACC_SerialOp : OpenACC_Op<"serial",
    [AttrSizedOperandSegments, RecursiveMemoryEffects]> {
  let summary = "serial construct";
  let description = [{
    The "acc.serial" operation represents a serial construct block. It has
    one region to be executed in serial on the current device.

    Example:

    ```mlir
    acc.serial private(%c : memref<10xf32>) {
      // serial region
    }
    ```
  }];

  let arguments = (ins Optional<IntOrIndex>:$async,
                       UnitAttr:$asyncAttr,
                       Variadic<IntOrIndex>:$waitOperands,
                       UnitAttr:$waitAttr,
                       Optional<I1>:$ifCond,
                       Optional<I1>:$selfCond,
                       UnitAttr:$selfAttr,
                       Variadic<AnyType>:$reductionOperands,
                       OptionalAttr<SymbolRefArrayAttr>:$reductionRecipes,
                       Variadic<OpenACC_PointerLikeTypeInterface>:$gangPrivateOperands,
                       OptionalAttr<SymbolRefArrayAttr>:$privatizations,
                       Variadic<OpenACC_PointerLikeTypeInterface>:$gangFirstPrivateOperands,
                       OptionalAttr<SymbolRefArrayAttr>:$firstprivatizations,
                       Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands,
                       OptionalAttr<DefaultValueAttr>:$defaultAttr);

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);
  }];

  let assemblyFormat = [{
    oilist(
        `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
      | `async` `(` $async `:` type($async) `)`
      | `firstprivate` `(` custom<SymOperandList>($gangFirstPrivateOperands,
            type($gangFirstPrivateOperands), $firstprivatizations)
        `)`
      | `private` `(` custom<SymOperandList>(
            $gangPrivateOperands, type($gangPrivateOperands), $privatizations)
        `)`
      | `wait` `(` $waitOperands `:` type($waitOperands) `)`
      | `self` `(` $selfCond `)`
      | `if` `(` $ifCond `)`
      | `reduction` `(` custom<SymOperandList>(
            $reductionOperands, type($reductionOperands), $reductionRecipes)
        `)`
    )
    $region attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.5.1 kernels Construct
//===----------------------------------------------------------------------===//

def OpenACC_KernelsOp : OpenACC_Op<"kernels",
    [AttrSizedOperandSegments, RecursiveMemoryEffects]> {
  let summary = "kernels construct";
  let description = [{
    The "acc.kernels" operation represents a kernels construct block. It has
    one region to be compiled into a sequence of kernels for execution on the
    current device.

    Example:

    ```mlir
    acc.kernels num_gangs(%c10) num_workers(%c10)
        private(%c : memref<10xf32>) {
      // kernels region
    }
    ```
  }];

  let arguments = (ins Optional<IntOrIndex>:$async,
                       UnitAttr:$asyncAttr,
                       Variadic<IntOrIndex>:$waitOperands,
                       UnitAttr:$waitAttr,
                       Variadic<IntOrIndex>:$numGangs,
                       Optional<IntOrIndex>:$numWorkers,
                       Optional<IntOrIndex>:$vectorLength,
                       Optional<I1>:$ifCond,
                       Optional<I1>:$selfCond,
                       UnitAttr:$selfAttr,
                       Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands,
                       OptionalAttr<DefaultValueAttr>:$defaultAttr);

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);
  }];

  let assemblyFormat = [{
    oilist(
        `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
      | `async` `(` $async `:` type($async) `)`
      | `num_gangs` `(` $numGangs `:` type($numGangs) `)`
      | `num_workers` `(` $numWorkers `:` type($numWorkers) `)`
      | `vector_length` `(` $vectorLength `:` type($vectorLength) `)`
      | `wait` `(` $waitOperands `:` type($waitOperands) `)`
      | `self` `(` $selfCond `)`
      | `if` `(` $ifCond `)`
    )
    $region attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.6.5 data Construct
//===----------------------------------------------------------------------===//

def OpenACC_DataOp : OpenACC_Op<"data",
    [AttrSizedOperandSegments, RecursiveMemoryEffects]> {
  let summary = "data construct";

  let description = [{
    The "acc.data" operation represents a data construct. It defines vars to
    be allocated in the current device memory for the duration of the region,
    whether data should be copied from local memory to the current device
    memory upon region entry , and copied from device memory to local memory
    upon region exit.

    Example:

    ```mlir
    acc.data present(%a: memref<10x10xf32>, %b: memref<10x10xf32>,
        %c: memref<10xf32>, %d: memref<10xf32>) {
      // data region
    }
    ```
  }];


  let arguments = (ins Optional<I1>:$ifCond,
                       Optional<IntOrIndex>:$async,
                       UnitAttr:$asyncAttr,
                       Optional<IntOrIndex>:$waitDevnum,
                       Variadic<IntOrIndex>:$waitOperands,
                       UnitAttr:$waitAttr,
                       Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands,
                       OptionalAttr<DefaultValueAttr>:$defaultAttr);

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);
  }];

  let assemblyFormat = [{
    oilist(
        `if` `(` $ifCond `)`
      | `async` `(` $async `:` type($async) `)`
      | `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
      | `wait_devnum` `(` $waitDevnum `:` type($waitDevnum) `)`
      | `wait` `(` $waitOperands `:` type($waitOperands) `)`
    )
    $region attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

def OpenACC_TerminatorOp : OpenACC_Op<"terminator", [Terminator]> {
  let summary = "Generic terminator for OpenACC regions";

  let description = [{
    A terminator operation for regions that appear in the body of OpenACC
    operation. Generic OpenACC construct regions are not expected to return any
    value so the terminator takes no operands. The terminator op returns control
    to the enclosing op.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// 2.6.6 Enter Data Directive
//===----------------------------------------------------------------------===//

def OpenACC_EnterDataOp : OpenACC_Op<"enter_data", [AttrSizedOperandSegments]> {
  let summary = "enter data operation";

  let description = [{
    The "acc.enter_data" operation represents the OpenACC enter data directive.

    Example:

    ```mlir
    acc.enter_data create(%d1 : memref<10xf32>) attributes {async}
    ```
  }];

  let arguments = (ins Optional<I1>:$ifCond,
                       Optional<IntOrIndex>:$asyncOperand,
                       UnitAttr:$async,
                       Optional<IntOrIndex>:$waitDevnum,
                       Variadic<IntOrIndex>:$waitOperands,
                       UnitAttr:$wait,
                       Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);
  }];

  let assemblyFormat = [{
    oilist(
        `if` `(` $ifCond `)`
      | `async` `(` $asyncOperand `:` type($asyncOperand) `)`
      | `wait_devnum` `(` $waitDevnum `:` type($waitDevnum) `)`
      | `wait` `(` $waitOperands `:` type($waitOperands) `)`
      | `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
    )
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.6.6 Exit Data Directive
//===----------------------------------------------------------------------===//

def OpenACC_ExitDataOp : OpenACC_Op<"exit_data", [AttrSizedOperandSegments]> {
  let summary = "exit data operation";

  let description = [{
    The "acc.exit_data" operation represents the OpenACC exit data directive.

    Example:

    ```mlir
    acc.exit_data delete(%d1 : memref<10xf32>) attributes {async}
    ```
  }];

  let arguments = (ins Optional<I1>:$ifCond,
                       Optional<IntOrIndex>:$asyncOperand,
                       UnitAttr:$async,
                       Optional<IntOrIndex>:$waitDevnum,
                       Variadic<IntOrIndex>:$waitOperands,
                       UnitAttr:$wait,
                       Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands,
                       UnitAttr:$finalize);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);
  }];

  let assemblyFormat = [{
    oilist(
        `if` `(` $ifCond `)`
      | `async` `(` $asyncOperand `:` type($asyncOperand) `)`
      | `wait_devnum` `(` $waitDevnum `:` type($waitDevnum) `)`
      | `wait` `(` $waitOperands `:` type($waitOperands) `)`
      | `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
    )
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.8 Host_Data Construct
//===----------------------------------------------------------------------===//

def OpenACC_HostDataOp : OpenACC_Op<"host_data", [AttrSizedOperandSegments]> {
  let summary = "host_data construct";

  let description = [{
    The "acc.host_data" operation represents the OpenACC host_data construct.

    Example:

    ```mlir
    %0 = acc.use_device varPtr(%a : !llvm.ptr<f32>) -> !llvm.ptr<f32>
    acc.host_data dataOperands(%0 : !llvm.ptr<f32>) {

    }
    ```
  }];

  let arguments = (ins Optional<I1>:$ifCond,
                       Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands,
                       UnitAttr:$ifPresent);

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    oilist(
        `if` `(` $ifCond `)`
      | `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
    )
    $region attr-dict-with-keyword
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// 2.9 loop Construct
//===----------------------------------------------------------------------===//

def OpenACC_LoopOp : OpenACC_Op<"loop",
    [AttrSizedOperandSegments, RecursiveMemoryEffects]> {
  let summary = "loop construct";

  let description = [{
    The "acc.loop" operation represents the OpenACC loop construct.

    Example:

    ```mlir
    acc.loop gang vector {
      scf.for %arg3 = %c0 to %c10 step %c1 {
        scf.for %arg4 = %c0 to %c10 step %c1 {
          scf.for %arg5 = %c0 to %c10 step %c1 {
            // ... body
          }
        }
      }
      acc.yield
    } attributes { collapse = 3 }
    ```
  }];

  let arguments = (ins OptionalAttr<I64Attr>:$collapse,
                       Optional<IntOrIndex>:$gangNum,
                       Optional<IntOrIndex>:$gangDim,
                       Optional<IntOrIndex>:$gangStatic,
                       Optional<IntOrIndex>:$workerNum,
                       Optional<IntOrIndex>:$vectorLength,
                       UnitAttr:$seq,
                       UnitAttr:$independent,
                       UnitAttr:$auto_,
                       UnitAttr:$hasGang,
                       UnitAttr:$hasWorker,
                       UnitAttr:$hasVector,
                       Variadic<IntOrIndex>:$tileOperands,
                       Variadic<OpenACC_PointerLikeTypeInterface>:$privateOperands,
                       OptionalAttr<SymbolRefArrayAttr>:$privatizations,
                       Variadic<AnyType>:$reductionOperands,
                       OptionalAttr<SymbolRefArrayAttr>:$reductionRecipes);

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    static StringRef getAutoAttrStrName() { return "auto"; }
    static StringRef getGangNumKeyword() { return "num"; }
    static StringRef getGangDimKeyword() { return "dim"; }
    static StringRef getGangStaticKeyword() { return "static"; }
  }];

  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    oilist(
        `gang` `` custom<GangClause>($gangNum, type($gangNum), $gangDim, type($gangDim), $gangStatic, type($gangStatic), $hasGang)
      | `worker` `` custom<WorkerClause>($workerNum, type($workerNum), $hasWorker)
      | `vector` `` custom<VectorClause>($vectorLength, type($vectorLength), $hasVector)
      | `private` `(` custom<SymOperandList>(
            $privateOperands, type($privateOperands), $privatizations)
        `)`
      | `tile` `(` $tileOperands `:` type($tileOperands) `)`
      | `reduction` `(` custom<SymOperandList>(
            $reductionOperands, type($reductionOperands), $reductionRecipes)
        `)`
    )
    $region
    ( `(` type($results)^ `)` )?
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}

// Yield operation for the acc.loop and acc.parallel operations.
def OpenACC_YieldOp : OpenACC_Op<"yield", [ReturnLike, Terminator,
    ParentOneOf<["FirstprivateRecipeOp, LoopOp, ParallelOp, PrivateRecipeOp, ReductionRecipeOp, SerialOp"]>]> {
  let summary = "Acc yield and termination operation";

  let description = [{
    `acc.yield` is a special terminator operation for block inside regions in
    acc ops (parallel and loop). It returns values to the immediately enclosing
    acc op.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// 2.13 Declare Directive
//===----------------------------------------------------------------------===//

def OpenACC_DeclareEnterOp : OpenACC_Op<"declare_enter", []> {
  let summary = "declare directive - entry to implicit data region";

  let description = [{
    The "acc.declare_enter" operation represents the OpenACC declare directive
    and captures the entry semantics to the implicit data region.
    This operation is modeled similarly to "acc.enter_data".

    Example showing `acc declare create(a)`:

    ```mlir
    %0 = acc.create varPtr(%a : !llvm.ptr<f32>) -> !llvm.ptr<f32>
    acc.declare_enter dataOperands(%0 : !llvm.ptr<f32>)
    ```
  }];

  let arguments = (ins Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands);

  let assemblyFormat = [{
    oilist(
        `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
    )
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}

def OpenACC_DeclareExitOp : OpenACC_Op<"declare_exit", []> {
  let summary = "declare directive - exit from implicit data region";

  let description = [{
    The "acc.declare_exit" operation represents the OpenACC declare directive
    and captures the exit semantics from the implicit data region.
    This operation is modeled similarly to "acc.exit_data".

    Example showing `acc declare device_resident(a)`:

    ```mlir
    %0 = acc.getdeviceptr varPtr(%a : !llvm.ptr<f32>) -> !llvm.ptr<f32> {dataClause = #acc<data_clause declare_device_resident>}
    acc.declare_exit dataOperands(%0 : !llvm.ptr<f32>)
    acc.delete accPtr(%0 : !llvm.ptr<f32>) {dataClause = #acc<data_clause declare_device_resident>}
    ```
  }];

  let arguments = (ins Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands);

  let assemblyFormat = [{
    oilist(
        `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
    )
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}

def OpenACC_GlobalConstructorOp : OpenACC_Op<"global_ctor",
                                             [IsolatedFromAbove, Symbol]> {
  let summary = "Used to hold construction operations associated with globals such as declare";

  let description = [{
    The "acc.global_ctor" operation is used to capture OpenACC actions to apply
    on globals (such as `acc declare`) at the entry to the implicit data region.
    This operation is isolated and intended to be used in a module.

    Example showing `declare create` of global:

    ```mlir
    llvm.mlir.global external @globalvar() : i32 {
      %0 = llvm.mlir.constant(0 : i32) : i32
      llvm.return %0 : i32
    }
    acc.global_ctor @acc_constructor {
      %0 = llvm.mlir.addressof @globalvar : !llvm.ptr<i32>
      %1 = acc.create varPtr(%0 : !llvm.ptr<i32>) -> !llvm.ptr<i32>
      acc.declare_enter dataOperands(%1 : !llvm.ptr<i32>)
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    $sym_name $region attr-dict-with-keyword
  }];

  let hasVerifier = 0;
}

def OpenACC_GlobalDestructorOp : OpenACC_Op<"global_dtor",
                                            [IsolatedFromAbove, Symbol]> {
  let summary = "Used to hold destruction operations associated with globals such as declare";

  let description = [{
    The "acc.global_dtor" operation is used to capture OpenACC actions to apply
    on globals (such as `acc declare`) at the exit from the implicit data
    region. This operation is isolated and intended to be used in a module.

    Example showing delete associated with `declare create` of global:

    ```mlir
    llvm.mlir.global external @globalvar() : i32 {
      %0 = llvm.mlir.constant(0 : i32) : i32
      llvm.return %0 : i32
    }
    acc.global_dtor @acc_destructor {
      %0 = llvm.mlir.addressof @globalvar : !llvm.ptr<i32>
      %1 = acc.getdeviceptr varPtr(%0 : !llvm.ptr<i32>) -> !llvm.ptr<i32> {dataClause = #acc<data_clause create>}
      acc.declare_exit dataOperands(%1 : !llvm.ptr<i32>)
      acc.delete accPtr(%1 : !llvm.ptr<i32>) {dataClause = #acc<data_clause create>}
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
    $sym_name $region attr-dict-with-keyword
  }];

  let hasVerifier = 0;
}

//===----------------------------------------------------------------------===//
// 2.14.1. Init Directive
//===----------------------------------------------------------------------===//

def OpenACC_InitOp : OpenACC_Op<"init", [AttrSizedOperandSegments]> {
  let summary = "init operation";

  let description = [{
    The "acc.init" operation represents the OpenACC init executable
    directive.

    Example:

    ```mlir
    acc.init
    acc.init device_num(%dev1 : i32)
    ```
  }];

  let arguments = (ins Variadic<AnyInteger>:$deviceTypeOperands,
                       Optional<IntOrIndex>:$deviceNumOperand,
                       Optional<I1>:$ifCond);

  let assemblyFormat = [{
    oilist(
        `device_type` `(` $deviceTypeOperands `:` type($deviceTypeOperands) `)`
      | `device_num` `(` $deviceNumOperand `:` type($deviceNumOperand) `)`
      | `if` `(` $ifCond `)`
    ) attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.14.2. Shutdown
//===----------------------------------------------------------------------===//

def OpenACC_ShutdownOp : OpenACC_Op<"shutdown", [AttrSizedOperandSegments]> {
  let summary = "shutdown operation";

  let description = [{
    The "acc.shutdown" operation represents the OpenACC shutdown executable
    directive.

    Example:

    ```mlir
    acc.shutdown
    acc.shutdown device_num(%dev1 : i32)
    ```
  }];

  let arguments = (ins Variadic<AnyInteger>:$deviceTypeOperands,
                       Optional<IntOrIndex>:$deviceNumOperand,
                       Optional<I1>:$ifCond);

  let assemblyFormat = [{
    oilist(`device_type` `(` $deviceTypeOperands `:` type($deviceTypeOperands) `)`
    |`device_num` `(` $deviceNumOperand `:` type($deviceNumOperand) `)`
    |`if` `(` $ifCond `)`
    ) attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.14.4. Update Directive
//===----------------------------------------------------------------------===//

def OpenACC_UpdateOp : OpenACC_Op<"update", [AttrSizedOperandSegments]> {
  let summary = "update operation";

  let description = [{
    The "acc.udpate" operation represents the OpenACC update executable
    directive.
    As host and self clauses are synonyms, any operands for host and self are
    add to $hostOperands.

    Example:

    ```mlir
    acc.update device(%d1 : memref<10xf32>) attributes {async}
    ```
  }];

  let arguments = (ins Optional<I1>:$ifCond,
                       Optional<IntOrIndex>:$asyncOperand,
                       Optional<IntOrIndex>:$waitDevnum,
                       Variadic<IntOrIndex>:$waitOperands,
                       UnitAttr:$async,
                       UnitAttr:$wait,
                       Variadic<IntOrIndex>:$deviceTypeOperands,
                       Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands,
                       UnitAttr:$ifPresent);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);
  }];

  let assemblyFormat = [{
    oilist(
        `if` `(` $ifCond `)`
      | `async` `(` $asyncOperand `:` type($asyncOperand) `)`
      | `wait_devnum` `(` $waitDevnum `:` type($waitDevnum) `)`
      | `device_type` `(` $deviceTypeOperands `:`
          type($deviceTypeOperands) `)`
      | `wait` `(` $waitOperands `:` type($waitOperands) `)`
      | `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
    )
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.16.3. Wait Directive
//===----------------------------------------------------------------------===//

def OpenACC_WaitOp : OpenACC_Op<"wait", [AttrSizedOperandSegments]> {
  let summary = "wait operation";

  let description = [{
    The "acc.wait" operation represents the OpenACC wait executable
    directive.

    Example:

    ```mlir
    acc.wait(%value1: index)
    acc.wait() async(%async1: i32)
    ```
  }];

  let arguments = (ins Variadic<IntOrIndex>:$waitOperands,
                       Optional<IntOrIndex>:$asyncOperand,
                       Optional<IntOrIndex>:$waitDevnum,
                       UnitAttr:$async,
                       Optional<I1>:$ifCond);

  let assemblyFormat = [{
    ( `(` $waitOperands^ `:` type($waitOperands) `)` )?
    oilist(`async` `(` $asyncOperand `:` type($asyncOperand) `)`
      |`wait_devnum` `(` $waitDevnum `:` type($waitDevnum) `)`
      |`if` `(` $ifCond `)`
    ) attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

#endif // OPENACC_OPS
