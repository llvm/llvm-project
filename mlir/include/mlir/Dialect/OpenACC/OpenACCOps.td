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
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/OpenACC/OpenACCBase.td"
include "mlir/Dialect/OpenACC/OpenACCOpsTypes.td"
include "mlir/Dialect/OpenACC/OpenACCOpsInterfaces.td"
include "mlir/Dialect/OpenACC/OpenACCTypeInterfaces.td"
include "mlir/Dialect/OpenACCMPCommon/Interfaces/AtomicInterfaces.td"
include "mlir/Dialect/OpenACCMPCommon/Interfaces/OpenACCMPOpsInterfaces.td"

// AccCommon requires definition of OpenACC_Dialect.
include "mlir/Dialect/OpenACC/AccCommon.td"

// Base class for OpenACC dialect ops.
class OpenACC_Op<string mnemonic, list<Trait> traits = []> :
  Op<OpenACC_Dialect, mnemonic, traits>;

// Reduction operation enumeration.
def OpenACC_ReductionOperatorNone    : I32EnumAttrCase<"AccNone", 0, "none">;
def OpenACC_ReductionOperatorAdd     : I32EnumAttrCase<"AccAdd", 1, "add">;
def OpenACC_ReductionOperatorMul     : I32EnumAttrCase<"AccMul", 2, "mul">;
def OpenACC_ReductionOperatorMax     : I32EnumAttrCase<"AccMax", 3, "max">;
def OpenACC_ReductionOperatorMin     : I32EnumAttrCase<"AccMin", 4, "min">;
def OpenACC_ReductionOperatorAnd     : I32EnumAttrCase<"AccIand", 5, "iand">;
def OpenACC_ReductionOperatorOr      : I32EnumAttrCase<"AccIor", 6, "ior">;
def OpenACC_ReductionOperatorXor     : I32EnumAttrCase<"AccXor", 7, "xor">;
def OpenACC_ReductionOperatorLogEqv  : I32EnumAttrCase<"AccEqv", 8, "eqv">;
def OpenACC_ReductionOperatorLogNeqv : I32EnumAttrCase<"AccNeqv", 9, "neqv">;
def OpenACC_ReductionOperatorLogAnd  : I32EnumAttrCase<"AccLand", 10, "land">;
def OpenACC_ReductionOperatorLogOr   : I32EnumAttrCase<"AccLor", 11, "lor">;

def OpenACC_ReductionOperator : I32EnumAttr<"ReductionOperator",
    "built-in reduction operations supported by OpenACC",
    [OpenACC_ReductionOperatorNone, OpenACC_ReductionOperatorAdd, 
     OpenACC_ReductionOperatorMul, OpenACC_ReductionOperatorMax, OpenACC_ReductionOperatorMin,
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

// OpenACC variable type categorization. This is needed because OpenACC
// dialect is used with other dialects, and each dialect defines its own
// types. Thus, in order to be able to classify types and apply right semantics,
// it is needed to ensure the types can be categorized.
def OpenACC_VariableTypeUncategorized : I32BitEnumAttrCaseNone<"uncategorized">;

// The OpenACC spec definition of scalar type is as follows (from 3.3 spec,
// line 5454):
// Scalar datatype - an intrinsic or built-in datatype that is not an array or
// aggregate datatype. In Fortran, scalar datatypes are integer, real, double
// precision, complex, or logical. In C, scalar datatypes are char (signed or
// unsigned), int (signed or unsigned, with optional short, long or long long
// attribute), enum, float, double, long double, Complex (with optional float
// or long attribute), or any pointer datatype. In C++, scalar datatypes are
// char (signed or unsigned), wchar t, int (signed or unsigned, with optional
// short, long or long long attribute), enum, bool, float, double, long double,
// or any pointer datatype. Not all implementations or targets will support all
// of these datatypes.
// From an MLIR type perspective, the types that those language types map to
// will be categorized as scalar.
def OpenACC_VariableTypeScalar : I32BitEnumAttrCaseBit<"scalar", 0>;

// Not in OpenACC spec glossary as its own definition but used throughout the
// spec. One definition of array that can be assumed for purposes of type
// categorization is that it is a collection of elements of same type.
def OpenACC_VariableTypeArray : I32BitEnumAttrCaseBit<"array", 1>;

// The OpenACC spec definition of composite type is as follows (from 3.3 spec,
// line 5354):
// Composite datatype - a derived type in Fortran, or a struct or union type in
// C, or a class, struct, or union type in C++. (This is different from the use
// of the term composite data type in the C and C++ languages.)
def OpenACC_VariableTypeComposite : I32BitEnumAttrCaseBit<"composite", 2>;

// The OpenACC spec uses the type category "aggregate" to capture both arrays
// and composite types. However, it includes types which do not fall in either
// of those categories. Thus create a case for the others.
// For example, reading the definition of "Aggregate Variables" in the 3.3
// spec line 5346 shows this distinction:
// Aggregate variables - a variable of any non-scalar datatype, including array
// or composite variables. In Fortran, this includes any variable with
// allocatable or pointer attribute and character variables
def OpenACC_VariableTypeOtherNonScalar : I32BitEnumAttrCaseBit<"nonscalar", 3>;

// The OpenACC spec definition of aggregate type is as follows (from 3.3 spec,
// line 5342):
// Aggregate datatype - any non-scalar datatype such as array and composite
// datatypes. In Fortran, aggregate datatypes include arrays, derived types,
// character types. In C, aggregate datatypes include arrays, targets of
// pointers, structs, and unions. In C++, aggregate datatypes include arrays,
// targets of pointers, classes, structs, and unions.
def OpenACC_VariableTypeAggregate : I32BitEnumAttrCaseGroup<"aggregate",
  [OpenACC_VariableTypeArray, OpenACC_VariableTypeComposite,
  OpenACC_VariableTypeOtherNonScalar]>;

def OpenACC_VariableTypeCategory : I32BitEnumAttr<
    "VariableTypeCategory",
    "Captures different type categories described in OpenACC spec",
    [
      OpenACC_VariableTypeUncategorized, OpenACC_VariableTypeScalar,
      OpenACC_VariableTypeArray, OpenACC_VariableTypeComposite,
      OpenACC_VariableTypeOtherNonScalar, OpenACC_VariableTypeAggregate]> {
  let separator = ",";
  let cppNamespace = "::mlir::acc";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

// Type used in operation below.
def IntOrIndex : AnyTypeOf<[AnyInteger, Index]>;

// Simple alias to pointer-like interface to reduce verbosity.
def OpenACC_PointerLikeType : TypeAlias<OpenACC_PointerLikeTypeInterface,
    "pointer-like type">;
def OpenACC_MappableType : TypeAlias<OpenACC_MappableTypeInterface,
    "mappable type">;

def OpenACC_AnyPointerOrMappableLike : TypeConstraint<Or<[OpenACC_PointerLikeType.predicate,
    OpenACC_MappableType.predicate]>, "any pointer or mappable">;
def OpenACC_AnyPointerOrMappableType : Type<OpenACC_AnyPointerOrMappableLike.predicate,
    "any pointer or mappable">;

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
def OpenACC_Cache                 : I64EnumAttrCase<"acc_cache", 24>;
def OpenACC_CacheReadonly         : I64EnumAttrCase<"acc_cache_readonly", 25>;

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
     OpenACC_Cache, OpenACC_CacheReadonly,
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
  let parameters = (ins "DataClauseAttr":$dataClause,
                        DefaultValuedParameter<"bool", "false">:$implicit);
  let assemblyFormat = "`<` struct(params) `>`";
  let builders = [AttrBuilder<(ins "DataClauseAttr":$dataClause), [{
      return $_get($_ctxt, dataClause, /*implicit=*/false);
    }]>
  ];
}

// Attribute to attach functions that perform the pre/post allocation actions or
// pre/post deallocation actions as described in section 2.13.
def DeclareActionAttr : OpenACC_Attr<"DeclareAction", "declare_action"> {
  let parameters = (ins OptionalParameter<"SymbolRefAttr">:$preAlloc,
                        OptionalParameter<"SymbolRefAttr">:$postAlloc,
                        OptionalParameter<"SymbolRefAttr">:$preDealloc,
                        OptionalParameter<"SymbolRefAttr">:$postDealloc);
  let assemblyFormat = "`<` struct(params) `>`";
}

// Device type enumeration.
def OpenACC_DeviceTypeNone      : I32EnumAttrCase<"None", 0, "none">;
def OpenACC_DeviceTypeStar      : I32EnumAttrCase<"Star", 1, "star">;
def OpenACC_DeviceTypeDefault   : I32EnumAttrCase<"Default", 2, "default">;
def OpenACC_DeviceTypeHost      : I32EnumAttrCase<"Host", 3, "host">;
def OpenACC_DeviceTypeMulticore : I32EnumAttrCase<"Multicore", 4, "multicore">;
def OpenACC_DeviceTypeNvidia    : I32EnumAttrCase<"Nvidia", 5, "nvidia">;
def OpenACC_DeviceTypeRadeon    : I32EnumAttrCase<"Radeon", 6, "radeon">;

def OpenACC_DeviceType : I32EnumAttr<"DeviceType",
    "built-in device type supported by OpenACC",
    [OpenACC_DeviceTypeNone, OpenACC_DeviceTypeStar, OpenACC_DeviceTypeDefault,
     OpenACC_DeviceTypeHost, OpenACC_DeviceTypeMulticore,
     OpenACC_DeviceTypeNvidia, OpenACC_DeviceTypeRadeon
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::acc";
}

// Device type attribute is used to associate a value for for clauses that
// appear after a device_type clause. The list of clauses allowed after the
// device_type clause is defined per construct as follows:
// Loop construct: collapse, gang, worker, vector, seq, independent, auto,
//                 and tile
// Compute construct: async, wait, num_gangs, num_workers, and vector_length
// Data construct: async and wait 
// Routine: gang, worker, vector, seq and bind
//
// The `none` means that the value appears before any device_type clause.
//
def OpenACC_DeviceTypeAttr : EnumAttr<OpenACC_Dialect,
                                      OpenACC_DeviceType,
                                      "device_type"> {
  let assemblyFormat = [{ ```<` $value `>` }];
}

def DeviceTypeArrayAttr :
  TypedArrayAttrBase<OpenACC_DeviceTypeAttr, "device type array attribute"> {
  let constBuilderCall = ?;
}

// Gang arg type enumeration
def OpenACC_GangArgNum      : I32EnumAttrCase<"Num", 0, "Num">;
def OpenACC_GangArgDim      : I32EnumAttrCase<"Dim", 1, "Dim">;
def OpenACC_GangArgStatic   : I32EnumAttrCase<"Static", 2, "Static">;

def OpenACC_GangArgType : I32EnumAttr<"GangArgType",
    "Differentiate the different gang arg values",
    [OpenACC_GangArgNum, OpenACC_GangArgDim, OpenACC_GangArgStatic]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::acc";
}
def OpenACC_GangArgTypeAttr : EnumAttr<OpenACC_Dialect,
                                       OpenACC_GangArgType,
                                       "gang_arg_type"> {
  let assemblyFormat = [{ ```<` $value `>` }];
}
def GangArgTypeArrayAttr :
  TypedArrayAttrBase<OpenACC_GangArgTypeAttr, "gang arg type array attribute"> {
  let constBuilderCall = ?;
}

// Combined constructs enumerations
def OpenACC_KernelsLoop    : I32EnumAttrCase<"KernelsLoop", 1, "kernels_loop">;
def OpenACC_ParallelLoop   : I32EnumAttrCase<"ParallelLoop", 2, "parallel_loop">;
def OpenACC_SerialLoop     : I32EnumAttrCase<"SerialLoop", 3, "serial_loop">;

def OpenACC_CombinedConstructsType : I32EnumAttr<"CombinedConstructsType",
    "Differentiate between combined constructs",
    [OpenACC_KernelsLoop, OpenACC_ParallelLoop, OpenACC_SerialLoop]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::acc";
}

def OpenACC_CombinedConstructsAttr : EnumAttr<OpenACC_Dialect,
                                       OpenACC_CombinedConstructsType,
                                       "combined_constructs"> {
  let assemblyFormat = [{ ```<` $value `>` }];
}

def OpenACC_ParallelConstruct : I64EnumAttrCase<"acc_construct_parallel", 0>;
def OpenACC_KernelsConstruct : I64EnumAttrCase<"acc_construct_kernels", 1>;
def OpenACC_LoopConstruct : I64EnumAttrCase<"acc_construct_loop", 2>;
def OpenACC_DataConstruct : I64EnumAttrCase<"acc_construct_data", 3>;
def OpenACC_EnterDataConstruct : I64EnumAttrCase<"acc_construct_enter_data", 4>;
def OpenACC_ExitDataConstruct : I64EnumAttrCase<"acc_construct_exit_data", 5>;
def OpenACC_HostDataConstruct : I64EnumAttrCase<"acc_construct_host_data", 6>;
def OpenACC_AtomicConstruct : I64EnumAttrCase<"acc_construct_atomic", 7>;
def OpenACC_DeclareConstruct : I64EnumAttrCase<"acc_construct_declare", 8>;
def OpenACC_InitConstruct : I64EnumAttrCase<"acc_construct_init", 9>;
def OpenACC_ShutdownConstruct : I64EnumAttrCase<"acc_construct_shutdown", 10>;
def OpenACC_SetConstruct : I64EnumAttrCase<"acc_construct_set", 11>;
def OpenACC_UpdateConstruct : I64EnumAttrCase<"acc_construct_update", 12>;
def OpenACC_RoutineConstruct : I64EnumAttrCase<"acc_construct_routine", 13>;
def OpenACC_WaitConstruct : I64EnumAttrCase<"acc_construct_wait", 14>;
def OpenACC_RuntimeAPIConstruct : I64EnumAttrCase<"acc_construct_runtime_api", 15>;
def OpenACC_SerialConstruct : I64EnumAttrCase<"acc_construct_serial", 16>;

def OpenACC_ConstructEnum : I64EnumAttr<"Construct",
    "constructs supported by OpenACC",
    [OpenACC_ParallelConstruct, OpenACC_KernelsConstruct,
     OpenACC_LoopConstruct, OpenACC_DataConstruct,
     OpenACC_EnterDataConstruct, OpenACC_ExitDataConstruct,
     OpenACC_HostDataConstruct, OpenACC_AtomicConstruct,
     OpenACC_DeclareConstruct, OpenACC_InitConstruct,
     OpenACC_ShutdownConstruct, OpenACC_SetConstruct,
     OpenACC_UpdateConstruct, OpenACC_RoutineConstruct,
     OpenACC_WaitConstruct, OpenACC_RuntimeAPIConstruct,
     OpenACC_SerialConstruct
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::acc";
}

def OpenACC_ConstructAttr : EnumAttr<OpenACC_Dialect, OpenACC_ConstructEnum,
                                     "construct">;

// Define a resource for the OpenACC runtime counters.
def OpenACC_RuntimeCounters : Resource<"::mlir::acc::RuntimeCounters">;

// Define a resource for the OpenACC constructs.
// Useful to ensure that the constructs are not removed (even though
// the data semantics are encoded in the operations linked via their
// `dataOperands` list).
def OpenACC_ConstructResource : Resource<"::mlir::acc::ConstructResource">;

// Define a resource for the OpenACC current device setting.
def OpenACC_CurrentDeviceIdResource : Resource<"::mlir::acc::CurrentDeviceIdResource">;

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

    The `stride` represents the distance between consecutive elements. For
    multi-dimensional arrays, the `stride` for each outer dimension must account
    for the complete size of all inner dimensions.

    The `strideInBytes` flag indicates that the `stride` is specified in bytes
    rather than the number of elements.

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
    acc.bounds lb(1) ub(9) extent(9) startIdx(0) stride(1)
    ```

    Fortran:
    ```
    integer :: array(1:10)
    !$acc copy(array(2:10))
    ```
    =>
    ```mlir
    acc.bounds lb(1) ub(9) extent(9) startIdx(1) stride(1)
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

  let builders = [
    OpBuilder<(ins "::mlir::Value":$extent), [{
        build($_builder, $_state,
          ::mlir::acc::DataBoundsType::get($_builder.getContext()),
          /*lowerbound=*/{}, /*upperbound=*/{}, extent,
          /*stride=*/{}, /*strideInBytes=*/$_builder.getBoolAttr(false),
          /*startIdx=*/{});
      }]
    >,
    OpBuilder<(ins "::mlir::Value":$lowerbound,
                   "::mlir::Value":$upperbound), [{
        build($_builder, $_state,
          ::mlir::acc::DataBoundsType::get($_builder.getContext()),
          lowerbound, upperbound, /*extent=*/{},
          /*stride=*/{}, /*strideInBytes=*/$_builder.getBoolAttr(false),
          /*startIdx=*/{});
      }]
    >
  ];
}

// Data entry operation does not refer to OpenACC spec terminology, but to
// terminology used in this dialect. It refers to data operations that will
// appear before data or compute region. It will be used as the base of acc
// dialect operations for the following OpenACC data clauses: copyin, create,
// present, attach, deviceptr.
//
// The bounds are represented in rank order. Rank 0 (inner-most dimension) is
// the first.
//
class OpenACC_DataEntryOp<string mnemonic, string clause, string extraDescription,
                          list<Trait> traits = [], dag additionalArgs = (ins)> :
    OpenACC_Op<mnemonic, !listconcat(traits,
        [AttrSizedOperandSegments,
         MemoryEffects<[MemRead<OpenACC_CurrentDeviceIdResource>]>])> {
  let arguments = !con(
      additionalArgs,
      (ins TypeAttr:$varType,
          Optional<OpenACC_PointerLikeTypeInterface>:$varPtrPtr,
          Variadic<OpenACC_DataBoundsType>:$bounds, /* rank-0 to rank-{n-1} */
          Variadic<IntOrIndex>:$asyncOperands,
          OptionalAttr<DeviceTypeArrayAttr>:$asyncOperandsDeviceType,
          OptionalAttr<DeviceTypeArrayAttr>:$asyncOnly,
          DefaultValuedAttr<OpenACC_DataClauseAttr, clause>:$dataClause,
          DefaultValuedAttr<BoolAttr, "true">:$structured,
          DefaultValuedAttr<BoolAttr, "false">:$implicit,
          OptionalAttr<StrAttr>:$name));

  let description = !strconcat(extraDescription, [{
    Description of arguments:
    - `var`: The variable to copy. Must be either `MappableType` or
    `PointerLikeType`.
    - `varType`: The type of the variable that is being copied. When `var` is
    a `MappableType`, this matches the type of `var`. When `var` is a
    `PointerLikeType`, this type holds information about the target of the
    pointer.
    - `varPtrPtr`: Specifies the address of the address of `var` - only used
    when the variable copied is a field in a struct. This is important for
    OpenACC due to implicit attach semantics on data clauses (2.6.4).
    - `bounds`: Used when copying just slice of array or array's bounds are not
    encoded in type. They are in rank order where rank 0 is inner-most dimension.
    - `asyncOperands` and `asyncOperandsDeviceType`:
    pair-wise lists of the async clause values associated with device_type's.
    - `asyncOnly`: a list of device_type's for which async clause
    does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
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

    The async values attached to the data entry operation imply that the data
    action applies to all device types specified by the device_type clauses
    using the activity queues on these devices as defined by the async values.
  }]);

  code extraClassDeclarationBase = [{
    /// Return true if the op has the async attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasAsyncOnly() {
      return hasAsyncOnly(mlir::acc::DeviceType::None);
    }
    /// Return true if the op has the async attribute for the given device_type.
    bool hasAsyncOnly(mlir::acc::DeviceType deviceType) {
      mlir::ArrayAttr asyncOnly = getAsyncOnlyAttr();
      if (!asyncOnly)
        return false;
      for (auto attr : asyncOnly) {
        auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(attr);
        if (deviceTypeAttr.getValue() == deviceType)
          return true;
      }
      return false;
    }
    /// Return the value of the async clause if present.
    mlir::Value getAsyncValue() {
      return getAsyncValue(mlir::acc::DeviceType::None);
    }
    /// Return the value of the async clause for the given device_type if
    /// present.
    mlir::Value getAsyncValue(mlir::acc::DeviceType deviceType) {
      mlir::ArrayAttr deviceTypes = getAsyncOperandsDeviceTypeAttr();
      if (!deviceTypes)
        return nullptr;
      for (auto [attr, asyncValue] :
          llvm::zip(deviceTypes, getAsyncOperands())) {
        auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(attr);
        if (deviceTypeAttr.getValue() == deviceType)
          return asyncValue;
      }
      return nullptr;
    }
    mlir::TypedValue<mlir::acc::PointerLikeType> getVarPtr() {
      return mlir::dyn_cast<mlir::TypedValue<mlir::acc::PointerLikeType>>(getVar());
    }
    mlir::TypedValue<mlir::acc::PointerLikeType> getAccPtr() {
      return mlir::dyn_cast<mlir::TypedValue<mlir::acc::PointerLikeType>>(getAccVar());
    }
  }];

  let assemblyFormat = [{
    custom<Var>($var) `:` custom<VarPtrType>(type($var), $varType)
    oilist(
        `varPtrPtr` `(` $varPtrPtr `:` type($varPtrPtr) `)`
      | `bounds` `(` $bounds `)`
      | `async` `` custom<DeviceTypeOperandsWithKeywordOnly>($asyncOperands,
            type($asyncOperands), $asyncOperandsDeviceType, $asyncOnly)
    ) `->` type($accVar) attr-dict
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "::mlir::Value":$var,
                   "bool":$structured, "bool":$implicit,
                   CArg<"::mlir::ValueRange", "{}">:$bounds),
      [{
        auto ptrLikeTy = ::mlir::dyn_cast<::mlir::acc::PointerLikeType>(
          var.getType());
        build($_builder, $_state, var.getType(), var,
          /*varType=*/ptrLikeTy ?
            ::mlir::TypeAttr::get(ptrLikeTy.getElementType()) :
            ::mlir::TypeAttr::get(var.getType()),
          /*varPtrPtr=*/{}, bounds, /*asyncOperands=*/{},
          /*asyncOperandsDeviceType=*/nullptr,
          /*asyncOnly=*/nullptr, /*dataClause=*/nullptr,
          /*structured=*/$_builder.getBoolAttr(structured),
          /*implicit=*/$_builder.getBoolAttr(implicit), /*name=*/nullptr);
      }]>,
    OpBuilder<(ins "::mlir::Value":$var,
                   "bool":$structured, "bool":$implicit,
                   "const ::llvm::Twine &":$name,
                  CArg<"::mlir::ValueRange", "{}">:$bounds),
      [{
        auto ptrLikeTy = ::mlir::dyn_cast<::mlir::acc::PointerLikeType>(
          var.getType());
        build($_builder, $_state, var.getType(), var,
          /*varType=*/ptrLikeTy ?
            ::mlir::TypeAttr::get(ptrLikeTy.getElementType()) :
            ::mlir::TypeAttr::get(var.getType()),
          /*varPtrPtr=*/{}, bounds, /*asyncOperands=*/{},
          /*asyncOperandsDeviceType=*/nullptr,
          /*asyncOnly=*/nullptr, /*dataClause=*/nullptr,
          /*structured=*/$_builder.getBoolAttr(structured),
          /*implicit=*/$_builder.getBoolAttr(implicit),
          /*name=*/$_builder.getStringAttr(name));
      }]>];
}

//===----------------------------------------------------------------------===//
// 2.5.13 private clause
//===----------------------------------------------------------------------===//
def OpenACC_PrivateOp : OpenACC_DataEntryOp<"private",
    "mlir::acc::DataClause::acc_private", "", [],
    (ins OpenACC_AnyPointerOrMappableType:$var)> {
  let summary = "Represents private semantics for acc private clause.";
  let results = (outs Arg<OpenACC_AnyPointerOrMappableType,
                          "Accelerator mapped variable",[MemWrite]>:$accVar);
  let extraClassDeclaration = extraClassDeclarationBase;
}

//===----------------------------------------------------------------------===//
// 2.5.14 firstprivate clause
//===----------------------------------------------------------------------===//
def OpenACC_FirstprivateOp : OpenACC_DataEntryOp<"firstprivate",
    "mlir::acc::DataClause::acc_firstprivate", "", [],
    (ins Arg<OpenACC_AnyPointerOrMappableType,"Host variable",[MemRead]>:$var)> {
  let summary = "Represents firstprivate semantic for the acc firstprivate "
                "clause.";
  let results = (outs Arg<OpenACC_AnyPointerOrMappableType,
                          "Accelerator mapped variable",[MemWrite]>:$accVar);
  let extraClassDeclaration = extraClassDeclarationBase;
}

//===----------------------------------------------------------------------===//
// 2.5.15 reduction clause
//===----------------------------------------------------------------------===//
def OpenACC_ReductionOp : OpenACC_DataEntryOp<"reduction",
    "mlir::acc::DataClause::acc_reduction", "", [],
    (ins Arg<OpenACC_AnyPointerOrMappableType,"Host variable",[MemRead]>:$var)> {
  let summary = "Represents reduction semantics for acc reduction clause.";
  let results = (outs Arg<OpenACC_AnyPointerOrMappableType,
                          "Accelerator mapped variable",[MemWrite]>:$accVar);
  let extraClassDeclaration = extraClassDeclarationBase;
}

//===----------------------------------------------------------------------===//
// 2.7.4 deviceptr clause
//===----------------------------------------------------------------------===//
def OpenACC_DevicePtrOp : OpenACC_DataEntryOp<"deviceptr",
    "mlir::acc::DataClause::acc_deviceptr", "",
    [MemoryEffects<[MemRead<OpenACC_RuntimeCounters>]>],
    (ins OpenACC_AnyPointerOrMappableType:$var)> {
  let summary = "Specifies that the variable pointer is a device pointer.";
  let results = (outs OpenACC_AnyPointerOrMappableType:$accVar);
  let extraClassDeclaration = extraClassDeclarationBase;
}

//===----------------------------------------------------------------------===//
// 2.7.5 present clause
//===----------------------------------------------------------------------===//
def OpenACC_PresentOp : OpenACC_DataEntryOp<"present",
    "mlir::acc::DataClause::acc_present", "",
    [MemoryEffects<[MemRead<OpenACC_RuntimeCounters>,
                    MemWrite<OpenACC_RuntimeCounters>]>],
    (ins OpenACC_AnyPointerOrMappableType:$var)> {
  let summary = "Specifies that the variable is already present on device.";
  let results = (outs OpenACC_AnyPointerOrMappableType:$accVar);
  let extraClassDeclaration = extraClassDeclarationBase;
}

//===----------------------------------------------------------------------===//
// 2.7.7 copyin clause
//===----------------------------------------------------------------------===//
def OpenACC_CopyinOp : OpenACC_DataEntryOp<"copyin",
    "mlir::acc::DataClause::acc_copyin", "",
    [MemoryEffects<[MemRead<OpenACC_RuntimeCounters>,
                    MemWrite<OpenACC_RuntimeCounters>]>],
    (ins Arg<OpenACC_AnyPointerOrMappableType,"Host variable",[MemRead]>:$var)> {
  let summary = "Represents copyin semantics for acc data clauses like acc "
                "copyin and acc copy.";
  let results = (outs Arg<OpenACC_AnyPointerOrMappableType,
                          "Accelerator mapped variable",[MemWrite]>:$accVar);

  let extraClassDeclaration = extraClassDeclarationBase # [{
    /// Check if this is a copyin with readonly modifier.
    bool isCopyinReadonly();
  }];
}

//===----------------------------------------------------------------------===//
// 2.7.9 create clause
//===----------------------------------------------------------------------===//
def OpenACC_CreateOp : OpenACC_DataEntryOp<"create",
    "mlir::acc::DataClause::acc_create", "",
    [MemoryEffects<[MemRead<OpenACC_RuntimeCounters>,
                    MemWrite<OpenACC_RuntimeCounters>]>],
    (ins OpenACC_AnyPointerOrMappableType:$var)> {
  let summary = "Represents create semantics for acc data clauses like acc "
                "create and acc copyout.";
  let results = (outs Arg<OpenACC_AnyPointerOrMappableType,
                          "Accelerator mapped variable",[MemWrite]>:$accVar);

  let extraClassDeclaration = extraClassDeclarationBase # [{
    /// Check if this is a create with zero modifier.
    bool isCreateZero();
  }];
}

//===----------------------------------------------------------------------===//
// 2.7.10 no_create clause
//===----------------------------------------------------------------------===//
def OpenACC_NoCreateOp : OpenACC_DataEntryOp<"nocreate",
    "mlir::acc::DataClause::acc_no_create", "",
    [MemoryEffects<[MemRead<OpenACC_RuntimeCounters>,
                    MemWrite<OpenACC_RuntimeCounters>]>],
    (ins OpenACC_AnyPointerOrMappableType:$var)> {
  let summary = "Represents acc no_create semantics.";
  let results = (outs OpenACC_AnyPointerOrMappableType:$accVar);
  let extraClassDeclaration = extraClassDeclarationBase;
}

//===----------------------------------------------------------------------===//
// 2.7.12 attach clause
//===----------------------------------------------------------------------===//
def OpenACC_AttachOp : OpenACC_DataEntryOp<"attach",
    "mlir::acc::DataClause::acc_attach", "",
    [MemoryEffects<[MemRead<OpenACC_RuntimeCounters>,
                    MemWrite<OpenACC_RuntimeCounters>]>],
    (ins Arg<OpenACC_AnyPointerOrMappableType,"Host variable",[MemRead]>:$var)> {
  let summary = "Represents acc attach semantics which updates a pointer in "
                "device memory with the corresponding device address of the "
                "pointee.";
  let results = (outs OpenACC_AnyPointerOrMappableType:$accVar);
  let extraClassDeclaration = extraClassDeclarationBase;
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
    }], [MemoryEffects<[MemRead<OpenACC_RuntimeCounters>]>],
    (ins OpenACC_AnyPointerOrMappableType:$var)> {
  let summary = "Gets device address if variable exists on device.";
  let results = (outs OpenACC_AnyPointerOrMappableType:$accVar);
  let hasVerifier = 0;
  let extraClassDeclaration = extraClassDeclarationBase;
}

//===----------------------------------------------------------------------===//
// 2.14.4 device clause
//===----------------------------------------------------------------------===//
def OpenACC_UpdateDeviceOp : OpenACC_DataEntryOp<"update_device",
    "mlir::acc::DataClause::acc_update_device", "", [],
    (ins Arg<OpenACC_AnyPointerOrMappableType,"Host variable",[MemRead]>:$var)> {
  let summary = "Represents acc update device semantics.";
  let results = (outs Arg<OpenACC_AnyPointerOrMappableType,
                          "Accelerator mapped variable",[MemWrite]>:$accVar);
  let extraClassDeclaration = extraClassDeclarationBase;
}

//===----------------------------------------------------------------------===//
// 2.8 use_device clause
//===----------------------------------------------------------------------===//
def OpenACC_UseDeviceOp : OpenACC_DataEntryOp<"use_device",
    "mlir::acc::DataClause::acc_use_device", "",
    [MemoryEffects<[MemRead<OpenACC_RuntimeCounters>]>],
    (ins OpenACC_AnyPointerOrMappableType:$var)> {
  let summary = "Represents acc use_device semantics.";
  let results = (outs OpenACC_AnyPointerOrMappableType:$accVar);
  let extraClassDeclaration = extraClassDeclarationBase;
}

//===----------------------------------------------------------------------===//
// 2.13.1 device_resident clause
//===----------------------------------------------------------------------===//
def OpenACC_DeclareDeviceResidentOp : OpenACC_DataEntryOp<"declare_device_resident",
    "mlir::acc::DataClause::acc_declare_device_resident", "",
    [MemoryEffects<[MemWrite<OpenACC_RuntimeCounters>]>],
    (ins Arg<OpenACC_AnyPointerOrMappableType,"Host variable",[MemRead]>:$var)> {
  let summary = "Represents acc declare device_resident semantics.";
  let results = (outs OpenACC_AnyPointerOrMappableType:$accVar);
  let extraClassDeclaration = extraClassDeclarationBase;
}

//===----------------------------------------------------------------------===//
// 2.13.3 link clause
//===----------------------------------------------------------------------===//
def OpenACC_DeclareLinkOp : OpenACC_DataEntryOp<"declare_link",
    "mlir::acc::DataClause::acc_declare_link", "",
    [MemoryEffects<[MemWrite<OpenACC_RuntimeCounters>]>],
    (ins Arg<OpenACC_AnyPointerOrMappableType,"Host variable",[MemRead]>:$var)> {
  let summary = "Represents acc declare link semantics.";
  let results = (outs OpenACC_AnyPointerOrMappableType:$accVar);
  let extraClassDeclaration = extraClassDeclarationBase;
}

//===----------------------------------------------------------------------===//
// 2.10 cache directive
//===----------------------------------------------------------------------===//
def OpenACC_CacheOp : OpenACC_DataEntryOp<"cache",
    "mlir::acc::DataClause::acc_cache", "", [NoMemoryEffect],
    (ins OpenACC_AnyPointerOrMappableType:$var)> {
  let summary = "Represents the cache directive that is associated with a "
                "loop.";
  let results = (outs OpenACC_AnyPointerOrMappableType:$accVar);

  let extraClassDeclaration = extraClassDeclarationBase # [{
    /// Check if this is a cache with readonly modifier.
    bool isCacheReadonly() {
      return getDataClause() == acc::DataClause::acc_cache_readonly;
    }
  }];
}

// Data exit operation does not refer to OpenACC spec terminology, but to
// terminology used in this dialect. It refers to data operations that will appear
// after data or compute region. It will be used as the base of acc dialect
// operations for the following OpenACC data clauses: copyout, detach, delete.
class OpenACC_DataExitOp<string mnemonic, string clause, string extraDescription,
                         list<Trait> traits = [], dag additionalArgs = (ins)> :
    OpenACC_Op<mnemonic, !listconcat(traits,
        [AttrSizedOperandSegments,
         MemoryEffects<[MemRead<OpenACC_CurrentDeviceIdResource>]>])> {
  let arguments = !con(additionalArgs,
                      (ins Variadic<OpenACC_DataBoundsType>:$bounds,
                       Variadic<IntOrIndex>:$asyncOperands,
                       OptionalAttr<DeviceTypeArrayAttr>:$asyncOperandsDeviceType,
                       OptionalAttr<DeviceTypeArrayAttr>:$asyncOnly,
                       DefaultValuedAttr<OpenACC_DataClauseAttr,clause>:$dataClause,
                       DefaultValuedAttr<BoolAttr, "true">:$structured,
                       DefaultValuedAttr<BoolAttr, "false">:$implicit,
                       OptionalAttr<StrAttr>:$name));

  let description = !strconcat(extraDescription, [{
    - `accVar`: The acc variable. This is the link from the data-entry
    operation used.
    - `bounds`: Used when copying just slice of array or array's bounds are not
    encoded in type. They are in rank order where rank 0 is inner-most dimension.
    - `asyncOperands` and `asyncOperandsDeviceType`:
    pair-wise lists of the async clause values associated with device_type's.
    - `asyncOnly`: a list of device_type's for which async clause
    does not specify a value (default is acc_async_noval - OpenACC 3.3 2.16.1).
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

    The async values attached to the data exit operation imply that the data
    action applies to all device types specified by the device_type clauses
    using the activity queues on these devices as defined by the async values.
  }]);

  code extraClassDeclarationBase = [{
    /// Return true if the op has the async attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasAsyncOnly() {
      return hasAsyncOnly(mlir::acc::DeviceType::None);
    }
    /// Return true if the op has the async attribute for the given device_type.
    bool hasAsyncOnly(mlir::acc::DeviceType deviceType) {
      mlir::ArrayAttr asyncOnly = getAsyncOnlyAttr();
      if (!asyncOnly)
        return false;
      for (auto attr : asyncOnly) {
        auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(attr);
        if (deviceTypeAttr.getValue() == deviceType)
          return true;
      }
      return false;
    }
    /// Return the value of the async clause if present.
    mlir::Value getAsyncValue() {
      return getAsyncValue(mlir::acc::DeviceType::None);
    }
    /// Return the value of the async clause for the given device_type if
    /// present.
    mlir::Value getAsyncValue(mlir::acc::DeviceType deviceType) {
      mlir::ArrayAttr deviceTypes = getAsyncOperandsDeviceTypeAttr();
      if (!deviceTypes)
        return nullptr;
      for (auto [attr, asyncValue] :
          llvm::zip(deviceTypes, getAsyncOperands())) {
        auto deviceTypeAttr = mlir::dyn_cast<mlir::acc::DeviceTypeAttr>(attr);
        if (deviceTypeAttr.getValue() == deviceType)
          return asyncValue;
      }
      return nullptr;
    }
  }];

  let hasVerifier = 1;
}

class OpenACC_DataExitOpWithVarPtr<string mnemonic, string clause>
    : OpenACC_DataExitOp<
          mnemonic, clause,
          "- `varPtr`: The address of variable to copy back to.",
          [MemoryEffects<[MemRead<OpenACC_RuntimeCounters>,
                          MemWrite<OpenACC_RuntimeCounters>]>],
          (ins Arg<OpenACC_AnyPointerOrMappableType,
                   "Accelerator mapped variable", [MemRead]>:$accVar,
              Arg<OpenACC_AnyPointerOrMappableType,
                  "Host variable", [MemWrite]>:$var,
              TypeAttr:$varType)> {
  let assemblyFormat = [{
    custom<AccVar>($accVar, type($accVar))
    (`bounds` `(` $bounds^ `)` )?
    (`async` `` custom<DeviceTypeOperandsWithKeywordOnly>($asyncOperands,
            type($asyncOperands), $asyncOperandsDeviceType, $asyncOnly)^)?
    `to` custom<Var>($var) `:` custom<VarPtrType>(type($var), $varType)
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Value":$accVar,
                   "::mlir::Value":$var,
                   "bool":$structured, "bool":$implicit,
                   CArg<"::mlir::ValueRange", "{}">:$bounds),
      [{
        auto ptrLikeTy = ::mlir::dyn_cast<::mlir::acc::PointerLikeType>(
          var.getType());
        build($_builder, $_state, accVar, var,
          /*varType=*/ptrLikeTy ?
            ::mlir::TypeAttr::get(ptrLikeTy.getElementType()) :
            ::mlir::TypeAttr::get(var.getType()),
          bounds, /*asyncOperands=*/{}, /*asyncOperandsDeviceType=*/nullptr,
          /*asyncOnly=*/nullptr, /*dataClause=*/nullptr,
          /*structured=*/$_builder.getBoolAttr(structured),
          /*implicit=*/$_builder.getBoolAttr(implicit), /*name=*/nullptr);
      }]>,
    OpBuilder<(ins "::mlir::Value":$accVar,
                   "::mlir::Value":$var,
                   "bool":$structured, "bool":$implicit,
                   "const ::llvm::Twine &":$name,
                   CArg<"::mlir::ValueRange", "{}">:$bounds),
      [{
        auto ptrLikeTy = ::mlir::dyn_cast<::mlir::acc::PointerLikeType>(
          var.getType());
        build($_builder, $_state, accVar, var,
          /*varType=*/ptrLikeTy ?
            ::mlir::TypeAttr::get(ptrLikeTy.getElementType()) :
            ::mlir::TypeAttr::get(var.getType()),
          bounds, /*asyncOperands=*/{}, /*asyncOperandsDeviceType=*/nullptr,
          /*asyncOnly=*/nullptr, /*dataClause=*/nullptr,
          /*structured=*/$_builder.getBoolAttr(structured),
          /*implicit=*/$_builder.getBoolAttr(implicit),
          /*name=*/$_builder.getStringAttr(name));
      }]>];

  code extraClassDeclarationDataExit = [{
    mlir::TypedValue<mlir::acc::PointerLikeType> getVarPtr() {
      return mlir::dyn_cast<mlir::TypedValue<mlir::acc::PointerLikeType>>(getVar());
    }
    mlir::TypedValue<mlir::acc::PointerLikeType> getAccPtr() {
      return mlir::dyn_cast<mlir::TypedValue<mlir::acc::PointerLikeType>>(getAccVar());
    }
  }];
}

class OpenACC_DataExitOpNoVarPtr<string mnemonic, string clause> :
    OpenACC_DataExitOp<mnemonic, clause, "",
      [MemoryEffects<[MemRead<OpenACC_RuntimeCounters>,
                    MemWrite<OpenACC_RuntimeCounters>]>],
      (ins Arg<OpenACC_AnyPointerOrMappableType,"Accelerator mapped variable",
           [MemRead]>:$accVar)> {
  let assemblyFormat = [{
    custom<AccVar>($accVar, type($accVar))
    (`bounds` `(` $bounds^ `)` )?
    (`async` `` custom<DeviceTypeOperandsWithKeywordOnly>($asyncOperands,
            type($asyncOperands), $asyncOperandsDeviceType, $asyncOnly)^)?
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Value":$accVar,
                   "bool":$structured, "bool":$implicit,
                   CArg<"::mlir::ValueRange", "{}">:$bounds),
      [{
        build($_builder, $_state, accVar,
          bounds, /*asyncOperands=*/{}, /*asyncOperandsDeviceType=*/nullptr,
          /*asyncOnly=*/nullptr, /*dataClause=*/nullptr,
          /*structured=*/$_builder.getBoolAttr(structured),
          /*implicit=*/$_builder.getBoolAttr(implicit), /*name=*/nullptr);
      }]>,
    OpBuilder<(ins "::mlir::Value":$accVar,
                   "bool":$structured, "bool":$implicit,
                   "const ::llvm::Twine &":$name,
                   CArg<"::mlir::ValueRange", "{}">:$bounds),
      [{
        build($_builder, $_state, accVar,
          bounds, /*asyncOperands=*/{}, /*asyncOperandsDeviceType=*/nullptr,
          /*asyncOnly=*/nullptr, /*dataClause=*/nullptr,
          /*structured=*/$_builder.getBoolAttr(structured),
          /*implicit=*/$_builder.getBoolAttr(implicit),
          /*name=*/$_builder.getStringAttr(name));
      }]>
  ];

  code extraClassDeclarationDataExit = [{
    mlir::TypedValue<mlir::acc::PointerLikeType> getAccPtr() {
      return mlir::dyn_cast<mlir::TypedValue<mlir::acc::PointerLikeType>>(getAccVar());
    }
  }];
}

//===----------------------------------------------------------------------===//
// 2.7.8 copyout clause
//===----------------------------------------------------------------------===//
def OpenACC_CopyoutOp : OpenACC_DataExitOpWithVarPtr<"copyout",
    "mlir::acc::DataClause::acc_copyout"> {
  let summary = "Represents acc copyout semantics - reverse of copyin.";

  let extraClassDeclaration = extraClassDeclarationBase # extraClassDeclarationDataExit # [{
    /// Check if this is a copyout with zero modifier.
    bool isCopyoutZero();
  }];
}

//===----------------------------------------------------------------------===//
// 2.7.11 delete clause
//===----------------------------------------------------------------------===//
def OpenACC_DeleteOp : OpenACC_DataExitOpNoVarPtr<"delete",
    "mlir::acc::DataClause::acc_delete"> {
  let summary = "Represents acc delete semantics - reverse of create.";
  let extraClassDeclaration = extraClassDeclarationBase # extraClassDeclarationDataExit;
}

//===----------------------------------------------------------------------===//
// 2.7.13 detach clause
//===----------------------------------------------------------------------===//
def OpenACC_DetachOp : OpenACC_DataExitOpNoVarPtr<"detach",
    "mlir::acc::DataClause::acc_detach"> {
  let summary = "Represents acc detach semantics - reverse of attach.";
  let extraClassDeclaration = extraClassDeclarationBase # extraClassDeclarationDataExit;
}

//===----------------------------------------------------------------------===//
// 2.14.4 host clause
//===----------------------------------------------------------------------===//
def OpenACC_UpdateHostOp : OpenACC_DataExitOpWithVarPtr<"update_host",
    "mlir::acc::DataClause::acc_update_host"> {
  let summary = "Represents acc update host semantics.";
  let extraClassDeclaration = extraClassDeclarationBase # extraClassDeclarationDataExit # [{
    /// Check if this is an acc update self.
    bool isSelf() {
      return getDataClause() == acc::DataClause::acc_update_self;
    }
  }];
}

//===----------------------------------------------------------------------===//
// 2.5.13 private clause
//===----------------------------------------------------------------------===//

def OpenACC_PrivateRecipeOp
    : OpenACC_Op<"private.recipe", [IsolatedFromAbove, Symbol, RecipeInterface,
                                    AutomaticAllocationScope]> {
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

def OpenACC_FirstprivateRecipeOp
    : OpenACC_Op<"firstprivate.recipe", [IsolatedFromAbove, Symbol,
                                         RecipeInterface,
                                         AutomaticAllocationScope]> {
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
    ^bb0(%0: f32, %1: !llvm.ptr):
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

def OpenACC_ReductionRecipeOp
    : OpenACC_Op<"reduction.recipe", [IsolatedFromAbove, Symbol,
                                      RecipeInterface,
                                      AutomaticAllocationScope]> {
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
    [AttrSizedOperandSegments, AutomaticAllocationScope,
     RecursiveMemoryEffects,
     DeclareOpInterfaceMethods<ComputeRegionOpInterface>,
     MemoryEffects<[MemWrite<OpenACC_ConstructResource>,
                    MemRead<OpenACC_CurrentDeviceIdResource>]>]> {
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

    `async`, `wait`, `num_gangs`, `num_workers` and `vector_length` operands are
    supported with `device_type` information. They should only be accessed by
    the extra provided getters. If modified, the corresponding `device_type`
    attributes must be modified as well.
  }];

  let arguments = (ins
      Variadic<IntOrIndex>:$asyncOperands,
      OptionalAttr<DeviceTypeArrayAttr>:$asyncOperandsDeviceType,
      OptionalAttr<DeviceTypeArrayAttr>:$asyncOnly,
      Variadic<IntOrIndex>:$waitOperands,
      OptionalAttr<DenseI32ArrayAttr>:$waitOperandsSegments,
      OptionalAttr<DeviceTypeArrayAttr>:$waitOperandsDeviceType,
      OptionalAttr<BoolArrayAttr>:$hasWaitDevnum,
      OptionalAttr<DeviceTypeArrayAttr>:$waitOnly,
      Variadic<IntOrIndex>:$numGangs,
      OptionalAttr<DenseI32ArrayAttr>:$numGangsSegments,
      OptionalAttr<DeviceTypeArrayAttr>:$numGangsDeviceType,
      Variadic<IntOrIndex>:$numWorkers,
      OptionalAttr<DeviceTypeArrayAttr>:$numWorkersDeviceType,
      Variadic<IntOrIndex>:$vectorLength,
      OptionalAttr<DeviceTypeArrayAttr>:$vectorLengthDeviceType,
      Optional<I1>:$ifCond,
      Optional<I1>:$selfCond,
      UnitAttr:$selfAttr,
      Variadic<AnyType>:$reductionOperands,
      OptionalAttr<SymbolRefArrayAttr>:$reductionRecipes,
      Variadic<OpenACC_AnyPointerOrMappableType>:$privateOperands,
      OptionalAttr<SymbolRefArrayAttr>:$privatizationRecipes,
      Variadic<OpenACC_AnyPointerOrMappableType>:$firstprivateOperands,
      OptionalAttr<SymbolRefArrayAttr>:$firstprivatizationRecipes,
      Variadic<OpenACC_AnyPointerOrMappableType>:$dataClauseOperands,
      OptionalAttr<DefaultValueAttr>:$defaultAttr,
      UnitAttr:$combined);

  let regions = (region AnyRegion:$region);

  let builders = [
    OpBuilder<(ins
      CArg<"mlir::ValueRange", "{}">:$numGangs,
      CArg<"mlir::ValueRange", "{}">:$numWorkers,
      CArg<"mlir::ValueRange", "{}">:$vectorLength,
      CArg<"mlir::ValueRange", "{}">:$asyncOperands,
      CArg<"mlir::ValueRange", "{}">:$waitOperands,
      CArg<"mlir::Value", "{}">:$ifCond,
      CArg<"mlir::Value", "{}">:$selfCond,
      CArg<"mlir::ValueRange", "{}">:$reductionOperands,
      CArg<"mlir::ValueRange", "{}">:$privateOperands,
      CArg<"mlir::ValueRange", "{}">:$firstprivateOperands,
      CArg<"mlir::ValueRange", "{}">:$dataClauseOperands)>];

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);

    /// Used to retrieve the block inside the op's region.
    Block &getBody() { return getRegion().front(); }

    /// Return true if the op has the async attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasAsyncOnly();
    /// Return true if the op has the async attribute for the given device_type.
    bool hasAsyncOnly(mlir::acc::DeviceType deviceType);
    /// Return the value of the async clause if present.
    mlir::Value getAsyncValue();
    /// Return the value of the async clause for the given device_type if
    /// present.
    mlir::Value getAsyncValue(mlir::acc::DeviceType deviceType);

    /// Return the value of the num_workers clause if present.
    mlir::Value getNumWorkersValue();
    /// Return the value of the num_workers clause for the given device_type if
    /// present.
    mlir::Value getNumWorkersValue(mlir::acc::DeviceType deviceType);

    /// Return the value of the vector_length clause if present.
    mlir::Value getVectorLengthValue();
    /// Return the value of the vector_length clause for the given device_type 
    /// if present.
    mlir::Value getVectorLengthValue(mlir::acc::DeviceType deviceType);

    /// Return the values of the num_gangs clause if present.
    mlir::Operation::operand_range getNumGangsValues();
    /// Return the values of the num_gangs clause for the given device_type if
    /// present.
    mlir::Operation::operand_range
    getNumGangsValues(mlir::acc::DeviceType deviceType);

    /// Return true if the op has the wait attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasWaitOnly();
    /// Return true if the op has the wait attribute for the given device_type.
    bool hasWaitOnly(mlir::acc::DeviceType deviceType);
    /// Return the values of the wait clause if present.
    mlir::Operation::operand_range getWaitValues();
    /// Return the values of the wait clause for the given device_type if
    /// present.
    mlir::Operation::operand_range
    getWaitValues(mlir::acc::DeviceType deviceType);
    /// Return the wait devnum value clause if present;
    mlir::Value getWaitDevnum();
    /// Return the wait devnum value clause for the given device_type if
    /// present.
    mlir::Value getWaitDevnum(mlir::acc::DeviceType deviceType);
    static mlir::acc::Construct getConstructId() {
      return mlir::acc::Construct::acc_construct_parallel;
    }
    /// Add a value to 'num_workers' with the current list of device types.
    void addNumWorkersOperand(MLIRContext *, mlir::Value,
                              llvm::ArrayRef<DeviceType>);
    /// Add a value to 'vector_length' with the current list of device types.
    void addVectorLengthOperand(MLIRContext *, mlir::Value,
                                llvm::ArrayRef<DeviceType>);
    /// Add an entry to the 'async-only' attribute (clause spelled without
    /// arguments)for each of the additional device types (or a none if it is
    /// empty).
    void addAsyncOnly(MLIRContext *, llvm::ArrayRef<DeviceType>);
    /// Add a value to the 'async' with the current list of device types.
    void addAsyncOperand(MLIRContext *, mlir::Value,
                         llvm::ArrayRef<DeviceType>);
    /// Add an array-like entry to the 'num_gangs' with the current list of
    /// device types.
    void addNumGangsOperands(MLIRContext *, mlir::ValueRange,
                             llvm::ArrayRef<DeviceType>);
    /// Add an entry to the 'wait-only' attribute (clause spelled without
    /// arguments)for each of the additional device types (or a none if it is
    /// empty).
    void addWaitOnly(MLIRContext *, llvm::ArrayRef<DeviceType>);
    /// Add an array-like entry  to the 'wait' with the current list of device
    /// types.
    void addWaitOperands(MLIRContext *, bool hasDevnum, mlir::ValueRange,
                         llvm::ArrayRef<DeviceType>);
  }];

  let assemblyFormat = [{
    ( `combined` `(` `loop` `)` $combined^)?
    oilist(
        `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
      | `async` `` custom<DeviceTypeOperandsWithKeywordOnly>($asyncOperands,
            type($asyncOperands), $asyncOperandsDeviceType, $asyncOnly)
      | `firstprivate` `(` custom<SymOperandList>($firstprivateOperands,
            type($firstprivateOperands), $firstprivatizationRecipes)
        `)`
      | `num_gangs` `(` custom<NumGangs>($numGangs,
            type($numGangs), $numGangsDeviceType, $numGangsSegments) `)`
      | `num_workers` `(` custom<DeviceTypeOperands>($numWorkers,
            type($numWorkers), $numWorkersDeviceType) `)`
      | `private` `(` custom<SymOperandList>(
            $privateOperands, type($privateOperands), $privatizationRecipes)
        `)`
      | `vector_length` `(` custom<DeviceTypeOperands>($vectorLength,
            type($vectorLength), $vectorLengthDeviceType) `)`
      | `wait` `` custom<WaitClause>($waitOperands, type($waitOperands),
          $waitOperandsDeviceType, $waitOperandsSegments, $hasWaitDevnum,
          $waitOnly)
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
    [AttrSizedOperandSegments, AutomaticAllocationScope,
     RecursiveMemoryEffects,
     DeclareOpInterfaceMethods<ComputeRegionOpInterface>,
     MemoryEffects<[MemWrite<OpenACC_ConstructResource>,
                    MemRead<OpenACC_CurrentDeviceIdResource>]>]> {
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

    `async` and `wait` operands are supported with `device_type` information.
    They should only be accessed by the extra provided getters. If modified,
    the corresponding `device_type` attributes must be modified as well.
  }];

  let arguments = (ins
      Variadic<IntOrIndex>:$asyncOperands,
      OptionalAttr<DeviceTypeArrayAttr>:$asyncOperandsDeviceType,
      OptionalAttr<DeviceTypeArrayAttr>:$asyncOnly,
      Variadic<IntOrIndex>:$waitOperands,
      OptionalAttr<DenseI32ArrayAttr>:$waitOperandsSegments,
      OptionalAttr<DeviceTypeArrayAttr>:$waitOperandsDeviceType,
      OptionalAttr<BoolArrayAttr>:$hasWaitDevnum,
      OptionalAttr<DeviceTypeArrayAttr>:$waitOnly,
      Optional<I1>:$ifCond,
      Optional<I1>:$selfCond,
      UnitAttr:$selfAttr,
      Variadic<AnyType>:$reductionOperands,
      OptionalAttr<SymbolRefArrayAttr>:$reductionRecipes,
      Variadic<OpenACC_AnyPointerOrMappableType>:$privateOperands,
      OptionalAttr<SymbolRefArrayAttr>:$privatizationRecipes,
      Variadic<OpenACC_AnyPointerOrMappableType>:$firstprivateOperands,
      OptionalAttr<SymbolRefArrayAttr>:$firstprivatizationRecipes,
      Variadic<OpenACC_AnyPointerOrMappableType>:$dataClauseOperands,
      OptionalAttr<DefaultValueAttr>:$defaultAttr,
      UnitAttr:$combined);

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);

    /// Used to retrieve the block inside the op's region.
    Block &getBody() { return getRegion().front(); }

    /// Return true if the op has the async attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasAsyncOnly();
    /// Return true if the op has the async attribute for the given device_type.
    bool hasAsyncOnly(mlir::acc::DeviceType deviceType);
    /// Return the value of the async clause if present.
    mlir::Value getAsyncValue();
    /// Return the value of the async clause for the given device_type if
    /// present.
    mlir::Value getAsyncValue(mlir::acc::DeviceType deviceType);

    /// Return true if the op has the wait attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasWaitOnly();
    /// Return true if the op has the wait attribute for the given device_type.
    bool hasWaitOnly(mlir::acc::DeviceType deviceType);
    /// Return the values of the wait clause if present.
    mlir::Operation::operand_range getWaitValues();
    /// Return the values of the wait clause for the given device_type if
    /// present.
    mlir::Operation::operand_range
    getWaitValues(mlir::acc::DeviceType deviceType);
    /// Return the wait devnum value clause if present;
    mlir::Value getWaitDevnum();
    /// Return the wait devnum value clause for the given device_type if
    /// present.
    mlir::Value getWaitDevnum(mlir::acc::DeviceType deviceType);
    static mlir::acc::Construct getConstructId() {
      return mlir::acc::Construct::acc_construct_serial;
    }
    /// Add an entry to the 'async-only' attribute (clause spelled without
    /// arguments) for each of the additional device types (or a none if it is
    /// empty).
    void addAsyncOnly(MLIRContext *, llvm::ArrayRef<DeviceType>);
    /// Add a value to the 'async' with the current list of device types.
    void addAsyncOperand(MLIRContext *, mlir::Value,
                         llvm::ArrayRef<DeviceType>);
    /// Add an entry to the 'wait-only' attribute (clause spelled without
    /// arguments) for each of the additional device types (or a none if it is
    /// empty).
    void addWaitOnly(MLIRContext *, llvm::ArrayRef<DeviceType>);
    /// Add an array-like entry  to the 'wait' with the current list of device
    /// types.
    void addWaitOperands(MLIRContext *, bool hasDevnum, mlir::ValueRange,
                         llvm::ArrayRef<DeviceType>);
  }];

  let assemblyFormat = [{
    ( `combined` `(` `loop` `)` $combined^)?
    oilist(
        `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
      | `async` `` custom<DeviceTypeOperandsWithKeywordOnly>($asyncOperands,
            type($asyncOperands), $asyncOperandsDeviceType, $asyncOnly)
      | `firstprivate` `(` custom<SymOperandList>($firstprivateOperands,
            type($firstprivateOperands), $firstprivatizationRecipes)
        `)`
      | `private` `(` custom<SymOperandList>(
            $privateOperands, type($privateOperands), $privatizationRecipes)
        `)`
      | `wait` `` custom<WaitClause>($waitOperands, type($waitOperands),
          $waitOperandsDeviceType, $waitOperandsSegments, $hasWaitDevnum,
          $waitOnly)
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
    [AttrSizedOperandSegments, AutomaticAllocationScope,
     RecursiveMemoryEffects,
     DeclareOpInterfaceMethods<ComputeRegionOpInterface>,
     MemoryEffects<[MemWrite<OpenACC_ConstructResource>,
                    MemRead<OpenACC_CurrentDeviceIdResource>]>]> {
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

    `collapse`, `gang`, `worker`, `vector`, `seq`, `independent`, `auto` and
    `tile` operands are supported with `device_type` information. They should
    only be accessed by the extra provided getters. If modified, the
    corresponding `device_type` attributes must be modified as well.
  }];

  let arguments = (ins
      Variadic<IntOrIndex>:$asyncOperands,
      OptionalAttr<DeviceTypeArrayAttr>:$asyncOperandsDeviceType,
      OptionalAttr<DeviceTypeArrayAttr>:$asyncOnly,
      Variadic<IntOrIndex>:$waitOperands,
      OptionalAttr<DenseI32ArrayAttr>:$waitOperandsSegments,
      OptionalAttr<DeviceTypeArrayAttr>:$waitOperandsDeviceType,
      OptionalAttr<BoolArrayAttr>:$hasWaitDevnum,
      OptionalAttr<DeviceTypeArrayAttr>:$waitOnly,
      Variadic<IntOrIndex>:$numGangs,
      OptionalAttr<DenseI32ArrayAttr>:$numGangsSegments,
      OptionalAttr<DeviceTypeArrayAttr>:$numGangsDeviceType,
      Variadic<IntOrIndex>:$numWorkers,
      OptionalAttr<DeviceTypeArrayAttr>:$numWorkersDeviceType,
      Variadic<IntOrIndex>:$vectorLength,
      OptionalAttr<DeviceTypeArrayAttr>:$vectorLengthDeviceType,
      Optional<I1>:$ifCond,
      Optional<I1>:$selfCond,
      UnitAttr:$selfAttr,
      Variadic<OpenACC_AnyPointerOrMappableType>:$dataClauseOperands,
      OptionalAttr<DefaultValueAttr>:$defaultAttr,
      UnitAttr:$combined);

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);

    /// Used to retrieve the block inside the op's region.
    Block &getBody() { return getRegion().front(); }

    /// Return true if the op has the async attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasAsyncOnly();
    /// Return true if the op has the async attribute for the given device_type.
    bool hasAsyncOnly(mlir::acc::DeviceType deviceType);
    /// Return the value of the async clause if present.
    mlir::Value getAsyncValue();
    /// Return the value of the async clause for the given device_type if
    /// present.
    mlir::Value getAsyncValue(mlir::acc::DeviceType deviceType);

    /// Return the value of the num_workers clause if present.
    mlir::Value getNumWorkersValue();
    /// Return the value of the num_workers clause for the given device_type if
    /// present.
    mlir::Value getNumWorkersValue(mlir::acc::DeviceType deviceType);

    /// Return the value of the vector_length clause if present.
    mlir::Value getVectorLengthValue();
    /// Return the value of the vector_length clause for the given device_type 
    /// if present.
    mlir::Value getVectorLengthValue(mlir::acc::DeviceType deviceType);

    /// Return the values of the num_gangs clause if present.
    mlir::Operation::operand_range getNumGangsValues();
    /// Return the values of the num_gangs clause for the given device_type if
    /// present.
    mlir::Operation::operand_range
    getNumGangsValues(mlir::acc::DeviceType deviceType);

    /// Return true if the op has the wait attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasWaitOnly();
    /// Return true if the op has the wait attribute for the given device_type.
    bool hasWaitOnly(mlir::acc::DeviceType deviceType);
    /// Return the values of the wait clause if present.
    mlir::Operation::operand_range getWaitValues();
    /// Return the values of the wait clause for the given device_type if
    /// present.
    mlir::Operation::operand_range
    getWaitValues(mlir::acc::DeviceType deviceType);
    /// Return the wait devnum value clause if present;
    mlir::Value getWaitDevnum();
    /// Return the wait devnum value clause for the given device_type if
    /// present.
    mlir::Value getWaitDevnum(mlir::acc::DeviceType deviceType);
    static mlir::acc::Construct getConstructId() {
      return mlir::acc::Construct::acc_construct_kernels;
    }
    /// Add a value to 'num_workers' with the current list of device types.
    void addNumWorkersOperand(MLIRContext *, mlir::Value,
                              llvm::ArrayRef<DeviceType>);
    /// Add a value to 'vector_length' with the current list of device types.
    void addVectorLengthOperand(MLIRContext *, mlir::Value,
                                llvm::ArrayRef<DeviceType>);
    /// Add an entry to the 'async-only' attribute (clause spelled without
    /// arguments) for each of the additional device types (or a none if it is
    /// empty).
    void addAsyncOnly(MLIRContext *, llvm::ArrayRef<DeviceType>);
    /// Add a value to the 'async' with the current list of device types.
    void addAsyncOperand(MLIRContext *, mlir::Value,
                         llvm::ArrayRef<DeviceType>);
    /// Add an array-like entry to the 'num_gangs' with the current list of
    /// device types.
    void addNumGangsOperands(MLIRContext *, mlir::ValueRange,
                             llvm::ArrayRef<DeviceType>);
    /// Add an entry to the 'wait-only' attribute (clause spelled without
    /// arguments) for each of the additional device types (or a none if it is
    /// empty).
    void addWaitOnly(MLIRContext *, llvm::ArrayRef<DeviceType>);
    /// Add an array-like entry  to the 'wait' with the current list of device
    /// types.
    void addWaitOperands(MLIRContext *, bool hasDevnum, mlir::ValueRange,
                         llvm::ArrayRef<DeviceType>);
  }];

  let assemblyFormat = [{
    ( `combined` `(` `loop` `)` $combined^)?
    oilist(
        `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
      | `async` `` custom<DeviceTypeOperandsWithKeywordOnly>($asyncOperands,
            type($asyncOperands), $asyncOperandsDeviceType, $asyncOnly)
      | `num_gangs` `(` custom<NumGangs>($numGangs,
            type($numGangs), $numGangsDeviceType, $numGangsSegments) `)`
      | `num_workers` `(` custom<DeviceTypeOperands>($numWorkers,
            type($numWorkers), $numWorkersDeviceType) `)`
      | `vector_length` `(` custom<DeviceTypeOperands>($vectorLength,
            type($vectorLength), $vectorLengthDeviceType) `)`
      | `wait` `` custom<WaitClause>($waitOperands, type($waitOperands),
          $waitOperandsDeviceType, $waitOperandsSegments, $hasWaitDevnum,
          $waitOnly)
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
    [AttrSizedOperandSegments, RecursiveMemoryEffects,
     MemoryEffects<[MemWrite<OpenACC_ConstructResource>,
                    MemRead<OpenACC_CurrentDeviceIdResource>]>]> {
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

    `async` and `wait` operands are supported with `device_type` information.
    They should only be accessed by the extra provided getters. If modified,
    the corresponding `device_type` attributes must be modified as well.
  }];


  let arguments = (ins Optional<I1>:$ifCond,
      Variadic<IntOrIndex>:$asyncOperands,
      OptionalAttr<DeviceTypeArrayAttr>:$asyncOperandsDeviceType,
      OptionalAttr<DeviceTypeArrayAttr>:$asyncOnly,
      Variadic<IntOrIndex>:$waitOperands,
      OptionalAttr<DenseI32ArrayAttr>:$waitOperandsSegments,
      OptionalAttr<DeviceTypeArrayAttr>:$waitOperandsDeviceType,
      OptionalAttr<BoolArrayAttr>:$hasWaitDevnum,
      OptionalAttr<DeviceTypeArrayAttr>:$waitOnly,
      Variadic<OpenACC_AnyPointerOrMappableType>:$dataClauseOperands,
      OptionalAttr<DefaultValueAttr>:$defaultAttr);

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);

    /// Return true if the op has the async attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasAsyncOnly();
    /// Return true if the op has the async attribute for the given device_type.
    bool hasAsyncOnly(mlir::acc::DeviceType deviceType);
    /// Return the value of the async clause if present.
    mlir::Value getAsyncValue();
    /// Return the value of the async clause for the given device_type if
    /// present.
    mlir::Value getAsyncValue(mlir::acc::DeviceType deviceType);

    /// Return true if the op has the wait attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasWaitOnly();
    /// Return true if the op has the wait attribute for the given device_type.
    bool hasWaitOnly(mlir::acc::DeviceType deviceType);
    /// Return the values of the wait clause if present.
    mlir::Operation::operand_range getWaitValues();
    /// Return the values of the wait clause for the given device_type if
    /// present.
    mlir::Operation::operand_range
    getWaitValues(mlir::acc::DeviceType deviceType);
    /// Return the wait devnum value clause if present;
    mlir::Value getWaitDevnum();
    /// Return the wait devnum value clause for the given device_type if
    /// present.
    mlir::Value getWaitDevnum(mlir::acc::DeviceType deviceType);
    /// Add an entry to the 'async-only' attribute (clause spelled without
    /// arguments) for each of the additional device types (or a none if it is
    /// empty).
    void addAsyncOnly(MLIRContext *, llvm::ArrayRef<DeviceType>);
    /// Add a value to the 'async' with the current list of device types.
    void addAsyncOperand(MLIRContext *, mlir::Value,
                         llvm::ArrayRef<DeviceType>);
    /// Add an entry to the 'wait-only' attribute (clause spelled without
    /// arguments) for each of the additional device types (or a none if it is
    /// empty).
    void addWaitOnly(MLIRContext *, llvm::ArrayRef<DeviceType>);
    /// Add an array-like entry  to the 'wait' with the current list of device
    /// types.
    void addWaitOperands(MLIRContext *, bool hasDevnum, mlir::ValueRange,
                         llvm::ArrayRef<DeviceType>);
  }];

  let assemblyFormat = [{
    oilist(
        `if` `(` $ifCond `)`
      | `async` `` custom<DeviceTypeOperandsWithKeywordOnly>($asyncOperands,
            type($asyncOperands), $asyncOperandsDeviceType, $asyncOnly)
      | `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
      | `wait` `` custom<WaitClause>($waitOperands, type($waitOperands),
          $waitOperandsDeviceType, $waitOperandsSegments, $hasWaitDevnum,
          $waitOnly)
    )
    $region attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

def OpenACC_TerminatorOp : OpenACC_Op<"terminator", [Pure, Terminator]> {
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

def OpenACC_EnterDataOp : OpenACC_Op<"enter_data",
    [AttrSizedOperandSegments,
     MemoryEffects<[MemWrite<OpenACC_ConstructResource>,
                    MemRead<OpenACC_CurrentDeviceIdResource>]>]> {
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
                       Variadic<OpenACC_AnyPointerOrMappableType>:$dataClauseOperands);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);
  }];

  let assemblyFormat = [{
    oilist(
        `if` `(` $ifCond `)`
      | `async` `` custom<OperandWithKeywordOnly>($asyncOperand,
            type($asyncOperand), $async)
      | `wait_devnum` `(` $waitDevnum `:` type($waitDevnum) `)`
      | `wait` `` custom<OperandsWithKeywordOnly>($waitOperands,
            type($waitOperands), $wait)
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

def OpenACC_ExitDataOp : OpenACC_Op<"exit_data",
    [AttrSizedOperandSegments,
     MemoryEffects<[MemWrite<OpenACC_ConstructResource>,
                    MemRead<OpenACC_CurrentDeviceIdResource>]>]> {
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
                       Variadic<OpenACC_AnyPointerOrMappableType>:$dataClauseOperands,
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
      | `async` `` custom<OperandWithKeywordOnly>($asyncOperand,
            type($asyncOperand), $async)
      | `wait_devnum` `(` $waitDevnum `:` type($waitDevnum) `)`
      | `wait` `` custom<OperandsWithKeywordOnly>($waitOperands,
            type($waitOperands), $wait)
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

def OpenACC_HostDataOp : OpenACC_Op<"host_data",
    [AttrSizedOperandSegments,
     MemoryEffects<[MemWrite<OpenACC_ConstructResource>,
                    MemRead<OpenACC_CurrentDeviceIdResource>]>]> {
  let summary = "host_data construct";

  let description = [{
    The "acc.host_data" operation represents the OpenACC host_data construct.

    Example:

    ```mlir
    %0 = acc.use_device varPtr(%a : !llvm.ptr) -> !llvm.ptr
    acc.host_data dataOperands(%0 : !llvm.ptr) {

    }
    ```
  }];

  let arguments = (ins Optional<I1>:$ifCond,
                       Variadic<OpenACC_AnyPointerOrMappableType>:$dataClauseOperands,
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
    [AttrSizedOperandSegments, AutomaticAllocationScope,
     RecursiveMemoryEffects,
     DeclareOpInterfaceMethods<ComputeRegionOpInterface>,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>,
     MemoryEffects<[MemWrite<OpenACC_ConstructResource>]>]> {
  let summary = "loop construct";

  let description = [{
    The "acc.loop" operation represents the OpenACC loop construct. The lower
    and upper bounds specify a half-open range: the range includes the lower
    bound but does not include the upper bound. If the `inclusive` attribute is
    set then the upper bound is included.

    Example:

    ```mlir
    acc.loop gang() vector() (%arg3 : index, %arg4 : index, %arg5 : index) = 
        (%c0, %c0, %c0 : index, index, index) to 
        (%c10, %c10, %c10 : index, index, index) step 
        (%c1, %c1, %c1 : index, index, index) {
      // Loop body
      acc.yield
    } attributes { collapse = [3] }
    ```

    `collapse`, `gang`, `worker`, `vector`, `seq`, `independent`, `auto` and
    `tile` operands are supported with `device_type` information. They should
    only be accessed by the extra provided getters. If modified, the
    corresponding `device_type` attributes must be modified as well.
  }];

  let arguments = (ins
      Variadic<IntOrIndex>:$lowerbound,
      Variadic<IntOrIndex>:$upperbound,
      Variadic<IntOrIndex>:$step,
      OptionalAttr<DenseBoolArrayAttr>:$inclusiveUpperbound,
      OptionalAttr<I64ArrayAttr>:$collapse,
      OptionalAttr<DeviceTypeArrayAttr>:$collapseDeviceType,
      Variadic<IntOrIndex>:$gangOperands,
      OptionalAttr<GangArgTypeArrayAttr>:$gangOperandsArgType,
      OptionalAttr<DenseI32ArrayAttr>:$gangOperandsSegments,
      OptionalAttr<DeviceTypeArrayAttr>:$gangOperandsDeviceType,
      Variadic<IntOrIndex>:$workerNumOperands,
      OptionalAttr<DeviceTypeArrayAttr>:$workerNumOperandsDeviceType,
      Variadic<IntOrIndex>:$vectorOperands,
      OptionalAttr<DeviceTypeArrayAttr>:$vectorOperandsDeviceType,
      OptionalAttr<DeviceTypeArrayAttr>:$seq,
      OptionalAttr<DeviceTypeArrayAttr>:$independent,
      OptionalAttr<DeviceTypeArrayAttr>:$auto_,
      OptionalAttr<DeviceTypeArrayAttr>:$gang,
      OptionalAttr<DeviceTypeArrayAttr>:$worker,
      OptionalAttr<DeviceTypeArrayAttr>:$vector,
      Variadic<IntOrIndex>:$tileOperands,
      OptionalAttr<DenseI32ArrayAttr>:$tileOperandsSegments,
      OptionalAttr<DeviceTypeArrayAttr>:$tileOperandsDeviceType,
      Variadic<OpenACC_AnyPointerOrMappableType>:$cacheOperands,
      Variadic<OpenACC_AnyPointerOrMappableType>:$privateOperands,
      OptionalAttr<SymbolRefArrayAttr>:$privatizationRecipes,
      Variadic<AnyType>:$reductionOperands,
      OptionalAttr<SymbolRefArrayAttr>:$reductionRecipes,
      OptionalAttr<OpenACC_CombinedConstructsAttr>:$combined
  );

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    static StringRef getAutoAttrStrName() { return "auto"; }
    static StringRef getGangNumKeyword() { return "num"; }
    static StringRef getGangDimKeyword() { return "dim"; }
    static StringRef getGangStaticKeyword() { return "static"; }
    static StringRef getControlKeyword() { return "control"; }

    /// The number of private and reduction operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);

    /// Used to retrieve the block inside the op's region.
    Block &getBody() { return getLoopRegions().front()->front(); }

    /// Used to determine if this operation is merely a container for a loop
    /// operation instead of being loop-like itself.
    bool isLoopLike() { return !getLowerbound().empty(); }
    bool isContainerLike() { return !isLoopLike(); }

    /// Return true if the op has the auto attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasAuto();
    /// Return true if the op has the auto attribute for the given device_type.
    bool hasAuto(mlir::acc::DeviceType deviceType);
    /// Return true if the op has the independent attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasIndependent();
    /// Return true if the op has the independent attribute for the given
    /// device_type.
    bool hasIndependent(mlir::acc::DeviceType deviceType);
    /// Return true if the op has the seq attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasSeq();
    /// Return true if the op has the seq attribute for the given device_type.
    bool hasSeq(mlir::acc::DeviceType deviceType);

    /// Return the value of the vector clause if present.
    mlir::Value getVectorValue();
    /// Return the value of the vector clause for the given device_type 
    /// if present.
    mlir::Value getVectorValue(mlir::acc::DeviceType deviceType);
    /// Return true if the op has the vector attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasVector();
    /// Return true if the op has the vector attribute for the given
    /// device_type.
    bool hasVector(mlir::acc::DeviceType deviceType);

    /// Return the value of the worker clause if present.
    mlir::Value getWorkerValue();
    /// Return the value of the worker clause for the given device_type 
    /// if present.
    mlir::Value getWorkerValue(mlir::acc::DeviceType deviceType);
    /// Return true if the op has the worker attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasWorker();
    /// Return true if the op has the worker attribute for the given
    /// device_type.
    bool hasWorker(mlir::acc::DeviceType deviceType);

    /// Return the values of the tile clause if present.
    mlir::Operation::operand_range getTileValues();
    /// Return the values of the tile clause for the given device_type if
    /// present.
    mlir::Operation::operand_range
    getTileValues(mlir::acc::DeviceType deviceType);

    /// Return the value of the collapse clause if present.
    std::optional<int64_t> getCollapseValue();
    /// Return the value of the collapse clause for the given device_type 
    /// if present.
    std::optional<int64_t> getCollapseValue(mlir::acc::DeviceType deviceType);

    /// Return true if the op has the gang attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasGang();
    /// Return true if the op has the gang attribute for the given
    /// device_type.
    bool hasGang(mlir::acc::DeviceType deviceType);

    /// Return the value of the worker clause if present.
    mlir::Value getGangValue(mlir::acc::GangArgType gangArgType);
    /// Return the value of the worker clause for the given device_type 
    /// if present.
    mlir::Value getGangValue(mlir::acc::GangArgType gangArgType, mlir::acc::DeviceType deviceType);

    // Add an entry to the 'seq' attribute for each additional device types.
    void addSeq(MLIRContext *, llvm::ArrayRef<DeviceType>);
    // Add an entry to the 'independent' attribute for each additional device
    // types.
    void addIndependent(MLIRContext *, llvm::ArrayRef<DeviceType>);
    // Add an entry to the 'auto' attribute for each additional device types.
    void addAuto(MLIRContext *, llvm::ArrayRef<DeviceType>);

    // Sets the collapse value for this 'loop' for a set of DeviceTypes. Note
    // that this may only be set once per DeviceType, and will fail the verifier
    // if this is set multiple times.
    void setCollapseForDeviceTypes(MLIRContext *, llvm::ArrayRef<DeviceType>,
                                   llvm::APInt);
    // Sets the tile values for this 'loop' for a set of DeviceTypes. All of the
    // values should be integral constants, with the '*' represented as a '-1'.
    void setTileForDeviceTypes(MLIRContext *, llvm::ArrayRef<DeviceType>,
                               mlir::ValueRange);

    // Add a value to the 'vector' list with a current list of device_types.
    void addVectorOperand(MLIRContext *, mlir::Value,
                          llvm::ArrayRef<DeviceType>);
    // Add an empty value to the 'vector' list with a current list of
    // device_types. This is for the case where there is no expression specified
    // in a 'vector'.
    void addEmptyVector(MLIRContext *, llvm::ArrayRef<DeviceType>);
    // Add a value to the 'worker' list with a current list of device_types.
    void addWorkerNumOperand(MLIRContext *, mlir::Value,
                             llvm::ArrayRef<DeviceType>);
    // Add an empty value to the 'worker' list with a current list of
    // device_types. This is for the case where there is no expression specified
    // in a 'worker'.
    void addEmptyWorker(MLIRContext *, llvm::ArrayRef<DeviceType>);

    // Adds a collection of operands for a 'gang' clause that has various types
    // corresponding to each operand.
    void addGangOperands(MLIRContext *, llvm::ArrayRef<DeviceType>,
                         llvm::ArrayRef<GangArgType>, mlir::ValueRange);

    // Add an empty value to the 'gang' list with a current list of
    // device_types. This is for the case where there is no expression specified
    // in a 'gang'.
    void addEmptyGang(MLIRContext *, llvm::ArrayRef<DeviceType>);

    // Return whether this LoopOp has an auto, seq, or independent for the
    // specified device-type.
    bool hasParallelismFlag(DeviceType);

    // Return whether this LoopOp has a gang, worker, or vector applying to the
    // 'default'/None device-type.
    bool hasDefaultGangWorkerVector();
  }];

  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    ( `combined` `(` custom<CombinedConstructsLoop>($combined)^ `)` )?
    oilist(
        `gang` `` custom<GangClause>($gangOperands, type($gangOperands),
            $gangOperandsArgType, $gangOperandsDeviceType,
            $gangOperandsSegments, $gang)
      | `worker` `` custom<DeviceTypeOperandsWithKeywordOnly>(
            $workerNumOperands, type($workerNumOperands),
            $workerNumOperandsDeviceType, $worker)
      | `vector` `` custom<DeviceTypeOperandsWithKeywordOnly>($vectorOperands,
            type($vectorOperands), $vectorOperandsDeviceType, $vector)
      | `private` `(` custom<SymOperandList>(
            $privateOperands, type($privateOperands), $privatizationRecipes) `)`
      | `tile` `(` custom<DeviceTypeOperandsWithSegment>($tileOperands,
            type($tileOperands), $tileOperandsDeviceType, $tileOperandsSegments)
        `)`
      | `reduction` `(` custom<SymOperandList>(
            $reductionOperands, type($reductionOperands), $reductionRecipes)
        `)`
      | `cache` `(` $cacheOperands `:` type($cacheOperands) `)`
    )
    custom<LoopControl>($region, $lowerbound, type($lowerbound), $upperbound,
        type($upperbound), $step, type($step))
    ( `(` type($results)^ `)` )?
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}

// Yield operation for the acc.loop and acc.parallel operations.
def OpenACC_YieldOp : OpenACC_Op<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["FirstprivateRecipeOp, LoopOp, ParallelOp, PrivateRecipeOp,"
                 "ReductionRecipeOp, SerialOp, AtomicUpdateOp"]>]> {
  let summary = "Acc yield and termination operation";

  let description = [{
    `acc.yield` is a special terminator operation for block inside regions in
    various acc ops (including parallel, loop, atomic.update). It returns values
    to the immediately enclosing acc op.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// 2.12 atomic construct
//===----------------------------------------------------------------------===//

def AtomicReadOp : OpenACC_Op<"atomic.read", [AtomicReadOpInterface]> {

  let summary = "performs an atomic read";

  let description = [{
    This operation performs an atomic read.

    The operand `x` is the address from where the value is atomically read.
    The operand `v` is the address where the value is stored after reading.
  }];

  let arguments = (ins OpenACC_PointerLikeType:$x,
                       OpenACC_PointerLikeType:$v,
                       TypeAttr:$element_type);
  let assemblyFormat = [{
    $v `=` $x
    `:` type($v) `,` type($x) `,` $element_type attr-dict
  }];
  let hasVerifier = 1;
}

def AtomicWriteOp : OpenACC_Op<"atomic.write",[AtomicWriteOpInterface]> {

  let summary = "performs an atomic write";

  let description = [{
    This operation performs an atomic write.

    The operand `x` is the address to where the `expr` is atomically
    written w.r.t. multiple threads. The evaluation of `expr` need not be
    atomic w.r.t. the write to address. In general, the type(x) must
    dereference to type(expr).
  }];

  let arguments = (ins OpenACC_PointerLikeType:$x,
                       AnyType:$expr);
  let assemblyFormat = [{
    $x `=` $expr
    `:` type($x) `,` type($expr)
    attr-dict
  }];
  let hasVerifier = 1;
}

def AtomicUpdateOp : OpenACC_Op<"atomic.update",
                               [SingleBlockImplicitTerminator<"YieldOp">,
                                RecursiveMemoryEffects,
                                AtomicUpdateOpInterface]> {

  let summary = "performs an atomic update";

  let description = [{
    This operation performs an atomic update.

    The operand `x` is exactly the same as the operand `x` in the OpenACC
    Standard (OpenACC 3.3, section 2.12). It is the address of the variable
    that is being updated. `x` is atomically read/written.

    The region describes how to update the value of `x`. It takes the value at
    `x` as an input and must yield the updated value. Only the update to `x` is
    atomic. Generally the region must have only one instruction, but can
    potentially have more than one instructions too. The update is sematically
    similar to a compare-exchange loop based atomic update.

    The syntax of atomic update operation is different from atomic read and
    atomic write operations. This is because only the host dialect knows how to
    appropriately update a value. For example, while generating LLVM IR, if
    there are no special `atomicrmw` instructions for the operation-type
    combination in atomic update, a compare-exchange loop is generated, where
    the core update operation is directly translated like regular operations by
    the host dialect. The front-end must handle semantic checks for allowed
    operations.
  }];

  let arguments = (ins Arg<OpenACC_PointerLikeType,
                           "Address of variable to be updated",
                           [MemRead, MemWrite]>:$x);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    $x `:` type($x) $region attr-dict
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let hasCanonicalizeMethod = 1;
  let extraClassDeclaration = [{
    Operation* getFirstOp() {
      return &getRegion().front().getOperations().front();
    }
  }];
}

def AtomicCaptureOp : OpenACC_Op<"atomic.capture",
    [SingleBlockImplicitTerminator<"TerminatorOp">,
     RecursiveMemoryEffects, AtomicCaptureOpInterface]> {
  let summary = "performs an atomic capture";
  let description = [{
    This operation performs an atomic capture.

    The region has the following allowed forms:

    ```
      acc.atomic.capture {
        acc.atomic.update ...
        acc.atomic.read ...
        acc.terminator
      }

      acc.atomic.capture {
        acc.atomic.read ...
        acc.atomic.update ...
        acc.terminator
      }

      acc.atomic.capture {
        acc.atomic.read ...
        acc.atomic.write ...
        acc.terminator
      }
    ```

  }];

  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
    $region attr-dict
  }];
  let hasRegionVerifier = 1;
  let extraClassDeclaration = [{
    /// Returns the `atomic.read` operation inside the region, if any.
    /// Otherwise, it returns nullptr.
    AtomicReadOp getAtomicReadOp();

    /// Returns the `atomic.write` operation inside the region, if any.
    /// Otherwise, it returns nullptr.
    AtomicWriteOp getAtomicWriteOp();

    /// Returns the `atomic.update` operation inside the region, if any.
    /// Otherwise, it returns nullptr.
    AtomicUpdateOp getAtomicUpdateOp();
  }];
}

//===----------------------------------------------------------------------===//
// 2.13 Declare Directive
//===----------------------------------------------------------------------===//

def OpenACC_DeclareEnterOp : OpenACC_Op<"declare_enter",
    [MemoryEffects<[MemWrite<OpenACC_ConstructResource>,
                    MemRead<OpenACC_CurrentDeviceIdResource>]>]> {
  let summary = "declare directive - entry to implicit data region";

  let description = [{
    The "acc.declare_enter" operation represents the OpenACC declare directive
    and captures the entry semantics to the implicit data region.
    This operation is modeled similarly to "acc.enter_data".

    Example showing `acc declare create(a)`:

    ```mlir
    %0 = acc.create varPtr(%a : !llvm.ptr) -> !llvm.ptr
    acc.declare_enter dataOperands(%0 : !llvm.ptr)
    ```
  }];

  let arguments = (ins Variadic<OpenACC_AnyPointerOrMappableType>:$dataClauseOperands);
  let results = (outs OpenACC_DeclareTokenType:$token);

  let assemblyFormat = [{
    oilist(
        `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
    )
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}

def OpenACC_DeclareExitOp : OpenACC_Op<"declare_exit",
    [AttrSizedOperandSegments,
     MemoryEffects<[MemWrite<OpenACC_ConstructResource>,
                    MemRead<OpenACC_CurrentDeviceIdResource>]>]> {
  let summary = "declare directive - exit from implicit data region";

  let description = [{
    The "acc.declare_exit" operation represents the OpenACC declare directive
    and captures the exit semantics from the implicit data region.
    This operation is modeled similarly to "acc.exit_data".

    Example showing `acc declare device_resident(a)`:

    ```mlir
    %0 = acc.getdeviceptr varPtr(%a : !llvm.ptr) -> !llvm.ptr {dataClause = #acc<data_clause declare_device_resident>}
    acc.declare_exit dataOperands(%0 : !llvm.ptr)
    acc.delete accPtr(%0 : !llvm.ptr) {dataClause = #acc<data_clause declare_device_resident>}
    ```
  }];

  let arguments = (ins
      Optional<OpenACC_DeclareTokenType>:$token,
      Variadic<OpenACC_AnyPointerOrMappableType>:$dataClauseOperands);

  let assemblyFormat = [{
    oilist(
        `token` `(` $token `)` |
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
      %0 = llvm.mlir.addressof @globalvar : !llvm.ptr
      %1 = acc.create varPtr(%0 : !llvm.ptr) -> !llvm.ptr
      acc.declare_enter dataOperands(%1 : !llvm.ptr)
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
      %0 = llvm.mlir.addressof @globalvar : !llvm.ptr
      %1 = acc.getdeviceptr varPtr(%0 : !llvm.ptr) -> !llvm.ptr {dataClause = #acc<data_clause create>}
      acc.declare_exit dataOperands(%1 : !llvm.ptr)
      acc.delete accPtr(%1 : !llvm.ptr) {dataClause = #acc<data_clause create>}
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

def OpenACC_DeclareOp : OpenACC_Op<"declare",
    [RecursiveMemoryEffects,
     MemoryEffects<[MemWrite<OpenACC_ConstructResource>]>]> {
  let summary = "declare implicit region";

  let description = [{
    The "acc.declare" operation represents an implicit declare region in
    function (and subroutine in Fortran).

    Example:

    ```mlir
    %pa = acc.present varPtr(%a : memref<10x10xf32>) -> memref<10x10xf32>
    acc.declare dataOperands(%pa: memref<10x10xf32>) {
      // implicit region
    }
    ```
  }];

  let arguments = (ins
      Variadic<OpenACC_AnyPointerOrMappableType>:$dataClauseOperands);

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
      `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
    $region attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.15.1 Routine Directive
//===----------------------------------------------------------------------===//

def OpenACC_RoutineOp : OpenACC_Op<"routine", [IsolatedFromAbove]> {
  let summary = "acc routine operation";

  let description = [{
    The `acc.routine` operation is used to capture the clauses of acc
    routine directive, including the associated function name. The associated
    function keeps track of its corresponding routine declaration through
    the `RoutineInfoAttr`.

    Example:

    ```mlir
    func.func @acc_func(%a : i64) -> () attributes 
        {acc.routine_info = #acc.routine_info<[@acc_func_rout1]>} {
      return
    }
    acc.routine @acc_func_rout1 func(@acc_func) gang
    ```

    `bind`, `gang`, `worker`, `vector` and `seq` operands are supported with
    `device_type` information. They should only be accessed by the extra
    provided getters. If modified, the corresponding `device_type` attributes
    must be modified as well.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       SymbolNameAttr:$func_name,
                       OptionalAttr<StrArrayAttr>:$bindName,
                       OptionalAttr<DeviceTypeArrayAttr>:$bindNameDeviceType,
                       OptionalAttr<DeviceTypeArrayAttr>:$worker,
                       OptionalAttr<DeviceTypeArrayAttr>:$vector,
                       OptionalAttr<DeviceTypeArrayAttr>:$seq,
                       UnitAttr:$nohost,
                       UnitAttr:$implicit,
                       OptionalAttr<DeviceTypeArrayAttr>:$gang,
                       OptionalAttr<I64ArrayAttr>:$gangDim,
                       OptionalAttr<DeviceTypeArrayAttr>:$gangDimDeviceType);

  let extraClassDeclaration = [{
    static StringRef getGangDimKeyword() { return "dim"; }

    /// Return true if the op has the worker attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasWorker();
    /// Return true if the op has the worker attribute for the given
    /// device_type.
    bool hasWorker(mlir::acc::DeviceType deviceType);

    /// Return true if the op has the vector attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasVector();
    /// Return true if the op has the vector attribute for the given
    /// device_type.
    bool hasVector(mlir::acc::DeviceType deviceType);

    /// Return true if the op has the seq attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasSeq();
    /// Return true if the op has the seq attribute for the given
    /// device_type.
    bool hasSeq(mlir::acc::DeviceType deviceType);

    /// Return true if the op has the gang attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasGang();
    /// Return true if the op has the gang attribute for the given
    /// device_type.
    bool hasGang(mlir::acc::DeviceType deviceType);

    std::optional<int64_t> getGangDimValue();
    std::optional<int64_t> getGangDimValue(mlir::acc::DeviceType deviceType);

    std::optional<llvm::StringRef> getBindNameValue();
    std::optional<llvm::StringRef> getBindNameValue(mlir::acc::DeviceType deviceType);
  }];

  let assemblyFormat = [{
    $sym_name `func` `(` $func_name `)`
    oilist (
        `bind` `(` custom<BindName>($bindName, $bindNameDeviceType) `)`
      | `gang` `` custom<RoutineGangClause>($gang, $gangDim, $gangDimDeviceType)
      | `worker` custom<DeviceTypeArrayAttr>($worker)
      | `vector` custom<DeviceTypeArrayAttr>($vector)
      | `seq` custom<DeviceTypeArrayAttr>($seq)
      | `nohost` $nohost
      | `implicit` $implicit
    ) attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}

def RoutineInfoAttr : OpenACC_Attr<"RoutineInfo", "routine_info"> {
  let summary = "Keeps track of associated acc routine information";

  let description = [{
    This attribute is used to create the association between a function and
    its `acc.routine` operation. A `func.func` uses this if its name
    was referenced in an `acc routine` directive.
  }];

  let parameters = (ins ArrayRefParameter<"SymbolRefAttr", "">:$accRoutines);
  let assemblyFormat = "`<` `[` `` $accRoutines `]` `>`";
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

  let arguments = (ins OptionalAttr<TypedArrayAttrBase<OpenACC_DeviceTypeAttr, "Device type attributes">>:$device_types,
                       Optional<IntOrIndex>:$deviceNum,
                       Optional<I1>:$ifCond);

  let extraClassDeclaration = [{
    /// Adds a device type to the list of device types for this directive.
    void addDeviceType(MLIRContext *, mlir::acc::DeviceType);
  }];

  let assemblyFormat = [{
    oilist(`device_num` `(` $deviceNum `:` type($deviceNum) `)`
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

  let arguments = (ins OptionalAttr<TypedArrayAttrBase<OpenACC_DeviceTypeAttr, "Device type attributes">>:$device_types,
                       Optional<IntOrIndex>:$deviceNum,
                       Optional<I1>:$ifCond);

  let extraClassDeclaration = [{
    /// Adds a device type to the list of device types for this directive.
    void addDeviceType(MLIRContext *, mlir::acc::DeviceType);
  }];

  let assemblyFormat = [{
    oilist(`device_num` `(` $deviceNum `:` type($deviceNum) `)`
    |`if` `(` $ifCond `)`
    ) attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.14.3. Set
//===----------------------------------------------------------------------===//

def OpenACC_SetOp : OpenACC_Op<"set", [AttrSizedOperandSegments,
    MemoryEffects<[MemWrite<OpenACC_CurrentDeviceIdResource>]>]> {
  let summary = "set operation";

  let description = [{
    The "acc.set" operation represents the OpenACC set directive.

    Example:

    ```mlir
    acc.set device_num(%dev1 : i32)
    ```
  }];

  let arguments = (ins OptionalAttr<OpenACC_DeviceTypeAttr>:$device_type,
                       Optional<IntOrIndex>:$defaultAsync,
                       Optional<IntOrIndex>:$deviceNum,
                       Optional<I1>:$ifCond);

  let assemblyFormat = [{
    oilist(`default_async` `(` $defaultAsync `:` type($defaultAsync) `)`
    | `device_num` `(` $deviceNum `:` type($deviceNum) `)`
    | `if` `(` $ifCond `)`
    ) attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.14.4. Update Directive
//===----------------------------------------------------------------------===//

def OpenACC_UpdateOp : OpenACC_Op<"update",
    [AttrSizedOperandSegments,
     MemoryEffects<[MemWrite<OpenACC_ConstructResource>,
                    MemRead<OpenACC_CurrentDeviceIdResource>]>]> {
  let summary = "update operation";

  let description = [{
    The `acc.update` operation represents the OpenACC update executable
    directive.
    As host and self clauses are synonyms, any operands for host and self are
    add to $hostOperands.

    Example:

    ```mlir
    acc.update device(%d1 : memref<10xf32>) attributes {async}
    ```

    `async` and `wait` operands are supported with `device_type` information.
    They should only be accessed by the extra provided getters. If modified,
    the corresponding `device_type` attributes must be modified as well.
  }];

  let arguments = (ins Optional<I1>:$ifCond,
      Variadic<IntOrIndex>:$asyncOperands,
      OptionalAttr<DeviceTypeArrayAttr>:$asyncOperandsDeviceType,
      OptionalAttr<DeviceTypeArrayAttr>:$asyncOnly,
      Variadic<IntOrIndex>:$waitOperands,
      OptionalAttr<DenseI32ArrayAttr>:$waitOperandsSegments,
      OptionalAttr<DeviceTypeArrayAttr>:$waitOperandsDeviceType,
      OptionalAttr<BoolArrayAttr>:$hasWaitDevnum,
      OptionalAttr<DeviceTypeArrayAttr>:$waitOnly,
      Variadic<OpenACC_AnyPointerOrMappableType>:$dataClauseOperands,
      UnitAttr:$ifPresent);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);

    /// Return true if the op has the async attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasAsyncOnly();
    /// Return true if the op has the async attribute for the given device_type.
    bool hasAsyncOnly(mlir::acc::DeviceType deviceType);
    /// Return the value of the async clause if present.
    mlir::Value getAsyncValue();
    /// Return the value of the async clause for the given device_type if
    /// present.
    mlir::Value getAsyncValue(mlir::acc::DeviceType deviceType);

    /// Return true if the op has the wait attribute for the
    /// mlir::acc::DeviceType::None device_type.
    bool hasWaitOnly();
    /// Return true if the op has the wait attribute for the given device_type.
    bool hasWaitOnly(mlir::acc::DeviceType deviceType);
    /// Return the values of the wait clause if present.
    mlir::Operation::operand_range getWaitValues();
    /// Return the values of the wait clause for the given device_type if
    /// present.
    mlir::Operation::operand_range
    getWaitValues(mlir::acc::DeviceType deviceType);
    /// Return the wait devnum value clause if present;
    mlir::Value getWaitDevnum();
    /// Return the wait devnum value clause for the given device_type if
    /// present.
    mlir::Value getWaitDevnum(mlir::acc::DeviceType deviceType);
  }];

  let assemblyFormat = [{
    oilist(
        `if` `(` $ifCond `)`
      | `async` `` custom<DeviceTypeOperandsWithKeywordOnly>($asyncOperands,
            type($asyncOperands), $asyncOperandsDeviceType, $asyncOnly)
      | `wait` `` custom<WaitClause>($waitOperands, type($waitOperands),
          $waitOperandsDeviceType, $waitOperandsSegments, $hasWaitDevnum,
          $waitOnly)
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

    acc.wait does not implement MemoryEffects interface,
    so it affects all the resources. This is conservatively
    correct. More precise modelling of the memory effects
    seems to be impossible without the whole program analysis.
  }];

  let arguments = (ins Variadic<IntOrIndex>:$waitOperands,
                       Optional<IntOrIndex>:$asyncOperand,
                       Optional<IntOrIndex>:$waitDevnum,
                       UnitAttr:$async,
                       Optional<I1>:$ifCond);

  let assemblyFormat = [{
    ( `(` $waitOperands^ `:` type($waitOperands) `)` )?
    oilist(
        `async` `` custom<OperandWithKeywordOnly>($asyncOperand,
            type($asyncOperand), $async)
      |  `wait_devnum` `(` $waitDevnum `:` type($waitDevnum) `)`
      | `if` `(` $ifCond `)`
    ) attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

#endif // OPENACC_OPS
