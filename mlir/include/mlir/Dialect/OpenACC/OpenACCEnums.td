
//===-- OpenACCEnums.td - OpenACC dialect enum file ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_ENUMS
#define OPENACC_ENUMS

include "mlir/IR/EnumAttr.td"

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

// These are parallelism determination modes for `acc loop`.
// In the enum names, we use the "loop_" prefix because "auto" is
// a language keyword - and thus for consistency all other cases
// do the same.
def OpenACC_LoopSeq : I32EnumAttrCase<"loop_seq", 0>;
def OpenACC_LoopAuto : I32EnumAttrCase<"loop_auto", 1>;
def OpenACC_LoopIndependent : I32EnumAttrCase<"loop_independent", 2>;

def OpenACC_LoopParMode : I32EnumAttr<
    "LoopParMode",
    "Encodes the options for loop parallelism determination mode",
    [
      OpenACC_LoopAuto, OpenACC_LoopIndependent,
      OpenACC_LoopSeq]> {
  let cppNamespace = "::mlir::acc";
  let genSpecializedAttr = 0;
}

// Parallelism level (gang/worker/vector/seq).
// GangDim1 is the default gang level (equivalent to just "gang").
// GangDim2/GangDim3 are for gang(dim:2) and gang(dim:3).
def OpenACC_ParLevelSeq      : I32EnumAttrCase<"seq", 0>;
def OpenACC_ParLevelGangDim1 : I32EnumAttrCase<"gang_dim1", 1>;
def OpenACC_ParLevelGangDim2 : I32EnumAttrCase<"gang_dim2", 2>;
def OpenACC_ParLevelGangDim3 : I32EnumAttrCase<"gang_dim3", 3>;
def OpenACC_ParLevelWorker   : I32EnumAttrCase<"worker", 4>;
def OpenACC_ParLevelVector   : I32EnumAttrCase<"vector", 5>;

def OpenACC_ParLevel : I32EnumAttr<"ParLevel",
    "Parallelism level (gang/worker/vector/seq)",
    [OpenACC_ParLevelSeq,
     OpenACC_ParLevelGangDim1, OpenACC_ParLevelGangDim2,
     OpenACC_ParLevelGangDim3,
     OpenACC_ParLevelWorker, OpenACC_ParLevelVector]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::acc";
}

def OpenACC_ParLevelAttr : EnumAttr<OpenACC_Dialect,
                                    OpenACC_ParLevel,
                                    "par_level"> {
  let assemblyFormat = [{ ```<` $value `>` }];
}

def OpenACC_PrivateRecipe : I32EnumAttrCase<"private_recipe", 0>;
def OpenACC_FirstprivateRecipe : I32EnumAttrCase<"firstprivate_recipe", 1>;
def OpenACC_ReductionRecipe : I32EnumAttrCase<"reduction_recipe", 2>;

def OpenACC_RecipeKind : I32EnumAttr<
    "RecipeKind",
    "Encodes the options for kinds of recipes availabie in acc dialect",
    [
      OpenACC_PrivateRecipe, OpenACC_FirstprivateRecipe,
      OpenACC_ReductionRecipe]> {
  let cppNamespace = "::mlir::acc";
  let genSpecializedAttr = 0;
}

def OpenACC_RecipeKindAttr : EnumAttr<OpenACC_Dialect,
                                             OpenACC_RecipeKind,
                                             "recipe_kind"> {
  let assemblyFormat = [{ ```<` $value `>` }];
}

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

// Data clause modifiers:
// * readonly: Added in OpenACC 2.7 to copyin and cache.
// * zero: Added in OpenACC 3.0 for create and copyout.
// * always, alwaysin, alwaysout: Added in OpenACC 3.4 for
//       copy, copyin, and copyout clauses.
// * capture: Added in OpenACC 3.4 for copy, copyin, copyout and create clauses.
def OpenACC_DataClauseModifierNone : I32BitEnumAttrCaseNone<"none">;
// All of the modifiers below are bit flags - so the value noted is `1 << bit`.
// Thus the `zero` modifier is `1 << 0` = 1, `readonly` is `1 << 1` = 2, etc.
def OpenACC_DataClauseModifierZero : I32BitEnumAttrCaseBit<"zero", 0>;
def OpenACC_DataClauseModifierReadonly : I32BitEnumAttrCaseBit<"readonly", 1>;
def OpenACC_DataClauseModifierAlwaysIn : I32BitEnumAttrCaseBit<"alwaysin", 2>;
def OpenACC_DataClauseModifierAlwaysOut : I32BitEnumAttrCaseBit<"alwaysout", 3>;
def OpenACC_DataClauseModifierAlways : I32BitEnumAttrCaseGroup<"always",
  [OpenACC_DataClauseModifierAlwaysIn, OpenACC_DataClauseModifierAlwaysOut]>;
def OpenACC_DataClauseModifierCapture : I32BitEnumAttrCaseBit<"capture", 4>;

def OpenACC_DataClauseModifierEnum : I32BitEnumAttr<
    "DataClauseModifier",
    "Captures data clause modifiers",
    [
      OpenACC_DataClauseModifierNone, OpenACC_DataClauseModifierZero,
      OpenACC_DataClauseModifierReadonly, OpenACC_DataClauseModifierAlwaysIn,
      OpenACC_DataClauseModifierAlwaysOut, OpenACC_DataClauseModifierAlways,
      OpenACC_DataClauseModifierCapture]> {
  let separator = ",";
  let cppNamespace = "::mlir::acc";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def OpenACC_DataClauseModifierAttr : EnumAttr<OpenACC_Dialect,
                                      OpenACC_DataClauseModifierEnum,
                                      "data_clause_modifier">;

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

#endif // OPENACC_ENUMS