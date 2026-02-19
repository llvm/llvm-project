//===-- OpenMPEnums.td - OpenMP dialect enum file ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_ENUMS
#define OPENMP_ENUMS

include "mlir/Dialect/OpenMP/OpenMPDialect.td"
include "mlir/IR/EnumAttr.td"

include "mlir/Dialect/OpenMP/OmpCommon.td"

//===----------------------------------------------------------------------===//
// Base classes for OpenMP enum attributes.
//===----------------------------------------------------------------------===//

class OpenMP_I32EnumAttr<string name, string summary,
                         list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::omp";
}

class OpenMP_BitEnumAttr<string name, string summary,
                         list<BitEnumAttrCaseBase> cases>
    : I32BitEnumAttr<name, summary, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::omp";
}

class OpenMP_EnumAttr<EnumAttrInfo enumInfo, string name>
    : EnumAttr<OpenMP_Dialect, enumInfo, name>;


//===----------------------------------------------------------------------===//
// capture_clause enum.
//===----------------------------------------------------------------------===//

def CaptureClauseNone : I32EnumAttrCase<"none", 0>;
def CaptureClauseTo : I32EnumAttrCase<"to", 1>;
def CaptureClauseLink : I32EnumAttrCase<"link", 2>;
def CaptureClauseEnter : I32EnumAttrCase<"enter", 3>;

def DeclareTargetCaptureClause : OpenMP_I32EnumAttr<
    "DeclareTargetCaptureClause",
    "capture clause", [
      CaptureClauseNone,
      CaptureClauseTo,
      CaptureClauseLink,
      CaptureClauseEnter
    ]>;

def DeclareTargetCaptureClauseAttr : OpenMP_EnumAttr<DeclareTargetCaptureClause,
                                                     "capture_clause"> {
  let assemblyFormat = "`(` $value `)`";
}

//===----------------------------------------------------------------------===//
// clause_depend enum.
//===----------------------------------------------------------------------===//

def ClauseDependSource : I32EnumAttrCase<"dependsource", 0>;
def ClauseDependSink : I32EnumAttrCase<"dependsink", 1>;

def ClauseDepend : OpenMP_I32EnumAttr<
    "ClauseDepend",
    "depend clause", [
      ClauseDependSource,
      ClauseDependSink
    ]>;

def ClauseDependAttr : OpenMP_EnumAttr<ClauseDepend, "clause_depend"> {
  let assemblyFormat = "`(` $value `)`";
}

//===----------------------------------------------------------------------===//
// clause_requires enum.
//===----------------------------------------------------------------------===//

// atomic_default_mem_order clause values not defined here because they can be
// represented by the OMPC_MemoryOrder enumeration instead.
def ClauseRequiresNone : I32BitEnumAttrCaseNone<"none">;
def ClauseRequiresReverseOffload : I32BitEnumAttrCaseBit<"reverse_offload", 0>;
def ClauseRequiresUnifiedAddress : I32BitEnumAttrCaseBit<"unified_address", 1>;
def ClauseRequiresUnifiedSharedMemory
    : I32BitEnumAttrCaseBit<"unified_shared_memory", 2>;
def ClauseRequiresDynamicAllocators
    : I32BitEnumAttrCaseBit<"dynamic_allocators", 3>;

def ClauseRequires : OpenMP_BitEnumAttr<
    "ClauseRequires",
    "requires clauses", [
      ClauseRequiresNone,
      ClauseRequiresReverseOffload,
      ClauseRequiresUnifiedAddress,
      ClauseRequiresUnifiedSharedMemory,
      ClauseRequiresDynamicAllocators
    ]>;

def ClauseRequiresAttr : OpenMP_EnumAttr<ClauseRequires, "clause_requires">;


//===----------------------------------------------------------------------===//
// clause_map_flag enum.
//===----------------------------------------------------------------------===//

def ClauseMapFlagsNone : I32BitEnumAttrCaseNone<"none">;
def ClauseMapFlagsStorage : I32BitEnumAttrCaseBit<"storage", 0>; // alloc/release synonym
def ClauseMapFlagsTo : I32BitEnumAttrCaseBit<"to", 1>;
def ClauseMapFlagsFrom : I32BitEnumAttrCaseBit<"from", 2>;
def ClauseMapFlagsAlways : I32BitEnumAttrCaseBit<"always", 3>;
def ClauseMapFlagsDelete : I32BitEnumAttrCaseBit<"del", 4>; // delete, is reserved by C/C++
def ClauseMapFlagsReturnParam : I32BitEnumAttrCaseBit<"return_param", 5>;
def ClauseMapFlagsPrivate : I32BitEnumAttrCaseBit<"priv", 6>; // private, is reserved by C/C++
def ClauseMapFlagsLiteral : I32BitEnumAttrCaseBit<"literal", 7>;
def ClauseMapFlagsImplicit : I32BitEnumAttrCaseBit<"implicit", 8>;
def ClauseMapFlagsClose : I32BitEnumAttrCaseBit<"close", 9>;
def ClauseMapFlagsPresent : I32BitEnumAttrCaseBit<"present", 10>;
def ClauseMapFlagsOMPXHold : I32BitEnumAttrCaseBit<"ompx_hold", 11>;
def ClauseMapFlagsAttach : I32BitEnumAttrCaseBit<"attach", 12>;
def ClauseMapFlagsAttachAlways : I32BitEnumAttrCaseBit<"attach_always", 13>;
def ClauseMapFlagsAttachNever : I32BitEnumAttrCaseBit<"attach_never", 14>;
def ClauseMapFlagsAttachAuto : I32BitEnumAttrCaseBit<"attach_auto", 15>;
def ClauseMapFlagsRefPtr : I32BitEnumAttrCaseBit<"ref_ptr", 16>;
def ClauseMapFlagsRefPtee : I32BitEnumAttrCaseBit<"ref_ptee", 17>;
def ClauseMapFlagsRefPtrPtee : I32BitEnumAttrCaseBit<"ref_ptr_ptee", 18>;
def ClauseMapFlagsIsDevicePtr : I32BitEnumAttrCaseBit<"is_device_ptr", 19>;

def ClauseMapFlags : OpenMP_BitEnumAttr<
    "ClauseMapFlags",
    "Map types and modifiers tied to data maps", [
      ClauseMapFlagsNone,
      ClauseMapFlagsStorage,
      ClauseMapFlagsTo,
      ClauseMapFlagsFrom,
      ClauseMapFlagsAlways,
      ClauseMapFlagsDelete,
      ClauseMapFlagsReturnParam,
      ClauseMapFlagsPrivate,
      ClauseMapFlagsLiteral,
      ClauseMapFlagsImplicit,
      ClauseMapFlagsClose,
      ClauseMapFlagsPresent,
      ClauseMapFlagsOMPXHold,
      ClauseMapFlagsAttach,
      ClauseMapFlagsAttachAlways,
      ClauseMapFlagsAttachNever,
      ClauseMapFlagsAttachAuto,
      ClauseMapFlagsRefPtr,
      ClauseMapFlagsRefPtee,
      ClauseMapFlagsRefPtrPtee,
      ClauseMapFlagsIsDevicePtr
    ]>;

def ClauseMapFlagsAttr : OpenMP_EnumAttr<ClauseMapFlags,
                                         "clause_map_flags">;

//===----------------------------------------------------------------------===//
// clause_task_depend enum.
//===----------------------------------------------------------------------===//

def ClauseTaskDependIn : I32EnumAttrCase<"taskdependin", 0>;
def ClauseTaskDependOut : I32EnumAttrCase<"taskdependout", 1>;
def ClauseTaskDependInOut : I32EnumAttrCase<"taskdependinout", 2>;
def ClauseTaskDependMutexInOutSet
    : I32EnumAttrCase<"taskdependmutexinoutset", 3>;
def ClauseTaskDependInOutSet : I32EnumAttrCase<"taskdependinoutset", 4>;

def ClauseTaskDepend
    : OpenMP_I32EnumAttr<
          "ClauseTaskDepend", "depend clause in a target or task construct",
          [ClauseTaskDependIn, ClauseTaskDependOut, ClauseTaskDependInOut,
           ClauseTaskDependMutexInOutSet, ClauseTaskDependInOutSet]>;

def ClauseTaskDependAttr : OpenMP_EnumAttr<ClauseTaskDepend,
                                           "clause_task_depend"> {
  let assemblyFormat = "`(` $value `)`";
}

//===----------------------------------------------------------------------===//
// data_sharing_type enum.
//===----------------------------------------------------------------------===//

def DataSharingTypePrivate : I32EnumAttrCase<"Private", 0, "private">;
def DataSharingTypeFirstPrivate
    : I32EnumAttrCase<"FirstPrivate", 1, "firstprivate">;

def DataSharingClauseType : OpenMP_I32EnumAttr<
    "DataSharingClauseType",
    "Type of a data-sharing clause", [
      DataSharingTypePrivate,
      DataSharingTypeFirstPrivate
    ]>;

def DataSharingClauseTypeAttr : OpenMP_EnumAttr<DataSharingClauseType,
                                                "data_sharing_type"> {
  let assemblyFormat = "`{` `type` `=` $value `}`";
}

//===----------------------------------------------------------------------===//
// device_type enum.
//===----------------------------------------------------------------------===//

def DeviceTypeAny : I32EnumAttrCase<"any", 0>;
def DeviceTypeHost : I32EnumAttrCase<"host", 1>;
def DeviceTypeNoHost : I32EnumAttrCase<"nohost", 2>;

def DeclareTargetDeviceType : OpenMP_I32EnumAttr<
    "DeclareTargetDeviceType",
    "device_type clause", [
      DeviceTypeAny,
      DeviceTypeHost,
      DeviceTypeNoHost
    ]>;

def DeclareTargetDeviceTypeAttr : OpenMP_EnumAttr<DeclareTargetDeviceType,
                                                  "device_type"> {
  let assemblyFormat = "`(` $value `)`";
}

//===----------------------------------------------------------------------===//
// order_modifer enum.
//===----------------------------------------------------------------------===//

def OMP_OrderModReproducible       : I32EnumAttrCase<"reproducible", 0>;
def OMP_OrderModUnconstrained      : I32EnumAttrCase<"unconstrained", 1>;
def OrderModifier
    : I32EnumAttr<"OrderModifier", "OpenMP Order Modifier",
                  [OMP_OrderModReproducible, OMP_OrderModUnconstrained]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::omp";
}
def OrderModifierAttr : EnumAttr<OpenMP_Dialect, OrderModifier,
                                    "order_mod">;

//===----------------------------------------------------------------------===//
// reduction_modifier enum.
//===----------------------------------------------------------------------===//

def ReductionModifierDefault : I32EnumAttrCase<"defaultmod", 0>;
def ReductionModifierInscan : I32EnumAttrCase<"inscan", 1>;
def ReductionModifierTask : I32EnumAttrCase<"task", 2>;

def ReductionModifier : OpenMP_I32EnumAttr<
    "ReductionModifier",
    "reduction modifier", [
      ReductionModifierDefault,
      ReductionModifierInscan,
      ReductionModifierTask
    ]>;

def ReductionModifierAttr : OpenMP_EnumAttr<ReductionModifier,
                                            "reduction_modifier"> {
  let assemblyFormat = "`(` $value `)`";
}

//===----------------------------------------------------------------------===//
// sched_mod enum.
//===----------------------------------------------------------------------===//

def OpenMP_ScheduleModNone : I32EnumAttrCase<"none", 0>;
def OpenMP_ScheduleModMonotonic : I32EnumAttrCase<"monotonic", 1>;
def OpenMP_ScheduleModNonmonotonic : I32EnumAttrCase<"nonmonotonic", 2>;
// FIXME: remove this value for the modifier because this is handled using a
// separate attribute
def OpenMP_ScheduleModSimd : I32EnumAttrCase<"simd", 3>;

def ScheduleModifier : OpenMP_I32EnumAttr<
    "ScheduleModifier",
    "OpenMP Schedule Modifier", [
      OpenMP_ScheduleModNone,
      OpenMP_ScheduleModMonotonic,
      OpenMP_ScheduleModNonmonotonic,
      OpenMP_ScheduleModSimd
    ]>;

def ScheduleModifierAttr : OpenMP_EnumAttr<ScheduleModifier, "sched_mod">;

//===----------------------------------------------------------------------===//
// target_region_flags enum.
//===----------------------------------------------------------------------===//

def TargetRegionFlagsNone : I32BitEnumAttrCaseNone<"none">;
def TargetRegionFlagsGeneric : I32BitEnumAttrCaseBit<"generic", 0>;
def TargetRegionFlagsSpmd : I32BitEnumAttrCaseBit<"spmd", 1>;
def TargetRegionFlagsTripCount : I32BitEnumAttrCaseBit<"trip_count", 2>;
def TargetRegionFlagsNoLoop : I32BitEnumAttrCaseBit<"no_loop", 3>;

def TargetRegionFlags : OpenMP_BitEnumAttr<
    "TargetRegionFlags",
    "These flags describe properties of the target kernel. "
    "TargetRegionFlagsGeneric - denotes generic kernel. "
    "TargetRegionFlagsSpmd - denotes SPMD kernel. "
    "TargetRegionFlagsNoLoop - denotes kernel where "
    "num_teams * num_threads >= loop_trip_count. It allows the conversion "
    "of loops into sequential code by ensuring that each team/thread "
    "executes at most one iteration. "
    "TargetRegionFlagsTripCount - checks if the loop trip count should be "
    "calculated.", [
      TargetRegionFlagsNone,
      TargetRegionFlagsGeneric,
      TargetRegionFlagsSpmd,
      TargetRegionFlagsTripCount,
      TargetRegionFlagsNoLoop
    ]>;

//===----------------------------------------------------------------------===//
// variable_capture_kind enum.
//===----------------------------------------------------------------------===//

def CaptureThis : I32EnumAttrCase<"This", 0>;
def CaptureByRef : I32EnumAttrCase<"ByRef", 1>;
def CaptureByCopy : I32EnumAttrCase<"ByCopy", 2>;
def CaptureVLAType : I32EnumAttrCase<"VLAType", 3>;

def VariableCaptureKind : OpenMP_I32EnumAttr<
    "VariableCaptureKind",
    "variable capture kind", [
      CaptureThis,
      CaptureByRef,
      CaptureByCopy,
      CaptureVLAType
    ]>;

def VariableCaptureKindAttr : OpenMP_EnumAttr<VariableCaptureKind,
                                              "variable_capture_kind"> {
  let assemblyFormat = "`(` $value `)`";
}

#endif // OPENMP_ENUMS
