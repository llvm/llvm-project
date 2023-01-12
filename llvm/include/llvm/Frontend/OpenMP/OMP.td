//===-- OMP.td - OpenMP directive definition file ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for OpenMP directives and clauses.
//
//===----------------------------------------------------------------------===//

include "llvm/Frontend/Directive/DirectiveBase.td"

//===----------------------------------------------------------------------===//
// Definition of general OpenMP information
//===----------------------------------------------------------------------===//

def OpenMP : DirectiveLanguage {
  let name = "OpenMP";
  let cppNamespace = "omp"; // final namespace will be llvm::omp
  let directivePrefix = "OMPD_";
  let clausePrefix = "OMPC_";
  let makeEnumAvailableInNamespace = true;
  let enableBitmaskEnumInNamespace = true;
  let clauseEnumSetClass = "OmpClauseSet";
  let flangClauseBaseClass = "OmpClause";
}

//===----------------------------------------------------------------------===//
// Definition of OpenMP clauses
//===----------------------------------------------------------------------===//

def OMPC_Allocator : Clause<"allocator"> {
  let clangClass = "OMPAllocatorClause";
  let flangClass = "ScalarIntExpr";
}
def OMPC_If : Clause<"if"> {
  let clangClass = "OMPIfClause";
  let flangClass = "OmpIfClause";
}
def OMPC_Final : Clause<"final"> {
  let clangClass = "OMPFinalClause";
  let flangClass = "ScalarLogicalExpr";
}
def OMPC_NumThreads : Clause<"num_threads"> {
  let clangClass = "OMPNumThreadsClause";
  let flangClass = "ScalarIntExpr";
}
def OMPC_SafeLen : Clause<"safelen"> {
  let clangClass = "OMPSafelenClause";
  let flangClass = "ScalarIntConstantExpr";
}
def OMPC_SimdLen : Clause<"simdlen"> {
  let clangClass = "OMPSimdlenClause";
  let flangClass = "ScalarIntConstantExpr";
}
def OMPC_Collapse : Clause<"collapse"> {
  let clangClass = "OMPCollapseClause";
  let flangClass = "ScalarIntConstantExpr";
}
def OMPC_Default : Clause<"default"> {
  let clangClass = "OMPDefaultClause";
  let flangClass = "OmpDefaultClause";
}
def OMPC_Private : Clause<"private"> {
  let clangClass = "OMPPrivateClause";
  let flangClass = "OmpObjectList";
}
def OMPC_Sizes: Clause<"sizes"> { let clangClass = "OMPSizesClause"; }
def OMPC_Full: Clause<"full"> { let clangClass = "OMPFullClause"; }
def OMPC_Partial: Clause<"partial"> { let clangClass = "OMPPartialClause"; }
def OMPC_FirstPrivate : Clause<"firstprivate"> {
  let clangClass = "OMPFirstprivateClause";
  let flangClass = "OmpObjectList";
}
def OMPC_LastPrivate : Clause<"lastprivate"> {
  let clangClass = "OMPLastprivateClause";
  let flangClass = "OmpObjectList";
}
def OMPC_Shared : Clause<"shared"> {
  let clangClass = "OMPSharedClause";
  let flangClass = "OmpObjectList";
}
def OMPC_Reduction : Clause<"reduction"> {
  let clangClass = "OMPReductionClause";
  let flangClass = "OmpReductionClause";
}
def OMPC_Linear : Clause<"linear"> {
  let clangClass = "OMPLinearClause";
  let flangClass = "OmpLinearClause";
}
def OMPC_Aligned : Clause<"aligned"> {
  let clangClass = "OMPAlignedClause";
  let flangClass = "OmpAlignedClause";
}
def OMPC_Copyin : Clause<"copyin"> {
  let clangClass = "OMPCopyinClause";
  let flangClass = "OmpObjectList";
}
def OMPC_CopyPrivate : Clause<"copyprivate"> {
  let clangClass = "OMPCopyprivateClause";
  let flangClass = "OmpObjectList";
}
def OMP_PROC_BIND_master : ClauseVal<"master",2,1> {}
def OMP_PROC_BIND_close : ClauseVal<"close",3,1> {}
def OMP_PROC_BIND_spread : ClauseVal<"spread",4,1> {}
def OMP_PROC_BIND_primary : ClauseVal<"primary",5,1> {}
def OMP_PROC_BIND_default : ClauseVal<"default",6,0> {}
def OMP_PROC_BIND_unknown : ClauseVal<"unknown",7,0> { let isDefault = true; }
def OMPC_ProcBind : Clause<"proc_bind"> {
  let clangClass = "OMPProcBindClause";
  let flangClass = "OmpProcBindClause";
  let enumClauseValue = "ProcBindKind";
  let allowedClauseValues = [
    OMP_PROC_BIND_primary,
    OMP_PROC_BIND_master,
    OMP_PROC_BIND_close,
    OMP_PROC_BIND_spread,
    OMP_PROC_BIND_default,
    OMP_PROC_BIND_unknown
  ];
}

def OMP_SCHEDULE_Static : ClauseVal<"static", 2, 1> {}
def OMP_SCHEDULE_Dynamic : ClauseVal<"dynamic", 3, 1> {}
def OMP_SCHEDULE_Guided : ClauseVal<"guided", 4, 1> {}
def OMP_SCHEDULE_Auto : ClauseVal<"auto", 5, 1> {}
def OMP_SCHEDULE_Runtime : ClauseVal<"runtime", 6, 1> {}
def OMP_SCHEDULE_Default : ClauseVal<"default", 7, 0> { let isDefault = 1; }

def OMPC_Schedule : Clause<"schedule"> {
  let clangClass = "OMPScheduleClause";
  let flangClass = "OmpScheduleClause";
  let enumClauseValue = "ScheduleKind";
  let allowedClauseValues = [
    OMP_SCHEDULE_Static,
    OMP_SCHEDULE_Dynamic,
    OMP_SCHEDULE_Guided,
    OMP_SCHEDULE_Auto,
    OMP_SCHEDULE_Runtime,
    OMP_SCHEDULE_Default
  ];
}

def OMP_MEMORY_ORDER_SeqCst : ClauseVal<"seq_cst", 1, 1> {}
def OMP_MEMORY_ORDER_AcqRel : ClauseVal<"acq_rel", 2, 1> {}
def OMP_MEMORY_ORDER_Acquire : ClauseVal<"acquire", 3, 1> {}
def OMP_MEMORY_ORDER_Release : ClauseVal<"release", 4, 1> {}
def OMP_MEMORY_ORDER_Relaxed : ClauseVal<"relaxed", 5, 1> {}
def OMP_MEMORY_ORDER_Default : ClauseVal<"default", 6, 0> {
  let isDefault = 1;
}
def OMPC_MemoryOrder : Clause<"memory_order"> {
  let enumClauseValue = "MemoryOrderKind";
  let allowedClauseValues = [
    OMP_MEMORY_ORDER_SeqCst,
    OMP_MEMORY_ORDER_AcqRel,
    OMP_MEMORY_ORDER_Acquire,
    OMP_MEMORY_ORDER_Release,
    OMP_MEMORY_ORDER_Relaxed,
    OMP_MEMORY_ORDER_Default
  ];
}

def OMP_CANCELLATION_CONSTRUCT_Parallel : ClauseVal<"parallel", 1, 1> {}
def OMP_CANCELLATION_CONSTRUCT_Loop : ClauseVal<"loop", 2, 1> {}
def OMP_CANCELLATION_CONSTRUCT_Sections : ClauseVal<"sections", 3, 1> {}
def OMP_CANCELLATION_CONSTRUCT_Taskgroup : ClauseVal<"taskgroup", 4, 1> {}
def OMP_CANCELLATION_CONSTRUCT_None : ClauseVal<"none", 5, 0> {
  let isDefault = 1;
}

def OMPC_CancellationConstructType : Clause<"cancellation_construct_type"> {
  let enumClauseValue = "CancellationConstructType";
  let allowedClauseValues = [
    OMP_CANCELLATION_CONSTRUCT_Parallel,
    OMP_CANCELLATION_CONSTRUCT_Loop,
    OMP_CANCELLATION_CONSTRUCT_Sections,
    OMP_CANCELLATION_CONSTRUCT_Taskgroup,
    OMP_CANCELLATION_CONSTRUCT_None
  ];
}

def OMPC_Ordered : Clause<"ordered"> {
  let clangClass = "OMPOrderedClause";
  let flangClass = "ScalarIntConstantExpr";
  let isValueOptional = true;
}
def OMPC_NoWait : Clause<"nowait"> {
  let clangClass = "OMPNowaitClause";
}
def OMPC_Untied : Clause<"untied"> { let clangClass = "OMPUntiedClause"; }
def OMPC_Mergeable : Clause<"mergeable"> {
  let clangClass = "OMPMergeableClause";
}
def OMPC_Read : Clause<"read"> { let clangClass = "OMPReadClause"; }
def OMPC_Write : Clause<"write"> { let clangClass = "OMPWriteClause"; }
def OMPC_Update : Clause<"update"> { let clangClass = "OMPUpdateClause"; }
def OMPC_Capture : Clause<"capture"> { let clangClass = "OMPCaptureClause"; }
def OMPC_Compare : Clause<"compare"> { let clangClass = "OMPCompareClause"; }
def OMPC_SeqCst : Clause<"seq_cst"> { let clangClass = "OMPSeqCstClause"; }
def OMPC_AcqRel : Clause<"acq_rel"> { let clangClass = "OMPAcqRelClause"; }
def OMPC_Acquire : Clause<"acquire"> { let clangClass = "OMPAcquireClause"; }
def OMPC_Release : Clause<"release"> { let clangClass = "OMPReleaseClause"; }
def OMPC_Relaxed : Clause<"relaxed"> { let clangClass = "OMPRelaxedClause"; }
def OMPC_Depend : Clause<"depend"> {
  let clangClass = "OMPDependClause";
  let flangClass = "OmpDependClause";
}
def OMPC_Device : Clause<"device"> {
  let clangClass = "OMPDeviceClause";
  let flangClass = "OmpDeviceClause";
}
def OMPC_Threads : Clause<"threads"> { let clangClass = "OMPThreadsClause"; }
def OMPC_Simd : Clause<"simd"> { let clangClass = "OMPSIMDClause"; }
def OMPC_Map : Clause<"map"> {
  let clangClass = "OMPMapClause";
  let flangClass = "OmpMapClause";
}
def OMPC_NumTeams : Clause<"num_teams"> {
  let clangClass = "OMPNumTeamsClause";
  let flangClass = "ScalarIntExpr";
}
def OMPC_ThreadLimit : Clause<"thread_limit"> {
  let clangClass = "OMPThreadLimitClause";
  let flangClass = "ScalarIntExpr";
}
def OMPC_Priority : Clause<"priority"> {
  let clangClass = "OMPPriorityClause";
  let flangClass = "ScalarIntExpr";
}

def OMP_GRAINSIZE_Strict : ClauseVal<"strict", 1, 1> {}
def OMP_GRAINSIZE_Unknown : ClauseVal<"unkonwn", 2, 0> { let isDefault = 1; }

def OMPC_GrainSize : Clause<"grainsize"> {
  let clangClass = "OMPGrainsizeClause";
  let flangClass = "ScalarIntExpr";
  let enumClauseValue = "GrainsizeType";
  let allowedClauseValues = [
    OMP_GRAINSIZE_Strict,
    OMP_GRAINSIZE_Unknown
  ];
}
def OMPC_NoGroup : Clause<"nogroup"> {
  let clangClass = "OMPNogroupClause";
}

def OMP_NUMTASKS_Strict : ClauseVal<"strict", 1, 1> {}
def OMP_NUMTASKS_Unknown : ClauseVal<"unkonwn", 2, 0> { let isDefault = 1; }

def OMPC_NumTasks : Clause<"num_tasks"> {
  let clangClass = "OMPNumTasksClause";
  let flangClass = "ScalarIntExpr";
  let enumClauseValue = "NumTasksType";
  let allowedClauseValues = [
    OMP_NUMTASKS_Strict,
    OMP_NUMTASKS_Unknown
  ];
}
def OMPC_Hint : Clause<"hint"> {
  let clangClass = "OMPHintClause";
  let flangClass = "ConstantExpr";
}
def OMPC_DistSchedule : Clause<"dist_schedule"> {
  let clangClass = "OMPDistScheduleClause";
  let flangClass = "ScalarIntExpr";
  let isValueOptional = true;
}
def OMPC_DefaultMap : Clause<"defaultmap"> {
  let clangClass = "OMPDefaultmapClause";
  let flangClass = "OmpDefaultmapClause";
}
def OMPC_To : Clause<"to"> {
  let clangClass = "OMPToClause";
  let flangClass = "OmpObjectList";
}
def OMPC_From : Clause<"from"> {
  let clangClass = "OMPFromClause";
  let flangClass = "OmpObjectList";
}
def OMPC_UseDevicePtr : Clause<"use_device_ptr"> {
  let clangClass = "OMPUseDevicePtrClause";
  let flangClass = "Name";
  let isValueList = true;
}
def OMPC_IsDevicePtr : Clause<"is_device_ptr"> {
  let clangClass = "OMPIsDevicePtrClause";
  let flangClass = "Name";
  let isValueList = true;
}
def OMPC_HasDeviceAddr : Clause<"has_device_addr"> {
  let clangClass = "OMPHasDeviceAddrClause";
  let flangClass = "Name";
  let isValueList = true;
}
def OMPC_TaskReduction : Clause<"task_reduction"> {
  let clangClass = "OMPTaskReductionClause";
  let flangClass = "OmpReductionClause";
}
def OMPC_InReduction : Clause<"in_reduction"> {
  let clangClass = "OMPInReductionClause";
  let flangClass = "OmpInReductionClause";
}
def OMPC_UnifiedAddress : Clause<"unified_address"> {
  let clangClass = "OMPUnifiedAddressClause";
}
def OMPC_UnifiedSharedMemory : Clause<"unified_shared_memory"> {
  let clangClass = "OMPUnifiedSharedMemoryClause";
}
def OMPC_ReverseOffload : Clause<"reverse_offload"> {
  let clangClass = "OMPReverseOffloadClause";
}
def OMPC_DynamicAllocators : Clause<"dynamic_allocators"> {
  let clangClass = "OMPDynamicAllocatorsClause";
}
def OMPC_AtomicDefaultMemOrder : Clause<"atomic_default_mem_order"> {
  let clangClass = "OMPAtomicDefaultMemOrderClause";
  let flangClass = "OmpAtomicDefaultMemOrderClause";
}
def OMPC_At : Clause<"at"> {
  let clangClass = "OMPAtClause";
}
def OMPC_Severity : Clause<"severity"> {
  let clangClass = "OMPSeverityClause";
}
def OMPC_Message : Clause<"message"> {
  let clangClass = "OMPMessageClause";
}
def OMPC_Allocate : Clause<"allocate"> {
  let clangClass = "OMPAllocateClause";
  let flangClass = "OmpAllocateClause";
}
def OMPC_NonTemporal : Clause<"nontemporal"> {
  let clangClass = "OMPNontemporalClause";
  let flangClass = "Name";
  let isValueList = true;
}

def OMP_ORDER_concurrent : ClauseVal<"concurrent",1,1> {}
def OMP_ORDER_unknown : ClauseVal<"unknown",2,0> { let isDefault = 1; }
def OMPC_Order : Clause<"order"> {
  let clangClass = "OMPOrderClause";
  let enumClauseValue = "OrderKind";
  let allowedClauseValues = [
    OMP_ORDER_unknown,
    OMP_ORDER_concurrent
  ];
}
def OMPC_Init : Clause<"init"> {
  let clangClass = "OMPInitClause";
}
def OMPC_Use : Clause<"use"> {
  let clangClass = "OMPUseClause";
}
def OMPC_Destroy : Clause<"destroy"> {
  let clangClass = "OMPDestroyClause";
}
def OMPC_Novariants : Clause<"novariants"> {
  let clangClass = "OMPNovariantsClause";
  let flangClass = "ScalarLogicalExpr";
}
def OMPC_Nocontext : Clause<"nocontext"> {
  let clangClass = "OMPNocontextClause";
  let flangClass = "ScalarLogicalExpr";
}
def OMPC_Detach : Clause<"detach"> {
  let clangClass = "OMPDetachClause";
}
def OMPC_Inclusive : Clause<"inclusive"> {
  let clangClass = "OMPInclusiveClause";
}
def OMPC_Exclusive : Clause<"exclusive"> {
  let clangClass = "OMPExclusiveClause";
}
def OMPC_UsesAllocators : Clause<"uses_allocators"> {
  let clangClass = "OMPUsesAllocatorsClause";
}
def OMPC_Affinity : Clause<"affinity"> {
  let clangClass = "OMPAffinityClause";
}
def OMPC_UseDeviceAddr : Clause<"use_device_addr"> {
  let clangClass = "OMPUseDeviceAddrClause";
}
def OMPC_Uniform : Clause<"uniform"> {
  let flangClass = "Name";
  let isValueList = true;
}
def OMPC_DeviceType : Clause<"device_type"> {}
def OMPC_Match : Clause<"match"> {}
def OMPC_AdjustArgs : Clause<"adjust_args"> { }
def OMPC_AppendArgs : Clause<"append_args"> { }
def OMPC_Depobj : Clause<"depobj"> {
  let clangClass = "OMPDepobjClause";
  let isImplicit = true;
}
def OMPC_Flush : Clause<"flush"> {
  let clangClass = "OMPFlushClause";
  let isImplicit = true;
}
def OMPC_ThreadPrivate : Clause<"threadprivate"> {
  let alternativeName = "threadprivate or thread local";
  let isImplicit = true;
}
def OMPC_Unknown : Clause<"unknown"> {
  let isImplicit = true;
  let isDefault = true;
}
def OMPC_Link : Clause<"link"> {
  let flangClass = "OmpObjectList";
}
def OMPC_Indirect : Clause<"indirect"> {}
def OMPC_Inbranch : Clause<"inbranch"> {}
def OMPC_Notinbranch : Clause<"notinbranch"> {}
def OMPC_Filter : Clause<"filter"> {
  let clangClass = "OMPFilterClause";
  let flangClass = "ScalarIntExpr";
}
def OMPC_Align : Clause<"align"> {
  let clangClass = "OMPAlignClause";
}
def OMPC_When: Clause<"when"> {}

def OMPC_Bind : Clause<"bind"> {
  let clangClass = "OMPBindClause";
}

//===----------------------------------------------------------------------===//
// Definition of OpenMP directives
//===----------------------------------------------------------------------===//

def OMP_ThreadPrivate : Directive<"threadprivate"> {}
def OMP_Parallel : Directive<"parallel"> {
  let allowedClauses = [
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Allocate>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
  ];
}
def OMP_Task : Directive<"task"> {
  let allowedClauses = [
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Untied>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_InReduction>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Detach, 50>,
    VersionedClause<OMPC_Affinity, 50>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_Priority>
  ];
}
def OMP_Simd : Directive<"simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_If, 50>,
  ];
}
def OMP_Tile : Directive<"tile"> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_Sizes, 51>,
  ];
}
def OMP_Unroll : Directive<"unroll"> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_Full, 51>,
    VersionedClause<OMPC_Partial, 51>,
  ];
}
def OMP_For : Directive<"for"> {
  let allowedClauses = [
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Order, 50>
  ];
}
def OMP_Do : Directive<"do"> {
  let allowedClauses = [
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Reduction>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_NoWait>
  ];
}
def OMP_Sections : Directive<"sections"> {
  let allowedClauses = [
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Allocate>
  ];
}
def OMP_Section : Directive<"section"> {}
def OMP_Single : Directive<"single"> {
  let allowedClauses = [
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_CopyPrivate>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Allocate>
  ];
}
def OMP_Master : Directive<"master"> {}
def OMP_Critical : Directive<"critical"> {
  let allowedClauses = [
    VersionedClause<OMPC_Hint>
  ];
}
def OMP_TaskYield : Directive<"taskyield"> {}
def OMP_Barrier : Directive<"barrier"> {}
def OMP_Error : Directive<"error"> {
  let allowedClauses = [
    VersionedClause<OMPC_At, 51>,
    VersionedClause<OMPC_Severity, 51>,
    VersionedClause<OMPC_Message, 51>
  ];
}
def OMP_TaskWait : Directive<"taskwait"> {
  let allowedClauses = [
    VersionedClause<OMPC_Depend, 50>,
    VersionedClause<OMPC_NoWait, 51>
  ];
}
def OMP_TaskGroup : Directive<"taskgroup"> {
  let allowedClauses = [
    VersionedClause<OMPC_TaskReduction, 50>,
    VersionedClause<OMPC_Allocate, 50>
  ];
}
def OMP_Flush : Directive<"flush"> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_AcqRel, 50>,
    VersionedClause<OMPC_Acquire, 50>,
    VersionedClause<OMPC_Release, 50>,
    // TODO This should ne `none` instead. Comment carried over from
    // OMPKinds.def.
    VersionedClause<OMPC_Flush>
  ];
}
def OMP_Ordered : Directive<"ordered"> {
  let allowedClauses = [
    VersionedClause<OMPC_Depend>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Threads>,
    VersionedClause<OMPC_Simd>
  ];
}
def OMP_Atomic : Directive<"atomic"> {
  let allowedClauses = [
    VersionedClause<OMPC_Read>,
    VersionedClause<OMPC_Write>,
    VersionedClause<OMPC_Update>,
    VersionedClause<OMPC_Capture>,
    VersionedClause<OMPC_Compare, 51>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_SeqCst>,
    VersionedClause<OMPC_AcqRel, 50>,
    VersionedClause<OMPC_Acquire, 50>,
    VersionedClause<OMPC_Release, 50>,
    VersionedClause<OMPC_Relaxed, 50>,
    VersionedClause<OMPC_Hint, 50>
  ];
}
def OMP_Target : Directive<"target"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_InReduction, 50>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_UsesAllocators, 50>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_ThreadLimit, 51>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_NoWait>
  ];
}
def OMP_Teams : Directive<"teams"> {
  let allowedClauses = [
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Allocate>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>
  ];
}
def OMP_Cancel : Directive<"cancel"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>
  ];
}
def OMP_Requires : Directive<"requires"> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_UnifiedAddress>,
    VersionedClause<OMPC_UnifiedSharedMemory>,
    // OpenMP 5.2 Spec: If an implementation is not supporting a requirement
    // (reverse offload in this case) then it should give compile-time error
    // termination.
    // Seeting supported version for reverse_offload to a distant future version
    // 9.9 so that its partial support can be tested in the meantime.
    //
    // TODO: Correct this supprted version number whenever complete
    // implementation of reverse_offload is available.
    VersionedClause<OMPC_ReverseOffload, 99>,
    VersionedClause<OMPC_DynamicAllocators>,
    VersionedClause<OMPC_AtomicDefaultMemOrder>
  ];
}
def OMP_Nothing : Directive<"nothing"> {}
def OMP_TargetData : Directive<"target data"> {
  let allowedClauses = [
    VersionedClause<OMPC_UseDevicePtr>,
    VersionedClause<OMPC_UseDeviceAddr, 50>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_If>
  ];
  let requiredClauses = [
    VersionedClause<OMPC_Map>
  ];
}
def OMP_TargetEnterData : Directive<"target enter data"> {
  let allowedClauses = [
    VersionedClause<OMPC_Depend>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_NoWait>
  ];
  let requiredClauses = [
    VersionedClause<OMPC_Map>
  ];
}
def OMP_TargetExitData : Directive<"target exit data"> {
  let allowedClauses = [
    VersionedClause<OMPC_Depend>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NoWait>
  ];
  let requiredClauses = [
    VersionedClause<OMPC_Map>
  ];
}
def OMP_TargetParallel : Directive<"target parallel"> {
  let allowedClauses = [
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_UsesAllocators, 50>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>
  ];
}
def OMP_TargetParallelFor : Directive<"target parallel for"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_UsesAllocators, 50>
  ];
}
def OMP_TargetParallelDo : Directive<"target parallel do"> {
  let allowedClauses = [
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_Allocator>,
    VersionedClause<OMPC_Order>,
    VersionedClause<OMPC_UsesAllocators>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Copyin>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_NoWait>
  ];
}
def OMP_TargetUpdate : Directive<"target update"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_To>,
    VersionedClause<OMPC_From>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Depend>
  ];
}
def OMP_ParallelFor : Directive<"parallel for"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Order, 50>
  ];
}
def OMP_ParallelDo : Directive<"parallel do"> {
  let allowedClauses = [
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Collapse>
  ];
}
def OMP_ParallelForSimd : Directive<"parallel for simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>
  ];
}
def OMP_ParallelDoSimd : Directive<"parallel do simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NonTemporal>,
    VersionedClause<OMPC_Order>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>
  ];
}
def OMP_ParallelMaster : Directive<"parallel master"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Allocate>
  ];
}
def OMP_ParallelMasked : Directive<"parallel masked"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Filter>
  ];
}
def OMP_ParallelSections : Directive<"parallel sections"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Allocate>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_NumThreads>
  ];
}
def OMP_ForSimd : Directive<"for simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_If, 50>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>,
  ];
}
def OMP_DoSimd : Directive<"do simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Reduction>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_NoWait>
  ];
}
def OMP_CancellationPoint : Directive<"cancellation point"> {}
def OMP_DeclareReduction : Directive<"declare reduction"> {}
def OMP_DeclareMapper : Directive<"declare mapper"> {
  let allowedClauses = [
    VersionedClause<OMPC_Map>
  ];
}
def OMP_DeclareSimd : Directive<"declare simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Uniform>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_SimdLen>
  ];
  let allowedExclusiveClauses = [
    VersionedClause<OMPC_Inbranch>,
    VersionedClause<OMPC_Notinbranch>
  ];
}
def OMP_TaskLoop : Directive<"taskloop"> {
  let allowedClauses = [
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Untied>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_InReduction>,
    VersionedClause<OMPC_Allocate>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_Priority>,
  ];
  let allowedExclusiveClauses = [
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_NumTasks>
  ];
}
def OMP_TaskLoopSimd : Directive<"taskloop simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_InReduction>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Untied>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_Priority>
  ];
  let allowedExclusiveClauses = [
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_NumTasks>
  ];
}
def OMP_Distribute : Directive<"distribute"> {
  let allowedClauses = [
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Allocate>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>
  ];
}
def OMP_BeginDeclareTarget : Directive<"begin declare target"> {
  let allowedClauses = [
    VersionedClause<OMPC_To>,
    VersionedClause<OMPC_Link>,
    VersionedClause<OMPC_DeviceType>,
    VersionedClause<OMPC_Indirect>
  ];
}
def OMP_DeclareTarget : Directive<"declare target"> {
  let allowedClauses = [
    VersionedClause<OMPC_To>,
    VersionedClause<OMPC_Link>,
    VersionedClause<OMPC_Indirect>
  ];
}
def OMP_EndDeclareTarget : Directive<"end declare target"> {}
def OMP_DistributeParallelFor : Directive<"distribute parallel for"> {
  let allowedClauses = [
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Order, 50>
  ];
}
def OMP_DistributeParallelDo : Directive<"distribute parallel do"> {
  let allowedClauses = [
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Order>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Linear>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Ordered>
  ];
}
def OMP_DistributeParallelForSimd : Directive<"distribute parallel for simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>
  ];
}
def OMP_DistributeParallelDoSimd : Directive<"distribute parallel do simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NonTemporal>,
    VersionedClause<OMPC_Order>
  ];
}
def OMP_DistributeSimd : Directive<"distribute simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_If, 50>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>
  ];
}

def OMP_TargetParallelForSimd : Directive<"target parallel for simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_UsesAllocators, 50>
  ];
}
def OMP_TargetParallelDoSimd : Directive<"target parallel do simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NonTemporal>,
    VersionedClause<OMPC_Order>,
    VersionedClause<OMPC_UsesAllocators>
  ];
}
def OMP_TargetSimd : Directive<"target simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators, 50>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Schedule>
  ];
}
def OMP_TeamsDistribute : Directive<"teams distribute"> {
  let allowedClauses = [
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_Allocate>
  ];
}
def OMP_TeamsDistributeSimd : Directive<"teams distribute simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_If, 50>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_ThreadLimit>
  ];
}

def OMP_TeamsDistributeParallelForSimd :
    Directive<"teams distribute parallel for simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>
  ];
}
def OMP_TeamsDistributeParallelDoSimd :
    Directive<"teams distribute parallel do simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Order>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_NonTemporal>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_If>,
  ];
}
def OMP_TeamsDistributeParallelFor :
    Directive<"teams distribute parallel for"> {
  let allowedClauses = [
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Order, 50>
  ];
}
def OMP_TeamsDistributeParallelDo :
    Directive<"teams distribute parallel do"> {
  let allowedClauses = [
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Linear>
  ];
let allowedOnceClauses = [
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Order>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>
  ];
}
def OMP_TargetTeams : Directive<"target teams"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_UsesAllocators, 50>,
    VersionedClause<OMPC_Shared>
  ];

  let allowedOnceClauses = [
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>
  ];
}
def OMP_TargetTeamsDistribute : Directive<"target teams distribute"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_UsesAllocators, 50>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_LastPrivate>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>
  ];
}

def OMP_TargetTeamsDistributeParallelFor :
    Directive<"target teams distribute parallel for"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_UsesAllocators, 50>
  ];
}
def OMP_TargetTeamsDistributeParallelDo :
    Directive<"target teams distribute parallel do"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_UsesAllocators>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Order>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>,
  ];
}
def OMP_TargetTeamsDistributeParallelForSimd :
    Directive<"target teams distribute parallel for simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_UsesAllocators, 50>
  ];
}
def OMP_TargetTeamsDistributeParallelDoSimd :
    Directive<"target teams distribute parallel do simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_UsesAllocators>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Order>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_NonTemporal>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>
  ];
}
def OMP_TargetTeamsDistributeSimd :
    Directive<"target teams distribute simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators, 50>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
  ];
}
def OMP_Allocate : Directive<"allocate"> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_Allocator>,
    VersionedClause<OMPC_Align, 51>
  ];
}
def OMP_DeclareVariant : Directive<"declare variant"> {
  let allowedClauses = [
    VersionedClause<OMPC_Match>
  ];
  let allowedExclusiveClauses = [
    VersionedClause<OMPC_AdjustArgs, 51>,
    VersionedClause<OMPC_AppendArgs, 51>
  ];
}
def OMP_MasterTaskloop : Directive<"master taskloop"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_Untied>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_InReduction>,
    VersionedClause<OMPC_Allocate>
  ];
}
def OMP_MaskedTaskloop : Directive<"masked taskloop"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_Untied>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_InReduction>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Filter>
  ];
}
def OMP_ParallelMasterTaskloop :
    Directive<"parallel master taskloop"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_Untied>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Copyin>
  ];
}
def OMP_ParallelMaskedTaskloop :
    Directive<"parallel masked taskloop"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_Untied>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Filter>
  ];
}
def OMP_MasterTaskloopSimd : Directive<"master taskloop simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_Untied>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_InReduction>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>
  ];
}
def OMP_MaskedTaskloopSimd : Directive<"masked taskloop simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_Untied>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_InReduction>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Filter>
  ];
}
def OMP_ParallelMasterTaskloopSimd :
    Directive<"parallel master taskloop simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_Untied>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>
  ];
}
def OMP_ParallelMaskedTaskloopSimd :
    Directive<"parallel masked taskloop simd"> {
  let allowedClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_Untied>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Filter>
  ];
}
def OMP_Depobj : Directive<"depobj"> {
  let allowedClauses = [
    VersionedClause<OMPC_Depend, 50>,
    VersionedClause<OMPC_Destroy, 50>,
    VersionedClause<OMPC_Update, 50>,
    // TODO This should ne `none` instead. Comment carried over from
    // OMPKinds.def.
    VersionedClause<OMPC_Depobj, 50>
  ];
}
def OMP_Scan : Directive<"scan"> {
  let allowedClauses = [
    VersionedClause<OMPC_Inclusive, 50>,
    VersionedClause<OMPC_Exclusive, 50>
  ];
}
def OMP_Assumes : Directive<"assumes"> {}
def OMP_BeginAssumes : Directive<"begin assumes"> {}
def OMP_EndAssumes : Directive<"end assumes"> {}
def OMP_BeginDeclareVariant : Directive<"begin declare variant"> {}
def OMP_EndDeclareVariant : Directive<"end declare variant"> {}
def OMP_ParallelWorkshare : Directive<"parallel workshare"> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>
  ];
}
def OMP_Workshare : Directive<"workshare"> {}
def OMP_EndDo : Directive<"end do"> {}
def OMP_EndDoSimd : Directive<"end do simd"> {}
def OMP_EndSections : Directive<"end sections"> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_NoWait>
  ];
}
def OMP_EndSingle : Directive<"end single"> {
  let allowedClauses = [
    VersionedClause<OMPC_CopyPrivate>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_NoWait>
  ];
}
def OMP_EndWorkshare : Directive<"end workshare"> {
  let allowedClauses = [
    VersionedClause<OMPC_NoWait>
  ];
}
def OMP_interop : Directive<"interop"> {
  let allowedClauses = [
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Destroy>,
    VersionedClause<OMPC_Init>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Use>,
  ];
}
def OMP_dispatch : Directive<"dispatch"> {
  let allowedClauses = [
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Novariants>,
    VersionedClause<OMPC_Nocontext>
  ];
}
def OMP_masked : Directive<"masked"> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_Filter>
  ];
}
def OMP_loop : Directive<"loop"> {
  let allowedClauses = [
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Bind, 50>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Order>,
  ];
}
def OMP_teams_loop : Directive<"teams loop"> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Bind, 50>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_Order>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
}
def OMP_target_teams_loop : Directive<"target teams loop"> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators, 50>
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Bind, 50>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_Order>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
}
def OMP_parallel_loop : Directive<"parallel loop"> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Bind, 50>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Order>,
    VersionedClause<OMPC_ProcBind>,
  ];
}
def OMP_target_parallel_loop : Directive<"target parallel loop"> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators, 50>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Bind, 50>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Order>,
    VersionedClause<OMPC_ProcBind>,
  ];
}
def OMP_Metadirective : Directive<"metadirective"> {
  let allowedClauses = [VersionedClause<OMPC_When>];
  let allowedOnceClauses = [VersionedClause<OMPC_Default>];
}
def OMP_Unknown : Directive<"unknown"> {
  let isDefault = true;
}
