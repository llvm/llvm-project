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
// Definitions of OpenMP clauses
// Sorted alphabetically wrt clause spelling.
//===----------------------------------------------------------------------===//

def OMPC_Absent : Clause<[Spelling<"absent">]> {
  let clangClass = "OMPAbsentClause";
  let flangClass = "OmpAbsentClause";
}
def OMPC_Acquire : Clause<[Spelling<"acquire">]> {
  let clangClass = "OMPAcquireClause";
}
def OMPC_AcqRel : Clause<[Spelling<"acq_rel">]> {
  let clangClass = "OMPAcqRelClause";
}
def OMPC_AdjustArgs : Clause<[Spelling<"adjust_args">]> {
  let flangClass = "OmpAdjustArgsClause";
}
def OMPC_Affinity : Clause<[Spelling<"affinity">]> {
  let clangClass = "OMPAffinityClause";
  let flangClass = "OmpAffinityClause";
}
def OMPC_Align : Clause<[Spelling<"align">]> {
  let clangClass = "OMPAlignClause";
  let flangClass = "OmpAlignClause";
}
def OMPC_Aligned : Clause<[Spelling<"aligned">]> {
  let clangClass = "OMPAlignedClause";
  let flangClass = "OmpAlignedClause";
}
def OMPC_Allocate : Clause<[Spelling<"allocate">]> {
  let clangClass = "OMPAllocateClause";
  let flangClass = "OmpAllocateClause";
}
def OMPC_Allocator : Clause<[Spelling<"allocator">]> {
  let clangClass = "OMPAllocatorClause";
  let flangClass = "ScalarIntExpr";
}
def OMPC_AppendArgs : Clause<[Spelling<"append_args">]> {
  let flangClass = "OmpAppendArgsClause";
}
def OMPC_At : Clause<[Spelling<"at">]> {
  let clangClass = "OMPAtClause";
  let flangClass = "OmpAtClause";
}
def OMPC_AtomicDefaultMemOrder
    : Clause<[Spelling<"atomic_default_mem_order">]> {
  let clangClass = "OMPAtomicDefaultMemOrderClause";
  let flangClass = "OmpAtomicDefaultMemOrderClause";
}

def OMP_BIND_parallel : EnumVal<"parallel",1,1> {}
def OMP_BIND_teams : EnumVal<"teams",2,1> {}
def OMP_BIND_thread : EnumVal<"thread",3,1> { let isDefault = true; }
def OMPC_Bind : Clause<[Spelling<"bind">]> {
  let clangClass = "OMPBindClause";
  let flangClass = "OmpBindClause";
  let enumClauseValue = "BindKind";
  let allowedClauseValues = [
    OMP_BIND_parallel,
    OMP_BIND_teams,
    OMP_BIND_thread
  ];
}

def OMP_CANCELLATION_CONSTRUCT_Parallel : EnumVal<"parallel", 1, 1> {}
def OMP_CANCELLATION_CONSTRUCT_Loop : EnumVal<"loop", 2, 1> {}
def OMP_CANCELLATION_CONSTRUCT_Sections : EnumVal<"sections", 3, 1> {}
def OMP_CANCELLATION_CONSTRUCT_Taskgroup : EnumVal<"taskgroup", 4, 1> {}
def OMP_CANCELLATION_CONSTRUCT_None : EnumVal<"none", 5, 0> {
  let isDefault = 1;
}
def OMPC_CancellationConstructType
    : Clause<[Spelling<"cancellation_construct_type">]> {
  let enumClauseValue = "CancellationConstructType";
  let allowedClauseValues = [
    OMP_CANCELLATION_CONSTRUCT_Parallel,
    OMP_CANCELLATION_CONSTRUCT_Loop,
    OMP_CANCELLATION_CONSTRUCT_Sections,
    OMP_CANCELLATION_CONSTRUCT_Taskgroup,
    OMP_CANCELLATION_CONSTRUCT_None
  ];
  let flangClass = "OmpCancellationConstructTypeClause";
  let skipFlangUnparser = true;
}
def OMPC_Contains : Clause<[Spelling<"contains">]> {
  let clangClass = "OMPContainsClause";
  let flangClass = "OmpContainsClause";
}
def OMPC_Capture : Clause<[Spelling<"capture">]> {
  let clangClass = "OMPCaptureClause";
}
def OMPC_Collapse : Clause<[Spelling<"collapse">]> {
  let clangClass = "OMPCollapseClause";
  let flangClass = "ScalarIntConstantExpr";
}
def OMPC_Compare : Clause<[Spelling<"compare">]> {
  let clangClass = "OMPCompareClause";
}
def OMPC_Copyin : Clause<[Spelling<"copyin">]> {
  let clangClass = "OMPCopyinClause";
  let flangClass = "OmpObjectList";
}
def OMPC_CopyPrivate : Clause<[Spelling<"copyprivate">]> {
  let clangClass = "OMPCopyprivateClause";
  let flangClass = "OmpObjectList";
}
def OMPC_Default : Clause<[Spelling<"default">]> {
  let clangClass = "OMPDefaultClause";
  let flangClass = "OmpDefaultClause";
}
def OMPC_DefaultMap : Clause<[Spelling<"defaultmap">]> {
  let clangClass = "OMPDefaultmapClause";
  let flangClass = "OmpDefaultmapClause";
}
def OMPC_Depend : Clause<[Spelling<"depend">]> {
  let clangClass = "OMPDependClause";
  let flangClass = "OmpDependClause";
}
def OMPC_Depobj : Clause<[Spelling<"depobj">]> {
  let clangClass = "OMPDepobjClause";
  let isImplicit = true;
}
def OMPC_Destroy : Clause<[Spelling<"destroy">]> {
  let clangClass = "OMPDestroyClause";
  let flangClass = "OmpDestroyClause";
  let isValueOptional = true;
}
def OMPC_Detach : Clause<[Spelling<"detach">]> {
  let clangClass = "OMPDetachClause";
  let flangClass = "OmpDetachClause";
}
def OMPC_Device : Clause<[Spelling<"device">]> {
  let clangClass = "OMPDeviceClause";
  let flangClass = "OmpDeviceClause";
}
def OMPC_DeviceType : Clause<[Spelling<"device_type">]> {
  let flangClass = "OmpDeviceTypeClause";
}
def OMPC_DistSchedule : Clause<[Spelling<"dist_schedule">]> {
  let clangClass = "OMPDistScheduleClause";
  let flangClass = "ScalarIntExpr";
  let isValueOptional = true;
}
def OMPC_Doacross : Clause<[Spelling<"doacross">]> {
  let clangClass = "OMPDoacrossClause";
  let flangClass = "OmpDoacrossClause";
}
def OMPC_DynamicAllocators : Clause<[Spelling<"dynamic_allocators">]> {
  let clangClass = "OMPDynamicAllocatorsClause";
}
def OMPC_DynGroupprivate : Clause<[Spelling<"dyn_groupprivate">]> {
  let flangClass = "OmpDynGroupprivateClause";
}
def OMPC_Enter : Clause<[Spelling<"enter">]> {
  let flangClass = "OmpEnterClause";
}
def OMPC_Exclusive : Clause<[Spelling<"exclusive">]> {
  let clangClass = "OMPExclusiveClause";
  let flangClass = "OmpObjectList";
}
def OMPC_Fail : Clause<[Spelling<"fail">]> {
  let clangClass = "OMPFailClause";
  let flangClass = "OmpFailClause";
}
def OMPC_Filter : Clause<[Spelling<"filter">]> {
  let clangClass = "OMPFilterClause";
  let flangClass = "ScalarIntExpr";
}
def OMPC_Final : Clause<[Spelling<"final">]> {
  let clangClass = "OMPFinalClause";
  let flangClass = "ScalarLogicalExpr";
}
def OMPC_FirstPrivate : Clause<[Spelling<"firstprivate">]> {
  let clangClass = "OMPFirstprivateClause";
  let flangClass = "OmpObjectList";
}
def OMPC_Flush : Clause<[Spelling<"flush">]> {
  let clangClass = "OMPFlushClause";
  let isImplicit = true;
}
def OMPC_From : Clause<[Spelling<"from">]> {
  let clangClass = "OMPFromClause";
  let flangClass = "OmpFromClause";
}
def OMPC_Full: Clause<[Spelling<"full">]> {
  let clangClass = "OMPFullClause";
}
def OMP_GRAINSIZE_Strict : EnumVal<"strict", 1, 1> {}
def OMP_GRAINSIZE_Unknown : EnumVal<"unknown", 2, 0> { let isDefault = 1; }
def OMPC_GrainSize : Clause<[Spelling<"grainsize">]> {
  let clangClass = "OMPGrainsizeClause";
  let flangClass = "OmpGrainsizeClause";
  let enumClauseValue = "GrainsizeType";
  let allowedClauseValues = [
    OMP_GRAINSIZE_Strict,
    OMP_GRAINSIZE_Unknown
  ];
}
def OMPC_GraphId : Clause<[Spelling<"graph_id">]> {
  let flangClass = "OmpGraphIdClause";
}
def OMPC_GraphReset : Clause<[Spelling<"graph_reset">]> {
  let flangClass = "OmpGraphResetClause";
  let isValueOptional = true;
}
def OMPC_HasDeviceAddr : Clause<[Spelling<"has_device_addr">]> {
  let clangClass = "OMPHasDeviceAddrClause";
  let flangClass = "OmpObjectList";
}
def OMPC_Hint : Clause<[Spelling<"hint">]> {
  let clangClass = "OMPHintClause";
  let flangClass = "OmpHintClause";
}
def OMPC_Holds : Clause<[Spelling<"holds">]> {
  let clangClass = "OMPHoldsClause";
  let flangClass = "OmpHoldsClause";
}
def OMPC_If : Clause<[Spelling<"if">]> {
  let clangClass = "OMPIfClause";
  let flangClass = "OmpIfClause";
}
def OMPC_Inbranch : Clause<[Spelling<"inbranch">]> {
}
def OMPC_Inclusive : Clause<[Spelling<"inclusive">]> {
  let clangClass = "OMPInclusiveClause";
  let flangClass = "OmpObjectList";
}
def OMPC_Indirect : Clause<[Spelling<"indirect">]> {
  let flangClass = "OmpIndirectClause";
}
def OMPC_Init : Clause<[Spelling<"init">]> {
  let clangClass = "OMPInitClause";
  let flangClass = "OmpInitClause";
}
def OMPC_Initializer : Clause<[Spelling<"initializer">]> {
  let flangClass = "OmpInitializerClause";
}
def OMPC_InReduction : Clause<[Spelling<"in_reduction">]> {
  let clangClass = "OMPInReductionClause";
  let flangClass = "OmpInReductionClause";
}
def OMPC_IsDevicePtr : Clause<[Spelling<"is_device_ptr">]> {
  let clangClass = "OMPIsDevicePtrClause";
  let flangClass = "OmpObjectList";
}
def OMPC_LastPrivate : Clause<[Spelling<"lastprivate">]> {
  let clangClass = "OMPLastprivateClause";
  let flangClass = "OmpLastprivateClause";
}
def OMPC_Linear : Clause<[Spelling<"linear">]> {
  let clangClass = "OMPLinearClause";
  let flangClass = "OmpLinearClause";
}
def OMPC_Link : Clause<[Spelling<"link">]> {
  let flangClass = "OmpObjectList";
}
def OMPC_LoopRange : Clause<[Spelling<"looprange">]> {
  let clangClass = "OMPLoopRangeClause";
  let flangClass = "OmpLoopRangeClause";
}
def OMPC_Map : Clause<[Spelling<"map">]> {
  let clangClass = "OMPMapClause";
  let flangClass = "OmpMapClause";
}
def OMPC_Match : Clause<[Spelling<"match">]> {
  let flangClass = "OmpMatchClause";
}
def OMP_MEMORY_ORDER_SeqCst : EnumVal<"seq_cst", 1, 1> {}
def OMP_MEMORY_ORDER_AcqRel : EnumVal<"acq_rel", 2, 1> {}
def OMP_MEMORY_ORDER_Acquire : EnumVal<"acquire", 3, 1> {}
def OMP_MEMORY_ORDER_Release : EnumVal<"release", 4, 1> {}
def OMP_MEMORY_ORDER_Relaxed : EnumVal<"relaxed", 5, 1> {}
def OMP_MEMORY_ORDER_Default : EnumVal<"default", 6, 0> {
  let isDefault = 1;
}
def OMPC_MemoryOrder : Clause<[Spelling<"memory_order">]> {
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
def OMPC_Mergeable : Clause<[Spelling<"mergeable">]> {
  let clangClass = "OMPMergeableClause";
}
def OMPC_Message : Clause<[Spelling<"message">]> {
  let clangClass = "OMPMessageClause";
  let flangClass = "OmpMessageClause";
}
def OMPC_NoOpenMP : Clause<[Spelling<"no_openmp">]> {
  let clangClass = "OMPNoOpenMPClause";
}
def OMPC_NoOpenMPRoutines : Clause<[Spelling<"no_openmp_routines">]> {
  let clangClass = "OMPNoOpenMPRoutinesClause";
}
def OMPC_NoOpenMPConstructs : Clause<[Spelling<"no_openmp_constructs">]> {
  let clangClass = "OMPNoOpenMPConstructsClause";
}
def OMPC_NoParallelism : Clause<[Spelling<"no_parallelism">]> {
  let clangClass = "OMPNoParallelismClause";
}
def OMPC_Nocontext : Clause<[Spelling<"nocontext">]> {
  let clangClass = "OMPNocontextClause";
  let flangClass = "ScalarLogicalExpr";
}
def OMPC_NoGroup : Clause<[Spelling<"nogroup">]> {
  let clangClass = "OMPNogroupClause";
}
def OMPC_NonTemporal : Clause<[Spelling<"nontemporal">]> {
  let clangClass = "OMPNontemporalClause";
  let flangClass = "Name";
  let isValueList = true;
}
def OMPC_Notinbranch : Clause<[Spelling<"notinbranch">]> {
}
def OMPC_Novariants : Clause<[Spelling<"novariants">]> {
  let clangClass = "OMPNovariantsClause";
  let flangClass = "ScalarLogicalExpr";
}
def OMPC_NoWait : Clause<[Spelling<"nowait">]> {
  let clangClass = "OMPNowaitClause";
  let isValueOptional = true;
}
def OMP_NUMTASKS_Strict : EnumVal<"strict", 1, 1> {}
def OMP_NUMTASKS_Unknown : EnumVal<"unknown", 2, 0> { let isDefault = 1; }
def OMPC_NumTasks : Clause<[Spelling<"num_tasks">]> {
  let clangClass = "OMPNumTasksClause";
  let flangClass = "OmpNumTasksClause";
  let enumClauseValue = "NumTasksType";
  let allowedClauseValues = [
    OMP_NUMTASKS_Strict,
    OMP_NUMTASKS_Unknown
  ];
}
def OMPC_NumTeams : Clause<[Spelling<"num_teams">]> {
  let clangClass = "OMPNumTeamsClause";
  let flangClass = "ScalarIntExpr";
}
def OMP_NUMTHREADS_Strict : EnumVal<"strict", 1, 1> {}
def OMP_NUMTHREADS_Unknown : EnumVal<"unknown", 2, 0> { let isDefault = 1; }
def OMPC_NumThreads : Clause<[Spelling<"num_threads">]> {
  let clangClass = "OMPNumThreadsClause";
  let flangClass = "ScalarIntExpr";
  let enumClauseValue = "NumThreadsType";
  let allowedClauseValues = [
    OMP_NUMTHREADS_Strict,
    OMP_NUMTHREADS_Unknown
  ];
}
def OMPC_OMPX_Attribute : Clause<[Spelling<"ompx_attribute">]> {
  let clangClass = "OMPXAttributeClause";
}
def OMPC_OMPX_Bare : Clause<[Spelling<"ompx_bare">]> {
  let clangClass = "OMPXBareClause";
}
def OMPC_OMPX_DynCGroupMem : Clause<[Spelling<"ompx_dyn_cgroup_mem">]> {
  let clangClass = "OMPXDynCGroupMemClause";
  let flangClass = "ScalarIntExpr";
}
def OMP_ORDER_concurrent : EnumVal<"concurrent",1,1> {}
def OMP_ORDER_unknown : EnumVal<"unknown",2,0> { let isDefault = 1; }
def OMPC_Order : Clause<[Spelling<"order">]> {
  let clangClass = "OMPOrderClause";
  let flangClass = "OmpOrderClause";
  let enumClauseValue = "OrderKind";
  let allowedClauseValues = [
    OMP_ORDER_unknown,
    OMP_ORDER_concurrent
  ];
}
def OMPC_Ordered : Clause<[Spelling<"ordered">]> {
  let clangClass = "OMPOrderedClause";
  let flangClass = "ScalarIntConstantExpr";
  let isValueOptional = true;
}
def OMPC_Otherwise : Clause<[Spelling<"otherwise">]> {
  let flangClass = "OmpOtherwiseClause";
  let isValueOptional = true;
}
def OMPC_Partial: Clause<[Spelling<"partial">]> {
  let clangClass = "OMPPartialClause";
  let flangClass = "ScalarIntConstantExpr";
  let isValueOptional = true;
}
def OMPC_Permutation: Clause<[Spelling<"permutation">]> {
  let clangClass = "OMPPermutationClause";
  let flangClass = "ScalarIntExpr";
  let isValueList = true;
}
def OMPC_Priority : Clause<[Spelling<"priority">]> {
  let clangClass = "OMPPriorityClause";
  let flangClass = "ScalarIntExpr";
}
def OMPC_Private : Clause<[Spelling<"private">]> {
  let clangClass = "OMPPrivateClause";
  let flangClass = "OmpObjectList";
}
def OMP_PROC_BIND_master : EnumVal<"master",2,1> {}
def OMP_PROC_BIND_close : EnumVal<"close",3,1> {}
def OMP_PROC_BIND_spread : EnumVal<"spread",4,1> {}
def OMP_PROC_BIND_primary : EnumVal<"primary",5,1> {}
def OMP_PROC_BIND_default : EnumVal<"default",6,0> {}
def OMP_PROC_BIND_unknown : EnumVal<"unknown",7,0> { let isDefault = true; }
def OMPC_ProcBind : Clause<[Spelling<"proc_bind">]> {
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
def OMPC_Read : Clause<[Spelling<"read">]> {
  let clangClass = "OMPReadClause";
}
def OMPC_Reduction : Clause<[Spelling<"reduction">]> {
  let clangClass = "OMPReductionClause";
  let flangClass = "OmpReductionClause";
}
def OMPC_Relaxed : Clause<[Spelling<"relaxed">]> {
  let clangClass = "OMPRelaxedClause";
}
def OMPC_Release : Clause<[Spelling<"release">]> {
  let clangClass = "OMPReleaseClause";
}
def OMPC_Replayable : Clause<[Spelling<"replayable">]> {
  let flangClass = "OmpReplayableClause";
  let isValueOptional = true;
}
def OMPC_ReverseOffload : Clause<[Spelling<"reverse_offload">]> {
  let clangClass = "OMPReverseOffloadClause";
}
def OMPC_SafeLen : Clause<[Spelling<"safelen">]> {
  let clangClass = "OMPSafelenClause";
  let flangClass = "ScalarIntConstantExpr";
}
def OMP_SCHEDULE_Static : EnumVal<"static", 2, 1> {}
def OMP_SCHEDULE_Dynamic : EnumVal<"dynamic", 3, 1> {}
def OMP_SCHEDULE_Guided : EnumVal<"guided", 4, 1> {}
def OMP_SCHEDULE_Auto : EnumVal<"auto", 5, 1> {}
def OMP_SCHEDULE_Runtime : EnumVal<"runtime", 6, 1> {}
def OMP_SCHEDULE_Default : EnumVal<"default", 7, 0> { let isDefault = 1; }
def OMPC_Schedule : Clause<[Spelling<"schedule">]> {
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
def OMPC_SeqCst : Clause<[Spelling<"seq_cst">]> {
  let clangClass = "OMPSeqCstClause";
}
def OMPC_Severity : Clause<[Spelling<"severity">]> {
  let clangClass = "OMPSeverityClause";
  let flangClass = "OmpSeverityClause";
}
def OMPC_Shared : Clause<[Spelling<"shared">]> {
  let clangClass = "OMPSharedClause";
  let flangClass = "OmpObjectList";
}
def OMPC_Simd : Clause<[Spelling<"simd">]> {
  let clangClass = "OMPSIMDClause";
}
def OMPC_SimdLen : Clause<[Spelling<"simdlen">]> {
  let clangClass = "OMPSimdlenClause";
  let flangClass = "ScalarIntConstantExpr";
}
def OMPC_Sizes: Clause<[Spelling<"sizes">]> {
  let clangClass = "OMPSizesClause";
  let flangClass = "ScalarIntExpr";
  let isValueList = true;
}
def OMPC_TaskReduction : Clause<[Spelling<"task_reduction">]> {
  let clangClass = "OMPTaskReductionClause";
  let flangClass = "OmpTaskReductionClause";
}
def OMPC_ThreadLimit : Clause<[Spelling<"thread_limit">]> {
  let clangClass = "OMPThreadLimitClause";
  let flangClass = "ScalarIntExpr";
}
def OMPC_ThreadPrivate : Clause<[Spelling<"threadprivate">]> {
  let isImplicit = true;
}
def OMPC_GroupPrivate : Clause<[Spelling<"groupprivate">]> { 
  let isImplicit = true;
}
def OMPC_Threads : Clause<[Spelling<"threads">]> {
  let clangClass = "OMPThreadsClause";
}
def OMPC_To : Clause<[Spelling<"to">]> {
  let clangClass = "OMPToClause";
  let flangClass = "OmpToClause";
}
def OMPC_Transparent : Clause<[Spelling<"transparent">]> {
  let flangClass = "OmpTransparentClause";
  let isValueOptional = true;
}
def OMPC_UnifiedAddress : Clause<[Spelling<"unified_address">]> {
  let clangClass = "OMPUnifiedAddressClause";
}
def OMPC_UnifiedSharedMemory : Clause<[Spelling<"unified_shared_memory">]> {
  let clangClass = "OMPUnifiedSharedMemoryClause";
}
def OMPC_SelfMaps : Clause<[Spelling<"self_maps">]> {
  let clangClass = "OMPSelfMapsClause";
}
def OMPC_Uniform : Clause<[Spelling<"uniform">]> {
  let flangClass = "Name";
  let isValueList = true;
}
def OMPC_Unknown : Clause<[Spelling<"unknown">]> {
  let isImplicit = true;
  let isDefault = true;
}
def OMPC_Untied : Clause<[Spelling<"untied">]> {
  let clangClass = "OMPUntiedClause";
}
def OMPC_Update : Clause<[Spelling<"update">]> {
  let clangClass = "OMPUpdateClause";
  let flangClass = "OmpUpdateClause";
  let isValueOptional = true;
}
def OMPC_Use : Clause<[Spelling<"use">]> {
  let clangClass = "OMPUseClause";
  let flangClass = "OmpUseClause";
}
def OMPC_UsesAllocators : Clause<[Spelling<"uses_allocators">]> {
  let clangClass = "OMPUsesAllocatorsClause";
}
def OMPC_UseDeviceAddr : Clause<[Spelling<"use_device_addr">]> {
  let clangClass = "OMPUseDeviceAddrClause";
  let flangClass = "OmpObjectList";
}
def OMPC_UseDevicePtr : Clause<[Spelling<"use_device_ptr">]> {
  let clangClass = "OMPUseDevicePtrClause";
  let flangClass = "OmpObjectList";
}
def OMPC_Weak : Clause<[Spelling<"weak">]> {
  let clangClass = "OMPWeakClause";
}
def OMPC_When: Clause<[Spelling<"when">]> {
  let flangClass = "OmpWhenClause";
}
def OMPC_Write : Clause<[Spelling<"write">]> {
  let clangClass = "OMPWriteClause";
}

//===----------------------------------------------------------------------===//
// Definitions of OpenMP leaf directives
// Sorted alphabetically wrt directive spelling, except "end xyz" immediately
// follows "xyz".
//===----------------------------------------------------------------------===//

def OMP_Allocate : Directive<[Spelling<"allocate">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_Align, 51>,
    VersionedClause<OMPC_Allocator>,
  ];
  let association = AS_None;
  let category = CA_Declarative;
}
def OMP_Allocators : Directive<[Spelling<"allocators">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_Assumes : Directive<[Spelling<"assumes">]> {
  let association = AS_None;
  let category = CA_Informational;
  let allowedOnceClauses = [
    VersionedClause<OMPC_Absent, 51>,
    VersionedClause<OMPC_Contains, 51>,
    VersionedClause<OMPC_Holds, 51>,
    VersionedClause<OMPC_NoOpenMP, 51>,
    VersionedClause<OMPC_NoOpenMPRoutines, 51>,
    VersionedClause<OMPC_NoParallelism, 51>,
  ];
}
def OMP_Assume : Directive<[Spelling<"assume">]> {
  let association = AS_Block;
  let category = CA_Informational;
  let allowedOnceClauses = [
    VersionedClause<OMPC_Absent, 51>,
    VersionedClause<OMPC_Contains, 51>,
    VersionedClause<OMPC_Holds, 51>,
    VersionedClause<OMPC_NoOpenMP, 51>,
    VersionedClause<OMPC_NoOpenMPRoutines, 51>,
    VersionedClause<OMPC_NoParallelism, 51>,
    VersionedClause<OMPC_NoOpenMPConstructs, 60>,
  ];
}
def OMP_Atomic : Directive<[Spelling<"atomic">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_AcqRel, 50>,
    VersionedClause<OMPC_Acquire, 50>,
    VersionedClause<OMPC_Capture>,
    VersionedClause<OMPC_Compare, 51>,
    VersionedClause<OMPC_Fail, 51>,
    VersionedClause<OMPC_Hint, 50>,
    VersionedClause<OMPC_Read>,
    VersionedClause<OMPC_Relaxed, 50>,
    VersionedClause<OMPC_Release, 50>,
    VersionedClause<OMPC_SeqCst>,
    VersionedClause<OMPC_Update>,
    VersionedClause<OMPC_Weak, 51>,
    VersionedClause<OMPC_Write>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_Barrier : Directive<[Spelling<"barrier">]> {
  let association = AS_None;
  let category = CA_Executable;
}
def OMP_BeginAssumes : Directive<[Spelling<"begin assumes">]> {
  let association = AS_Delimited;
  let category = CA_Informational;
  let allowedOnceClauses = [
    VersionedClause<OMPC_Absent, 51>,
    VersionedClause<OMPC_Contains, 51>,
    VersionedClause<OMPC_Holds, 51>,
    VersionedClause<OMPC_NoOpenMP, 51>,
    VersionedClause<OMPC_NoOpenMPRoutines, 51>,
    VersionedClause<OMPC_NoParallelism, 51>,
  ];
  let languages = [L_C];
}
def OMP_EndAssumes : Directive<[Spelling<"end assumes">]> {
  let association = AS_Delimited;
  let category = OMP_BeginAssumes.category;
  let languages = OMP_BeginAssumes.languages;
}
def OMP_BeginDeclareTarget : Directive<[Spelling<"begin declare target", 1, 52>,
                                        Spelling<"begin declare_target", 60>]> {
  let allowedClauses = [
    VersionedClause<OMPC_DeviceType>,
    VersionedClause<OMPC_Indirect, 51>,
    VersionedClause<OMPC_Link>,
    VersionedClause<OMPC_To>,
  ];
  let association = AS_Delimited;
  let category = CA_Declarative;
  let languages = [L_C];
}
def OMP_EndDeclareTarget : Directive<[Spelling<"end declare target", 1, 52>,
                                      Spelling<"end declare_target", 60>]> {
  let association = AS_Delimited;
  let category = OMP_BeginDeclareTarget.category;
  let languages = OMP_BeginDeclareTarget.languages;
}
def OMP_BeginDeclareVariant
    : Directive<[Spelling<"begin declare variant", 1, 52>,
                 Spelling<"begin declare_variant", 60>]> {
  let association = AS_Delimited;
  let category = CA_Declarative;
  let languages = [L_C];
}
def OMP_EndDeclareVariant : Directive<[Spelling<"end declare variant", 1, 52>,
                                       Spelling<"end declare_variant", 60>]> {
  let association = AS_Delimited;
  let category = OMP_BeginDeclareVariant.category;
  let languages = OMP_BeginDeclareVariant.languages;
}
def OMP_Cancel : Directive<[Spelling<"cancel">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_CancellationConstructType>,
    VersionedClause<OMPC_If>,
  ];
  let association = AS_None;
  let category = CA_Executable;
}
def OMP_CancellationPoint : Directive<[Spelling<"cancellation point", 1, 52>,
                                       Spelling<"cancellation_point", 60>]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_CancellationConstructType>,
  ];
  let association = AS_None;
  let category = CA_Executable;
}
def OMP_Critical : Directive<[Spelling<"critical">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_Hint>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_DeclareMapper : Directive<[Spelling<"declare mapper", 1, 52>,
                                   Spelling<"declare_mapper", 60>]> {
  let requiredClauses = [
    VersionedClause<OMPC_Map, 45>,
  ];
  let association = AS_None;
  let category = CA_Declarative;
}
def OMP_DeclareReduction : Directive<[Spelling<"declare reduction", 1, 52>,
                                      Spelling<"declare_reduction", 60>]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_Initializer>,
  ];
  let association = AS_None;
  let category = CA_Declarative;
}
def OMP_DeclareSimd : Directive<[Spelling<"declare simd", 1, 52>,
                                 Spelling<"declare_simd", 60>]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Uniform>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_SimdLen>,
  ];
  let allowedExclusiveClauses = [
    VersionedClause<OMPC_Inbranch>,
    VersionedClause<OMPC_Notinbranch>,
  ];
  let association = AS_Declaration;
  let category = CA_Declarative;
}
def OMP_DeclareTarget : Directive<[Spelling<"declare target", 1, 52>,
                                   Spelling<"declare_target", 60>]> {
  let allowedClauses = [
    VersionedClause<OMPC_Enter, 52>,
    VersionedClause<OMPC_Indirect, 51>,
    VersionedClause<OMPC_Link>,
    VersionedClause<OMPC_To>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_DeviceType, 50>,
  ];
  let association = AS_None;
  let category = CA_Declarative;
}
def OMP_DeclareVariant : Directive<[Spelling<"declare variant", 1, 52>,
                                    Spelling<"declare_variant", 60>]> {
  let allowedClauses = [
    VersionedClause<OMPC_AdjustArgs, 51>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Match>,
    VersionedClause<OMPC_AppendArgs, 51>,
  ];
  let association = AS_Declaration;
  let category = CA_Declarative;
  let languages = [L_C];
}
def OMP_Depobj : Directive<[Spelling<"depobj">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Depend, 50>,
    // TODO This should ne `none` instead. Comment carried over from
    // OMPKinds.def.
    VersionedClause<OMPC_Depobj, 50>,
    VersionedClause<OMPC_Destroy, 50>,
    VersionedClause<OMPC_Update, 50>,
  ];
  let association = AS_None;
  let category = CA_Executable;
}
def OMP_dispatch : Directive<[Spelling<"dispatch">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_Nocontext>,
    VersionedClause<OMPC_Novariants>,
    VersionedClause<OMPC_NoWait>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_Distribute : Directive<[Spelling<"distribute">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Private>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_Order, 50>,
  ];
  let association = AS_Loop;
  let category = CA_Executable;
}
def OMP_Do : Directive<[Spelling<"do">]> {
  let allowedClauses = [
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Schedule>,
  ];
  let association = AS_Loop;
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_EndDo : Directive<[Spelling<"end do">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_NoWait>,
  ];
  let leafConstructs = OMP_Do.leafConstructs;
  let association = OMP_Do.association;
  let category = OMP_Do.category;
  let languages = OMP_Do.languages;
}
def OMP_Error : Directive<[Spelling<"error">]> {
  let allowedClauses = [
    VersionedClause<OMPC_At, 51>,
    VersionedClause<OMPC_Message, 51>,
    VersionedClause<OMPC_Severity, 51>,
  ];
  let association = AS_None;
  let category = CA_Utility;
}
def OMP_Flush : Directive<[Spelling<"flush">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_AcqRel, 50>,
    VersionedClause<OMPC_Acquire, 50>,
    // TODO This should ne `none` instead. Comment carried over from
    // OMPKinds.def.
    VersionedClause<OMPC_Flush>,
    VersionedClause<OMPC_Release, 50>,
    VersionedClause<OMPC_SeqCst, 51>,
  ];
  let association = AS_None;
  let category = CA_Executable;
}
def OMP_For : Directive<[Spelling<"for">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Schedule>,
  ];
  let association = AS_Loop;
  let category = CA_Executable;
  let languages = [L_C];
}
def OMP_Groupprivate : Directive<[Spelling<"groupprivate">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_DeviceType>,
  ];
  let association = AS_None;
  let category = CA_Declarative;
  let languages = [L_C, L_Fortran];
}
def OMP_Fuse : Directive<[Spelling<"fuse">]> {
  let allowedOnceClauses = [VersionedClause<OMPC_LoopRange, 60>];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_Interchange : Directive<[Spelling<"interchange">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_Permutation>,
  ];
  let association = AS_Loop;
  let category = CA_Executable;
}
def OMP_interop : Directive<[Spelling<"interop">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Destroy>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_Init>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Use>,
  ];
  let association = AS_None;
  let category = CA_Executable;
}
def OMP_loop : Directive<[Spelling<"loop">]> {
  let allowedClauses = [
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Bind, 50>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Order, 50>,
  ];
  let association = AS_Loop;
  let category = CA_Executable;
}
def OMP_masked : Directive<[Spelling<"masked">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_Filter>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_Master : Directive<[Spelling<"master">]> {
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_Metadirective : Directive<[Spelling<"metadirective">]> {
  let allowedClauses = [
    VersionedClause<OMPC_When>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Otherwise, 52>,
    VersionedClause<OMPC_Default, 50, 51>,
  ];
  let association = AS_None;
  let category = CA_Meta;
}
def OMP_Nothing : Directive<[Spelling<"nothing">]> {
  let association = AS_None;
  let category = CA_Utility;
}
def OMP_Ordered : Directive<[Spelling<"ordered">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Doacross, 52>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Simd>,
    VersionedClause<OMPC_Threads>,
  ];
  let association = AS_None;
  // There is also a block-associated "ordered" directive.
  let category = CA_Executable;
}
def OMP_Parallel : Directive<[Spelling<"parallel">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
  ];
  let allowedOnceClauses = [VersionedClause<OMPC_Default>,
                            VersionedClause<OMPC_If>,
                            VersionedClause<OMPC_Message, 60>,
                            VersionedClause<OMPC_NumThreads>,
                            VersionedClause<OMPC_ProcBind>,
                            VersionedClause<OMPC_Severity, 60>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_Requires : Directive<[Spelling<"requires">]> {
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
    VersionedClause<OMPC_AtomicDefaultMemOrder>,
    VersionedClause<OMPC_DynamicAllocators>, VersionedClause<OMPC_SelfMaps>,
    VersionedClause<OMPC_ReverseOffload, 99>,
  ];
  let association = AS_None;
  let category = CA_Informational;
}
def OMP_Reverse : Directive<[Spelling<"reverse">]> {
  let association = AS_Loop;
  let category = CA_Executable;
}
def OMP_Scan : Directive<[Spelling<"scan">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Exclusive>,
    VersionedClause<OMPC_Inclusive>,
  ];
  let association = AS_Separating;
  let category = CA_Subsidiary;
}
def OMP_Scope : Directive<[Spelling<"scope">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Private, 51>,
    VersionedClause<OMPC_Reduction, 51>,
    VersionedClause<OMPC_FirstPrivate, 52>,
    VersionedClause<OMPC_Allocate, 52>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_NoWait, 51>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_EndScope : Directive<[Spelling<"end scope">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_NoWait>,
  ];
  let leafConstructs = OMP_Scope.leafConstructs;
  let association = OMP_Scope.association;
  let category = OMP_Scope.category;
  let languages = [L_Fortran];
}
def OMP_Section : Directive<[Spelling<"section">]> {
  let association = AS_Separating;
  let category = CA_Subsidiary;
}
def OMP_Sections : Directive<[Spelling<"sections">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_EndSections : Directive<[Spelling<"end sections">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_NoWait>,
  ];
  let leafConstructs = OMP_Sections.leafConstructs;
  let association = OMP_Sections.association;
  let category = OMP_Sections.category;
  let languages = [L_Fortran];
}
def OMP_Simd : Directive<[Spelling<"simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_If, 50>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
  ];
  let association = AS_Loop;
  let category = CA_Executable;
}
def OMP_Single : Directive<[Spelling<"single">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_CopyPrivate>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Private>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_NoWait>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_EndSingle : Directive<[Spelling<"end single">]> {
  let allowedClauses = [
    VersionedClause<OMPC_CopyPrivate>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_NoWait>,
  ];
  let leafConstructs = OMP_Single.leafConstructs;
  let association = OMP_Single.association;
  let category = OMP_Single.category;
  let languages = [L_Fortran];
}
def OMP_Target : Directive<[Spelling<"target">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_InReduction, 50>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_UsesAllocators, 50>,
    VersionedClause<OMPC_Default, 60>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_OMPX_Bare>,
    VersionedClause<OMPC_OMPX_DynCGroupMem>,
    VersionedClause<OMPC_Replayable, 60>,
    VersionedClause<OMPC_ThreadLimit, 51>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_TargetData : Directive<[Spelling<"target data", 1, 52>,
                                Spelling<"target_data", 60>]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Default, 60>,
    VersionedClause<OMPC_Transparent, 60>,
  ];
  let requiredClauses = [
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_UseDeviceAddr, 50>,
    VersionedClause<OMPC_UseDevicePtr>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_TargetEnterData : Directive<[Spelling<"target enter data", 1, 52>,
                                     Spelling<"target_enter_data", 60>]> {
  let allowedClauses = [
    VersionedClause<OMPC_Depend>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Replayable, 60>,
  ];
  let requiredClauses = [
    VersionedClause<OMPC_Map>,
  ];
  let association = AS_None;
  let category = CA_Executable;
}
def OMP_TargetExitData : Directive<[Spelling<"target exit data", 1, 52>,
                                    Spelling<"target_exit_data", 60>]> {
  let allowedClauses = [
    VersionedClause<OMPC_Depend>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Replayable, 60>,
  ];
  let requiredClauses = [
    VersionedClause<OMPC_Map>,
  ];
  let association = AS_None;
  let category = CA_Executable;
}
def OMP_TargetUpdate : Directive<[Spelling<"target update", 1, 52>,
                                  Spelling<"target_update", 60>]> {
  let allowedClauses = [
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_From>,
    VersionedClause<OMPC_To>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Replayable, 60>,
  ];
  let association = AS_None;
  let category = CA_Executable;
}
def OMP_Task : Directive<[Spelling<"task">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Affinity, 50>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_InReduction>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Untied>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Detach, 50>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_Replayable, 60>,
    VersionedClause<OMPC_Transparent, 60>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_Taskgraph : Directive<[Spelling<"taskgraph">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_GraphId, 60>,
    VersionedClause<OMPC_GraphReset, 60>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_NoGroup>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_TaskGroup : Directive<[Spelling<"taskgroup">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate, 50>,
    VersionedClause<OMPC_TaskReduction, 50>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_TaskLoop : Directive<[Spelling<"taskloop">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_InReduction>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Untied>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_Replayable, 60>,
    VersionedClause<OMPC_Transparent, 60>,
  ];
  let allowedExclusiveClauses = [
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_NumTasks>,
  ];
  let association = AS_Loop;
  let category = CA_Executable;
}
def OMP_TaskWait : Directive<[Spelling<"taskwait">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Depend, 50>,
    VersionedClause<OMPC_NoWait, 51>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Replayable, 60>,
  ];
  let association = AS_None;
  let category = CA_Executable;
}
def OMP_TaskYield : Directive<[Spelling<"taskyield">]> {
  let association = AS_None;
  let category = CA_Executable;
}
def OMP_Teams : Directive<[Spelling<"teams">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_If, 52>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
}
def OMP_ThreadPrivate : Directive<[Spelling<"threadprivate">]> {
  let association = AS_None;
  let category = CA_Declarative;
}
def OMP_Tile : Directive<[Spelling<"tile">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_Sizes, 51>,
  ];
  let requiredClauses = [
    VersionedClause<OMPC_Sizes, 51>,
  ];
  let association = AS_Loop;
  let category = CA_Executable;
}
def OMP_Stripe : Directive<[Spelling<"stripe">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_Sizes, 60>,
  ];
  let association = AS_Loop;
  let category = CA_Executable;
}
def OMP_Unknown : Directive<[Spelling<"unknown">]> {
  let isDefault = true;
  let association = AS_None;
  let category = CA_Utility;
}
def OMP_Unroll : Directive<[Spelling<"unroll">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_Full, 51>,
    VersionedClause<OMPC_Partial, 51>,
  ];
  let association = AS_Loop;
  let category = CA_Executable;
}
def OMP_Workshare : Directive<[Spelling<"workshare">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_NoWait>,
  ];
  let association = AS_Block;
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_EndWorkshare : Directive<[Spelling<"end workshare">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_NoWait>,
  ];
  let leafConstructs = OMP_Workshare.leafConstructs;
  let association = OMP_Workshare.association;
  let category = OMP_Workshare.category;
  let languages = [L_Fortran];
}
def OMP_Workdistribute : Directive<[Spelling<"workdistribute">]> {
  let association = AS_Block;
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_EndWorkdistribute : Directive<[Spelling<"end workdistribute">]> {
  let leafConstructs = OMP_Workdistribute.leafConstructs;
  let association = OMP_Workdistribute.association;
  let category = OMP_Workdistribute.category;
  let languages = [L_Fortran];
}

//===----------------------------------------------------------------------===//
// Definitions of OpenMP compound directives
// Sorted alphabetically wrt directive spelling, except "end xyz" immediately
// follows "xyz".
//===----------------------------------------------------------------------===//

def OMP_DistributeParallelDo : Directive<[Spelling<"distribute parallel do">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
  ];
  let leafConstructs = [OMP_Distribute, OMP_Parallel, OMP_Do];
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_DistributeParallelDoSimd
    : Directive<[Spelling<"distribute parallel do simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NonTemporal>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_SimdLen>,
  ];
  let leafConstructs = [OMP_Distribute, OMP_Parallel, OMP_Do, OMP_Simd];
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_DistributeParallelFor
    : Directive<[Spelling<"distribute parallel for">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
  ];
  let leafConstructs = [OMP_Distribute, OMP_Parallel, OMP_For];
  let category = CA_Executable;
  let languages = [L_C];
}
def OMP_DistributeParallelForSimd
    : Directive<[Spelling<"distribute parallel for simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_SimdLen>,
  ];
  let leafConstructs = [OMP_Distribute, OMP_Parallel, OMP_For, OMP_Simd];
  let category = CA_Executable;
  let languages = [L_C];
}
def OMP_DistributeSimd : Directive<[Spelling<"distribute simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_If, 50>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_SimdLen>,
  ];
  let leafConstructs = [OMP_Distribute, OMP_Simd];
  let category = CA_Executable;
}
def OMP_DoSimd : Directive<[Spelling<"do simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_If, 50>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_SimdLen>,
  ];
  let leafConstructs = [OMP_Do, OMP_Simd];
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_EndDoSimd : Directive<[Spelling<"end do simd">]> {
  let allowedOnceClauses = [
    VersionedClause<OMPC_NoWait>,
  ];
  let leafConstructs = OMP_DoSimd.leafConstructs;
  let association = OMP_DoSimd.association;
  let category = OMP_DoSimd.category;
  let languages = [L_Fortran];
}
def OMP_ForSimd : Directive<[Spelling<"for simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If, 50>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_SimdLen>,
  ];
  let leafConstructs = [OMP_For, OMP_Simd];
  let category = CA_Executable;
}
def OMP_target_loop : Directive<[Spelling<"target loop">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_UsesAllocators, 50>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_InReduction, 50>,

  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Bind, 50>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Order>,
    VersionedClause<OMPC_ThreadLimit>,
    VersionedClause<OMPC_OMPX_DynCGroupMem>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_NoWait>,
  ];
  let leafConstructs = [OMP_Target, OMP_loop];
  let category = CA_Executable;
}
def OMP_MaskedTaskloop : Directive<[Spelling<"masked taskloop">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Filter>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_InReduction>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Untied>,
  ];
  let leafConstructs = [OMP_masked, OMP_TaskLoop];
  let category = CA_Executable;
}
def OMP_MaskedTaskloopSimd : Directive<[Spelling<"masked taskloop simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Filter>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_InReduction>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_Untied>,
  ];
  let leafConstructs = [OMP_masked, OMP_TaskLoop, OMP_Simd];
  let category = CA_Executable;
}
def OMP_MasterTaskloop : Directive<[Spelling<"master taskloop">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_InReduction>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Untied>,
  ];
  let leafConstructs = [OMP_Master, OMP_TaskLoop];
  let category = CA_Executable;
}
def OMP_MasterTaskloopSimd : Directive<[Spelling<"master taskloop simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_InReduction>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_Untied>,
  ];
  let leafConstructs = [OMP_Master, OMP_TaskLoop, OMP_Simd];
  let category = CA_Executable;
}
def OMP_ParallelDo : Directive<[Spelling<"parallel do">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
  ];
  let leafConstructs = [OMP_Parallel, OMP_Do];
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_ParallelDoSimd : Directive<[Spelling<"parallel do simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_NonTemporal>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_Severity, 60>,
  ];
  let leafConstructs = [OMP_Parallel, OMP_Do, OMP_Simd];
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_ParallelFor : Directive<[Spelling<"parallel for">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
  ];
  let leafConstructs = [OMP_Parallel, OMP_For];
  let category = CA_Executable;
  let languages = [L_C];
}
def OMP_ParallelForSimd : Directive<[Spelling<"parallel for simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_SimdLen>,
  ];
  let leafConstructs = [OMP_Parallel, OMP_For, OMP_Simd];
  let category = CA_Executable;
  let languages = [L_C];
}
def OMP_parallel_loop : Directive<[Spelling<"parallel loop">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Bind, 50>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Order>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Severity, 60>,
  ];
  let leafConstructs = [OMP_Parallel, OMP_loop];
  let category = CA_Executable;
}
def OMP_ParallelMasked : Directive<[Spelling<"parallel masked">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Filter>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
  ];
  let leafConstructs = [OMP_Parallel, OMP_masked];
  let category = CA_Executable;
}
def OMP_ParallelMaskedTaskloop
    : Directive<[Spelling<"parallel masked taskloop">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Filter>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Untied>,
  ];
  let leafConstructs = [OMP_Parallel, OMP_masked, OMP_TaskLoop];
  let category = CA_Executable;
}
def OMP_ParallelMaskedTaskloopSimd
    : Directive<[Spelling<"parallel masked taskloop simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Filter>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_Untied>,
  ];
  let leafConstructs = [OMP_Parallel, OMP_masked, OMP_TaskLoop, OMP_Simd];
  let category = CA_Executable;
}
def OMP_ParallelMaster : Directive<[Spelling<"parallel master">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
  ];
  let leafConstructs = [OMP_Parallel, OMP_Master];
  let category = CA_Executable;
}
def OMP_ParallelMasterTaskloop
    : Directive<[Spelling<"parallel master taskloop">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Untied>,
  ];
  let leafConstructs = [OMP_Parallel, OMP_Master, OMP_TaskLoop];
  let category = CA_Executable;
}
def OMP_ParallelMasterTaskloopSimd
    : Directive<[Spelling<"parallel master taskloop simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_NumTasks>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_Untied>,
  ];
  let leafConstructs = [OMP_Parallel, OMP_Master, OMP_TaskLoop, OMP_Simd];
  let category = CA_Executable;
}
def OMP_ParallelSections : Directive<[Spelling<"parallel sections">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Severity, 60>,
  ];
  let leafConstructs = [OMP_Parallel, OMP_Sections];
  let category = CA_Executable;
}
def OMP_ParallelWorkshare : Directive<[Spelling<"parallel workshare">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Severity, 60>,
  ];
  let leafConstructs = [OMP_Parallel, OMP_Workshare];
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_TargetParallel : Directive<[Spelling<"target parallel">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators, 50>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_DynCGroupMem>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_ThreadLimit, 51>,
  ];
  let leafConstructs = [OMP_Target, OMP_Parallel];
  let category = CA_Executable;
}
def OMP_TargetParallelDo : Directive<[Spelling<"target parallel do">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocator>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
  ];
  let leafConstructs = [OMP_Target, OMP_Parallel, OMP_Do];
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_TargetParallelDoSimd
    : Directive<[Spelling<"target parallel do simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NonTemporal>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_UsesAllocators>,
  ];
  let leafConstructs = [OMP_Target, OMP_Parallel, OMP_Do, OMP_Simd];
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_TargetParallelFor : Directive<[Spelling<"target parallel for">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators, 50>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_OMPX_DynCGroupMem>,
    VersionedClause<OMPC_ThreadLimit, 51>,
  ];
  let leafConstructs = [OMP_Target, OMP_Parallel, OMP_For];
  let category = CA_Executable;
  let languages = [L_C];
}
def OMP_TargetParallelForSimd
    : Directive<[Spelling<"target parallel for simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Ordered>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_UsesAllocators, 50>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_OMPX_DynCGroupMem>,
    VersionedClause<OMPC_ThreadLimit, 51>,
  ];
  let leafConstructs = [OMP_Target, OMP_Parallel, OMP_For, OMP_Simd];
  let category = CA_Executable;
  let languages = [L_C];
}
def OMP_target_parallel_loop : Directive<[Spelling<"target parallel loop">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_OMPX_Attribute>,
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
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_DynCGroupMem>,
    VersionedClause<OMPC_Order>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_ThreadLimit, 51>,
  ];
  let leafConstructs = [OMP_Target, OMP_Parallel, OMP_loop];
  let category = CA_Executable;
}
def OMP_TargetSimd : Directive<[Spelling<"target simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators, 50>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_DynCGroupMem>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_ThreadLimit, 51>,
  ];
  let leafConstructs = [OMP_Target, OMP_Simd];
  let category = CA_Executable;
}
def OMP_TargetTeams : Directive<[Spelling<"target teams">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators, 50>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_OMPX_DynCGroupMem>,
    VersionedClause<OMPC_OMPX_Bare>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let leafConstructs = [OMP_Target, OMP_Teams];
  let category = CA_Executable;
}
def OMP_TargetTeamsDistribute
    : Directive<[Spelling<"target teams distribute">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators, 50>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_OMPX_DynCGroupMem>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let leafConstructs = [OMP_Target, OMP_Teams, OMP_Distribute];
  let category = CA_Executable;
}
def OMP_TargetTeamsDistributeParallelDo
    : Directive<[Spelling<"target teams distribute parallel do">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let leafConstructs =
      [OMP_Target, OMP_Teams, OMP_Distribute, OMP_Parallel, OMP_Do];
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_TargetTeamsDistributeParallelDoSimd
    : Directive<[Spelling<"target teams distribute parallel do simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_NonTemporal>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let leafConstructs =
      [OMP_Target, OMP_Teams, OMP_Distribute, OMP_Parallel, OMP_Do, OMP_Simd];
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_TargetTeamsDistributeParallelFor
    : Directive<[Spelling<"target teams distribute parallel for">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_ThreadLimit>,
    VersionedClause<OMPC_UsesAllocators, 50>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_OMPX_DynCGroupMem>,
  ];
  let leafConstructs =
      [OMP_Target, OMP_Teams, OMP_Distribute, OMP_Parallel, OMP_For];
  let category = CA_Executable;
  let languages = [L_C];
}
def OMP_TargetTeamsDistributeParallelForSimd
    : Directive<[Spelling<"target teams distribute parallel for simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_ThreadLimit>,
    VersionedClause<OMPC_UsesAllocators, 50>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_OMPX_DynCGroupMem>,
  ];
  let leafConstructs =
      [OMP_Target, OMP_Teams, OMP_Distribute, OMP_Parallel, OMP_For, OMP_Simd];
  let category = CA_Executable;
  let languages = [L_C];
}
def OMP_TargetTeamsDistributeSimd
    : Directive<[Spelling<"target teams distribute simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators, 50>,
    VersionedClause<OMPC_Default, 60>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_OMPX_DynCGroupMem>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let leafConstructs = [OMP_Target, OMP_Teams, OMP_Distribute, OMP_Simd];
  let category = CA_Executable;
}
def OMP_TargetTeamsWorkdistribute : Directive<[Spelling<"target teams workdistribute">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators, 50>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_OMPX_DynCGroupMem>,
    VersionedClause<OMPC_OMPX_Bare>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let leafConstructs = [OMP_Target, OMP_Teams, OMP_Workdistribute];
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_target_teams_loop : Directive<[Spelling<"target teams loop">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_DefaultMap>,
    VersionedClause<OMPC_Depend>,
    VersionedClause<OMPC_Device>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_HasDeviceAddr, 51>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_IsDevicePtr>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Map>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_UsesAllocators, 50>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Bind, 50>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_NoWait>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_OMPX_DynCGroupMem>,
    VersionedClause<OMPC_Order>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let leafConstructs = [OMP_Target, OMP_Teams, OMP_loop];
  let category = CA_Executable;
}
def OMP_TaskLoopSimd : Directive<[Spelling<"taskloop simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_InReduction>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Mergeable>,
    VersionedClause<OMPC_NoGroup>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_Untied>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Final>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Priority>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
  ];
  let allowedExclusiveClauses = [
    VersionedClause<OMPC_GrainSize>,
    VersionedClause<OMPC_NumTasks>,
  ];
  let leafConstructs = [OMP_TaskLoop, OMP_Simd];
  let category = CA_Executable;
}
def OMP_TeamsDistribute : Directive<[Spelling<"teams distribute">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_Order, 50>,
  ];
  let leafConstructs = [OMP_Teams, OMP_Distribute];
  let category = CA_Executable;
}
def OMP_TeamsDistributeParallelDo
    : Directive<[Spelling<"teams distribute parallel do">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let leafConstructs = [OMP_Teams, OMP_Distribute, OMP_Parallel, OMP_Do];
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_TeamsDistributeParallelDoSimd
    : Directive<[Spelling<"teams distribute parallel do simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_NonTemporal>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let leafConstructs =
      [OMP_Teams, OMP_Distribute, OMP_Parallel, OMP_Do, OMP_Simd];
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_TeamsDistributeParallelFor
    : Directive<[Spelling<"teams distribute parallel for">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Copyin>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let leafConstructs = [OMP_Teams, OMP_Distribute, OMP_Parallel, OMP_For];
  let category = CA_Executable;
  let languages = [L_C];
}
def OMP_TeamsDistributeParallelForSimd
    : Directive<[Spelling<"teams distribute parallel for simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_Message, 60>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_NumThreads>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_ProcBind>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_Schedule>,
    VersionedClause<OMPC_Severity, 60>,
    VersionedClause<OMPC_Shared>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let leafConstructs =
      [OMP_Teams, OMP_Distribute, OMP_Parallel, OMP_For, OMP_Simd];
  let category = CA_Executable;
  let languages = [L_C];
}
def OMP_TeamsDistributeSimd : Directive<[Spelling<"teams distribute simd">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Aligned>,
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_If, 50>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_Linear>,
    VersionedClause<OMPC_NonTemporal, 50>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Collapse>,
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_DistSchedule>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_Order, 50>,
    VersionedClause<OMPC_SafeLen>,
    VersionedClause<OMPC_SimdLen>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let leafConstructs = [OMP_Teams, OMP_Distribute, OMP_Simd];
  let category = CA_Executable;
}
def OMP_TeamsWorkdistribute : Directive<[Spelling<"teams workdistribute">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_OMPX_Attribute>,
    VersionedClause<OMPC_Private>,
    VersionedClause<OMPC_Reduction>,
    VersionedClause<OMPC_Shared>,
  ];
  let allowedOnceClauses = [
    VersionedClause<OMPC_Default>,
    VersionedClause<OMPC_If, 52>,
    VersionedClause<OMPC_NumTeams>,
    VersionedClause<OMPC_ThreadLimit>,
  ];
  let leafConstructs = [OMP_Teams, OMP_Workdistribute];
  let category = CA_Executable;
  let languages = [L_Fortran];
}
def OMP_teams_loop : Directive<[Spelling<"teams loop">]> {
  let allowedClauses = [
    VersionedClause<OMPC_Allocate>,
    VersionedClause<OMPC_DynGroupprivate, 61>,
    VersionedClause<OMPC_FirstPrivate>,
    VersionedClause<OMPC_LastPrivate>,
    VersionedClause<OMPC_OMPX_Attribute>,
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
  let leafConstructs = [OMP_Teams, OMP_loop];
  let category = CA_Executable;
}
