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
  let makeEnumAvailableInNamespace = 1;
  let enableBitmaskEnumInNamespace = 1;
}

//===----------------------------------------------------------------------===//
// Definition of OpenMP clauses
//===----------------------------------------------------------------------===//

def OMPC_Allocator : Clause<"allocator"> {
  let clangClass = "OMPAllocatorClause";
}
def OMPC_If : Clause<"if"> { let clangClass = "OMPIfClause"; }
def OMPC_Final : Clause<"final"> { let clangClass = "OMPFinalClause"; }
def OMPC_NumThreads : Clause<"num_threads"> {
  let clangClass = "OMPNumThreadsClause";
}
def OMPC_SafeLen : Clause<"safelen"> { let clangClass = "OMPSafelenClause"; }
def OMPC_SimdLen : Clause<"simdlen"> { let clangClass = "OMPSimdlenClause"; }
def OMPC_Collapse : Clause<"collapse"> { let clangClass = "OMPCollapseClause"; }
def OMPC_Default : Clause<"default"> { let clangClass = "OMPDefaultClause"; }
def OMPC_Private : Clause<"private"> { let clangClass = "OMPPrivateClause"; }
def OMPC_FirstPrivate : Clause<"firstprivate"> {
  let clangClass = "OMPFirstprivateClause";
}
def OMPC_LastPrivate : Clause<"lastprivate"> {
  let clangClass = "OMPLastprivateClause";
}
def OMPC_Shared : Clause<"shared"> { let clangClass = "OMPSharedClause"; }
def OMPC_Reduction : Clause<"reduction"> {
  let clangClass = "OMPReductionClause";
}
def OMPC_Linear : Clause<"linear"> { let clangClass = "OMPLinearClause"; }
def OMPC_Aligned : Clause<"aligned"> { let clangClass = "OMPAlignedClause"; }
def OMPC_Copyin : Clause<"copyin"> { let clangClass = "OMPCopyinClause"; }
def OMPC_CopyPrivate : Clause<"copyprivate"> {
  let clangClass = "OMPCopyprivateClause";
}
def OMPC_ProcBind : Clause<"proc_bind"> {
  let clangClass = "OMPProcBindClause";
}
def OMPC_Schedule : Clause<"schedule"> { let clangClass = "OMPScheduleClause"; }
def OMPC_Ordered : Clause<"ordered"> { let clangClass = "OMPOrderedClause"; }
def OMPC_NoWait : Clause<"nowait"> { let clangClass = "OMPNowaitClause"; }
def OMPC_Untied : Clause<"untied"> { let clangClass = "OMPUntiedClause"; }
def OMPC_Mergeable : Clause<"mergeable"> {
  let clangClass = "OMPMergeableClause";
}
def OMPC_Read : Clause<"read"> { let clangClass = "OMPReadClause"; }
def OMPC_Write : Clause<"write"> { let clangClass = "OMPWriteClause"; }
def OMPC_Update : Clause<"update"> { let clangClass = "OMPUpdateClause"; }
def OMPC_Capture : Clause<"capture"> { let clangClass = "OMPCaptureClause"; }
def OMPC_SeqCst : Clause<"seq_cst"> { let clangClass = "OMPSeqCstClause"; }
def OMPC_AcqRel : Clause<"acq_rel"> { let clangClass = "OMPAcqRelClause"; }
def OMPC_Acquire : Clause<"acquire"> { let clangClass = "OMPAcquireClause"; }
def OMPC_Release : Clause<"release"> { let clangClass = "OMPReleaseClause"; }
def OMPC_Relaxed : Clause<"relaxed"> { let clangClass = "OMPRelaxedClause"; }
def OMPC_Depend : Clause<"depend"> { let clangClass = "OMPDependClause"; }
def OMPC_Device : Clause<"device"> { let clangClass = "OMPDeviceClause"; }
def OMPC_Threads : Clause<"threads"> { let clangClass = "OMPThreadsClause"; }
def OMPC_Simd : Clause<"simd"> { let clangClass = "OMPSIMDClause"; }
def OMPC_Map : Clause<"map"> { let clangClass = "OMPMapClause"; }
def OMPC_NumTeams : Clause<"num_teams"> {
  let clangClass = "OMPNumTeamsClause";
}
def OMPC_ThreadLimit : Clause<"thread_limit"> {
  let clangClass = "OMPThreadLimitClause";
}
def OMPC_Priority : Clause<"priority"> {
  let clangClass = "OMPPriorityClause";
}
def OMPC_GrainSize : Clause<"grainsize"> {
  let clangClass = "OMPGrainsizeClause";
}
def OMPC_NoGroup : Clause<"nogroup"> {
  let clangClass = "OMPNogroupClause";
}
def OMPC_NumTasks : Clause<"num_tasks"> {
  let clangClass = "OMPNumTasksClause";
}
def OMPC_Hint : Clause<"hint"> {
  let clangClass = "OMPHintClause";
}
def OMPC_DistSchedule : Clause<"dist_schedule"> {
  let clangClass = "OMPDistScheduleClause";
}
def OMPC_DefaultMap : Clause<"defaultmap"> {
  let clangClass = "OMPDefaultmapClause";
}
def OMPC_To : Clause<"to"> {
  let clangClass = "OMPToClause";
}
def OMPC_From : Clause<"from"> { let clangClass = "OMPFromClause"; }
def OMPC_UseDevicePtr : Clause<"use_device_ptr"> {
  let clangClass = "OMPUseDevicePtrClause";
}
def OMPC_IsDevicePtr : Clause<"is_device_ptr"> {
  let clangClass = "OMPIsDevicePtrClause";
}
def OMPC_TaskReduction : Clause<"task_reduction"> {
  let clangClass = "OMPTaskReductionClause";
}
def OMPC_InReduction : Clause<"in_reduction"> {
  let clangClass = "OMPInReductionClause";
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
}
def OMPC_Allocate : Clause<"allocate"> {
  let clangClass = "OMPAllocateClause";
}
def OMPC_NonTemporal : Clause<"nontemporal"> {
  let clangClass = "OMPNontemporalClause";
}
def OMPC_Order : Clause<"order"> {
  let clangClass = "OMPOrderClause";
}
def OMPC_Destroy : Clause<"destroy"> {
  let clangClass = "OMPDestroyClause";
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
def OMPC_Uniform : Clause<"uniform"> {}
def OMPC_DeviceType : Clause<"device_type"> {}
def OMPC_Match : Clause<"match"> {}
def OMPC_Depobj : Clause<"depobj"> {
  let clangClass = "OMPDepobjClause";
  let isImplicit = 1;
}
def OMPC_Flush : Clause<"flush"> {
  let clangClass = "OMPFlushClause";
  let isImplicit = 1;
}
def OMPC_ThreadPrivate : Clause<"threadprivate"> {
  // threadprivate or thread local
  let isImplicit = 1;
}
def OMPC_Unknown : Clause<"unknown"> {
  let isImplicit = 1;
}

//===----------------------------------------------------------------------===//
// Definition of OpenMP directives
//===----------------------------------------------------------------------===//

def OMP_ThreadPrivate : Directive<"threadprivate"> {}
def OMP_Parallel : Directive<"parallel"> {
  let allowedClauses = [OMPC_If, OMPC_NumThreads, OMPC_Default, OMPC_ProcBind,
      OMPC_Private, OMPC_FirstPrivate, OMPC_Shared, OMPC_Reduction, OMPC_Copyin,
      OMPC_Allocate];
}
def OMP_Task : Directive<"task"> {
  let allowedClauses = [OMPC_If, OMPC_Final, OMPC_Default, OMPC_Private,
      OMPC_FirstPrivate, OMPC_Shared, OMPC_Untied, OMPC_Mergeable, OMPC_Depend,
      OMPC_Priority, OMPC_InReduction, OMPC_Allocate, OMPC_Detach,
      OMPC_Affinity];
}
def OMP_Simd : Directive<"simd"> {
  let allowedClauses = [OMPC_Private, OMPC_LastPrivate, OMPC_Linear,
      OMPC_Aligned, OMPC_SafeLen, OMPC_SimdLen, OMPC_Collapse, OMPC_Reduction,
      OMPC_Allocate, OMPC_If, OMPC_NonTemporal, OMPC_Order];
}
def OMP_For : Directive<"for"> {
  let allowedClauses = [OMPC_Private, OMPC_LastPrivate, OMPC_FirstPrivate,
      OMPC_Reduction, OMPC_Collapse, OMPC_Schedule, OMPC_Ordered, OMPC_NoWait,
      OMPC_Linear, OMPC_Allocate, OMPC_Order];
}
def OMP_Sections : Directive<"sections"> {
  let allowedClauses = [OMPC_Private, OMPC_LastPrivate, OMPC_FirstPrivate,
      OMPC_Reduction, OMPC_NoWait, OMPC_Allocate];
}
def OMP_Section : Directive<"section"> {}
def OMP_Single : Directive<"single"> {
  let allowedClauses = [OMPC_Private, OMPC_FirstPrivate, OMPC_CopyPrivate,
      OMPC_NoWait, OMPC_Allocate];
}
def OMP_Master : Directive<"master"> {}
def OMP_Critical : Directive<"critical"> {
  let allowedClauses = [OMPC_Hint];
}
def OMP_TaskYield : Directive<"taskyield"> {}
def OMP_Barrier : Directive<"barrier"> {}
def OMP_TaskWait : Directive<"taskwait"> {}
def OMP_TaskGroup : Directive<"taskgroup"> {
  let allowedClauses = [OMPC_TaskReduction, OMPC_Allocate];
}
def OMP_Flush : Directive<"flush"> {
  let allowedClauses = [OMPC_AcqRel, OMPC_Acquire, OMPC_Release, OMPC_Flush];
}
def OMP_Ordered : Directive<"ordered"> {
  let allowedClauses = [OMPC_Threads, OMPC_Simd, OMPC_Depend];
}
def OMP_Atomic : Directive<"atomic"> {
  let allowedClauses = [OMPC_Read, OMPC_Write, OMPC_Update, OMPC_Capture,
      OMPC_SeqCst, OMPC_AcqRel, OMPC_Acquire, OMPC_Release, OMPC_Relaxed,
      OMPC_Hint];
}
def OMP_Target : Directive<"target"> {
  let allowedClauses = [OMPC_If, OMPC_Device, OMPC_Map, OMPC_Private,
      OMPC_NoWait, OMPC_Depend, OMPC_DefaultMap, OMPC_FirstPrivate,
      OMPC_IsDevicePtr, OMPC_Reduction, OMPC_Allocate, OMPC_UsesAllocators];
}
def OMP_Teams : Directive<"teams"> {
  let allowedClauses = [OMPC_Default, OMPC_Private, OMPC_FirstPrivate,
      OMPC_Shared, OMPC_Reduction, OMPC_NumTeams, OMPC_ThreadLimit,
      OMPC_Allocate];
}
def OMP_Cancel : Directive<"cancel"> {
  let allowedClauses = [OMPC_If];
}
def OMP_Requires : Directive<"requires"> {
  let allowedClauses = [OMPC_UnifiedAddress, OMPC_UnifiedSharedMemory,
      OMPC_ReverseOffload, OMPC_DynamicAllocators, OMPC_AtomicDefaultMemOrder];
}
def OMP_TargetData : Directive<"target data"> {
  let allowedClauses = [OMPC_If, OMPC_Device, OMPC_Map, OMPC_UseDevicePtr,
      OMPC_UseDevicePtr];
}
def OMP_TargetEnterData : Directive<"target enter data"> {
  let allowedClauses = [OMPC_If, OMPC_Device, OMPC_Map, OMPC_NoWait,
      OMPC_Depend];
}
def OMP_TargetExitData : Directive<"target exit data"> {
  let allowedClauses = [OMPC_If, OMPC_Device, OMPC_Map, OMPC_NoWait,
      OMPC_Depend];
}
def OMP_TargetParallel : Directive<"target parallel"> {
  let allowedClauses = [OMPC_If, OMPC_Device, OMPC_Map, OMPC_NoWait,
      OMPC_Depend, OMPC_Private, OMPC_FirstPrivate, OMPC_DefaultMap,
      OMPC_NumThreads, OMPC_Default, OMPC_ProcBind, OMPC_Shared, OMPC_Reduction,
      OMPC_IsDevicePtr, OMPC_Allocator, OMPC_UsesAllocators];
}
def OMP_TargetParallelFor : Directive<"target parallel for"> {
  let allowedClauses = [OMPC_If, OMPC_Device, OMPC_Map, OMPC_Private,
      OMPC_FirstPrivate, OMPC_LastPrivate, OMPC_NoWait, OMPC_Depend,
      OMPC_DefaultMap, OMPC_NumThreads, OMPC_DefaultMap, OMPC_ProcBind,
      OMPC_Shared, OMPC_Reduction, OMPC_Collapse, OMPC_Schedule, OMPC_Ordered,
      OMPC_Linear, OMPC_IsDevicePtr, OMPC_Allocator, OMPC_Order,
      OMPC_UsesAllocators];
}
def OMP_TargetUpdate : Directive<"target update"> {
  let allowedClauses = [OMPC_If, OMPC_Device, OMPC_To, OMPC_From, OMPC_NoWait,
      OMPC_Depend];
}
def OMP_ParallelFor : Directive<"parallel for"> {
  let allowedClauses = [OMPC_If, OMPC_NumThreads, OMPC_Default, OMPC_ProcBind,
      OMPC_Private, OMPC_FirstPrivate, OMPC_Shared, OMPC_Reduction, OMPC_Copyin,
      OMPC_LastPrivate, OMPC_Collapse, OMPC_Schedule, OMPC_Ordered, OMPC_Linear,
      OMPC_Allocate, OMPC_Order];
}
def OMP_ParallelForSimd : Directive<"parallel for simd"> {
  let allowedClauses = [OMPC_If, OMPC_NumThreads, OMPC_Default, OMPC_ProcBind,
      OMPC_Private, OMPC_FirstPrivate, OMPC_Shared, OMPC_Reduction, OMPC_Copyin,
      OMPC_LastPrivate, OMPC_Collapse, OMPC_Schedule, OMPC_SafeLen,
      OMPC_SimdLen, OMPC_Linear, OMPC_Aligned, OMPC_Ordered, OMPC_Allocate,
      OMPC_NonTemporal, OMPC_Order];
}
def OMP_ParallelMaster : Directive<"parallel master"> {
  let allowedClauses = [OMPC_If, OMPC_NumThreads, OMPC_Default, OMPC_Private,
      OMPC_FirstPrivate, OMPC_Shared, OMPC_Copyin, OMPC_Reduction,
      OMPC_ProcBind, OMPC_Allocate];
}
def OMP_ParallelSections : Directive<"parallel sections"> {
  let allowedClauses = [OMPC_If, OMPC_NumThreads, OMPC_Default, OMPC_ProcBind,
      OMPC_Private, OMPC_FirstPrivate, OMPC_Shared, OMPC_Reduction, OMPC_Copyin,
      OMPC_LastPrivate, OMPC_Allocate];
}
def OMP_ForSimd : Directive<"for simd"> {
  let allowedClauses = [OMPC_Private, OMPC_FirstPrivate, OMPC_LastPrivate,
      OMPC_Reduction, OMPC_Schedule, OMPC_Collapse, OMPC_NoWait, OMPC_SafeLen,
      OMPC_SimdLen, OMPC_Linear, OMPC_Aligned, OMPC_Ordered, OMPC_Allocate,
      OMPC_If, OMPC_NonTemporal, OMPC_Order];
}
def OMP_CancellationPoint : Directive<"cancellation point"> {}
def OMP_DeclareReduction : Directive<"declare reduction"> {}
def OMP_DeclareMapper : Directive<"declare mapper"> {
  let allowedClauses = [OMPC_Map];
}
def OMP_DeclareSimd : Directive<"declare simd"> {}
def OMP_TaskLoop : Directive<"taskloop"> {
  let allowedClauses = [OMPC_If, OMPC_Shared, OMPC_Private, OMPC_FirstPrivate,
      OMPC_LastPrivate, OMPC_Default, OMPC_Collapse, OMPC_Final, OMPC_Untied,
      OMPC_Mergeable, OMPC_Priority, OMPC_GrainSize, OMPC_NoGroup,
      OMPC_NumTasks, OMPC_Reduction, OMPC_InReduction, OMPC_Allocate];
}
def OMP_TaskLoopSimd : Directive<"taskloop simd"> {
  let allowedClauses = [OMPC_If, OMPC_Shared, OMPC_Private, OMPC_FirstPrivate,
      OMPC_LastPrivate, OMPC_Default, OMPC_Collapse, OMPC_Final, OMPC_Untied,
      OMPC_Mergeable, OMPC_Priority, OMPC_Linear, OMPC_Aligned, OMPC_SafeLen,
      OMPC_SimdLen, OMPC_GrainSize, OMPC_NoGroup, OMPC_NumTasks, OMPC_Reduction,
      OMPC_InReduction, OMPC_Allocator, OMPC_NonTemporal, OMPC_Order];
}
def OMP_Distribute : Directive<"distribute"> {
  let allowedClauses = [OMPC_Private, OMPC_FirstPrivate, OMPC_LastPrivate,
      OMPC_Collapse, OMPC_DistSchedule, OMPC_Allocate];
}
def OMP_DeclareTarget : Directive<"declare target"> {}
def OMP_EndDeclareTarget : Directive<"end declare target"> {}
def OMP_DistributeParallelFor : Directive<"distribute parallel for"> {
  let allowedClauses = [OMPC_FirstPrivate, OMPC_LastPrivate, OMPC_Collapse,
      OMPC_DistSchedule, OMPC_If, OMPC_NumThreads, OMPC_Default, OMPC_ProcBind,
      OMPC_Private, OMPC_Shared, OMPC_Reduction, OMPC_Copyin, OMPC_Schedule,
      OMPC_Allocate, OMPC_Order];
}
def OMP_DistributeParallelForSimd : Directive<"distribute parallel for simd"> {
  let allowedClauses = [OMPC_FirstPrivate, OMPC_LastPrivate, OMPC_Collapse,
      OMPC_DistSchedule, OMPC_If, OMPC_NumThreads, OMPC_Default, OMPC_ProcBind,
      OMPC_Private, OMPC_Shared, OMPC_Reduction, OMPC_Copyin, OMPC_Schedule,
      OMPC_Linear, OMPC_Aligned, OMPC_SafeLen, OMPC_SimdLen, OMPC_Allocate,
      OMPC_NonTemporal, OMPC_Order];
}
def OMP_DistributeSimd : Directive<"distribute simd"> {
  let allowedClauses = [OMPC_Private, OMPC_FirstPrivate, OMPC_LastPrivate,
      OMPC_Collapse, OMPC_DistSchedule, OMPC_Linear, OMPC_Aligned, OMPC_SafeLen,
      OMPC_SimdLen, OMPC_Reduction, OMPC_Allocate, OMPC_If, OMPC_NonTemporal,
      OMPC_Order];
}
def OMP_TargetParallelForSimd : Directive<"target parallel for simd"> {
  let allowedClauses = [OMPC_If, OMPC_Device, OMPC_Map, OMPC_Private,
      OMPC_FirstPrivate, OMPC_LastPrivate, OMPC_NoWait, OMPC_Depend,
      OMPC_DefaultMap, OMPC_NumThreads, OMPC_Default, OMPC_ProcBind,
      OMPC_Shared, OMPC_Reduction, OMPC_Collapse, OMPC_Schedule, OMPC_Ordered,
      OMPC_Linear, OMPC_SafeLen, OMPC_SimdLen, OMPC_Aligned, OMPC_IsDevicePtr,
      OMPC_Allocate, OMPC_NonTemporal, OMPC_Order, OMPC_UsesAllocators];
}
def OMP_TargetSimd : Directive<"target simd"> {
  let allowedClauses = [OMPC_If, OMPC_Device, OMPC_Map, OMPC_Private,
      OMPC_NoWait, OMPC_Depend, OMPC_DefaultMap, OMPC_FirstPrivate,
      OMPC_IsDevicePtr, OMPC_LastPrivate, OMPC_Linear, OMPC_Aligned,
      OMPC_SafeLen, OMPC_SimdLen, OMPC_Collapse, OMPC_Reduction, OMPC_Allocate,
      OMPC_NonTemporal, OMPC_Order, OMPC_UsesAllocators];
}
def OMP_TeamsDistribute : Directive<"teams distribute"> {
  let allowedClauses = [OMPC_Default, OMPC_Private, OMPC_FirstPrivate,
      OMPC_Shared, OMPC_Reduction, OMPC_NumTeams, OMPC_ThreadLimit,
      OMPC_LastPrivate, OMPC_Collapse, OMPC_DistSchedule, OMPC_Allocate];
}
def OMP_TeamsDistributeSimd : Directive<"teams distribute simd"> {
  let allowedClauses = [OMPC_Default, OMPC_Private, OMPC_FirstPrivate,
      OMPC_Shared, OMPC_Reduction, OMPC_NumTeams, OMPC_ThreadLimit,
      OMPC_LastPrivate, OMPC_Collapse, OMPC_DistSchedule, OMPC_Linear,
      OMPC_Aligned, OMPC_SafeLen, OMPC_SimdLen, OMPC_Allocate, OMPC_If,
      OMPC_NonTemporal, OMPC_Order];
}
def OMP_TeamsDistributeParallelForSimd :
    Directive<"teams distribute parallel for simd"> {
  let allowedClauses = [OMPC_FirstPrivate, OMPC_LastPrivate, OMPC_Collapse,
      OMPC_DistSchedule, OMPC_If, OMPC_NumThreads, OMPC_Default, OMPC_ProcBind,
      OMPC_Private, OMPC_Shared, OMPC_Reduction, OMPC_Schedule, OMPC_Linear,
      OMPC_Aligned, OMPC_SafeLen, OMPC_SimdLen, OMPC_NumTeams, OMPC_ThreadLimit,
      OMPC_Allocate, OMPC_NonTemporal, OMPC_Order];
}
def OMP_TeamsDistributeParallelFor :
    Directive<"teams distribute parallel for"> {
  let allowedClauses = [OMPC_FirstPrivate, OMPC_LastPrivate, OMPC_Collapse,
      OMPC_DistSchedule, OMPC_If, OMPC_NumThreads, OMPC_Default, OMPC_ProcBind,
      OMPC_Private, OMPC_Shared, OMPC_Reduction, OMPC_Schedule, OMPC_NumTeams,
      OMPC_ThreadLimit, OMPC_Copyin, OMPC_Allocate, OMPC_Order];
}
def OMP_TargetTeams : Directive<"target teams"> {
  let allowedClauses = [OMPC_If, OMPC_Device, OMPC_Map, OMPC_Private,
      OMPC_NoWait, OMPC_Depend, OMPC_DefaultMap, OMPC_FirstPrivate,
      OMPC_IsDevicePtr, OMPC_Default, OMPC_Shared, OMPC_Reduction,
      OMPC_NumTeams, OMPC_ThreadLimit, OMPC_Allocate, OMPC_UsesAllocators];
}
def OMP_TargetTeamsDistribute : Directive<"target teams distribute"> {
  let allowedClauses = [OMPC_If, OMPC_Device, OMPC_Map, OMPC_Private,
      OMPC_NoWait, OMPC_Depend, OMPC_DefaultMap, OMPC_FirstPrivate,
      OMPC_IsDevicePtr, OMPC_Default, OMPC_Shared, OMPC_Reduction,
      OMPC_NumTeams, OMPC_ThreadLimit, OMPC_LastPrivate, OMPC_Collapse,
      OMPC_DistSchedule, OMPC_Allocate, OMPC_UsesAllocators];
}
def OMP_TargetTeamsDistributeParallelFor :
    Directive<"target teams distribute parallel for"> {
  let allowedClauses = [OMPC_If, OMPC_Device, OMPC_Map, OMPC_Private,
      OMPC_NoWait, OMPC_Depend, OMPC_DefaultMap, OMPC_FirstPrivate,
      OMPC_IsDevicePtr, OMPC_Default, OMPC_Shared, OMPC_Reduction,
      OMPC_NumTeams, OMPC_ThreadLimit, OMPC_LastPrivate, OMPC_Collapse,
      OMPC_DistSchedule, OMPC_NumThreads, OMPC_ProcBind, OMPC_Schedule,
      OMPC_Allocate, OMPC_Order, OMPC_UsesAllocators];
}
def OMP_TargetTeamsDistributeParallelForSimd :
    Directive<"target teams distribute parallel for simd"> {
  let allowedClauses = [OMPC_If, OMPC_Device, OMPC_Map, OMPC_Private,
      OMPC_NoWait, OMPC_Depend, OMPC_DefaultMap, OMPC_FirstPrivate,
      OMPC_IsDevicePtr, OMPC_Default, OMPC_Shared, OMPC_Reduction,
      OMPC_NumTeams, OMPC_ThreadLimit, OMPC_LastPrivate, OMPC_Collapse,
      OMPC_DistSchedule, OMPC_NumThreads, OMPC_ProcBind, OMPC_Schedule,
      OMPC_Linear, OMPC_Aligned, OMPC_SafeLen, OMPC_SimdLen, OMPC_Allocate,
      OMPC_NonTemporal, OMPC_Order, OMPC_UsesAllocators];
}
def OMP_TargetTeamsDistributeSimd :
    Directive<"target teams distribute simd"> {
  let allowedClauses = [OMPC_If, OMPC_Device, OMPC_Map, OMPC_Private,
      OMPC_NoWait, OMPC_Depend, OMPC_DefaultMap, OMPC_FirstPrivate,
      OMPC_LastPrivate, OMPC_IsDevicePtr, OMPC_Shared, OMPC_Reduction,
      OMPC_NumTeams, OMPC_ThreadLimit, OMPC_Collapse, OMPC_DistSchedule,
      OMPC_Linear, OMPC_Aligned, OMPC_SafeLen, OMPC_SimdLen, OMPC_Allocate,
      OMPC_NonTemporal, OMPC_Order, OMPC_UsesAllocators];
}
def OMP_Allocate : Directive<"allocate"> {
  let allowedClauses = [OMPC_Allocator];
}
def OMP_DeclareVariant : Directive<"declare variant"> {
  let allowedClauses = [OMPC_Match];
}
def OMP_MasterTaskloop : Directive<"master taskloop"> {
  let allowedClauses = [
      OMPC_If, OMPC_Shared, OMPC_Private, OMPC_FirstPrivate, OMPC_LastPrivate,
      OMPC_Default, OMPC_Collapse, OMPC_Final, OMPC_Untied, OMPC_Mergeable,
      OMPC_Priority, OMPC_GrainSize, OMPC_NoGroup, OMPC_NumTasks,
      OMPC_Reduction, OMPC_InReduction, OMPC_Allocate];
}
def OMP_ParallelMasterTaskloop :
    Directive<"parallel master taskloop"> {
  let allowedClauses = [OMPC_If, OMPC_Shared, OMPC_Private, OMPC_FirstPrivate,
      OMPC_LastPrivate, OMPC_Default, OMPC_Collapse, OMPC_Final, OMPC_Untied,
      OMPC_Mergeable, OMPC_Priority, OMPC_GrainSize, OMPC_NoGroup,
      OMPC_NumTasks, OMPC_Reduction, OMPC_Allocate, OMPC_NumThreads,
      OMPC_ProcBind, OMPC_Copyin];
}
def OMP_MasterTaskloopSimd : Directive<"master taskloop simd"> {
  let allowedClauses = [OMPC_If, OMPC_Shared, OMPC_Private, OMPC_FirstPrivate,
      OMPC_LastPrivate, OMPC_DefaultMap, OMPC_Collapse, OMPC_Final, OMPC_Untied,
      OMPC_Mergeable, OMPC_Priority, OMPC_Linear, OMPC_Aligned, OMPC_SafeLen,
      OMPC_SimdLen, OMPC_GrainSize, OMPC_NoGroup, OMPC_NumTasks, OMPC_Reduction,
      OMPC_InReduction, OMPC_Allocate, OMPC_NonTemporal, OMPC_Order];
}
def OMP_ParallelMasterTaskloopSimd :
    Directive<"parallel master taskloop simd"> {
  let allowedClauses = [OMPC_If, OMPC_Shared, OMPC_Private, OMPC_FirstPrivate,
      OMPC_LastPrivate, OMPC_Default, OMPC_Collapse, OMPC_Final, OMPC_Untied,
      OMPC_Mergeable, OMPC_Priority, OMPC_GrainSize, OMPC_NoGroup,
      OMPC_NumTasks, OMPC_Reduction, OMPC_Allocate, OMPC_NumThreads,
      OMPC_ProcBind, OMPC_Copyin, OMPC_Linear, OMPC_Aligned, OMPC_SafeLen,
      OMPC_SimdLen, OMPC_NonTemporal, OMPC_Order];
}
def OMP_Depobj : Directive<"depobj"> {
  let allowedClauses = [OMPC_Depend, OMPC_Destroy, OMPC_Update, OMPC_Depobj];
}
def OMP_Scan : Directive<"scan"> {
  let allowedClauses = [OMPC_Inclusive, OMPC_Exclusive];
}
def OMP_BeginDeclareVariant : Directive<"begin declare variant"> {}
def OMP_EndDeclareVariant : Directive<"end declare variant"> {}
def OMP_Unknown : Directive<"unknown"> {}
