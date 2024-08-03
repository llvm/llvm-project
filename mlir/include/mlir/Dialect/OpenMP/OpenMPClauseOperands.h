//===-- OpenMPClauseOperands.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the structures defining MLIR operands associated with each
// OpenMP clause, and structures grouping the appropriate operands for each
// construct.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_OPENMPCLAUSEOPERANDS_H_
#define MLIR_DIALECT_OPENMP_OPENMPCLAUSEOPERANDS_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/OpenMP/OpenMPOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/OpenMP/OpenMPOpsAttributes.h.inc"

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// Mixin structures defining MLIR operands associated with each OpenMP clause.
//===----------------------------------------------------------------------===//

struct AlignedClauseOps {
  llvm::SmallVector<Value> alignedVars;
  llvm::SmallVector<Attribute> alignments;
};

struct AllocateClauseOps {
  llvm::SmallVector<Value> allocateVars, allocatorVars;
};

struct CancelDirectiveNameClauseOps {
  ClauseCancellationConstructTypeAttr cancelDirective;
};

struct CopyprivateClauseOps {
  llvm::SmallVector<Value> copyprivateVars;
  llvm::SmallVector<Attribute> copyprivateSyms;
};

struct CriticalNameClauseOps {
  /// This field has a generic name because it's mirroring the `sym_name`
  /// argument of the `OpenMP_CriticalNameClause` tablegen definition. That one
  /// can't be renamed to anything more specific because the `sym_name` name is
  /// a requirement of the `Symbol` MLIR trait associated with that clause.
  StringAttr symName;
};

struct DependClauseOps {
  llvm::SmallVector<Attribute> dependKinds;
  llvm::SmallVector<Value> dependVars;
};

struct DeviceClauseOps {
  Value device;
};

struct DeviceTypeClauseOps {
  // The default capture type.
  DeclareTargetDeviceType deviceType = DeclareTargetDeviceType::any;
};

struct DistScheduleClauseOps {
  UnitAttr distScheduleStatic;
  Value distScheduleChunkSize;
};

struct DoacrossClauseOps {
  ClauseDependAttr doacrossDependType;
  IntegerAttr doacrossNumLoops;
  llvm::SmallVector<Value> doacrossDependVars;
};

struct FilterClauseOps {
  Value filteredThreadId;
};

struct FinalClauseOps {
  Value final;
};

struct GrainsizeClauseOps {
  Value grainsize;
};

struct HasDeviceAddrClauseOps {
  llvm::SmallVector<Value> hasDeviceAddrVars;
};

struct HintClauseOps {
  IntegerAttr hint;
};

struct IfClauseOps {
  Value ifVar;
};

struct InReductionClauseOps {
  llvm::SmallVector<Value> inReductionVars;
  llvm::SmallVector<bool> inReductionByref;
  llvm::SmallVector<Attribute> inReductionSyms;
};

struct IsDevicePtrClauseOps {
  llvm::SmallVector<Value> isDevicePtrVars;
};

struct LinearClauseOps {
  llvm::SmallVector<Value> linearVars, linearStepVars;
};

struct LoopRelatedOps {
  llvm::SmallVector<Value> loopLowerBounds, loopUpperBounds, loopSteps;
  UnitAttr loopInclusive;
};

struct MapClauseOps {
  llvm::SmallVector<Value> mapVars;
};

struct MergeableClauseOps {
  UnitAttr mergeable;
};

struct NogroupClauseOps {
  UnitAttr nogroup;
};

struct NontemporalClauseOps {
  llvm::SmallVector<Value> nontemporalVars;
};

struct NowaitClauseOps {
  UnitAttr nowait;
};

struct NumTasksClauseOps {
  Value numTasks;
};

struct NumTeamsClauseOps {
  Value numTeamsLower, numTeamsUpper;
};

struct NumThreadsClauseOps {
  Value numThreads;
};

struct OrderClauseOps {
  ClauseOrderKindAttr order;
  OrderModifierAttr orderMod;
};

struct OrderedClauseOps {
  IntegerAttr ordered;
};

struct ParallelizationLevelClauseOps {
  UnitAttr parLevelSimd;
};

struct PriorityClauseOps {
  Value priority;
};

struct PrivateClauseOps {
  // SSA values that correspond to "original" values being privatized.
  // They refer to the SSA value outside the OpenMP region from which a clone is
  // created inside the region.
  llvm::SmallVector<Value> privateVars;
  // The list of symbols referring to delayed privatizer ops (i.e. `omp.private`
  // ops).
  llvm::SmallVector<Attribute> privateSyms;
};

struct ProcBindClauseOps {
  ClauseProcBindKindAttr procBindKind;
};

struct ReductionClauseOps {
  llvm::SmallVector<Value> reductionVars;
  llvm::SmallVector<bool> reductionByref;
  llvm::SmallVector<Attribute> reductionSyms;
};

struct SafelenClauseOps {
  IntegerAttr safelen;
};

struct ScheduleClauseOps {
  ClauseScheduleKindAttr scheduleKind;
  Value scheduleChunk;
  ScheduleModifierAttr scheduleMod;
  UnitAttr scheduleSimd;
};

struct SimdlenClauseOps {
  IntegerAttr simdlen;
};

struct TaskReductionClauseOps {
  llvm::SmallVector<Value> taskReductionVars;
  llvm::SmallVector<bool> taskReductionByref;
  llvm::SmallVector<Attribute> taskReductionSyms;
};

struct ThreadLimitClauseOps {
  Value threadLimit;
};

struct UntiedClauseOps {
  UnitAttr untied;
};

struct UseDeviceAddrClauseOps {
  llvm::SmallVector<Value> useDeviceAddrVars;
};

struct UseDevicePtrClauseOps {
  llvm::SmallVector<Value> useDevicePtrVars;
};

//===----------------------------------------------------------------------===//
// Structures defining clause operands associated with each OpenMP leaf
// construct.
//
// These mirror the arguments expected by the corresponding OpenMP MLIR ops.
//===----------------------------------------------------------------------===//

namespace detail {
template <typename... Mixins>
struct Clauses : public Mixins... {};
} // namespace detail

using CancelOperands =
    detail::Clauses<CancelDirectiveNameClauseOps, IfClauseOps>;

using CancellationPointOperands = detail::Clauses<CancelDirectiveNameClauseOps>;

using CriticalDeclareOperands =
    detail::Clauses<CriticalNameClauseOps, HintClauseOps>;

// TODO `indirect` clause.
using DeclareTargetOperands = detail::Clauses<DeviceTypeClauseOps>;

using DistributeOperands =
    detail::Clauses<AllocateClauseOps, DistScheduleClauseOps, OrderClauseOps,
                    PrivateClauseOps>;

using LoopNestOperands = detail::Clauses<LoopRelatedOps>;

using MaskedOperands = detail::Clauses<FilterClauseOps>;

using OrderedOperands = detail::Clauses<DoacrossClauseOps>;

using OrderedRegionOperands = detail::Clauses<ParallelizationLevelClauseOps>;

using ParallelOperands =
    detail::Clauses<AllocateClauseOps, IfClauseOps, NumThreadsClauseOps,
                    PrivateClauseOps, ProcBindClauseOps, ReductionClauseOps>;

using SectionsOperands = detail::Clauses<AllocateClauseOps, NowaitClauseOps,
                                         PrivateClauseOps, ReductionClauseOps>;

using SimdOperands =
    detail::Clauses<AlignedClauseOps, IfClauseOps, LinearClauseOps,
                    NontemporalClauseOps, OrderClauseOps, PrivateClauseOps,
                    ReductionClauseOps, SafelenClauseOps, SimdlenClauseOps>;

using SingleOperands = detail::Clauses<AllocateClauseOps, CopyprivateClauseOps,
                                       NowaitClauseOps, PrivateClauseOps>;

// TODO `defaultmap`, `uses_allocators` clauses.
using TargetOperands =
    detail::Clauses<AllocateClauseOps, DependClauseOps, DeviceClauseOps,
                    HasDeviceAddrClauseOps, IfClauseOps, InReductionClauseOps,
                    IsDevicePtrClauseOps, MapClauseOps, NowaitClauseOps,
                    PrivateClauseOps, ThreadLimitClauseOps>;

using TargetDataOperands =
    detail::Clauses<DeviceClauseOps, IfClauseOps, MapClauseOps,
                    UseDeviceAddrClauseOps, UseDevicePtrClauseOps>;

using TargetEnterExitUpdateDataOperands =
    detail::Clauses<DependClauseOps, DeviceClauseOps, IfClauseOps, MapClauseOps,
                    NowaitClauseOps>;

// TODO `affinity`, `detach` clauses.
using TaskOperands =
    detail::Clauses<AllocateClauseOps, DependClauseOps, FinalClauseOps,
                    IfClauseOps, InReductionClauseOps, MergeableClauseOps,
                    PriorityClauseOps, PrivateClauseOps, UntiedClauseOps>;

using TaskgroupOperands =
    detail::Clauses<AllocateClauseOps, TaskReductionClauseOps>;

using TaskloopOperands =
    detail::Clauses<AllocateClauseOps, FinalClauseOps, GrainsizeClauseOps,
                    IfClauseOps, InReductionClauseOps, MergeableClauseOps,
                    NogroupClauseOps, NumTasksClauseOps, PriorityClauseOps,
                    PrivateClauseOps, ReductionClauseOps, UntiedClauseOps>;

using TaskwaitOperands = detail::Clauses<DependClauseOps, NowaitClauseOps>;

using TeamsOperands =
    detail::Clauses<AllocateClauseOps, IfClauseOps, NumTeamsClauseOps,
                    PrivateClauseOps, ReductionClauseOps, ThreadLimitClauseOps>;

using WsloopOperands =
    detail::Clauses<AllocateClauseOps, LinearClauseOps, NowaitClauseOps,
                    OrderClauseOps, OrderedClauseOps, PrivateClauseOps,
                    ReductionClauseOps, ScheduleClauseOps>;

} // namespace omp
} // namespace mlir

#endif // MLIR_DIALECT_OPENMP_OPENMPCLAUSEOPERANDS_H_
