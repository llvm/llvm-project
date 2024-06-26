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
  llvm::SmallVector<Attribute> alignmentAttrs;
};

struct AllocateClauseOps {
  llvm::SmallVector<Value> allocatorVars, allocateVars;
};

struct CollapseClauseOps {
  llvm::SmallVector<Value> loopLBVar, loopUBVar, loopStepVar;
};

struct CopyprivateClauseOps {
  llvm::SmallVector<Value> copyprivateVars;
  llvm::SmallVector<Attribute> copyprivateFuncs;
};

struct DependClauseOps {
  llvm::SmallVector<Attribute> dependTypeAttrs;
  llvm::SmallVector<Value> dependVars;
};

struct DeviceClauseOps {
  Value deviceVar;
};

struct DeviceTypeClauseOps {
  // The default capture type.
  DeclareTargetDeviceType deviceType = DeclareTargetDeviceType::any;
};

struct DistScheduleClauseOps {
  UnitAttr distScheduleStaticAttr;
  Value distScheduleChunkSizeVar;
};

struct DoacrossClauseOps {
  llvm::SmallVector<Value> doacrossVectorVars;
  ClauseDependAttr doacrossDependTypeAttr;
  IntegerAttr doacrossNumLoopsAttr;
};

struct FinalClauseOps {
  Value finalVar;
};

struct GrainsizeClauseOps {
  Value grainsizeVar;
};

struct HasDeviceAddrClauseOps {
  llvm::SmallVector<Value> hasDeviceAddrVars;
};
struct HintClauseOps {
  IntegerAttr hintAttr;
};

struct IfClauseOps {
  Value ifVar;
};

struct InReductionClauseOps {
  llvm::SmallVector<Value> inReductionVars;
  llvm::SmallVector<Attribute> inReductionDeclSymbols;
};

struct IsDevicePtrClauseOps {
  llvm::SmallVector<Value> isDevicePtrVars;
};

struct LinearClauseOps {
  llvm::SmallVector<Value> linearVars, linearStepVars;
};

struct LoopRelatedOps {
  UnitAttr loopInclusiveAttr;
};

struct MapClauseOps {
  llvm::SmallVector<Value> mapVars;
};

struct MergeableClauseOps {
  UnitAttr mergeableAttr;
};

struct NameClauseOps {
  StringAttr nameAttr;
};

struct NogroupClauseOps {
  UnitAttr nogroupAttr;
};

struct NontemporalClauseOps {
  llvm::SmallVector<Value> nontemporalVars;
};

struct NowaitClauseOps {
  UnitAttr nowaitAttr;
};

struct NumTasksClauseOps {
  Value numTasksVar;
};

struct NumTeamsClauseOps {
  Value numTeamsLowerVar, numTeamsUpperVar;
};

struct NumThreadsClauseOps {
  Value numThreadsVar;
};

struct OrderClauseOps {
  ClauseOrderKindAttr orderAttr;
  OrderModifierAttr orderModAttr;
};

struct OrderedClauseOps {
  IntegerAttr orderedAttr;
};

struct ParallelizationLevelClauseOps {
  UnitAttr parLevelSimdAttr;
};

struct PriorityClauseOps {
  Value priorityVar;
};

struct PrivateClauseOps {
  // SSA values that correspond to "original" values being privatized.
  // They refer to the SSA value outside the OpenMP region from which a clone is
  // created inside the region.
  llvm::SmallVector<Value> privateVars;
  // The list of symbols referring to delayed privatizer ops (i.e. `omp.private`
  // ops).
  llvm::SmallVector<Attribute> privatizers;
};

struct ProcBindClauseOps {
  ClauseProcBindKindAttr procBindKindAttr;
};

struct ReductionClauseOps {
  llvm::SmallVector<Value> reductionVars;
  llvm::SmallVector<bool> reduceVarByRef;
  llvm::SmallVector<Attribute> reductionDeclSymbols;
};

struct SafelenClauseOps {
  IntegerAttr safelenAttr;
};

struct ScheduleClauseOps {
  ClauseScheduleKindAttr scheduleValAttr;
  ScheduleModifierAttr scheduleModAttr;
  Value scheduleChunkVar;
  UnitAttr scheduleSimdAttr;
};

struct SimdlenClauseOps {
  IntegerAttr simdlenAttr;
};

struct TaskReductionClauseOps {
  llvm::SmallVector<Value> taskReductionVars;
  llvm::SmallVector<Attribute> taskReductionDeclSymbols;
};

struct ThreadLimitClauseOps {
  Value threadLimitVar;
};

struct UntiedClauseOps {
  UnitAttr untiedAttr;
};

struct UseDeviceClauseOps {
  llvm::SmallVector<Value> useDevicePtrVars, useDeviceAddrVars;
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

using CriticalClauseOps = detail::Clauses<HintClauseOps, NameClauseOps>;

// TODO `indirect` clause.
using DeclareTargetClauseOps = detail::Clauses<DeviceTypeClauseOps>;

using DistributeClauseOps =
    detail::Clauses<AllocateClauseOps, DistScheduleClauseOps, OrderClauseOps,
                    PrivateClauseOps>;

using LoopNestClauseOps = detail::Clauses<CollapseClauseOps, LoopRelatedOps>;

// TODO `filter` clause.
using MaskedClauseOps = detail::Clauses<>;

using OrderedOpClauseOps = detail::Clauses<DoacrossClauseOps>;

using OrderedRegionClauseOps = detail::Clauses<ParallelizationLevelClauseOps>;

using ParallelClauseOps =
    detail::Clauses<AllocateClauseOps, IfClauseOps, NumThreadsClauseOps,
                    PrivateClauseOps, ProcBindClauseOps, ReductionClauseOps>;

using SectionsClauseOps = detail::Clauses<AllocateClauseOps, NowaitClauseOps,
                                          PrivateClauseOps, ReductionClauseOps>;

// TODO `linear` clause.
using SimdClauseOps =
    detail::Clauses<AlignedClauseOps, IfClauseOps, NontemporalClauseOps,
                    OrderClauseOps, PrivateClauseOps, ReductionClauseOps,
                    SafelenClauseOps, SimdlenClauseOps>;

using SingleClauseOps = detail::Clauses<AllocateClauseOps, CopyprivateClauseOps,
                                        NowaitClauseOps, PrivateClauseOps>;

// TODO `defaultmap`, `uses_allocators` clauses.
using TargetClauseOps =
    detail::Clauses<AllocateClauseOps, DependClauseOps, DeviceClauseOps,
                    HasDeviceAddrClauseOps, IfClauseOps, InReductionClauseOps,
                    IsDevicePtrClauseOps, MapClauseOps, NowaitClauseOps,
                    PrivateClauseOps, ReductionClauseOps, ThreadLimitClauseOps>;

using TargetDataClauseOps = detail::Clauses<DeviceClauseOps, IfClauseOps,
                                            MapClauseOps, UseDeviceClauseOps>;

using TargetEnterExitUpdateDataClauseOps =
    detail::Clauses<DependClauseOps, DeviceClauseOps, IfClauseOps, MapClauseOps,
                    NowaitClauseOps>;

// TODO `affinity`, `detach` clauses.
using TaskClauseOps =
    detail::Clauses<AllocateClauseOps, DependClauseOps, FinalClauseOps,
                    IfClauseOps, InReductionClauseOps, MergeableClauseOps,
                    PriorityClauseOps, PrivateClauseOps, UntiedClauseOps>;

using TaskgroupClauseOps =
    detail::Clauses<AllocateClauseOps, TaskReductionClauseOps>;

using TaskloopClauseOps =
    detail::Clauses<AllocateClauseOps, FinalClauseOps, GrainsizeClauseOps,
                    IfClauseOps, InReductionClauseOps, MergeableClauseOps,
                    NogroupClauseOps, NumTasksClauseOps, PriorityClauseOps,
                    PrivateClauseOps, ReductionClauseOps, UntiedClauseOps>;

using TaskwaitClauseOps = detail::Clauses<DependClauseOps, NowaitClauseOps>;

using TeamsClauseOps =
    detail::Clauses<AllocateClauseOps, IfClauseOps, NumTeamsClauseOps,
                    PrivateClauseOps, ReductionClauseOps, ThreadLimitClauseOps>;

using WsloopClauseOps =
    detail::Clauses<AllocateClauseOps, LinearClauseOps, NowaitClauseOps,
                    OrderClauseOps, OrderedClauseOps, PrivateClauseOps,
                    ReductionClauseOps, ScheduleClauseOps>;

} // namespace omp
} // namespace mlir

#endif // MLIR_DIALECT_OPENMP_OPENMPCLAUSEOPERANDS_H_
