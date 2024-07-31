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

#include "mlir/Dialect/OpenMP/OpenMPClauseOps.h.inc"

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// Extra clause operand structures.
//===----------------------------------------------------------------------===//

struct DeviceTypeClauseOps {
  /// The default capture type.
  DeclareTargetDeviceType deviceType = DeclareTargetDeviceType::any;
};

//===----------------------------------------------------------------------===//
// Extra operation operand structures.
//===----------------------------------------------------------------------===//

// TODO: Add `indirect` clause.
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

using WorkshareOperands = detail::Clauses<NowaitClauseOps>;

using WsloopOperands =
    detail::Clauses<AllocateClauseOps, LinearClauseOps, NowaitClauseOps,
                    OrderClauseOps, OrderedClauseOps, PrivateClauseOps,
                    ReductionClauseOps, ScheduleClauseOps>;

} // namespace omp
} // namespace mlir

#endif // MLIR_DIALECT_OPENMP_OPENMPCLAUSEOPERANDS_H_
