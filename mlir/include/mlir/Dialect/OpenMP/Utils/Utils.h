//===- Utils.h - OpenMP dialect utilities -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for various OpenMP utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_UTILS_UTILS_H_
#define MLIR_DIALECT_OPENMP_UTILS_UTILS_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace omp {

/// Offload-specific OpenMP module attributes, associated to the
/// OffloadModuleInterface.
struct OffloadModuleOpts {
  OffloadModuleOpts() = default;
  OffloadModuleOpts(uint32_t targetDebugKind, bool assumeTeamsOversubscription,
                    bool assumeThreadsOversubscription,
                    bool assumeNoThreadState, bool assumeNoNestedParallelism,
                    bool isTargetDevice, bool isGPU, bool forceUSM,
                    uint32_t openMPDeviceVersion, const Twine &hostIRFile,
                    ArrayRef<llvm::Triple> targetTriples = {},
                    bool noGPULib = false)
      : targetDebugKind(targetDebugKind),
        assumeTeamsOversubscription(assumeTeamsOversubscription),
        assumeThreadsOversubscription(assumeThreadsOversubscription),
        assumeNoThreadState(assumeNoThreadState),
        assumeNoNestedParallelism(assumeNoNestedParallelism),
        isTargetDevice(isTargetDevice), isGPU(isGPU), forceUSM(forceUSM),
        openMPDeviceVersion(openMPDeviceVersion), hostIRFile(hostIRFile.str()),
        targetTriples(targetTriples.begin(), targetTriples.end()),
        noGPULib(noGPULib) {}

  uint32_t targetDebugKind = 0;
  bool assumeTeamsOversubscription = false;
  bool assumeThreadsOversubscription = false;
  bool assumeNoThreadState = false;
  bool assumeNoNestedParallelism = false;
  bool isTargetDevice = false;
  bool isGPU = false;
  bool forceUSM = false;
  uint32_t openMPDeviceVersion = 31;
  std::string hostIRFile = {};
  std::vector<llvm::Triple> targetTriples = {};
  bool noGPULib = false;
};

/// Sets OpenMP offload module interface attributes on a ModuleOp, shared
/// between Flang and Clang (CIR) frontends.
void setOffloadModuleInterfaceAttributes(ModuleOp module,
                                         const OffloadModuleOpts &opts);

/// Adds or updates the omp.version attribute.
void setOpenMPVersionAttribute(ModuleOp module, int64_t version);

/// Add the omp.integer_wrap_around attribute.
void setOpenMPIntegerWrapAround(ModuleOp module, bool value);

/// Returns the value of the omp.version attribute, if present, or the fallback.
int64_t getOpenMPVersionAttribute(ModuleOp module, int64_t fallback = -1);

/// Checks whether this is an OpenMP-enabled module.
bool isOpenMPModule(ModuleOp module);

/// Check whether the value representing an allocation, assumed to have been
/// defined in a shared device context, is used in a manner that would require
/// device shared memory for correctness.
///
/// When a use takes place inside an omp.parallel region and it's not as a
/// private clause argument, or when it is a reduction argument passed to
/// omp.parallel or a function call argument, then the defining allocation is
/// eligible for replacement with shared memory.
///
/// \see mlir::omp::opInSharedDeviceContext().
bool allocaUsesRequireSharedMem(Value alloc);

/// Check whether the given operation is located in a context where an
/// allocation to be used by multiple threads in a parallel region would have to
/// be placed in device shared memory to be accessible.
///
/// That means that it is inside of a target device module, it is a non-SPMD
/// target region, is inside of one or it's located in a device function, and it
/// is not not inside of a parallel region.
///
/// This represents a necessary but not sufficient set of conditions to use
/// device shared memory in place of regular allocas. For some variables, the
/// associated OpenMP construct or their uses might also need to be taken into
/// account.
///
/// \see mlir::omp::allocaUsesRequireSharedMem().
bool opInSharedDeviceContext(Operation &op);

} // namespace omp
} // namespace mlir

#endif // MLIR_DIALECT_OPENMP_UTILS_UTILS_H_
