//===- OpenMPOffloadUtils.h - OpenMP offload utilities ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Shared utilities for setting OpenMP offload module interface attributes.
/// These are used by both Flang and Clang (CIR) frontends.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_OPENMPOFFLOADUTILS_H_
#define MLIR_DIALECT_OPENMP_OPENMPOFFLOADUTILS_H_

#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/Dialect/OpenMP/OpenMPOpsAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdint>
#include <string>
#include <vector>

namespace mlir::omp {

struct OffloadModuleOpts {
  OffloadModuleOpts() = default;
  OffloadModuleOpts(uint32_t openMPTargetDebug, bool openMPTeamSubscription,
                    bool openMPThreadSubscription, bool openMPNoThreadState,
                    bool openMPNoNestedParallelism, bool openMPIsTargetDevice,
                    bool openMPIsGPU, bool openMPForceUSM,
                    uint32_t openMPVersion, std::string ompHostIRFile = {},
                    const std::vector<llvm::Triple> &ompTargetTriples = {},
                    bool noGPULib = false)
      : OpenMPTargetDebug(openMPTargetDebug),
        OpenMPTeamSubscription(openMPTeamSubscription),
        OpenMPThreadSubscription(openMPThreadSubscription),
        OpenMPNoThreadState(openMPNoThreadState),
        OpenMPNoNestedParallelism(openMPNoNestedParallelism),
        OpenMPIsTargetDevice(openMPIsTargetDevice), OpenMPIsGPU(openMPIsGPU),
        OpenMPForceUSM(openMPForceUSM), OpenMPVersion(openMPVersion),
        OMPHostIRFile(std::move(ompHostIRFile)),
        OMPTargetTriples(ompTargetTriples.begin(), ompTargetTriples.end()),
        NoGPULib(noGPULib) {}

  uint32_t OpenMPTargetDebug = 0;
  bool OpenMPTeamSubscription = false;
  bool OpenMPThreadSubscription = false;
  bool OpenMPNoThreadState = false;
  bool OpenMPNoNestedParallelism = false;
  bool OpenMPIsTargetDevice = false;
  bool OpenMPIsGPU = false;
  bool OpenMPForceUSM = false;
  uint32_t OpenMPVersion = 31;
  std::string OMPHostIRFile = {};
  std::vector<llvm::Triple> OMPTargetTriples = {};
  bool NoGPULib = false;
};

/// Sets OpenMP offload module interface attributes on a ModuleOp, shared
/// between Flang and Clang (CIR) frontends.
[[maybe_unused]] static void
setOffloadModuleInterfaceAttributes(ModuleOp module, OffloadModuleOpts opts) {
  if (auto offloadMod =
          llvm::dyn_cast<OffloadModuleInterface>(module.getOperation())) {
    offloadMod.setIsTargetDevice(opts.OpenMPIsTargetDevice);
    offloadMod.setIsGPU(opts.OpenMPIsGPU);
    if (opts.OpenMPForceUSM)
      offloadMod.setRequires(ClauseRequires::unified_shared_memory);
    if (opts.OpenMPIsTargetDevice) {
      offloadMod.setFlags(
          opts.OpenMPTargetDebug, opts.OpenMPTeamSubscription,
          opts.OpenMPThreadSubscription, opts.OpenMPNoThreadState,
          opts.OpenMPNoNestedParallelism, opts.OpenMPVersion, opts.NoGPULib);
      if (!opts.OMPHostIRFile.empty())
        offloadMod.setHostIRFilePath(opts.OMPHostIRFile);
    }
    auto strTriples = llvm::to_vector(
        llvm::map_range(opts.OMPTargetTriples, [](llvm::Triple triple) {
          return triple.normalize();
        }));
    offloadMod.setTargetTriples(strTriples);
  }
}

[[maybe_unused]] static void setOpenMPVersionAttribute(ModuleOp module,
                                                       int64_t version) {
  module.getOperation()->setAttr(
      StringAttr::get(module.getContext(), llvm::Twine{"omp.version"}),
      VersionAttr::get(module.getContext(), version));
}

[[maybe_unused]] static int64_t
getOpenMPVersionAttribute(ModuleOp module, int64_t fallback = -1) {
  if (Attribute verAttr = module->getAttr("omp.version"))
    return llvm::cast<VersionAttr>(verAttr).getVersion();
  return fallback;
}

} // namespace mlir::omp

#endif // MLIR_DIALECT_OPENMP_OPENMPOFFLOADUTILS_H_
