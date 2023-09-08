//===- Utils.h - MLIR ROCDL target utils ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files declares ROCDL target related utility classes and functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVM_ROCDL_UTILS_H
#define MLIR_TARGET_LLVM_ROCDL_UTILS_H

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Target/LLVM/ModuleToObject.h"

namespace mlir {
namespace ROCDL {
/// Searches & returns the path ROCM toolkit path, the search order is:
/// 1. The `ROCM_PATH` environment variable.
/// 2. The `ROCM_ROOT` environment variable.
/// 3. The `ROCM_HOME` environment variable.
/// 4. The ROCM path detected by CMake.
/// 5. Returns an empty string.
StringRef getROCMPath();

/// Base class for all ROCDL serializations from GPU modules into binary
/// strings. By default this class serializes into LLVM bitcode.
class SerializeGPUModuleBase : public LLVM::ModuleToObject {
public:
  /// Initializes the `toolkitPath` with the path in `targetOptions` or if empty
  /// with the path in `getROCMPath`.
  SerializeGPUModuleBase(Operation &module, ROCDLTargetAttr target,
                         const gpu::TargetOptions &targetOptions = {});

  /// Initializes the LLVM AMDGPU target by safely calling
  /// `LLVMInitializeAMDGPU*` methods if available.
  static void init();

  /// Returns the target attribute.
  ROCDLTargetAttr getTarget() const;

  /// Returns the ROCM toolkit path.
  StringRef getToolkitPath() const;

  /// Returns the bitcode files to be loaded.
  ArrayRef<std::string> getFileList() const;

  /// Appends standard ROCm device libraries like `ocml.bc`, `ockl.bc`, etc.
  LogicalResult appendStandardLibs();

  /// Loads the bitcode files in `fileList`.
  virtual std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
  loadBitcodeFiles(llvm::Module &module,
                   llvm::TargetMachine &targetMachine) override;

  /// Adds `oclc` control variables to the LLVM module.
  void handleModulePreLink(llvm::Module &module,
                           llvm::TargetMachine &targetMachine) override;

  /// Removes unnecessary metadata from the loaded bitcode files.
  LogicalResult handleBitcodeFile(llvm::Module &module,
                                  llvm::TargetMachine &targetMachine) override;

protected:
  /// Appends the paths of common ROCm device libraries to `libs`.
  LogicalResult getCommonBitcodeLibs(llvm::SmallVector<std::string> &libs,
                                     SmallVector<char, 256> &libPath,
                                     StringRef isaVersion);

  /// Adds `oclc` control variables to the LLVM module.
  void addControlVariables(llvm::Module &module, bool wave64, bool daz,
                           bool finiteOnly, bool unsafeMath, bool fastMath,
                           bool correctSqrt, StringRef abiVer);

  /// Returns the assembled ISA.
  std::optional<SmallVector<char, 0>> assembleIsa(StringRef isa);

  /// ROCDL target attribute.
  ROCDLTargetAttr target;

  /// ROCM toolkit path.
  std::string toolkitPath;

  /// List of LLVM bitcode files to link to.
  SmallVector<std::string> fileList;
};
} // namespace ROCDL
} // namespace mlir

#endif // MLIR_TARGET_LLVM_ROCDL_UTILS_H
