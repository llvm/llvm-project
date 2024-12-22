//===- Utils.h - MLIR NVVM target utils -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files declares NVVM target related utility classes and functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_LLVM_NVVM_UTILS_H
#define MLIR_TARGET_LLVM_NVVM_UTILS_H

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Target/LLVM/ModuleToObject.h"

namespace mlir {
namespace NVVM {
/// Searches & returns the path CUDA toolkit path, the search order is:
/// 1. The `CUDA_ROOT` environment variable.
/// 2. The `CUDA_HOME` environment variable.
/// 3. The `CUDA_PATH` environment variable.
/// 4. The CUDA toolkit path detected by CMake.
/// 5. Returns an empty string.
StringRef getCUDAToolkitPath();

/// Base class for all NVVM serializations from GPU modules into binary strings.
/// By default this class serializes into LLVM bitcode.
class SerializeGPUModuleBase : public LLVM::ModuleToObject {
public:
  /// Initializes the `toolkitPath` with the path in `targetOptions` or if empty
  /// with the path in `getCUDAToolkitPath`.
  SerializeGPUModuleBase(Operation &module, NVVMTargetAttr target,
                         const gpu::TargetOptions &targetOptions = {});

  /// Initializes the LLVM NVPTX target by safely calling `LLVMInitializeNVPTX*`
  /// methods if available.
  static void init();

  /// Returns the target attribute.
  NVVMTargetAttr getTarget() const;

  /// Returns the CUDA toolkit path.
  StringRef getToolkitPath() const;

  /// Returns the bitcode files to be loaded.
  ArrayRef<std::string> getFileList() const;

  /// Appends `nvvm/libdevice.bc` into `fileList`. Returns failure if the
  /// library couldn't be found.
  LogicalResult appendStandardLibs();

  /// Loads the bitcode files in `fileList`.
  virtual std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
  loadBitcodeFiles(llvm::Module &module) override;

protected:
  /// NVVM target attribute.
  NVVMTargetAttr target;

  /// CUDA toolkit path.
  std::string toolkitPath;

  /// List of LLVM bitcode files to link to.
  SmallVector<std::string> fileList;
};
} // namespace NVVM
} // namespace mlir

#endif // MLIR_TARGET_LLVM_NVVM_UTILS_H
