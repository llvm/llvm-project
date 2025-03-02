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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/LLVM.h"
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

/// Helper enum for specifying the AMD GCN device libraries required for
/// compilation.
enum class AMDGCNLibraries : uint32_t {
  None = 0,
  Ockl = 1,
  Ocml = 2,
  OpenCL = 4,
  Hip = 8,
  LastLib = Hip,
  LLVM_MARK_AS_BITMASK_ENUM(LastLib),
  All = (LastLib << 1) - 1
};

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

  /// Returns the LLVM bitcode libraries to be linked.
  ArrayRef<Attribute> getLibrariesToLink() const;

  /// Appends standard ROCm device libraries to `fileList`.
  LogicalResult appendStandardLibs(AMDGCNLibraries libs);

  /// Loads the bitcode files in `fileList`.
  virtual std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
  loadBitcodeFiles(llvm::Module &module) override;

  /// Determines required Device Libraries and adds `oclc` control variables to
  /// the LLVM Module if needed. Also sets
  /// `amdhsa_code_object_version` module flag.
  void handleModulePreLink(llvm::Module &module) override;

  /// Removes unnecessary metadata from the loaded bitcode files.
  LogicalResult handleBitcodeFile(llvm::Module &module) override;

protected:
  /// Adds `oclc` control variables to the LLVM Module if needed. It also sets
  /// `amdhsa_code_object_version` module flag which is equal to ABI version and
  /// it uses "llvm::Module::Error" to set that flag.
  void addControlVariables(llvm::Module &module, AMDGCNLibraries libs,
                           bool wave64, bool daz, bool finiteOnly,
                           bool unsafeMath, bool fastMath, bool correctSqrt,
                           StringRef abiVer);

  /// Compiles assembly to a binary.
  virtual std::optional<SmallVector<char, 0>>
  compileToBinary(const std::string &serializedISA);

  /// Default implementation of `ModuleToObject::moduleToObject`.
  std::optional<SmallVector<char, 0>>
  moduleToObjectImpl(const gpu::TargetOptions &targetOptions,
                     llvm::Module &llvmModule);

  /// Returns the assembled ISA.
  std::optional<SmallVector<char, 0>> assembleIsa(StringRef isa);

  /// ROCDL target attribute.
  ROCDLTargetAttr target;

  /// ROCM toolkit path.
  std::string toolkitPath;

  /// List of LLVM bitcode files to link to.
  SmallVector<Attribute> librariesToLink;

  /// AMD GCN libraries to use when linking, the default is using none.
  AMDGCNLibraries deviceLibs = AMDGCNLibraries::None;
};

/// Returns a map containing the `amdhsa.kernels` ELF metadata for each of the
/// kernels in the binary, or `std::nullopt` if the metadata couldn't be
/// retrieved. The map associates the name of the kernel with the list of named
/// attributes found in `amdhsa.kernels`. For more information on the ELF
/// metadata see: https://llvm.org/docs/AMDGPUUsage.html#amdhsa
std::optional<DenseMap<StringAttr, NamedAttrList>>
getAMDHSAKernelsELFMetadata(Builder &builder, ArrayRef<char> elfData);

/// Returns a `#gpu.kernel_table` containing kernel metadata for each of the
/// kernels in `gpuModule`. If `elfData` is valid, then the `amdhsa.kernels` ELF
/// metadata will be added to the `#gpu.kernel_table`.
gpu::KernelTableAttr getKernelMetadata(Operation *gpuModule,
                                       ArrayRef<char> elfData = {});
} // namespace ROCDL
} // namespace mlir

#endif // MLIR_TARGET_LLVM_ROCDL_UTILS_H
