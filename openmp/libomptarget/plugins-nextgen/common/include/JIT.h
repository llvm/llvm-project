//===- JIT.h - Target independent JIT infrastructure ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_JIT_H
#define OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_JIT_H

#include "Shared/EnvironmentVar.h"
#include "Shared/Utils.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>

struct __tgt_device_image;

namespace llvm {
class MemoryBuffer;

namespace omp {
namespace target {
namespace plugin {
struct GenericDeviceTy;
} // namespace plugin

/// The JIT infrastructure and caching mechanism.
struct JITEngine {
  /// Function type for a callback that will be called after the backend is
  /// called.
  using PostProcessingFn =
      std::function<Expected<std::unique_ptr<MemoryBuffer>>(
          std::unique_ptr<MemoryBuffer>)>;

  JITEngine(Triple::ArchType TA);

  /// Run jit compilation if \p Image is a bitcode image, otherwise simply
  /// return \p Image. It is expected to return a memory buffer containing the
  /// generated device image that could be loaded to the device directly.
  Expected<const __tgt_device_image *>
  process(const __tgt_device_image &Image,
          target::plugin::GenericDeviceTy &Device);

  /// Return true if \p Image is a bitcode image that can be JITed for the given
  /// architecture.
  bool checkBitcodeImage(const __tgt_device_image &Image);

private:
  /// Compile the bitcode image \p Image and generate the binary image that can
  /// be loaded to the target device of the triple \p Triple architecture \p
  /// MCpu. \p PostProcessing will be called after codegen to handle cases such
  /// as assember as an external tool.
  Expected<const __tgt_device_image *>
  compile(const __tgt_device_image &Image, const std::string &ComputeUnitKind,
          PostProcessingFn PostProcessing);

  /// Create or retrieve the object image file from the file system or via
  /// compilation of the \p Image.
  Expected<std::unique_ptr<MemoryBuffer>>
  getOrCreateObjFile(const __tgt_device_image &Image, LLVMContext &Ctx,
                     const std::string &ComputeUnitKind);

  /// Run backend, which contains optimization and code generation.
  Expected<std::unique_ptr<MemoryBuffer>>
  backend(Module &M, const std::string &ComputeUnitKind, unsigned OptLevel);

  /// Run optimization pipeline.
  void opt(TargetMachine *TM, TargetLibraryInfoImpl *TLII, Module &M,
           unsigned OptLevel);

  /// Run code generation.
  void codegen(TargetMachine *TM, TargetLibraryInfoImpl *TLII, Module &M,
               raw_pwrite_stream &OS);

  /// The target triple used by the JIT.
  const Triple TT;

  struct ComputeUnitInfo {
    /// LLVM Context in which the modules will be constructed.
    LLVMContext Context;

    /// Output images generated from LLVM backend.
    SmallVector<std::unique_ptr<MemoryBuffer>, 4> JITImages;

    /// A map of embedded IR images to JITed images.
    DenseMap<const __tgt_device_image *, __tgt_device_image *> TgtImageMap;
  };

  /// Map from (march) "CPUs" (e.g., sm_80, or gfx90a), which we call compute
  /// units as they are not CPUs, to the image information we cached for them.
  StringMap<ComputeUnitInfo> ComputeUnitMap;
  std::mutex ComputeUnitMapMutex;

  /// Control environment variables.
  StringEnvar ReplacementObjectFileName =
      StringEnvar("LIBOMPTARGET_JIT_REPLACEMENT_OBJECT");
  StringEnvar ReplacementModuleFileName =
      StringEnvar("LIBOMPTARGET_JIT_REPLACEMENT_MODULE");
  StringEnvar PreOptIRModuleFileName =
      StringEnvar("LIBOMPTARGET_JIT_PRE_OPT_IR_MODULE");
  StringEnvar PostOptIRModuleFileName =
      StringEnvar("LIBOMPTARGET_JIT_POST_OPT_IR_MODULE");
  UInt32Envar JITOptLevel = UInt32Envar("LIBOMPTARGET_JIT_OPT_LEVEL", 3);
  BoolEnvar JITSkipOpt = BoolEnvar("LIBOMPTARGET_JIT_SKIP_OPT", false);
};

} // namespace target
} // namespace omp
} // namespace llvm

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_JIT_H
