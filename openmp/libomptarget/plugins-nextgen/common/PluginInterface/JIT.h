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

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <memory>
#include <string>

struct __tgt_device_image;

namespace llvm {
class MemoryBuffer;

namespace omp {
namespace jit {

/// Function type for a callback that will be called after the backend is
/// called.
using PostProcessingFn = std::function<Expected<std::unique_ptr<MemoryBuffer>>(
    std::unique_ptr<MemoryBuffer>)>;

/// Check if \p Image contains bitcode with triple \p Triple.
bool checkBitcodeImage(__tgt_device_image *Image, Triple::ArchType TA);

/// Compile the bitcode image \p Image and generate the binary image that can be
/// loaded to the target device of the triple \p Triple architecture \p MCpu. \p
/// PostProcessing will be called after codegen to handle cases such as assember
/// as an external tool.
Expected<__tgt_device_image *> compile(__tgt_device_image *Image,
                                       Triple::ArchType TA, std::string MCpu,
                                       unsigned OptLevel,
                                       PostProcessingFn PostProcessing);
} // namespace jit
} // namespace omp
} // namespace llvm

#endif // OPENMP_LIBOMPTARGET_PLUGINS_NEXTGEN_COMMON_JIT_H
