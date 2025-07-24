//===----- OffloadWrapper.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_SYCL_OFFLOAD_WRAPPER_H
#define LLVM_FRONTEND_SYCL_OFFLOAD_WRAPPER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/OffloadBinary.h"

#include <string>

namespace llvm {

class Module;

namespace offloading {
namespace sycl {

struct SYCLWrappingOptions {
  // target/compiler specific options what are suggested to use to "compile"
  // program at runtime.
  std::string CompileOptions;
  // Target/Compiler specific options that are suggested to use to "link"
  // program at runtime.
  std::string LinkOptions;
};

/// Wraps OffloadBinaries in the given \p Buffers into the module \p M
/// as global symbols and registers the images with the SYCL Runtime.
/// \param Options Settings that allows to turn on optional data and settings.
llvm::Error
wrapSYCLBinaries(llvm::Module &M, llvm::ArrayRef<llvm::ArrayRef<char>> Buffers,
                 SYCLWrappingOptions Options = SYCLWrappingOptions());

} // namespace sycl
} // namespace offloading
} // namespace llvm

#endif // LLVM_FRONTEND_SYCL_OFFLOAD_WRAPPER_H
