//===- OffloadWrapper.h --r-------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_OFFLOADING_OFFLOADWRAPPER_H
#define LLVM_FRONTEND_OFFLOADING_OFFLOADWRAPPER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Compiler.h"

#include <string>

namespace llvm {
namespace offloading {
using EntryArrayTy = std::pair<GlobalVariable *, GlobalVariable *>;
/// Wraps the input device images into the module \p M as global symbols and
/// registers the images with the OpenMP Offloading runtime libomptarget.
/// \param EntryArray Optional pair pointing to the `__start` and `__stop`
/// symbols holding the `__tgt_offload_entry` array.
/// \param Suffix An optional suffix appended to the emitted symbols.
/// \param Relocatable Indicate if we need to change the offloading section to
/// create a relocatable object.
LLVM_ABI llvm::Error
wrapOpenMPBinaries(llvm::Module &M, llvm::ArrayRef<llvm::ArrayRef<char>> Images,
                   EntryArrayTy EntryArray, llvm::StringRef Suffix = "",
                   bool Relocatable = false);

/// Wraps the input fatbinary image into the module \p M as global symbols and
/// registers the images with the CUDA runtime.
/// \param EntryArray Optional pair pointing to the `__start` and `__stop`
/// symbols holding the `__tgt_offload_entry` array.
/// \param Suffix An optional suffix appended to the emitted symbols.
/// \param EmitSurfacesAndTextures Whether to emit surface and textures
/// registration code. It defaults to false.
LLVM_ABI llvm::Error wrapCudaBinary(llvm::Module &M,
                                    llvm::ArrayRef<char> Images,
                                    EntryArrayTy EntryArray,
                                    llvm::StringRef Suffix = "",
                                    bool EmitSurfacesAndTextures = true);

/// Wraps the input bundled image into the module \p M as global symbols and
/// registers the images with the HIP runtime.
/// \param EntryArray Optional pair pointing to the `__start` and `__stop`
/// symbols holding the `__tgt_offload_entry` array.
/// \param Suffix An optional suffix appended to the emitted symbols.
/// \param EmitSurfacesAndTextures Whether to emit surface and textures
/// registration code. It defaults to false.
LLVM_ABI llvm::Error wrapHIPBinary(llvm::Module &M, llvm::ArrayRef<char> Images,
                                   EntryArrayTy EntryArray,
                                   llvm::StringRef Suffix = "",
                                   bool EmitSurfacesAndTextures = true);

struct SYCLJITOptions {
  // Target/compiler specific options that are suggested to use to "compile"
  // program at runtime.
  std::string CompileOptions;
  // Target/compiler specific options that are suggested to use to "link"
  // program at runtime.
  std::string LinkOptions;
};

/// Wraps OffloadBinaries in the given \p Buffers into the module \p M
/// as global symbols and registers the images with the SYCL Runtime.
/// \param Options Compiler and linker options to be encoded for the later
///  use by a runtime for JIT compilation.
LLVM_ABI llvm::Error
wrapSYCLBinaries(llvm::Module &M, llvm::ArrayRef<char> Buffer,
                 SYCLJITOptions Options = SYCLJITOptions());

} // namespace offloading
} // namespace llvm

#endif // LLVM_FRONTEND_OFFLOADING_OFFLOADWRAPPER_H
