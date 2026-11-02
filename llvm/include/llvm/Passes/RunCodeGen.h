//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_RUN_CODEGEN_H
#define LLVM_PASSES_RUN_CODEGEN_H

#include "llvm/IR/Module.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

LLVM_ABI Error runCodeGenPipeline(
    TargetMachine &TM, Module &M, raw_pwrite_stream &OS,
    raw_pwrite_stream *DwoOS, CodeGenFileType CGFT, bool DisableVerify = true,
    bool DisableSimplifyLibCalls = false,
    IntrusiveRefCntPtr<vfs::FileSystem> VFS = vfs::getRealFileSystem());

} // namespace llvm

#endif // LLVM_PASSES_RUN_CODEGEN_H
