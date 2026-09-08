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
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

Error runCodeGenPipeline(
    TargetMachine &TM, Module &M, raw_pwrite_stream &OS,
    std::unique_ptr<ToolOutputFile> &DwoOS, CodeGenFileType CGFT,
    bool PrintPipelinePasses = false, bool DisableVerify = true,
    IntrusiveRefCntPtr<vfs::FileSystem> VFS = vfs::getRealFileSystem());

} // namespace llvm

#endif // LLVM_PASSES_RUN_CODEGEN_H
