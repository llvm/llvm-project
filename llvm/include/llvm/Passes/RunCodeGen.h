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

class ModuleSummaryIndex;

/// Options for runCodeGenPipeline().
struct CodeGenPipelineConfig {
  /// FIXME: Need to be implemented in runCodeGenPipeline().
  bool PrintPipelinePasses = false;

  bool DisableVerify = true;
  bool DisableSimplifyLibCalls = false;
  /// Log each pass as it runs. New pass manager only.
  bool DebugPassManager = false;
  /// Verify the IR between passes. New pass manager only.
  bool VerifyEach = false;
  /// LTO uses this to expose the combined summary index to summary-consuming
  /// codegen passes.
  const ModuleSummaryIndex *SummaryIndex = nullptr;
  /// For passes needing file access.
  IntrusiveRefCntPtr<vfs::FileSystem> VFS = nullptr;
};

Error runCodeGenPipeline(TargetMachine &TM, Module &M, raw_pwrite_stream &OS,
                         std::unique_ptr<ToolOutputFile> &DwoOS,
                         CodeGenFileType CGFT,
                         const CodeGenPipelineConfig &Config = {});

} // namespace llvm

#endif // LLVM_PASSES_RUN_CODEGEN_H
