//===------ ScopInliner.h ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef POLLY_POLLYINLINER_H
#define POLLY_POLLYINLINER_H

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
namespace vfs {
class FileSystem;
}
} // namespace llvm

namespace polly {
class ScopInlinerPass : public llvm::OptionalPassInfoMixin<ScopInlinerPass> {
  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;

public:
  explicit ScopInlinerPass(llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS);

  llvm::PreservedAnalyses run(llvm::LazyCallGraph::SCC &C,
                              llvm::CGSCCAnalysisManager &AM,
                              llvm::LazyCallGraph &CG,
                              llvm::CGSCCUpdateResult &UR);
};
} // namespace polly

#endif /* POLLY_POLLYINLINER_H */
