//===- RemarkStreamer.h - MLIR Optimization Remark ---------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines LLVMRemarkStreamer plugging class that uses LLVM's
// streamer.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Remarks.h"

#include "llvm/Remarks/RemarkStreamer.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir::remark::detail {

/// Concrete streamer that writes LLVM optimization remarks to a file
/// (YAML or Bitstream). Lives outside core.
class LLVMRemarkStreamer final : public MLIRRemarkStreamerBase {
public:
  static FailureOr<std::unique_ptr<MLIRRemarkStreamerBase>>
  createToFile(llvm::StringRef path, llvm::remarks::Format fmt);

  void streamOptimizationRemark(const Remark &remark) override;
  void finalize() override;
  ~LLVMRemarkStreamer() override;

private:
  LLVMRemarkStreamer() = default;

  std::unique_ptr<class llvm::ToolOutputFile> file;
  // RemarkStreamer must be destructed before file is destroyed!
  std::unique_ptr<class llvm::remarks::RemarkStreamer> remarkStreamer;
};
} // namespace mlir::remark::detail

namespace mlir::remark {
/// Enable optimization remarks to a file with the given path and format.
/// The remark categories are used to filter the remarks that are emitted.
/// If the printAsEmitRemarks flag is set, remarks will also be printed using
/// mlir::emitRemarks.
LogicalResult enableOptimizationRemarksWithLLVMStreamer(
    MLIRContext &ctx, StringRef filePath, llvm::remarks::Format fmt,
    std::unique_ptr<detail::RemarkEmittingPolicyBase> remarkEmittingPolicy,
    const RemarkCategories &cat, bool printAsEmitRemarks = false);

} // namespace mlir::remark
