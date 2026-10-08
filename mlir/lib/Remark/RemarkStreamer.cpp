#include "mlir/Remark/RemarkStreamer.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Remarks.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Remarks/RemarkSerializer.h"
#include "llvm/Remarks/RemarkStreamer.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir::remark::detail {
FailureOr<std::unique_ptr<MLIRRemarkStreamerBase>>
LLVMRemarkStreamer::createToFile(llvm::StringRef path,
                                 llvm::remarks::Format fmt,
                                 std::string *errorMessage) {
  std::unique_ptr<llvm::ToolOutputFile> f = openOutputFile(path, errorMessage);
  if (!f)
    return failure();

  auto serOr = llvm::remarks::createRemarkSerializer(fmt, f->os());
  if (!serOr) {
    std::string reason = llvm::toString(serOr.takeError());
    if (errorMessage)
      *errorMessage =
          "cannot create remark serializer for '" + path.str() + "': " + reason;
    return failure();
  }

  auto rs =
      std::make_unique<llvm::remarks::RemarkStreamer>(std::move(*serOr), path);

  auto impl = std::unique_ptr<LLVMRemarkStreamer>(new LLVMRemarkStreamer());
  impl->remarkStreamer = std::move(rs);
  impl->file = std::move(f);
  return std::unique_ptr<MLIRRemarkStreamerBase>(std::move(impl));
}

void LLVMRemarkStreamer::streamOptimizationRemark(const Remark &remark) {
  if (!remarkStreamer->matchesFilter(remark.getCategoryName()))
    return;

  // First, convert the diagnostic to a remark.
  llvm::remarks::Remark r = remark.generateRemark();
  // Then, emit the remark through the serializer.
  remarkStreamer->getSerializer().emit(r);
}

LLVMRemarkStreamer::~LLVMRemarkStreamer() {
  if (file && remarkStreamer)
    file->keep();
}

void LLVMRemarkStreamer::finalize() {
  if (!remarkStreamer)
    return;
  remarkStreamer->releaseSerializer();
}
} // namespace mlir::remark::detail

namespace mlir::remark {
LogicalResult enableOptimizationRemarksWithLLVMStreamer(
    MLIRContext &ctx, StringRef path, llvm::remarks::Format fmt,
    std::unique_ptr<detail::RemarkEmittingPolicyBase> remarkEmittingPolicy,
    const RemarkCategories &cat, bool printAsEmitRemarks) {

  std::string errorMessage;
  FailureOr<std::unique_ptr<detail::MLIRRemarkStreamerBase>> sOr =
      detail::LLVMRemarkStreamer::createToFile(path, fmt, &errorMessage);
  if (failed(sOr))
    return emitError(UnknownLoc::get(&ctx)) << errorMessage;

  return remark::enableOptimizationRemarks(ctx, std::move(*sOr),
                                           std::move(remarkEmittingPolicy), cat,
                                           printAsEmitRemarks);
}

} // namespace mlir::remark
