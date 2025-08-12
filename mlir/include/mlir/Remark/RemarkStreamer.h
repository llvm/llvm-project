#include "mlir/IR/Remarks.h"

#include "llvm/Remarks/RemarkStreamer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
namespace mlir::remark {

/// Concrete streamer that writes LLVM optimization remarks to a file
/// (YAML or Bitstream). Lives outside core.
class LLVMRemarkStreamer final : public MLIRRemarkStreamerBase {
public:
  static FailureOr<std::unique_ptr<MLIRRemarkStreamerBase>>
  createToFile(llvm::StringRef path, llvm::remarks::Format fmt);

  void streamOptimizationRemark(const Remark &remark) override;
  void finalize() override {}
  ~LLVMRemarkStreamer() override;

private:
  LLVMRemarkStreamer() = default;

  std::unique_ptr<class llvm::remarks::RemarkStreamer> remarkStreamer;
  std::unique_ptr<class llvm::ToolOutputFile> file;
};

/// Enable optimization remarks to a file with the given path and format.
/// The remark categories are used to filter the remarks that are emitted.
/// If the printAsEmitRemarks flag is set, remarks will also be printed using
/// mlir::emitRemarks.
LogicalResult enableOptimizationRemarksToFile(
    MLIRContext &ctx, StringRef path, llvm::remarks::Format fmt,
    const MLIRContext::RemarkCategories &cat, bool printAsEmitRemarks = false);

} // namespace mlir::remark
