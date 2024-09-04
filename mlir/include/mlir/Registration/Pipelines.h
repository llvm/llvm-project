#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include <cstdint>

namespace mlir {
    struct TosaFuserPipelineOptions : public PassPipelineOptions<TosaFuserPipelineOptions> {

    };
    void createTosaFuserPipeline(OpPassManager &pm, const TosaFuserPipelineOptions &options, unsigned optLevel);
    void registerTosaFuserPipeline();
}