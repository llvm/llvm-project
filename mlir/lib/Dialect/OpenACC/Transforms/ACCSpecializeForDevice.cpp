//===- ACCSpecializeForDevice.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass strips OpenACC constructs that are invalid or unnecessary inside
// device code (specialized acc routines or compute construct regions).
//
// Overview:
// ---------
// In a specialized acc routine or compute construct, many OpenACC operations
// do not make sense because they are host-side constructs. This pass removes
// or transforms these operations appropriately:
//
// - Data operations that manage device memory from host perspective
// - Compute constructs that launch kernels (we're already on device)
// - Runtime operations like init/shutdown/set/wait
//
// Transformations:
// ----------------
// The pass applies the following transformations:
//
// 1. Data Entry Ops (replaced with var operand):
//    acc.attach, acc.copyin, acc.create, acc.declare_device_resident,
//    acc.declare_link, acc.deviceptr, acc.get_deviceptr, acc.nocreate,
//    acc.present, acc.update_device, acc.use_device
//
// 2. Data Exit Ops (erased):
//    acc.copyout, acc.delete, acc.detach, acc.update_host
//
// 3. Structured Data/Compute Constructs (region inlined):
//    acc.data, acc.host_data, acc.kernel_environment, acc.parallel,
//    acc.serial, acc.kernels
//
// 4. Unstructured Data Ops (erased):
//    acc.enter_data, acc.exit_data, acc.update, acc.declare_enter,
//    acc.declare_exit
//
// 5. Runtime Ops (erased):
//    acc.init, acc.shutdown, acc.set, acc.wait
//
// Scope of Application:
// ---------------------
// - For functions with `acc.specialized_routine` attribute: patterns are
//   applied to the entire function body.
// - For non-specialized functions: patterns are applied only to ACC
//   operations INSIDE compute constructs (parallel, serial, kernels),
//   not to the compute constructs themselves or their data operands.
//
// Note: acc.cache, acc.private, acc.reduction, acc.firstprivate are NOT
// transformed by this pass as they are valid in device code.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/Transforms/ACCSpecializePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCSPECIALIZEFORDEVICE
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

using namespace mlir;
using namespace mlir::acc;

namespace {

class ACCSpecializeForDevice
    : public acc::impl::ACCSpecializeForDeviceBase<ACCSpecializeForDevice> {
public:
  using ACCSpecializeForDeviceBase<
      ACCSpecializeForDevice>::ACCSpecializeForDeviceBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    RewritePatternSet patterns(&getContext());
    acc::populateACCSpecializeForDevicePatterns(patterns);
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);

    if (acc::isSpecializedAccRoutine(func)) {
      // For specialized acc routines, apply patterns to the entire function
      (void)applyPatternsGreedily(func, std::move(patterns), config);
    } else {
      // For non-specialized functions, apply patterns only to ACC operations
      // inside compute constructs (not to the compute constructs themselves).
      SmallVector<Operation *> opsToTransform;
      func.walk([&](Operation *op) {
        if (isa<ACC_COMPUTE_CONSTRUCT_OPS>(op)) {
          // Walk inside the compute construct and collect ACC ops
          op->walk([&](Operation *innerOp) {
            // Skip the compute construct itself
            if (innerOp == op) {
              return;
            }
            if (isa<acc::OpenACCDialect>(innerOp->getDialect())) {
              opsToTransform.push_back(innerOp);
            }
          });
        }
      });
      if (!opsToTransform.empty()) {
        (void)applyOpPatternsGreedily(opsToTransform, std::move(patterns),
                                      config);
      }
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population functions
//===----------------------------------------------------------------------===//

void mlir::acc::populateACCSpecializeForDevicePatterns(
    RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();

  // Declare patterns - erase declare_enter and its associated declare_exit
  patterns.insert<ACCDeclareEnterOpConversion>(context);

  // Data entry ops - replaced with their var operand
  // Note: acc.cache, acc.private, acc.reduction, acc.firstprivate are NOT
  // included here - they are valid in device code
  patterns.insert<ACCOpReplaceWithVarConversion<acc::AttachOp>,
                  ACCOpReplaceWithVarConversion<acc::CopyinOp>,
                  ACCOpReplaceWithVarConversion<acc::CreateOp>,
                  ACCOpReplaceWithVarConversion<acc::DeclareDeviceResidentOp>,
                  ACCOpReplaceWithVarConversion<acc::DeclareLinkOp>,
                  ACCOpReplaceWithVarConversion<acc::DevicePtrOp>,
                  ACCOpReplaceWithVarConversion<acc::GetDevicePtrOp>,
                  ACCOpReplaceWithVarConversion<acc::NoCreateOp>,
                  ACCOpReplaceWithVarConversion<acc::PresentOp>,
                  ACCOpReplaceWithVarConversion<acc::UpdateDeviceOp>,
                  ACCOpReplaceWithVarConversion<acc::UseDeviceOp>>(context);

  // Data exit ops - simply erased (no results)
  patterns.insert<ACCOpEraseConversion<acc::CopyoutOp>,
                  ACCOpEraseConversion<acc::DeleteOp>,
                  ACCOpEraseConversion<acc::DetachOp>,
                  ACCOpEraseConversion<acc::UpdateHostOp>>(context);

  // Structured data constructs - unwrap their regions
  patterns.insert<ACCRegionUnwrapConversion<acc::DataOp>,
                  ACCRegionUnwrapConversion<acc::HostDataOp>,
                  ACCRegionUnwrapConversion<acc::KernelEnvironmentOp>>(context);

  // Compute constructs - unwrap their regions
  patterns.insert<ACCRegionUnwrapConversion<acc::ParallelOp>,
                  ACCRegionUnwrapConversion<acc::SerialOp>,
                  ACCRegionUnwrapConversion<acc::KernelsOp>>(context);

  // Unstructured data operations - erase them
  patterns.insert<ACCOpEraseConversion<acc::EnterDataOp>,
                  ACCOpEraseConversion<acc::ExitDataOp>,
                  ACCOpEraseConversion<acc::UpdateOp>>(context);

  // Runtime operations - erase them
  patterns.insert<ACCOpEraseConversion<acc::InitOp>,
                  ACCOpEraseConversion<acc::ShutdownOp>,
                  ACCOpEraseConversion<acc::SetOp>,
                  ACCOpEraseConversion<acc::WaitOp>>(context);
}

