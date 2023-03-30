//===- TestAvailability.cpp - Pass to test SPIR-V op availability ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Printing op availability pass
//===----------------------------------------------------------------------===//

namespace {
/// A pass for testing SPIR-V op availability.
struct PrintOpAvailability
    : public PassWrapper<PrintOpAvailability, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintOpAvailability)

  void runOnOperation() override;
  StringRef getArgument() const final { return "test-spirv-op-availability"; }
  StringRef getDescription() const final {
    return "Test SPIR-V op availability";
  }
};
} // namespace

void PrintOpAvailability::runOnOperation() {
  auto f = getOperation();
  llvm::outs() << f.getName() << "\n";

  Dialect *spirvDialect = getContext().getLoadedDialect("spirv");

  f->walk([&](Operation *op) {
    if (op->getDialect() != spirvDialect)
      return WalkResult::advance();

    auto opName = op->getName();
    auto &os = llvm::outs();

    if (auto minVersionIfx = dyn_cast<spirv::QueryMinVersionInterface>(op)) {
      std::optional<spirv::Version> minVersion = minVersionIfx.getMinVersion();
      os << opName << " min version: ";
      if (minVersion)
        os << spirv::stringifyVersion(*minVersion) << "\n";
      else
        os << "None\n";
    }

    if (auto maxVersionIfx = dyn_cast<spirv::QueryMaxVersionInterface>(op)) {
      std::optional<spirv::Version> maxVersion = maxVersionIfx.getMaxVersion();
      os << opName << " max version: ";
      if (maxVersion)
        os << spirv::stringifyVersion(*maxVersion) << "\n";
      else
        os << "None\n";
    }

    if (auto extension = dyn_cast<spirv::QueryExtensionInterface>(op)) {
      os << opName << " extensions: [";
      for (const auto &exts : extension.getExtensions()) {
        os << " [";
        llvm::interleaveComma(exts, os, [&](spirv::Extension ext) {
          os << spirv::stringifyExtension(ext);
        });
        os << "]";
      }
      os << " ]\n";
    }

    if (auto capability = dyn_cast<spirv::QueryCapabilityInterface>(op)) {
      os << opName << " capabilities: [";
      for (const auto &caps : capability.getCapabilities()) {
        os << " [";
        llvm::interleaveComma(caps, os, [&](spirv::Capability cap) {
          os << spirv::stringifyCapability(cap);
        });
        os << "]";
      }
      os << " ]\n";
    }
    os.flush();

    return WalkResult::advance();
  });
}

namespace mlir {
void registerPrintSpirvAvailabilityPass() {
  PassRegistration<PrintOpAvailability>();
}
} // namespace mlir

//===----------------------------------------------------------------------===//
// Converting target environment pass
//===----------------------------------------------------------------------===//

namespace {
/// A pass for testing SPIR-V op availability.
struct ConvertToTargetEnv
    : public PassWrapper<ConvertToTargetEnv, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertToTargetEnv)

  StringRef getArgument() const override { return "test-spirv-target-env"; }
  StringRef getDescription() const override {
    return "Test SPIR-V target environment";
  }
  void runOnOperation() override;
};

struct ConvertToAtomCmpExchangeWeak : RewritePattern {
  ConvertToAtomCmpExchangeWeak(MLIRContext *context)
      : RewritePattern("test.convert_to_atomic_compare_exchange_weak_op", 1,
                       context, {"spirv.AtomicCompareExchangeWeak"}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value ptr = op->getOperand(0);
    Value value = op->getOperand(1);
    Value comparator = op->getOperand(2);

    // Create a spirv.AtomicCompareExchangeWeak op with AtomicCounterMemory bits
    // in memory semantics to additionally require AtomicStorage capability.
    rewriter.replaceOpWithNewOp<spirv::AtomicCompareExchangeWeakOp>(
        op, value.getType(), ptr, spirv::Scope::Workgroup,
        spirv::MemorySemantics::AcquireRelease |
            spirv::MemorySemantics::AtomicCounterMemory,
        spirv::MemorySemantics::Acquire, value, comparator);
    return success();
  }
};

struct ConvertToBitReverse : RewritePattern {
  ConvertToBitReverse(MLIRContext *context)
      : RewritePattern("test.convert_to_bit_reverse_op", 1, context,
                       {"spirv.BitReverse"}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value predicate = op->getOperand(0);
    rewriter.replaceOpWithNewOp<spirv::BitReverseOp>(
        op, op->getResult(0).getType(), predicate);
    return success();
  }
};

struct ConvertToGroupNonUniformBallot : RewritePattern {
  ConvertToGroupNonUniformBallot(MLIRContext *context)
      : RewritePattern("test.convert_to_group_non_uniform_ballot_op", 1,
                       context, {"spirv.GroupNonUniformBallot"}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value predicate = op->getOperand(0);
    rewriter.replaceOpWithNewOp<spirv::GroupNonUniformBallotOp>(
        op, op->getResult(0).getType(), spirv::Scope::Workgroup, predicate);
    return success();
  }
};

struct ConvertToModule : RewritePattern {
  ConvertToModule(MLIRContext *context)
      : RewritePattern("test.convert_to_module_op", 1, context,
                       {"spirv.module"}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<spirv::ModuleOp>(
        op, spirv::AddressingModel::PhysicalStorageBuffer64,
        spirv::MemoryModel::Vulkan);
    return success();
  }
};

struct ConvertToSubgroupBallot : RewritePattern {
  ConvertToSubgroupBallot(MLIRContext *context)
      : RewritePattern("test.convert_to_subgroup_ballot_op", 1, context,
                       {"spirv.KHR.SubgroupBallot"}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    Value predicate = op->getOperand(0);
    rewriter.replaceOpWithNewOp<spirv::KHRSubgroupBallotOp>(
        op, op->getResult(0).getType(), predicate);
    return success();
  }
};

template <const char *TestOpName, typename SPIRVOp>
struct ConvertToIntegerDotProd : RewritePattern {
  ConvertToIntegerDotProd(MLIRContext *context)
      : RewritePattern(TestOpName, 1, context, {SPIRVOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<SPIRVOp>(op, op->getResultTypes(),
                                         op->getOperands(), op->getAttrs());
    return success();
  }
};
} // namespace

void ConvertToTargetEnv::runOnOperation() {
  MLIRContext *context = &getContext();
  func::FuncOp fn = getOperation();

  auto targetEnv = dyn_cast_or_null<spirv::TargetEnvAttr>(
      fn.getOperation()->getAttr(spirv::getTargetEnvAttrName()));
  if (!targetEnv) {
    fn.emitError("missing 'spirv.target_env' attribute");
    return signalPassFailure();
  }

  auto target = SPIRVConversionTarget::get(targetEnv);

  static constexpr char sDotTestOpName[] = "test.convert_to_sdot_op";
  static constexpr char suDotTestOpName[] = "test.convert_to_sudot_op";
  static constexpr char uDotTestOpName[] = "test.convert_to_udot_op";
  static constexpr char sDotAccSatTestOpName[] =
      "test.convert_to_sdot_acc_sat_op";
  static constexpr char suDotAccSatTestOpName[] =
      "test.convert_to_sudot_acc_sat_op";
  static constexpr char uDotAccSatTestOpName[] =
      "test.convert_to_udot_acc_sat_op";

  RewritePatternSet patterns(context);
  patterns.add<
      ConvertToAtomCmpExchangeWeak, ConvertToBitReverse,
      ConvertToGroupNonUniformBallot, ConvertToModule, ConvertToSubgroupBallot,
      ConvertToIntegerDotProd<sDotTestOpName, spirv::SDotOp>,
      ConvertToIntegerDotProd<suDotTestOpName, spirv::SUDotOp>,
      ConvertToIntegerDotProd<uDotTestOpName, spirv::UDotOp>,
      ConvertToIntegerDotProd<sDotAccSatTestOpName, spirv::SDotAccSatOp>,
      ConvertToIntegerDotProd<suDotAccSatTestOpName, spirv::SUDotAccSatOp>,
      ConvertToIntegerDotProd<uDotAccSatTestOpName, spirv::UDotAccSatOp>>(
      context);

  if (failed(applyPartialConversion(fn, *target, std::move(patterns))))
    return signalPassFailure();
}

namespace mlir {
void registerConvertToTargetEnvPass() {
  PassRegistration<ConvertToTargetEnv>();
}
} // namespace mlir
