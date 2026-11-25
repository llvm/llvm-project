// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "CVNSS/CVNSS.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
using namespace mlir;
using namespace mlir::cvnss;

#include "CVNSS/CVNSSOpsDialect.cpp.inc"
#define GET_OP_CLASSES
#include "CVNSS/CVNSSOps.cpp.inc"

void CVNSSDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "CVNSS/CVNSSOps.cpp.inc"
  >();
}
LogicalResult Pack27Op::verify() {
  for (auto a : {getHiAttr(), getMidAttr(), getLoAttr()})
    if (a.getInt() < -1 || a.getInt() > 1)
      return emitOpError("trits must be -1, 0 or 1");
  return success();
}
LogicalResult Unpack27Op::verify() {
  if (getIdAttr().getInt() < 0 || getIdAttr().getInt() > 26)
    return emitOpError("ID must be in [0, 26]");
  return success();
}
LogicalResult MappingOp::verify() {
  auto mode = getModeAttr().getValue();
  if (mode != "CQN" && mode != "CVN" && mode != "KHD")
    return emitOpError("mode must be CQN, CVN or KHD");
  if (getMappingVersionAttr().getValue().empty())
    return emitOpError("mapping_version must not be empty");
  auto policy = getLossPolicyAttr().getValue();
  if (policy != "report" && policy != "preserve_candidates")
    return emitOpError("loss_policy must report or preserve_candidates");
  return success();
}

LogicalResult ReferenceOp::verify() {
  auto lang = getLanguageAttr().getValue();
  if (lang != "bila" && lang != "c" && lang != "cpp" &&
      lang != "javascript" && lang != "rust" && lang != "python")
    return emitOpError("unsupported source language");
  if (getSchemaVersionAttr().getValue() != "0.2")
    return emitOpError("unsupported semantic schema version");
  auto digest = getSourceDigestAttr().getValue();
  if (digest.size() != 64)
    return emitOpError("source_digest must be lowercase SHA-256 hex");
  for (char c : digest)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
      return emitOpError("source_digest must be lowercase SHA-256 hex");
  auto ref = getReferenceIdAttr().getValue();
  if (ref.size() != 70 || ref.take_front(6) != "cvnss:")
    return emitOpError("reference_id must be cvnss:<64 hex>");
  for (char c : ref.drop_front(6))
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
      return emitOpError("reference_id must be cvnss:<64 hex>");
  if (getSymbolAttr().getValue().empty() ||
      getSourceUriAttr().getValue().empty())
    return emitOpError("symbol and source_uri must not be empty");
  auto kind = getSemanticKindAttr().getValue();
  if (kind != "function" && kind != "type" && kind != "value" &&
      kind != "external")
    return emitOpError("unsupported semantic kind");
  return success();
}

namespace {
struct PackLowering : OpConversionPattern<Pack27Op> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(Pack27Op op, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto id = 13 + 9 * op.getHiAttr().getInt() +
              3 * op.getMidAttr().getInt() + op.getLoAttr().getInt();
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, id, 32);
    return success();
  }
};
struct UnpackLowering : OpConversionPattern<Unpack27Op> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(Unpack27Op op, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto id = op.getIdAttr().getInt();
    SmallVector<Value> values;
    for (auto trit : {id / 9 - 1, (id / 3) % 3 - 1, id % 3 - 1})
      values.push_back(rewriter.create<arith::ConstantIntOp>(op.getLoc(), trit, 32));
    rewriter.replaceOp(op, values);
    return success();
  }
};
struct LowerTritPass : PassWrapper<LowerTritPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTritPass)
  StringRef getArgument() const final { return "cvnss-lower-trit"; }
  StringRef getDescription() const final { return "Lower verified Base27 constants to arith"; }
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect>();
  }
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalOp<Pack27Op, Unpack27Op>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    RewritePatternSet patterns(&getContext());
    patterns.add<PackLowering, UnpackLowering>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
}
void mlir::cvnss::registerCVNSSPasses() { PassRegistration<LowerTritPass>(); }
