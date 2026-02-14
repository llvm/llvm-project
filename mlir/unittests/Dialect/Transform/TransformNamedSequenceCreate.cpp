#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::transform;

TEST(NamedSequenceOpTest, ArgAttrsAreHonoredByBuilder) {
  MLIRContext ctx;
  ctx.loadDialect<TransformDialect>();

  OpBuilder builder(&ctx);
  Location loc = UnknownLoc::get(&ctx);
  auto module = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(module.getBody());

  NamedSequenceOp seqOp = NamedSequenceOp::create(
      builder, loc,
      /*sym_name=*/transform::TransformDialect::kTransformEntryPointSymbolName,
      /*rootType=*/builder.getType<AnyOpType>(),
      /*resultType=*/TypeRange{},
      [](OpBuilder &b, Location nested, Value rootH) {
        YieldOp::create(b, nested, ValueRange());
      },
      /*args=*/ArrayRef<NamedAttribute>{},
      /*attrArgs=*/
      ArrayRef<DictionaryAttr>{
          builder.getDictionaryAttr(ArrayRef<NamedAttribute>{
              builder.getNamedAttr(TransformDialect::kArgConsumedAttrName,
                                   builder.getUnitAttr())})});

  // Check if body argument contains any attributes.
  Block &body = seqOp.getBody().front();
  ASSERT_EQ(body.getNumArguments(), 1u);

  auto arg0Attr = seqOp.getArgAttrDict(0);
  EXPECT_TRUE(arg0Attr);

  auto arg0Name = arg0Attr.getNamed(TransformDialect::kArgConsumedAttrName);
  EXPECT_TRUE(arg0Name.has_value());

  EXPECT_EQ(arg0Name.value().getName(), TransformDialect::kArgConsumedAttrName);

  auto expectedFalse =
      arg0Attr.getNamed(TransformDialect::kArgReadOnlyAttrName);
  EXPECT_FALSE(expectedFalse.has_value());
}
