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
  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  builder.setInsertionPointToEnd(module.getBody());

  Location loc = UnknownLoc::get(&ctx);

  static constexpr StringLiteral kMainSequenceName = "__transform_main";

  NamedSequenceOp seqOp = builder.create<NamedSequenceOp>(
      loc,
      /*sym_name=*/kMainSequenceName,
      /*rootType=*/builder.getType<AnyOpType>(),
      /*resultType=*/TypeRange{},
      [](OpBuilder &b, Location nested, Value rootH) {
        b.create<YieldOp>(nested, ValueRange());
      },
      /*args=*/ArrayRef<NamedAttribute>{},
      /*attrArgs=*/
      ArrayRef<DictionaryAttr>{
          builder.getDictionaryAttr(ArrayRef<NamedAttribute>{
              builder.getNamedAttr(TransformDialect::kArgConsumedAttrName,
                                   builder.getUnitAttr())})});

  // 检查 body argument 上有没有 transform.consumed
  Block &body = seqOp.getBody().front();
  ASSERT_EQ(body.getNumArguments(), 1u);

  StringAttr arg0Name = seqOp.getArgAttrsAttrName();
  EXPECT_TRUE(arg0Name);
}