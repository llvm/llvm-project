//===- builder-api-test.cpp - Tests for Declarative Builder APIs ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-edsc-builder-api-test | FileCheck %s -dump-input-on-failure

#include "mlir/Dialect/AffineOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/LoopOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/VectorOps/EDSC/Intrinsics.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#include "APITest.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;

static MLIRContext &globalContext() {
  static bool init_once = []() {
    registerDialect<AffineOpsDialect>();
    registerDialect<linalg::LinalgDialect>();
    registerDialect<loop::LoopOpsDialect>();
    registerDialect<StandardOpsDialect>();
    registerDialect<vector::VectorOpsDialect>();
    return true;
  }();
  (void)init_once;
  static thread_local MLIRContext context;
  return context;
}

static FuncOp makeFunction(StringRef name, ArrayRef<Type> results = {},
                           ArrayRef<Type> args = {}) {
  auto &ctx = globalContext();
  auto function = FuncOp::create(UnknownLoc::get(&ctx), name,
                                 FunctionType::get(args, results, &ctx));
  function.addEntryBlock();
  return function;
}

TEST_FUNC(builder_dynamic_for_func_args) {
  auto indexType = IndexType::get(&globalContext());
  auto f32Type = FloatType::getF32(&globalContext());
  auto f =
      makeFunction("builder_dynamic_for_func_args", {}, {indexType, indexType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  ValueHandle i(indexType), j(indexType), lb(f.getArgument(0)),
      ub(f.getArgument(1));
  ValueHandle f7(std_constant_float(llvm::APFloat(7.0f), f32Type));
  ValueHandle f13(std_constant_float(llvm::APFloat(13.0f), f32Type));
  ValueHandle i7(std_constant_int(7, 32));
  ValueHandle i13(std_constant_int(13, 32));
  AffineLoopNestBuilder(&i, lb, ub, 3)([&] {
    using namespace edsc::op;
    lb *std_constant_index(3) + ub;
    lb + std_constant_index(3);
    AffineLoopNestBuilder(&j, lb, ub, 2)([&] {
      ceilDiv(std_constant_index(31) * floorDiv(i + j * std_constant_index(3),
                                                std_constant_index(32)),
              std_constant_index(32));
      ((f7 + f13) / f7) % f13 - f7 *f13;
      ((i7 + i13) / i7) % i13 - i7 *i13;
    });
  });

  // clang-format off
  // CHECK-LABEL: func @builder_dynamic_for_func_args(%{{.*}}: index, %{{.*}}: index) {
  //     CHECK:  affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%{{.*}}) to affine_map<(d0) -> (d0)>(%{{.*}}) step 3 {
  //     CHECK:  {{.*}} = affine.apply affine_map<()[s0] -> (s0 * 3)>()[%{{.*}}]
  //     CHECK:  {{.*}} = affine.apply affine_map<()[s0, s1] -> (s1 + s0 * 3)>()[%{{.*}}, %{{.*}}]
  //     CHECK:  {{.*}} = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%{{.*}}]
  //     CHECK:  affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%{{.*}}) to affine_map<(d0) -> (d0)>(%{{.*}}) step 2 {
  //     CHECK:    {{.*}} = affine.apply affine_map<(d0, d1) -> ((d0 + d1 * 3) floordiv 32)>(%{{.*}}, %{{.*}})
  //     CHECK:    {{.*}} = affine.apply affine_map<(d0, d1) -> (((d0 + d1 * 3) floordiv 32) * 31)>(%{{.*}}, %{{.*}})
  //     CHECK:    {{.*}} = affine.apply affine_map<(d0, d1) -> ((((d0 + d1 * 3) floordiv 32) * 31) ceildiv 32)>(%{{.*}}, %{{.*}})
  // CHECK-DAG:    [[rf1:%[0-9]+]] = addf {{.*}}, {{.*}} : f32
  // CHECK-DAG:    [[rf2:%[0-9]+]] = divf [[rf1]], {{.*}} : f32
  // CHECK-DAG:    [[rf3:%[0-9]+]] = remf [[rf2]], {{.*}} : f32
  // CHECK-DAG:    [[rf4:%[0-9]+]] = mulf {{.*}}, {{.*}} : f32
  //     CHECK:    {{.*}} = subf [[rf3]], [[rf4]] : f32
  // CHECK-DAG:    [[ri1:%[0-9]+]] = addi {{.*}}, {{.*}} : i32
  // CHECK-DAG:    [[ri2:%[0-9]+]] = divi_signed [[ri1]], {{.*}} : i32
  // CHECK-DAG:    [[ri3:%[0-9]+]] = remi_signed [[ri2]], {{.*}} : i32
  // CHECK-DAG:    [[ri4:%[0-9]+]] = muli {{.*}}, {{.*}} : i32
  //     CHECK:    {{.*}} = subi [[ri3]], [[ri4]] : i32
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

TEST_FUNC(builder_dynamic_for) {
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("builder_dynamic_for", {},
                        {indexType, indexType, indexType, indexType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  ValueHandle i(indexType), a(f.getArgument(0)), b(f.getArgument(1)),
      c(f.getArgument(2)), d(f.getArgument(3));
  using namespace edsc::op;
  AffineLoopNestBuilder(&i, a - b, c + d, 2)();

  // clang-format off
  // CHECK-LABEL: func @builder_dynamic_for(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
  // CHECK-DAG:    [[r0:%[0-9]+]] = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%{{.*}}, %{{.*}}]
  // CHECK-DAG:    [[r1:%[0-9]+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%{{.*}}, %{{.*}}]
  // CHECK-NEXT:   affine.for %{{.*}} = affine_map<(d0) -> (d0)>([[r0]]) to affine_map<(d0) -> (d0)>([[r1]]) step 2 {
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

TEST_FUNC(builder_loop_for) {
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("builder_loop_for", {},
                        {indexType, indexType, indexType, indexType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  ValueHandle i(indexType), a(f.getArgument(0)), b(f.getArgument(1)),
      c(f.getArgument(2)), d(f.getArgument(3));
  using namespace edsc::op;
  LoopNestBuilder(&i, a - b, c + d, a)();

  // clang-format off
  // CHECK-LABEL: func @builder_loop_for(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
  // CHECK-DAG:    [[r0:%[0-9]+]] = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%{{.*}}, %{{.*}}]
  // CHECK-DAG:    [[r1:%[0-9]+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%{{.*}}, %{{.*}}]
  // CHECK-NEXT:   loop.for %{{.*}} = [[r0]] to [[r1]] step {{.*}} {
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

TEST_FUNC(builder_max_min_for) {
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("builder_max_min_for", {},
                        {indexType, indexType, indexType, indexType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  ValueHandle i(indexType), lb1(f.getArgument(0)), lb2(f.getArgument(1)),
      ub1(f.getArgument(2)), ub2(f.getArgument(3));
  AffineLoopNestBuilder(&i, {lb1, lb2}, {ub1, ub2}, 1)();
  std_ret();

  // clang-format off
  // CHECK-LABEL: func @builder_max_min_for(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
  // CHECK:  affine.for %{{.*}} = max affine_map<(d0, d1) -> (d0, d1)>(%{{.*}}, %{{.*}}) to min affine_map<(d0, d1) -> (d0, d1)>(%{{.*}}, %{{.*}}) {
  // CHECK:  return
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

TEST_FUNC(builder_blocks) {
  using namespace edsc::op;
  auto f = makeFunction("builder_blocks");

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  ValueHandle c1(ValueHandle::create<ConstantIntOp>(42, 32)),
      c2(ValueHandle::create<ConstantIntOp>(1234, 32));
  ValueHandle arg1(c1.getType()), arg2(c1.getType()), arg3(c1.getType()),
      arg4(c1.getType()), r(c1.getType());

  BlockHandle b1, b2, functionBlock(&f.front());
  BlockBuilder(&b1, {&arg1, &arg2})(
      // b2 has not yet been constructed, need to come back later.
      // This is a byproduct of non-structured control-flow.
  );
  BlockBuilder(&b2, {&arg3, &arg4})([&] { br(b1, {arg3, arg4}); });
  // The insertion point within the toplevel function is now past b2, we will
  // need to get back the entry block.
  // This is what happens with unstructured control-flow..
  BlockBuilder(b1, Append())([&] {
    r = arg1 + arg2;
    br(b2, {arg1, r});
  });
  // Get back to entry block and add a branch into b1
  BlockBuilder(functionBlock, Append())([&] { br(b1, {c1, c2}); });

  // clang-format off
  // CHECK-LABEL: @builder_blocks
  // CHECK:        %{{.*}} = constant 42 : i32
  // CHECK-NEXT:   %{{.*}} = constant 1234 : i32
  // CHECK-NEXT:   br ^bb1(%{{.*}}, %{{.*}} : i32, i32)
  // CHECK-NEXT: ^bb1(%{{.*}}: i32, %{{.*}}: i32):   // 2 preds: ^bb0, ^bb2
  // CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : i32
  // CHECK-NEXT:   br ^bb2(%{{.*}}, %{{.*}} : i32, i32)
  // CHECK-NEXT: ^bb2(%{{.*}}: i32, %{{.*}}: i32):   // pred: ^bb1
  // CHECK-NEXT:   br ^bb1(%{{.*}}, %{{.*}} : i32, i32)
  // CHECK-NEXT: }
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

TEST_FUNC(builder_blocks_eager) {
  using namespace edsc::op;
  auto f = makeFunction("builder_blocks_eager");

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  ValueHandle c1(ValueHandle::create<ConstantIntOp>(42, 32)),
      c2(ValueHandle::create<ConstantIntOp>(1234, 32));
  ValueHandle arg1(c1.getType()), arg2(c1.getType()), arg3(c1.getType()),
      arg4(c1.getType()), r(c1.getType());

  // clang-format off
  BlockHandle b1, b2;
  { // Toplevel function scope.
    // Build a new block for b1 eagerly.
    br(&b1, {&arg1, &arg2}, {c1, c2});
    // Construct a new block b2 explicitly with a branch into b1.
    BlockBuilder(&b2, {&arg3, &arg4})([&]{
        br(b1, {arg3, arg4});
    });
    /// And come back to append into b1 once b2 exists.
    BlockBuilder(b1, Append())([&]{
        r = arg1 + arg2;
        br(b2, {arg1, r});
    });
  }

  // CHECK-LABEL: @builder_blocks_eager
  // CHECK:        %{{.*}} = constant 42 : i32
  // CHECK-NEXT:   %{{.*}} = constant 1234 : i32
  // CHECK-NEXT:   br ^bb1(%{{.*}}, %{{.*}} : i32, i32)
  // CHECK-NEXT: ^bb1(%{{.*}}: i32, %{{.*}}: i32):   // 2 preds: ^bb0, ^bb2
  // CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : i32
  // CHECK-NEXT:   br ^bb2(%{{.*}}, %{{.*}} : i32, i32)
  // CHECK-NEXT: ^bb2(%{{.*}}: i32, %{{.*}}: i32):   // pred: ^bb1
  // CHECK-NEXT:   br ^bb1(%{{.*}}, %{{.*}} : i32, i32)
  // CHECK-NEXT: }
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

TEST_FUNC(builder_cond_branch) {
  auto f = makeFunction("builder_cond_branch", {},
                        {IntegerType::get(1, &globalContext())});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  ValueHandle funcArg(f.getArgument(0));
  ValueHandle c32(ValueHandle::create<ConstantIntOp>(32, 32)),
      c64(ValueHandle::create<ConstantIntOp>(64, 64)),
      c42(ValueHandle::create<ConstantIntOp>(42, 32));
  ValueHandle arg1(c32.getType()), arg2(c64.getType()), arg3(c32.getType());

  BlockHandle b1, b2, functionBlock(&f.front());
  BlockBuilder(&b1, {&arg1})([&] { std_ret(); });
  BlockBuilder(&b2, {&arg2, &arg3})([&] { std_ret(); });
  // Get back to entry block and add a conditional branch
  BlockBuilder(functionBlock, Append())([&] {
    cond_br(funcArg, b1, {c32}, b2, {c64, c42});
  });

  // clang-format off
  // CHECK-LABEL: @builder_cond_branch
  // CHECK:   %{{.*}} = constant 32 : i32
  // CHECK-NEXT:   %{{.*}} = constant 64 : i64
  // CHECK-NEXT:   %{{.*}} = constant 42 : i32
  // CHECK-NEXT:   cond_br %{{.*}}, ^bb1(%{{.*}} : i32), ^bb2(%{{.*}}, %{{.*}} : i64, i32)
  // CHECK-NEXT: ^bb1(%{{.*}}: i32):   // pred: ^bb0
  // CHECK-NEXT:   return
  // CHECK-NEXT: ^bb2(%{{.*}}: i64, %{{.*}}: i32):  // pred: ^bb0
  // CHECK-NEXT:   return
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

TEST_FUNC(builder_cond_branch_eager) {
  using namespace edsc::op;
  auto f = makeFunction("builder_cond_branch_eager", {},
                        {IntegerType::get(1, &globalContext())});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  ValueHandle funcArg(f.getArgument(0));
  ValueHandle c32(ValueHandle::create<ConstantIntOp>(32, 32)),
      c64(ValueHandle::create<ConstantIntOp>(64, 64)),
      c42(ValueHandle::create<ConstantIntOp>(42, 32));
  ValueHandle arg1(c32.getType()), arg2(c64.getType()), arg3(c32.getType());

  // clang-format off
  BlockHandle b1, b2;
  cond_br(funcArg, &b1, {&arg1}, {c32}, &b2, {&arg2, &arg3}, {c64, c42});
  BlockBuilder(b1, Append())([]{
      std_ret();
  });
  BlockBuilder(b2, Append())([]{
      std_ret();
  });

  // CHECK-LABEL: @builder_cond_branch_eager
  // CHECK:   %{{.*}} = constant 32 : i32
  // CHECK-NEXT:   %{{.*}} = constant 64 : i64
  // CHECK-NEXT:   %{{.*}} = constant 42 : i32
  // CHECK-NEXT:   cond_br %{{.*}}, ^bb1(%{{.*}} : i32), ^bb2(%{{.*}}, %{{.*}} : i64, i32)
  // CHECK-NEXT: ^bb1(%{{.*}}: i32):   // pred: ^bb0
  // CHECK-NEXT:   return
  // CHECK-NEXT: ^bb2(%{{.*}}: i64, %{{.*}}: i32):  // pred: ^bb0
  // CHECK-NEXT:   return
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

TEST_FUNC(builder_helpers) {
  using namespace edsc::op;
  auto indexType = IndexType::get(&globalContext());
  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType =
      MemRefType::get({ShapedType::kDynamicSize, ShapedType::kDynamicSize,
                       ShapedType::kDynamicSize},
                      f32Type, {}, 0);
  auto f =
      makeFunction("builder_helpers", {}, {memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  // clang-format off
  ValueHandle f7(
      ValueHandle::create<ConstantFloatOp>(llvm::APFloat(7.0f), f32Type));
  MemRefBoundsCapture vA(f.getArgument(0)), vB(f.getArgument(1)),
      vC(f.getArgument(2));
  AffineIndexedValue A(f.getArgument(0)), B(f.getArgument(1)), C(f.getArgument(2));
  ValueHandle i(indexType), j(indexType), k1(indexType), k2(indexType),
      lb0(indexType), lb1(indexType), lb2(indexType),
      ub0(indexType), ub1(indexType), ub2(indexType);
  int64_t step0, step1, step2;
  std::tie(lb0, ub0, step0) = vA.range(0);
  std::tie(lb1, ub1, step1) = vA.range(1);
  lb2 = vA.lb(2);
  ub2 = vA.ub(2);
  step2 = vA.step(2);
  AffineLoopNestBuilder({&i, &j}, {lb0, lb1}, {ub0, ub1}, {step0, step1})([&]{
    AffineLoopNestBuilder(&k1, lb2, ub2, step2)([&]{
      C(i, j, k1) = f7 + A(i, j, k1) + B(i, j, k1);
    });
    AffineLoopNestBuilder(&k2, lb2, ub2, step2)([&]{
      C(i, j, k2) += A(i, j, k2) + B(i, j, k2);
    });
  });

  // CHECK-LABEL: @builder_helpers
  //      CHECK:   affine.for %{{.*}} = affine_map<(d0) -> (d0)>({{.*}}) to affine_map<(d0) -> (d0)>({{.*}}) {
  // CHECK-NEXT:     affine.for %{{.*}} = affine_map<(d0) -> (d0)>({{.*}}) to affine_map<(d0) -> (d0)>({{.*}}) {
  // CHECK-NEXT:       affine.for %{{.*}} = affine_map<(d0) -> (d0)>({{.*}}) to affine_map<(d0) -> (d0)>({{.*}}) {
  //  CHECK-DAG:         [[a:%.*]] = affine.load %arg0[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-DAG:         [[b:%.*]] = addf {{.*}}, [[a]] : f32
  //  CHECK-DAG:         [[c:%.*]] = affine.load %arg1[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-DAG:         [[d:%.*]] = addf [[b]], [[c]] : f32
  // CHECK-NEXT:         affine.store [[d]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  // CHECK-NEXT:       }
  // CHECK-NEXT:       affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%{{.*}}) to affine_map<(d0) -> (d0)>(%{{.*}}) {
  //  CHECK-DAG:         [[a:%.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-DAG:         [[b:%.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-DAG:         [[c:%.*]] = addf [[b]], [[a]] : f32
  //  CHECK-DAG:         [[d:%.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-DAG:         [[e:%.*]] = addf [[d]], [[c]] : f32
  // CHECK-NEXT:         affine.store [[e]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

TEST_FUNC(custom_ops) {
  using namespace edsc::op;
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("custom_ops", {}, {indexType, indexType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  CustomOperation<ValueHandle> MY_CUSTOM_OP("my_custom_op");
  CustomOperation<OperationHandle> MY_CUSTOM_OP_0("my_custom_op_0");
  CustomOperation<OperationHandle> MY_CUSTOM_OP_2("my_custom_op_2");

  // clang-format off
  ValueHandle vh(indexType), vh20(indexType), vh21(indexType);
  OperationHandle ih0, ih2;
  ValueHandle m(indexType), n(indexType);
  ValueHandle M(f.getArgument(0)), N(f.getArgument(1));
  ValueHandle ten(std_constant_index(10)), twenty(std_constant_index(20));
  AffineLoopNestBuilder({&m, &n}, {M, N}, {M + ten, N + twenty}, {1, 1})([&]{
    vh = MY_CUSTOM_OP({m, m + n}, {indexType}, {});
    ih0 = MY_CUSTOM_OP_0({m, m + n}, {});
    ih2 = MY_CUSTOM_OP_2({m, m + n}, {indexType, indexType});
    // These captures are verbose for now, can improve when used in practice.
    vh20 = ValueHandle(ih2.getOperation()->getResult(0));
    vh21 = ValueHandle(ih2.getOperation()->getResult(1));
    MY_CUSTOM_OP({vh20, vh21}, {indexType}, {});
  });

  // CHECK-LABEL: @custom_ops
  // CHECK: affine.for %{{.*}} {{.*}}
  // CHECK:   affine.for %{{.*}} {{.*}}
  // CHECK:     {{.*}} = "my_custom_op"{{.*}} : (index, index) -> index
  // CHECK:     "my_custom_op_0"{{.*}} : (index, index) -> ()
  // CHECK:     [[TWO:%[a-z0-9]+]]:2 = "my_custom_op_2"{{.*}} : (index, index) -> (index, index)
  // CHECK:     {{.*}} = "my_custom_op"([[TWO]]#0, [[TWO]]#1) : (index, index) -> index
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

TEST_FUNC(insertion_in_block) {
  using namespace edsc::op;
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("insertion_in_block", {}, {indexType, indexType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  BlockHandle b1;
  // clang-format off
  ValueHandle::create<ConstantIntOp>(0, 32);
  BlockBuilder(&b1, {})([]{
    ValueHandle::create<ConstantIntOp>(1, 32);
  });
  ValueHandle::create<ConstantIntOp>(2, 32);
  // CHECK-LABEL: @insertion_in_block
  // CHECK: {{.*}} = constant 0 : i32
  // CHECK: {{.*}} = constant 2 : i32
  // CHECK: ^bb1:   // no predecessors
  // CHECK: {{.*}} = constant 1 : i32
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

TEST_FUNC(zero_and_std_sign_extendi_op_i1_to_i8) {
  using namespace edsc::op;
  auto i1Type = IntegerType::get(1, &globalContext());
  auto i8Type = IntegerType::get(8, &globalContext());
  auto memrefType = MemRefType::get({}, i1Type, {}, 0);
  auto f = makeFunction("zero_and_std_sign_extendi_op", {},
                        {memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  AffineIndexedValue A(f.getArgument(0));
  AffineIndexedValue B(f.getArgument(1));
  // clang-format off
  edsc::intrinsics::std_zero_extendi(*A, i8Type);
  edsc::intrinsics::std_sign_extendi(*B, i8Type);
  // CHECK-LABEL: @zero_and_std_sign_extendi_op
  //      CHECK:     %[[SRC1:.*]] = affine.load
  //      CHECK:     zexti %[[SRC1]] : i1 to i8
  //      CHECK:     %[[SRC2:.*]] = affine.load
  //      CHECK:     sexti %[[SRC2]] : i1 to i8
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

TEST_FUNC(select_op_i32) {
  using namespace edsc::op;
  auto indexType = IndexType::get(&globalContext());
  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get(
      {ShapedType::kDynamicSize, ShapedType::kDynamicSize}, f32Type, {}, 0);
  auto f = makeFunction("select_op", {}, {memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  // clang-format off
  ValueHandle zero = std_constant_index(0), one = std_constant_index(1);
  MemRefBoundsCapture vA(f.getArgument(0));
  AffineIndexedValue A(f.getArgument(0));
  ValueHandle i(indexType), j(indexType);
  AffineLoopNestBuilder({&i, &j}, {zero, zero}, {one, one}, {1, 1})([&]{
    // This test exercises AffineIndexedValue::operator Value.
    // Without it, one must force conversion to ValueHandle as such:
    //   std_select(
    //      i == zero, ValueHandle(A(zero, zero)), ValueHandle(ValueA(i, j)))
    using edsc::op::operator==;
    std_select(i == zero, *A(zero, zero), *A(i, j));
  });

  // CHECK-LABEL: @select_op
  //      CHECK: affine.for %{{.*}} = 0 to 1 {
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to 1 {
  //  CHECK-DAG:     {{.*}} = cmpi "eq"
  //  CHECK-DAG:     {{.*}} = affine.load
  //  CHECK-DAG:     {{.*}} = affine.load
  // CHECK-NEXT:     {{.*}} = select
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

TEST_FUNC(select_op_f32) {
  auto indexType = IndexType::get(&globalContext());
  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get(
      {ShapedType::kDynamicSize, ShapedType::kDynamicSize}, f32Type, {}, 0);
  auto f = makeFunction("select_op", {}, {memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  // clang-format off
  ValueHandle zero = std_constant_index(0), one = std_constant_index(1);
  MemRefBoundsCapture vA(f.getArgument(0)), vB(f.getArgument(1));
  AffineIndexedValue A(f.getArgument(0)), B(f.getArgument(1));
  ValueHandle i(indexType), j(indexType);
  AffineLoopNestBuilder({&i, &j}, {zero, zero}, {one, one}, {1, 1})([&]{
    using namespace edsc::op;
    std_select(B(i, j) == B(i + one, j), *A(zero, zero), *A(i, j));
    std_select(B(i, j) != B(i + one, j), *A(zero, zero), *A(i, j));
    std_select(B(i, j) >= B(i + one, j), *A(zero, zero), *A(i, j));
    std_select(B(i, j) <= B(i + one, j), *A(zero, zero), *A(i, j));
    std_select(B(i, j) < B(i + one, j), *A(zero, zero), *A(i, j));
    std_select(B(i, j) > B(i + one, j), *A(zero, zero), *A(i, j));
  });

  // CHECK-LABEL: @select_op
  //      CHECK: affine.for %{{.*}} = 0 to 1 {
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to 1 {
  //  CHECK-DAG:     cmpf "oeq"
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  //  CHECK-DAG:     cmpf "one"
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  //  CHECK-DAG:     cmpf "oge"
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  //  CHECK-DAG:     cmpf "ole"
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  //  CHECK-DAG:     cmpf "olt"
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  //  CHECK-DAG:     cmpf "ogt"
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

// Inject an EDSC-constructed computation to exercise imperfectly nested 2-d
// tiling.
TEST_FUNC(tile_2d) {
  auto indexType = IndexType::get(&globalContext());
  auto memrefType =
      MemRefType::get({ShapedType::kDynamicSize, ShapedType::kDynamicSize,
                       ShapedType::kDynamicSize},
                      FloatType::getF32(&globalContext()), {}, 0);
  auto f = makeFunction("tile_2d", {}, {memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  ValueHandle zero = std_constant_index(0);
  MemRefBoundsCapture vA(f.getArgument(0)), vB(f.getArgument(1)),
      vC(f.getArgument(2));
  AffineIndexedValue A(f.getArgument(0)), B(f.getArgument(1)),
      C(f.getArgument(2));
  ValueHandle i(indexType), j(indexType), k1(indexType), k2(indexType);
  ValueHandle M(vC.ub(0)), N(vC.ub(1)), O(vC.ub(2));

  // clang-format off
  using namespace edsc::op;
  AffineLoopNestBuilder({&i, &j}, {zero, zero}, {M, N}, {1, 1})([&]{
    AffineLoopNestBuilder(&k1, zero, O, 1)([&]{
      C(i, j, k1) = A(i, j, k1) + B(i, j, k1);
    });
    AffineLoopNestBuilder(&k2, zero, O, 1)([&]{
      C(i, j, k2) = A(i, j, k2) + B(i, j, k2);
    });
  });
  // clang-format on

  auto li = getForInductionVarOwner(i.getValue()),
       lj = getForInductionVarOwner(j.getValue()),
       lk1 = getForInductionVarOwner(k1.getValue()),
       lk2 = getForInductionVarOwner(k2.getValue());
  auto indicesL1 = mlir::tile({li, lj}, {512, 1024}, {lk1, lk2});
  auto lii1 = indicesL1[0][0], ljj1 = indicesL1[1][0];
  mlir::tile({ljj1, lii1}, {32, 16}, ljj1);

  // clang-format off
  // CHECK-LABEL: func @tile_2d
  //       CHECK: %[[ZERO:.*]] = constant 0 : index
  //       CHECK: %[[M:[0-9]+]] = dim %arg2, 0 : memref<?x?x?xf32>
  //  CHECK-NEXT: %[[N:[0-9]+]] = dim %arg2, 1 : memref<?x?x?xf32>
  //  CHECK-NEXT: %[[P:[0-9]+]] = dim %arg2, 2 : memref<?x?x?xf32>
  //       CHECK:   affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%[[ZERO]]) to affine_map<(d0) -> (d0)>(%[[M]]) step 512 {
  //  CHECK-NEXT:     affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%[[ZERO]]) to affine_map<(d0) -> (d0)>(%[[N]]) step 1024 {
  //  CHECK-NEXT:       affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%[[ZERO]]) to affine_map<(d0) -> (d0)>(%[[P]]) {
  //  CHECK-NEXT:         affine.for %{{.*}} = max affine_map<(d0) -> (0, d0)>(%{{.*}}) to min affine_map<(d0)[s0] -> (s0, d0 + 512)>(%{{.*}})[%[[M]]] step 16 {
  //  CHECK-NEXT:           affine.for %{{.*}} = max affine_map<(d0) -> (0, d0)>(%{{.*}}) to min affine_map<(d0)[s0] -> (s0, d0 + 1024)>(%{{.*}})[%[[N]]] step 32 {
  //  CHECK-NEXT:             affine.for %{{.*}} = max affine_map<(d0, d1) -> (0, d0, d1)>(%{{.*}}, %{{.*}}) to min affine_map<(d0, d1)[s0] -> (s0, d0 + 1024, d1 + 32)>(%{{.*}}, %{{.*}})[%[[N]]] {
  //  CHECK-NEXT:               affine.for %{{.*}} = max affine_map<(d0, d1) -> (0, d0, d1)>(%{{.*}}, %{{.*}}) to min affine_map<(d0, d1)[s0] -> (s0, d0 + 512, d1 + 16)>(%{{.*}}, %{{.*}})[%[[M]]] {
  //  CHECK-NEXT:                 {{.*}} = affine.load {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-NEXT:                 {{.*}} = affine.load {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-NEXT:                 {{.*}} = addf {{.*}}, {{.*}} : f32
  //  CHECK-NEXT:                 affine.store {{.*}}, {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //       CHECK:               }
  //  CHECK-NEXT:             }
  //  CHECK-NEXT:           }
  //  CHECK-NEXT:         }
  //  CHECK-NEXT:       }
  //  CHECK-NEXT:       affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%[[ZERO]]) to affine_map<(d0) -> (d0)>(%[[P]]) {
  //  CHECK-NEXT:         affine.for %{{.*}} = max affine_map<(d0) -> (0, d0)>(%{{.*}}) to min affine_map<(d0)[s0] -> (s0, d0 + 512)>(%{{.*}})[%[[M]]] {
  //  CHECK-NEXT:           affine.for %{{.*}} = max affine_map<(d0) -> (0, d0)>(%{{.*}}) to min affine_map<(d0)[s0] -> (s0, d0 + 1024)>(%{{.*}})[%[[N]]] {
  //  CHECK-NEXT:             {{.*}} = affine.load {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-NEXT:             {{.*}} = affine.load {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-NEXT:             {{.*}}= addf {{.*}}, {{.*}} : f32
  //  CHECK-NEXT:             affine.store {{.*}}, {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

// Exercise StdIndexedValue for loads and stores.
TEST_FUNC(indirect_access) {
  using namespace edsc::op;
  auto memrefType = MemRefType::get({ShapedType::kDynamicSize},
                                    FloatType::getF32(&globalContext()), {}, 0);
  auto f = makeFunction("indirect_access", {},
                        {memrefType, memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  ValueHandle zero = std_constant_index(0);
  MemRefBoundsCapture vC(f.getArgument(2));
  AffineIndexedValue B(f.getArgument(1)), D(f.getArgument(3));
  StdIndexedValue A(f.getArgument(0)), C(f.getArgument(2));
  ValueHandle i(builder.getIndexType()), N(vC.ub(0));

  // clang-format off
  AffineLoopNestBuilder(&i, zero, N, 1)([&]{
      C((ValueHandle)D(i)) = A((ValueHandle)B(i));
  });
  // clang-format on

  // clang-format off
  // CHECK-LABEL: func @indirect_access
  // CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: memref<?xf32>)
  // CHECK-DAG:  [[B:%.*]] = affine.load %[[ARG1]]
  // CHECK-DAG:  [[D:%.*]] = affine.load %[[ARG3]]
  // CHECK:  load %{{.*}}{{\[}}[[B]]{{\]}}
  // CHECK:  store %{{.*}}, %{{.*}}{{\[}}[[D]]{{\]}}
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

// Exercise affine loads and stores build with empty maps.
TEST_FUNC(empty_map_load_store) {
  using namespace edsc::op;
  auto memrefType =
      MemRefType::get({}, FloatType::getF32(&globalContext()), {}, 0);
  auto f = makeFunction("empty_map_load_store", {},
                        {memrefType, memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  ValueHandle zero = std_constant_index(0);
  ValueHandle one = std_constant_index(1);
  AffineIndexedValue input(f.getArgument(0)), res(f.getArgument(1));
  ValueHandle iv(builder.getIndexType());

  // clang-format off
  AffineLoopNestBuilder(&iv, zero, one, 1)([&]{
      res() = input();
  });
  // clang-format on

  // clang-format off
  // CHECK-LABEL: func @empty_map_load_store(
  // CHECK:  [[A:%.*]] = affine.load %{{.*}}[]
  // CHECK:  affine.store [[A]], %{{.*}}[]
  // clang-format on
  f.print(llvm::outs());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @affine_if_op
// CHECK:       affine.if affine_set<([[d0:.*]], [[d1:.*]]){{\[}}[[s0:.*]], [[s1:.*]]{{\]}}
// CHECK-NOT:   else
// CHECK:       affine.if affine_set<([[d0:.*]], [[d1:.*]]){{\[}}[[s0:.*]], [[s1:.*]]{{\]}}
// CHECK-NEXT: } else {
// clang-format on
TEST_FUNC(affine_if_op) {
  using namespace edsc::op;
  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get(
      {ShapedType::kDynamicSize, ShapedType::kDynamicSize}, f32Type, {}, 0);
  auto f = makeFunction("affine_if_op", {}, {memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());

  ValueHandle zero = std_constant_index(0), ten = std_constant_index(10);

  SmallVector<bool, 4> isEq = {false, false, false, false};
  SmallVector<AffineExpr, 4> affineExprs = {
      builder.getAffineDimExpr(0),    // d0 >= 0
      builder.getAffineDimExpr(1),    // d1 >= 0
      builder.getAffineSymbolExpr(0), // s0 >= 0
      builder.getAffineSymbolExpr(1)  // s1 >= 0
  };
  auto intSet = IntegerSet::get(2, 2, affineExprs, isEq);

  SmallVector<Value, 4> affineIfArgs = {zero, zero, ten, ten};
  intrinsics::affine_if(intSet, affineIfArgs, /*withElseRegion=*/false);
  intrinsics::affine_if(intSet, affineIfArgs, /*withElseRegion=*/true);

  f.print(llvm::outs());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @linalg_pointwise
//       CHECK:   linalg.generic {args_in = 2 : i64, args_out = 1 : i64,
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
//       CHECK:       addf
//       CHECK:     }: memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
//       CHECK:   linalg.generic {args_in = 2 : i64, args_out = 1 : i64,
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
//       CHECK:       cmpf "ogt"
//       CHECK:       select
//       CHECK:   }: memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
//       CHECK:   linalg.generic {args_in = 1 : i64, args_out = 1 : i64,
// CHECK-SAME:      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]}
//       CHECK:     tanh
//       CHECK:   }: memref<?x?xf32>, memref<?x?xf32>
// clang-format on
TEST_FUNC(linalg_pointwise_test) {
  using namespace edsc;
  using namespace edsc::ops;

  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get(
      {ShapedType::kDynamicSize, ShapedType::kDynamicSize}, f32Type, {}, 0);
  auto f = makeFunction("linalg_pointwise", {},
                        {memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  ValueHandle A(f.getArgument(0)), B(f.getArgument(1)), C(f.getArgument(2));
  AffineExpr i, j;
  bindDims(&globalContext(), i, j);
  StructuredIndexed SA(A), SB(B), SC(C);
  linalg_pointwise_add(SA({i, j}), SB({i, j}), SC({i, j}));
  linalg_pointwise_max(SA({i, j}), SB({i, j}), SC({i, j}));
  linalg_pointwise_tanh(SA({i, j}), SC({i, j}));

  f.print(llvm::outs());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @linalg_matmul
//       CHECK:   linalg.generic {args_in = 2 : i64, args_out = 1 : i64,
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]}
///      CHECK:   ^bb0(%[[a0:.*]]: f32, %[[a1:.*]]: f32, %[[a2:.*]]: f32):
//       CHECK:     %[[a3:.*]] = mulf %[[a0]], %[[a1]] : f32
//       CHECK:     %[[a4:.*]] = addf %[[a2]], %[[a3]] : f32
//       CHECK:     linalg.yield %[[a4]] : f32
//       CHECK:   }: memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>
// clang-format on
TEST_FUNC(linalg_matmul_test) {
  using namespace edsc;
  using namespace edsc::ops;

  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get(
      {ShapedType::kDynamicSize, ShapedType::kDynamicSize}, f32Type, {}, 0);
  auto f =
      makeFunction("linalg_matmul", {}, {memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  linalg_matmul(makeValueHandles(llvm::to_vector<3>(f.getArguments())));

  f.print(llvm::outs());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @linalg_conv_nhwc
//       CHECK:   linalg.generic {args_in = 2 : i64, args_out = 1 : i64,
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2 * 3 + d4 * 5, d3 * 4 + d5 * 6, d6)>,
// CHECK-SAME: affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d1)>,
// CHECK-SAME: affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d3, d1)>],
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
///      CHECK:   ^bb0(%[[a0:.*]]: f32, %[[a1:.*]]: f32, %[[a2:.*]]: f32):
//       CHECK:     %[[a3:.*]] = mulf %[[a0]], %[[a1]] : f32
//       CHECK:     %[[a4:.*]] = addf %[[a2]], %[[a3]] : f32
//       CHECK:     linalg.yield %[[a4]] : f32
//       CHECK:   }: memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
// clang-format on
TEST_FUNC(linalg_conv_nhwc) {
  using namespace edsc;
  using namespace edsc::ops;

  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType =
      MemRefType::get({ShapedType::kDynamicSize, ShapedType::kDynamicSize,
                       ShapedType::kDynamicSize, ShapedType::kDynamicSize},
                      f32Type, {}, 0);
  auto f = makeFunction("linalg_conv_nhwc", {},
                        {memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  linalg_conv_nhwc(makeValueHandles(llvm::to_vector<3>(f.getArguments())),
                   /*strides=*/{3, 4}, /*dilations=*/{5, 6});

  f.print(llvm::outs());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @linalg_dilated_conv_nhwc
//       CHECK:   linalg.generic {args_in = 2 : i64, args_out = 1 : i64,
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d3 * 3 + d5 * 5, d4 * 4 + d6 * 6, d2)>,
// CHECK-SAME: affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d2, d1)>,
// CHECK-SAME: affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d3, d4, d1 + d2 * 7)>],
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
//       CHECK:   ^bb0(%[[a0:.*]]: f32, %[[a1:.*]]: f32, %[[a2:.*]]: f32):
//       CHECK:     %[[a3:.*]] = mulf %[[a0]], %[[a1]] : f32
//       CHECK:     %[[a4:.*]] = addf %[[a2]], %[[a3]] : f32
//       CHECK:     linalg.yield %[[a4]] : f32
//       CHECK:   }: memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, memref<?x?x?x?xf32>
// clang-format on
TEST_FUNC(linalg_dilated_conv_nhwc) {
  using namespace edsc;
  using namespace edsc::ops;

  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType =
      MemRefType::get({ShapedType::kDynamicSize, ShapedType::kDynamicSize,
                       ShapedType::kDynamicSize, ShapedType::kDynamicSize},
                      f32Type, {}, 0);
  auto f = makeFunction("linalg_dilated_conv_nhwc", {},
                        {memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  linalg_dilated_conv_nhwc(
      makeValueHandles(llvm::to_vector<3>(f.getArguments())),
      /*depth_multiplier=*/7,
      /*strides=*/{3, 4}, /*dilations=*/{5, 6});

  f.print(llvm::outs());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @linalg_metadata_ops
//       CHECK: linalg.reshape {{.*}} [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d2)>] : memref<4x8x16xf32> into memref<32x16xf32>
//       CHECK: linalg.reshape {{.*}} [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d2)>] : memref<32x16xf32> into memref<4x8x16xf32>
// clang-format on
TEST_FUNC(linalg_metadata_ops) {
  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get({4, 8, 16}, f32Type, {}, 0);
  auto f = makeFunction("linalg_metadata_ops", {}, {memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  AffineExpr i, j, k;
  bindDims(&globalContext(), i, j, k);
  ValueHandle v(f.getArgument(0));
  auto reshaped = linalg_reshape(v, ArrayRef<ArrayRef<AffineExpr>>{{i, j}, k});
  linalg_reshape(memrefType, reshaped,
                 ArrayRef<ArrayRef<AffineExpr>>{{i, j}, k});

  f.print(llvm::outs());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @linalg_tensors
//       CHECK:   linalg.generic {args_in = 2 : i64, args_out = 1 : i64,
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
//       CHECK:       addf
//       CHECK:     }: tensor<?x?xf32>, memref<?x?xf32> -> tensor<?x?xf32>
//       CHECK:   linalg.generic {args_in = 2 : i64, args_out = 1 : i64,
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
//       CHECK:       cmpf "ogt"
//       CHECK:       select
//       CHECK:   }: tensor<?x?xf32>, memref<?x?xf32> -> tensor<?x?xf32>
//       CHECK:   linalg.generic {args_in = 1 : i64, args_out = 1 : i64,
// CHECK-SAME:      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]}
//       CHECK:     tanh
//       CHECK:   }: tensor<?x?xf32> -> tensor<?x?xf32>
//       CHECK:   linalg.generic {args_in = 2 : i64, args_out = 1 : i64,
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d2, d1)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d0, d1)>],
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]}
//       CHECK:     mulf
//       CHECK:   }: tensor<?x?xf32>, memref<?x?xf32> -> tensor<?x?xf32>
//       CHECK:   linalg.generic {args_in = 3 : i64, args_out = 1 : i64,
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d2, d1)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d0, d1)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d0, d1)>],
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//       CHECK:     mulf
//       CHECK:     addf
//       CHECK:   }: tensor<?x?xf32>, memref<?x?xf32>, tensor<?x?xf32> -> tensor<?x?xf32>
// clang-format on
TEST_FUNC(linalg_tensors_test) {
  using namespace edsc;
  using namespace edsc::ops;

  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get(
      {ShapedType::kDynamicSize, ShapedType::kDynamicSize}, f32Type, {}, 0);
  auto tensorType = RankedTensorType::get(
      {ShapedType::kDynamicSize, ShapedType::kDynamicSize}, f32Type);
  auto f = makeFunction("linalg_tensors", {}, {tensorType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  ValueHandle A(f.getArgument(0)), B(f.getArgument(1));
  AffineExpr i, j;
  bindDims(&globalContext(), i, j);
  StructuredIndexed SA(A), SB(B), SC(tensorType);
  linalg_pointwise_add(SA({i, j}), SB({i, j}), SC({i, j}));
  linalg_pointwise_max(SA({i, j}), SB({i, j}), SC({i, j}));
  linalg_pointwise_tanh(SA({i, j}), SC({i, j}));
  Value o1 = linalg_matmul(A, B, tensorType)->getResult(0);
  linalg_matmul(A, B, ValueHandle(o1), tensorType);

  f.print(llvm::outs());
  f.erase();
}

// CHECK-LABEL: func @vector_matmul_test(
//  CHECK-SAME:   %[[A:.*]]: vector<4x16xf32>,
//  CHECK-SAME:   %[[B:.*]]: vector<16x8xf32>,
//  CHECK-SAME:   %[[C:.*]]: vector<4x8xf32>)
//  CHECK:   vector.contract {{.*}}[affine_map<(d0, d1, d2) -> (d0, d2)>,
//  CHECK-SAME:                     affine_map<(d0, d1, d2) -> (d2, d1)>,
//  CHECK-SAME:                     affine_map<(d0, d1, d2) -> (d0, d1)>],
//  CHECK-SAME:              {{.*}}["parallel", "parallel", "reduction"]
//  CHECK-SAME: %[[A]], %[[B]], %[[C]]
//  CHECK-SAME:   vector<4x16xf32>, vector<16x8xf32> into vector<4x8xf32>
TEST_FUNC(vector_matmul_test) {
  using namespace edsc;
  using namespace edsc::ops;

  int64_t M = 4, N = 8, K = 16;
  auto f32Type = FloatType::getF32(&globalContext());
  auto mkVectorType = VectorType::get({M, K}, f32Type);
  auto knVectorType = VectorType::get({K, N}, f32Type);
  auto mnVectorType = VectorType::get({M, N}, f32Type);
  auto f = makeFunction("vector_matmul_test", {},
                        {mkVectorType, knVectorType, mnVectorType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  ValueHandle A(f.getArgument(0)), B(f.getArgument(1)), C(f.getArgument(2));
  vector_matmul(A, B, C);
  f.print(llvm::outs());
  f.erase();
}

int main() {
  RUN_TESTS();
  return 0;
}
