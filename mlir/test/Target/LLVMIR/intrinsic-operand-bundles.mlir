// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define i32 @ctpop_with_bundle(
// CHECK-SAME: i32 %[[ARG:.*]], i64 %[[STATE:.*]])
// CHECK: %[[RESULT:.*]] = call i32 @llvm.ctpop.i32(i32 %[[ARG]]) [ "tag"(i64 %[[STATE]]) ]
// CHECK: ret i32 %[[RESULT]]
llvm.func @ctpop_with_bundle(%arg: i32, %state: i64) -> i32 {
  %result = llvm.call_intrinsic "llvm.ctpop.i32"(%arg) ["tag"(%state : i64)] : (i32) -> i32
  llvm.return %result : i32
}
