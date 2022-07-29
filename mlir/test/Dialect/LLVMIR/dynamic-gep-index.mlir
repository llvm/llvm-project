// RUN: mlir-opt %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<[32, 64]> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>} {
  // CHECK: llvm.func @foo(%[[ARG0:.+]]: !llvm.ptr<struct<"my_struct", {{.+}}>>, %[[ARG1:.+]]: i32)
  llvm.func @foo(%arg0: !llvm.ptr<struct<"my_struct", (struct<"sub_struct", (i32, i8)>, array<4 x i32>)>>, %arg1: i32) {
    // CHECK: %[[C0:.+]] = llvm.mlir.constant(0 : i32)
    %0 = llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.getelementptr %[[ARG0]][%[[C0]], 1, %[[ARG1]]]
    %1 = "llvm.getelementptr"(%arg0, %0, %arg1) {rawConstantIndices = [:i32 -2147483648, 1, -2147483648]} : (!llvm.ptr<struct<"my_struct", (struct<"sub_struct", (i32, i8)>, array<4 x i32>)>>, i32, i32) -> !llvm.ptr<i32>
    llvm.return
  }
}
