// RUN: mlir-opt %s -split-input-file | FileCheck %s

llvm.func local_unnamed_addr @testfn(!llvm.array<2 x f32> {llvm.alignstack = 8 : i64})
llvm.func internal @g(%arg0: !llvm.array<2 x f32>) attributes {dso_local} {
  // CHECK-LABEL: @g
  // CHECK: llvm.call @testfn(%arg0) : (!llvm.array<2 x f32> {llvm.alignstack = 8 : i64}) -> ()
  llvm.call @testfn(%arg0) : (!llvm.array<2 x f32> {llvm.alignstack = 8 : i64}) -> ()
  llvm.return
}
llvm.func local_unnamed_addr @testfn2(!llvm.struct<(i8, i8)> {llvm.alignstack = 8 : i64})
llvm.func internal @h(%arg0: !llvm.struct<(i8, i8)>) attributes {dso_local} {
  // CHECK-LABEL: @h
  // CHECK: llvm.call @testfn2(%arg0) : (!llvm.struct<(i8, i8)> {llvm.alignstack = 8 : i64}) -> ()
  llvm.call @testfn2(%arg0) : (!llvm.struct<(i8, i8)> {llvm.alignstack = 8 : i64}) -> ()
  llvm.return
}
llvm.func local_unnamed_addr @testfn3(i32 {llvm.alignstack = 8 : i64})
llvm.func internal @i(%arg0: i32) attributes {dso_local} {
  // CHECK-LABEL: @i
  // CHECK: llvm.call @testfn3(%arg0) : (i32 {llvm.alignstack = 8 : i64}) -> ()
  llvm.call @testfn3(%arg0) : (i32 {llvm.alignstack = 8 : i64}) -> ()
  llvm.return
}
