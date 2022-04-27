; RUN: mlir-translate --import-llvm %s | FileCheck %s

; CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(7 : i32) : i32
; CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(8 : i16) : i16
; CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(4 : i8) : i8
; CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(9 : i32) : i32
; CHECK: %[[ROOT:.+]] = llvm.mlir.undef : !llvm.struct<"SimpleAggType", (i32, i8, i16, i32)>
; CHECK: %[[CHAIN0:.+]] = llvm.insertvalue %[[C3]], %[[ROOT]][0 : i32]
; CHECK: %[[CHAIN1:.+]] = llvm.insertvalue %[[C2]], %[[CHAIN0]][1 : i32]
; CHECK: %[[CHAIN2:.+]] = llvm.insertvalue %[[C1]], %[[CHAIN1]][2 : i32]
; CHECK: %[[CHAIN3:.+]] = llvm.insertvalue %[[C0]], %[[CHAIN2]][3 : i32]
; CHECK: llvm.return %[[CHAIN3]]
%SimpleAggType = type {i32, i8, i16, i32}
@simpleAgg = global %SimpleAggType {i32 9, i8 4, i16 8, i32 7}

; CHECK: %[[NP:.+]] = llvm.mlir.null : !llvm.ptr<struct<"SimpleAggType", (i32, i8, i16, i32)>>
; CHECK-DAG: %[[C0:.+]] = llvm.mlir.constant(4 : i32) : i32
; CHECK-DAG: %[[C1:.+]] = llvm.mlir.constant(3 : i16) : i16
; CHECK-DAG: %[[C2:.+]] = llvm.mlir.constant(2 : i8) : i8
; CHECK-DAG: %[[C3:.+]] = llvm.mlir.constant(1 : i32) : i32
; CHECK: %[[ROOT:.+]] = llvm.mlir.undef : !llvm.struct<"SimpleAggType", (i32, i8, i16, i32)>
; CHECK: %[[CHAIN0:.+]] = llvm.insertvalue %[[C3]], %[[ROOT]][0 : i32]
; CHECK: %[[CHAIN1:.+]] = llvm.insertvalue %[[C2]], %[[CHAIN0]][1 : i32]
; CHECK: %[[CHAIN2:.+]] = llvm.insertvalue %[[C1]], %[[CHAIN1]][2 : i32]
; CHECK: %[[CHAIN3:.+]] = llvm.insertvalue %[[C0]], %[[CHAIN2]][3 : i32]
; CHECK: %[[ROOT2:.+]] = llvm.mlir.undef : !llvm.struct<"NestedAggType", (struct<"SimpleAggType", (i32, i8, i16, i32)>, ptr<struct<"SimpleAggType", (i32, i8, i16, i32)>>)>
; CHECK: %[[CHAIN4:.+]] = llvm.insertvalue %[[CHAIN3]], %[[ROOT2]][0 : i32]
; CHECK: %[[CHAIN5:.+]] = llvm.insertvalue %[[NP]], %[[CHAIN4]][1 : i32]
; CHECK: llvm.return %[[CHAIN5]]
%NestedAggType = type {%SimpleAggType, %SimpleAggType*}
@nestedAgg = global %NestedAggType { %SimpleAggType{i32 1, i8 2, i16 3, i32 4}, %SimpleAggType* null }

