; RUN: mlir-translate --import-llvm %s | FileCheck %s

%Domain = type { %Domain**, %Domain* }

; CHECK: llvm.mlir.global external @D()
; CHECK-SAME: !llvm.struct<"Domain", (ptr<ptr<struct<"Domain">>>, ptr<struct<"Domain">>)>
; CHECK-DAG: %[[E0:.+]] = llvm.mlir.null : !llvm.ptr<struct<"Domain", (ptr<ptr<struct<"Domain">>>, ptr<struct<"Domain">>)>>
; CHECK-DAG: %[[E1:.+]] = llvm.mlir.null : !llvm.ptr<ptr<struct<"Domain", (ptr<ptr<struct<"Domain">>>, ptr<struct<"Domain">>)>>>
; CHECK: %[[ROOT:.+]] = llvm.mlir.undef : !llvm.struct<"Domain", (ptr<ptr<struct<"Domain">>>, ptr<struct<"Domain">>)>
; CHECK: %[[CHAIN:.+]] = llvm.insertvalue %[[E1]], %[[ROOT]][0]
; CHECK: %[[RES:.+]] = llvm.insertvalue %[[E0]], %[[CHAIN]][1]
; CHECK: llvm.return %[[RES]]
@D = global %Domain zeroinitializer
