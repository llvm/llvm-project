; RUN: mlir-translate -opaque-pointers=0 --import-llvm %s | FileCheck %s

%Domain = type { %Domain**, %Domain* }

; CHECK: llvm.mlir.global external @D()
; CHECK-SAME: !llvm.struct<"Domain", (ptr<ptr<struct<"Domain">>>, ptr<struct<"Domain">>)>
; CHECK: %[[E0:.+]] = llvm.mlir.null : !llvm.ptr<ptr<struct<"Domain", (ptr<ptr<struct<"Domain">>>, ptr<struct<"Domain">>)>>>
; CHECK: %[[E1:.+]] = llvm.mlir.null : !llvm.ptr<struct<"Domain", (ptr<ptr<struct<"Domain">>>, ptr<struct<"Domain">>)>>
; CHECK: %[[ROOT:.+]] = llvm.mlir.undef : !llvm.struct<"Domain", (ptr<ptr<struct<"Domain">>>, ptr<struct<"Domain">>)>
; CHECK: %[[CHAIN:.+]] = llvm.insertvalue %[[E0]], %[[ROOT]][0]
; CHECK: %[[RES:.+]] = llvm.insertvalue %[[E1]], %[[CHAIN]][1]
; CHECK: llvm.return %[[RES]]
@D = global %Domain zeroinitializer
