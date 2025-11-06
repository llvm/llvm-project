; RUN: mlir-translate --import-llvm %s | FileCheck %s

%Domain = type { ptr, ptr }

; CHECK: llvm.mlir.global external @D()
; CHECK-SAME: !llvm.struct<"Domain", (ptr, ptr)>
; CHECK: %[[E0:.+]] = llvm.mlir.zero : !llvm.ptr
; CHECK: %[[ROOT:.+]] = llvm.mlir.undef : !llvm.struct<"Domain", (ptr, ptr)>
; CHECK: %[[CHAIN:.+]] = llvm.insertvalue %[[E0]], %[[ROOT]][0]
; CHECK: %[[RES:.+]] = llvm.insertvalue %[[E0]], %[[CHAIN]][1]
; CHECK: llvm.return %[[RES]]
@D = global %Domain zeroinitializer
