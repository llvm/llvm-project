; RUN: mlir-translate --import-llvm %s | FileCheck %s

%Domain = type { ptr, ptr }

; CHECK: llvm.mlir.global external @D()
; CHECK-SAME: !llvm.struct<"Domain", (ptr, ptr)>
; CHECK: %[[E0:.+]] = llvm.mlir.zero : !llvm.ptr
; CHECK: %[[RES:.+]] = llvm.mlir.zero : !llvm.struct<"Domain", (ptr, ptr)>
; CHECK: llvm.return %[[RES]] : !llvm.struct<"Domain", (ptr, ptr)>
@D = global %Domain zeroinitializer
