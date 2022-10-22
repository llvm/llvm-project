; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+amx-tile,+amx-fp16 | FileCheck %s

; CHECK-LABEL: test_amx:
; CHECK:       # %bb.0:
; CHECK:    tdpfp16ps       %tmm1, %tmm2, %tmm3

define void @test_amx() {
call void @llvm.x86.tdpfp16ps(i8 3, i8 2, i8 1)

ret void
}
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
declare void @llvm.x86.tdpfp16ps(i8 %tile3, i8 %tile2, i8 %tile1)
