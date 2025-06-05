; RUN: llc -O2 -mtriple=aarch64-linux-gnu -mcpu=generic -mattr=+long-calls < %s | FileCheck %s
; RUN: llc -O0 -mtriple=aarch64-linux-gnu -mcpu=generic -mattr=+long-calls < %s | FileCheck %s

declare void @far_func()
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

define void @test() {
entry:
  call void @far_func()
  ret void
}

define void @test2(ptr %dst, i8 %val, i64 %len) {
entry:
  call void @llvm.memset.p0.i64(ptr %dst, i8 %val, i64 %len, i1 false)
  ret void
}

; CHECK-LABEL: test:
; CHECK: adrp {{x[0-9]+}}, far_func
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, :lo12:far_func
; CHECK: blr {{x[0-9]+}}

; CHECK-LABEL: test2:
; CHECK: adrp {{x[0-9]+}}, memset
; CHECK: add {{x[0-9]+}}, {{x[0-9]+}}, :lo12:memset
; CHECK: blr {{x[0-9]+}}
