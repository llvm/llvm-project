; RUN: llc < %s -mtriple=aarch64 -mattr=+mte | FileCheck %s

define ptr @tagp2(ptr %p, ptr %tag) {
entry:
; CHECK-LABEL: tagp2:
; CHECK: subp [[R:x[0-9]+]], x0, x1
; CHECK: add  [[R]], [[R]], x1
; CHECK: addg x0, [[R]], #0, #2
; CHECK: ret
  %q = call ptr @llvm.aarch64.tagp.p0(ptr %p, ptr %tag, i64 2)
  ret ptr %q
}

define ptr @irg_tagp_unrelated(ptr %p, ptr %q) {
entry:
; CHECK-LABEL: irg_tagp_unrelated:
; CHECK: irg  [[R0:x[0-9]+]], x0{{$}}
; CHECK: subp [[R:x[0-9]+]], [[R0]], x1
; CHECK: add  [[R]], [[R0]], x1
; CHECK: addg x0, [[R]], #0, #1
; CHECK: ret
  %p1 = call ptr @llvm.aarch64.irg(ptr %p, i64 0)
  %q1 = call ptr @llvm.aarch64.tagp.p0(ptr %p1, ptr %q, i64 1)
  ret ptr %q1
}

define ptr @tagp_alloca(ptr %tag) {
entry:
; CHECK-LABEL: tagp_alloca:
; CHECK: mov  [[R0:x[0-9]+]], sp{{$}}
; CHECK: subp [[R:x[0-9]+]], [[R0]], x0{{$}}
; CHECK: add  [[R]], [[R0]], x0{{$}}
; CHECK: addg x0, [[R]], #0, #3
; CHECK: ret
  %a = alloca i8, align 16
  %q = call ptr @llvm.aarch64.tagp.p0(ptr %a, ptr %tag, i64 3)
  ret ptr %q
}

declare ptr @llvm.aarch64.irg(ptr %p, i64 %exclude)
declare ptr @llvm.aarch64.tagp.p0(ptr %p, ptr %tag, i64 %ofs)
