; RUN: llc < %s -mtriple=aarch64 -mattr=+mte | FileCheck %s

define i8* @tagp2(i8* %p, i8* %tag) {
entry:
; CHECK-LABEL: tagp2:
; CHECK: subp [[R:x[0-9]+]], x0, x1
; CHECK: add  [[R]], [[R]], x1
; CHECK: addg x0, [[R]], #0, #2
; CHECK: ret
  %q = call i8* @llvm.aarch64.tagp.p0i8(i8* %p, i8* %tag, i64 2)
  ret i8* %q
}

define i8* @irg_tagp_unrelated(i8* %p, i8* %q) {
entry:
; CHECK-LABEL: irg_tagp_unrelated:
; CHECK: irg  [[R0:x[0-9]+]], x0{{$}}
; CHECK: subp [[R:x[0-9]+]], [[R0]], x1
; CHECK: add  [[R]], [[R0]], x1
; CHECK: addg x0, [[R]], #0, #1
; CHECK: ret
  %p1 = call i8* @llvm.aarch64.irg(i8* %p, i64 0)
  %q1 = call i8* @llvm.aarch64.tagp.p0i8(i8* %p1, i8* %q, i64 1)
  ret i8* %q1
}

define i8* @tagp_alloca(i8* %tag) {
entry:
; CHECK-LABEL: tagp_alloca:
; CHECK: mov  [[R0:x[0-9]+]], sp{{$}}
; CHECK: subp [[R:x[0-9]+]], [[R0]], x0{{$}}
; CHECK: add  [[R]], [[R0]], x0{{$}}
; CHECK: addg x0, [[R]], #0, #3
; CHECK: ret
  %a = alloca i8, align 16
  %q = call i8* @llvm.aarch64.tagp.p0i8(i8* %a, i8* %tag, i64 3)
  ret i8* %q
}

declare i8* @llvm.aarch64.irg(i8* %p, i64 %exclude)
declare i8* @llvm.aarch64.tagp.p0i8(i8* %p, i8* %tag, i64 %ofs)
